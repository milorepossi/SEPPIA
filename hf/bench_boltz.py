# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["boltz==2.2.1", "numpy", "pandas"]
# [tool.uv]
# exclude-newer = "2026-10-01T00:00:00Z"
# ///
"""Boltz-2 trunk-embedding benchmark for peptide-HLA complexes.

Runs on a Hugging Face Job. Measures, on one GPU, the quantities needed to cost
a full 28,166-complex extraction:

  1. GPU identity, compute capability, bf16 support
  2. boltz install + weight download time (fixed per-container overhead)
  3. seconds/complex under several conditions, isolating the two cost knobs:
       - diffusion sampling steps (we do not need coordinates, only the trunk)
       - MSA depth (empty vs server-generated for the HLA chain)
  4. the shapes and dtypes of the s/z tensors actually returned

Writes a JSON report to --out. Nothing here trains anything; this exists so the
scale-up decision is made on measurements rather than estimates.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Four alleles spanning both loci and the extremes of mean stability, three
# peptides each (lowest / median / highest half-life for that allele).
CASES = [
    # HLA-A*80:01: n=351, mean log1p t1/2 = 0.249; peptide t1/2 = [0.0, 0.1, 12.9]
    ("HLA-A*80:01",
     "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDSQFVQFDSDAASQRMEPRAPWIEQEEPEYWDEETRNVKAHSQTNRANLGTLRGYYNQSEDGSHTIQIMYGCDVGSDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAARRAEQLRAYLEGECVDGLRRYLENGKETLQRT",
     ['LMSGKDVFY', 'YVFVGTSRY', 'RLASYGLYY']),
    # HLA-A*02:11: n=379, mean log1p t1/2 = 2.18; peptide t1/2 = [0.0, 9.5, 39.2]
    ("HLA-A*02:11",
     "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQIDRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRT",
     ['FLIGELANL', 'QLLGWYSRV', 'ALMEVTHVL']),
    # HLA-B*41:01: n=368, mean log1p t1/2 = 0.255; peptide t1/2 = [0.0, 0.0, 8.8]
    ("HLA-B*41:01",
     "GSHSMRYFHTAMSRPGRGEPRFITVGYVDDTLFVRFDSDATSPRKEPRAPWIEQEGPEYWDRETQISKTNTQTYRESLRNLRGYYNQSEAGSHTWQRMYGCDVGPDGRLLRGHNQYAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAARVAEQDRAYLEGTCVEWLRRYLENGKDTLERA",
     ['KEAENGDEL', 'YERGNIIIF', 'YEHYFVFAA']),
    # HLA-B*42:01: n=350, mean log1p t1/2 = 2.251; peptide t1/2 = [0.0, 10.7, 49.2]
    ("HLA-B*42:01",
     "GSHSMRYFYTSVSRPGRGEPRFISVGYVDDTQFVRFDSDAASPREEPRAPWIEQEGPEYWDRNTQIYKAQAQTDRESLRNLRGYYNQSEAGSHTLQSMYGCDVGPDGRLLRGHNQYAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAARVAEQDRAYLEGTCVEWLRRYLENGKDTLERA",
     ['QPEMVTLTI', 'WPEIVGAIV', 'APRPPGAAM']),
]


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def gpu_report() -> dict:
    out: dict = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return out | {"torch": f"import failed: {exc}"}
    out["torch"] = torch.__version__
    out["cuda_available"] = bool(torch.cuda.is_available())
    if not torch.cuda.is_available():
        return out
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    out |= {
        "gpu_name": props.name,
        "gpu_total_GiB": round(props.total_memory / 1024**3, 2),
        "compute_capability": f"{cap[0]}.{cap[1]}",
        # trifast triangle-attention kernels need sm_80+; T4 is sm_75.
        "sm80_or_newer": cap[0] >= 8,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }
    return out


def write_yaml(path: Path, hla_seq: str, peptide: str, hla_msa: str | None) -> None:
    """One Boltz YAML. The peptide always gets msa: empty.

    A 9-mer alignment is meaningless, and letting the MSA server resolve 5,633
    distinct peptides would dominate the entire run for no modelling benefit.
    """
    hla_msa_line = f"      msa: {hla_msa}\n" if hla_msa is not None else ""
    path.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {hla_seq}\n"
        f"{hla_msa_line}"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {peptide}\n"
        "      msa: empty\n"
    )


def build_inputs(root: Path, hla_msa: str | None) -> int:
    root.mkdir(parents=True, exist_ok=True)
    n = 0
    for allele, hla_seq, peptides in CASES:
        assert len(hla_seq) == 182, f"{allele} hla_seq is {len(hla_seq)} aa, expected 182"
        for peptide in peptides:
            assert len(peptide) == 9
            tag = f"{allele.replace('*', '').replace(':', '')}_{peptide}"
            write_yaml(root / f"{tag}.yaml", hla_seq, peptide, hla_msa)
            n += 1
    return n


def run_condition(name: str, in_dir: Path, out_dir: Path, cache: Path,
                  sampling_steps: int, recycling_steps: int,
                  use_msa_server: bool, n: int, no_kernels: bool = True) -> dict:
    """One timed `boltz predict` invocation over the whole input directory."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        "boltz", "predict", str(in_dir),
        "--out_dir", str(out_dir),
        "--cache", str(cache),
        "--write_embeddings",
        "--output_format", "pdb",
        "--sampling_steps", str(sampling_steps),
        "--recycling_steps", str(recycling_steps),
        "--diffusion_samples", "1",
        "--num_workers", "2",
        "--override",
    ]
    if no_kernels:
        # Boltz-2 reaches for the cuEquivariance triangle-multiplication kernel
        # and hard-fails if it is absent. --no_kernels takes the pure-PyTorch
        # path, which always works; the kernel path is benchmarked separately.
        cmd.append("--no_kernels")
    if use_msa_server:
        cmd.append("--use_msa_server")
    started = time.monotonic()
    proc = sh(cmd)
    elapsed = time.monotonic() - started

    result = {
        "condition": name,
        "sampling_steps": sampling_steps,
        "recycling_steps": recycling_steps,
        "use_msa_server": use_msa_server,
        "no_kernels": no_kernels,
        "n_complexes": n,
        "returncode": proc.returncode,
        "wall_seconds": round(elapsed, 2),
        "seconds_per_complex": round(elapsed / n, 3) if n else None,
    }
    if proc.returncode != 0:
        result["stderr_tail"] = proc.stderr[-3000:]
        result["stdout_tail"] = proc.stdout[-1500:]
        return result

    # Inspect whatever embeddings landed, and record their real shapes.
    import numpy as np
    found = sorted(out_dir.rglob("embeddings_*.npz"))
    result["embedding_files"] = len(found)
    if found:
        with np.load(found[0]) as z:
            result["tensors"] = {
                k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype),
                    "MB": round(z[k].nbytes / 1024**2, 3)}
                for k in z.files
            }
        result["npz_on_disk_MB"] = round(
            sum(p.stat().st_size for p in found) / 1024**2 / len(found), 3)
    return result


def extrapolate(seconds_per_complex: float, dollars_per_hour: float,
                budget: float, total_rows: int = 28166) -> dict:
    gpu_hours = total_rows * seconds_per_complex / 3600
    return {
        "seconds_per_complex": round(seconds_per_complex, 3),
        "gpu_hours_full_dataset": round(gpu_hours, 2),
        "cost_full_dataset_usd": round(gpu_hours * dollars_per_hour, 2),
        "complexes_affordable_on_budget": int(budget / dollars_per_hour * 3600
                                              / seconds_per_complex),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/bench_report.json")
    ap.add_argument("--work", default="/tmp/bench")
    ap.add_argument("--cache", default=os.environ.get("BOLTZ_CACHE", "/tmp/boltz_cache"))
    ap.add_argument("--dollars-per-hour", type=float, default=0.80)
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--skip-msa-server", action="store_true")
    args = ap.parse_args()

    work = Path(args.work)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    report: dict = {"environment": gpu_report(), "conditions": []}
    print(json.dumps(report["environment"], indent=2), flush=True)

    # Weight download is a fixed per-container cost; time it separately so the
    # per-complex number is not polluted by it.
    started = time.monotonic()
    warm_in = work / "warm"
    n_warm = build_inputs(warm_in, "empty")
    report["boltz_cache_prepopulated"] = any(cache.iterdir()) if cache.exists() else False
    warm = run_condition("warmup_and_weight_download", warm_in, work / "out_warm",
                         cache, sampling_steps=10, recycling_steps=0,
                         use_msa_server=False, n=n_warm)
    warm["includes_weight_download"] = True
    report["weight_download_plus_first_run_seconds"] = round(time.monotonic() - started, 2)
    report["conditions"].append(warm)
    print(json.dumps(warm, indent=2), flush=True)
    if warm["returncode"] != 0:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print("WARMUP FAILED - stopping before spending more GPU time", file=sys.stderr)
        return 1

    cold_in = work / "cases"
    n = build_inputs(cold_in, "empty")

    # The two knobs worth measuring. Trunk-only is what we actually need: the
    # distogram comes from z, so coordinates are surplus.
    plan = [
        # (name, sampling_steps, recycling_steps, use_msa_server, no_kernels)
        ("empty_msa_steps10_recycle0", 10, 0, False, True),
        ("empty_msa_steps10_recycle3", 10, 3, False, True),
        ("empty_msa_steps200_recycle3", 200, 3, False, True),
        ("empty_msa_steps10_recycle3_kernels", 10, 3, False, False),
    ]
    if not args.skip_msa_server:
        plan.append(("server_msa_steps10_recycle3", 10, 3, True, True))

    for name, steps, recycles, server, nokern in plan:
        if server:
            src = work / "cases_srv"
            build_inputs(src, None)  # no msa field -> --use_msa_server fills it
        else:
            src = cold_in
        res = run_condition(name, src, work / f"out_{name}", cache,
                            steps, recycles, server, n, no_kernels=nokern)
        report["conditions"].append(res)
        print(json.dumps(res, indent=2), flush=True)

    ok = [c for c in report["conditions"]
          if c["returncode"] == 0 and not c.get("includes_weight_download")]
    if ok:
        best = min(ok, key=lambda c: c["seconds_per_complex"])
        report["extrapolation"] = {
            "basis_condition": best["condition"],
            "dollars_per_hour": args.dollars_per_hour,
            "budget_usd": args.budget,
            **extrapolate(best["seconds_per_complex"], args.dollars_per_hour, args.budget),
        }
        print(json.dumps(report["extrapolation"], indent=2), flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
