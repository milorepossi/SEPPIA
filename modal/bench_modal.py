"""Modal benchmark: pick the cheapest GPU per complex, and test fine-tunability.

The question is **not** which GPU is fastest. Extraction cost is
`$/hour ÷ complexes/hour`, so a B200 that is 3x faster than an L4 but 8x the
price is a worse buy. At 191 tokens a complex is tiny, so the large cards are
also badly under-occupied by a single `boltz predict` process -- which is why
this benchmark measures **concurrency scaling** as well as raw speed. Running
K workers on one card is what makes an H100 competitive, if anything does.

Three entrypoints:

  modal run modal/bench_modal.py::probe           # identity + kernel availability
  modal run modal/bench_modal.py::sweep           # $/complex across GPUs
  modal run modal/bench_modal.py::finetune        # forward+backward feasibility

Prices are Modal's published per-second rates, hardcoded so the report carries
its own cost model. Verify against modal.com/pricing before trusting a total.
"""
from __future__ import annotations

import json
import os
import pathlib

import modal

# Modal per-second rates (USD), converted to per-hour in the report.
PRICES = {
    "T4": 0.000164, "L4": 0.000222, "A10G": 0.000306, "L40S": 0.000542,
    "A100-40GB": 0.000583, "A100-80GB": 0.000694, "H100": 0.001097,
    "H200": 0.001261, "B200": 0.001736,
}
# T4 is sm_75: no bf16, and Boltz-2 autocasts to bf16. Included anyway because
# it is the cheapest card and "it does not work" is a useful measured result.
SWEEP_ORDER = ["L4", "A10G", "L40S", "A100-40GB", "A100-80GB", "H100", "H200", "B200", "T4"]

HERE = pathlib.Path(__file__).parent
CASES = json.loads((HERE / "bench_cases.json").read_text())
HLA_LEN, PEPTIDE_LEN = 182, 9
N_TOKENS = HLA_LEN + PEPTIDE_LEN

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("boltz==2.2.1", "numpy", "pandas")
    # cuequivariance supplies the triangle-multiplication kernel Boltz-2 imports
    # and hard-fails without. Installed best-effort: its wheels are built per
    # CUDA version, so this may be a no-op and the fallback is --no_kernels.
    .run_commands(
        "pip install --no-cache-dir cuequivariance-torch cuequivariance-ops-torch-cu12 "
        "|| echo 'CUEQ_INSTALL_FAILED'",
    )
    .env({"BOLTZ_CACHE": "/cache/boltz", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_file(HERE / "bench_cases.json", "/root/bench_cases.json")
)

app = modal.App("boltz-pmhc-bench", image=image)
cache = modal.Volume.from_name("boltz-pmhc-cache", create_if_missing=True)
VOLUMES = {"/cache": cache}


# --------------------------------------------------------------------------- #
# helpers that run inside the container
# --------------------------------------------------------------------------- #
def _gpu_facts() -> dict:
    import torch

    out = {"torch": str(torch.__version__), "cuda": bool(torch.cuda.is_available())}
    if not torch.cuda.is_available():
        return out
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    out |= {
        "device": str(props.name),
        "memory_GiB": round(props.total_memory / 1024**3, 1),
        "capability": f"{cap[0]}.{cap[1]}",
        "sm80_plus": cap[0] >= 8,
        "bf16": bool(torch.cuda.is_bf16_supported()),
        "sms": int(props.multi_processor_count),
    }
    try:
        import cuequivariance_torch  # noqa: F401
        out["cuequivariance"] = True
    except Exception as exc:
        out["cuequivariance"] = False
        out["cuequivariance_error"] = type(exc).__name__
    return out


def _write_yamls(root: pathlib.Path, n: int) -> list[str]:
    """n distinct peptide-HLA YAMLs, cycling the embedded case list.

    The peptide always gets `msa: empty`; so does the HLA here, because MSA
    depth is a separate variable and single-sequence mode keeps the timing
    comparable across GPUs.
    """
    root.mkdir(parents=True, exist_ok=True)
    flat = [(c["allele"], c["hla_seq"], p) for c in CASES for p in c["peptides"]]
    ids = []
    for i in range(n):
        allele, hla, pep = flat[i % len(flat)]
        rid = f"c{i:04d}"
        (root / f"{rid}.yaml").write_text(
            "version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: {hla}\n      msa: empty\n"
            f"  - protein:\n      id: B\n      sequence: {pep}\n      msa: empty\n"
        )
        ids.append(rid)
    return ids


def _boltz_cmd(in_dir, out_dir, *, steps: int, recycles: int, kernels: bool,
               workers: int = 2) -> list[str]:
    cmd = [
        "boltz", "predict", str(in_dir), "--out_dir", str(out_dir),
        "--cache", "/cache/boltz", "--write_embeddings", "--output_format", "pdb",
        "--sampling_steps", str(steps), "--recycling_steps", str(recycles),
        "--diffusion_samples", "1", "--num_workers", str(workers), "--override",
    ]
    if not kernels:
        cmd.append("--no_kernels")
    return cmd


def _gpu_mem_used_MiB() -> int:
    """Device-wide memory in use, from nvidia-smi.

    torch.cuda.max_memory_allocated() is useless here: boltz runs in
    subprocesses, so the parent's allocator sees nothing. Concurrency is partly
    a memory question -- each worker loads its own 2.3 GB checkpoint -- so this
    has to come from the driver.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def _run_once(n_per_worker: int, steps: int, recycles: int, kernels: bool,
              concurrency: int, tag: str) -> dict:
    """Time `concurrency` parallel boltz workers, each over n_per_worker complexes.

    n is **per worker**, not a total that gets divided. If it were a total,
    raising concurrency would shrink each worker's slice until the measurement
    was dominated by the per-process checkpoint load, and the concurrency
    comparison would be meaningless.
    """
    import shutil
    import subprocess
    import threading
    import time

    work = pathlib.Path(f"/tmp/{tag}")
    shutil.rmtree(work, ignore_errors=True)
    shards = []
    for k in range(concurrency):
        d = work / f"in{k}"
        _write_yamls(d, n_per_worker)
        shards.append(d)

    peak_mib = [_gpu_mem_used_MiB()]
    stop = threading.Event()

    def sample_memory() -> None:
        while not stop.wait(2.0):
            peak_mib.append(_gpu_mem_used_MiB())

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    started = time.monotonic()
    procs = [subprocess.Popen(_boltz_cmd(d, work / f"out{k}", steps=steps,
                                         recycles=recycles, kernels=kernels),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for k, d in enumerate(shards)]
    fails = []
    for p in procs:
        _, err = p.communicate()
        if p.returncode != 0:
            fails.append(err[-900:])
    elapsed = time.monotonic() - started
    stop.set()
    sampler.join(timeout=5)
    total = n_per_worker * concurrency
    produced = len(list(work.rglob("embeddings_*.npz")))

    result = {
        "tag": tag, "n_requested": total, "n_produced": produced,
        "n_per_worker": n_per_worker,
        "concurrency": concurrency, "sampling_steps": steps,
        "recycling_steps": recycles, "kernels": kernels,
        "wall_seconds": round(elapsed, 2),
        "ok": not fails and produced == total,
        "peak_device_MiB": max(peak_mib),
    }
    if produced:
        result["seconds_per_complex"] = round(elapsed / produced, 3)
    if fails:
        result["error"] = fails[0]
    shutil.rmtree(work, ignore_errors=True)
    return result


# --------------------------------------------------------------------------- #
# remote functions, one per GPU type
# --------------------------------------------------------------------------- #
def _make(gpu: str, timeout: int = 3600):
    return app.function(gpu=gpu, volumes=VOLUMES, timeout=timeout,
                        name=f"bench_{gpu.replace('-', '_').lower()}")


def _bench_body(gpu: str, n: int, concurrencies: list[int], warm: bool) -> dict:
    """Shared benchmark body; n is complexes **per worker**.

    Weights are fetched once into the Volume, so only the first GPU to run
    pays the download.
    """
    report = {"gpu": gpu, "facts": _gpu_facts(),
              "price_per_hour": round(PRICES[gpu] * 3600, 4), "runs": []}
    if not report["facts"].get("cuda"):
        report["fatal"] = "no CUDA device"
        return report

    os.makedirs("/cache/boltz", exist_ok=True)
    use_kernels = bool(report["facts"].get("cuequivariance"))

    # One untimed warm run so the weight download and any JIT are not charged
    # to the measurement.
    if warm:
        w = _run_once(2, 10, 0, use_kernels, 1, "warm")
        report["warmup"] = w
        cache.commit()
        if not w["ok"]:
            # retry on the pure-PyTorch path before giving up on this GPU
            w2 = _run_once(2, 10, 0, False, 1, "warm2")
            report["warmup_no_kernels"] = w2
            if not w2["ok"]:
                report["fatal"] = "boltz failed on both kernel paths"
                return report
            use_kernels = False
    report["kernels_used"] = use_kernels

    for c in concurrencies:
        report["runs"].append(_run_once(n, 10, 3, use_kernels, c, f"c{c}"))

    ok = [r for r in report["runs"] if r.get("ok") and r.get("seconds_per_complex")]
    if ok:
        best = min(ok, key=lambda r: r["seconds_per_complex"])
        sec = best["seconds_per_complex"]
        report["best"] = {
            "concurrency": best["concurrency"],
            "seconds_per_complex": sec,
            "dollars_per_complex": round(PRICES[gpu] * sec, 6),
            "cost_full_28166": round(PRICES[gpu] * sec * 28166, 2),
            "hours_full_28166": round(28166 * sec / 3600, 2),
        }
    return report


# Explicit per-GPU functions: Modal resolves the gpu= at decoration time.
@_make("L4")
def bench_l4(n: int = 8, concurrencies: list[int] = [1, 2, 4]):
    return json.dumps(_bench_body("L4", n, concurrencies, True))


@_make("A10G")
def bench_a10g(n: int = 8, concurrencies: list[int] = [1, 2, 4]):
    return json.dumps(_bench_body("A10G", n, concurrencies, True))


@_make("L40S")
def bench_l40s(n: int = 8, concurrencies: list[int] = [1, 2, 4]):
    return json.dumps(_bench_body("L40S", n, concurrencies, True))


@_make("A100-40GB")
def bench_a100_40(n: int = 8, concurrencies: list[int] = [1, 2, 4]):
    return json.dumps(_bench_body("A100-40GB", n, concurrencies, True))


@_make("A100-80GB")
def bench_a100_80(n: int = 8, concurrencies: list[int] = [1, 2, 4]):
    return json.dumps(_bench_body("A100-80GB", n, concurrencies, True))


@_make("H100")
def bench_h100(n: int = 8, concurrencies: list[int] = [1, 2, 4, 8]):
    return json.dumps(_bench_body("H100", n, concurrencies, True))


@_make("H200")
def bench_h200(n: int = 8, concurrencies: list[int] = [1, 2, 4, 8]):
    return json.dumps(_bench_body("H200", n, concurrencies, True))


@_make("B200")
def bench_b200(n: int = 8, concurrencies: list[int] = [1, 2, 4, 8]):
    return json.dumps(_bench_body("B200", n, concurrencies, True))


@_make("T4")
def bench_t4(n: int = 4, concurrencies: list[int] = [1]):
    return json.dumps(_bench_body("T4", n, concurrencies, True))


FUNCS = {
    "L4": bench_l4, "A10G": bench_a10g, "L40S": bench_l40s,
    "A100-40GB": bench_a100_40, "A100-80GB": bench_a100_80,
    "H100": bench_h100, "H200": bench_h200, "B200": bench_b200, "T4": bench_t4,
}


# --------------------------------------------------------------------------- #
# fine-tuning feasibility
# --------------------------------------------------------------------------- #
def _finetune_body(gpu: str, blocks_list: list[int]) -> dict:
    """Cost of backprop through the last N Pairformer blocks at 191 tokens.

    Fine-tuning Boltz-2 means gradients through the trunk's pairwise stack,
    which is the expensive part by a wide margin. This builds a PairformerModule
    of N blocks at the real widths and times forward-only against
    forward+backward on synthetic tensors of the true shape, which isolates the
    compute without needing the data pipeline.

    Caching the input to the last block is not an escape route: z is
    [191, 191, 128] float32 = 17.8 MB per complex, so caching it for 28,166
    rows is 500 GB. Any fine-tune pays a trunk forward every step.
    """
    import time

    import torch
    from boltz.model.layers.pairformer import PairformerModule

    report = {"gpu": gpu, "facts": _gpu_facts(),
              "price_per_hour": round(PRICES[gpu] * 3600, 4), "probes": []}
    if not report["facts"].get("cuda"):
        report["fatal"] = "no CUDA device"
        return report

    dev = torch.device("cuda")
    token_s, token_z = 384, 128
    # Boltz-2's trunk is 48 blocks, 16 heads, dropout 0.25, and it trains with
    # activation checkpointing ON (scripts/train/configs/structure.yaml). Both
    # memory modes are probed: checkpointing trades roughly a third more
    # compute for a large memory saving, and which one fits decides whether a
    # full-trunk fine-tune is possible at all.
    combos = [(n, ckpt) for n in blocks_list for ckpt in (False, True)]
    for n_blocks, ckpt in combos:
        # v2=True is required: PairformerModule defaults to v2=False, which
        # builds the v1 attention and then fails with an unexpected 'k_in'
        # keyword. Boltz-2's checkpoint sets it through pairformer_args.
        module = PairformerModule(token_s, token_z, num_blocks=n_blocks,
                                  num_heads=16, dropout=0.25,
                                  activation_checkpointing=ckpt, v2=True).to(dev)
        module.train()
        probe = {"blocks": n_blocks, "activation_checkpointing": ckpt,
                 "trainable_params_M": round(
                     sum(q.numel() for q in module.parameters()) / 1e6, 2)}

        s = torch.randn(1, N_TOKENS, token_s, device=dev)
        z = torch.randn(1, N_TOKENS, N_TOKENS, token_z, device=dev)
        mask = torch.ones(1, N_TOKENS, device=dev)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        def once(backward: bool) -> float:
            torch.cuda.synchronize()
            t0 = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                so, zo = module(s.clone().requires_grad_(backward),
                                z.clone().requires_grad_(backward),
                                mask=mask, pair_mask=pair_mask, use_kernels=False)
                loss = so.float().mean() + zo.float().mean()
            if backward:
                loss.backward()
                module.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            return time.monotonic() - t0

        try:
            with torch.no_grad():
                once(False)                                  # warm
                fwd = min(once(False) for _ in range(3))
            torch.cuda.reset_peak_memory_stats()
            bwd = min(once(True) for _ in range(3))
            probe |= {
                "forward_s": round(fwd, 4),
                "forward_backward_s": round(bwd, 4),
                "backward_overhead_x": round(bwd / fwd, 2) if fwd else None,
                "peak_GiB": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            }
            # one epoch over the pilot and the full set, at this step cost
            for label, rows in (("pilot_2814", 2814), ("full_28166", 28166)):
                hours = rows * bwd / 3600
                probe[f"epoch_{label}_hours"] = round(hours, 2)
                probe[f"epoch_{label}_usd"] = round(hours * PRICES[gpu] * 3600, 2)
        except torch.cuda.OutOfMemoryError:
            probe["oom"] = True
            torch.cuda.empty_cache()
        except Exception as exc:
            probe["error"] = f"{type(exc).__name__}: {exc}"
        report["probes"].append(probe)
        del module
        torch.cuda.empty_cache()
    return report


@app.function(gpu="A100-80GB", volumes=VOLUMES, timeout=2400, name="ft_a100_80")
def ft_a100_80(blocks: list[int] = [1, 4, 48]):
    return json.dumps(_finetune_body("A100-80GB", blocks))


@app.function(gpu="H100", volumes=VOLUMES, timeout=2400, name="ft_h100")
def ft_h100(blocks: list[int] = [1, 4, 48]):
    return json.dumps(_finetune_body("H100", blocks))


@app.function(gpu="L4", volumes=VOLUMES, timeout=2400, name="ft_l4")
def ft_l4(blocks: list[int] = [1, 4, 48]):
    return json.dumps(_finetune_body("L4", blocks))


@app.function(gpu="B200", volumes=VOLUMES, timeout=2400, name="ft_b200")
def ft_b200(blocks: list[int] = [1, 4, 48]):
    return json.dumps(_finetune_body("B200", blocks))


# --------------------------------------------------------------------------- #
# local entrypoints
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def probe(gpus: str = "L4,H100"):
    """Cheap identity + kernel-availability check before spending on a sweep."""
    for gpu in [g.strip() for g in gpus.split(",") if g.strip()]:
        out = json.loads(FUNCS[gpu].remote(n=2, concurrencies=[1]))
        print(json.dumps({k: out[k] for k in ("gpu", "facts", "kernels_used", "warmup")
                          if k in out}, indent=2), flush=True)


@app.local_entrypoint()
def sweep(gpus: str = "", n: int = 8, out: str = "modal/sweep_results.json"):
    """Run every GPU in parallel and rank by dollars per complex."""
    names = [g.strip() for g in (gpus.split(",") if gpus else SWEEP_ORDER) if g.strip()]
    handles = {g: FUNCS[g].spawn(n=n) for g in names}
    results = {}
    for g, h in handles.items():
        try:
            results[g] = json.loads(h.get())
        except Exception as exc:
            results[g] = {"gpu": g, "fatal": f"{type(exc).__name__}: {exc}"}
        print(f"--- {g} ---", flush=True)
        print(json.dumps(results[g].get("best") or results[g], indent=2), flush=True)

    ranked = sorted((r for r in results.values() if r.get("best")),
                    key=lambda r: r["best"]["dollars_per_complex"])
    print("\n=== ranked by $/complex ===", flush=True)
    for r in ranked:
        b = r["best"]
        print("%-11s %5.2f s/cx  conc=%d  $%.6f/cx  full set $%6.2f (%5.1f h)  %s"
              % (r["gpu"], b["seconds_per_complex"], b["concurrency"],
                 b["dollars_per_complex"], b["cost_full_28166"],
                 b["hours_full_28166"],
                 "kernels" if r.get("kernels_used") else "no_kernels"), flush=True)
    pathlib.Path(out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}", flush=True)


@app.local_entrypoint()
def finetune(gpus: str = "L4,A100-80GB,H100", out: str = "modal/finetune_results.json"):
    """Is fine-tuning the trunk affordable on $150?"""
    fns = {"L4": ft_l4, "A100-80GB": ft_a100_80, "H100": ft_h100, "B200": ft_b200}
    names = [g.strip() for g in gpus.split(",") if g.strip()]
    handles = {g: fns[g].spawn() for g in names}
    results = {}
    for g, h in handles.items():
        try:
            results[g] = json.loads(h.get())
        except Exception as exc:
            results[g] = {"gpu": g, "fatal": f"{type(exc).__name__}: {exc}"}
        print(f"--- {g} ---", flush=True)
        print(json.dumps(results[g], indent=2), flush=True)
    pathlib.Path(out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


@app.local_entrypoint()
def refine(out: str = "modal/refine_results.json"):
    """Second pass: push concurrency to the memory limit and strip startup.

    The first sweep understated every GPU twice over. It capped concurrency at
    4 or 8 while per-worker memory is only ~2.8-3.5 GiB, so an 80 GiB H100 was
    using a third of its memory; and at 6 complexes per worker the ~55 s
    `boltz predict` start-up dominated, which inflates s/complex by roughly
    half.

    Both are fixed here. Concurrency is set from the measured per-worker
    footprint against each card's memory, and each GPU is run at two values of
    n so the **marginal** cost per complex falls out of the slope:

        steady_state = (wall(n_hi) - wall(n_lo)) / ((n_hi - n_lo) * concurrency)

    That slope is what production pays, because the extractor runs
    --batch-size 256 and amortises start-up over the whole shard.
    """
    import json as _json

    # concurrency chosen from peak_device_MiB in the first sweep vs card memory,
    # leaving ~25% headroom
    plan = {"L4": 6, "A10G": 6, "L40S": 12, "H100": 20, "B200": 32}
    n_lo, n_hi = 6, 18
    results = {}
    for gpu, conc in plan.items():
        lo = _json.loads(FUNCS[gpu].spawn(n=n_lo, concurrencies=[conc]).get())
        hi = _json.loads(FUNCS[gpu].spawn(n=n_hi, concurrencies=[conc]).get())
        rec = {"gpu": gpu, "concurrency": conc,
               "price_per_hour": lo["price_per_hour"], "lo": lo, "hi": hi}
        try:
            rlo, rhi = lo["runs"][0], hi["runs"][0]
            if rlo["ok"] and rhi["ok"]:
                d_wall = rhi["wall_seconds"] - rlo["wall_seconds"]
                d_cx = rhi["n_produced"] - rlo["n_produced"]
                steady = d_wall / d_cx if d_cx else None
                rec |= {
                    "apparent_s_per_cx_hi": rhi.get("seconds_per_complex"),
                    "steady_s_per_cx": round(steady, 3) if steady else None,
                    "implied_startup_s": round(
                        rhi["wall_seconds"] - steady * rhi["n_produced"], 1) if steady else None,
                    "peak_device_MiB": max(rlo.get("peak_device_MiB", 0),
                                           rhi.get("peak_device_MiB", 0)),
                }
                if steady:
                    rec |= {
                        "dollars_per_complex": round(PRICES[gpu] * steady, 6),
                        "cost_full_28166": round(PRICES[gpu] * steady * 28166, 2),
                        "cost_pilot_2814": round(PRICES[gpu] * steady * 2814, 2),
                    }
        except (KeyError, IndexError, TypeError) as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        results[gpu] = rec
        print(_json.dumps({k: v for k, v in rec.items() if k not in ("lo", "hi")},
                          indent=2), flush=True)

    ranked = sorted((r for r in results.values() if r.get("dollars_per_complex")),
                    key=lambda r: r["dollars_per_complex"])
    print("\n=== refined, steady state, ranked by $/complex ===", flush=True)
    for r in ranked:
        print("%-6s conc=%-3d %5.2f s/cx (apparent %5.2f)  $%.6f/cx  "
              "full $%6.2f  pilot $%5.2f  %5d MiB"
              % (r["gpu"], r["concurrency"], r["steady_s_per_cx"],
                 r["apparent_s_per_cx_hi"], r["dollars_per_complex"],
                 r["cost_full_28166"], r["cost_pilot_2814"],
                 r["peak_device_MiB"]), flush=True)
    pathlib.Path(out).write_text(_json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}", flush=True)
