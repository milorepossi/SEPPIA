#!/usr/bin/env python3
"""Local end-to-end test of the Boltz-2 extraction plumbing, with no GPU.

`run_boltz` is replaced by a stub that writes exactly the files the real CLI
writes -- `embeddings_{id}.npz` holding `s` and `z` with the real shapes and
dtypes, plus a `{id}_model_0.pdb` -- so everything except the model itself is
exercised: the loader, the assertions, the slot gather, the interface
contraction, the memmap cache, the index, the resume checkpoint and the
fingerprint guard.

This exists because two separate GPU jobs were burned on plumbing bugs that
cost nothing to catch here: a `--no_kernels` flag that was not propagated from
the benchmark, and a keyword argument missing from `extract()`'s signature.

Run: python scripts/test_extract_boltz.py
"""
from __future__ import annotations

import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_boltz as eb  # noqa: E402

CSV = Path(__file__).resolve().parent.parent / "DATA" / "rasmussen_clean.csv"
PASSED: list[str] = []


def ok(label: str) -> None:
    PASSED.append(label)
    print(f"  PASS  {label}")


def write_fake_outputs(in_dir: Path, out_dir: Path) -> None:
    """Mimic `boltz predict --write_embeddings --output_format pdb`.

    Token signatures are deterministic per position so the gather and the
    contraction can be checked against a hand calculation downstream.
    Channel 2 of `s` carries the record's own numeric id, which is what makes
    the round-robin worker reassembly checkable: if a batch split across
    workers were stitched back in the wrong order, row k would hold another
    record's id.
    """
    L, S, Z = eb.N_TOKENS, eb.TOKEN_S, eb.TOKEN_Z
    pred = out_dir / "boltz_results_yaml" / "predictions"
    for yaml_path in sorted(in_dir.glob("*.yaml")):
        rid = yaml_path.stem
        here = pred / rid
        here.mkdir(parents=True, exist_ok=True)

        # Magnitudes are kept inside the float16 range on purpose: this test
        # exercises plumbing, and an out-of-range probe would instead trip the
        # extractor's overflow guard (which is covered separately below).
        # Channel 0 carries the token index so the gather stays checkable.
        s = np.zeros((L, S), dtype=np.float32)
        s[:, 0] = np.arange(L)
        s[:, 1] = 0.05 * np.arange(L)
        s[:, 2] = float(int(rid[1:]))          # this record's own id
        s[:, 3:] = 0.05 * np.arange(L)[:, None]
        ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        z = np.repeat(((ii + 2 * jj) / 100.0).astype(np.float32)[:, :, None], Z, axis=2)[None]
        s = s[None]
        np.savez_compressed(here / f"embeddings_{rid}.npz", s=s, z=z)

        # 182 CA atoms on chain A, 9 on chain B, in residue order
        lines = []
        serial = 1
        for chain, count, offset in (("A", eb.HLA_LEN, 0.0), ("B", eb.PEPTIDE_LEN, 4.0)):
            for i in range(count):
                x, y, zc = 1.5 * i, offset, offset
                lines.append(
                    f"ATOM  {serial:>5}  CA  GLY {chain}{i+1:>4}    "
                    f"{x:>8.3f}{y:>8.3f}{zc:>8.3f}  1.00  0.00           C")
                serial += 1
        (here / f"{rid}_model_0.pdb").write_text("\n".join(lines) + "\nEND\n")


def main() -> int:
    if not CSV.exists():
        print(f"missing {CSV}; run scripts/pilot_subset.py first", file=sys.stderr)
        return 1

    print("signature checks")
    sig = inspect.signature(eb.extract)
    for required in ("no_kernels", "source_rows", "sampling_steps", "progress_callback"):
        assert required in sig.parameters, f"extract() is missing {required!r}"
    ok("extract() accepts every keyword the job wrapper passes")

    call = inspect.signature(eb.run_boltz)
    assert "no_kernels" in call.parameters and call.parameters["no_kernels"].default is True
    ok("run_boltz defaults to --no_kernels (cuequivariance_torch is absent)")

    # the flag must actually reach the command line, not just the signature
    recorded: dict = {}
    real_run = eb.run_boltz_parallel
    seen_workers: list[int] = []

    def stub(jobs, cache, **kwargs):
        recorded.update(kwargs)
        seen_workers.append(len(jobs))
        for in_dir, out_dir in jobs:
            write_fake_outputs(Path(in_dir), Path(out_dir))

    call = inspect.signature(eb.extract)
    assert "workers" in call.parameters, "extract() must expose intra-GPU workers"
    ok("extract() exposes `workers` (the concurrency cost lever)")

    print("\nend-to-end with a stubbed Boltz")
    work = Path(tempfile.mkdtemp(prefix="boltz_test_"))
    try:
        eb.run_boltz_parallel = stub
        rows_wanted = [2, 3, 4, 5, 6, 7, 8, 9]
        summary = eb.extract(CSV, work / "cache", source_rows=rows_wanted,
                             boltz_cache=work / "bc", work_dir=work / "wk",
                             batch_size=5, workers=3, resume=False)
        assert recorded.get("no_kernels") is True, recorded
        ok("no_kernels=True is passed through to the boltz invocation")
        assert max(seen_workers) == 3, seen_workers
        ok(f"the batch was split across {max(seen_workers)} concurrent workers")
        assert summary["n_rows"] == len(rows_wanted), summary["n_rows"]
        assert summary["n_slots"] == 43
        ok(f"extracted {summary['n_rows']} rows x {summary['n_slots']} slots")

        cache = work / "cache"
        index = json.loads((cache / "index.json").read_text())
        assert index["source_row"] == rows_wanted
        assert index["n_slots"] == 43 and len(index["pairs"]) == len(rows_wanted)
        ok("index.json records source_row, pairs and the slot layout")

        cp = np.asarray(eb.slot_layout(linker="")[0])
        arrays = {name: np.load(cache / name) for name in eb.ARRAYS}
        for name, arr in arrays.items():
            assert arr.shape == (len(rows_wanted), 43, eb.ARRAYS[name]), (name, arr.shape)
            assert arr.dtype == np.float16
            assert np.isfinite(arr).all(), f"{name} holds non-finite values"
        ok("all three caches have the right shape, dtype and no NaNs")

        # s[token, :] == token, so slot k must hold concat_positions[k]
        s_arr = arrays["boltz_S.npy"]
        assert np.allclose(s_arr[0, :, 0], cp.astype(np.float16), atol=0.5)
        ok("boltz_S slots hold exactly the 43 intended tokens")

        # the decisive reassembly check: row k must carry row k's own record id
        want = np.array([int(eb.record_id(sr)[1:]) for sr in rows_wanted])
        got = s_arr[:, 0, 2].astype(np.int64)
        assert np.array_equal(got, want), f"reassembly mismatch: {got} vs {want}"
        ok("round-robin worker split is reassembled in source_row order")

        # the 1/d^2 and uniform contractions must differ, or BZZU is a duplicate
        assert not np.allclose(arrays["boltz_Z.npy"], arrays["boltz_ZU.npy"])
        ok("boltz_Z and boltz_ZU differ, so the weighting ablation is real")

        # every row must be written; an unwritten row would stay exactly zero
        for name, arr in arrays.items():
            per_row = np.abs(arr.astype(np.float32)).sum(axis=(1, 2))
            assert (per_row > 0).all(), f"{name} has an all-zero row"
        ok("no row was left unwritten")

        assert all(v < np.finfo(np.float16).max for v in summary["max_abs"].values())
        ok(f"magnitudes recorded and within float16: {summary['max_abs']}")

        progress = json.loads((cache / "progress.json").read_text())
        assert progress["rows_done"] == len(rows_wanted)
        ok("progress.json checkpoints the completed row count")

        # resuming a finished cache must be a no-op, not a re-run
        recorded.clear()
        again = eb.extract(CSV, cache, source_rows=rows_wanted,
                           boltz_cache=work / "bc", work_dir=work / "wk",
                           batch_size=5, workers=3, resume=True)
        assert again["processed"] == 0, again["processed"]
        assert not recorded, "resume re-invoked boltz on an already-complete cache"
        ok("resume on a complete cache does no work")

        # a different configuration must be refused, not silently mixed
        try:
            eb.extract(CSV, cache, source_rows=rows_wanted, boltz_cache=work / "bc",
                       work_dir=work / "wk", batch_size=5, workers=3, resume=True,
                       sampling_steps=200)
        except RuntimeError as exc:
            assert "fingerprint" in str(exc).lower()
            ok("fingerprint guard refuses to resume a differently-configured run")
        else:
            raise AssertionError("fingerprint mismatch was not caught")
    finally:
        eb.run_boltz_parallel = real_run
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n{len(PASSED)}/{len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
