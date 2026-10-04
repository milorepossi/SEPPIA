#!/usr/bin/env python3
"""Tests for the combined arm `BZSZ` and the pre-pooling standardisation.

Both exist for one reason, and this file is the proof of it: concatenating
`boltz_S` (peaks ~1383) with `boltz_Z` (peaks ~294) and then fitting PCA lets
the larger block monopolise the components, so the "combined" arm would silently
be `BZS` alone. Per-dimension standardisation before pooling, fitted on the
training rows, is what makes the arm mean what it says.

Run: python scripts/test_combined.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_features as af  # noqa: E402
import build_combined as bc  # noqa: E402

PASSED: list[str] = []


def ok(label: str) -> None:
    PASSED.append(label)
    print(f"  PASS  {label}")


def make_cache(root: Path, n_rows: int = 40, s_scale: float = 500.0,
               z_scale: float = 1.0):
    """A two-array cache with a deliberate 500x scale gap between the blocks."""
    root.mkdir(parents=True, exist_ok=True)
    specs = [("boltz_S.npy", 384, s_scale, 0), ("boltz_Z.npy", 128, z_scale, 1)]
    for name, dim, scale, seed in specs:
        arr = np.lib.format.open_memmap(root / name, mode="w+", dtype=np.float16,
                                        shape=(n_rows, 43, dim))
        rng = np.random.default_rng(seed)
        arr[:] = (rng.normal(size=(n_rows, 43, dim)) * scale).astype(np.float16)
        arr.flush()
    rows = list(range(2, 2 + n_rows))
    (root / "index.json").write_text(json.dumps({
        "n_rows": n_rows, "source_row": rows,
        "pairs": [[f"H{r}", f"P{r}"] for r in rows],
        "peptide_slots": list(range(9)), "pseudoseq_slots": list(range(9, 43)),
        "n_slots": 43,
        "arrays": {"boltz_S.npy": [43, 384], "boltz_Z.npy": [43, 128]},
        "fingerprint": "fp", "model": "boltz2"}) + "\n")
    dataset = {"source_row": np.array(rows),
               "hla_seq": np.array([f"H{r}" for r in rows]),
               "peptide": np.array([f"P{r}" for r in rows])}
    return dataset


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="combined_test_"))
    cache = work / "cache"
    try:
        dataset = make_cache(cache)

        print("building the combined arm")
        summary = bc.build(cache)
        assert summary["shape"] == [40, 43, 512], summary["shape"]
        ok("BZSZ is (n, 43, 384+128=512)")

        checked = bc.verify(cache)
        assert checked["columns_verified"] == checked["total_columns"] == 512
        ok("every column of BZSZ reproduces its source block exactly")

        index = json.loads((cache / "index.json").read_text())
        assert index["arrays"]["boltz_SZ.npy"] == [43, 512]
        assert index["combined"]["boltz_SZ.npy"] == ["boltz_S.npy", "boltz_Z.npy"]
        ok("index.json records the new array and what it was built from")

        assert "BZSZ" in af.ARMS and af.BOLTZ_ARMS["BZSZ"] == "boltz_SZ.npy"
        ok("BZSZ is a registered ladder arm")

        print("\nthe scale trap, with and without prescaling")
        shares = {}
        for prescale in (False, True):
            enc = af.CachedArmEncoder(cache, "BZSZ", pooling="pca:20",
                                      verify=False, prescale=prescale)
            enc.fit(dataset)
            components = enc.projection[1]                   # (20, 512)
            total = (components ** 2).sum()
            shares[prescale] = float((components[:, :384] ** 2).sum() / total)

        # s occupies 384 of 512 columns, so a fair share is 0.75
        assert shares[False] > 0.99, shares[False]
        ok("without prescaling, the s block takes %.1f%% of the PCA components — "
           "the pair tensor is invisible" % (100 * shares[False]))
        assert abs(shares[True] - 0.75) < 0.10, shares[True]
        ok("with prescaling it takes %.1f%%, close to its %.0f%% column share"
           % (100 * shares[True], 100 * 384 / 512))

        print("\nprescaling is fitted on the given rows only")
        enc = af.CachedArmEncoder(cache, "BZSZ", pooling="flatten",
                                  verify=False, prescale=True)
        train = {k: v[:30] for k, v in dataset.items()}
        enc.fit(train)
        center, scale = enc.dim_center.copy(), enc.dim_scale.copy()
        enc.fit(train)
        assert np.array_equal(center, enc.dim_center) and np.array_equal(scale, enc.dim_scale)
        ok("refitting on the same rows is deterministic")

        enc_all = af.CachedArmEncoder(cache, "BZSZ", pooling="flatten",
                                      verify=False, prescale=True).fit(dataset)
        assert not np.allclose(center, enc_all.dim_center), \
            "fitting on all rows gave the same statistics as fitting on 30 — not row-dependent"
        ok("statistics depend on which rows are passed, so train-only fitting is real")

        block = enc.block(train)
        flat = block.reshape(-1, 512)
        assert abs(float(flat.mean())) < 0.05 and abs(float(flat.std()) - 1.0) < 0.1
        ok("the standardised training block is centred and unit-scaled")

        print("\nunfitted use is refused, not silently wrong")
        enc_bad = af.CachedArmEncoder(cache, "BZSZ", pooling="flatten",
                                      verify=False, prescale=True)
        try:
            enc_bad(dataset)
        except RuntimeError as exc:
            assert "fit()" in str(exc)
            ok("calling a prescaled encoder before fit() raises")
        else:
            raise AssertionError("unfitted prescaled encoder did not raise")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n{len(PASSED)}/{len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
