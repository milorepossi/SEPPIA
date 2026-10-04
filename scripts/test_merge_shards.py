#!/usr/bin/env python3
"""Tests for merging sharded Boltz caches.

The cache is deliberately **heterogeneous**: the four ladder arms use 43 slots
while `boltz_ZRAW` uses 306, because it holds the raw 9x34 interface block as a
derivation source. An earlier version of the merge sized every output array
from the index's global `n_slots`, which worked for the arms and then died on
ZRAW with `could not broadcast input array from shape (306,128) into shape
(43,128)`. That bug reached a paid run because the first version of this test
only built 43-slot arrays, so a mixed cache is now the default fixture.

Run: python scripts/test_merge_shards.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_shards as ms  # noqa: E402

# name -> (slots, dim), mirroring the real cache including the odd one out
LAYOUT = {
    "boltz_S.npy": (43, 384),
    "boltz_Z.npy": (43, 128),
    "boltz_ZU.npy": (43, 128),
    "boltz_PLDDT.npy": (43, 1),
    "boltz_ZRAW.npy": (306, 128),
}
PASSED: list[str] = []


def ok(label: str) -> None:
    PASSED.append(label)
    print(f"  PASS  {label}")


def build_shards(root: Path, all_rows: list[int], n_shards: int,
                 partial_shard: int | None = 1) -> list[int]:
    """Write shard caches by stride, optionally leaving one half-finished."""
    expected: list[int] = []
    for shard in range(n_shards):
        mine = all_rows[shard::n_shards]
        here = root / f"shard_{shard:03d}"
        here.mkdir(parents=True)
        done = len(mine) if shard != partial_shard else len(mine) // 2
        expected.extend(mine[:done])
        for name, (slots, dim) in LAYOUT.items():
            arr = np.lib.format.open_memmap(here / name, mode="w+",
                                            dtype=np.float16,
                                            shape=(len(mine), slots, dim))
            for i, source_row in enumerate(mine):
                arr[i, :, :] = np.float16(source_row)   # signature = source_row
            arr.flush()
        (here / "index.json").write_text(json.dumps({
            "n_rows": len(mine), "source_row": mine,
            "pairs": [[f"H{r}", f"P{r}"] for r in mine],
            "peptide_slots": list(range(9)), "pseudoseq_slots": list(range(9, 43)),
            "n_slots": 43, "arrays": {k: list(v) for k, v in LAYOUT.items()},
            "fingerprint": "fp", "model": "boltz2"}) + "\n")
        (here / "progress.json").write_text(
            json.dumps({"rows_done": done, "fingerprint": "fp"}) + "\n")
    return sorted(expected)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="merge_test_"))
    root = work / "shards"
    try:
        all_rows = list(range(2, 52))
        expected = build_shards(root, all_rows, n_shards=3, partial_shard=1)

        summary = ms.merge(root, work / "merged")
        assert summary["merged_rows"] == len(expected), summary
        ok(f"merged {summary['merged_rows']} of {len(all_rows)} rows "
           f"(shard 1 deliberately half-finished)")

        merged = work / "merged"
        index = json.loads((merged / "index.json").read_text())
        assert index["source_row"] == expected
        ok("merged rows are ordered by source_row")

        # the regression this file exists for
        for name, (slots, dim) in LAYOUT.items():
            arr = np.load(merged / name, mmap_mode="r")
            assert arr.shape == (len(expected), slots, dim), (name, arr.shape)
        ok("every array keeps its own slot count, including the 306-slot ZRAW")

        for name in LAYOUT:
            arr = np.load(merged / name, mmap_mode="r")
            got = arr[:, 0, 0].astype(np.int64)
            assert np.array_equal(got, np.array(expected)), name
        ok("in all five arrays, row k carries the payload of its own source_row")

        assert index["pairs"][0] == [f"H{expected[0]}", f"P{expected[0]}"]
        ok("pairs travel with their rows")

        progress = json.loads((merged / "progress.json").read_text())
        assert progress["rows_done"] == len(expected)
        ok("merged progress.json records the merged row count")

        # a duplicated shard must be refused, not silently double-counted
        shutil.copytree(root / "shard_000", root / "shard_009")
        try:
            ms.merge(root, work / "m2")
        except ValueError as exc:
            assert "more than one shard" in str(exc)
            ok("a source_row present in two shards is rejected")
        else:
            raise AssertionError("duplicate source_row was not caught")
        shutil.rmtree(root / "shard_009")

        # shards whose array shapes disagree must be refused
        bad = root / "shard_000"
        idx = json.loads((bad / "index.json").read_text())
        idx["arrays"]["boltz_ZRAW.npy"] = [999, 128]
        (bad / "index.json").write_text(json.dumps(idx) + "\n")
        try:
            ms.merge(root, work / "m3")
        except ValueError as exc:
            assert "array shapes" in str(exc)
            ok("shards disagreeing on array shapes are rejected")
        else:
            raise AssertionError("shape disagreement was not caught")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n{len(PASSED)}/{len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
