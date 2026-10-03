#!/usr/bin/env python3
"""Merge sharded Boltz-2 caches into one cache ordered by source_row.

Extraction is fanned out across Modal containers, each writing its own cache
directory. This reassembles them. It lives here rather than inside the Modal
app so it can be tested locally with synthetic shards, which is cheaper than
discovering an off-by-one on a paid run.

Shards are assigned by stride (`rows[shard::n_shards]`), so the original order
is restored by sorting on source_row -- which is also the key
`arm_features.row_positions` joins a split on, making the merged cache a
drop-in for the ladder.

Partial shards are fine: each shard's `progress.json` records how many of its
rows completed, and only those are merged. A container that died halfway
contributes its finished rows and nothing else.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def discover(root: Path) -> list[Path]:
    """Shard directories that have both an index and a checkpoint."""
    return sorted(p for p in root.glob("shard_*")
                  if (p / "index.json").exists() and (p / "progress.json").exists())


def plan(shards: list[Path]) -> tuple[list[tuple[int, int, int]], dict, dict]:
    """Return (sorted (source_row, shard_i, pos) rows, pairs map, first index)."""
    indices = [json.loads((p / "index.json").read_text()) for p in shards]
    slots = {idx["n_slots"] for idx in indices}
    if len(slots) != 1:
        raise ValueError(f"shards disagree on n_slots: {slots}")
    fps = {idx["fingerprint"] for idx in indices}
    if len(fps) != 1:
        # Different row lists give different fingerprints by design, so this is
        # only a warning signal, not fatal. Config differences are what matter
        # and those are caught per shard at extraction time.
        pass

    rows: list[tuple[int, int, int]] = []
    pairs: dict[int, list] = {}
    seen: set[int] = set()
    for si, (path, idx) in enumerate(zip(shards, indices)):
        done = int(json.loads((path / "progress.json").read_text())["rows_done"])
        for pos, source_row in enumerate(idx["source_row"][:done]):
            source_row = int(source_row)
            if source_row in seen:
                raise ValueError(f"source_row {source_row} appears in more than one shard")
            seen.add(source_row)
            rows.append((source_row, si, pos))
            pairs[source_row] = idx["pairs"][pos]
    rows.sort()
    return rows, pairs, indices[0]


def merge(root, dest, *, dry_run: bool = False) -> dict:
    root, dest = Path(root), Path(dest)
    shards = discover(root)
    if not shards:
        raise FileNotFoundError(f"no completed shards under {root}")
    rows, pairs, first = plan(shards)
    names = list(first["arrays"])
    summary = {"shards": len(shards), "merged_rows": len(rows),
               "n_slots": first["n_slots"], "arrays": names,
               "shard_names": [p.name for p in shards]}
    if dry_run:
        return summary

    dest.mkdir(parents=True, exist_ok=True)
    opened = [{n: np.load(p / n, mmap_mode="r") for n in names} for p in shards]
    out = {}
    for n in names:
        dim = int(opened[0][n].shape[-1])
        out[n] = np.lib.format.open_memmap(
            dest / n, mode="w+", dtype=np.float16,
            shape=(len(rows), first["n_slots"], dim))
    for k, (_sr, si, pos) in enumerate(rows):
        for n in names:
            out[n][k] = opened[si][n][pos]
    for n in names:
        out[n].flush()

    index = dict(first)
    index |= {"n_rows": len(rows),
              "source_row": [sr for sr, _, _ in rows],
              "pairs": [pairs[sr] for sr, _, _ in rows],
              "merged_from": [p.name for p in shards]}
    (dest / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (dest / "progress.json").write_text(
        json.dumps({"rows_done": len(rows), "fingerprint": index["fingerprint"]}) + "\n")
    summary["arrays_shape"] = {n: list(out[n].shape) for n in names}
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory holding shard_* subdirectories")
    ap.add_argument("dest", help="output cache directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(json.dumps(merge(args.root, args.dest, dry_run=args.dry_run), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
