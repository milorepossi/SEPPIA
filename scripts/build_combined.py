#!/usr/bin/env python3
"""Build the combined arm `BZSZ` by concatenating boltz_S and boltz_Z per slot.

Runs offline on CPU from an existing cache. No GPU, no re-extraction.

Why this arm exists
-------------------
`BZS` and `BZZ` test different things, and PreFold-dG's architecture combines
its tensors rather than scoring them separately: it projects each pooled
embedding to a common width and then **averages** them before a two-layer MLP.
Scoring our two tensors as isolated arms answers "is either sufficient", which
is not the question their result bears on.

The two are plausibly complementary. `BZS` carries clean per-position residue
identity and context, 384 dimensions per slot -- it is close to a like-for-like
rerun of A3. `BZZ` carries interface pair structure, which is the only thing
here a sequence model cannot express, but its contraction averages over 34
groove positions and may wash out the peptide residue's own identity in the
process. A head with only `BZZ` has no direct identity channel; a head with
only `BZS` has no pair channel.

Concatenation rather than averaging
-----------------------------------
Averaging forces all sources into one shared d-dimensional space, which is a
strong regulariser and keeps the head's width independent of how many tensors
you feed it. That matters at PreFold-dG's scale: 5,817 rows over 334 complexes.
We have 28,166 rows, so the extra parameters of a concatenation are affordable,
and concatenating lets the head weight each source independently instead of
assuming the two views are commensurable.

Both arms share the 43-slot layout, so concatenating along the feature axis is
well defined and preserves the slot structure: each slot becomes 384 + 128 =
512 features.

The scale trap
--------------
`boltz_S` peaks around 1383 and `boltz_Z` around 294. Concatenating raw and
then fitting PCA would let `s` monopolise the components, and the "combined"
arm would be `BZS` with the pair tensor as rounding error. This is handled in
`CachedArmEncoder` by per-dimension standardisation fitted on the training rows
*before* pooling, which is why that had to be added alongside this arm. The
cache stays raw and split-agnostic; the scaling happens at train time where it
cannot leak.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SOURCES = ("boltz_S.npy", "boltz_Z.npy")
TARGET = "boltz_SZ.npy"


def build(cache_dir, sources=SOURCES, target=TARGET, chunk=2048) -> dict:
    cache = Path(cache_dir)
    index_path = cache / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} is missing; is {cache} an extraction cache?")
    index = json.loads(index_path.read_text())

    arrays = []
    for name in sources:
        path = cache / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; extract it before combining")
        arrays.append(np.load(path, mmap_mode="r"))

    n_rows = arrays[0].shape[0]
    n_slots = arrays[0].shape[1]
    for name, arr in zip(sources, arrays):
        if arr.shape[0] != n_rows or arr.shape[1] != n_slots:
            raise ValueError(f"{name} has shape {arr.shape}, expected "
                             f"({n_rows}, {n_slots}, *) to match {sources[0]}")
    dim = sum(int(a.shape[2]) for a in arrays)

    out = np.lib.format.open_memmap(cache / target, mode="w+", dtype=np.float16,
                                    shape=(n_rows, n_slots, dim))
    # Chunked so a full-dataset build never holds more than a slice in memory.
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        out[start:stop] = np.concatenate(
            [np.asarray(a[start:stop], dtype=np.float32) for a in arrays],
            axis=-1).astype(np.float16)
    out.flush()

    index.setdefault("arrays", {})[target] = [n_slots, dim]
    index["combined"] = {target: list(sources)}
    tmp = cache / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2) + "\n")
    tmp.replace(index_path)

    return {"target": target, "shape": [n_rows, n_slots, dim],
            "sources": list(sources),
            "source_dims": [int(a.shape[2]) for a in arrays],
            "MB": round(out.nbytes / 1024**2, 1)}


def verify(cache_dir, sources=SOURCES, target=TARGET, n_probe=16) -> dict:
    """Spot-check that the concatenation really is the sources side by side."""
    cache = Path(cache_dir)
    combined = np.load(cache / target, mmap_mode="r")
    arrays = [np.load(cache / n, mmap_mode="r") for n in sources]
    rng = np.random.default_rng(0)
    rows = rng.choice(combined.shape[0], size=min(n_probe, combined.shape[0]),
                      replace=False)
    offset, checked = 0, 0
    for name, arr in zip(sources, arrays):
        width = arr.shape[2]
        a = np.asarray(combined[rows][:, :, offset:offset + width])
        b = np.asarray(arr[rows])
        if not np.array_equal(a, b):
            raise AssertionError(f"{target} does not reproduce {name} at columns "
                                 f"{offset}:{offset + width}")
        offset += width
        checked += width
    return {"rows_probed": len(rows), "columns_verified": checked,
            "total_columns": int(combined.shape[2])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir")
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--target", default=TARGET)
    args = ap.parse_args()
    summary = build(args.cache_dir, tuple(args.sources), args.target)
    summary["verification"] = verify(args.cache_dir, tuple(args.sources), args.target)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
