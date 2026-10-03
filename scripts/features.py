#!/usr/bin/env python3
"""Build the A0 one-hot baseline features.

Arm A0 of the ladder: a plain sequence encoding with no protein language model.
Each of the same 43 residue positions the ESM-2 arms cache is encoded as a
20-dimensional indicator over the amino-acid alphabet.

    concat_onehot.npy   (28166, 43, 20) float32

The slot layout, the pseudosequence indices and the row order are imported from
extract_embeddings rather than restated, so A0 and the ESM-2 arms are guaranteed
to describe the same positions of the same rows in the same order:

    slots  0..8   peptide            (concat indices 182..190)
    slots  9..42  HLA pseudosequence (PSEUDOSEQ_INDICES)

Like the embedding cache this is split-agnostic: row i corresponds to
index.json["pairs"][i] and to source_row i + SOURCE_ROW_OFFSET, and a split
selects its rows by joining on source_row, the spreadsheet-row convention
split_dataset.py defines. The file is written into the same directory as the
embedding arrays and shares their index.json, which is verified to agree
(or written, if extraction has not run yet).

Why one-hot rather than BLOSUM62: ESM-2 layer 0 is a full-rank linear map of
one-hot (rank 20, condition number 2.3 on this dataset's alphabet), so arm A1 is
an exactly information-matched control for A0. A BLOSUM62 encoding would instead
inject a biochemical prior that layer 0 does not have, confounding A0 vs A1.

Usage:
    python scripts/features.py rasmussen_et_al_dataset.xlsx --out-dir embeddings
"""
import argparse
import json
from pathlib import Path

import numpy as np

from extract_embeddings import (
    HLA_COLUMN,
    PEPTIDE_COLUMN,
    PSEUDOSEQ_COLUMN,
    PSEUDOSEQ_INDICES,
    LINKER,
    assert_pseudoseq_indices,
    load_rows,
    slot_layout,
    write_index,
)

# The dataset uses exactly these 20 residues, so no unknown/ambiguity column is
# needed; an unexpected letter raises rather than becoming a zero vector.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {letter: i for i, letter in enumerate(AMINO_ACIDS)}
N_AA = len(AMINO_ACIDS)

FEATURE_NAME = "onehot"


def residue_letters(row, concat_positions, linker=LINKER):
    """The 43 residues this row contributes, in output-slot order."""
    sequence = row[HLA_COLUMN] + linker + row[PEPTIDE_COLUMN]
    return [sequence[p] for p in concat_positions]


def encode_row(row, concat_positions, linker=LINKER):
    """One-hot encode one row's 43 kept positions -> (43, 20) float32."""
    letters = residue_letters(row, concat_positions, linker)
    out = np.zeros((len(concat_positions), N_AA), dtype=np.float32)
    for slot, letter in enumerate(letters):
        try:
            out[slot, AA_INDEX[letter]] = 1.0
        except KeyError:
            raise ValueError(
                f"residue {letter!r} at slot {slot} is not in {AMINO_ACIDS!r} "
                f"(source_row {row['source_row']})"
            ) from None
    return out


def decode_row(encoded):
    """Invert encode_row, for the round-trip assertion."""
    assert np.all(encoded.sum(axis=1) == 1), "not one-hot: a slot has != 1 hot entry"
    return "".join(AMINO_ACIDS[i] for i in encoded.argmax(axis=1))


def assert_round_trip(array, rows, peptide_slots, pseudoseq_slots):
    """Decoding the encoded slots must return the peptide and the pseudosequence.

    The A0 analogue of the token-alignment assertion in extract_embeddings: it
    checks the slots really hold what the layout claims, rather than trusting
    the indices that produced them.
    """
    for offset, row in enumerate(rows):
        decoded = decode_row(array[offset])
        peptide = "".join(decoded[s] for s in peptide_slots)
        pseudoseq = "".join(decoded[s] for s in pseudoseq_slots)
        assert peptide == row[PEPTIDE_COLUMN], (
            f"peptide slots decode to {peptide!r}, expected {row[PEPTIDE_COLUMN]!r} "
            f"(source_row {row['source_row']})"
        )
        assert pseudoseq == row[PSEUDOSEQ_COLUMN], (
            f"pseudoseq slots decode to {pseudoseq!r}, "
            f"expected {row[PSEUDOSEQ_COLUMN]!r} (source_row {row['source_row']})"
        )
    return len(rows)


def repair_index(out_dir, rows, pseudoseq_indices=PSEUDOSEQ_INDICES, linker=LINKER):
    """Rewrite an existing index.json's source_row to the current convention.

    For a cache extracted before source_row followed split_dataset.py. The .npy
    arrays stay valid because row order never changed, so only the labels need
    replacing -- and only source_row is touched, so the recorded model, layer
    list and slot layout survive. Deleting and rebuilding the index instead
    would lose the model name, since this module only knows about its own arm.
    """
    out_dir = Path(out_dir)
    path = out_dir/"index.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; nothing to repair")
    index = json.loads(path.read_text())

    if index["n_rows"] != len(rows):
        raise ValueError(
            f"{path} describes {index['n_rows']} rows but the dataset has {len(rows)}. "
            "Repair needs the same rows the cache was extracted over "
            "(match --limit).")
    expected_pairs = [[row[HLA_COLUMN], row[PEPTIDE_COLUMN]] for row in rows]
    if index["pairs"] != expected_pairs:
        raise ValueError(
            f"{path} pairs do not match this dataset in this order, so the "
            "arrays are not a cache of these rows. Refusing to repair; re-extract.")

    before = index["source_row"][:1]
    index["source_row"] = [int(row["source_row"]) for row in rows]
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index))
    temporary.replace(path)
    print(f"repaired {path}: source_row now starts at {index['source_row'][0]} "
          f"(was {before[0] if before else 'empty'}); model "
          f"{index.get('model')!r} and layers {index.get('layers')} preserved",
          flush=True)
    return index


def check_index_agrees(out_dir, rows, n_slots):
    """If the ESM-2 cache already wrote index.json, A0 must match it exactly."""
    path = Path(out_dir) / "index.json"
    if not path.exists():
        return False
    index = json.loads(path.read_text())
    if index["n_rows"] != len(rows):
        raise ValueError(
            f"{path} describes {index['n_rows']} rows but this run has {len(rows)}. "
            "Build A0 over the same rows as the embedding cache (same --limit)."
        )
    if index["source_row"] != [int(row["source_row"]) for row in rows]:
        raise ValueError(f"{path} source_row order does not match this run")
    expected_pairs = [[row[HLA_COLUMN], row[PEPTIDE_COLUMN]] for row in rows]
    if index["pairs"] != expected_pairs:
        raise ValueError(f"{path} pairs do not match this run")
    if len(index["peptide_slots"]) + len(index["pseudoseq_slots"]) != n_slots:
        raise ValueError(f"{path} slot layout does not match this run")
    return True


def check_against_layer0(out_dir, array, rows, concat_positions, atol=1e-3):
    """Cross-check A0 against the ESM-2 layer-0 cache when it is present.

    Layer 0 is a pure per-residue lookup, so every slot that A0 says holds
    residue c must carry the same layer-0 vector everywhere, and two different
    residues must never share one. This ties the two arms' slot layouts
    together: if either one were shifted, this fails.
    """
    path = Path(out_dir) / "concat_L0.npy"
    if not path.exists():
        return None
    layer0 = np.load(path, mmap_mode="r")
    if layer0.shape[0] != len(rows) or layer0.shape[1] != array.shape[1]:
        return None
    letter_vector = {}
    checked = min(len(rows), 512)
    for i in range(checked):
        decoded = decode_row(array[i])
        for slot, letter in enumerate(decoded):
            vector = np.asarray(layer0[i, slot], dtype=np.float32)
            if letter in letter_vector:
                assert np.allclose(letter_vector[letter], vector, atol=atol), (
                    f"residue {letter!r} has two different layer-0 vectors: "
                    "A0 and the ESM-2 cache disagree about slot contents"
                )
            else:
                letter_vector[letter] = vector
    letters = sorted(letter_vector)
    stacked = np.stack([letter_vector[c] for c in letters])
    rank = int(np.linalg.matrix_rank(stacked.astype(np.float64)))
    assert rank == len(letters), (
        f"layer-0 vectors for {len(letters)} residues have rank {rank}; "
        "one-hot would not be an information-matched control"
    )
    return len(letters), rank, checked


def build_onehot(dataset_path, out_dir, limit=None, linker=LINKER,
                 pseudoseq_indices=PSEUDOSEQ_INDICES, check_layer0=True):
    """Encode the A0 features and write them beside the embedding cache.

    Plain function taking all paths as arguments, like extract_embeddings, so a
    Modal entrypoint or a notebook can call it directly.
    """
    out_dir = Path(out_dir)
    concat_positions, peptide_slots, pseudoseq_slots = slot_layout(
        linker=linker, pseudoseq_indices=pseudoseq_indices)
    n_slots = len(concat_positions)

    frame, rows = load_rows(dataset_path, limit=limit)
    n_rows = len(rows)

    n_alleles = assert_pseudoseq_indices(frame, pseudoseq_indices)
    print(f"[assert 1/3] pseudosequence indices reconstruct hla_pseudoseq "
          f"for {n_alleles}/{n_alleles} alleles: PASS", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    array = np.zeros((n_rows, n_slots, N_AA), dtype=np.float32)
    for i, row in enumerate(rows):
        array[i] = encode_row(row, concat_positions, linker)

    n_checked = assert_round_trip(array, rows, peptide_slots, pseudoseq_slots)
    print(f"[assert 2/3] all {n_checked} rows round-trip: slots "
          f"{peptide_slots[0]}..{peptide_slots[-1]} decode to the peptide, "
          f"{pseudoseq_slots[0]}..{pseudoseq_slots[-1]} to hla_pseudoseq: PASS",
          flush=True)

    shared = check_index_agrees(out_dir, rows, n_slots)
    if not shared:
        write_index(out_dir, rows, FEATURE_NAME, [], concat_positions,
                    peptide_slots, pseudoseq_slots, pseudoseq_indices, linker,
                    n_slots, dim=N_AA)
        print("index.json written (no embedding cache present yet)", flush=True)
    else:
        print("index.json from the embedding cache agrees: same rows, same "
              "source_row order, same slot layout", flush=True)

    path = out_dir / f"concat_{FEATURE_NAME}.npy"
    np.save(path, array)

    layer0 = check_against_layer0(out_dir, array, rows, concat_positions) if check_layer0 else None
    if layer0 is not None:
        n_letters, rank, checked = layer0
        print(f"[assert 3/3] cross-check vs concat_L0.npy over {checked} rows: "
              f"{n_letters} residues map 1:1 onto layer-0 vectors, rank "
              f"{rank}/{n_letters} (one-hot is an information-matched control "
              f"for A1): PASS", flush=True)
    else:
        print("[assert 3/3] skipped: no comparable concat_L0.npy in this directory",
              flush=True)

    density = float(array.sum() / array.size)
    print(f"\nwrote {path}  shape={array.shape} dtype={array.dtype} "
          f"{path.stat().st_size / 2**20:.1f} MiB")
    print(f"features per row = {n_slots * N_AA} "
          f"(one hot per slot, density {density:.4f} = 1/{N_AA})")
    return {
        "path": str(path),
        "shape": list(array.shape),
        "n_rows": n_rows,
        "slots": n_slots,
        "dim": N_AA,
        "features_per_row": n_slots * N_AA,
        "shared_index": bool(shared),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Encode only the first N rows (must match the "
                             "embedding cache's --limit)")
    parser.add_argument("--linker", default=LINKER)
    parser.add_argument("--no-check-layer0", dest="check_layer0",
                        action="store_false",
                        help="Skip the cross-check against concat_L0.npy")
    parser.add_argument("--repair-index", action="store_true",
                        help="Rewrite an existing index.json's source_row to the "
                             "current convention and exit. For a cache extracted "
                             "before source_row followed split_dataset.py; the .npy "
                             "arrays are left alone and the recorded model and layer "
                             "list are preserved.")
    args = parser.parse_args()

    if args.repair_index:
        _, rows = load_rows(args.dataset, limit=args.limit)
        repair_index(args.out_dir, rows, linker=args.linker)
        return

    summary = build_onehot(dataset_path=args.dataset, out_dir=args.out_dir,
                           limit=args.limit, linker=args.linker,
                           check_layer0=args.check_layer0)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
