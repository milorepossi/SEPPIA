#!/usr/bin/env python3
"""Extract cached ESM-2 representations for peptide-HLA pairs.

One forward pass per dataset row over the concatenation hla_seq + LINKER +
peptide, keeping 43 of the 191 residue positions from each of several layer
depths. The cache is split-agnostic: row i of every array corresponds to
entry i of index.json["pairs"] and to source_row i, so a single extraction
serves every train/test split and every arm of the experiment. Nothing is
ever keyed by split.

Example:
    python scripts/extract_embeddings.py rasmussen_et_al_dataset.xlsx \
        --out-dir embeddings --limit 500

Layer semantics (fair-esm `repr_layers` convention, verified against
esm/model/esm2.py):
  - layer 0  is embed_scale * embed_tokens(tokens), recorded before the
    first transformer block. ESM-2 applies rotary position embeddings inside
    each attention module rather than adding them here, so layer 0 is
    genuinely context-free and position-free: a usable control arm.
  - layer 33 is recorded AFTER emb_layer_norm_after, so the final-layer
    representation is post-layer-norm while layer 15 is not.

The extraction logic is a plain function, `extract_embeddings`, that takes all
paths as arguments; model loading, extraction and file writing are separate so
a Modal entrypoint can call them without this module knowing anything
environment-specific.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

# torch is imported lazily inside load_model/extract_embeddings so that the
# CPU-only consumers of this module's constants and assertions (features.py,
# audit_target.py) do not need it installed.

# --- Top-level configuration -------------------------------------------------
MODEL_NAME = "esm2_t33_650M_UR50D"
LAYERS = [0, 15, 33]
EMBED_DIM = 1280

# Plain concatenation: ESM-2 has no chain-break token. Kept as a constant so a
# "GGGGS" variant can be ablated later without touching the indexing logic,
# which derives the peptide offset from len(LINKER).
LINKER = ""

HLA_LEN = 182
PEPTIDE_LEN = 9

# 0-based indices into hla_seq of the 34 NetMHCpan contact positions. These
# reconstruct hla_pseudoseq exactly for every allele in this dataset, which
# assert_pseudoseq_indices verifies before any compute is spent.
PSEUDOSEQ_INDICES = [
    6, 8, 23, 44, 58, 61, 62, 65, 66, 68, 69, 72, 73, 75, 76, 79, 80, 83,
    94, 96, 98, 113, 115, 117, 142, 146, 149, 151, 155, 157, 158, 162,
    166, 170,
]

BATCH_SIZE = 32
CHECKPOINT_EVERY = 2000

HLA_COLUMN = "hla_seq"
PEPTIDE_COLUMN = "peptide"
PSEUDOSEQ_COLUMN = "hla_pseudoseq"


# --- Position bookkeeping ----------------------------------------------------
def slot_layout(linker=LINKER, hla_len=HLA_LEN, peptide_len=PEPTIDE_LEN,
                pseudoseq_indices=PSEUDOSEQ_INDICES):
    """Return (concat_positions, peptide_slots, pseudoseq_slots).

    concat_positions are 0-based offsets into the concatenated residue string,
    in output-slot order: the peptide first, then the HLA pseudosequence. The
    peptide offset is derived from len(linker), so a linker variant shifts the
    peptide positions automatically.
    """
    peptide_start = hla_len + len(linker)
    peptide_positions = list(range(peptide_start, peptide_start + peptide_len))
    concat_positions = peptide_positions + list(pseudoseq_indices)
    peptide_slots = list(range(peptide_len))
    pseudoseq_slots = list(range(peptide_len, peptide_len + len(pseudoseq_indices)))
    return concat_positions, peptide_slots, pseudoseq_slots


def build_sequence(hla_seq, peptide, linker=LINKER):
    return hla_seq + linker + peptide


# --- Assertion 1: the pseudosequence indices are the right ones --------------
def assert_pseudoseq_indices(frame, pseudoseq_indices=PSEUDOSEQ_INDICES,
                             hla_column=HLA_COLUMN,
                             pseudoseq_column=PSEUDOSEQ_COLUMN):
    """Verify hla_seq[pseudoseq_indices] == hla_pseudoseq for every allele.

    Checked over unique HLA sequences, which is every distinct allele in the
    file. Raises AssertionError naming the offending allele on any mismatch.
    """
    alleles = frame[[hla_column, pseudoseq_column]].drop_duplicates()
    mismatches = []
    for hla_seq, pseudoseq in alleles.itertuples(index=False):
        reconstructed = "".join(hla_seq[i] for i in pseudoseq_indices)
        if reconstructed != pseudoseq:
            mismatches.append((reconstructed, pseudoseq))
    assert not mismatches, (
        f"pseudosequence reconstruction failed for {len(mismatches)}/{len(alleles)} "
        f"alleles; first: got {mismatches[0][0]!r} expected {mismatches[0][1]!r}"
    )
    return len(alleles)


# --- Assertion 2: BOS/EOS stripped, so slots really are what we claim --------
def assert_token_alignment(tokens, alphabet, rows, concat_positions,
                           peptide_slots, pseudoseq_slots, linker=LINKER):
    """Verify the token grid aligns with the residue strings on this batch.

    An off-by-one from the BOS token would silently shift every peptide
    representation, so this checks three things explicitly:
      - BOS is at token 0 and EOS at the last token,
      - stripping them leaves exactly hla_seq followed by the peptide,
      - the token positions actually gathered decode back to the peptide and
        to hla_pseudoseq, which tests the gather indices themselves rather
        than just the layout they were derived from.
    """
    token_positions = [p + 1 for p in concat_positions]  # +1 for BOS
    expected_tokens = 1 + HLA_LEN + len(linker) + PEPTIDE_LEN + 1
    assert tokens.shape[1] == expected_tokens, (
        f"expected {expected_tokens} tokens per row, got {tokens.shape[1]}"
    )

    for row_offset, row in enumerate(rows):
        letters = [alphabet.get_tok(int(t)) for t in tokens[row_offset]]
        assert int(tokens[row_offset, 0]) == alphabet.cls_idx, "token 0 is not BOS"
        assert int(tokens[row_offset, -1]) == alphabet.eos_idx, "last token is not EOS"

        residues = letters[1:-1]
        sequence = build_sequence(row[HLA_COLUMN], row[PEPTIDE_COLUMN], linker)
        assert "".join(residues) == sequence, "residues do not match the concatenation"
        assert "".join(residues[:HLA_LEN]) == row[HLA_COLUMN], (
            "concat index 0..181 is not hla_seq"
        )
        peptide_start = HLA_LEN + len(linker)
        assert "".join(residues[peptide_start:peptide_start + PEPTIDE_LEN]) == row[PEPTIDE_COLUMN], (
            f"concat index {peptide_start}..{peptide_start + PEPTIDE_LEN - 1} is not the peptide"
        )

        gathered_peptide = "".join(letters[token_positions[s]] for s in peptide_slots)
        assert gathered_peptide == row[PEPTIDE_COLUMN], (
            f"peptide slots gather {gathered_peptide!r}, expected {row[PEPTIDE_COLUMN]!r}"
        )
        gathered_pseudoseq = "".join(letters[token_positions[s]] for s in pseudoseq_slots)
        assert gathered_pseudoseq == row[PSEUDOSEQ_COLUMN], (
            f"pseudoseq slots gather {gathered_pseudoseq!r}, expected {row[PSEUDOSEQ_COLUMN]!r}"
        )
    return len(rows)


# --- Data --------------------------------------------------------------------
def load_rows(dataset_path, limit=None, sheet=0):
    """Read the dataset and return (frame, rows) with a source_row column.

    source_row is the row's position in the input file, which is what a split's
    rows are joined on at training time. Taking a --limit keeps the first N
    rows, so source_row stays equal to the array index.
    """
    frame = pd.read_excel(dataset_path, sheet_name=sheet)
    for column in (HLA_COLUMN, PEPTIDE_COLUMN, PSEUDOSEQ_COLUMN):
        if column not in frame.columns:
            raise ValueError(f"dataset is missing column {column!r}")
    frame = frame.reset_index(drop=True)
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)

    lengths = frame[HLA_COLUMN].str.len().unique()
    assert set(lengths) == {HLA_LEN}, f"expected all hla_seq to be {HLA_LEN} aa, saw {sorted(lengths)}"
    lengths = frame[PEPTIDE_COLUMN].str.len().unique()
    assert set(lengths) == {PEPTIDE_LEN}, f"expected all peptides to be {PEPTIDE_LEN} aa, saw {sorted(lengths)}"

    if limit is not None:
        frame = frame.iloc[:limit].copy()
    rows = frame[[HLA_COLUMN, PEPTIDE_COLUMN, PSEUDOSEQ_COLUMN, "source_row"]].to_dict("records")
    return frame, rows


def fingerprint(rows, model_name, layers, linker, pseudoseq_indices):
    """Identity of this extraction, so a resume cannot mix incompatible runs."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row[HLA_COLUMN].encode())
        digest.update(b"\x00")
        digest.update(row[PEPTIDE_COLUMN].encode())
        digest.update(b"\x00")
    return hashlib.sha256(json.dumps({
        "model": model_name,
        "layers": list(layers),
        "linker": linker,
        "pseudoseq_indices": list(pseudoseq_indices),
        "n_rows": len(rows),
        "pairs": digest.hexdigest(),
    }, sort_keys=True).encode()).hexdigest()


# --- Model -------------------------------------------------------------------
def load_model(model_name=MODEL_NAME, device=None):
    """Load an ESM-2 model in eval mode. Returns (model, alphabet, device)."""
    import esm
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, alphabet, device


# --- Output files ------------------------------------------------------------
def open_outputs(out_dir, layers, n_rows, n_slots, dim=EMBED_DIM, resume=False):
    """Open one memmapped .npy per layer, shaped (n_rows, n_slots, dim) fp16.

    Arrays are filled incrementally through the memmaps; batches are never
    accumulated in a Python list.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = (n_rows, n_slots, dim)
    arrays = {}
    for layer in layers:
        path = out_dir / f"concat_L{layer}.npy"
        if resume:
            if not path.exists():
                raise FileNotFoundError(f"cannot resume: {path} is missing")
            array = np.lib.format.open_memmap(path, mode="r+")
            if array.shape != shape or array.dtype != np.float16:
                raise ValueError(
                    f"cannot resume: {path} has shape {array.shape} dtype {array.dtype}, "
                    f"expected {shape} float16"
                )
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        arrays[layer] = array
    return arrays


def write_index(out_dir, rows, model_name, layers, concat_positions,
                peptide_slots, pseudoseq_slots, pseudoseq_indices, linker,
                n_slots, dim=None):
    """Write the split-agnostic row index next to the arrays."""
    index = {
        "pairs": [[row[HLA_COLUMN], row[PEPTIDE_COLUMN]] for row in rows],
        "source_row": [int(row["source_row"]) for row in rows],
        "model": model_name,
        "layers": list(layers),
        "peptide_slots": list(peptide_slots),
        "pseudoseq_slots": list(pseudoseq_slots),
        "pseudoseq_indices": list(pseudoseq_indices),
        "concat_positions": list(concat_positions),
        "linker": linker,
        "n_rows": len(rows),
        "shape": [len(rows), n_slots, dim],
        "dtype": "float16",
    }
    path = Path(out_dir) / "index.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index))
    temporary.replace(path)
    return path


def _progress_path(out_dir):
    return Path(out_dir) / "progress.json"


def read_progress(out_dir, expected_fingerprint):
    """Return rows already written, or 0 if there is no usable checkpoint."""
    path = _progress_path(out_dir)
    if not path.exists():
        return 0
    state = json.loads(path.read_text())
    if state.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            f"{path} belongs to a different extraction (fingerprint mismatch). "
            "Delete the output directory to start over."
        )
    return int(state.get("rows_done", 0))


def write_progress(out_dir, rows_done, n_rows, fingerprint_value, model_name, layers):
    path = _progress_path(out_dir)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "rows_done": int(rows_done),
        "n_rows": int(n_rows),
        "complete": bool(rows_done >= n_rows),
        "fingerprint": fingerprint_value,
        "model": model_name,
        "layers": list(layers),
    }))
    temporary.replace(path)


def check_free_space(out_dir, n_rows, n_slots, layers, dim=EMBED_DIM):
    """Warn if the destination cannot hold the arrays."""
    needed = n_rows * n_slots * dim * 2 * len(layers)
    free = shutil.disk_usage(Path(out_dir)).free
    return needed, free


# --- Extraction --------------------------------------------------------------
def extract_embeddings(dataset_path, out_dir, model_name=MODEL_NAME,
                       layers=LAYERS, batch_size=BATCH_SIZE, limit=None,
                       device=None, checkpoint_every=CHECKPOINT_EVERY,
                       linker=LINKER, pseudoseq_indices=PSEUDOSEQ_INDICES,
                       resume=True, progress_callback=None):
    """Extract and cache representations. Returns a summary dict.

    Takes every path as an argument and nothing from the environment, so a
    Modal entrypoint can mount volumes wherever it likes and call this.
    """
    import torch

    layers = list(layers)
    out_dir = Path(out_dir)
    concat_positions, peptide_slots, pseudoseq_slots = slot_layout(
        linker=linker, pseudoseq_indices=pseudoseq_indices)
    n_slots = len(concat_positions)

    frame, rows = load_rows(dataset_path, limit=limit)
    n_rows = len(rows)

    # Assertion 1, before any compute is spent.
    n_alleles = assert_pseudoseq_indices(frame, pseudoseq_indices)
    print(f"[assert 1/2] pseudosequence indices reconstruct hla_pseudoseq "
          f"for {n_alleles}/{n_alleles} alleles: PASS", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    expected_fingerprint = fingerprint(rows, model_name, layers, linker, pseudoseq_indices)
    rows_done = read_progress(out_dir, expected_fingerprint) if resume else 0
    if rows_done >= n_rows and n_rows > 0:
        print(f"already complete: {rows_done}/{n_rows} rows", flush=True)
        existing = np.lib.format.open_memmap(out_dir / f"concat_L{layers[0]}.npy", mode="r")
        write_index(out_dir, rows, model_name, layers, concat_positions,
                    peptide_slots, pseudoseq_slots, pseudoseq_indices, linker,
                    n_slots, dim=int(existing.shape[-1]))
        return {"n_rows": n_rows, "rows_done": rows_done, "resumed_from": rows_done,
                "seconds": 0.0, "sequences_per_second": None, "complete": True}

    model, alphabet, device = load_model(model_name, device)
    batch_converter = alphabet.get_batch_converter()
    dim = int(model.embed_dim)
    max_layer = max(layers)
    assert max_layer <= model.num_layers, (
        f"{model_name} has {model.num_layers} layers, cannot request layer {max_layer}"
    )

    needed, free = check_free_space(out_dir, n_rows, n_slots, layers, dim=dim)
    print(f"model {model_name}: {model.num_layers} layers, d={dim}, device={device}",
          flush=True)
    print(f"output: {len(layers)} x {(n_rows, n_slots, dim)} fp16 "
          f"= {needed / 2**30:.2f} GiB, {free / 2**30:.2f} GiB free at {out_dir}",
          flush=True)
    if free < needed:
        raise OSError(f"need {needed / 2**30:.2f} GiB at {out_dir}, only "
                      f"{free / 2**30:.2f} GiB free")

    arrays = open_outputs(out_dir, layers, n_rows, n_slots, dim=dim,
                          resume=rows_done > 0)
    write_index(out_dir, rows, model_name, layers, concat_positions,
                peptide_slots, pseudoseq_slots, pseudoseq_indices, linker,
                n_slots, dim=dim)
    use_autocast = device.type == "cuda"
    token_positions = torch.tensor([p + 1 for p in concat_positions], dtype=torch.long)

    if rows_done:
        print(f"resuming at row {rows_done}/{n_rows}", flush=True)
    started = time.monotonic()
    rows_at_start = rows_done
    last_checkpoint = rows_done
    alignment_checked = False

    for batch_start in range(rows_done, n_rows, batch_size):
        batch_rows = rows[batch_start:batch_start + batch_size]
        data = [(str(row["source_row"]),
                 build_sequence(row[HLA_COLUMN], row[PEPTIDE_COLUMN], linker))
                for row in batch_rows]
        _, _, tokens = batch_converter(data)

        # Assertion 2, on the first batch actually pushed through the model.
        if not alignment_checked:
            n_checked = assert_token_alignment(
                tokens, alphabet, batch_rows, concat_positions,
                peptide_slots, pseudoseq_slots, linker)
            print(f"[assert 2/2] BOS/EOS stripped; concat 0..{HLA_LEN - 1} is hla_seq, "
                  f"{HLA_LEN + len(linker)}..{HLA_LEN + len(linker) + PEPTIDE_LEN - 1} is the peptide; "
                  f"gathered slots decode to peptide + pseudoseq for all {n_checked} "
                  f"rows of batch 0: PASS", flush=True)
            alignment_checked = True

        tokens = tokens.to(device, non_blocking=True)
        with torch.inference_mode():
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(tokens, repr_layers=layers, return_contacts=False)
            else:
                out = model(tokens, repr_layers=layers, return_contacts=False)
            positions = token_positions.to(device)
            for layer in layers:
                kept = out["representations"][layer].index_select(1, positions)
                arrays[layer][batch_start:batch_start + len(batch_rows)] = (
                    kept.to(torch.float16).cpu().numpy())
        del out

        rows_done = batch_start + len(batch_rows)
        if rows_done - last_checkpoint >= checkpoint_every:
            for array in arrays.values():
                array.flush()
            write_progress(out_dir, rows_done, n_rows, expected_fingerprint, model_name, layers)
            last_checkpoint = rows_done
            elapsed = time.monotonic() - started
            rate = (rows_done - rows_at_start) / elapsed if elapsed else float("nan")
            print(f"  {rows_done}/{n_rows} rows, {rate:.2f} seq/s", flush=True)
            if progress_callback is not None:
                progress_callback(rows_done, n_rows, rate)

    for array in arrays.values():
        array.flush()
    write_progress(out_dir, rows_done, n_rows, expected_fingerprint, model_name, layers)
    elapsed = time.monotonic() - started
    processed = rows_done - rows_at_start
    rate = processed / elapsed if elapsed else float("nan")
    return {
        "n_rows": n_rows,
        "rows_done": rows_done,
        "resumed_from": rows_at_start,
        "processed": processed,
        "seconds": elapsed,
        "sequences_per_second": rate,
        "complete": rows_done >= n_rows,
        "out_dir": str(out_dir),
        "slots": n_slots,
    }


# --- CLI ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows (development runs)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--linker", default=LINKER)
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore any existing checkpoint and overwrite")
    args = parser.parse_args()

    summary = extract_embeddings(
        dataset_path=args.dataset, out_dir=args.out_dir, model_name=args.model,
        layers=args.layers, batch_size=args.batch_size, limit=args.limit,
        device=args.device, checkpoint_every=args.checkpoint_every,
        linker=args.linker, resume=args.resume)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
