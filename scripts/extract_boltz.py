#!/usr/bin/env python3
"""Boltz-2 trunk-embedding extraction for peptide-HLA complexes (arm B*).

Produces caches shaped ``(n_rows, 43, D)`` that drop into the existing ladder
(`scripts/arm_features.py`, `train_mlp.py`) with no change to the training code:
the 43 slots are the same 9 peptide residues + 34 NetMHCpan contact positions
that the ESM-2 arms use, so B-arms and A-arms are scored on identical positions
of identical rows.

Why the trunk and not Boltz-2's affinity module
-----------------------------------------------
Boltz-2's affinity head cannot take a peptide. Its PairFormer attends only over
protein-ligand and intra-ligand interactions, and the CLI requires the affinity
``binder`` to be a SMILES/CCD ligand of <=128 atoms (~56 recommended); affinity
training discarded ligands above 50 heavy atoms. This dataset's 9-mers are
48-100 heavy atoms (median 75). So we build our own head on the trunk, which is
the PreFold-dG recipe (Bioinformatics 2026, PMID 42635209).

What gets cached
----------------
``boltz predict --write_embeddings`` writes ``embeddings_{id}.npz`` holding the
trunk's single representation ``s`` [L, 384] and pair representation ``z``
[L, L, 128], with L = 182 + 9 = 191 tokens.

  boltz_S.npy    (n, 43, 384)  trunk single representation at the 43 slots
  boltz_Z.npy    (n, 43, 128)  interface pair contraction, 1/D^2 weighted
  boltz_ZU.npy   (n, 43, 128)  same contraction with uniform weights (ablation)
  index.json     row -> (hla_seq, peptide), source_row, slot layout

The ``z`` reduction, and why it is not PreFold-dG's
---------------------------------------------------
PreFold-dG pools an outer product down to a fixed [384x384] block because its
complexes vary in length. Ours never do: always 182 + 9, always the same
geometry. So we keep per-position structure instead, which matters because
peptide-HLA binding is dominated by anchor residues (P2, P9) and pooling across
peptide positions destroys exactly that signal. `docs/02_next_steps.md` makes
the same argument for why the one-hot arm is flattened rather than mean-pooled.

What we do keep from PreFold-dG is the good part: inverse-square distance
weighting restricted to *interchain* pairs. In a two-chain pMHC system the
interchain block is precisely the peptide-groove interface, so for peptide slot
p and contact position q,

    zc[p]     = sum_q w[p,q] z[pep_p, hla_q] / sum_q w[p,q]
    zc[9 + q] = sum_p w[p,q] z[pep_p, hla_q] / sum_p w[p,q]
    w[p,q]    = 1 / max(d(p,q), d_min)^2

with d the CA-CA distance from the predicted structure. Using the predicted
coordinates rather than the distogram expectation is both simpler and sharper,
and costs nothing extra because the CLI writes the structure anyway.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_embeddings import (  # noqa: E402  reuse, never restate, the layout
    HLA_COLUMN,
    HLA_LEN,
    PEPTIDE_COLUMN,
    PEPTIDE_LEN,
    PSEUDOSEQ_COLUMN,
    PSEUDOSEQ_INDICES,
    SOURCE_ROW_OFFSET,
    assert_pseudoseq_indices,
    slot_layout,
)


def load_rows_any(dataset_path, limit=None, sheet=0):
    """``extract_embeddings.load_rows`` but accepting CSV as well as XLSX.

    The upstream loader is ``pd.read_excel`` only, and the clean target lives in
    a CSV (the committed XLSX has an Excel-mangled half-life column that is not
    losslessly invertible -- see scripts/pilot_subset.py). Everything else here
    follows the upstream convention exactly, including SOURCE_ROW_OFFSET, so
    array index i holds source_row i + SOURCE_ROW_OFFSET.
    """
    import pandas as pd

    path = Path(dataset_path)
    frame = (pd.read_csv(path) if path.suffix.lower() == ".csv"
             else pd.read_excel(path, sheet_name=sheet))
    for column in (HLA_COLUMN, PEPTIDE_COLUMN, PSEUDOSEQ_COLUMN):
        if column not in frame.columns:
            raise ValueError(f"dataset is missing column {column!r}")
    frame = frame.reset_index(drop=True)
    frame["source_row"] = np.arange(
        SOURCE_ROW_OFFSET, len(frame) + SOURCE_ROW_OFFSET, dtype=np.int64)

    lengths = set(frame[HLA_COLUMN].str.len().unique())
    assert lengths == {HLA_LEN}, f"expected all hla_seq to be {HLA_LEN} aa, saw {sorted(lengths)}"
    lengths = set(frame[PEPTIDE_COLUMN].str.len().unique())
    assert lengths == {PEPTIDE_LEN}, f"expected all peptides to be {PEPTIDE_LEN} aa, saw {sorted(lengths)}"

    if limit is not None:
        frame = frame.iloc[:limit].copy()
    rows = frame[[HLA_COLUMN, PEPTIDE_COLUMN, PSEUDOSEQ_COLUMN, "source_row"]].to_dict("records")
    return frame, rows

TOKEN_S = 384
TOKEN_Z = 128
N_TOKENS = HLA_LEN + PEPTIDE_LEN          # 191
D_MIN = 3.0                                # floor on CA-CA distance, angstrom
ARRAYS = {"boltz_S.npy": TOKEN_S, "boltz_Z.npy": TOKEN_Z, "boltz_ZU.npy": TOKEN_Z}
CHECKPOINT_EVERY = 200


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def write_yaml(path: Path, hla_seq: str, peptide: str, hla_msa: str | None) -> None:
    """One Boltz YAML. The peptide always gets ``msa: empty``.

    A 9-mer alignment carries no information, and resolving 5,633 distinct
    peptides through the MSA server would dominate the run for no benefit.
    """
    msa_line = f"      msa: {hla_msa}\n" if hla_msa is not None else ""
    path.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {hla_seq}\n"
        f"{msa_line}"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {peptide}\n"
        "      msa: empty\n"
    )


def record_id(source_row: int) -> str:
    """Stable per-complex id. Boltz names its outputs after the YAML stem."""
    return f"r{int(source_row):06d}"


def msa_path_for(hla_seq: str, msa_dir: Path | None) -> str | None:
    """Precomputed .a3m for this HLA, or ``empty``, or None to use the server.

    Only 75 unique HLA sequences exist, so alignments are generated once and
    shared by every row of that allele.
    """
    if msa_dir is None:
        return "empty"
    digest = hashlib.sha256(hla_seq.encode()).hexdigest()[:16]
    candidate = Path(msa_dir) / f"{digest}.a3m"
    return str(candidate) if candidate.exists() else "empty"


def build_batch_inputs(rows, in_dir: Path, msa_dir: Path | None) -> list[str]:
    if in_dir.exists():
        shutil.rmtree(in_dir)
    in_dir.mkdir(parents=True)
    ids = []
    for row in rows:
        rid = record_id(row["source_row"])
        write_yaml(in_dir / f"{rid}.yaml", row["hla_seq"], row["peptide"],
                   msa_path_for(row["hla_seq"], msa_dir))
        ids.append(rid)
    return ids


# --------------------------------------------------------------------------- #
# reduction
# --------------------------------------------------------------------------- #
def read_ca_coords(structure_path: Path) -> np.ndarray:
    """CA coordinates in token order, shape (191, 3).

    Chain A is the 182-residue HLA, chain B the 9-mer, matching the YAML order.
    """
    per_chain: dict[str, list[tuple[int, np.ndarray]]] = {}
    for line in structure_path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        chain = line[21]
        resseq = int(line[22:26])
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        per_chain.setdefault(chain, []).append((resseq, xyz))
    if "A" not in per_chain or "B" not in per_chain:
        raise ValueError(f"{structure_path} has chains {sorted(per_chain)}, expected A and B")
    hla = [xyz for _, xyz in sorted(per_chain["A"])]
    pep = [xyz for _, xyz in sorted(per_chain["B"])]
    if len(hla) != HLA_LEN or len(pep) != PEPTIDE_LEN:
        raise ValueError(f"{structure_path}: chain A {len(hla)} res, chain B {len(pep)} res; "
                         f"expected {HLA_LEN} and {PEPTIDE_LEN}")
    return np.asarray(hla + pep, dtype=np.float64)


def interface_weights(coords: np.ndarray) -> np.ndarray:
    """1/d^2 weights over the (9 peptide) x (34 contact) interchain block."""
    pep = coords[HLA_LEN:]                      # (9, 3)
    hla = coords[np.asarray(PSEUDOSEQ_INDICES)]  # (34, 3)
    d = np.linalg.norm(pep[:, None, :] - hla[None, :, :], axis=-1)
    return 1.0 / np.maximum(d, D_MIN) ** 2


def reduce_s(s: np.ndarray, concat_positions: np.ndarray) -> np.ndarray:
    """(191, 384) -> (43, 384) by gathering the 43 slot positions."""
    if s.shape != (N_TOKENS, TOKEN_S):
        raise ValueError(f"s has shape {s.shape}, expected {(N_TOKENS, TOKEN_S)}")
    return s[concat_positions]


def reduce_z(z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(191, 191, 128) -> (43, 128) interface contraction, weights (9, 34).

    Slots 0..8 contract each peptide position over the 34 contact positions;
    slots 9..42 contract each contact position over the 9 peptide positions.
    Both directions of the pair tensor are averaged, since z is not symmetric.
    """
    if z.shape != (N_TOKENS, N_TOKENS, TOKEN_Z):
        raise ValueError(f"z has shape {z.shape}, expected {(N_TOKENS, N_TOKENS, TOKEN_Z)}")
    pep_idx = np.arange(HLA_LEN, N_TOKENS)
    hla_idx = np.asarray(PSEUDOSEQ_INDICES)
    block = 0.5 * (z[np.ix_(pep_idx, hla_idx)] + z[np.ix_(hla_idx, pep_idx)].transpose(1, 0, 2))
    w = weights.astype(np.float64)
    pep_side = np.einsum("pq,pqd->pd", w, block) / w.sum(axis=1, keepdims=True)
    hla_side = np.einsum("pq,pqd->qd", w, block) / w.sum(axis=0)[:, None]
    return np.concatenate([pep_side, hla_side], axis=0)


def reduce_one(emb_path: Path, structure_path: Path,
               concat_positions: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(emb_path) as handle:
        s = np.asarray(handle["s"], dtype=np.float32)
        z = np.asarray(handle["z"], dtype=np.float32)
    s = np.squeeze(s)
    z = np.squeeze(z)
    coords = read_ca_coords(structure_path)
    w = interface_weights(coords)
    return {
        "boltz_S.npy": reduce_s(s, concat_positions),
        "boltz_Z.npy": reduce_z(z, w),
        "boltz_ZU.npy": reduce_z(z, np.ones_like(w)),
    }


# --------------------------------------------------------------------------- #
# cache plumbing (mirrors scripts/extract_embeddings.py conventions)
# --------------------------------------------------------------------------- #
def fingerprint(rows, n_slots: int, extra: str) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(f"{row['source_row']}|{row['hla_seq']}|{row['peptide']}\n".encode())
    h.update(f"slots={n_slots}|{extra}".encode())
    return h.hexdigest()


def open_outputs(out_dir: Path, n_rows: int, n_slots: int, resume: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, dim in ARRAYS.items():
        path = out_dir / name
        if resume and path.exists():
            arrays[name] = np.lib.format.open_memmap(path, mode="r+")
        else:
            arrays[name] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float16, shape=(n_rows, n_slots, dim))
    return arrays


def write_index(out_dir: Path, rows, peptide_slots, pseudoseq_slots, fp: str) -> None:
    payload = {
        "n_rows": len(rows),
        "source_row": [int(r["source_row"]) for r in rows],
        "pairs": [[r["hla_seq"], r["peptide"]] for r in rows],
        "peptide_slots": list(map(int, peptide_slots)),
        "pseudoseq_slots": list(map(int, pseudoseq_slots)),
        "pseudoseq_indices": list(map(int, PSEUDOSEQ_INDICES)),
        "n_slots": len(peptide_slots) + len(pseudoseq_slots),
        "model": "boltz2",
        "arrays": {k: v for k, v in ARRAYS.items()},
        "fingerprint": fp,
    }
    tmp = out_dir / "index.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(out_dir / "index.json")


def read_progress(out_dir: Path, fp: str) -> int:
    path = out_dir / "progress.json"
    if not path.exists():
        return 0
    state = json.loads(path.read_text())
    if state.get("fingerprint") != fp:
        raise RuntimeError(
            f"{path} was written for a different run (fingerprint mismatch). "
            "Use --no-resume to start a fresh cache, or point --out-dir elsewhere.")
    return int(state.get("rows_done", 0))


def write_progress(out_dir: Path, rows_done: int, fp: str) -> None:
    tmp = out_dir / "progress.json.tmp"
    tmp.write_text(json.dumps({"rows_done": rows_done, "fingerprint": fp}) + "\n")
    tmp.replace(out_dir / "progress.json")


def check_free_space(out_dir: Path, n_rows: int, n_slots: int) -> dict:
    need = sum(n_rows * n_slots * dim * 2 for dim in ARRAYS.values())
    free = shutil.disk_usage(out_dir).free
    if free < need * 1.15:
        raise RuntimeError(f"need ~{need/1024**3:.2f} GiB for the cache but only "
                           f"{free/1024**3:.2f} GiB is free at {out_dir}")
    return {"cache_GiB": round(need / 1024**3, 3), "free_GiB": round(free / 1024**3, 2)}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def boltz_cmd(in_dir: Path, out_dir: Path, cache: Path, *, sampling_steps: int,
              recycling_steps: int, use_msa_server: bool, num_workers: int,
              no_kernels: bool = True,
              extra_args: list[str] | None = None) -> list[str]:
    """The argv for one `boltz predict` invocation."""
    cmd = [
        "boltz", "predict", str(in_dir),
        "--out_dir", str(out_dir),
        "--cache", str(cache),
        "--write_embeddings",
        "--output_format", "pdb",
        "--sampling_steps", str(sampling_steps),
        "--recycling_steps", str(recycling_steps),
        "--diffusion_samples", "1",
        "--num_workers", str(num_workers),
        "--override",
        *(extra_args or []),
    ]
    if no_kernels:
        cmd.append("--no_kernels")
    if use_msa_server:
        cmd.append("--use_msa_server")
    return cmd


def run_boltz_parallel(jobs: list[tuple[Path, Path]], cache: Path, **kw) -> None:
    """Run several `boltz predict` processes concurrently on one GPU.

    This is where most of the money is saved. One process badly under-occupies
    any GPU at 191 tokens: measured on Modal, an L4 goes from 14.4 to 6.1
    s/complex between 1 and 4 concurrent workers, and an H100 from 8.7 to 2.0
    at 8. Each worker holds its own ~2.8-3.5 GiB copy of the model, so the
    ceiling is device memory -- and, as the benchmark also found, the
    container's CPU allocation, since every worker spawns its own dataloader
    threads.
    """
    procs = [(subprocess.Popen(boltz_cmd(i, o, cache, **kw),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True), i)
             for i, o in jobs]
    failures = []
    for proc, in_dir in procs:
        out, err = proc.communicate()
        if proc.returncode != 0:
            failures.append(f"{in_dir.name}:\n{out[-800:]}\n{err[-2500:]}")
    if failures:
        raise RuntimeError("boltz predict failed in %d/%d workers\n%s"
                           % (len(failures), len(procs), "\n---\n".join(failures)))


def run_boltz(in_dir: Path, out_dir: Path, cache: Path, *, sampling_steps: int,
              recycling_steps: int, use_msa_server: bool, num_workers: int,
              no_kernels: bool = True, extra_args: list[str] | None = None) -> None:
    cmd = [
        "boltz", "predict", str(in_dir),
        "--out_dir", str(out_dir),
        "--cache", str(cache),
        "--write_embeddings",
        "--output_format", "pdb",
        "--sampling_steps", str(sampling_steps),
        "--recycling_steps", str(recycling_steps),
        "--diffusion_samples", "1",
        "--num_workers", str(num_workers),
        "--override",
        *(extra_args or []),
    ]
    if no_kernels:
        # Boltz-2 2.2.1 imports cuequivariance_torch for triangle
        # multiplication and raises ModuleNotFoundError if it is absent rather
        # than falling back. --no_kernels takes the pure-PyTorch path. Measured
        # in BENCHMARK.md; the kernel path is faster but its wheels are built
        # per CUDA version and the job image ships CUDA 13.
        cmd.append("--no_kernels")
    if use_msa_server:
        cmd.append("--use_msa_server")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("boltz predict failed\n"
                           f"stdout tail:\n{proc.stdout[-2000:]}\n"
                           f"stderr tail:\n{proc.stderr[-4000:]}")


def locate_outputs(boltz_out: Path, rid: str) -> tuple[Path, Path]:
    """Find this record's embeddings npz and model-0 structure."""
    emb = next(boltz_out.rglob(f"embeddings_{rid}.npz"), None)
    pdb = next(boltz_out.rglob(f"{rid}_model_0.pdb"), None)
    if emb is None or pdb is None:
        raise FileNotFoundError(
            f"missing outputs for {rid}: embeddings={emb}, structure={pdb}")
    return emb, pdb


def extract(dataset_path, out_dir, *, source_rows=None, msa_dir=None,
            boltz_cache=None, work_dir=None, batch_size=32, sampling_steps=10,
            recycling_steps=3, use_msa_server=False, num_workers=2,
            workers=4, no_kernels=True, limit=None, resume=True,
            progress_callback=None) -> dict:
    """Extract and cache Boltz-2 trunk features. Returns a summary dict.

    Takes every path as an argument and reads nothing from the environment, so
    a Hugging Face Job entrypoint can mount storage wherever it likes.
    """
    out_dir = Path(out_dir)
    work_dir = Path(work_dir or (out_dir / "_work"))
    boltz_cache = Path(boltz_cache or (out_dir / "_boltz_cache"))
    boltz_cache.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    frame, rows = load_rows_any(dataset_path, limit=None)

    # Assertion 1: the contact indices really do reconstruct hla_pseudoseq, for
    # every allele. Runs before any GPU time is spent, as in the ESM-2
    # extractor, so a bad index list costs nothing.
    assert_pseudoseq_indices(frame, PSEUDOSEQ_INDICES)

    if source_rows is not None:
        wanted = {int(v) for v in source_rows}
        rows = [r for r in rows if int(r["source_row"]) in wanted]
        missing = wanted - {int(r["source_row"]) for r in rows}
        if missing:
            raise ValueError(f"{len(missing)} requested source_rows are not in the dataset")
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no rows selected")

    concat_positions, peptide_slots, pseudoseq_slots = slot_layout(linker="")
    concat_positions = np.asarray(concat_positions)
    n_slots = len(peptide_slots) + len(pseudoseq_slots)

    fp = fingerprint(rows, n_slots, f"boltz2|steps={sampling_steps}|"
                                    f"recycle={recycling_steps}|msa={bool(msa_dir) or use_msa_server}")
    space = check_free_space(out_dir if out_dir.exists() else Path("."), len(rows), n_slots)
    arrays = open_outputs(out_dir, len(rows), n_slots, resume)
    write_index(out_dir, rows, peptide_slots, pseudoseq_slots, fp)
    done = read_progress(out_dir, fp) if resume else 0

    summary = {"n_rows": len(rows), "n_slots": n_slots, "resumed_at": done,
               "sampling_steps": sampling_steps, "recycling_steps": recycling_steps,
               "workers": workers,
               **space, "batches": [], "max_abs": {k: 0.0 for k in ARRAYS}}
    # The cache is float16 (ceiling 65504). ESM-2 layer 15 peaked at 223, but
    # Boltz-2's z is a different distribution and its range is not documented,
    # so the magnitude is tracked and overflow is an error rather than a silent
    # inf. Caught by scripts/test_extract_boltz.py.
    f16_max = float(np.finfo(np.float16).max)
    started = time.monotonic()

    for start in range(done, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        boltz_out = work_dir / "boltz_out"
        if boltz_out.exists():
            shutil.rmtree(boltz_out)

        # Split the batch across `workers` concurrent boltz processes on the one
        # GPU. This is the main cost lever: measured on Modal, an L4 drops from
        # 14.4 to 6.1 s/complex going from 1 to 4 workers. Round-robin rather
        # than contiguous slices so every worker gets a mix of alleles and no
        # single worker inherits a pathological tail.
        n_workers = max(1, min(workers, len(chunk)))
        sub_chunks = [chunk[w::n_workers] for w in range(n_workers)]
        jobs, ids_per_worker = [], []
        for w, sub in enumerate(sub_chunks):
            if not sub:
                continue
            in_dir = work_dir / f"yaml_{w}"
            ids_per_worker.append(build_batch_inputs(sub, in_dir, msa_dir))
            jobs.append((in_dir, boltz_out / f"w{w}"))

        t0 = time.monotonic()
        run_boltz_parallel(jobs, boltz_cache, sampling_steps=sampling_steps,
                           recycling_steps=recycling_steps,
                           use_msa_server=use_msa_server,
                           num_workers=num_workers, no_kernels=no_kernels)
        predict_seconds = time.monotonic() - t0

        # Rebuild the batch-local order so each record lands at its own row.
        ids = [None] * len(chunk)
        for w, worker_ids in enumerate(ids_per_worker):
            for j, rid in enumerate(worker_ids):
                ids[j * n_workers + w] = rid
        if any(r is None for r in ids):
            raise RuntimeError("worker id reassembly left a gap")

        for offset, rid in enumerate(ids):
            emb, pdb = locate_outputs(boltz_out, rid)
            reduced = reduce_one(emb, pdb, concat_positions)
            for name, block in reduced.items():
                peak = float(np.abs(block).max())
                if peak > f16_max:
                    raise OverflowError(
                        f"{name} row {start + offset} peaks at {peak:.4g}, above the "
                        f"float16 ceiling {f16_max:.0f}; store this array as float32")
                summary["max_abs"][name] = max(summary["max_abs"][name], peak)
                arrays[name][start + offset] = block.astype(np.float16)

        batch_stat = {"start": start, "n": len(chunk),
                      "predict_seconds": round(predict_seconds, 2),
                      "seconds_per_complex": round(predict_seconds / len(chunk), 3)}
        summary["batches"].append(batch_stat)
        if progress_callback:
            progress_callback(batch_stat)
        print(json.dumps(batch_stat), flush=True)

        if (start + len(chunk) - done) % CHECKPOINT_EVERY < batch_size:
            for array in arrays.values():
                array.flush()
            write_progress(out_dir, start + len(chunk), fp)
        shutil.rmtree(boltz_out, ignore_errors=True)

    for array in arrays.values():
        array.flush()
    write_progress(out_dir, len(rows), fp)
    elapsed = time.monotonic() - started
    processed = len(rows) - done
    summary |= {
        "wall_seconds": round(elapsed, 2),
        "processed": processed,
        "seconds_per_complex": round(elapsed / processed, 3) if processed else None,
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="rasmussen CSV or XLSX")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-rows", type=Path,
                    help="JSON or newline list of source_row values to extract")
    ap.add_argument("--msa-dir", default=None,
                    help="directory of <sha256[:16]>.a3m files, one per unique hla_seq")
    ap.add_argument("--boltz-cache", default=os.environ.get("BOLTZ_CACHE"))
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--use-msa-server", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2,
                    help="boltz dataloader workers per process")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent boltz processes on the one GPU; the main "
                         "cost lever (L4: 14.4 s/cx at 1, 6.1 at 4). Capped by "
                         "device memory (~3 GiB each) and container CPU")
    ap.add_argument("--use-kernels", dest="no_kernels", action="store_false",
                    help="use the cuequivariance triangle kernels (needs "
                         "cuequivariance-torch matching the image's CUDA)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    source_rows = None
    if args.source_rows:
        text = args.source_rows.read_text().strip()
        source_rows = json.loads(text) if text.startswith("[") else [
            int(line) for line in text.split() if line]

    summary = extract(args.dataset, args.out_dir, source_rows=source_rows,
                      msa_dir=args.msa_dir, boltz_cache=args.boltz_cache,
                      work_dir=args.work_dir, batch_size=args.batch_size,
                      sampling_steps=args.sampling_steps,
                      recycling_steps=args.recycling_steps,
                      use_msa_server=args.use_msa_server,
                      num_workers=args.num_workers, workers=args.workers,
                      no_kernels=args.no_kernels,
                      limit=args.limit, resume=args.resume)
    print(json.dumps({k: v for k, v in summary.items() if k != "batches"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
