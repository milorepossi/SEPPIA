# Task 4 — Boltz-2 trunk-embedding arm (B*)

Technical report for `scripts/extract_boltz.py`, `scripts/pilot_subset.py`,
`hf/bench_boltz.py` and `hf/extract_job.py`.

Status: **benchmarked on Hugging Face L4, extractor validated, pilot selected.**
Measured cost in [`BENCHMARK.md`](../BENCHMARK.md).

---

## 1. The question this arm asks

`RESULTS.md` establishes that frozen ESM-2 representations do not beat a plain
one-hot encoding of the same 43 positions (A0 0.771, best PLM arm 0.767). The
natural follow-up is not "a bigger language model" but "a representation that
sees the *complex*".

Boltz-2's single representation is 384-dimensional against ESM-2 650M's 1280,
so Boltz-2 is **not** the richer per-residue featurizer. The only thing it adds
that no sequence model can express is its **pair** representation: a learned
embedding of residue *pairs*, conditioned on the actual peptide-HLA complex. If
the B arms are built only on the single representation, they are testing the
weak form of the hypothesis and should be expected to reproduce A3's result.

So arm `BZZ`, the interface pair contraction, carries the hypothesis. `BZS` is
its control.

## 2. Why not Boltz-2's own affinity module

Boltz-2 ships a binding-affinity head, and the obvious first instinct is to
point it at the peptide. That does not work, for three independent reasons:

- the affinity PairFormer attends **only** over protein-ligand and intra-ligand
  interactions (Boltz-2 preprint, §Architecture);
- the CLI requires the affinity `binder` to be a ligand chain given as SMILES or
  CCD, capped at 128 atoms with ~56 recommended;
- affinity training discarded ligands above 50 heavy atoms, and this dataset's
  9-mers are 48-100 heavy atoms (median 75).

The route is closed at the interface, the atom limit, and the training
distribution. We build our own head on the trunk instead, which is what
PreFold-dG does (Bioinformatics 2026, PMID 42635209).

## 3. What gets cached, and the shape decision

`boltz predict --write_embeddings` writes `embeddings_{id}.npz` with the trunk's
`s` `[1, 191, 384]` and `z` `[1, 191, 191, 128]`, both float32. L = 182 + 9.

The caches are deliberately shaped **`(n_rows, 43, D)`** — the same 43 slots
(9 peptide residues + 34 NetMHCpan contact positions) and the same row order as
the ESM-2 arms. That means:

- `train_mlp.py` needs **no change**; the B arms are selected with `--arm BZZ`
  and `--embeddings-dir <boltz cache>`;
- A arms and B arms are scored on identical positions of identical rows, so the
  five-split comparison stays paired;
- the slot count is read from `index.json`, not hardcoded, so `D` is free to
  differ per arm (384 for `BZS`, 128 for `BZZ`).

| File | Shape | Content |
|---|---|---|
| `boltz_S.npy` | `(n, 43, 384)` | trunk single representation at the 43 slots |
| `boltz_Z.npy` | `(n, 43, 128)` | interface pair contraction, 1/d² weighted |
| `boltz_ZU.npy` | `(n, 43, 128)` | same contraction, uniform weights (ablation) |

float16, so the full dataset would be 1.44 GB and the pilot 0.14 GB. The raw
`z` tensors are 16.6 MB per complex on disk, which is why the extractor reduces
and deletes per batch — keeping them for 28,166 rows would be 468 GB.

## 4. The `z` reduction, and the one place we deliberately diverge from PreFold-dG

PreFold-dG pools an outer product of the single representations down to a fixed
`[384x384]` block, weighting residue pairs by inverse-square distance and giving
**zero weight to intrachain pairs**. The pooling exists because its complexes
vary in length and the head needs a fixed-size input.

Our complexes never vary: always 182 + 9, always the same geometry. Copying the
pooling would therefore buy nothing and cost something real — **pooling across
peptide positions destroys anchor-residue identity**, and P2/P9 anchors carry
most of the binding signal in class I presentation. `docs/02_next_steps.md`
already makes exactly this argument for why A0 is flattened rather than
mean-pooled; the same logic applies here.

What we keep is the good part: the inverse-square **interchain** weighting. In a
two-chain pMHC system the interchain block *is* the peptide-groove interface, so
for peptide slot `p` and contact position `q`:

```
w[p,q]    = 1 / max(d(p,q), 3.0)^2
zc[p]     = sum_q w[p,q] * zblk[p,q] / sum_q w[p,q]      slots 0..8
zc[9 + q] = sum_p w[p,q] * zblk[p,q] / sum_p w[p,q]      slots 9..42
zblk[p,q] = 0.5 * (z[pep_p, hla_q] + z[hla_q, pep_p])
```

Both contraction directions are kept because `z` is not symmetric, and the two
directions are averaged before contracting.

`d` is the **CA-CA distance from the predicted structure**, not the distogram
expectation. The CLI writes the structure anyway, so this is free, and it is
sharper than a binned expectation. `BZZU` repeats the contraction with uniform
weights, which makes "does distance weighting earn its keep?" a one-line
ablation rather than a second extraction.

## 5. Assertions

Both run before any GPU time is spent, mirroring `docs/01`.

| Check | Where | Result |
|---|---|---|
| `hla_seq[PSEUDOSEQ_INDICES] == hla_pseudoseq`, every allele | `assert_pseudoseq_indices`, reused from `extract_embeddings` | **PASS, 75/75** |
| peptide tokens are exactly 182..190 | `slot_layout(linker="")` | **PASS** |
| `reduce_s` gathers exactly the 43 slot tokens | synthetic `s[i,:] = i` | **PASS** |
| both `reduce_z` contraction directions match an explicit hand calculation | synthetic `z[i,j,:] = 1000i + j` | **PASS** |
| 1/d² and uniform contractions differ | guards against `BZZU` being a duplicate arm | **PASS** |
| chain A has 182 residues and chain B has 9 in the predicted structure | `read_ca_coords` | enforced per complex |

The synthetic-tensor checks are the load-bearing ones: they validate the gather
and contraction indices themselves, not the layout they were derived from. An
off-by-one or a transposed contraction fails row 1.

## 6. Boltz token order matches the ESM-2 concat by construction

The YAML lists chain A (182-aa HLA) then chain B (9-mer), so Boltz's token order
is HLA-then-peptide. That is identical to the ESM-2 arms' concatenation with an
empty linker, which is the configured `LINKER = ""`. So `slot_layout()`'s
`concat_positions` index Boltz tokens directly, with no separate mapping to keep
in sync. Verified: `concat_positions[:9] == [182..190]`.

## 7. MSA handling

Only **75 unique HLA sequences** exist, so alignments are generated once and
shared by every row of that allele; `msa_path_for()` looks them up by a hash of
the sequence. The peptide always gets `msa: empty` — a 9-mer alignment carries
no information, and resolving 5,633 distinct peptides through the MSA server
would dominate the run for no modelling benefit.

**The benchmark and the pilot both run with `msa: empty` on the HLA too.** This
is a real limitation, not a design choice: Boltz-2 warns that single-sequence
mode degrades predictions. It is recorded here as the first thing to vary if the
pilot shows signal, because MSA depth is also a cost driver and the trade-off is
unmeasured.

## 8. Two things the benchmark taught the extractor

**`--no_kernels` is mandatory.** Boltz-2 2.2.1 imports
`cuequivariance_torch` for triangle multiplication and raises
`ModuleNotFoundError` rather than falling back. The first smoke run of the
extractor crashed on exactly this after the benchmark had already found it —
the lesson had not been propagated. It now is, and the flag is the default.

**Run few large jobs, not many small ones.** The 6.2 GB of Boltz-2 checkpoints
takes 116 s to fetch. Persisting them to a mounted bucket makes that a one-off,
so per-container overhead stops mattering, which is why the pilot runs as a
single job rather than sharded.

## 9. Target correction

`scripts/pilot_subset.py:audit_target()` records a bug in the existing pipeline.

The committed `.xlsx` had 42% of its `thalf_hours` cells mangled by Excel into
datetimes, and `split_dataset.py:half_life()` inverts them as
`float(day.month)`. `docs/02_next_steps.md` §0.1 states that inversion is
lossless, citing zero overlap between recovered and surviving string values.

It is not lossless. Values of the form `x.0Y` collapse: a true `1.05` was stored
as `2026-05-01` and recovers as `1.5`, because the hundredths-place leading zero
is lost. Both `1.5` and `1.05` parse to the same datetime, so the collapse is
between two *mangled* values — which is why an audit comparing against surviving
values could not see it.

| | |
|---|---|
| Rows affected | **26** of 28,166 (0.092%) |
| Distinct true values affected | 19 (1.04, 1.05, 1.06, 1.09, 2.05, …, 18.05) |
| Max absolute error | 0.81 h |
| Distinct values, clean CSV vs recovery | **944** vs 925 |

The clean CSV is ground truth. Split **membership** is unaffected in any way
that matters — 26 rows cannot move a moment-matching constraint on 28,166 — so
the committed splits stand and only the target is re-read, from
`DATA/rasmussen_clean.csv`. The bug is baked into the committed
`DATA/training_*.npz` targets (130 cells = 26 rows x 5 splits), so any arm
trained from those npz targets carries it.

## 10. Reproducing

```bash
# CPU, free: target audit + pilot subset selection
python scripts/pilot_subset.py

# measure the GPU cost before buying anything
hf jobs uv run hf/bench_boltz.py --flavor l4x1 --timeout 50m \
  --volume hf://buckets/<user>/boltz-cache:/bcache:rw \
  -- --cache /bcache/boltz --out /bcache/bench_report.json --skip-msa-server

# extract the pilot (one job, ~$5.89 at the measured end-to-end 9.41 s/complex)
hf jobs uv run hf/extract_job.py --flavor l4x1 --timeout 8h \
  --volume hf://buckets/<user>/boltz-cache:/bcache:rw \
  -- --dataset /bcache/inputs/rasmussen_clean.csv \
     --source-rows /bcache/inputs/pilot_source_rows.json \
     --out-dir /bcache/boltz_pilot --boltz-cache /bcache/boltz \
     --scripts-dir /bcache/scripts --batch-size 256 --max-seconds 28800

# score an arm on the existing five splits, head unchanged
python train_mlp.py --arm BZZ --embeddings-dir <boltz cache> --pooling pca:20
```

`--max-seconds` makes the budget enforceable from outside: the job flushes the
memmap cache and the resume checkpoint, then exits 0. A follow-up job continues
from exactly where it stopped, guarded by the fingerprint.

## 11. Open items

- The pilot extraction has not been run at scale yet. The 16-complex
  end-to-end validation passed (16/16 rows, 9.41 s/complex, peak magnitudes
  1383/294/215 against the float16 ceiling of 65504) and the 12-complex
  benchmark is in BENCHMARK.md.
- Each batch spawns a fresh `boltz predict` that reloads the 2.3 GB checkpoint,
  so `--batch-size` should be large (256) to amortise it. The 9.41 s/complex
  was measured at batch 16 and is therefore conservative.
- `A0` must be **re-scored on the pilot rows only** before any B-arm number is
  interpreted. B arms see 2,814 rows against A0-A3's 28,166, so a B-arm loss is
  not evidence about Boltz-2 until the comparison is row-matched.
- MSA depth is unvaried (`empty` throughout). It is both a quality and a cost
  variable and deserves one measured condition.
- The cuEquivariance kernel path is unmeasured and would reduce cost.
- `BZS` + `BZZ` concatenated is the closest analogue to PreFold-dG and is not
  yet a registered arm.
