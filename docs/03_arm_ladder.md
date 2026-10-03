# The arm ladder — one MLP, swappable input representation

Technical report for `train_mlp.py --arm`, `scripts/arm_features.py` and
`scripts/compare_arms.py`.

Status: **implemented and validated.** A0 runs end to end. A1–A3 are
implemented and their code path is verified, but the comparison itself is
pending the full embedding cache (§8).

- **To run it:** §9 — one process per layer, all five splits each.
- **If the cache was extracted on another machine:** §10.
- **The decision that actually matters:** §5, pooling.

---

## 1. What the ladder tests

Whether protein language model representations beat a plain sequence encoding at
predicting pMHC half-life. Every arm feeds the **same MLP block** with the
**same hyperparameter budget**; only the input differs.

| Arm | `--arm` | Input | Source |
|---|---|---|---|
| A0 | `onehot` | one-hot of the 43 kept positions, no PLM | in-memory, no cache |
| A1 | `L0` | ESM-2 layer 0 — context-free control | `concat_L0.npy` |
| A2 | `L15` | ESM-2 layer 15 — mid-stack | `concat_L15.npy` |
| A3 | `L33` | ESM-2 layer 33 — final | `concat_L33.npy` |
| A4 | — | fine-tune the last ESM-2 block | not implemented, see `02_next_steps.md` §6 |

The MLP is unchanged from the A0 work: `in → 256 → 128 → 1`, ReLU, dropout 0.2,
Adam at 1e-3, weight decay 1e-5, batch 256, ≤300 epochs, early stopping with
patience 25 on a validation slice carved out of train.

---

## 2. Design decision: extend the trainer, do not fork it

The ladder runs through the existing `train_mlp.py` rather than a second
trainer. The arms must share a hyperparameter budget, and one training loop
makes that true by construction instead of by discipline — a forked trainer
drifts the moment either copy is tuned. It also means one metrics format, one
set of figures and one error-decomposition path for all four arms.

`--arm onehot` is the default and reproduces the committed A0 results exactly,
so the change is additive: the existing entry point behaves identically.

The seam is one argument threaded through `features_and_target`:

```python
features = encoder(dataset) if encoder is not None else <one-hot of the two blocks>
```

Everything downstream — target transform, validation split, standardisation of
the target, training loop, metrics, figures — is untouched.

---

## 3. Data flow

```
DATA/training_{i}.npz   (source_row, peptide, hla_seq, hla_pseudoseq, thalf_hours)
         |
         |  rng.permutation -> train_idx / validation_idx     (test never touched)
         v
   arm == onehot ?
         |                                    \
         | yes: one_hot(peptide) +              \ no: CachedArmEncoder
         |      one_hot(hla_pseudoseq)           v
         |      -> (rows, 860)            embeddings/index.json
         |                                       |  row_positions()  source_row -> cache row
         |                                       |  verify_join()    rows carry the split's own sequences
         |                                       v
         |                                concat_L{0,15,33}.npy  ->  (rows, 43, 1280) fp16->fp32
         |                                       |
         |                                       |  .fit(train_idx rows only)   <- pca, if used
         |                                       |  apply_pooling()
         |                                       v
         |                                 (rows, features)
         v                                       v
         +---------------------+-----------------+
                               |
                   Standardiser().fit(x_train)   <- training rows only
                               |
                        MLP 256-128-1  ->  metrics.json, figures
```

Nothing is keyed by split anywhere in the feature path; a split is only ever a
set of `source_row` values to select.

---

## 4. Code structure

### `scripts/arm_features.py`

| Function | Role |
|---|---|
| `parse_pooling()` / `pooled_width()` | validate `flatten` / `mean` / `pca:K`, compute the feature count without materialising anything |
| `apply_pooling()` | `(rows, 43, D)` → `(rows, features)` |
| `fit_projection()` | one `D→K` PCA over the training rows' position vectors, by economy SVD; returns the explained-variance fraction |
| `open_arm()` | memory-map one layer's cache, cross-check its row count against `index.json` |
| `row_positions()` | `source_row` → cache position **through `index.json`**, then assert the `SOURCE_ROW_OFFSET` shortcut agrees |
| `verify_join()` | every joined row must carry the split's own `peptide` and `hla_seq` |
| `CachedArmEncoder` | the callable the trainer uses; `fit()` is a no-op unless pooling is `pca` |
| `Standardiser` | per-column centre/scale, constant columns left alone rather than divided by ~0 |
| `should_standardise()` | resolves the `auto` / `always` / `never` policy |

### `scripts/compare_arms.py`

Reads several runs' `metrics.json`, prints mean ± sd per arm plus **paired
per-split deltas** against a reference arm, and writes `ladder.csv`. It refuses
to tabulate runs whose `hidden`, `epsilon` or split count differ, since those
are no longer budget-matched.

---

## 5. The real decision is pooling, not the MLP

Each cached row is `(43, 1280)`. How that becomes a vector sets the first
layer's width, so it — not the network — is the live experimental choice.

| `--pooling` | A0 features | PLM features | First layer vs A0 | Peak RAM / split |
|---|---|---|---|---|
| `flatten` | 860 | 55,040 | **64×** the parameters | ~15 GiB (measured 14.5) |
| `mean` | 40 | 2,560 | 3×, discards position within each block | ~5 GiB |
| `pca:20` | 860 | **860** | **identical** | ~5 GiB |

Peak memory is dominated by materialising the training rows as float32
(`19,716 × 43 × 1280 × 4` = 4.3 GiB), not by the pooled width, so the three
pooling modes differ far less than their feature counts suggest. Only
`flatten` is substantially worse, because it also keeps standardised copies of
the full-width matrices.

`flatten` is the most literal reading of "only the input features differ", but
a 55,040 → 256 first layer is 14.1M parameters against A0's 220k, trained on
~17.7k rows. The arms then differ in capacity as well as in features, and the
comparison partly measures regularisation.

**`pca:20` is the recommended headline.** One `1280 → 20` projection is fitted
on the training rows' position vectors and applied at every position, so every
arm has exactly A0's 860 inputs and a byte-identical head. One projection shared
across positions rather than 43 separate ones: the positions inhabit the same
representation space, so pooling them uses 43× the samples and keeps features
comparable position to position, mirroring one-hot's identical 20-d code
everywhere.

It is also principled rather than a round number. Layer 0's embedding matrix is
**rank exactly 20** on this dataset's alphabet (condition number 2.3, measured
in `01_embedding_extraction.md` §4), so `pca:20` is **lossless for A1**. That
makes A0 vs A1 information-, dimension- and parameter-matched at once — a real
null hypothesis.

It is lossy for layers 15 and 33, which is the question the ladder actually
asks: *given one budget, which representation carries more signal?* Run
`flatten` as a secondary check on whether 20 components starve the deeper
layers, but report it as a separate experiment.

---

## 6. Standardisation, and why it is not optional

Layer 33 is recorded **after** `emb_layer_norm_after`; layer 15 is raw. Mean
absolute values are 0.145 and 2.344 — a 16× scale gap (§8.1 of
`01_embedding_extraction.md`). Without standardisation, A2 vs A3 partly measures
input scaling rather than representation quality.

`--standardise auto` (default) standardises the PLM arms and leaves one-hot
alone, which keeps A0's published numbers reproducible. `always` puts all four
arms through an identical pipeline; measured cost on A0 is small but not zero
(§7).

### Leakage discipline

Both fitted transforms see the **training rows only**:

- the PCA projection is fitted on `train_idx` rows, excluding the validation
  rows used for early stopping;
- the standardiser is fitted on `x_train` after the validation split;
- the test set enters only at final scoring, as before.

---

## 7. Validation

The cached-arm code path was exercised before any ESM-2 array existed, by
pointing it at `concat_onehot.npy` — a cached arm whose correct answer is known
independently.

| Check | Result |
|---|---|
| `--arm onehot` default unchanged | 860 features, RMSE 1.097, ρ 0.797, epoch 23/48 — identical to before |
| Cached path, `flatten` + `never` | **reproduces those numbers exactly** |
| Joined cache vs `train_mlp.one_hot` | **bitwise identical** over all 19,716 training rows |
| `row_positions` / `verify_join` on split 0 | 19,716 rows, every one matching the split's own sequences |
| Cache not covering a split | raises, naming the missing `source_row` values |
| Missing layer array | raises with the extraction command to run |
| `pca:0`, `pca:x`, `nope` | all rejected |

### Three findings worth carrying forward

**1. `mean` pooling is as destructive as predicted.** ρ 0.797 → **0.613** on
identical information. Averaging one-hot over the peptide block yields amino-acid
composition and erases the P2/P9 anchors. Do not use it for the headline.

**2. Preprocessing alone moves ρ by 0.025.** `pca:20` on one-hot retains 100% of
variance (`ev = 1.000`; at `D = 20` it is a lossless rotation) and still scored
ρ 0.772 against `flatten`'s 0.797. Same information, different conditioning.

> **Therefore: gaps below ~0.03 between arms are not evidence of a better
> representation.** Fix one pooling, apply it to every arm, and read small
> deltas as paired per-split wins rather than as mean differences — which is
> what `compare_arms.py` reports.

**3. Standardising one-hot is nearly free but not inert.** `flatten` + `always`
gave ρ 0.794 vs 0.797, with early stopping moving from epoch 23 to 68. Safe to
use for a strictly identical pipeline; just do not mix policies across arms and
then compare.

Measured sensitivity, split 0, for reference:

| pooling | standardise | features | RMSE | ρ | best epoch | explained var |
|---|---|---|---|---|---|---|
| `flatten` | never | 860 | 1.097 | 0.797 | 23/48 | — |
| `flatten` | always | 860 | 1.101 | 0.794 | 68/93 | — |
| `pca:20` | always | 860 | 1.160 | 0.772 | 33/58 | 1.000 |
| `pca:5` | always | 215 | 1.428 | 0.620 | 32/57 | 0.528 |
| `mean` | always | 40 | 1.438 | 0.613 | 50/75 | — |

---

## 8. Status and what is pending

**The embedding cache is being built.** A1–A3 cannot run without
`concat_L{0,15,33}.npy`. The full extraction is running locally on the P100 at
~21 seq/s — consistent with the 22.67 seq/s measured on 500 rows — for 8.66 GiB
across the three layers, about 21 minutes total. Modal is therefore not on the
critical path for a first result.

Once it lands, run the ladder as described in §9.

### Known limitations, to settle before reading the comparison

1. **One seed per split.** The reported sd mixes split difficulty with seed
   noise, and finding 2 says preprocessing noise is ~0.025 in ρ. ≥3 seeds per
   (arm, split) is needed before any small gap is interpretable. `--seed` shifts
   the base; the loop uses `seed + split_index`.
2. **`unseen_hla` is keyed on `hla_pseudoseq`, not `hla_seq`** (see
   `02_next_steps.md` §0.4). On splits 1 and 4 this misclassifies 382 rows per
   split, contaminating both that stratum and `seen_both`. Fix before comparing
   per-stratum generalisation across arms.
3. **A0 and A1 share a ceiling.** Both see only the 34 pseudosequence
   positions, so the two `C67S` alleles (756 rows) are indistinguishable to
   them; A2 and A3 do see the difference. Do not read the whole A1→A2 gain as
   "context helps".
4. **The subsets are not a partition.** `seen_both + unseen_peptide +
   unseen_hla` exceeds the test row count, so ~99 rows are double-counted.
5. **Spearman should stay the headline.** RMSE is dominated by the censored zero
   stratum and the long tail, where the target itself is least trustworthy
   (`02_next_steps.md` §0.1).

### The contrasts that will matter

- **A1 vs A3** — does context help beyond amino-acid identity? This is the
  experiment.
- **A0 vs A1** — a pipeline check, not a result. Under `pca:20` they are
  information-matched, so they should land within noise. A large gap means a
  bug or a conditioning problem, not a discovery.

---

## 9. Running the ladder — Slurm job array

`slurm/ladder.sbatch` is a seven-task array, one task per (arm, pooling) pair,
each training all five splits:

```bash
sbatch slurm/ladder.sbatch              # all 7
sbatch --array=1-3 slurm/ladder.sbatch  # just the pca:20 arms
sbatch --array=0 slurm/ladder.sbatch    # just A0, a 2-minute smoke test
```

| Task | Arm | Pooling | Output |
|---|---|---|---|
| 0 | `onehot` | — | `RESULTS/A0_onehot` |
| 1–3 | `L0`, `L15`, `L33` | `pca:20` | `RESULTS/A{1,2,3}_*_pca20` |
| 4–6 | `L0`, `L15`, `L33` | `flatten` | `RESULTS/A{1,2,3}_*_flatten` |

`--partition=tau --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00`.
48G covers `flatten`'s ~15 GiB peak; the pca tasks need ~5 GiB. `PROJECT`,
`PYTHON`, `EMBEDDINGS`, `SPLITS_DIR` and `RESULTS` are all overridable from the
environment.

Then:

```bash
python scripts/compare_arms.py RESULTS/A0_onehot RESULTS/A1_L0_pca20 \
    RESULTS/A2_L15_pca20 RESULTS/A3_L33_pca20 --csv RESULTS/ladder_pca20.csv
```

### Trap: submitting from inside an allocation

`sbatch` exports the submitting environment. Submitted from inside an
interactive `srun`/`salloc` shell, that job's variables leak into the new one
and contradict its allocation — all seven tasks failed instantly with:

```
srun: error: CPU binding outside of job step allocation, allocated CPUs are: 0x00AA00AA
srun: error: Unable to satisfy cpu bind request
No devices were found
```

The script therefore unsets `SLURM_CPU_BIND*`, `SLURM_STEP_GPUS`,
`SLURM_GPUS_ON_NODE` and `GPU_DEVICE_ORDINAL`, and rebuilds
`CUDA_VISIBLE_DEVICES` from `SLURM_JOB_GPUS`. It also does **not** wrap the
python call in `srun`: for a single-task job that adds nothing and is what
actually enforces the stale binding.

### Sharding: one layer per task, not one split per task

**Shard by layer, not by split.** Each task already does all five splits, so
one task per layer is the right granularity and needs no extra flags. Each
writes a **complete five-split `metrics.json`**, so `compare_arms.py` reads the
output directories directly and **nothing needs merging**. Sharding at the
split level would fragment `metrics.json` and need a merge step for no gain,
since a whole arm takes only a few minutes.

To run outside Slurm, the same thing by hand:

```bash
python train_mlp.py --output-dir RESULTS/A0_onehot          # no GPU, no cache
python train_mlp.py --arm L33 --pooling pca:20 --device cuda \
    --embeddings-dir /scratch/lmensi/peptide-HLA-stability/embeddings \
    --output-dir RESULTS/A3_L33_pca20
```

### Three things to get right

1. **A distinct `--output-dir` per arm.** Concurrent runs otherwise overwrite
   each other's `metrics.json`, figures and `mlp_split_*.pt`.
2. **The same `--pooling` and `--standardise` for every arm**, or the
   comparison is not budget-matched. `compare_arms.py` rejects runs whose
   `hidden`, `epsilon` or split count differ, but it cannot treat a pooling
   mismatch as an error — it only prints it in the arm label, so this one is on
   the operator.
3. **~5 GiB RAM per shard** under `pca:20` or `mean`: the cache is
   memory-mapped and concurrent reads are safe, but each process materialises
   the training rows as float32 (4.3 GiB). Under `flatten` it is ~15 GiB per
   shard, so three concurrent `flatten` runs need ~45 GiB.

### Cost, measured

Per split, on one GPU:

| Arm | Features | Epochs run | Seconds/split |
|---|---|---|---|
| `onehot` | 860 | 48–85 | ~22 |
| `L33` `pca:20` | 860 | ~250 | ~205 |
| `L33` `flatten` | 55,040 | 77–115 | 128–171 |

So a whole arm is **2–17 minutes**, not hours. Two things are counter-intuitive
here:

- **`pca:20` is slower per split than `flatten`**, despite 64× fewer features,
  because it runs ~3× the epochs before early stopping. Per *epoch* `flatten`
  is far more expensive; it just converges sooner.
- **`flatten` genuinely uses the GPU** — the 55,040 × 256 first layer is one
  large matmul, measured at 59% utilisation and 2.8 GiB. The narrow arms do
  not: at 860 features the head is negligible and the time goes to loading and
  pooling 3.1 GB per layer, which is CPU and I/O bound.

A GPU per task is therefore worth it for `flatten` and close to irrelevant for
the narrow arms, which would run about as fast sharing one device or on CPU.

---

## 10. Using an embedding cache extracted elsewhere

### Where the cache lives

On this cluster the cache is on **`/scratch`**, not in the repository and not
in a session temp directory:

```
/scratch/lmensi/peptide-HLA-stability/embeddings/
```

```bash
--embeddings-dir /scratch/lmensi/peptide-HLA-stability/embeddings
```

Why not elsewhere:

| Location | Why not |
|---|---|
| repo / `/home` | NFS at **97% full**, 267 GB free and shared. The cache is 8.8 GB now, but A4 needs a 193-position layer-32 cache (13.9 GB) and a `GGGGS` ablation another 8.8 GB |
| session temp dir | **`tmpfs`, i.e. RAM** — it would consume 8.8 GB of memory the training needs, vanish on reboot, and be invisible to a collaborator |
| `/scratch` | 3.4 TB free, persistent, and `/scratch/<user>` is world-readable, so one extraction serves the whole team |

Nothing hardcodes the path — `--embeddings-dir` exists for exactly this. Point
`TORCH_HOME` at `/scratch` too, or the 2.5 GB of ESM-2 weights are
re-downloaded every session.

### Reusing a cache built elsewhere

A collaborator who extracted on another machine needs the cache directory to
hold `concat_L{0,15,33}.npy` and `index.json` together, passed as
`--embeddings-dir`. Three failure modes are handled explicitly:

| Situation | Behaviour |
|---|---|
| Cache extracted **before** `source_row` followed `split_dataset.py` (pre-`1d68d28`) | Refused with the convention named and the repair command; see below |
| A layer array missing | Refused, naming the `extract_embeddings.py --layers` call that builds it |
| Cache covers only some rows (a `--limit` run) | Refused, listing the absent `source_row` values |

The arrays from a pre-fix extraction are **still valid** — row order never
changed, only the `source_row` labels — so the index is repaired in place
rather than re-extracting 8.66 GiB:

```bash
python scripts/features.py DATA/rasmussen_et_al_dataset.xlsx \
    --out-dir <embeddings-dir> --repair-index
```

Repair, not delete-and-rebuild: `features.py` only knows about its own arm and
would stamp `model="onehot"` over an ESM-2 cache. `--repair-index` rewrites
`source_row` and nothing else, preserving the recorded model and layer list, and
refuses unless the stored `pairs` still match the dataset in the same order —
which is what makes the arrays trustworthy afterwards.

The run line and `metrics.json` now record the cache's model and width:

```
[L33] split 0: 860 features from esm2_t33_650M_UR50D d=1280  best epoch ...
```

That is there because a cache built with the 8M development model would
otherwise be used silently at `d=320` instead of `d=1280`, which is invisible in
every other output.

Dependencies for a PLM arm are only numpy, pandas and torch — `fair-esm` is
needed to *build* a cache, not to train on one, since `extract_embeddings`
imports torch and esm lazily.
