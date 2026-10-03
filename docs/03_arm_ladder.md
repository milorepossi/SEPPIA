# The arm ladder — one MLP, swappable input representation

Technical report for `train_mlp.py --arm`, `scripts/arm_features.py` and
`scripts/compare_arms.py`.

Status: **implemented and validated.** A0 runs end to end. A1–A3 are
implemented and their code path is verified, but the comparison itself is
pending the full embedding cache (§8).

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
| `flatten` | 860 | 55,040 | **64×** the parameters | ~15.6 GiB |
| `mean` | 40 | 2,560 | 3×, discards position within each block | ~0.7 GiB |
| `pca:20` | 860 | **860** | **identical** | ~0.2 GiB |

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

Once it lands:

```bash
python train_mlp.py --output-dir RESULTS/A0_onehot
for arm in L0 L15 L33; do
  python train_mlp.py --arm $arm --pooling pca:20 --output-dir RESULTS/A_$arm
done
python scripts/compare_arms.py RESULTS/A0_onehot RESULTS/A_L0 \
    RESULTS/A_L15 RESULTS/A_L33 --csv RESULTS/ladder.csv
```

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
