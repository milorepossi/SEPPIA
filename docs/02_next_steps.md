# Next implementation steps

Plan after Task 1 (`docs/01_embedding_extraction.md`). The experiment is a ladder
of MLP heads with identical architecture and hyperparameter budget, differing
**only** in input features, each evaluated on all 5 splits.

| Arm | Input features | Needs |
|---|---|---|
| A0 | BLOSUM62 encoding (no PLM) | — |
| A1 | ESM-2 layer 0 (context-free control) | `concat_L0.npy` |
| A2 | ESM-2 layer 15 (mid-stack) | `concat_L15.npy` |
| A3 | ESM-2 layer 33 (final) | `concat_L33.npy` |
| A4 | fine-tune last ESM-2 block + head | separate cache, see §6 |

---

## 0. Decide these before writing training code

Four choices that are confounds, not details. Each must be fixed once and applied
identically across arms.

### 0.1 Target variable — `thalf_hours` is Excel-corrupted (recovery verified)

42% of the cells were mangled by Excel into datetimes. This is a property of the
committed `.xlsx`, not of any code here; `extract_embeddings.py` never reads the
column, so the embedding cache is unaffected. It goes live the moment training
code reads the target.

| Property | Value |
|---|---|
| Raw cell types | **16,259 `str` + 11,907 `datetime`** (42.3% mangled) |
| Example | row 0 reads `2026-08-01 00:00:00`, i.e. `"1.8"` parsed as day.month |
| Recovered range | 0.0 – 256.7 h, median **1.10**, mean 5.45, std 11.71 |
| Zeros | **5,679 rows (20.2%)** |
| Tail | p25 = 0.20, p75 = 5.60, p90 = 15.40, p99 = 52.63, max 256.7 |
| Distinct values | 925 |

**Do not write another parser.** `split_dataset.py:half_life()` already inverts it
(`float(f'{value.day}.{value.month}')`). Import that function so splits and
training agree by construction.

`scripts/audit_target.py` proves the inversion is lossless rather than assuming
it — run it in CI or before any training run:

| Check | Result |
|---|---|
| All datetime cells share one Excel parse | year `[2026]`, all midnight ✓ |
| Orientation is `day.month`, not `month.day` | day range 1..31, month range 1..12 ✓ |
| Recovered values ∩ surviving string values | **0 overlap** ✓ |
| Recovered range inside the clean-cell range | 1.10..31.80 ⊂ 0.000..256.7 ✓ |

The zero overlap is the load-bearing result. The corruption is a deterministic
function of the *value*: anything readable as day.month (`1.8`) was eaten in every
row it occurred in, anything unreadable (`1.15` — no month 15; `1.0` — no month 0)
survived in every row. Disjoint populations ⇒ the inverse map is well defined. An
overlap would have meant two distinct source values collapse onto one number and
the column could not be trusted at all.

Residual risk: **none for value recovery**, but the corruption is silent and
locale-dependent. If the `.xlsx` is ever re-saved or re-exported, re-run the
audit — a different Excel locale would flip the orientation to `month.day` and
`half_life()` would then be wrong while still returning plausible numbers.

Implications:
- Heavy right skew ⇒ regress on `log` with a floor, matching the splits, which are
  moment-matched on `E[ln x]` and `E[ln x]^2` with zeros mapped to `0.001`.
  Reusing that exact convention keeps the split constraints meaningful.
- 20% exact zeros are **left-censored**, not real zero half-lives. `log(0.001)`
  puts them 7 natural-log units below the median — a large artificial spike.
  Options: (a) keep the `0.001` floor for comparability with the splits,
  (b) censored/two-part model, (c) report metrics with and without the zero
  stratum. Pick one and state it.
- Report **Spearman ρ** as the headline metric — it is invariant to the transform
  and immune to the censoring choice — plus Pearson *r* and RMSE on the log
  scale.

### 0.2 Pooling 43 × 1280 → a fixed vector

Flattening is 55,040 features against ~19,700 training rows. A 256-unit first
layer would be 14M parameters — the head would overfit and the comparison would
measure regularisation, not representation.

Recommended, cheap, and identical across arms: **mean-pool the two blocks
separately**, keeping the peptide/HLA distinction.

```
peptide  slots 0..8   -> mean -> 1280
pseudoseq slots 9..42 -> mean -> 1280
concat                       -> 2560 features
```

Alternatives if time allows (apply to every PLM arm or none): per-slot learned
attention pooling; or flatten only the 9 peptide slots (9×1280 = 11,520) and
mean-pool the HLA side.

Input dimension necessarily differs between arms (A0 ≈ 860, PLM arms 2560). That
is the intended difference. Keep **hidden widths, depth, dropout, optimiser, LR
schedule, epochs, early-stopping rule and seed count identical.**

### 0.3 Feature standardisation — required, because of the layer-norm asymmetry

Layer 33 is post-`emb_layer_norm_after`, layer 15 is raw (mean abs 0.145 vs
2.344, a 16× scale gap — see `docs/01_embedding_extraction.md` §8.1). Fit a
`StandardScaler` **per arm on the training split only** and reuse it for test.
Without this, A2 vs A3 partly measures input scaling.

### 0.4 A0's HLA encoding

`HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` (756 rows) share a pseudosequence and
differ only at `hla_seq` index 10, a non-contact position. A pseudosequence-only
BLOSUM baseline makes them identical; the PLM arms see the difference.

To keep A0 a fair baseline, encode the **same 43 positions** the PLM arms use
(9 peptide + 34 pseudoseq) × 20 BLOSUM62 columns = **860 features**, and note the
collapse as a known limitation.

---

## 1. Modal app for the full extraction — `modal_app.py`

`extract_embeddings()` is already a plain function taking all paths as arguments.

```python
image = modal.Image.debian_slim().pip_install("torch", "fair-esm", "pandas",
                                              "openpyxl", "numpy")
vol = modal.Volume.from_name("pmhc-embeddings", create_if_missing=True)

@app.function(image=image, gpu="A10G", volumes={"/cache": vol},
              timeout=60*60, retries=2)
def run(limit=None):
    from scripts.extract_embeddings import extract_embeddings
    return extract_embeddings(dataset_path="/cache/rasmussen_et_al_dataset.xlsx",
                              out_dir="/cache/embeddings", limit=limit)
```

Checklist:
- Bake the 2.6 GB ESM-2 weights into the image (or a volume) — do not re-download
  per invocation. Set `TORCH_HOME` to a persisted path.
- `vol.commit()` after the run; the memmap checkpoint already survives a timeout,
  and `retries` plus the resume logic make a mid-run kill recoverable.
- Verify **9.3 GB** of volume headroom.
- Smoke-test with `limit=500` on Modal and diff against the local 500-row cache —
  it should be bitwise identical for the same model and dtype.
- Budget ~20 min on a P100-class GPU; A10G/A100 should be well under that.

## 2. Generate the 5 splits

Not yet produced (`DATA/split/` does not exist). `split_dataset.py` writes
`training_{i}.xlsx`, `testing_{i}.xlsx`, `metadata_{i}.json`,
`log_distributions_{i}.png` for `i = 0..4`.

```bash
~/miniconda3/envs/myconda/bin/python split_dataset.py \
    rasmussen_et_al_dataset.xlsx --thresholds 0.05 0.1 --output-dir DATA/split
```

Needs `matplotlib` + `openpyxl` (both present in `myconda`). Exit code 2 means the
moment thresholds were not met — loosen them rather than reinterpreting the
output. Commit `metadata_*.json`; the `.xlsx` splits are derived and large, so
consider gitignoring them and regenerating from the recorded seed.

## 3. Feature assembly — `scripts/features.py`

The only place that joins splits to the cache.

```
index.json["source_row"]  -->  position in concat_L*.npy
training_{i}.xlsx.source_row  -->  np.searchsorted / dict lookup  -->  row indices
```

- `np.load(..., mmap_mode="r")`, fancy-index the split's rows, then pool. Pooled
  train+test for one layer is ~0.1 GB, so cache pooled arrays to `.npy` and
  training becomes CPU-cheap and instant to iterate on.
- **Assert** `index.json["source_row"]` covers every `source_row` in both split
  files and that the recovered `(hla_seq, peptide)` pair matches the xlsx row —
  this is the one place a silent misalignment could still enter.
- Assert the union of train and test `source_row` is exactly `0..28165`, once per
  split.

## 4. Shared head and training loop — `scripts/train_head.py`

One module, one architecture, `--features {blosum,L0,L15,L33}` selecting the input
only.

- MLP: `in -> 512 -> 256 -> 1`, GELU, dropout 0.1, LayerNorm; AdamW, cosine
  schedule, early stopping on a validation slice **carved out of train** (never
  test).
- Fixed budget: identical epochs/LR/batch for every arm. If any hyperparameter is
  tuned, tune it per arm with the same search budget and report the budget.
- Seeds: ≥3 per (arm, split) ⇒ 4 arms × 5 splits × 3 seeds = 60 cheap runs on
  pooled features.
- Log per run: split, arm, seed, Spearman, Pearson, RMSE, n_train, n_test, plus
  metrics on the test-exclusive HLA and peptide strata (the splits guarantee ≥5
  unseen HLAs and ≥100 unseen peptides — the generalisation question worth
  answering).
- Write one tidy `results.csv`; keep plotting separate.

## 5. Analysis

- Primary: mean ± sd Spearman per arm across the 5 splits.
- The informative contrasts are **A1 vs A3** (does context help beyond amino-acid
  identity?) and **A0 vs A1** (does a context-free PLM embedding beat BLOSUM62 at
  all?). A1 is the control that makes the claim falsifiable — if A3 ≈ A1, depth
  buys nothing here.
- Paired comparison across splits (same splits for every arm) ⇒ use a paired test
  or report per-split deltas, not independent-sample statistics.
- Break out unseen-HLA and unseen-peptide strata separately.

## 6. A4 — fine-tuning the last block

Naive full fine-tuning is the expensive path: forward+backward over 33 layers at
~11 seq/s ⇒ ~43 min per epoch per split on the P100, × 5 splits × epochs.

**Only block 33 and the head have gradients, so cache layer 32 once and train on
that.** Note this needs *all 193 token positions*, not the 43 slots, because
attention in block 33 mixes positions:

```
28166 x 193 x 1280 fp16 = 13.9 GB   (one extra extraction, repr_layers=[32])
```

Then each epoch runs one transformer block plus the head — seconds, not
40 minutes, and it makes the 5-split × multi-seed protocol affordable.

Caveats: re-apply `emb_layer_norm_after` after the fine-tuned block to match A3's
feature definition; the pre-trained block-33 weights are the init; use a much
lower LR for the block than the head. A4 is **not** budget-matched to A0–A3 by
construction — report it as a separate claim.

## 7. Suggested order

1. §0 decisions written down (target transform, pooling, scaling, A0 encoding).
2. §2 splits — unblocks everything and is CPU-only.
3. §3 feature assembly + the join assertions, validated against the existing
   local 500-row cache.
4. §4 head trained on the 500-row cache end-to-end to shake out plumbing.
5. §1 Modal full extraction (9.3 GB).
6. §4/§5 all arms × 5 splits × seeds, then analysis.
7. §6 A4 last.
