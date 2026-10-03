# Boltz-2 arm — measured cost, and what $20 buys

Everything here is measured on Hugging Face Jobs, not extrapolated from the
paper. Neither the Boltz-2 preprint nor its repository publishes a wall-clock
figure, so this had to be run.

Reproduce with `hf/bench_boltz.py`; the raw report is at
`hf://buckets/<user>/boltz-cache/bench_report.json`.

---

## 1. Hardware

```
flavor l4x1   NVIDIA L4, 22.03 GiB, compute capability 8.9, bf16 supported
torch 2.14.1+cu130, python 3.12.12, $0.80/hour
```

L4 is the right class of card. The complex is 182 + 9 = **191 tokens**, so the
pair representation is about 19 MB and GPU memory is nowhere near binding. An
A100 at $2.50/hour would buy nothing here.

**Do not use `t4-small`** despite it being $0.40/hour. T4 is compute capability
7.5, and Boltz-2's triangle-attention path needs sm_80 or newer.

## 2. Per-complex cost, 12 complexes over 4 alleles

| Condition | s/complex | vs baseline |
|---|---:|---|
| `sampling_steps=10, recycling=0` | **5.54** | floor |
| `sampling_steps=10, recycling=3` | **7.73** | +2.19 for default recycling |
| `sampling_steps=200, recycling=3` | 15.10 | +7.37 for full diffusion |
| weight download, one-off per container | 115.7 s | not per-complex |

Two findings that matter more than the numbers.

**Full diffusion is wasted money.** Dropping 200 sampling steps to 10 halves
the per-complex cost. We only need coordinates for the inverse-square interface
weighting, and the groove is well determined; the trunk representations `s` and
`z` are produced before the diffusion module runs at all.

**The weight download is now free.** Boltz-2's checkpoints and ligand
dictionary are 6.2 GB (`boltz2_conf.ckpt` 2.29 GB, `boltz2_aff.ckpt` 2.06 GB,
`mols.tar` 1.86 GB). Mounting a bucket at `--cache` persists them, so only the
first job ever pays the 116 s. This is why the extraction runs as **few large
jobs, not many small ones**.

### A blocker worth recording

Boltz-2 2.2.1 reaches for `cuequivariance_torch` for triangle multiplication
and hard-fails with `ModuleNotFoundError` if it is absent — it does not fall
back on its own. Every number above is therefore the **`--no_kernels`**
pure-PyTorch path. The kernel path would be faster and is unmeasured; adding
`cuequivariance-torch` is the obvious next optimisation, but its wheels are
built per CUDA version and the job image ships CUDA 13, so it is not a
one-liner. Budgeting on `--no_kernels` is the conservative choice.

## 3. What `--write_embeddings` actually returns

| Tensor | Shape | dtype | Size |
|---|---|---|---|
| `s` | `[1, 191, 384]` | float32 | 0.28 MB |
| `z` | `[1, 191, 191, 128]` | float32 | 17.81 MB |

Compressed npz on disk: **16.62 MB per complex**. That is the reason the
extractor reduces and deletes per batch rather than keeping the raw output —
retaining it for the full dataset would be 468 GB, against 1.44 GB for the
reduced cache.

## 4. Reduced cache size

43 slots, float16, three arrays (`boltz_S` 384-dim, `boltz_Z` and `boltz_ZU`
128-dim each):

| Scope | Cache |
|---|---:|
| pilot, 2,814 complexes | 0.14 GB |
| full, 28,166 complexes | 1.44 GB |

Storage is a non-issue. GPU seconds are the only real constraint.

## 5. The budget

At $0.80/hour, **$20 is 25 GPU-hours = 90,000 GPU-seconds.**

| Scope | s/complex | GPU-hours | Cost | Fits in $20? |
|---|---:|---:|---:|:--|
| Pilot, 2,814 | 7.73 | 6.0 | **$4.83** | yes, with room to spare |
| Full, 28,166 | 7.73 | 60.5 | $48.38 | **no, 2.4x over** |
| Full, 28,166 | 5.54 | 43.3 | $34.64 | **no, 1.7x over** |
| Budget ceiling | 7.73 | 25.0 | $20.00 | 11,640 complexes |

### Recommendation

**Run the 2,814-complex pilot, not the full dataset.** It costs $4.83, leaves
roughly $15 for reruns and a second pass, and answers the actual question.
Spending the entire budget on 11,640 rows would buy 4x the data for an arm that
has not yet been shown to beat a one-hot encoding, and would leave nothing for
the mistakes.

The pilot is chosen for breadth rather than depth: 28 alleles over both loci,
~110 peptides each, stratified across each allele's own half-life range, and it
covers **every** HLA sequence the five splits hold out as test-exclusive. That
last property is not optional — without it the unseen-allele stratum, which is
the generalisation question the splits were built for, cannot be scored at all.

| Pilot property | Value |
|---|---|
| Complexes | 2,814 |
| Alleles / unique peptides | 28 / 2,122 |
| Locus balance (A / B rows) | 1,357 / 1,457 |
| Zero (left-censored) fraction | 17.95% vs 20.16% full |
| Covers all test-exclusive HLA | yes |

Spent so far on benchmarking: **about $0.17** (two jobs, 763 GPU-seconds).

## 6. What the pilot has to beat

From `RESULTS.md`, scored on these same five splits with the same head:

| Arm | Spearman | Splits won vs A0 |
|---|---:|---:|
| **A0 one-hot** | **0.771 ± 0.020** | — |
| A1 ESM-2 layer 0 | 0.761 ± 0.017 | 0/5 |
| A2 ESM-2 layer 15 | 0.750 ± 0.020 | 0/5 |
| A3 ESM-2 layer 33 | 0.736 ± 0.030 | 0/5 |
| A3 layer 33, flatten | 0.767 ± 0.021 | 2/5 |

No frozen ESM-2 arm beats a plain one-hot encoding. The bar is 0.771.

One caveat on reading the comparison: the B arms are trained on 2,814 rows
while A0-A3 had all 28,166. A B-arm loss is therefore **not** evidence against
Boltz-2 until A0 is re-scored on the same pilot rows. That re-scoring is CPU-
only and free, and must be reported alongside.

## 7. Which embeddings to keep, and why `BZZ` carries the hypothesis

Boltz-2's single representation is 384-dimensional. ESM-2 650M's is 1280.
Boltz-2 is **not** the richer per-residue featurizer, so an arm built only on
`s` is the weak form of the experiment and would most likely reproduce the
ESM-2 result. What Boltz adds that no sequence model can express is the pair
tensor: a representation of residue *pairs* at the interface, conditioned on
the actual complex.

| Arm | Content | Slots x dim | Role |
|---|---|---|---|
| `BZS` | trunk single representation | 43 x 384 | comparable to A3; expected to tie |
| `BZZ` | interface pair contraction, 1/d² | 43 x 128 | **the actual hypothesis** |
| `BZZU` | same contraction, uniform weights | 43 x 128 | does distance weighting matter? |

All three come from one extraction pass, so the ablation is free once the pilot
has run. The ranking decides what a full run would cache, if a full run is ever
bought.

`BZS` + `BZZ` concatenated is the strongest configuration and should be scored
too; it is the closest analogue to what PreFold-dG actually does.
