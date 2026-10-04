# peptide-HLA-stability

**SEPPIA** — Structure Enabled Protein Protein Interaction Assistant.

Predicting the dissociation half-life (*t*½) of peptide-HLA class I complexes by reading
the internal representations of a structure prediction model, rather than encoding the
sequences directly.

*t*½ is among the strongest correlates of immunogenicity, so it matters for neoantigen
selection and vaccine design. This repository holds the experiment that led to SEPPIA:
a ladder of input representations, each feeding an identical regression head, which
isolates what each representation actually contributes.

## The result in one table

| Representation | Features | Spearman ρ | RMSE | Unseen-allele RMSE | Splits won |
|---|---:|---:|---:|---:|---:|
| One-hot contact positions (baseline) | 860 | 0.767 ± 0.018 | 1.172 | 1.643 | reference |
| ESM-2 layer 33, matched budget | 860 | 0.736 ± 0.030 | 1.230 | 1.749 | 0 / 5 |
| Boltz-2 single track, matched budget | 860 | 0.747 ± 0.023 | 1.203 | 1.584 | 0 / 5 |
| **Boltz-2 pair tensor, matched budget** | 860 | **0.821 ± 0.018** | **1.036** | **1.365** | **5 / 5** |
| **Boltz-2 pair tensor, full width** | 5,504 | **0.836 ± 0.015** | **0.992** | **1.292** | **5 / 5** |

Mean ± SD over five splits; RMSE in natural-log units of hours, lower is better. "Splits
won" counts paired per-split wins in Spearman ρ against the one-hot baseline. Full tables:
[`RESULTS/ladder_all.md`](RESULTS/ladder_all.md) and
[`docs/06_boltz2_ladder_results.md`](docs/06_boltz2_ladder_results.md).

Three findings:

1. **A one-hot encoding of the 43 groove-contact positions is a strong baseline**, ρ = 0.767.
2. **ESM-2 embeddings do not beat it.** At a matched feature budget its layers score 0.761,
   0.750 and 0.736, losing on all five splits; at 64× the input width they reach only parity.
   Per-residue sequence context adds little to amino-acid identity for this task.
3. **Boltz-2 trunk representations do beat it**, by ρ = +0.054 (matched) and +0.069 (full
   width), winning 5/5 splits, with RMSE on test-exclusive HLA alleles down 21%. The gain is
   carried specifically by the **pair tensor** *z*ᵢⱼ, which encodes residue-residue geometry:
   Boltz-2's per-residue single track scores 0.747, right where ESM-2 lands, and per-residue
   confidence (pLDDT) alone carries almost no signal. The structure model is used frozen and
   is never fine-tuned.

## Data and evaluation protocol

28,166 measured peptide-HLA class I complexes (Rasmussen et al.),
`DATA/rasmussen_et_al_dataset.xlsx`. Each row is a 9-mer peptide, an HLA allele with its
182-residue sequence and 34-residue pseudosequence, and a measured half-life in hours.

[`split_dataset.py`](split_dataset.py) builds five 70/30 splits that are, per split:

- **moment-matched** — train and test means of ln(*x*) and ln(*x*)² agree within 0.05 and 0.10,
- **holdout-constrained** — at least 5 HLA sequences and 100 peptides appear only in test,

so generalisation to unseen sequences is measured rather than assumed. The splits are
independent draws, not cross-validation folds, and their test sets may overlap.

The target is `ln(thalf_hours + ε)` with **ε = 0.1**, the reporting resolution of the assay;
20% of measured half-lives are exactly zero, and a smaller ε pushes those rows far below the
rest and lets them dominate the squared error. Errors are therefore in log units and are
comparable only between runs sharing the same ε.

Comparisons are **paired**: every arm sees the same five splits with the same hyperparameters,
so the across-split SD is mostly split difficulty and swamps the difference between arms. The
consistency of the sign across splits carries the argument, not the means.

## The ladder

One MLP block (860 → 256 → 128 → 1, dropout 0.2, Adam, early stopping) serves every arm;
`--arm` changes only the input representation, and `--pooling` decides how each row's
(43, D) block becomes a vector.

| Arm | Representation |
|---|---|
| `onehot` | 43 contact positions, one-hot over 20 amino acids (the baseline) |
| `L0`, `L15`, `L33` | ESM-2 layers 0 (context-free control), 15, 33 |
| `BZS` | Boltz-2 trunk single representation *s* |
| `BZZ` | Boltz-2 pair tensor, 1/*d*² distance weighted |
| `BZZU` | Boltz-2 pair tensor, uniform groove contraction — **the winning arm** |
| `BZSZ` | Combined single + pair |
| `BZP` | Per-token pLDDT confidence, a zero-training control |

`pca:20` is the budget-matched pooling: every arm gets exactly 43 × 20 = 860 features and a
byte-identical head, with the projection fitted on training rows only. `flatten` keeps all
features and is a capacity experiment, not a representation one.

## Layout

```
split_dataset.py          Five moment-matched, holdout-constrained splits
train_mlp.py              The shared MLP head; --arm selects the representation
analyze_errors.py         Error decomposition, stability calls, scatter plots
scripts/extract_embeddings.py   ESM-2 feature cache
scripts/extract_boltz.py        Boltz-2 trunk cache, (n_rows, 43, D)
scripts/arm_features.py         Pooling and standardisation for every arm
scripts/compare_arms.py, plot_ladder.py   Paired comparisons and ladder figures
slurm/                    Cluster submission for the full ladder and width sweep
docs/                     One technical report per task, 01-06
paper/                    Manuscript sources and the SEPPIA project description
RESULTS/                  Per-arm metrics.json, figures, reports
video/                    Remotion project for animated figures
```

## Reproducing

```bash
# 1. Splits (writes DATA/training_i.npz, DATA/testing_i.npz for i = 0..4)
python split_dataset.py DATA/rasmussen_et_al_dataset.xlsx --thresholds 0.05 0.1

# 2. The baseline arm: no cache, no GPU, a couple of minutes on CPU
python train_mlp.py --output-dir RESULTS/A0_onehot

# 3. A representation arm, against a feature cache
python train_mlp.py --arm BZZU --pooling pca:20 \
    --embeddings-dir <boltz cache> --output-dir RESULTS/BZZU_pca20

# 4. Paired comparison and ladder tables: both take the run directories
python scripts/compare_arms.py RESULTS/A0_onehot RESULTS/BZZU_pca20 --csv RESULTS/ladder.csv
python scripts/plot_ladder.py RESULTS/A0_onehot RESULTS/BZZU_pca20 --out-dir RESULTS

# The whole ladder on a cluster
sbatch slurm/ladder.sbatch
```

Error analysis for a trained arm:

```bash
python analyze_errors.py --splits-dir DATA --results-dir RESULTS
```

## Caveats worth knowing

- **Baselines are internal.** The one-hot MLP is a matched-budget baseline, not a published
  predictor; no external tool (NetMHCstabpan, NetMHCpan) has yet been run on these splits.
- **Capacity is under-regularised.** The baseline arm's test error is about 2.9× its training
  error even with dropout, weight decay and early stopping.
- **Only 74 distinct HLA pseudosequences** exist in the data, so allele-level generalisation is
  estimated from 5 held-out alleles per split and the unseen-allele figures carry wide
  uncertainty.
- **42% of half-lives were stored as dates** in the source workbook and are recovered as
  `day.month`; the recovery is consistent with one-decimal values but is lossy in principle.
- **Zeros are censored, not measured.** ε = 0.1 asserts a value for measurements that have
  none; a censored likelihood would treat the detection limit explicitly.

## Environment

Python 3.14, PyTorch 2.14, NumPy 2.5, pandas 3.0, Matplotlib 3.11. The ESM-2 and Boltz-2
caches are produced on GPU (documented in `docs/01_embedding_extraction.md` and
`docs/04_boltz2_arm.md`); everything downstream of a cache runs on CPU. The embedding caches
are large (9.3 GB for the full ESM-2 run) and are gitignored — regenerate them or pass
`--embeddings-dir`.
