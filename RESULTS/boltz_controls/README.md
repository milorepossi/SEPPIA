# Controls for the Boltz-2 ladder

## Standardisation asymmetry (control 1)

The ladder's `auto` policy resolves to "standardise everything except one-hot"
(`scripts/arm_features.py:should_standardise`), so the A0 baseline reaches the
head as raw 0/1 while every Boltz arm passes through a fitted per-dimension
prescaling and a fitted feature standardiser with clipping. That made the
headline +0.0543 partly a pipeline comparison rather than a pure
representation swap.

Measured, 5 splits, one seed per split, `--arm onehot`:

| run | standardised | Spearman | paired Δ |
|---|---|---|---|
| `onehot_standardise_never.json` | False | 0.7687 ± 0.0217 | reference |
| `onehot_standardise_always.json` | True | 0.7742 ± 0.0161 | +0.0055 (3/5) |

**Standardising the baseline is worth +0.0055**, roughly a tenth of the
+0.0543 that `BZZU (pca:20)` gains over it. The asymmetry does not explain the
result.

Two caveats. These runs used the committed `DATA/*.npz` targets *without*
`--clean-target-csv`, because `DATA/rasmussen_clean.csv` is gitignored and has
no committed generator — so this is a within-baseline paired comparison, not a
drop-in replacement for the committed reference row. The `never` arm lands at
0.7687 against the committed 0.7667, so ~0.002 is attributable to the target
correction plus environment drift (different torch build).

Reproduce:

```bash
python train_mlp.py --arm onehot --standardise never  --device cpu \
    --output-dir RESULTS/boltz_controls/onehot_never  --no-save-models
python train_mlp.py --arm onehot --standardise always --device cpu \
    --output-dir RESULTS/boltz_controls/onehot_always --no-save-models
```

## Still outstanding

- **Replicate seeds.** Every committed arm is one seed per split, so no
  seed-variance estimate exists and the quoted ± is split spread. Needs ≥3
  seeds for `onehot` and `BZZU`.
- **A null representation.** A random projection at 860 features, or a
  randomly initialised Boltz-2, to show the gain is the trained representation
  rather than dimensionality.
- **`BZP` sequence control.** Nothing supports a claim that pLDDT tracks
  stability until a sequence-only model matched on peptide composition is
  scored on the same rows.
- **Default-fidelity re-extraction.** `BZZU` beats `BZZ` under MSA-free,
  10-step extraction. Discriminating "distance weighting injects coordinate
  noise" from "distance weighting is wrong" needs a sample re-extracted at
  default sampling steps with real alignments. Requires GPU.
