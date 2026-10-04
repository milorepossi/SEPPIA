# Boltz-2 arm (fork)

Fork of [milorepossi/peptide-HLA-stability](https://github.com/milorepossi/peptide-HLA-stability),
branch `boltz2-arm`. Adds a Boltz-2 trunk-embedding arm to the existing A0-A3
ladder, with runners for both **Modal** (current, $150) and **Hugging Face
Jobs** (earlier, $20).

Start here:

| Document | What it covers |
|---|---|
| **[`MODAL_BENCHMARK.md`](MODAL_BENCHMARK.md)** | best GPU, nine-card cost sweep, fine-tuning feasibility — **read first** |
| [`BENCHMARK.md`](BENCHMARK.md) | the earlier Hugging Face measurement on a $20 budget |
| [`docs/04_boltz2_arm.md`](docs/04_boltz2_arm.md) | the method, assertions and failure modes |

## The claim being tested

`RESULTS.md` found that frozen ESM-2 representations do not beat a one-hot
encoding of the same 43 positions (A0 0.771 Spearman, best PLM arm 0.767). The
follow-up is not a bigger language model but a representation that sees the
*complex*.

Boltz-2's single representation is 384-dimensional against ESM-2 650M's 1280, so
it is not the richer per-residue featurizer. The only thing it adds that no
sequence model can express is its **pair** representation over residue pairs at
the interface. Arm `BZZ` carries that hypothesis; `BZS` is its control.

## The embedding-regressor, end to end

Four stages. Boltz-2 is **frozen** throughout; nothing about it is trained.

```
  peptide + HLA  ->  Boltz-2 trunk  ->  43-slot cache  ->  pooling  ->  MLP  ->  ln(t½)
   9aa    182aa      (frozen)           (n,43,D) fp16     (n,F)      256-128-1
```

### Stage 1 — which embeddings

One `boltz predict --write_embeddings` pass per complex returns the trunk's
single representation `s` `[191, 384]` and pair representation `z`
`[191, 191, 128]`, where 191 = 182 HLA residues + 9 peptide residues. We reduce
each to **43 slots**: the 9 peptide positions plus the 34 NetMHCpan
contact positions of the groove. Those are the same 43 positions the ESM-2 arms
use, which is what keeps the ladder comparable.

| Arm | Cache | Shape | What it is |
|---|---|---|---|
| `BZS` | `boltz_S.npy` | `(n, 43, 384)` | `s` gathered at the 43 slots |
| `BZZ` | `boltz_Z.npy` | `(n, 43, 128)` | `z` contracted over the interface, 1/d² weighted |
| `BZZU` | `boltz_ZU.npy` | `(n, 43, 128)` | same contraction, uniform weights |
| `BZP` | `boltz_PLDDT.npy` | `(n, 43, 1)` | per-token pLDDT — **zero-training arm** |
| — | `boltz_ZRAW.npy` | `(n, 306, 128)` | raw 9×34 interface block, for offline derivation |

`BZP` costs nothing: the confidence module writes per-token pLDDT on every run
regardless, so this is 43 numbers already being computed and discarded. It asks
whether the model's own confidence in *where* the peptide sits predicts *how
long* it stays there — a single scalar per slot, no training, no parameters.

`boltz_ZRAW` is not an arm. It caches the raw interface block so any further
z-derived feature can be computed offline on CPU rather than paying for a second
extraction. 3.5 GB for the full dataset, all five arrays together.

`BZZ` is the arm that carries the hypothesis. Boltz-2's single representation is
384-dimensional against ESM-2 650M's 1280, so it is **not** the richer
per-residue featurizer — `BZS` is really a like-for-like rerun of A3. The pair
tensor is the only thing here that a sequence model cannot express, because it
describes residue *pairs* at an interface and is conditioned on the actual
complex.

The contraction keeps per-position structure rather than pooling it away. For
peptide slot `p` and contact position `q`, with `d` the Cα-Cα distance from the
predicted structure:

```
w[p,q]    = 1 / max(d(p,q), 3.0)²
zc[p]     = Σ_q w[p,q]·z[p,q] / Σ_q w[p,q]     -> slots 0..8
zc[9+q]   = Σ_p w[p,q]·z[p,q] / Σ_p w[p,q]     -> slots 9..42
```

This is PreFold-dG's inverse-square interchain weighting (PMID 42635209), but
*not* its outer-product pooling. That pooling exists to give variable-length
complexes a fixed-size vector; ours are always 182+9, and collapsing the peptide
axis would destroy the P2/P9 anchor signal that dominates class I binding.
`BZZU` makes "does the distance weighting earn its keep" a one-line ablation
rather than a second extraction.

### Which parts of Boltz-2 actually run

Not trunk-only. The extractor runs `boltz predict` with `--sampling_steps 10
--recycling_steps 3 --diffusion_samples 1`, so the **trunk, the diffusion module
and the confidence module all execute**. Three of the five cached arrays are
trunk-only; two depend on the diffusion path:

| Array | Trunk `s`/`z` | Diffusion coords | Confidence |
|---|:--:|:--:|:--:|
| `boltz_S.npy` | yes | — | — |
| `boltz_ZU.npy` | yes | — | — |
| `boltz_ZRAW.npy` | yes | — | — |
| `boltz_Z.npy` | yes | **yes** (1/d² weights from Cα) | — |
| `boltz_PLDDT.npy` | — | yes (indirectly) | **yes** |

Diffusion is run on purpose, for two reasons. The 1/d² interface weighting uses
**real Cα-Cα distances from the predicted structure**, which is sharper than
the distogram's binned expectation. And pLDDT comes from the confidence module,
which scores the sampled coordinates — so arm `BZP` does not exist without it.

The cost is small. From the measured conditions, the marginal cost of a sampling
step is 0.0388 s, so 10 steps is **0.39 s, about 5%** of the 7.73 s/complex
single-process rate. By contrast 200 steps would be 51% and recycling 0→3 is
28%. Going genuinely trunk-only (`skip_run_structure`) would save that 5% plus
some unmeasured fixed overhead — `diffusion_conditioning` and the confidence
module run once per complex regardless of step count — and would cost `BZP`
entirely and force distogram-based weighting. Not a trade worth taking. A
`sampling_steps=1` condition would pin the fixed part if it ever matters.

### Stage 2 — joining a split to the cache

The cache is **split-agnostic**: row `i` of every array is `source_row` `i`, and
`index.json` records the mapping. A split selects its rows by joining on
`source_row`, so one extraction serves all five splits and every arm. Nothing in
the feature path is aware of splits.

### Stage 3 — pooling 43 × D down to a fixed width

| Pooling | Features | `BZS` | `BZZ` | one-hot |
|---|---|---:|---:|---:|
| `flatten` | 43·D | 16,512 | 5,504 | 860 |
| `mean` | 2·D, peptide and HLA blocks separately | 768 | 256 | 40 |
| `pca:20` | 43·20, one D→20 projection fitted on train rows | 860 | 860 | 860 |

`pca:20` equalises the feature count so the head is byte-identical across arms.
**Treat it with suspicion.** In the ESM-2 ladder, Spearman tracks the variance
each arm retained at `pca:20` with r = 0.94 (L0 100% → 0.761, L15 79.4% →
0.750, L33 72.5% → 0.736), and running A3 uncompressed recovers +0.031, which
shrinks its deficit against one-hot from 0.035 to 0.004. The ladder's ordering
may be measuring the compression rather than the representation.

PCA also ranks directions by variance, not relevance, and `pca:20` would keep
15.6% of `BZZ`'s directions against 1.6% of ESM-2's — so it would quietly
*favour* the Boltz arms. Report `flatten`, fixed-*variance* PCA, and a random
projection, print the variance retained beside every score, and treat
disagreement between them as the result. Full analysis in
[`docs/05_finetune.md`](docs/05_finetune.md) §5.

### Stage 4 — the regressor

A plain MLP, deliberately. Anything clever here would confound the comparison.

| | |
|---|---|
| Architecture | `F → 256 → 128 → 1`, ReLU, dropout 0.2 |
| First-layer params at `pca:20` | 220,416 (identical for every arm) |
| Loss | MSE on the standardised log target |
| Target | `ln(t½ + 0.1)`; the 0.1 keeps the 20% left-censored zeros only 0.22 sd from the next observed value |
| Optimiser | Adam, lr 1e-3, weight decay 1e-5, batch 256 |
| Early stopping | on a 10% validation slice **carved out of train**, never test |
| Protocol | 5 splits × ≥3 seeds, paired across arms |
| Metric | Spearman ρ headline, plus Pearson and RMSE in log units |

Everything fitted is fitted on **training rows only**: the PCA projection, the
feature standardiser, and the target centering. The validation slice used for
early stopping is held out of all three.

```bash
python train_mlp.py --arm BZZ --embeddings-dir <boltz cache> --pooling pca:20
```

### Two things to get right before reading any number

**Re-read the target from the clean CSV.** `train_mlp.py` takes `thalf_hours`
from the split `.npz` files, which inherit the Excel recovery bug: 26 rows
across 19 distinct values are wrong (a true `1.05` reads as `1.5`). See
[`docs/04_boltz2_arm.md`](docs/04_boltz2_arm.md) §9.

**Re-score A0 on the same rows.** If the Boltz arms run on a subset, a loss
against A0's full-data 0.771 says nothing until A0 is re-scored on identical
rows. That re-scoring is CPU-only and free.

## Measured cost

### Modal, nine cards, ranked by $/complex (not speed)

| GPU | s/complex | conc | $/complex | Full 28,166 |
|---|---:|---:|---:|---:|
| **L4** | 6.14 | 4 | **$0.001364** | **$38.41** |
| A10G | 4.75 | 4 | $0.001455 | $40.97 |
| L40S | 3.01 | 4 | $0.001630 | $45.92 |
| H100 | 2.03 | 8 | $0.002229 | $62.78 |
| B200 | 1.80 | 8 | $0.003128 | $88.11 |
| T4 | 26.37 | 1 | $0.004325 | $121.81 |

**L4, concurrency 4-6, sharded.** The big cards are faster but cost more per
complex, and Modal fans out, so wall clock is bought with shards. T4 is the
trap: cheapest per hour, dearest per complex.

With $150 the **full dataset is affordable** at ~$38, so the pilot is no longer
a budget necessity — only a way to get an answer sooner.

### Fine-tuning is feasible

The 48-block Pairformer OOMs on a 22 GiB L4 without activation checkpointing
and needs 2.37 GiB with it. Counting Boltz's 3 recycles:

| Scope | GPU | 10 epochs, full set |
|---|---|---:|
| last 4 blocks, 12.3M params | L4 | **~$19** |
| all 48 blocks, 147M params | L4 | ~$274 (pilot only, ~$27) |

The binding constraint is statistical: 147M parameters against 28,166 labels
over 75 alleles will overfit whatever it costs.

### Earlier Hugging Face measurement

L4 (`l4x1`, $0.80/h), single process, `--no_kernels`:

| Condition | s/complex |
|---|---:|
| 10 sampling steps, 0 recycling | 5.54 |
| 10 sampling steps, 3 recycling | 7.73 |
| 200 sampling steps, 3 recycling | 15.10 |
| **end to end, incl. reduction** | **9.41** |

| Scope | Cost at 9.41 s | Fits $20? |
|---|---:|:--|
| Pilot, 2,814 complexes | **$5.89** | yes |
| Full, 28,166 complexes | $58.93 | no, 2.9x over |

**Recommendation: run the pilot.** It answers the question, costs under a
quarter of the budget, and covers every HLA sequence the five splits hold out as
test-exclusive, so the unseen-allele stratum is scorable.

## Run order

```bash
python scripts/test_extract_boltz.py   # GPU-free, 13 checks, run this first
python scripts/pilot_subset.py         # target audit + pilot selection
# then the two HF jobs; see docs/04_boltz2_arm.md §10
```

## Two findings that change the existing pipeline

**`docs/02` §0.1 is wrong about the target.** The Excel-mangled half-life column
is *not* losslessly invertible. Values of the form `x.0Y` collapse: a true
`1.05` recovers as `1.5`. 26 rows, 19 distinct values, and the bug is baked into
the committed `DATA/training_*.npz` targets. `DATA/rasmussen_clean.csv` is
ground truth. Split membership is unaffected.

**Boltz-2's affinity head cannot be used.** Its PairFormer attends only over
protein-ligand and intra-ligand interactions, the CLI requires a SMILES/CCD
binder of ≤128 atoms, and affinity training dropped ligands above 50 heavy
atoms, while these 9-mers are 48-100 (median 75). Hence a custom head on the
trunk, following PreFold-dG (PMID 42635209).

## Interpreting a B-arm number

B arms see 2,814 rows; A0-A3 saw all 28,166. **A B-arm loss is not evidence
about Boltz-2 until A0 is re-scored on the pilot rows only.** That re-scoring is
CPU-only and free, and must be reported alongside.
