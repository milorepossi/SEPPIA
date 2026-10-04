# Task 5 — fine-tuning: timing, gating, and the zero half-life question

Companion to [`MODAL_BENCHMARK.md`](../MODAL_BENCHMARK.md). Covers three things:
how long the extraction takes and how to make it fit overnight, whether to drop
the zero half-life rows, and the guards that stop a fine-tune from competing
with the extraction.

---

## 1. Overnight timing: buy shards, not a bigger card

Wall clock and cost are separable here. Sharding splits the work across
containers, each on its own GPU, so **S shards finish S times sooner for the
same total GPU-seconds**. The GPU choice sets the price; the shard count sets
the clock.

Hours of wall clock for all 28,166 complexes:

| GPU | GPU-hours | Cost | S=1 | S=4 | **S=8** | S=16 |
|---|---:|---:|---:|---:|---:|---:|
| **L4** | 48.0 | **$38.39** | 48.0 | 12.0 | **6.0** | 3.0 |
| A10G | 37.2 | $40.94 | 37.2 | 9.3 | 4.6 | 2.3 |
| L40S | 23.5 | $45.95 | 23.5 | 5.9 | 2.9 | 1.5 |
| H100 | 15.9 | $62.72 | 15.9 | 4.0 | 2.0 | 1.0 |
| B200 | 14.1 | $88.01 | 14.1 | 3.5 | 1.8 | 0.9 |

**L4 with 8 shards is 6.0 hours for $38.39.** That fits overnight with four
hours to spare. Paying for a B200 to reach 1.8 hours costs $88, and 16 L4
shards reach 3.0 hours for the same $38.

### Verified, not assumed

Sharding only works if the containers genuinely run at once. `modal
run modal/capacity_test.py::check --shards 8` tests it on real L4s:

| | |
|---|---|
| Containers returned | 8 of 8 |
| Peak concurrent | **8** |
| Start spread | 2.3 s |
| Verdict | full parallelism |

All eight started within 2.3 seconds of each other on real `NVIDIA L4`
devices. 16 shards is untested; 8 is confirmed and already sufficient.

The per-complex rate behind these numbers (6.14 s at concurrency 4) is
**conservative**, because the benchmark ran 6 complexes per worker and the
~50-60 s checkpoint load is baked in. Production uses `--batch-size 256`, which
amortises it, so 6.0 hours is an upper bound.

## 2. Should the zero half-life rows be dropped? No

The argument was that rows with a half-life of exactly zero "contain no binding
information". The premise is inverted, and the data says so in three ways.

### A zero is a measurement, not a gap

These are **left-censored** observations: the complex dissociated faster than
the assay could resolve. That is the single most informative thing you can know
about a peptide-HLA pair when the task is ranking which peptides a given allele
presents. Dropping them removes the negative class and leaves a model that has
never seen a non-binder being asked to assign low scores.

The same peptides recur across alleles, so the zeros are not a separate
population of peptides: only 4.5% of peptides (253 of 5,633) disappear
entirely, and all 75 alleles survive. What the zeros carry is **allele
specificity** — a peptide with a 10-hour half-life on one allele and zero on
another is describing the groove, and that contrast is exactly what a
pan-specific model has to learn.

### It would strip the weakest-binding alleles

Zero fraction per allele ranges from 0.000 to 0.921. Seven of 68 well-populated
alleles would lose more than half their rows:

| Allele | rows | zero fraction |
|---|---:|---:|
| HLA-B*39:06(C67S) | 379 | 0.921 |
| HLA-B*14:01(C67S) | 374 | 0.890 |
| HLA-B*14:02(C67S) | 382 | 0.749 |
| HLA-B*45:01 | 368 | 0.671 |

By locus, HLA-B is 26.5% zeros against HLA-A's 13.1%. Dropping zeros therefore
deletes HLA-B data at twice the rate and nearly erases the three engineered
C67S constructs. Those are the hard cases the held-out-allele question turns
on.

### It breaks the committed splits

The five splits are moment-matched on `E[ln x]` and `E[ln x]²` **with zeros
included**. Dropping them shifts the mean of `ln(t + 0.1)` by **+0.64 natural
log units, 35% of its standard deviation**. The matching property the splits
were constructed for no longer holds, and any comparison against A0-A3 stops
being like-for-like.

### The instinct points at something real, but the fix is the loss

There is a genuine problem nearby: a floor turns 20% of the data into an
artificial spike that squared error will chase. How bad depends on the epsilon:

| epsilon | zeros land at | gap to next value |
|---|---:|---:|
| 0.001 (used by the split's moment matching) | ln = −6.91 | 3.93 log units (1.16 sd) |
| **0.1 (used by `train_mlp.py`)** | ln = −2.30 | **0.41 log units (0.22 sd)** |

At the 0.1 the repo already uses, the censored rows sit only 0.22 standard
deviations from the next observed value. The spike is largely handled. Where it
still bites, the textbook answer is a **censored (Tobit) likelihood** or a
two-part model — deleting censored observations is the known-wrong option,
because it biases the fit upward by construction.

### And the saving is negligible

| | rows | 10 epochs, last-4-blocks, L4 |
|---|---:|---:|
| All rows | 28,166 | $19.30 |
| Zeros dropped | 22,487 | $15.41 |

**$3.89.** Not a reason to invalidate the evaluation.

### What to do instead

Keep every row, train on `ln(t + 0.1)`, and **report metrics on the non-zero
stratum separately**. That answers "does it rank binders well among binders"
without biasing the fit, and it costs nothing. If the censoring still looks
like it is hurting, switch the loss, not the dataset.

## 3. Guards: the fine-tune cannot starve the extraction

`modal/finetune_modal.py` refuses to start unless three conditions hold. The
ordering is deliberate — the free checks run first.

| Guard | Check | Why |
|---|---|---|
| 1 | No live `boltz-pmhc-extract` app | the two would compete for GPUs and the container quota |
| 2 | Extraction cache has all 28,166 rows | the frozen-embedding arms are the cheap experiment that says whether fine-tuning is worth buying |
| 3 | `--max-usd` converted to GPU-seconds and enforced in-loop | the job checkpoints and exits at the ceiling instead of overrunning |

Guard 2 is overridable with `--no-require-cache`, which exists for debugging
and should not be used to skip the frozen-arm result.

### Budget against $170

| Item | Cost |
|---|---:|
| Benchmarking already spent (sweeps, probes, capacity, smoke) | ~$4 |
| Full extraction, 28,166 complexes, L4 x 8 shards, 6.0 h | $38 |
| Frozen-arm ladder, all arms x 5 splits x seeds | $0 (CPU) |
| Fine-tune last 4 blocks, 10 epochs, full set | $19 |
| **Committed** | **~$61** |
| Remaining | ~$109 |

Extraction and fine-tuning together are about 36% of the budget, and they are
**sequential by construction**, so the fine-tune cannot take capacity from the
extraction even if someone tries to launch it early.

A full 48-block fine-tune would be $274 over the whole dataset and does not
fit; over the pilot it is $27 and does. But the binding constraint there is
statistical rather than financial: 147M trainable parameters against 28,166
labels over 75 alleles will overfit whatever it costs, and `RESULTS.md` already
shows a 14.1M-parameter head losing to an 860-feature one-hot encoding.

## 4. Order of operations

1. Extract all 28,166 complexes. L4, 8 shards, 4 workers each. 6 hours, $38.
2. Merge the shards and score `BZS`, `BZZ`, `BZZU` on the five splits, plus A0
   re-scored on identical rows. CPU, free.
3. **Decide.** If `BZZ` does not move against A0's 0.771, a fine-tune of the
   same representation is unlikely to rescue it, and the negative result is
   itself a submission.
4. Only then fine-tune the last 4 blocks, 10 epochs, $19.

---

## 5. Is `pca:20` a good choice? No, and your own results show why

`pca:20` was introduced to make the ladder fair: every arm gets 43 x 20 = 860
features, so the head is byte-identical and only the feature content differs.
The intent is right. The instrument is not.

### The ladder's ordering tracks the compression, not the representation

| Arm | variance kept at `pca:20` | Spearman |
|---|---:|---:|
| A1 ESM-2 layer 0 | 1.000 | 0.761 |
| A2 ESM-2 layer 15 | 0.794 | 0.750 |
| A3 ESM-2 layer 33 | 0.725 | 0.736 |

Spearman and variance retained correlate at **r = 0.94** across the three arms,
and the ordering is identical. Then the uncompressed check:

| A3 layer 33 | Spearman |
|---|---:|
| `pca:20` | 0.736 |
| `flatten` | 0.767 |
| A0 one-hot, for reference | 0.771 |

Removing the compression recovers **+0.031**, and A3's apparent 0.035 deficit
against one-hot collapses to **0.004**. So the headline finding that deeper
ESM-2 layers are progressively worse may be substantially an artifact of how
hard each one was compressed. `RESULTS.md` concedes this in passing ("partly a
budget artifact"); the correlation above says it is more than partly.

### Three reasons PCA is the wrong instrument here

**It ranks by variance, not by relevance.** The leading directions of an
embedding's covariance are not the directions carrying binding information.
PCA is unsupervised, so it will happily spend all 20 components on whatever
dominates the variance — sequence composition, position, chain identity — and
discard a low-variance direction that happens to be the informative one.

**It is lossless for exactly one arm.** ESM-2 layer 0 has rank exactly 20 on
the 20-letter alphabet, so `pca:20` is lossless there and A0-vs-A1 is a clean
null. For every other arm it is lossy by an arm-dependent amount, which means
the rungs are not on the footing the design assumed.

**Dimension-matching is not information-matching.** Equal feature counts
equalise first-layer parameters, nothing more. Compressing 1280 dimensions to
20 and 128 dimensions to 20 are not comparable operations.

For the Boltz arms the fraction of directions kept would be:

| Arm | D | directions kept at K=20 |
|---|---:|---:|
| one-hot / L0 | 20 | 100% (lossless, rank 20) |
| ESM-2 L15 / L33 | 1280 | 1.6% |
| `BZS` | 384 | 5.2% |
| `BZZ` | 128 | 15.6% |

`BZZ` would be compressed far less harshly than the ESM-2 arms were, which
means `pca:20` would quietly *favour* it. A win under that setting would be as
suspect as A3's loss was.

### What to run instead

Report all three, and treat disagreement between them as the finding.

| Setting | What it answers | Parameter-matched? |
|---|---|---|
| `flatten` + strong weight decay | is the information there at all? | no, report separately |
| **fixed-variance PCA** (K per arm for 95%) | information-matched comparison | no, but report K |
| **random projection** to K=20 | unbiased dimension-match | yes |

Random projection is the better parameter-matched control: it equalises the
head exactly as PCA does, but it has no preference for high-variance
directions, so it cannot systematically discard the informative ones. The
Johnson-Lindenstrauss bound gives distance preservation in expectation.

Non-negotiable either way: **print the variance retained next to every score.**
That single column would have flagged the ESM-2 confound immediately.

## 6. How this regressor compares to PreFold-dG's

| | PreFold-dG | this arm |
|---|---|---|
| Tensors used | `s_inputs`, `s`, `z`, distogram | `s`, `z` (+ pLDDT) |
| Interchain weighting | 1/d², zero on intrachain | same, from Cα coordinates |
| Spatial pooling | outer product -> fixed [384x384] | per-position, 43 slots kept |
| Feature width | 147,456, then projected | 860 to 16,512 |
| Combining tensors | project each, then **average** | separate arms (not yet combined) |
| Head | 2-layer MLP | `F -> 256 -> 128 -> 1` |
| Loss | joint MSE over ΔG_wt, ΔG_mut, ΔΔG | single MSE on ln(t½+0.1) |
| Training data | SKEMPI 2.0, 5,817 rows / 334 complexes | 28,166 rows / 75 alleles |

The heads are near-identical. The real differences are three, and two of them
are gaps on our side.

**Pooling (ours is better here).** Their outer-product pooling exists to give
variable-length complexes a fixed size. Ours are always 182+9, so we keep
per-position structure and preserve the P2/P9 anchor signal their pooling would
average away.

**`s_inputs` is missing (gap).** Their ablation found that excluding
`s_inputs` "has the most deleterious effect" on **ΔG** specifically — absolute
affinity, which is the closer analogue of our absolute half-life — while `s`
mattered most for ΔΔG. We use only `s`. Adding `s_inputs` is nearly free: it is
the input embedder's output, available in the same forward pass.

**Tensors are never combined (gap).** They project each embedding and average
the three. We score `BZS` and `BZZ` as separate arms and have never run the
concatenation, which is the configuration closest to theirs and the one most
likely to work. It should be a registered arm.

Their multi-task loss also has a direct analogue we are not using: training on
half-life jointly with binding-affinity data, which is the TLStab transfer
result (PMID 38577265) from a different angle.

## 7. Why the distogram is not included — and the part that should be

The distogram is **not** an information-bearing addition, and this is provable
rather than a judgement call.

`DistogramModule` is a single `nn.Linear(128, 64)` applied to `z`. Our `BZZ`
contraction is a weighted sum over interface pairs. Both are linear, so they
commute: contracting then projecting equals projecting then contracting.
Verified numerically to 1.07e-14, and by least squares, the contracted
distogram logits are recoverable from contracted `z` at **R² = 1.000000**.

Given `BZZ`, the distogram logits add exactly nothing — the MLP's first linear
layer can reproduce them. PreFold-dG gains from including it because their
pooling collapses `z` to 128 numbers total and the distogram survives as a
differently-pooled view; ours keeps `z` per-position, so it is strictly
redundant.

**The part worth adding is the nonlinear readout.** Softmax over the 64 bins is
not linear, so statistics derived from the *distribution* are not recoverable:

| Derived statistic | best affine fit from contracted `z` |
|---|---:|
| distogram logits (control) | R² = 1.0000 |
| expected distance | R² = 0.2046 |
| **bin entropy (geometric uncertainty)** | **R² = 0.0252** |

Bin entropy is 97.5% unexplained by any affine function of `z`. And it is the
mechanistically interesting one for this task: entropy over the predicted
distance distribution is the model's uncertainty about where a residue sits,
which is a proxy for a floppy interface — and half-life is an off-rate, which
is dynamics. (Measured on random `z`; the analytic argument holds regardless,
the magnitudes on real `z` are not yet known.)

This needs no GPU re-run. `boltz_ZRAW.npy` now caches the raw 9x34 interface
block, so the distogram head's weights can be applied offline on CPU and any
nonlinear statistic derived from it afterwards.

## 8. The combined arm `BZSZ`, and the scale trap it exposed

### Why combine at all

PreFold-dG does not score its tensors separately. It projects each pooled
embedding to a common width, applies batchnorm, a nonlinearity and dropout,
then **averages** the three into one aggregated vector before a two-layer MLP.
Scoring `BZS` and `BZZ` as isolated arms answers "is either sufficient on its
own", which is not the question their result bears on.

The two are plausibly complementary rather than redundant:

| | `BZS` | `BZZ` |
|---|---|---|
| What it carries | per-position residue identity and trunk context | interface pair structure |
| Width per slot | 384 | 128 |
| Closest existing arm | A3 ESM-2 layer 33 | nothing — no sequence model has this |
| Weakness | no pair channel at all | contraction averages over 34 groove positions, which can wash out the peptide residue's own identity |

Note `z` is not pure geometry: it is initialised from outer sums of
`s_inputs`, so it already carries residue identity for both members of each
pair. The concern is not that `BZZ` lacks identity but that averaging over 34
partners dilutes it, which is precisely the gap a clean per-position `s`
channel fills.

### Concatenate, don't average

| | Averaging (theirs) | Concatenation (ours) |
|---|---|---|
| Head input width | `d`, independent of tensor count | `Σ dᵢ` |
| Inductive bias | the views must be commensurable | each source weighted independently |
| Suits | small data — 5,817 rows / 334 complexes | 28,166 rows |

Averaging is the stronger regulariser and makes sense at their scale. We have
roughly 5x the rows, so the extra parameters are affordable, and letting the
head weight each source independently is the less presumptuous choice. Both
arms share the 43-slot layout, so concatenating along the feature axis is well
defined: each slot becomes 384 + 128 = **512** features, and the pooling
machinery works unchanged.

Built offline from an existing cache, no GPU:

```bash
python scripts/build_combined.py <cache dir>     # writes boltz_SZ.npy, arm BZSZ
```

### The trap: it would have silently been `BZS`

`boltz_S` peaks around 1383 and `boltz_Z` around 294. Concatenating raw and
fitting PCA lets the larger block monopolise the components. Measured on a
synthetic cache with that scale gap:

| | share of PCA component mass on the `s` block |
|---|---:|
| No prescaling | **100.0%** |
| With prescaling | 74.8% (its 75% column share) |

Without prescaling the pair tensor is **invisible** — the "combined" arm would
have reproduced `BZS` and we would have concluded the combination does not
help. This is the kind of failure that produces a confident wrong answer rather
than an error.

The fix is per-dimension standardisation of the cache block **before** pooling,
fitted on the training rows. The existing `Standardiser` runs *after* pooling,
which is too late for PCA. Now in `CachedArmEncoder(prescale=True)`, covered by
`scripts/test_combined.py` (10 checks).

### It also fixes a flaw already on the record

`docs/01_embedding_extraction.md` §8.1 records that ESM-2 layer 33 is recorded
post-`emb_layer_norm_after` (mean abs 0.145) while layer 15 is raw (2.344), a
16x gap, and warns that "A2 vs A3 partly measures input scaling rather than
representation quality". Pre-pooling standardisation removes that asymmetry.

**This changes the published A1/A2/A3 numbers.** Prescaling is on by default
because it is the more correct setting; pass `--no-prescale` to reproduce
`RESULTS.md` as committed. Re-running the ESM-2 arms with it on is CPU-only and
free, and worth doing so the whole ladder is scored under one convention —
especially given §5, which suggests those numbers were already partly an
artifact of how the arms were compressed.
