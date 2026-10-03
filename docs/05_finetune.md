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
