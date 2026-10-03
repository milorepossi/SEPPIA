# Modal: best setup, measured cost, and whether fine-tuning is affordable

All numbers measured on Modal, nine GPU types, `boltz==2.2.1`, 191-token
peptide-HLA complexes. Reproduce with:

```bash
modal run modal/bench_modal.py::sweep      # $/complex across all nine cards
modal run modal/bench_modal.py::refine     # steady state at the memory limit
modal run modal/bench_modal.py::finetune   # forward+backward feasibility
```

Raw results: `modal/sweep_results.json`, `modal/refine_results.json`,
`modal/finetune_results.json`.

---

## 1. Verdict

**Use L4 at concurrency 4-6, sharded across containers.** Full 28,166-complex
extraction lands at roughly $38 against a $150 budget, leaving ample room for
fine-tuning experiments.

Two things changed versus the earlier Hugging Face run. The cuEquivariance
triangle kernels **install cleanly on Modal** and were used on every card,
where on HF Jobs they could not be (that image ships CUDA 13 and the
`cuequivariance-ops-torch` wheels are built against CUDA 12). And Modal fans
out, so **wall clock is bought with shards, not with a bigger card**.

## 2. The sweep: ranked by dollars per complex

Cost is `$/hour ÷ complexes/hour`, so the fastest card is not the cheapest.
Each GPU at its best concurrency:

| GPU | s/complex | conc | $/complex | Full 28,166 | GPU-hours |
|---|---:|---:|---:|---:|---:|
| **L4** | 6.14 | 4 | **$0.001364** | **$38.41** | 48.0 |
| A10G | 4.75 | 4 | $0.001455 | $40.97 | 37.2 |
| L40S | 3.01 | 4 | $0.001630 | $45.92 | 23.5 |
| H100 | 2.03 | 8 | $0.002229 | $62.78 | 15.9 |
| A100-40GB | 3.83 | 4 | $0.002235 | $62.96 | 30.0 |
| H200 | 1.97 | 8 | $0.002479 | $69.83 | 15.4 |
| A100-80GB | 3.76 | 4 | $0.002609 | $73.50 | 29.4 |
| B200 | 1.80 | 8 | $0.003128 | $88.11 | 14.1 |
| T4 | 26.37 | 1 | $0.004325 | $121.81 | 206.3 |

The large cards really are faster. A B200 at 1.80 s/complex is 3.4x an L4. It
is also 2.3x the price per complex, and since sharding makes wall clock cheap,
there is no reason to pay for it.

**T4 is the trap.** It is the cheapest card per hour at $0.59 and the most
expensive per complex. It did not crash, and `torch.cuda.is_bf16_supported()`
even reports `True`, but at compute capability 7.5 with no real bf16 tensor
cores it runs 4.3x slower than an L4 at the same concurrency. Avoid it.

## 3. Concurrency is the main lever

A single `boltz predict` process wastes most of any GPU. At 191 tokens the
per-worker footprint is only 2.8-3.5 GiB:

| GPU | conc 1 | conc 2 | conc 4 | conc 8 | MiB/worker |
|---|---:|---:|---:|---:|---:|
| L4 | 14.37 | 8.53 | 6.14 | — | 2817 |
| A10G | 11.54 | 7.04 | 4.75 | — | 2897 |
| L40S | 10.01 | 5.11 | 3.01 | — | 3099 |
| A100-80GB | 11.43 | 6.51 | 3.76 | — | 3095 |
| H100 | 8.70 | 4.51 | 2.73 | 2.03 | 3367 |
| B200 | 8.90 | 4.67 | 2.74 | 1.80 | 3468 |

Going from 1 to 4 workers is a 2.3x throughput gain on an L4 and 3.2x on an
H100. **Running one complex at a time is the single most expensive mistake
available here.**

Two caveats the sweep itself exposes, both addressed by the `refine`
entrypoint:

- Concurrency was capped well below the memory limit. An 80 GiB H100 at
  concurrency 8 uses 26.9 GiB. At 3.4 GiB per worker it has room for ~20.
- At 6 complexes per worker, the measurement is start-up dominated: `boltz
  predict` spends roughly 50-60 s loading the 2.3 GB checkpoint before the
  first complex. Production runs `--batch-size 256`, which amortises that
  away, so these s/complex figures are **conservative**. `refine` extracts the
  marginal rate from a two-point slope instead.

## 4. Is fine-tuning feasible? Yes, with one hard constraint

The probe times forward against forward+backward through the last N Pairformer
blocks at the real 191-token shapes. Two things make this the right thing to
measure. The pairwise stack is the trunk's expensive part by a wide margin.
And caching the last block's input to skip the trunk is **not** an escape
route: `z` is `[191, 191, 128]` float32 = 17.8 MB per complex, so the full
dataset would be 498 GB (the pilot, 50 GB, is borderline feasible).

### Activation checkpointing is mandatory below 40 GiB

| blocks | checkpointing | peak | L4 (22 GiB) |
|---|---|---:|---|
| 48 | off | 40.0 GiB | **OOM** |
| 48 | on | 2.37 GiB | fits |

A 17x memory reduction for about 1.3x the compute. Boltz-2's own training
config sets `activation_checkpointing: true`, so this is the supported path.

### Cost per epoch

A real step is not one Pairformer pass. Boltz uses `recycling_steps=3`, so four
trunk passes with gradients only on the last. Step cost is modelled as
`3 x forward + 1 x (forward+backward)`. The MSA module and input embedder are
**not** covered, so these are lower bounds.

| Scope | GPU | step | 1 epoch pilot | 1 epoch full | 10 epochs full |
|---|---|---:|---:|---:|---:|
| last 4 blocks, 12.3M params | L4 | 0.31 s | $0.19 | $1.93 | **$19** |
| last 4 blocks | H100 | 0.08 s | $0.25 | $2.47 | $25 |
| all 48 blocks, 147M params | L4 (ckpt) | 4.38 s | $2.73 | $27.37 | $274 |
| all 48 blocks | A100-80GB | 1.45 s | $2.83 | $28.28 | $283 |
| all 48 blocks | B200 | 0.97 s | $4.73 | $47.38 | $474 |

### What to actually do

**Fine-tune the last 4 Pairformer blocks over the full dataset.** Ten epochs is
about $19 on an L4, which is inside the budget by a wide margin.

A full-trunk fine-tune is affordable over the **pilot only** (10 epochs, ~$27).
Over the full dataset it is $274 and does not fit.

The binding constraint is statistical, not financial. 147M trainable parameters
against 28,166 labels, of which 20% are left-censored at zero and which span
only 75 alleles, will overfit badly however cheap the compute is. For context,
`RESULTS.md` already shows a 14.1M-parameter head losing to an 860-feature
one-hot encoding. The defensible ladder is: head only, then last 1-4 blocks,
then LoRA on the pairwise stack. Full fine-tuning is the least interesting rung
even though it is now purchasable.

## 5. Recommended budget split

| Item | GPU | Cost |
|---|---|---:|
| Pilot extraction, 2,814 complexes | L4 conc 4-6 | ~$4 |
| Full extraction, 28,166 complexes | L4, 8-16 shards | ~$38 |
| Head-only ladder, all arms x 5 splits x seeds | CPU | $0 |
| Last-4-block fine-tune, 10 epochs, full set | L4 | ~$19 |
| Benchmarking already spent | all nine | ~$2.10 |
| **Committed** | | **~$63** |
| Reserve for reruns and a second configuration | | ~$87 |

Extraction is the cheap half. With sharding, the full extraction is about
6 hours of wall clock at 8 shards, or 3 hours at 16, for the same $38.

## 6. What is still unmeasured

- **MSA depth.** Everything above runs `msa: empty` on both chains. Boltz-2
  warns that single-sequence mode degrades predictions, and MSA depth is also
  a cost driver. One measured condition is owed here; there are only 75 unique
  HLA sequences, so the alignments themselves are nearly free.
- **The MSA module and input embedder** in the fine-tune step cost.
- **Concurrency above 8**, and the start-up-free steady-state rate. Both are
  what `refine` measures; its numbers supersede section 2 where they differ.
- **CPU allocation per container.** The `refine` pass at concurrency 6 on an L4
  ran markedly slower than concurrency 4 did in the sweep, which points at CPU
  starvation rather than a GPU limit: six `boltz predict` processes each spawn
  `--num_workers 2` dataloader workers, and the Modal functions here request no
  explicit `cpu=`, so they get the default allocation. Before raising
  concurrency past 4, request CPU in proportion
  (`@app.function(gpu=..., cpu=4*2)`) and re-measure. This is the likeliest
  reason the large cards did not scale as far as their memory allows.
