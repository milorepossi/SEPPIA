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

## Arms

| Arm | Content | Shape | Role |
|---|---|---|---|
| `BZS` | trunk single representation | `(n, 43, 384)` | comparable to A3, expected to tie |
| `BZZ` | interface pair contraction, 1/d² | `(n, 43, 128)` | **the hypothesis** |
| `BZZU` | same, uniform weights | `(n, 43, 128)` | is distance weighting earning its keep? |

All three come from one extraction pass. Caches use the same 43 slots and row
order as the ESM-2 arms, so `train_mlp.py` is unchanged:

```bash
python train_mlp.py --arm BZZ --embeddings-dir <boltz cache> --pooling pca:20
```

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
