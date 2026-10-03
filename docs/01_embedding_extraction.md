# Task 1 — ESM-2 embedding extraction

Technical report for `scripts/extract_embeddings.py`.

Status: **implemented, validated on GPU, 500-row run green.** Full 28,166-row run
deferred to Modal.

---

## 1. What this produces

One forward pass of ESM-2 per dataset row over the concatenation
`hla_seq + LINKER + peptide`, caching 43 residue positions from each of three
layer depths.

```
embeddings/
  concat_L0.npy    (28166, 43, 1280) fp16   3.10 GB   token embeddings (control)
  concat_L15.npy   (28166, 43, 1280) fp16   3.10 GB   mid-stack
  concat_L33.npy   (28166, 43, 1280) fp16   3.10 GB   final layer
  index.json       row -> (hla_seq, peptide), source_row, slot layout
  progress.json    resume checkpoint (rows_done + fingerprint)
                                     total  9.30 GB
```

**The cache is split-agnostic.** Row `i` of every array corresponds to
`index.json["pairs"][i]` and to `source_row == i`. A split selects its rows by
joining on `source_row`. One extraction serves all 5 splits and all arms; nothing
is keyed by split.

### Slot layout (43 positions)

| Slots | Count | Content | Source indices (into the 191-residue concat) |
|---|---|---|---|
| `0..8` | 9 | peptide | `182..190` |
| `9..42` | 34 | HLA pseudosequence | `PSEUDOSEQ_INDICES` (NetMHCpan contact positions) |

The peptide offset is **derived** as `HLA_LEN + len(LINKER)`, so the `GGGGS`
ablation needs no index edits.

---

## 2. Data flow

```
rasmussen_et_al_dataset.xlsx  (28166 x [allele, peptide, thalf_hours, hla_seq, hla_pseudoseq])
         |
         |  load_rows()        + source_row = 0..n-1 ; assert all |hla_seq|=182, |peptide|=9
         v
   ASSERT 1  assert_pseudoseq_indices()      <-- before any compute is spent
         |   hla_seq[PSEUDOSEQ_INDICES] == hla_pseudoseq, all 75 alleles
         v
   load_model()  ->  model, alphabet, device ; dim = model.embed_dim
         |
         |  disk pre-check ; open_outputs() -> 3 memmapped .npy ; write_index()
         v
   per batch of 32:
         build_sequence()  hla_seq + LINKER + peptide          191 residues
         batch_converter() [BOS] + 191 + [EOS]                 193 tokens
             |
             +--> ASSERT 2 (first batch only) assert_token_alignment()
             |
         model(tokens, repr_layers=[0,15,33])  under fp16 autocast
             |
         index_select(dim=1, [p+1 for p in concat_positions])  <-- +1 skips BOS
             |
         .to(float16).cpu().numpy()  ->  arrays[L][start:start+len] = ...
         v
   every 2000 rows: array.flush() + atomic write_progress()
```

No batch is ever accumulated in a Python list; writes go straight through
`np.lib.format.open_memmap`.

---

## 3. Code structure

Deliberately factored so a Modal entrypoint can call the pieces. Every path is an
argument; nothing is read from the environment.

| Function | Role |
|---|---|
| `slot_layout()` | → `(concat_positions, peptide_slots, pseudoseq_slots)`; derives peptide offset from `len(linker)` |
| `build_sequence()` | `hla_seq + linker + peptide` |
| `load_rows()` | read xlsx, attach `source_row`, validate sequence lengths, apply `--limit` |
| `assert_pseudoseq_indices()` | **Assertion 1** |
| `assert_token_alignment()` | **Assertion 2** |
| `load_model()` | fair-esm load → `(model, alphabet, device)`, eval mode, `requires_grad_(False)` |
| `open_outputs()` | one memmapped `.npy` per layer, `w+` fresh / `r+` resume |
| `write_index()` | atomic `index.json` |
| `read_progress()` / `write_progress()` | resume checkpoint + fingerprint guard |
| `check_free_space()` | pre-flight disk check |
| `extract_embeddings()` | **the plain function a Modal entrypoint calls** |
| `main()` | argparse CLI only |

Top-level constants: `MODEL_NAME`, `LAYERS`, `LINKER`, `PSEUDOSEQ_INDICES`,
`HLA_LEN`, `PEPTIDE_LEN`, `BATCH_SIZE`, `CHECKPOINT_EVERY`.

### CLI

```bash
python scripts/extract_embeddings.py rasmussen_et_al_dataset.xlsx \
    --out-dir embeddings [--model esm2_t33_650M_UR50D] [--layers 0 15 33] \
    [--batch-size 32] [--limit N] [--device cuda] [--checkpoint-every 2000] \
    [--linker ""] [--no-resume]
```

---

## 4. The two required assertions — both PASS

### Assertion 1 — pseudosequence indices

`hla_seq[PSEUDOSEQ_INDICES] == hla_pseudoseq` for **75/75 alleles, 0 mismatches.**
Runs before the model is loaded, so a bad index list costs no GPU time.

### Assertion 2 — BOS/EOS alignment

ESM-2's alphabet is `prepend_bos=True, append_eos=True` (`cls_idx=0, eos_idx=2`),
so residue `j` of the concat lives at token `j+1`. On the first batch actually
pushed through the model, for all 32 rows:

- token count is exactly `1 + 182 + 0 + 9 + 1 = 193`;
- `tokens[0] == cls_idx`, `tokens[-1] == eos_idx`;
- stripping them leaves concat `0..181 == hla_seq` and `182..190 == peptide`;
- **the token positions actually gathered decode back to the peptide and to
  `hla_pseudoseq`.**

That last check is the load-bearing one: it validates the gather indices
themselves rather than the layout they were derived from. It re-runs on resume.

### Independent third check (layer 0 is provably context-free)

Not required, but it is the strongest available correctness probe, so it was run
as a dev script:

Layer 0 is `embed_scale * embed_tokens(tokens)`, recorded before the first
transformer block. ESM-2 applies **rotary** position embeddings inside each
attention module rather than adding them here, so a layer-0 slot vector is a pure
function of that slot's amino-acid letter — no position, no context.

| Check | Result |
|---|---|
| Layer 0: same letter ⇒ same vector, over all 43 slots × all rows | 20 distinct residues, **0 violations** |
| Layer 0: same peptide, different HLA ⇒ identical peptide slots | PASS (5 row pairs) |
| Layer 15: same residue, different HLA context ⇒ differs | max abs Δ = 15.88 |
| Layer 33: same residue, different HLA context ⇒ differs | max abs Δ = 0.80 |

Any off-by-one or slot transposition in the gather breaks row 1 immediately.

---

## 5. Measurements (Tesla P100-PCIE-16GB, batch 32, fp16 autocast)

Real 500-row run: **22.67 seq/s**, both assertions PASS, arrays reload clean.

| batch | seq/s | peak GPU |
|---|---|---|
| 8 | 21.16 | 2.62 GiB |
| 16 | 22.02 | 2.74 GiB |
| **32** | **22.99** | 2.98 GiB |
| 64 | 23.47 | 3.45 GiB |

Compute-bound: batch 64 buys ~3%, so batch 32 was kept as specified. GPU memory is
a non-issue (3 GiB of 16 GiB).

### Extrapolated wall clock

```
28166 rows / 22.67 seq/s = 1242 s = 20.7 min   (this P100, excl. model download)
```

A Modal A100/H100 has no P100 fp16 tensor-core handicap and should land several
times faster; the 2.6 GB weight download and the 9.3 GB volume write will be a
meaningful share of the total there.

### fp16 numerics

fp16 autocast vs full fp32 on identical rows:

| layer | max abs diff | mean abs value | max rel | cosine |
|---|---|---|---|---|
| 0 | 0.00000 | 0.059 | 0 | 1.0000 |
| 15 | 0.139 | 2.344 | 6.2e-04 | 1.0000 |
| 33 | 0.0083 | 0.145 | 9.8e-04 | 1.0000 |

fp16 **storage** round-trip error ≤ 0.0625 on layer 15 (abs max 223, vs the fp16
ceiling of 65504 — no overflow risk). Precision is not a concern at this scale.

---

## 6. Robustness

| Property | Implementation | Verified |
|---|---|---|
| Resumable | `progress.json` + `r+` memmap reopen; restart at `rows_done` | Rolled back to row 32, zeroed tail, resumed → **bitwise identical** (`np.array_equal`) on all 3 layers |
| Wrong-run guard | SHA-256 fingerprint over pairs + model + layers + linker + n_rows | Re-running with `--layers 0 3` on a `0 3 6` cache **refuses** instead of mixing |
| Crash-safe checkpoints | `flush()` then atomic `.tmp` → `replace()` | — |
| Disk pre-flight | `shutil.disk_usage` vs computed need, raises before compute | Relevant: `/home` is 97% full (277 GB free, shared NFS) |
| Layer bounds | asserts `max(layers) <= model.num_layers` | — |
| Model-agnostic | `dim` read from `model.embed_dim`, not hardcoded | 8M dev model (d=320) and 650M (d=1280) both work |

---

## 7. Environment notes (repo differed from the brief)

| Brief assumed | Actual | Handling |
|---|---|---|
| `DATA/rasmussen_et_al_dataset.xlsx` | dataset at repo root | path is a CLI argument |
| vendored `esm/` repo present | **absent** | `pip install fair-esm==2.0.0` — same `repr_layers` convention, so indexing is unaffected |
| torch available | absent in active `base` conda env | used `~/miniconda3/envs/myconda` (torch 2.6.0+cu124, CUDA OK) |

GPU: Tesla P100-PCIE-16GB, compute capability 6.0 — fp16 compute works, **no
tensor cores, no bf16**. Do not switch the autocast dtype to bf16 on this box.

Dataset confirmed against the brief: 28,166 rows; peptide all 9-mers, 5,633
unique; `hla_seq` all 182 aa, 75 unique; `hla_pseudoseq` 34 aa; `(hla_seq,
peptide)` pairs unique (0 duplicates); no nulls; standard 20-letter alphabet.

---

## 8. Two findings that affect later arms

### 8.1 Layer 33 is post-layer-norm, layer 15 is not

In `esm/model/esm2.py`, the final representation is recorded **after**
`emb_layer_norm_after`; intermediate layers are recorded raw:

```python
for layer_idx, layer in enumerate(self.layers):
    x, attn = layer(...)
    if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x.transpose(0, 1)   # raw
x = self.emb_layer_norm_after(x)
if (layer_idx + 1) in repr_layers:
    hidden_representations[layer_idx + 1] = x                        # layer-normed
```

So **A3 receives layer-normed features and A2 raw ones** (mean abs 0.145 vs
2.344 — a 16× scale gap). The ladder is supposed to differ *only* in input
features, so feature standardisation must be fitted per arm (train split only) or
A2 vs A3 partly measures input scaling rather than representation quality.

### 8.2 Two alleles share a pseudosequence

`HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` — **756 rows** — have identical
34-mer pseudosequences. Their `hla_seq` differ at exactly one position, index
**10**, which is not a contact position.

- The concat input (full 182-aa `hla_seq`) **does** see the difference.
- The **A0 one-hot baseline collapses them**, because it encodes only the 34
  pseudosequence positions — two distinct alleles become indistinguishable.
- Note the cached *pseudoseq slots* are identical for these two at layer 0 too,
  by construction; at layers 15/33 they differ, because attention has seen
  position 10.

So A0 and A1 share this limitation, while A2/A3 do not. 756 of 28,166 rows
(2.7%) are affected — record it as a known ceiling on A0/A1 rather than reading
any A1→A2 gain as purely "context helps".

---

## 9. Reproducing

```bash
PY=~/miniconda3/envs/myconda/bin/python

# dev smoke test, 8M model, ~1 s
$PY scripts/extract_embeddings.py rasmussen_et_al_dataset.xlsx \
    --out-dir /tmp/emb8m --model esm2_t6_8M_UR50D --layers 0 3 6 --limit 64

# the validated 500-row 650M run, ~22 s + weight download
$PY scripts/extract_embeddings.py rasmussen_et_al_dataset.xlsx \
    --out-dir /tmp/emb650 --limit 500
```

`embeddings/` and `__pycache__/` are gitignored — the full cache is 9.3 GB and
must never be committed.
