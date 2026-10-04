"""
Modal Training Pipeline for 2-Instance ESMc-6B with LoRA Fine-Tuning.

Architecture:
1. Two Dedicated Foundation Instances:
   - Allele ESMc-6B: Dedicated instance encoding 182-aa HLA binding cleft.
   - Peptide ESMc-6B: Dedicated instance encoding 9-mer peptide epitope.
2. Parameter-Efficient LoRA Adapters:
   - Attached to attention output projections (attn.out_proj) on the top 4 transformer blocks (76..79)
     of both instances (rank r=8, alpha=16, dropout=0.05).
   - Base blocks (0..75) are cached once for unique sequences.
3. Downstream DeepExtract Interaction Architecture:
   - PeptideEncodingInstance (d_in=2560 -> 384, positional embeddings P1..P9, intra-peptide self-attention, FFN).
   - AlleleEncodingInstance (d_in=2560 -> 384, positional embeddings 1..182, intra-cleft self-attention, FFN).
   - 2D Pairwise Contact & Affinity Track (outer product + bidirectional attention maps + dot affinity + cosine similarity -> 35 channels).
   - 2D ResNet Blocks (downsampling HLA to 9 positions to create aligned 9x9 interaction map).
   - Position-weighted and Anchor Readout (P2 B-pocket anchor, P9 F-pocket anchor, weighted & max pooling).
4. Multi-Objective Stability Loss:
   - Regression (Smooth L1) + Pearson correlation loss + margin ranking loss + binary classification (where applicable).
"""

import json
import math
import os
import time
from pathlib import Path
import modal

app = modal.App("esmc-6b-lora-2inst")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = str(PROJECT_ROOT / "splits")
DATA_DIR = str(PROJECT_ROOT / "data")

hf_volume = modal.Volume.from_name("esmc-hf-cache", create_if_missing=True)

esmc_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch",
        "transformers",
        "esm==3.4.1.post1",
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "tqdm",
    )
    .add_local_dir(SPLITS_DIR, remote_path="/root/splits")
    .add_local_dir(DATA_DIR, remote_path="/root/data")
)


# -----------------------------------------------------------------------------
# REMOTE TRAINING FUNCTION
# -----------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    image=esmc_image,
    volumes={"/root/.cache/huggingface": hf_volume},
    timeout=2400,
)
def run_lora_2inst_experiment(
    split_name: str = "iid",
    epochs: int = 30,
    batch_size: int = 128,
    lr_head: float = 3e-4,
    lr_lora: float = 1e-4,
    weight_decay: float = 1e-4,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    num_lora_layers: int = 4,
    d_model: int = 384,
    n_heads: int = 8,
    d_pair: int = 32,
):
    import gc
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        mean_squared_error,
        mean_absolute_error,
    )
    from esm.models.esmc import EsmcModel, EsmcTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print("=" * 80)
    print(f"2-INSTANCE ESMC-6B LORA FINE-TUNING: SPLIT '{split_name.upper()}'")
    print("=" * 80)
    print(f"Compute Device: {device} ({gpu_name}) | Total VRAM: {total_mem_gb:.2f} GB")
    print(f"Config: epochs={epochs}, batch_size={batch_size}, lr_head={lr_head}, lr_lora={lr_lora}")
    print(f"LoRA: rank={lora_rank}, alpha={lora_alpha}, top {num_lora_layers} layers of each instance")
    print(f"Downstream: d_model={d_model}, n_heads={n_heads}, d_pair={d_pair}")
    print("=" * 80)

    t_start_total = time.time()

    # 1. Load Data Splits
    splits_dir = Path("/root/splits")
    train_path = splits_dir / f"{split_name}_train.csv"
    val_path = splits_dir / f"{split_name}_val.csv"
    test_path = splits_dir / f"{split_name}_test.csv"

    if not train_path.exists():
        train_path = splits_dir / "binding_modes" / f"{split_name}_train.csv"
        val_path = splits_dir / "binding_modes" / f"{split_name}_val.csv"
        test_path = splits_dir / "binding_modes" / f"{split_name}_test.csv"

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    print(f"\n[Step 1/5] Loaded '{split_name}' Dataset:")
    print(f"  Train: {len(df_train):,} pairs ({df_train['peptide'].nunique()} unique peptides, {df_train['allele'].nunique()} alleles)")
    print(f"  Val:   {len(df_val):,} pairs ({df_val['peptide'].nunique()} unique peptides, {df_val['allele'].nunique()} alleles)")
    print(f"  Test:  {len(df_test):,} pairs ({df_test['peptide'].nunique()} unique peptides, {df_test['allele'].nunique()} alleles)")

    # For binding mode splits, also prepare cross-mode test sets for cross-regime evaluation
    is_binding_mode = split_name in ["non_binding", "mild_binding", "strong_binding"]
    cross_test_dfs = {}
    if is_binding_mode:
        for bm in ["non_binding", "mild_binding", "strong_binding"]:
            bmp = splits_dir / f"{bm}_test.csv"
            if not bmp.exists():
                bmp = splits_dir / "binding_modes" / f"{bm}_test.csv"
            cross_test_dfs[bm] = pd.read_csv(bmp)

    # 2. Collect unique sequences across all splits
    all_pep_set = set(df_train["peptide"]).union(set(df_val["peptide"])).union(set(df_test["peptide"]))
    all_hla_set = set(df_train["hla_seq"]).union(set(df_val["hla_seq"])).union(set(df_test["hla_seq"]))
    for o_df in cross_test_dfs.values():
        all_pep_set.update(o_df["peptide"])
        all_hla_set.update(o_df["hla_seq"])

    unique_peptides = sorted(list(all_pep_set))
    unique_hlas = sorted(list(all_hla_set))
    pep_to_idx = {p: i for i, p in enumerate(unique_peptides)}
    hla_to_idx = {h: i for i, h in enumerate(unique_hlas)}

    print(f"\n[Step 2/5] Sequence Extraction Requirements:")
    print(f"  Unique peptides: {len(unique_peptides):,} (9 residues)")
    print(f"  Unique HLAs:     {len(unique_hlas):,} (182 residues)")

    # 3. LoRA Module Definition
    class LoRALinear(nn.Module):
        def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
            super().__init__()
            self.base_linear = base_linear
            self.rank = rank
            self.scaling = alpha / rank

            for p in self.base_linear.parameters():
                p.requires_grad = False

            in_features = base_linear.in_features
            out_features = base_linear.out_features

            self.lora_A = nn.Parameter(
                torch.empty(rank, in_features, dtype=base_linear.weight.dtype, device=base_linear.weight.device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, rank, dtype=base_linear.weight.dtype, device=base_linear.weight.device)
            )
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base_linear(x)
            lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling
            return base_out + lora_out

    # 4. Load Models & Attach LoRA
    print("\n[Step 3/5] Loading 2 Dedicated ESMc-6B Foundation Instances and Attaching LoRA...")
    t0 = time.time()
    tokenizer = EsmcTokenizer.from_pretrained("biohub/ESMC-6B")

    # Instance 1: Allele Foundation Model
    print("  -> Loading Instance 1 (Allele Foundation Instance)...")
    t_load1 = time.time()
    allele_model = EsmcModel.from_pretrained("biohub/ESMC-6B", dtype=torch.bfloat16, device="cuda")
    print(f"     Instance 1 loaded in {time.time() - t_load1:.2f}s | VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # Instance 2: Peptide Foundation Model
    print("  -> Loading Instance 2 (Peptide Foundation Instance)...")
    t_load2 = time.time()
    pep_model = EsmcModel.from_pretrained("biohub/ESMC-6B", dtype=torch.bfloat16, device="cuda")
    print(f"     Instance 2 loaded in {time.time() - t_load2:.2f}s | VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # Freeze base weights
    for p in allele_model.parameters():
        p.requires_grad = False
    for p in pep_model.parameters():
        p.requires_grad = False

    total_blocks = len(allele_model.transformer.blocks)
    split_idx = total_blocks - num_lora_layers  # e.g., 80 - 4 = 76
    print(f"  -> Attaching LoRA (r={lora_rank}, alpha={lora_alpha}) to top {num_lora_layers} blocks (blocks {split_idx}..{total_blocks-1})...")

    allele_lora_params = []
    pep_lora_params = []

    for i in range(split_idx, total_blocks):
        # Allele instance LoRA
        b_allele = allele_model.transformer.blocks[i]
        b_allele.attn.out_proj = LoRALinear(b_allele.attn.out_proj, rank=lora_rank, alpha=lora_alpha)
        allele_lora_params.extend([b_allele.attn.out_proj.lora_A, b_allele.attn.out_proj.lora_B])

        # Peptide instance LoRA
        b_pep = pep_model.transformer.blocks[i]
        b_pep.attn.out_proj = LoRALinear(b_pep.attn.out_proj, rank=lora_rank, alpha=lora_alpha)
        pep_lora_params.extend([b_pep.attn.out_proj.lora_A, b_pep.attn.out_proj.lora_B])

    total_allele_lora = sum(p.numel() for p in allele_lora_params)
    total_pep_lora = sum(p.numel() for p in pep_lora_params)
    print(f"     Allele LoRA parameters: {total_allele_lora:,} ({total_allele_lora * 2 / (1024**2):.2f} MB)")
    print(f"     Peptide LoRA parameters: {total_pep_lora:,} ({total_pep_lora * 2 / (1024**2):.2f} MB)")

    # 5. Pre-compute Prefix Cache through Frozen Blocks (0..split_idx-1)
    print(f"\n[Step 4/5] Pre-computing exact prefix cache for blocks 0..{split_idx-1}...")
    t_cache = time.time()

    # Pre-compute HLA prefix cache
    hla_batch_sz = 16
    hla_prefixes = []
    with torch.no_grad():
        for i in range(0, len(unique_hlas), hla_batch_sz):
            batch = unique_hlas[i : i + hla_batch_sz]
            tok = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            x = allele_model.embed(tok["input_ids"])
            for b_idx in range(split_idx):
                x, _ = allele_model.transformer.blocks[b_idx](x, sequence_id=None)
            hla_prefixes.append(x.cpu())
    hla_prefix_cache = torch.cat(hla_prefixes, dim=0).to(device=device, dtype=torch.bfloat16)
    print(f"  -> Cached Allele prefix tensor: shape {hla_prefix_cache.shape} ({hla_prefix_cache.numel() * 2 / (1024**2):.2f} MB) in {time.time() - t_cache:.2f}s")

    # Pre-compute Peptide prefix cache
    t_pep_c = time.time()
    pep_batch_sz = 64
    pep_prefixes = []
    with torch.no_grad():
        for i in range(0, len(unique_peptides), pep_batch_sz):
            batch = unique_peptides[i : i + pep_batch_sz]
            tok = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            x = pep_model.embed(tok["input_ids"])
            for b_idx in range(split_idx):
                x, _ = pep_model.transformer.blocks[b_idx](x, sequence_id=None)
            pep_prefixes.append(x.cpu())
    pep_prefix_cache = torch.cat(pep_prefixes, dim=0).to(device=device, dtype=torch.bfloat16)
    print(f"  -> Cached Peptide prefix tensor: shape {pep_prefix_cache.shape} ({pep_prefix_cache.numel() * 2 / (1024**2):.2f} MB) in {time.time() - t_pep_c:.2f}s")

    # We only need the top blocks and norm on GPU for forward/backward
    allele_top_blocks = allele_model.transformer.blocks[split_idx:]
    allele_norm = allele_model.transformer.norm
    pep_top_blocks = pep_model.transformer.blocks[split_idx:]
    pep_norm = pep_model.transformer.norm

    # 6. Downstream Architecture: DeepExtract Interaction Network
    class ResNetBlock2D(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.act = nn.GELU()

        def forward(self, x):
            res = x
            x = self.act(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return self.act(x + res)

    class PeptideEncodingInstance(nn.Module):
        def __init__(self, d_in=2560, d_model=384, n_heads=8, dropout=0.1):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.pos_emb = nn.Parameter(torch.randn(1, 9, d_model) * 0.02)
            self.norm1 = nn.LayerNorm(d_model)
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model)
            )
            self.out_norm = nn.LayerNorm(d_model)

        def forward(self, x):
            h = self.proj(x) + self.pos_emb
            h_attn, _ = self.self_attn(h, h, h)
            h = self.norm1(h + h_attn)
            h = self.norm2(h + self.ffn(h))
            return self.out_norm(h)

    class AlleleEncodingInstance(nn.Module):
        def __init__(self, d_in=2560, d_model=384, n_heads=8, dropout=0.1):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.pos_emb = nn.Parameter(torch.randn(1, 182, d_model) * 0.02)
            self.norm1 = nn.LayerNorm(d_model)
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model)
            )
            self.out_norm = nn.LayerNorm(d_model)

        def forward(self, x):
            h = self.proj(x) + self.pos_emb
            h_attn, _ = self.self_attn(h, h, h)
            h = self.norm1(h + h_attn)
            h = self.norm2(h + self.ffn(h))
            return self.out_norm(h)

    class DeepESMceExtractLoRANet(nn.Module):
        def __init__(self, d_in=2560, d_model=384, n_heads=8, d_pair=32):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.peptide_encoder = PeptideEncodingInstance(d_in, d_model, n_heads=n_heads)
            self.allele_encoder = AlleleEncodingInstance(d_in, d_model, n_heads=n_heads)

            # Bidirectional Cross-Attention
            self.cross_p2h = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
            self.norm_p1 = nn.LayerNorm(d_model)
            self.ffn_p1 = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model)
            )
            self.norm_p2 = nn.LayerNorm(d_model)

            self.cross_h2p = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
            self.norm_h1 = nn.LayerNorm(d_model)
            self.ffn_h1 = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model)
            )
            self.norm_h2 = nn.LayerNorm(d_model)

            # 2D Pairwise Contact Track
            self.outer_p = nn.Linear(d_model, d_pair)
            self.outer_h = nn.Linear(d_model, d_pair)

            in_2d_ch = d_pair + n_heads * 3 + 1
            self.conv2d_in = nn.Sequential(
                nn.Conv2d(in_2d_ch, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU()
            )
            self.res2d_1 = ResNetBlock2D(64)
            self.conv2d_pool1 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(128),
                nn.GELU()
            )
            self.res2d_2 = ResNetBlock2D(128)
            self.conv2d_pool2 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(256),
                nn.GELU()
            )
            self.adaptive_hla_pool = nn.AdaptiveAvgPool2d((9, 1))

            # Fusion & Self-Attention
            fused_dim = d_model + 256
            self.pep_self_attn = nn.MultiheadAttention(fused_dim, 8, batch_first=True, dropout=0.1)
            self.norm_fused = nn.LayerNorm(fused_dim)

            # Readout Heads
            self.pos_weights = nn.Parameter(torch.ones(1, 9, 1) / 9.0)
            total_readout_dim = fused_dim * 4
            self.regressor = nn.Sequential(
                nn.Linear(total_readout_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
            self.classifier = nn.Sequential(
                nn.Linear(total_readout_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )

        def forward(self, pep_emb, hla_emb):
            B = pep_emb.size(0)

            # Dedicated Encoders
            p = self.peptide_encoder(pep_emb)  # [B, 9, D]
            h = self.allele_encoder(hla_emb)   # [B, 182, D]

            # Cross-Attention
            p_attn, attn_p2h_weights = self.cross_p2h(p, h, h, need_weights=True, average_attn_weights=False)
            p = self.norm_p1(p + p_attn)
            p = self.norm_p2(p + self.ffn_p1(p))

            h_attn, attn_h2p_weights = self.cross_h2p(h, p, p, need_weights=True, average_attn_weights=False)
            h = self.norm_h1(h + h_attn)
            h = self.norm_h2(h + self.ffn_h1(h))

            # Dot affinity logits
            q_p = p.view(B, 9, self.n_heads, self.d_k).permute(0, 2, 1, 3)
            k_h = h.view(B, 182, self.n_heads, self.d_k).permute(0, 2, 1, 3)
            logits_p2h = torch.matmul(q_p, k_h.transpose(-2, -1)) / (self.d_k ** 0.5)

            # 2D Pair Track
            op_p = self.outer_p(p).unsqueeze(3)
            op_h = self.outer_h(h).unsqueeze(1)
            pair_outer = (op_p * op_h.transpose(2, 3)).permute(0, 2, 1, 3)

            pep_norm_vec = F.normalize(pep_emb, p=2, dim=-1)
            hla_norm_vec = F.normalize(hla_emb, p=2, dim=-1)
            cos_sim = torch.bmm(pep_norm_vec, hla_norm_vec.transpose(1, 2)).unsqueeze(1)

            attn_h2p_t = attn_h2p_weights.transpose(2, 3)
            pair_2d = torch.cat([pair_outer, attn_p2h_weights, attn_h2p_t, logits_p2h, cos_sim], dim=1)

            feat_2d = self.conv2d_in(pair_2d)
            feat_2d = self.res2d_1(feat_2d)
            feat_2d = self.conv2d_pool1(feat_2d)
            feat_2d = self.res2d_2(feat_2d)
            feat_2d = self.conv2d_pool2(feat_2d)
            feat_2d = self.adaptive_hla_pool(feat_2d)
            feat_2d = feat_2d.squeeze(3).transpose(1, 2)

            # Fusion
            fused = torch.cat([p, feat_2d], dim=-1)
            fused_attn, _ = self.pep_self_attn(fused, fused, fused)
            fused = self.norm_fused(fused + fused_attn)

            # Readout
            weights = F.softmax(self.pos_weights, dim=1)
            pooled_w = (fused * weights).sum(dim=1)
            pooled_max = fused.max(dim=1)[0]
            p2_feat = fused[:, 1, :]
            p9_feat = fused[:, 8, :]

            readout_rep = torch.cat([pooled_w, pooled_max, p2_feat, p9_feat], dim=-1)
            pred_log = self.regressor(readout_rep).squeeze(-1)
            pred_logits = self.classifier(readout_rep).squeeze(-1)

            return pred_log, pred_logits

    # 7. Create End-to-End Model Wrapping LoRA Blocks + DeepExtract Head
    class TwoInstanceESMc6BLoRAModel(nn.Module):
        def __init__(self, allele_blocks, allele_norm, pep_blocks, pep_norm, head):
            super().__init__()
            self.allele_blocks = allele_blocks
            self.allele_norm = allele_norm
            self.pep_blocks = pep_blocks
            self.pep_norm = pep_norm
            self.head = head

        def forward(self, batch_pep_prefix, batch_hla_prefix):
            # Pass peptide prefix through top LoRA blocks
            x_pep = batch_pep_prefix
            for b in self.pep_blocks:
                x_pep, _ = b(x_pep, sequence_id=None)
            x_pep = self.pep_norm(x_pep)
            pep_tokens = x_pep[:, 1:10, :]  # [B, 9, 2560]

            # Pass HLA prefix through top LoRA blocks
            x_hla = batch_hla_prefix
            for b in self.allele_blocks:
                x_hla, _ = b(x_hla, sequence_id=None)
            x_hla = self.allele_norm(x_hla)
            hla_tokens = x_hla[:, 1:183, :]  # [B, 182, 2560]

            # Downstream DeepExtract interaction head
            pred_log, pred_logits = self.head(pep_tokens, hla_tokens)
            return pred_log, pred_logits

    deep_extract_head = DeepESMceExtractLoRANet(d_in=2560, d_model=d_model, n_heads=n_heads, d_pair=d_pair).to(device=device, dtype=torch.bfloat16)
    full_model = TwoInstanceESMc6BLoRAModel(
        allele_top_blocks, allele_norm, pep_top_blocks, pep_norm, deep_extract_head
    ).to(device=device)

    # Differential Learning Rates
    param_groups = [
        {"params": allele_lora_params + pep_lora_params, "lr": lr_lora, "weight_decay": 1e-4},
        {"params": deep_extract_head.parameters(), "lr": lr_head, "weight_decay": weight_decay},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Multi-task Loss
    def compute_loss(pred_log, pred_logits, target_log, target_clf, eps=1e-8):
        loss_reg = F.smooth_l1_loss(pred_log, target_log)
        p_c = pred_log - pred_log.mean()
        t_c = target_log - target_log.mean()
        cov = (p_c * t_c).sum()
        p_std = torch.sqrt((p_c ** 2).sum() + eps)
        t_std = torch.sqrt((t_c ** 2).sum() + eps)
        r = cov / (p_std * t_std + eps)
        loss_pearson = 1.0 - torch.clamp(r, -1.0, 1.0)

        # Ranking loss on random pairs
        B_curr = pred_log.size(0)
        idx1 = torch.randperm(B_curr, device=device)
        idx2 = torch.randperm(B_curr, device=device)
        diff_pred = pred_log[idx1] - pred_log[idx2]
        diff_true = target_log[idx1] - target_log[idx2]
        target_sign = torch.sign(diff_true)
        valid = (torch.abs(diff_true) > 0.05).float()
        loss_rank = (F.relu(0.1 - target_sign * diff_pred) * valid).sum() / (valid.sum() + eps)

        if is_binding_mode:
            # Within single-regime binding splits, classification labels are homogeneous
            loss_total = loss_reg + 0.6 * loss_pearson + 0.2 * loss_rank
        else:
            loss_clf = F.binary_cross_entropy_with_logits(pred_logits, target_clf)
            loss_total = loss_reg + 0.6 * loss_pearson + 0.2 * loss_rank + 0.3 * loss_clf

        return loss_total, r.item(), loss_reg.item()

    # 8. Data Preparation (Map each sample to prefix index)
    def prep_tensors(df):
        p_idxs = torch.tensor([pep_to_idx[p] for p in df["peptide"]], dtype=torch.long)
        h_idxs = torch.tensor([hla_to_idx[h] for h in df["hla_seq"]], dtype=torch.long)
        y_raw = df["thalf_hours"].values.astype(np.float32)
        y_log = np.log10(1.0 + np.clip(y_raw, 0, None))
        y_clf = (y_raw >= 1.0).astype(np.float32)
        return (
            p_idxs,
            h_idxs,
            torch.tensor(y_log, dtype=torch.bfloat16),
            torch.tensor(y_clf, dtype=torch.bfloat16),
            torch.tensor(y_raw, dtype=torch.float32),
        )

    tr_p, tr_h, tr_ylog, tr_yclf, tr_yraw = prep_tensors(df_train)
    va_p, va_h, va_ylog, va_yclf, va_yraw = prep_tensors(df_val)
    te_p, te_h, te_ylog, te_yclf, te_yraw = prep_tensors(df_test)

    train_loader = DataLoader(TensorDataset(tr_p, tr_h, tr_ylog, tr_yclf, tr_yraw), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(va_p, va_h, va_ylog, va_yclf, va_yraw), batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(TensorDataset(te_p, te_h, te_ylog, te_yclf, te_yraw), batch_size=batch_size * 2, shuffle=False)

    # 9. Training Loop with Epoch History
    print(f"\n[Step 5/5] Training 2-Instance ESMc-6B LoRA for {epochs} epochs...")
    print("-" * 84)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val MSE':<10} | {'Pearson r':<12} | {'Spearman ρ':<12} | {'AUROC':<10}")
    print("-" * 84)

    best_val_r = -999.0
    best_weights = None
    train_history = []

    for epoch in range(1, epochs + 1):
        full_model.train()
        train_losses = []
        train_rs = []

        for p_idx, h_idx, y_log, y_clf, _ in train_loader:
            p_idx, h_idx = p_idx.to(device), h_idx.to(device)
            y_log, y_clf = y_log.to(device), y_clf.to(device)

            # Retrieve cached prefix tokens for the batch
            batch_pep_prefix = pep_prefix_cache[p_idx]
            batch_hla_prefix = hla_prefix_cache[h_idx]

            optimizer.zero_grad()
            pred_log, pred_logits = full_model(batch_pep_prefix, batch_hla_prefix)
            loss, r_val, _ = compute_loss(pred_log, pred_logits, y_log, y_clf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_rs.append(r_val)

        scheduler.step()

        # Validation Step
        full_model.eval()
        val_preds_log, val_preds_raw, val_preds_clf = [], [], []
        val_targets_log, val_targets_raw, val_targets_clf = [], [], []

        with torch.no_grad():
            for p_idx, h_idx, y_log, y_clf, y_raw in val_loader:
                p_idx, h_idx = p_idx.to(device), h_idx.to(device)
                batch_pep_prefix = pep_prefix_cache[p_idx]
                batch_hla_prefix = hla_prefix_cache[h_idx]

                pred_log, pred_logits = full_model(batch_pep_prefix, batch_hla_prefix)
                p_log = pred_log.to(torch.float32).cpu().numpy()
                p_raw = np.power(10.0, p_log) - 1.0
                p_raw = np.clip(p_raw, 0, None)
                p_prob = torch.sigmoid(pred_logits.to(torch.float32)).cpu().numpy()

                val_preds_log.extend(p_log)
                val_preds_raw.extend(p_raw)
                val_preds_clf.extend(p_prob)
                val_targets_log.extend(y_log.to(torch.float32).cpu().numpy())
                val_targets_raw.extend(y_raw.numpy())
                val_targets_clf.extend(y_clf.to(torch.float32).cpu().numpy())

        val_preds_log = np.array(val_preds_log)
        val_targets_log = np.array(val_targets_log)
        val_targets_clf = np.array(val_targets_clf)
        val_preds_clf = np.array(val_preds_clf)

        val_mse = mean_squared_error(val_targets_log, val_preds_log)
        val_r, _ = pearsonr(val_targets_log, val_preds_log)
        val_rho, _ = spearmanr(val_targets_log, val_preds_log)

        if not is_binding_mode and len(np.unique(val_targets_clf)) > 1:
            val_auroc = roc_auc_score(val_targets_clf, val_preds_clf)
        else:
            val_auroc = 0.5

        train_history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_pearson": float(np.mean(train_rs)),
            "val_mse": float(val_mse),
            "val_pearson": float(val_r),
            "val_spearman": float(val_rho),
            "val_auroc": float(val_auroc),
        })

        auroc_str = f"{val_auroc:.4f}" if val_auroc > 0.5 else "N/A"
        print(f"{epoch:<6} | {np.mean(train_losses):<12.4f} | {val_mse:<10.4f} | {val_r:<12.4f} | {val_rho:<12.4f} | {auroc_str:<10}")

        if val_r > best_val_r:
            best_val_r = val_r
            best_weights = {
                "head": {k: v.cpu().clone() for k, v in deep_extract_head.state_dict().items()},
                "allele_lora": [p.data.cpu().clone() for p in allele_lora_params],
                "pep_lora": [p.data.cpu().clone() for p in pep_lora_params],
            }

    print("-" * 84)
    print(f"  Training finished in {time.time() - t_start_total:.2f}s | Best Val Pearson r: {best_val_r:.4f}")

    # Restore best weights
    deep_extract_head.load_state_dict({k: v.to(device) for k, v in best_weights["head"].items()})
    for p_dest, p_src in zip(allele_lora_params, best_weights["allele_lora"]):
        p_dest.data.copy_(p_src.to(device))
    for p_dest, p_src in zip(pep_lora_params, best_weights["pep_lora"]):
        p_dest.data.copy_(p_src.to(device))

    # 10. Test Evaluation Function
    def evaluate_dataset(loader, df_eval, desc="Test Set"):
        full_model.eval()
        preds_log, preds_raw, preds_clf = [], [], []
        targets_log, targets_raw, targets_clf = [], [], []

        with torch.no_grad():
            for p_idx, h_idx, y_log, y_clf, y_raw in loader:
                p_idx, h_idx = p_idx.to(device), h_idx.to(device)
                batch_pep_prefix = pep_prefix_cache[p_idx]
                batch_hla_prefix = hla_prefix_cache[h_idx]

                pred_log, pred_logits = full_model(batch_pep_prefix, batch_hla_prefix)
                p_l = pred_log.to(torch.float32).cpu().numpy()
                p_r = np.power(10.0, p_l) - 1.0
                p_r = np.clip(p_r, 0, None)
                p_c = torch.sigmoid(pred_logits.to(torch.float32)).cpu().numpy()

                preds_log.extend(p_l)
                preds_raw.extend(p_r)
                preds_clf.extend(p_c)
                targets_log.extend(y_log.to(torch.float32).cpu().numpy())
                targets_raw.extend(y_raw.numpy())
                targets_clf.extend(y_clf.to(torch.float32).cpu().numpy())

        preds_log = np.array(preds_log)
        targets_log = np.array(targets_log)
        preds_raw = np.array(preds_raw)
        targets_raw = np.array(targets_raw)
        preds_clf = np.array(preds_clf)
        targets_clf = np.array(targets_clf)

        r_log, _ = pearsonr(targets_log, preds_log)
        rho, _ = spearmanr(targets_log, preds_log)
        r_raw, _ = pearsonr(targets_raw, preds_raw)
        mse_log = mean_squared_error(targets_log, preds_log)
        rmse_log = np.sqrt(mse_log)
        mae_raw = mean_absolute_error(targets_raw, preds_raw)
        rmse_raw = np.sqrt(mean_squared_error(targets_raw, preds_raw))

        metrics = {
            "pearson_r": float(r_log),
            "spearman_rho": float(rho),
            "pearson_r_raw": float(r_raw),
            "rmse": float(rmse_log),
            "mse": float(mse_log),
            "mae_raw": float(mae_raw),
            "rmse_raw": float(rmse_raw),
            "mean_pred_thalf": float(np.mean(preds_raw)),
            "mean_true_thalf": float(np.mean(targets_raw)),
        }

        if len(np.unique(targets_clf)) > 1:
            metrics["auroc_1h"] = float(roc_auc_score(targets_clf, preds_clf))
            metrics["aupr_1h"] = float(average_precision_score(targets_clf, preds_clf))
            t_2h = (targets_raw >= 2.0).astype(float)
            if len(np.unique(t_2h)) > 1:
                metrics["auroc_2h"] = float(roc_auc_score(t_2h, preds_log))
                metrics["aupr_2h"] = float(average_precision_score(t_2h, preds_log))

        print(f"\nResults for {desc}:")
        print(f"  Pearson r (log10):  {r_log:.4f}")
        print(f"  Spearman rho:       {rho:.4f}")
        print(f"  Pearson r (raw):    {r_raw:.4f}")
        print(f"  RMSE (log10):       {rmse_log:.4f}")
        print(f"  MAE (raw thalf h):  {mae_raw:.4f}h")
        if "auroc_1h" in metrics:
            print(f"  AUROC (>=1h):       {metrics['auroc_1h']:.4f}")
            print(f"  AUPR (>=1h):        {metrics['aupr_1h']:.4f}")

        df_out = df_eval.copy()
        df_out["pred_log_thalf"] = preds_log
        df_out["pred_thalf_hours"] = preds_raw
        df_out["pred_prob_binder_1h"] = preds_clf
        return metrics, df_out

    # 11. Evaluate on Test Set
    test_metrics, df_test_preds = evaluate_dataset(test_loader, df_test, f"{split_name.upper()} Test Set")

    # Cross-evaluation for binding modes
    cross_metrics = {}
    if is_binding_mode:
        for other_mode, other_df in cross_test_dfs.items():
            o_p, o_h, o_ylog, o_yclf, o_yraw = prep_tensors(other_df)
            o_loader = DataLoader(TensorDataset(o_p, o_h, o_ylog, o_yclf, o_yraw), batch_size=batch_size * 2, shuffle=False)
            m_cross, _ = evaluate_dataset(o_loader, other_df, f"Cross-Eval: {other_mode.upper()} Test Set")
            cross_metrics[other_mode] = m_cross

    # 12. Save Results to HuggingFace Volume & Return Dict
    out_dir = Path("/root/.cache/huggingface")
    preds_file = out_dir / f"esmc_lora_2inst_{split_name}_preds.csv"
    metrics_file = out_dir / f"esmc_lora_2inst_{split_name}_metrics.json"

    result_payload = {
        "split_name": split_name,
        "model_architecture": "DeepExtract 2-Instance ESMc-6B with LoRA Fine-Tuning",
        "backbone": "biohub/ESMC-6B",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "num_lora_layers": num_lora_layers,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
        "total_time_seconds": time.time() - t_start_total,
        "test_metrics": test_metrics,
        "cross_eval_metrics": cross_metrics,
        "train_history": train_history,
        "predictions": df_test_preds.to_dict(orient="records"),
    }

    df_test_preds.to_csv(preds_file, index=False)
    with open(metrics_file, "w") as f:
        json.dump(result_payload, f, indent=2)

    hf_volume.commit()
    print(f"\n[Saved] Persisted {preds_file.name} and {metrics_file.name} to Modal Volume.")
    return result_payload


# -----------------------------------------------------------------------------
# LOCAL ENTRYPOINT
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main(split: str = "iid"):
    splits_to_run = [split] if split != "all" else ["iid", "novel_pep", "novel_allele", "non_binding", "mild_binding", "strong_binding"]
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    for s in splits_to_run:
        print(f"\n>>> Dispatching Modal Job for Split: {s.upper()} <<<")
        res = run_lora_2inst_experiment.remote(split_name=s)
        
        # Save local predictions
        preds_list = res.pop("predictions", None)
        if preds_list:
            df_preds = pd.DataFrame(preds_list)
            local_preds_path = results_dir / f"esmc_lora_2inst_{s}_preds.csv"
            df_preds.to_csv(local_preds_path, index=False)
            print(f"[Local Save] Saved predictions to {local_preds_path}")

        # Save local metrics
        local_metrics_path = results_dir / f"esmc_lora_2inst_{s}_metrics.json"
        with open(local_metrics_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[Local Save] Saved metrics to {local_metrics_path}")
        print(f"Test Pearson r: {res['test_metrics']['pearson_r']:.4f} | Spearman: {res['test_metrics']['spearman_rho']:.4f}")
