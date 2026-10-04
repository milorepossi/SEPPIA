"""Modal App for Deep ESMC-6B Extraction & Training Dynamics Optimization on IID Split.

Architectural and Optimization Improvements:
1. Bidirectional residue-level cross-attention (Peptide queries HLA groove, HLA groove queries Peptide).
2. 2D pairwise interaction contact track (Bilinear outer-product + multi-head attention maps + dot-product logits + cosine similarity).
3. 2D ResNet structural feature extractor over the 9x182 contact space.
4. Peptide self-attention across the 9 residues with anchor positional encodings.
5. Anchor-aware readout preserving explicit B-pocket (P2) and F-pocket (P9) anchor representations.
6. Multi-objective loss directly optimizing Pearson correlation and Spearman pairwise ranking:
   L = L_SmoothL1 + 0.6 * L_Pearson + 0.2 * L_Rank + 0.3 * L_BCE
7. 35 epochs with 3-epoch linear warmup + cosine decay.
"""

import os
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandas as pd
import modal

# -----------------------------------------------------------------------------
# 1. MODAL APP & CONTAINER DEFINITION
# -----------------------------------------------------------------------------
app = modal.App("esmc-deep-extract-iid")

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
# 2. MODAL REMOTE TRAINING FUNCTION (NVIDIA A100-40GB)
# -----------------------------------------------------------------------------
@app.function(
    gpu="A100-40GB",
    image=esmc_image,
    volumes={"/root/.cache/huggingface": hf_volume},
    timeout=2400,
)
def run_deep_extract_experiment(
    model_name: str = "biohub/ESMC-6B",
    split_name: str = "iid",
    epochs: int = 35,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    d_model: int = 384,
    n_heads: int = 8,
    d_pair: int = 32,
):
    import time
    import gc
    import json
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
    from esm.models.esmc import EsmcModel, EsmcTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0

    print("=" * 78)
    print(f"ESMC-6B DEEP EXTRACTION & TRAINING DYNAMICS OPTIMIZATION ON {gpu_name} ({split_name.upper()} SPLIT)")
    print("=" * 78)
    print(f"Device: {device} | VRAM: {total_mem_gb:.2f} GB | PyTorch: {torch.__version__}")
    print(f"Hyperparameters: split={split_name}, epochs={epochs}, batch_size={batch_size}, lr={lr}, d_model={d_model}, heads={n_heads}")
    print("=" * 78)

    # 1. Load Data
    splits_dir = Path("/root/splits")
    df_train = pd.read_csv(splits_dir / f"{split_name}_train.csv")
    df_val = pd.read_csv(splits_dir / f"{split_name}_val.csv")
    df_test = pd.read_csv(splits_dir / f"{split_name}_test.csv")

    print(f"\n[Step 1/5] Loaded {split_name.upper()} Dataset:")
    print(f"  Train: {len(df_train):,} pairs ({df_train['peptide'].nunique()} peptides, {df_train['allele'].nunique()} alleles)")
    print(f"  Val:   {len(df_val):,} pairs ({df_val['peptide'].nunique()} peptides, {df_val['allele'].nunique()} alleles)")
    print(f"  Test:  {len(df_test):,} pairs ({df_test['peptide'].nunique()} peptides, {df_test['allele'].nunique()} alleles)")

    # 2. Extract Token-Level Residue Embeddings using 2 Distinct ESMc Foundation Instances
    print(f"\n[Step 2/5] Initializing Instance 1 (Allele Foundation Instance) from '{model_name}'...")
    t0 = time.time()
    tokenizer = EsmcTokenizer.from_pretrained(model_name)
    allele_esmc = EsmcModel.from_pretrained(model_name, dtype=torch.bfloat16, device="cuda").eval()
    d_in = allele_esmc.config.d_model if hasattr(allele_esmc.config, "d_model") else 2560
    param_count = sum(p.numel() for p in allele_esmc.parameters())
    print(f"  Allele Instance loaded in {time.time() - t0:.2f}s | Parameters: {param_count:,} | d_in: {d_in}")
    print(f"  GPU VRAM allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    all_peptides = sorted(list(set(df_train["peptide"]).union(set(df_val["peptide"])).union(set(df_test["peptide"]))))
    all_hlas = sorted(list(set(df_train["hla_seq"]).union(set(df_val["hla_seq"])).union(set(df_test["hla_seq"]))))
    print(f"  Unique peptides: {len(all_peptides):,} (9 residues each)")
    print(f"  Unique HLAs:     {len(all_hlas):,} (182 residues each)")

    pep_to_idx = {p: i for i, p in enumerate(all_peptides)}
    hla_to_idx = {h: i for i, h in enumerate(all_hlas)}

    # Batch extraction for HLAs using Allele Instance (182 residues -> token slice 1:183)
    hla_tokens_list = []
    hla_batch_sz = 16
    t_start = time.time()
    with torch.no_grad():
        for i in range(0, len(all_hlas), hla_batch_sz):
            batch = all_hlas[i : i + hla_batch_sz]
            tok = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            out = allele_esmc(**tok)
            res_tokens = out.last_hidden_state[:, 1:183, :].to(dtype=torch.float32).cpu()
            hla_tokens_list.append(res_tokens)
    hla_tokens_tensor = torch.cat(hla_tokens_list, dim=0) # (num_hlas, 182, d_in)
    print(f"  -> Extracted HLA token tensors in {time.time() - t_start:.2f}s: shape {hla_tokens_tensor.shape}")

    # Free Allele Instance from GPU memory
    del allele_esmc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Allele Instance unloaded. Free GPU VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB used.")

    # Initialize Instance 2 (Peptide Foundation Instance)
    print(f"\n[Step 3/5] Initializing Instance 2 (Peptide Foundation Instance) from '{model_name}'...")
    t0 = time.time()
    pep_esmc = EsmcModel.from_pretrained(model_name, dtype=torch.bfloat16, device="cuda").eval()
    print(f"  Peptide Instance loaded in {time.time() - t0:.2f}s | Parameters: {sum(p.numel() for p in pep_esmc.parameters()):,}")

    # Batch extraction for Peptides using Peptide Instance (9 residues -> token slice 1:10)
    pep_tokens_list = []
    pep_batch_sz = 64
    t_start = time.time()
    with torch.no_grad():
        for i in range(0, len(all_peptides), pep_batch_sz):
            batch = all_peptides[i : i + pep_batch_sz]
            tok = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            out = pep_esmc(**tok)
            res_tokens = out.last_hidden_state[:, 1:10, :].to(dtype=torch.float32).cpu()
            pep_tokens_list.append(res_tokens)
    pep_tokens_tensor = torch.cat(pep_tokens_list, dim=0) # (num_peptides, 9, d_in)
    print(f"  -> Extracted Peptide token tensors in {time.time() - t_start:.2f}s: shape {pep_tokens_tensor.shape}")

    # Free Peptide Instance from GPU memory
    del pep_esmc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Peptide Instance unloaded. Free GPU VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB used.")

    # Cache token tensors in GPU VRAM
    hla_tokens_gpu = hla_tokens_tensor.to(device)
    pep_tokens_gpu = pep_tokens_tensor.to(device)
    print(f"  Cached all token tensors in GPU VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB used.")

    # Helper to prepare index datasets
    def make_dataset(df):
        p_indices = torch.tensor([pep_to_idx[p] for p in df["peptide"]], dtype=torch.long)
        h_indices = torch.tensor([hla_to_idx[h] for h in df["hla_seq"]], dtype=torch.long)
        y_raw = df["thalf_hours"].values.astype(np.float32)
        y_log = np.log10(1.0 + np.clip(y_raw, 0, None))
        y_clf = (y_raw >= 1.0).astype(np.float32)
        return TensorDataset(
            p_indices,
            h_indices,
            torch.tensor(y_log, dtype=torch.float32),
            torch.tensor(y_clf, dtype=torch.float32),
            torch.tensor(y_raw, dtype=torch.float32),
        )

    train_ds = make_dataset(df_train)
    val_ds = make_dataset(df_val)
    test_ds = make_dataset(df_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # -------------------------------------------------------------------------
    # 4. ARCHITECTURE DEFINITION
    # -------------------------------------------------------------------------
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
        """
        Dedicated Instance 1: Peptide Residue Encoder
        Specializes in 9-mer epitope sequence representation, local residue context,
        and anchor pocket positioning (P1-P9).
        """
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
        """
        Dedicated Instance 2: Allele Groove Residue Encoder
        Specializes in 182-residue HLA alpha1/alpha2 cleft groove representation,
        groove topology, and spatial cleft context.
        """
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

    class DeepESMceExtractNet(nn.Module):
        def __init__(self, d_in=2560, d_model=384, n_heads=8, d_pair=32):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            # 1. Two Dedicated Encoding Instances (Peptide != Allele)
            self.peptide_encoder = PeptideEncodingInstance(d_in, d_model, n_heads=n_heads)
            self.allele_encoder = AlleleEncodingInstance(d_in, d_model, n_heads=n_heads)

            # 2. Bidirectional Cross-Attention
            # Peptide queries HLA (Groove to Peptide context)
            self.cross_p2h = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
            self.norm_p1 = nn.LayerNorm(d_model)
            self.ffn_p1 = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model)
            )
            self.norm_p2 = nn.LayerNorm(d_model)

            # HLA queries Peptide (Peptide to Groove context)
            self.cross_h2p = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
            self.norm_h1 = nn.LayerNorm(d_model)
            self.ffn_h1 = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model)
            )
            self.norm_h2 = nn.LayerNorm(d_model)

            # 3. 2D Pairwise Contact Track (Outer product + Attention maps + Dot-product logits + Cosine similarity)
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

            # 4. Fusion & Peptide Self-Attention
            fused_dim = d_model + 256
            self.pep_self_attn = nn.MultiheadAttention(fused_dim, 8, batch_first=True, dropout=0.1)
            self.norm_fused = nn.LayerNorm(fused_dim)

            # 5. Position-weighted & Anchor Readout
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

            # 1. Two Dedicated Encoding Instances (Peptide != Allele)
            p = self.peptide_encoder(pep_emb)  # [B, 9, D]
            h = self.allele_encoder(hla_emb)   # [B, 182, D]

            # 2. Bidirectional Cross-Attention
            # Pep queries HLA
            p_attn, attn_p2h_weights = self.cross_p2h(p, h, h, need_weights=True, average_attn_weights=False) # [B, H, 9, 182]
            p = self.norm_p1(p + p_attn)
            p = self.norm_p2(p + self.ffn_p1(p))

            # HLA queries Pep
            h_attn, attn_h2p_weights = self.cross_h2p(h, p, p, need_weights=True, average_attn_weights=False) # [B, H, 182, 9]
            h = self.norm_h1(h + h_attn)
            h = self.norm_h2(h + self.ffn_h1(h))

            # Dot-product interaction affinity logits
            q_p = p.view(B, 9, self.n_heads, self.d_k).permute(0, 2, 1, 3)
            k_h = h.view(B, 182, self.n_heads, self.d_k).permute(0, 2, 1, 3)
            logits_p2h = torch.matmul(q_p, k_h.transpose(-2, -1)) / (self.d_k ** 0.5) # [B, H, 9, 182]

            # 3. 2D Pair Track Construction
            # Bilinear outer product
            op_p = self.outer_p(p).unsqueeze(3) # [B, 9, d_pair, 1]
            op_h = self.outer_h(h).unsqueeze(1) # [B, 1, 182, d_pair]
            pair_outer = (op_p * op_h.transpose(2, 3)).permute(0, 2, 1, 3) # [B, d_pair, 9, 182]

            # Cosine similarity matrix between raw foundation embeddings
            pep_norm = F.normalize(pep_emb, p=2, dim=-1)
            hla_norm = F.normalize(hla_emb, p=2, dim=-1)
            cos_sim = torch.bmm(pep_norm, hla_norm.transpose(1, 2)).unsqueeze(1) # [B, 1, 9, 182]

            attn_h2p_t = attn_h2p_weights.transpose(2, 3) # [B, H, 9, 182]

            pair_2d = torch.cat([pair_outer, attn_p2h_weights, attn_h2p_t, logits_p2h, cos_sim], dim=1)

            # 2D ResNet
            feat_2d = self.conv2d_in(pair_2d)
            feat_2d = self.res2d_1(feat_2d)
            feat_2d = self.conv2d_pool1(feat_2d)
            feat_2d = self.res2d_2(feat_2d)
            feat_2d = self.conv2d_pool2(feat_2d)
            feat_2d = self.adaptive_hla_pool(feat_2d) # [B, 256, 9, 1]
            feat_2d = feat_2d.squeeze(3).transpose(1, 2) # [B, 9, 256]

            # 4. Fusion & Peptide Self-Attention
            fused = torch.cat([p, feat_2d], dim=-1) # [B, 9, fused_dim]
            fused_attn, _ = self.pep_self_attn(fused, fused, fused)
            fused = self.norm_fused(fused + fused_attn) # [B, 9, fused_dim]

            # 5. Readout Head
            weights = F.softmax(self.pos_weights, dim=1)
            pooled_w = (fused * weights).sum(dim=1) # [B, fused_dim]
            pooled_max = fused.max(dim=1)[0]        # [B, fused_dim]
            p2_feat = fused[:, 1, :]                # [B, fused_dim] - Anchor P2 (B-pocket)
            p9_feat = fused[:, 8, :]                # [B, fused_dim] - Anchor P9 (F-pocket)

            readout_rep = torch.cat([pooled_w, pooled_max, p2_feat, p9_feat], dim=-1)

            pred_log = self.regressor(readout_rep).squeeze(-1)
            pred_logits = self.classifier(readout_rep).squeeze(-1)

            return pred_log, pred_logits, attn_p2h_weights

    # Multi-task loss function
    def compute_loss(pred_log, pred_logits, target_log, target_clf, eps=1e-8):
        # 1. Smooth L1 (Scale calibration)
        loss_reg = F.smooth_l1_loss(pred_log, target_log)

        # 2. Pearson Correlation Loss
        p_c = pred_log - pred_log.mean()
        t_c = target_log - target_log.mean()
        cov = (p_c * t_c).sum()
        p_std = torch.sqrt((p_c ** 2).sum() + eps)
        t_std = torch.sqrt((t_c ** 2).sum() + eps)
        r = cov / (p_std * t_std + eps)
        loss_pearson = 1.0 - torch.clamp(r, -1.0, 1.0)

        # 3. Soft Margin Pairwise Ranking Loss
        t_diff = target_log.unsqueeze(1) - target_log.unsqueeze(0)
        p_diff = pred_log.unsqueeze(1) - pred_log.unsqueeze(0)
        margin = 0.05
        valid_pairs = (t_diff > margin).float()
        if valid_pairs.sum() > 0:
            loss_rank = (torch.relu(margin - p_diff) * valid_pairs).sum() / (valid_pairs.sum() + eps)
        else:
            loss_rank = torch.tensor(0.0, device=pred_log.device)

        # 4. BCE Classification Loss
        loss_clf = F.binary_cross_entropy_with_logits(pred_logits, target_clf)

        total_loss = loss_reg + 0.6 * loss_pearson + 0.2 * loss_rank + 0.3 * loss_clf
        return total_loss, loss_reg, loss_pearson, loss_rank, loss_clf

    # Initialize model
    model = DeepESMceExtractNet(d_in=d_in, d_model=d_model, n_heads=n_heads, d_pair=d_pair).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Step 4/5] Two-Instance DeepESMceExtractNet initialized with {n_params:,} parameters.")
    print(f"  Instance 1 (Peptide): PeptideEncodingInstance (9-mer dedicated transformer encoder tower)")
    print(f"  Instance 2 (Allele):  AlleleEncodingInstance (182-residue cleft dedicated transformer encoder tower)")
    print(f"  Contact Track: 2D Pairwise ResNet (9x182) + Bidirectional Cross-Attention + Anchor Heads")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * 3

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-6 / lr, 0.5 * (1.0 + np.cos(np.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Evaluation routine
    def evaluate(loader):
        model.eval()
        preds_log = []
        preds_clf = []
        targets_log = []
        targets_clf = []
        targets_raw = []

        with torch.no_grad():
            for p_idx, h_idx, t_log, t_clf, t_raw in loader:
                p_emb = pep_tokens_gpu[p_idx]
                h_emb = hla_tokens_gpu[h_idx]
                p_log, p_logits, _ = model(p_emb, h_emb)

                preds_log.extend(p_log.cpu().numpy())
                preds_clf.extend(torch.sigmoid(p_logits).cpu().numpy())
                targets_log.extend(t_log.cpu().numpy())
                targets_clf.extend(t_clf.cpu().numpy())
                targets_raw.extend(t_raw.cpu().numpy())

        preds_log = np.array(preds_log)
        preds_clf = np.array(preds_clf)
        targets_log = np.array(targets_log)
        targets_clf = np.array(targets_clf)
        targets_raw = np.array(targets_raw)

        preds_raw = np.clip(10.0 ** preds_log - 1.0, 0, None)

        p_r_log, _ = pearsonr(preds_log, targets_log)
        s_rho, _ = spearmanr(preds_log, targets_log)
        p_r_raw, _ = pearsonr(preds_raw, targets_raw)
        mse_log = float(np.mean((preds_log - targets_log) ** 2))
        rmse_log = float(np.sqrt(mse_log))
        mae_raw = float(np.mean(np.abs(preds_raw - targets_raw)))
        rmse_raw = float(np.sqrt(np.mean((preds_raw - targets_raw) ** 2)))

        auc_1h = float(roc_auc_score(targets_clf, preds_clf))
        pr_auc_1h = float(average_precision_score(targets_clf, preds_clf))

        targets_2h = (targets_raw >= 2.0).astype(int)
        auc_2h = float(roc_auc_score(targets_2h, preds_log))
        pr_auc_2h = float(average_precision_score(targets_2h, preds_log)) if len(np.unique(targets_2h)) > 1 else 0.5

        return {
            "pearson_r_log": float(p_r_log),
            "spearman_rho": float(s_rho),
            "pearson_r_raw": float(p_r_raw),
            "mse_log": mse_log,
            "rmse_log": rmse_log,
            "rmse_raw": rmse_raw,
            "mae_raw": mae_raw,
            "auc_roc_1h": auc_1h,
            "pr_auc_1h": pr_auc_1h,
            "auc_roc_2h": auc_2h,
            "pr_auc_2h": pr_auc_2h,
            "preds_log": preds_log,
            "preds_clf": preds_clf,
            "preds_raw": preds_raw,
        }

    # Training Loop
    step_history = []
    epoch_history = []
    best_val_r = -1.0
    best_model_state = None
    global_step = 0

    print("\n" + "=" * 78)
    print(f"[Step 5/5] Training DeepESMceExtractNet for {epochs} Epochs...")
    print("=" * 78)
    train_start_t = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tot_epoch_loss = 0.0
        reg_epoch_loss = 0.0
        pear_epoch_loss = 0.0
        rank_epoch_loss = 0.0
        clf_epoch_loss = 0.0

        for p_idx, h_idx, t_log, t_clf, _ in train_loader:
            optimizer.zero_grad()
            p_emb = pep_tokens_gpu[p_idx]
            h_emb = hla_tokens_gpu[h_idx]
            t_log_dev = t_log.to(device)
            t_clf_dev = t_clf.to(device)

            p_log, p_logits, _ = model(p_emb, h_emb)
            tot, reg, pear, rank, clf = compute_loss(p_log, p_logits, t_log_dev, t_clf_dev)
            tot.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            tot_epoch_loss += tot.item()
            reg_epoch_loss += reg.item()
            pear_epoch_loss += pear.item()
            rank_epoch_loss += rank.item()
            clf_epoch_loss += clf.item()

            if global_step % 25 == 0 or global_step == 1:
                step_history.append({
                    "step": global_step,
                    "epoch": epoch,
                    "total_loss": float(tot.item()),
                    "reg_loss": float(reg.item()),
                    "pearson_loss": float(pear.item()),
                    "rank_loss": float(rank.item()),
                    "clf_loss": float(clf.item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                })

        n_batches = len(train_loader)
        train_tot = tot_epoch_loss / n_batches
        train_reg = reg_epoch_loss / n_batches
        train_pear = pear_epoch_loss / n_batches
        train_rank = rank_epoch_loss / n_batches
        train_clf = clf_epoch_loss / n_batches

        # Validation
        val_res = evaluate(val_loader)
        is_best = val_res["pearson_r_log"] > best_val_r
        if is_best:
            best_val_r = val_res["pearson_r_log"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_tot,
            "train_reg_loss": train_reg,
            "train_pearson_loss": train_pear,
            "train_rank_loss": train_rank,
            "train_clf_loss": train_clf,
            "val_pearson_r_log": val_res["pearson_r_log"],
            "val_spearman_rho": val_res["spearman_rho"],
            "val_mse_log": val_res["mse_log"],
            "val_auc_roc_1h": val_res["auc_roc_1h"],
            "val_pr_auc_1h": val_res["pr_auc_1h"],
            "is_best": is_best,
        })

        star = " ★ BEST" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_tot:.4f} (Reg: {train_reg:.4f}, Pear: {train_pear:.4f}, Rank: {train_rank:.4f}, Clf: {train_clf:.4f}) | "
            f"Val r: {val_res['pearson_r_log']:.4f} | "
            f"Val ρ: {val_res['spearman_rho']:.4f} | "
            f"Val AUC: {val_res['auc_roc_1h']:.4f}{star}"
        )

    print(f"\nTraining completed in {time.time() - train_start_t:.2f}s!")

    # Test Evaluation with Best Checkpoint
    print(f"\nLoading best checkpoint (Val Pearson r = {best_val_r:.4f}) for Final Test Evaluation...")
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    test_res = evaluate(test_loader)

    print("\n" + "=" * 78)
    print("FINAL TEST EVALUATION RESULTS ON IID SPLIT (4,225 PAIRS):")
    print("=" * 78)
    print(f"  Pearson r (log10 stability):   {test_res['pearson_r_log']:.4f}")
    print(f"  Spearman rho (rank stability): {test_res['spearman_rho']:.4f}")
    print(f"  Pearson r (raw hours):         {test_res['pearson_r_raw']:.4f}")
    print(f"  Log RMSE:                      {test_res['rmse_log']:.4f}")
    print(f"  Raw MAE:                       {test_res['mae_raw']:.2f} h")
    print(f"  Raw RMSE:                      {test_res['rmse_raw']:.2f} h")
    print(f"  AUROC (>= 1.0h binder):        {test_res['auc_roc_1h']:.4f}")
    print(f"  PR-AUC (>= 1.0h binder):       {test_res['pr_auc_1h']:.4f}")
    print(f"  AUROC (>= 2.0h strong binder): {test_res['auc_roc_2h']:.4f}")
    print(f"  PR-AUC (>= 2.0h strong binder):{test_res['pr_auc_2h']:.4f}")
    print("=" * 78)

    # Sample cross-attention map
    model.eval()
    with torch.no_grad():
        p_sample = pep_tokens_gpu[0:1]
        h_sample = hla_tokens_gpu[0:1]
        _, _, sample_attn = model(p_sample, h_sample)
        sample_attn_map = sample_attn[0].mean(dim=0).cpu().numpy().tolist()

    df_test_out = df_test.copy()
    df_test_out["pred_log_thalf"] = test_res["preds_log"]
    df_test_out["pred_thalf_hours"] = test_res["preds_raw"]
    df_test_out["pred_prob_binding"] = test_res["preds_clf"]

    results_payload = {
        "model_name": "ESMC-6B-DeepExtract-2Instances",
        "split": split_name,
        "test_metrics": {
            "pearson_r_log": test_res["pearson_r_log"],
            "spearman_rho": test_res["spearman_rho"],
            "pearson_r_raw": test_res["pearson_r_raw"],
            "rmse_log": test_res["rmse_log"],
            "mse_log": test_res["mse_log"],
            "mae_raw": test_res["mae_raw"],
            "rmse_raw": test_res["rmse_raw"],
            "auc_roc_1h": test_res["auc_roc_1h"],
            "pr_auc_1h": test_res["pr_auc_1h"],
            "auc_roc_2h": test_res["auc_roc_2h"],
            "pr_auc_2h": test_res["pr_auc_2h"],
        },
        "best_val_pearson": best_val_r,
        "epoch_history": epoch_history,
        "step_history": step_history,
        "sample_attn_map": sample_attn_map,
        "total_training_time_seconds": time.time() - train_start_t,
    }

    try:
        with open(f"/root/.cache/huggingface/deep_extract_{split_name}_metrics.json", "w") as f:
            json.dump(results_payload, f, indent=2)
        df_test_out.to_csv(f"/root/.cache/huggingface/deep_extract_{split_name}_preds.csv", index=False)
        hf_volume.commit()
        print("  [Backup] Successfully persisted metrics and predictions to Modal Volume.")
    except Exception as e:
        print(f"  [Warning] Could not backup to volume: {e}")

    return results_payload, df_test_out.to_dict(orient="records")


# -----------------------------------------------------------------------------
# 3. LOCAL ENTRY POINT
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    split: str = "all",
    epochs: int = 35,
    batch_size: int = 128,
    lr: float = 3e-4,
):
    splits_to_run = ["iid", "novel_allele", "novel_pep"] if split.lower() == "all" else [split.lower()]
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"STARTING ESMC-6B TWO-INSTANCE DEEP EXTRACTION ON SPLITS: {splits_to_run}")
    print("=" * 78)

    for current_split in splits_to_run:
        print("\n" + "=" * 78)
        print(f"DISPATCHING MODAL JOB: ESMC-6B TWO-INSTANCE DEEP EXTRACTION ON {current_split.upper()} SPLIT")
        print("=" * 78)

        results_payload, test_records = run_deep_extract_experiment.remote(
            model_name="biohub/ESMC-6B",
            split_name=current_split,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )

        df_test_preds = pd.DataFrame(test_records)
        pred_path = results_dir / f"predictions_esmc_6b_deep_extract_{current_split}.csv"
        df_test_preds.to_csv(pred_path, index=False)
        print(f"\n[Saved] Test predictions to: {pred_path} ({len(df_test_preds):,} rows)")

        pred_path_2inst = results_dir / f"predictions_esmc_6b_deep_extract_2inst_{current_split}.csv"
        df_test_preds.to_csv(pred_path_2inst, index=False)

        metrics_path = results_dir / f"esmc_deep_extract_{current_split}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results_payload, f, indent=4)
        print(f"[Saved] Metrics and epoch history to: {metrics_path}")

        metrics_path_2inst = results_dir / f"esmc_deep_extract_2inst_{current_split}_metrics.json"
        with open(metrics_path_2inst, "w") as f:
            json.dump(results_payload, f, indent=4)

        tm = results_payload["test_metrics"]
        print(f"\n[RESULT {current_split.upper()}] Pearson r = {tm['pearson_r_log']:.4f} | Spearman rho = {tm['spearman_rho']:.4f} | AUROC 1h = {tm['auc_roc_1h']:.4f} | PR-AUC 1h = {tm['pr_auc_1h']:.4f}")

    print("\n" + "=" * 78)
    print("ALL TWO-INSTANCE DEEP EXTRACTION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    pass
