#!/usr/bin/env python3
"""
Peptide-HLA Stability Training Pipeline using biohub/ESMC-300M Embeddings.
Runs with hardware acceleration (MPS on Apple Silicon / CUDA on Linux/Cloud).
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from esm.models.esmc import EsmcModel, EsmcTokenizer


class PeptideHLAInteractionHead(nn.Module):
    """
    Bilinear interaction network combining peptide and HLA embeddings.
    Features: [e_pep, e_hla, e_pep * e_hla, |e_pep - e_hla|] (dim: 4 * d = 3840)
    """
    def __init__(self, d=960, h=512, p=0.2):
        super().__init__()
        in_features = 4 * d
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, h),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(h // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, e_pep, e_hla):
        diff = torch.abs(e_pep - e_hla)
        prod = e_pep * e_hla
        x = torch.cat([e_pep, e_hla, prod, diff], dim=-1)
        return self.net(x).squeeze(-1)


def run_experiment(
    split_name="iid",
    epochs=5,
    batch_size=256,
    lr=1e-3,
    subset_size=None,  # Set to an integer (e.g. 3000) for rapid testing
):
    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dev_desc = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dev_desc = "Apple Silicon MPS (Metal Performance Shaders)"
    else:
        device = torch.device("cpu")
        dev_desc = "CPU"

    print("=" * 72)
    print("PEPTIDE-HLA STABILITY TRAINING WITH ESMC-300M")
    print("=" * 72)
    print(f"Compute Device: {dev_desc}")
    print(f"Split Regime:   {split_name}")
    print(f"Target:         log10(1 + thalf_hours)")
    print("=" * 72)

    # 1. Load Data
    splits_dir = Path("splits")
    train_csv = splits_dir / f"{split_name}_train.csv"
    val_csv = splits_dir / f"{split_name}_val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Missing split files in {splits_dir}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    if subset_size is not None and subset_size < len(df_train):
        print(f"Using subset of {subset_size} samples for fast smoke testing...")
        df_train = df_train.sample(n=subset_size, random_state=42).reset_index(drop=True)
        df_val = df_val.sample(n=min(subset_size // 4, len(df_val)), random_state=42).reset_index(drop=True)

    print(f"\n[Step 1/4] Loaded datasets:")
    print(f"  Train: {len(df_train):,} pairs ({df_train['peptide'].nunique()} unique peptides, {df_train['hla_seq'].nunique()} unique HLAs)")
    print(f"  Val:   {len(df_val):,} pairs ({df_val['peptide'].nunique()} unique peptides, {df_val['hla_seq'].nunique()} unique HLAs)")

    # 2. Load ESMC-300M Model
    print("\n[Step 2/4] Loading biohub/ESMC-300M model and tokenizer...")
    t0 = time.time()
    tokenizer = EsmcTokenizer.from_pretrained("biohub/ESMC-300M")
    esmc_model = EsmcModel.from_pretrained("biohub/ESMC-300M").to(device).eval()
    print(f"  ESMC-300M ready in {time.time() - t0:.2f}s (Parameters: {sum(p.numel() for p in esmc_model.parameters()):,})")

    # 3. Two-Tower Feature Extraction (Cached by unique sequence)
    print("\n[Step 3/4] Caching ESMC-300M embeddings for unique sequences...")
    unique_peptides = list(set(df_train["peptide"]).union(set(df_val["peptide"])))
    unique_hlas = list(set(df_train["hla_seq"]).union(set(df_val["hla_seq"])))
    print(f"  Unique peptides to embed: {len(unique_peptides):,}")
    print(f"  Unique HLAs to embed:     {len(unique_hlas):,}")

    def embed_sequences(seq_list, batch_sz=64, desc="Sequences"):
        embeddings = {}
        t_start = time.time()
        with torch.no_grad():
            for i in range(0, len(seq_list), batch_sz):
                batch = seq_list[i : i + batch_sz]
                tokens = tokenizer(batch, return_tensors="pt", padding=True).to(device)
                out = esmc_model(**tokens)
                mask = tokens.attention_mask.unsqueeze(-1)
                token_embs = out.last_hidden_state * mask
                pooled = (token_embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).cpu()
                for seq, emb in zip(batch, pooled):
                    embeddings[seq] = emb
        elapsed = time.time() - t_start
        rate = len(seq_list) / max(elapsed, 0.001)
        print(f"  -> Embedded {len(seq_list):,} {desc} in {elapsed:.2f}s ({rate:.1f} seq/s)")
        return embeddings

    # Batch sizes tailored for memory efficiency
    batch_sz_hla = 16 if device.type == "mps" else 64
    batch_sz_pep = 64 if device.type == "mps" else 256

    hla_embeddings = embed_sequences(unique_hlas, batch_sz=batch_sz_hla, desc="HLAs (182-aa)")
    pep_embeddings = embed_sequences(unique_peptides, batch_sz=batch_sz_pep, desc="Peptides (9-mer)")

    # Assemble tensors
    train_p = torch.stack([pep_embeddings[p] for p in df_train["peptide"]])
    train_h = torch.stack([hla_embeddings[h] for h in df_train["hla_seq"]])
    train_y_reg = torch.tensor(np.log10(1.0 + df_train["thalf_hours"].values.astype(np.float32)))

    val_p = torch.stack([pep_embeddings[p] for p in df_val["peptide"]])
    val_h = torch.stack([hla_embeddings[h] for h in df_val["hla_seq"]])
    val_y_reg = torch.tensor(np.log10(1.0 + df_val["thalf_hours"].values.astype(np.float32)))
    val_y_bin = (df_val["thalf_hours"].values >= 1.0).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(train_p, train_h, train_y_reg),
        batch_size=batch_size,
        shuffle=True,
    )

    # 4. Train Interaction Head
    print("\n[Step 4/4] Training Bilinear Interaction Head...")
    head_model = PeptideHLAInteractionHead(d=960, h=512, p=0.2).to(device)
    optimizer = torch.optim.AdamW(head_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    print("-" * 72)
    print(f"{'Epoch':<6} | {'Train MSE':<10} | {'Val MSE':<10} | {'Pearson r':<10} | {'Spearman ρ':<11} | {'AUROC (1h)':<10}")
    print("-" * 72)

    best_spearman = -1.0
    for epoch in range(1, epochs + 1):
        head_model.train()
        train_loss = 0.0
        for bp, bh, by in train_loader:
            bp, bh, by = bp.to(device), bh.to(device), by.to(device)
            optimizer.zero_grad()
            preds = head_model(bp, bh)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= len(train_p)
        scheduler.step()

        # Validation
        head_model.eval()
        with torch.no_grad():
            val_preds_list = []
            for i in range(0, len(val_p), 512):
                bp = val_p[i : i + 512].to(device)
                bh = val_h[i : i + 512].to(device)
                val_preds_list.append(head_model(bp, bh).cpu())
            val_preds = torch.cat(val_preds_list).numpy()

        val_true = val_y_reg.numpy()
        val_mse = float(np.mean((val_preds - val_true) ** 2))
        val_pearson, _ = pearsonr(val_preds, val_true)
        val_spearman, _ = spearmanr(val_preds, val_true)

        try:
            val_auroc = float(roc_auc_score(val_y_bin, val_preds))
        except Exception:
            val_auroc = float("nan")

        print(f"{epoch:<6} | {train_loss:<10.4f} | {val_mse:<10.4f} | {val_pearson:<10.4f} | {val_spearman:<11.4f} | {val_auroc:<10.4f}")

    print("-" * 72)
    print("Training run finished successfully!")
    print(f"Final Val Pearson r:  {val_pearson:.4f}")
    print(f"Final Val Spearman ρ: {val_spearman:.4f}")
    print(f"Final Val AUROC (1h): {val_auroc:.4f}")
    print("=" * 72)

    # Save model checkpoint
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    ckpt_path = output_dir / f"esmc_interaction_head_{split_name}.pt"
    torch.save(head_model.state_dict(), ckpt_path)
    print(f"Model saved to: {ckpt_path}\n")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "iid"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_experiment(split_name=split, epochs=epochs)
