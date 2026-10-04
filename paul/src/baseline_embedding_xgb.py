#!/usr/bin/env python3
"""
Baseline Model Suite for Peptide-HLA Complex Stability Prediction.
Implements:
1. Pure Neural Network with Learned Embedding Layers (PMHCEmbeddingNet)
2. Tabular XGBoost on Classical One-Hot Sequences
3. Hybrid Baseline: XGBoost trained on Learned Latent Embeddings (PMHC-Embed-XGB)

Target: log(1 + thalf_hours) regression with robust metrics:
- Pearson r (log and linear scale)
- Spearman rho (rank correlation)
- RMSE and MAE (hours)
- Binary classification AUC-ROC & PR-AUC (at 1h and 2h stability thresholds)
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import xgboost as xgb


# =============================================================================
# 1. CONSTANTS & ENCODING
# =============================================================================
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX["<PAD>"] = 0
AA_TO_IDX["<UNK>"] = len(AMINO_ACIDS) + 1  # 21
VOCAB_SIZE = len(AA_TO_IDX)

PEP_LEN = 9
HLA_PSEUDO_LEN = 34


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def tokenize_sequence(seq: str, max_len: int) -> np.ndarray:
    """Tokenize an amino acid sequence to an integer vector with padding/truncation."""
    tokens = [AA_TO_IDX.get(char, AA_TO_IDX["<UNK>"]) for char in seq[:max_len]]
    if len(tokens) < max_len:
        tokens = tokens + [AA_TO_IDX["<PAD>"]] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.int64)


def encode_one_hot_sequence(seq: str, max_len: int) -> np.ndarray:
    """Encode sequence into flattened one-hot vector (max_len * 20)."""
    one_hot = np.zeros((max_len, len(AMINO_ACIDS)), dtype=np.float32)
    for i, char in enumerate(seq[:max_len]):
        if char in AA_TO_IDX and AA_TO_IDX[char] <= len(AMINO_ACIDS):
            idx = AA_TO_IDX[char] - 1
            one_hot[i, idx] = 1.0
    return one_hot.flatten()


def build_one_hot_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build flattened one-hot feature matrix for (peptide + hla_pseudoseq)."""
    n = len(df)
    pep_feats = np.zeros((n, PEP_LEN * len(AMINO_ACIDS)), dtype=np.float32)
    hla_feats = np.zeros((n, HLA_PSEUDO_LEN * len(AMINO_ACIDS)), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        pep_feats[i] = encode_one_hot_sequence(row["peptide"], PEP_LEN)
        hla_feats[i] = encode_one_hot_sequence(row["hla_pseudoseq"], HLA_PSEUDO_LEN)

    return np.hstack([pep_feats, hla_feats])


# =============================================================================
# 2. PYTORCH DATASET
# =============================================================================
class PMHCDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.peptides = np.vstack([tokenize_sequence(p, PEP_LEN) for p in df["peptide"]])
        self.hlas = np.vstack([tokenize_sequence(h, HLA_PSEUDO_LEN) for h in df["hla_pseudoseq"]])
        
        # Continuous target: log(1 + thalf_hours)
        raw_thalf = df["thalf_hours"].values.astype(np.float32)
        self.targets_raw = raw_thalf
        self.targets_log = np.log1p(np.maximum(0.0, raw_thalf))

    def __len__(self):
        return len(self.targets_raw)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.peptides[idx], dtype=torch.long),
            torch.tensor(self.hlas[idx], dtype=torch.long),
            torch.tensor(self.targets_log[idx], dtype=torch.float32),
            torch.tensor(self.targets_raw[idx], dtype=torch.float32),
        )


# =============================================================================
# 3. LEARNED EMBEDDING NEURAL NETWORK (PMHCEmbeddingNet)
# =============================================================================
class PMHCEmbeddingNet(nn.Module):
    """
    Dual-tower neural network with learned amino acid embeddings.
    Extracts structured biophysical representations from peptide and HLA pseudosequence,
    models their interaction cross-features, and predicts stability.
    """
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 32,
        repr_dim: int = 64,
        latent_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.repr_dim = repr_dim
        self.latent_dim = latent_dim

        # Shared amino acid embedding layer for both peptide and HLA pocket
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Peptide Tower (L=9)
        self.pep_conv1 = nn.Conv1d(embed_dim, repr_dim, kernel_size=3, padding=1)
        self.pep_bn1 = nn.BatchNorm1d(repr_dim)
        self.pep_conv2 = nn.Conv1d(repr_dim, repr_dim, kernel_size=3, padding=1)
        self.pep_bn2 = nn.BatchNorm1d(repr_dim)
        self.pep_proj = nn.Linear(repr_dim * 2, repr_dim)

        # HLA Pseudosequence Tower (L=34 contact residues)
        self.hla_conv1 = nn.Conv1d(embed_dim, repr_dim, kernel_size=3, padding=1)
        self.hla_bn1 = nn.BatchNorm1d(repr_dim)
        self.hla_conv2 = nn.Conv1d(repr_dim, repr_dim, kernel_size=5, padding=2)
        self.hla_bn2 = nn.BatchNorm1d(repr_dim)
        self.hla_proj = nn.Linear(repr_dim * 2, repr_dim)

        # Interaction & Cross-feature bottleneck
        # Concat: [h_pep, h_hla, h_pep * h_hla, |h_pep - h_hla|] -> 4 * repr_dim
        self.interaction_fc1 = nn.Linear(4 * repr_dim, latent_dim)
        self.interaction_ln1 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.interaction_fc2 = nn.Linear(latent_dim, latent_dim)
        self.interaction_ln2 = nn.LayerNorm(latent_dim)

        # Final Regressor Head: outputs log(1 + thalf_hours)
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def encode_towers(self, pep_seq: torch.Tensor, hla_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pep_seq: (B, 9) -> (B, 9, embed_dim) -> transpose -> (B, embed_dim, 9)
        pep_emb = self.embedding(pep_seq).transpose(1, 2)
        p = F.gelu(self.pep_bn1(self.pep_conv1(pep_emb)))
        p = F.gelu(self.pep_bn2(self.pep_conv2(p)))
        # Global max and avg pooling
        p_max = F.adaptive_max_pool1d(p, 1).squeeze(-1)
        p_avg = F.adaptive_avg_pool1d(p, 1).squeeze(-1)
        h_pep = F.gelu(self.pep_proj(torch.cat([p_max, p_avg], dim=-1)))  # (B, repr_dim)

        # hla_seq: (B, 34) -> (B, 34, embed_dim) -> transpose -> (B, embed_dim, 34)
        hla_emb = self.embedding(hla_seq).transpose(1, 2)
        h = F.gelu(self.hla_bn1(self.hla_conv1(hla_emb)))
        h = F.gelu(self.hla_bn2(self.hla_conv2(h)))
        h_max = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        h_avg = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        h_hla = F.gelu(self.hla_proj(torch.cat([h_max, h_avg], dim=-1)))  # (B, repr_dim)

        return h_pep, h_hla

    def compute_latent(self, h_pep: torch.Tensor, h_hla: torch.Tensor) -> torch.Tensor:
        cross_prod = h_pep * h_hla
        cross_diff = torch.abs(h_pep - h_hla)
        joint = torch.cat([h_pep, h_hla, cross_prod, cross_diff], dim=-1)

        z = F.gelu(self.interaction_ln1(self.interaction_fc1(joint)))
        z = self.dropout(z)
        z = F.gelu(self.interaction_ln2(self.interaction_fc2(z)))
        return z

    def forward(self, pep_seq: torch.Tensor, hla_seq: torch.Tensor) -> torch.Tensor:
        h_pep, h_hla = self.encode_towers(pep_seq, hla_seq)
        z = self.compute_latent(h_pep, h_hla)
        out = self.regressor(z)
        return out.squeeze(-1)

    def extract_features(self, pep_seq: torch.Tensor, hla_seq: torch.Tensor) -> torch.Tensor:
        """
        Extracts multi-level learned embeddings:
        [h_pep (repr_dim), h_hla (repr_dim), z_interaction (latent_dim), nn_pred (1)]
        """
        h_pep, h_hla = self.encode_towers(pep_seq, hla_seq)
        z = self.compute_latent(h_pep, h_hla)
        pred = self.regressor(z)
        features = torch.cat([h_pep, h_hla, z, pred], dim=-1)
        return features


# =============================================================================
# 4. TRAINING & EXTRACTION UTILITIES
# =============================================================================
def train_neural_network(
    model: PMHCEmbeddingNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 35,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    save_path: str = None,
) -> Dict[str, List[float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.SmoothL1Loss(beta=1.0)

    history = {"train_loss": [], "val_loss": [], "val_spearman": [], "val_pearson": []}
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    print(f"Training PMHCEmbeddingNet on {device} (Max Epochs: {epochs}, Patience: {patience})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for pep_t, hla_t, targets_log, _ in train_loader:
            pep_t, hla_t, targets_log = (
                pep_t.to(device),
                hla_t.to(device),
                targets_log.to(device),
            )
            optimizer.zero_grad()
            preds = model(pep_t, hla_t)
            loss = criterion(preds, targets_log)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_train_loss += loss.item() * len(targets_log)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds_list = []
        val_true_list = []

        with torch.no_grad():
            for pep_t, hla_t, targets_log, _ in val_loader:
                pep_t, hla_t, targets_log = (
                    pep_t.to(device),
                    hla_t.to(device),
                    targets_log.to(device),
                )
                preds = model(pep_t, hla_t)
                loss = criterion(preds, targets_log)
                total_val_loss += loss.item() * len(targets_log)
                val_preds_list.extend(preds.cpu().numpy())
                val_true_list.extend(targets_log.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)

        v_true = np.array(val_true_list)
        v_pred = np.array(val_preds_list)
        val_sp, _ = spearmanr(v_true, v_pred)
        val_pr, _ = pearsonr(v_true, v_pred)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_spearman"].append(float(val_sp))
        history["val_pearson"].append(float(val_pr))

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Spearman: {val_sp:.4f} | Val Pearson: {val_pr:.4f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
        if save_path:
            torch.save(best_weights, save_path)
            print(f"Saved best model checkpoint to {save_path}")

    return history


def extract_learned_features(
    model: PMHCEmbeddingNet, dataloader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts latent embeddings, predictions, and ground-truth targets from the model.
    """
    model.eval()
    model.to(device)
    all_features = []
    all_targets_log = []
    all_targets_raw = []

    with torch.no_grad():
        for pep_t, hla_t, targets_log, targets_raw in dataloader:
            pep_t, hla_t = pep_t.to(device), hla_t.to(device)
            feats = model.extract_features(pep_t, hla_t)
            all_features.append(feats.cpu().numpy())
            all_targets_log.append(targets_log.numpy())
            all_targets_raw.append(targets_raw.numpy())

    features = np.vstack(all_features)
    targets_log = np.concatenate(all_targets_log)
    targets_raw = np.concatenate(all_targets_raw)
    return features, targets_log, targets_raw


# =============================================================================
# 5. METRICS SUITE
# =============================================================================
def compute_comprehensive_metrics(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> Dict[str, float]:
    """
    Computes regression and classification stability metrics.
    Raw half-life is non-negative hours.
    """
    y_pred_raw_clamped = np.maximum(0.0, y_pred_raw)
    
    # Regression metrics
    p_r_log, _ = pearsonr(y_true_log, y_pred_log)
    p_r_raw, _ = pearsonr(y_true_raw, y_pred_raw_clamped)
    sp_rho, _ = spearmanr(y_true_raw, y_pred_raw_clamped)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    rmse_raw = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw_clamped))
    mae_raw = mean_absolute_error(y_true_raw, y_pred_raw_clamped)

    # Binary Classification metrics at established thresholds:
    # 1.0 hour (Binder vs Non-binder)
    b1_true = (y_true_raw >= 1.0).astype(int)
    auc_1h = (
        roc_auc_score(b1_true, y_pred_raw_clamped)
        if len(np.unique(b1_true)) > 1
        else np.nan
    )
    prauc_1h = (
        average_precision_score(b1_true, y_pred_raw_clamped)
        if len(np.unique(b1_true)) > 1
        else np.nan
    )

    # 2.0 hours (Stringent / Strong Binder)
    b2_true = (y_true_raw >= 2.0).astype(int)
    auc_2h = (
        roc_auc_score(b2_true, y_pred_raw_clamped)
        if len(np.unique(b2_true)) > 1
        else np.nan
    )
    prauc_2h = (
        average_precision_score(b2_true, y_pred_raw_clamped)
        if len(np.unique(b2_true)) > 1
        else np.nan
    )

    return {
        "pearson_r_log": round(float(p_r_log), 4),
        "pearson_r_raw": round(float(p_r_raw), 4),
        "spearman_rho": round(float(sp_rho), 4),
        "rmse_log": round(float(rmse_log), 4),
        "rmse_raw": round(float(rmse_raw), 4),
        "mae_raw": round(float(mae_raw), 4),
        "auc_roc_1h": round(float(auc_1h), 4) if not np.isnan(auc_1h) else None,
        "pr_auc_1h": round(float(prauc_1h), 4) if not np.isnan(prauc_1h) else None,
        "auc_roc_2h": round(float(auc_2h), 4) if not np.isnan(auc_2h) else None,
        "pr_auc_2h": round(float(prauc_2h), 4) if not np.isnan(prauc_2h) else None,
    }


# =============================================================================
# 6. PIPELINE RUNNER FOR A SINGLE SPLIT REGIME
# =============================================================================
def run_benchmark_on_split(
    split_name: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    results_dir: Path,
    device: torch.device,
    batch_size: int = 256,
    nn_epochs: int = 35,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"RUNNING BENCHMARK REGIME: {split_name.upper()}")
    print("=" * 80)

    seed_everything(seed)

    # 1. Load Data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Datasets: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")

    train_ds = PMHCDataset(train_df)
    val_ds = PMHCDataset(val_df)
    test_ds = PMHCDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # -------------------------------------------------------------------------
    # STAGE 1: Train Neural Network with Learned Embeddings
    # -------------------------------------------------------------------------
    print("\n--- [Stage 1] Training Neural Network (PMHCEmbeddingNet) ---")
    model_ckpt_path = results_dir / f"pmhc_embedding_net_{split_name}.pt"
    nn_model = PMHCEmbeddingNet(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        repr_dim=64,
        latent_dim=64,
        dropout=0.2,
    )

    t0 = time.time()
    train_history = train_neural_network(
        model=nn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=nn_epochs,
        lr=1e-3,
        weight_decay=1e-4,
        patience=8,
        save_path=str(model_ckpt_path),
    )
    nn_time = time.time() - t0
    print(f"Neural Net training completed in {nn_time:.1f}s")

    # Extract Learned Features & Predictions
    print("\n--- Extracting Learned Latent Representations ---")
    X_train_embed, y_train_log, y_train_raw = extract_learned_features(nn_model, train_loader, device)
    X_val_embed, y_val_log, y_val_raw = extract_learned_features(nn_model, val_loader, device)
    X_test_embed, y_test_log, y_test_raw = extract_learned_features(nn_model, test_loader, device)

    # Pure NN Predictions (last column of extracted features is nn_pred in log scale)
    nn_pred_val_log = X_val_embed[:, -1]
    nn_pred_val_raw = np.expm1(np.maximum(0.0, nn_pred_val_log))
    nn_pred_test_log = X_test_embed[:, -1]
    nn_pred_test_raw = np.expm1(np.maximum(0.0, nn_pred_test_log))

    nn_test_metrics = compute_comprehensive_metrics(
        y_true_raw=y_test_raw,
        y_pred_raw=nn_pred_test_raw,
        y_true_log=y_test_log,
        y_pred_log=nn_pred_test_log,
    )

    # -------------------------------------------------------------------------
    # STAGE 2: Baseline XGBoost on Classical One-Hot Sequences
    # -------------------------------------------------------------------------
    print("\n--- [Stage 2] Classical Baseline: XGBoost on One-Hot Sequences ---")
    t0 = time.time()
    X_train_oh = build_one_hot_matrix(train_df)
    X_val_oh = build_one_hot_matrix(val_df)
    X_test_oh = build_one_hot_matrix(test_df)

    xgb_onehot = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    xgb_onehot.fit(
        X_train_oh,
        y_train_log,
        eval_set=[(X_val_oh, y_val_log)],
        verbose=False,
    )
    xgb_oh_time = time.time() - t0
    print(f"XGBoost One-Hot training completed in {xgb_oh_time:.1f}s (Best iter: {xgb_onehot.best_iteration})")

    xgb_oh_pred_test_log = xgb_onehot.predict(X_test_oh)
    xgb_oh_pred_test_raw = np.expm1(np.maximum(0.0, xgb_oh_pred_test_log))

    xgb_oh_test_metrics = compute_comprehensive_metrics(
        y_true_raw=y_test_raw,
        y_pred_raw=xgb_oh_pred_test_raw,
        y_true_log=y_test_log,
        y_pred_log=xgb_oh_pred_test_log,
    )

    # -------------------------------------------------------------------------
    # STAGE 3: Hybrid Baseline: XGBoost on Learned Latent Embeddings
    # -------------------------------------------------------------------------
    print("\n--- [Stage 3] Hybrid Model: XGBoost on Learned Embeddings ---")
    t0 = time.time()
    xgb_embed = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=35,
        eval_metric="rmse",
    )
    xgb_embed.fit(
        X_train_embed,
        y_train_log,
        eval_set=[(X_val_embed, y_val_log)],
        verbose=False,
    )
    xgb_embed_time = time.time() - t0
    print(f"XGBoost on Learned Embeddings training completed in {xgb_embed_time:.1f}s (Best iter: {xgb_embed.best_iteration})")

    xgb_embed_pred_test_log = xgb_embed.predict(X_test_embed)
    xgb_embed_pred_test_raw = np.expm1(np.maximum(0.0, xgb_embed_pred_test_log))

    xgb_embed_test_metrics = compute_comprehensive_metrics(
        y_true_raw=y_test_raw,
        y_pred_raw=xgb_embed_pred_test_raw,
        y_true_log=y_test_log,
        y_pred_log=xgb_embed_pred_test_log,
    )

    # Save models
    xgb_embed_model_path = results_dir / f"xgb_on_embeddings_{split_name}.json"
    xgb_embed.save_model(str(xgb_embed_model_path))

    # -------------------------------------------------------------------------
    # PRINT COMPARISON SUMMARY
    # -------------------------------------------------------------------------
    summary_df = pd.DataFrame([
        {"Model": "1. One-Hot + XGBoost", **xgb_oh_test_metrics},
        {"Model": "2. Pure Neural Net (Learned Embeddings)", **nn_test_metrics},
        {"Model": "3. Hybrid: XGBoost on Learned Embeddings", **xgb_embed_test_metrics},
    ])

    print("\n" + "=" * 80)
    print(f"TEST RESULTS COMPARISON [{split_name.upper()}]")
    print("=" * 80)
    print(summary_df[["Model", "spearman_rho", "pearson_r_log", "rmse_raw", "auc_roc_1h", "auc_roc_2h"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # SAVE TEST PREDICTIONS CSV
    # -------------------------------------------------------------------------
    preds_df = test_df.copy()
    preds_df["pred_log_nn"] = nn_pred_test_log
    preds_df["pred_hours_nn"] = nn_pred_test_raw
    preds_df["pred_log_xgb_onehot"] = xgb_oh_pred_test_log
    preds_df["pred_hours_xgb_onehot"] = xgb_oh_pred_test_raw
    preds_df["pred_log_xgb_embed"] = xgb_embed_pred_test_log
    preds_df["pred_hours_xgb_embed"] = xgb_embed_pred_test_raw

    pred_csv_path = results_dir / f"predictions_{split_name}.csv"
    preds_df.to_csv(pred_csv_path, index=False)
    print(f"\nSaved test predictions to: {pred_csv_path}")

    # Return structured dict
    return {
        "split_name": split_name,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "metrics": {
            "onehot_xgboost": xgb_oh_test_metrics,
            "neural_network": nn_test_metrics,
            "xgboost_on_embeddings": xgb_embed_test_metrics,
        },
        "training_time_seconds": {
            "neural_network": round(nn_time, 2),
            "onehot_xgboost": round(xgb_oh_time, 2),
            "xgboost_on_embeddings": round(xgb_embed_time, 2),
        },
        "feature_importances_top10": [
            {"feat_idx": int(idx), "importance": float(imp)}
            for idx, imp in sorted(
                enumerate(xgb_embed.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        ],
    }


# =============================================================================
# 7. MAIN CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Baseline Learned Embedding + XGBoost Pipeline")
    parser.add_argument(
        "--split",
        type=str,
        default="iid",
        choices=["iid", "novel_pep", "novel_allele", "all"],
        help="Split regime to evaluate ('iid', 'novel_pep', 'novel_allele', or 'all')",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Max neural net training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store outputs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    regimes_to_run = []
    if args.split in ["iid", "all"]:
        regimes_to_run.append((
            "iid",
            "splits/iid_train.csv",
            "splits/iid_val.csv",
            "splits/iid_test.csv",
        ))
    if args.split in ["novel_pep", "all"]:
        regimes_to_run.append((
            "novel_pep",
            "splits/novel_pep_train.csv",
            "splits/novel_pep_val.csv",
            "splits/novel_pep_test.csv",
        ))
    if args.split in ["novel_allele", "all"]:
        regimes_to_run.append((
            "novel_allele",
            "splits/novel_allele_train.csv",
            "splits/novel_allele_val.csv",
            "splits/novel_allele_test.csv",
        ))

    all_results = {}
    for split_name, train_csv, val_csv, test_csv in regimes_to_run:
        res = run_benchmark_on_split(
            split_name=split_name,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            results_dir=results_dir,
            device=device,
            batch_size=args.batch_size,
            nn_epochs=args.epochs,
            seed=args.seed,
        )
        all_results[split_name] = res

    # Save overall summary JSON
    metrics_path = results_dir / "baseline_metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results successfully written to: {metrics_path}")


if __name__ == "__main__":
    main()
