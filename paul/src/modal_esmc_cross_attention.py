"""
Modal Training: ESMC-6B with Cross-Attention Matrix Interaction Head (IID Split).
Uses token-level residue embeddings from biohub/ESMC-6B and predicts stability
directly and exclusively from the cross-attention matrix between peptide and HLA residues.
Includes detailed step-level and epoch-level training loss logging.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import modal

# -----------------------------------------------------------------------------
# 1. MODAL APP & CONTAINER DEFINITION
# -----------------------------------------------------------------------------
app = modal.App("esmc-cross-attention-iid")

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
# 2. MODAL REMOTE FUNCTION: EXECUTION ON NVIDIA A100-40GB
# -----------------------------------------------------------------------------
@app.function(
    gpu="A100-40GB",
    image=esmc_image,
    volumes={"/root/.cache/huggingface": hf_volume},
    timeout=1800,
)
def run_cross_attention_experiment(
    model_name: str = "biohub/ESMC-6B",
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    num_heads: int = 8,
    head_dim: int = 64,
):
    import time
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
    print(f"ESMC-6B CROSS-ATTENTION MATRIX MODEL (IID SPLIT) ON {gpu_name}")
    print("=" * 78)
    print(f"Device: {device} | VRAM: {total_mem_gb:.2f} GB | PyTorch: {torch.__version__}")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, heads={num_heads}")
    print("=" * 78)

    # 1. Load Data
    splits_dir = Path("/root/splits")
    df_train = pd.read_csv(splits_dir / "iid_train.csv")
    df_val = pd.read_csv(splits_dir / "iid_val.csv")
    df_test = pd.read_csv(splits_dir / "iid_test.csv")

    print(f"\n[Step 1/5] Loaded IID Dataset:")
    print(f"  Train: {len(df_train):,} pairs ({df_train['peptide'].nunique()} peptides, {df_train['allele'].nunique()} alleles)")
    print(f"  Val:   {len(df_val):,} pairs ({df_val['peptide'].nunique()} peptides, {df_val['allele'].nunique()} alleles)")
    print(f"  Test:  {len(df_test):,} pairs ({df_test['peptide'].nunique()} peptides, {df_test['allele'].nunique()} alleles)")

    # 2. Load ESMC-6B Foundation Model
    print(f"\n[Step 2/5] Loading '{model_name}' onto GPU...")
    t0 = time.time()
    tokenizer = EsmcTokenizer.from_pretrained(model_name)
    esmc_model = EsmcModel.from_pretrained(model_name, dtype=torch.bfloat16, device="cuda").eval()
    d_model = esmc_model.config.d_model if hasattr(esmc_model.config, "d_model") else 2560
    load_time = time.time() - t0
    param_count = sum(p.numel() for p in esmc_model.parameters())
    print(f"  Loaded in {load_time:.2f}s | Parameters: {param_count:,} | d_model: {d_model}")
    print(f"  GPU VRAM allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # 3. Extract Token-Level Residue Embeddings for all Unique Peptides & HLAs
    print(f"\n[Step 3/5] Extracting token-level residue embeddings...")
    all_peptides = sorted(list(set(df_train["peptide"]).union(set(df_val["peptide"])).union(set(df_test["peptide"]))))
    all_hlas = sorted(list(set(df_train["hla_seq"]).union(set(df_val["hla_seq"])).union(set(df_test["hla_seq"]))))
    print(f"  Unique peptides: {len(all_peptides):,} (9 residues each)")
    print(f"  Unique HLAs:     {len(all_hlas):,} (182 residues each)")

    pep_to_idx = {p: i for i, p in enumerate(all_peptides)}
    hla_to_idx = {h: i for i, h in enumerate(all_hlas)}

    # Batch extraction for HLAs (182 residues -> token slice 1:183)
    hla_tokens_list = []
    hla_batch_sz = 16
    t_start = time.time()
    with torch.no_grad():
        for i in range(0, len(all_hlas), hla_batch_sz):
            batch = all_hlas[i : i + hla_batch_sz]
            tok = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            out = esmc_model(**tok)
            # Slice residue positions 1:183 (exact 182 residues)
            res_tokens = out.last_hidden_state[:, 1:183, :].to(dtype=torch.float32).cpu()
            hla_tokens_list.append(res_tokens)
    hla_tokens_tensor = torch.cat(hla_tokens_list, dim=0) # (num_hlas, 182, d_model)
    print(f"  -> Extracted HLA token tensors in {time.time() - t_start:.2f}s: shape {hla_tokens_tensor.shape}")

    # Batch extraction for Peptides (9 residues -> token slice 1:10)
    pep_tokens_list = []
    pep_batch_sz = 64
    t_start = time.time()
    with torch.no_grad():
        for i in range(0, len(all_peptides), pep_batch_sz):
            batch = all_peptides[i : i + pep_batch_sz]
            tok = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            out = esmc_model(**tok)
            # Slice residue positions 1:10 (exact 9 residues)
            res_tokens = out.last_hidden_state[:, 1:10, :].to(dtype=torch.float32).cpu()
            pep_tokens_list.append(res_tokens)
    pep_tokens_tensor = torch.cat(pep_tokens_list, dim=0) # (num_peptides, 9, d_model)
    print(f"  -> Extracted Peptide token tensors in {time.time() - t_start:.2f}s: shape {pep_tokens_tensor.shape}")

    # Free ESMc transformer weights from GPU memory
    del esmc_model
    torch.cuda.empty_cache()
    print(f"  ESMc unloaded. Free GPU VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB used.")

    # Move token tensors to GPU for zero-latency indexing during training
    hla_tokens_gpu = hla_tokens_tensor.to(device)
    pep_tokens_gpu = pep_tokens_tensor.to(device)
    print(f"  Cached all token tensors in GPU VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB used.")

    # Prepare index datasets
    def make_dataset(df):
        p_indices = torch.tensor([pep_to_idx[p] for p in df["peptide"]], dtype=torch.long)
        h_indices = torch.tensor([hla_to_idx[h] for h in df["hla_seq"]], dtype=torch.long)
        y_log = torch.tensor(np.log10(1.0 + df["thalf_hours"].values), dtype=torch.float32)
        y_bin = torch.tensor((df["thalf_hours"].values >= 1.0).astype(np.float32), dtype=torch.float32)
        y_raw = torch.tensor(df["thalf_hours"].values, dtype=torch.float32)
        return TensorDataset(p_indices, h_indices, y_log, y_bin, y_raw)

    ds_train = make_dataset(df_train)
    ds_val = make_dataset(df_val)
    ds_test = make_dataset(df_test)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # 4. Define Cross-Attention Matrix Interaction Head
    class CrossAttentionMatrixModel(nn.Module):
        """
        Predicts peptide-MHC stability solely from the residue-level
        cross-attention matrix between peptide (9) and HLA (182) residues.
        """
        def __init__(self, d_model=2560, num_heads=8, head_dim=64):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.scale = head_dim ** -0.5

            # Pre-LayerNorm on foundation embeddings
            self.ln_pep = nn.LayerNorm(d_model)
            self.ln_hla = nn.LayerNorm(d_model)

            # Query (Peptide) and Key (HLA) projections
            self.q_proj = nn.Linear(d_model, num_heads * head_dim)
            self.k_proj = nn.Linear(d_model, num_heads * head_dim)

            # In channels to the 2D contact matrix analyzer:
            # - num_heads forward attention matrices (softmax along HLA residues)
            # - num_heads reverse attention matrices (softmax along peptide residues)
            # - num_heads scaled affinity logit matrices
            # - 1 foundation model unprojected cosine similarity matrix
            in_channels = num_heads * 3 + 1

            # 2D Contact-Map Network operating directly on the 9x182 cross-attention matrix
            self.conv_net = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=(3, 5), padding=(1, 2)),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, 256, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.Conv2d(256, 256, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((9, 1)) # Preserves the 9 peptide positions explicitly!
            )

            # Dense regression and classification heads
            self.fc = nn.Sequential(
                nn.Linear(256 * 9, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            self.out_reg = nn.Linear(128, 1)
            self.out_clf = nn.Linear(128, 1)

        def forward(self, p_tokens, h_tokens):
            # p_tokens: (B, 9, d_model), h_tokens: (B, 182, d_model)
            B = p_tokens.size(0)

            # 1. Foundation cosine similarity matrix (B, 1, 9, 182)
            p_norm = F.normalize(p_tokens, p=2, dim=-1)
            h_norm = F.normalize(h_tokens, p=2, dim=-1)
            cos_sim = torch.bmm(p_norm, h_norm.transpose(1, 2)).unsqueeze(1)

            # 2. Multi-Head Cross Attention
            p_ln = self.ln_pep(p_tokens)
            h_ln = self.ln_hla(h_tokens)

            Q = self.q_proj(p_ln).view(B, 9, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.k_proj(h_ln).view(B, 182, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores: (B, H, 9, 182)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

            # Forward attention (softmax over HLA groove)
            attn_fwd = F.softmax(scores, dim=-1)
            # Reverse attention (softmax over peptide anchors)
            attn_rev = F.softmax(scores, dim=-2)
            # Scaled affinity logits
            scaled_logits = torch.tanh(scores / 3.0)

            # Concatenate into full cross-attention feature map
            attn_maps = torch.cat([attn_fwd, attn_rev, scaled_logits, cos_sim], dim=1) # (B, in_channels, 9, 182)

            # 3. Process the cross-attention matrix
            feat_map = self.conv_net(attn_maps) # (B, 256, 9, 1)
            flat = feat_map.view(B, -1)         # (B, 256 * 9)

            # 4. Predict stability and binding probability
            h = self.fc(flat)
            pred_reg = self.out_reg(h).squeeze(-1)
            pred_clf = self.out_clf(h).squeeze(-1)
            return pred_reg, pred_clf, attn_fwd

    model = CrossAttentionMatrixModel(d_model=d_model, num_heads=num_heads, head_dim=head_dim).to(device)
    print(f"\n[Step 4/5] Initialized CrossAttentionMatrixModel with {sum(p.numel() for p in model.parameters()):,} parameters.")

    criterion_reg = nn.SmoothL1Loss()
    criterion_clf = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 5. Training Loop with Detailed Step and Epoch Loss Logging
    print(f"\n[Step 5/5] Commencing training for {epochs} epochs on IID split...")
    step_loss_history = []
    epoch_history = []
    best_val_r = -1.0
    best_model_state = None
    global_step = 0

    t_train_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        total_reg_loss = 0.0
        total_clf_loss = 0.0
        num_batches = 0

        for p_idx, h_idx, y_log, y_bin, _ in train_loader:
            global_step += 1
            p_tokens = pep_tokens_gpu[p_idx]
            h_tokens = hla_tokens_gpu[h_idx]
            y_log = y_log.to(device)
            y_bin = y_bin.to(device)

            optimizer.zero_grad()
            pred_reg, pred_clf, _ = model(p_tokens, h_tokens)

            loss_r = criterion_reg(pred_reg, y_log)
            loss_c = criterion_clf(pred_clf, y_bin)
            loss = loss_r + 0.3 * loss_c

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_epoch_loss += loss.item()
            total_reg_loss += loss_r.item()
            total_clf_loss += loss_c.item()
            num_batches += 1

            if global_step % 25 == 0:
                step_loss_history.append({
                    "step": global_step,
                    "epoch": ep,
                    "train_loss": loss.item(),
                    "reg_loss": loss_r.item(),
                    "clf_loss": loss_c.item(),
                })

        scheduler.step()
        avg_train_loss = total_epoch_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        avg_clf_loss = total_clf_loss / num_batches

        # Validation evaluation
        model.eval()
        val_preds_reg, val_preds_clf, val_y_log_list = [], [], []
        with torch.no_grad():
            for p_idx, h_idx, y_log, _, _ in val_loader:
                p_tokens = pep_tokens_gpu[p_idx]
                h_tokens = hla_tokens_gpu[h_idx]
                pred_reg, pred_clf, _ = model(p_tokens, h_tokens)
                val_preds_reg.extend(pred_reg.cpu().numpy())
                val_preds_clf.extend(torch.sigmoid(pred_clf).cpu().numpy())
                val_y_log_list.extend(y_log.numpy())

        val_preds_reg = np.array(val_preds_reg)
        val_y_log_arr = np.array(val_y_log_list)
        val_preds_clf = np.array(val_preds_clf)

        val_mse = float(mean_squared_error(val_y_log_arr, val_preds_reg))
        val_r = float(pearsonr(val_y_log_arr, val_preds_reg)[0])
        val_rho = float(spearmanr(val_y_log_arr, val_preds_reg)[0])
        val_y_bin = (val_y_log_arr >= np.log10(2.0)).astype(int)
        val_auroc = float(roc_auc_score(val_y_bin, val_preds_clf)) if len(np.unique(val_y_bin)) > 1 else 0.5
        val_prauc = float(average_precision_score(val_y_bin, val_preds_clf)) if len(np.unique(val_y_bin)) > 1 else 0.5

        epoch_record = {
            "epoch": ep,
            "train_loss": float(avg_train_loss),
            "train_reg_loss": float(avg_reg_loss),
            "train_clf_loss": float(avg_clf_loss),
            "val_mse": val_mse,
            "val_pearson": val_r,
            "val_spearman": val_rho,
            "val_auroc_1h": val_auroc,
            "val_pr_auc_1h": val_prauc,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        epoch_history.append(epoch_record)

        if val_r > best_val_r:
            best_val_r = val_r
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = "★ BEST"
        else:
            mark = ""

        print(
            f"  Epoch {ep:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} (Reg: {avg_reg_loss:.4f}, Clf: {avg_clf_loss:.4f}) | "
            f"Val Pearson: {val_r:.4f} | Val Spearman: {val_rho:.4f} | "
            f"Val AUROC: {val_auroc:.4f} {mark}"
        )

    total_train_time = time.time() - t_train_start
    print(f"\nTraining completed in {total_train_time:.2f}s ({total_train_time / epochs:.2f}s/epoch).")

    # Load best model for Test Evaluation
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()

    test_preds_reg = []
    test_preds_clf = []
    sample_attn_maps = []
    with torch.no_grad():
        for i, (p_idx, h_idx, y_log, y_bin, y_raw) in enumerate(test_loader):
            p_tokens = pep_tokens_gpu[p_idx]
            h_tokens = hla_tokens_gpu[h_idx]
            pred_reg, pred_clf, attn_fwd = model(p_tokens, h_tokens)
            test_preds_reg.extend(pred_reg.cpu().numpy())
            test_preds_clf.extend(torch.sigmoid(pred_clf).cpu().numpy())
            # Save first batch sample attention maps for visualization
            if i == 0 and len(sample_attn_maps) == 0:
                sample_attn_maps = attn_fwd[:5].mean(dim=1).cpu().numpy() # mean across heads: (5, 9, 182)

    df_test_out = df_test.copy()
    test_preds_reg = np.array(test_preds_reg)
    test_preds_clf = np.array(test_preds_clf)

    df_test_out["pred_stability_log"] = test_preds_reg
    df_test_out["pred_thalf_hours"] = np.clip(10**test_preds_reg - 1.0, 0.0, None)
    df_test_out["pred_prob_binding_1h"] = test_preds_clf
    df_test_out["actual_log"] = np.log10(1.0 + df_test_out["thalf_hours"].values)

    y_test_log = df_test_out["actual_log"].values
    y_test_raw = df_test_out["thalf_hours"].values
    y_pred_log = df_test_out["pred_stability_log"].values
    y_pred_raw = df_test_out["pred_thalf_hours"].values

    test_r_log = float(pearsonr(y_test_log, y_pred_log)[0])
    test_r_raw = float(pearsonr(y_test_raw, y_pred_raw)[0])
    test_rho = float(spearmanr(y_test_log, y_pred_log)[0])
    test_rmse_log = float(np.sqrt(mean_squared_error(y_test_log, y_pred_log)))
    test_rmse_raw = float(np.sqrt(mean_squared_error(y_test_raw, y_pred_raw)))
    test_mae_raw = float(mean_absolute_error(y_test_raw, y_pred_raw))

    y_bin_1h = (y_test_raw >= 1.0).astype(int)
    y_bin_2h = (y_test_raw >= 2.0).astype(int)
    auc_roc_1h = float(roc_auc_score(y_bin_1h, test_preds_clf))
    pr_auc_1h = float(average_precision_score(y_bin_1h, test_preds_clf))
    auc_roc_2h = float(roc_auc_score(y_bin_2h, y_pred_log))
    pr_auc_2h = float(average_precision_score(y_bin_2h, y_pred_log))

    print("\n" + "=" * 78)
    print(f"FINAL TEST SET EVALUATION: ESMC-6B CROSS-ATTENTION MATRIX (IID SPLIT)")
    print("=" * 78)
    print(f"  Pearson r (log10 scale):       {test_r_log:.4f}")
    print(f"  Pearson r (raw hours scale):   {test_r_raw:.4f}")
    print(f"  Spearman rho (rank order):     {test_rho:.4f}")
    print(f"  RMSE (log10 scale):            {test_rmse_log:.4f}")
    print(f"  RMSE (raw hours scale):        {test_rmse_raw:.2f} h")
    print(f"  MAE (raw hours scale):         {test_mae_raw:.2f} h")
    print(f"  AUROC (>= 1.0h binder):        {auc_roc_1h:.4f}")
    print(f"  PR-AUC (>= 1.0h binder):       {pr_auc_1h:.4f}")
    print(f"  AUROC (>= 2.0h strong binder): {auc_roc_2h:.4f}")
    print(f"  PR-AUC (>= 2.0h strong binder):{pr_auc_2h:.4f}")
    print("=" * 78)

    results_payload = {
        "model": "ESMC-6B-CrossAttentionMatrix",
        "split": "iid",
        "metrics": {
            "pearson_r_log": test_r_log,
            "pearson_r_raw": test_r_raw,
            "spearman_rho": test_rho,
            "rmse_log": test_rmse_log,
            "rmse_raw": test_rmse_raw,
            "mae_raw": test_mae_raw,
            "auc_roc_1h": auc_roc_1h,
            "pr_auc_1h": pr_auc_1h,
            "auc_roc_2h": auc_roc_2h,
            "pr_auc_2h": pr_auc_2h,
        },
        "epoch_history": epoch_history,
        "step_loss_history": step_loss_history,
        "sample_attn_maps": sample_attn_maps.tolist() if len(sample_attn_maps) > 0 else [],
    }

    return results_payload, df_test_out.to_dict(orient="records")


# -----------------------------------------------------------------------------
# 3. LOCAL ENTRY POINT
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 3e-4,
):
    print("=" * 78)
    print("DISPATCHING MODAL JOB: ESMC-6B CROSS-ATTENTION MATRIX ON IID SPLIT")
    print("=" * 78)

    results_payload, test_records = run_cross_attention_experiment.remote(
        model_name="biohub/ESMC-6B",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Test Predictions
    df_test_preds = pd.DataFrame(test_records)
    pred_path = results_dir / "predictions_esmc_6b_cross_attention_iid.csv"
    df_test_preds.to_csv(pred_path, index=False)
    print(f"\n[Saved] Test predictions to: {pred_path} ({len(df_test_preds):,} rows)")

    # 2. Save Comprehensive Metrics & Histories
    metrics_path = results_dir / "esmc_cross_attention_iid_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results_payload, f, indent=4)
    print(f"[Saved] Metrics, step loss, and epoch history to: {metrics_path}")

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    pass
