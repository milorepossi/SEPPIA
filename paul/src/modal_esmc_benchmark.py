"""
Comprehensive Modal Benchmark: ESMC-6B & ESMC-300M on NVIDIA A100-40GB.
Full multi-task training on Peptide-HLA Complex Stability across:
- IID Split (in-distribution test)
- Novel Allele Split (pan-allele zero-shot generalization test)
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
# 1. MODAL APP & CONTAINER IMAGE DEFINITION
# -----------------------------------------------------------------------------
app = modal.App("esmc-benchmark-a100")

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
    timeout=1800,  # 30 minutes
)
def run_benchmark_on_split(
    model_name: str = "biohub/ESMC-6B",
    split_name: str = "iid",
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
):
    import os
    import time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
    from esm.models.esmc import EsmcModel, EsmcTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0

    print("=" * 76)
    print(f"BENCHMARK: {model_name} ON NVIDIA A100-40GB ({split_name.upper()} SPLIT)")
    print("=" * 76)
    print(f"Compute Device: {device} ({gpu_name}) | Total VRAM: {total_mem_gb:.2f} GB")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    print("=" * 76)

    # 1. Load Data
    splits_dir = Path("/root/splits")
    df_train = pd.read_csv(splits_dir / f"{split_name}_train.csv")
    df_val = pd.read_csv(splits_dir / f"{split_name}_val.csv")
    df_test = pd.read_csv(splits_dir / f"{split_name}_test.csv")

    print(f"\n[Step 1/5] Loaded splits for regime '{split_name}':")
    print(f"  Train: {len(df_train):,} pairs ({df_train['peptide'].nunique()} peptides, {df_train['allele'].nunique()} alleles)")
    print(f"  Val:   {len(df_val):,} pairs ({df_val['peptide'].nunique()} peptides, {df_val['allele'].nunique()} alleles)")
    print(f"  Test:  {len(df_test):,} pairs ({df_test['peptide'].nunique()} peptides, {df_test['allele'].nunique()} alleles)")

    # 2. Load ESMC Model & Tokenizer
    print(f"\n[Step 2/5] Loading '{model_name}' onto A100 GPU...")
    t0 = time.time()
    tokenizer = EsmcTokenizer.from_pretrained(model_name)
    # Load in bfloat16 for optimal memory and speed on Ampere A100
    esmc_model = EsmcModel.from_pretrained(model_name, dtype=torch.bfloat16, device="cuda").eval()
    d_model = esmc_model.config.d_model if hasattr(esmc_model.config, "d_model") else 2560
    load_time = time.time() - t0
    param_count = sum(p.numel() for p in esmc_model.parameters())
    print(f"  Loaded {model_name} in {load_time:.2f}s")
    print(f"  Parameters: {param_count:,} | Embedding Dimension (d_model): {d_model}")
    print(f"  GPU VRAM allocated after model load: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # 3. Two-Tower Feature Extraction (Cached by unique sequence)
    print(f"\n[Step 3/5] Extracting {d_model}-dim embeddings for all unique sequences...")
    all_peptides = list(set(df_train["peptide"]).union(set(df_val["peptide"])).union(set(df_test["peptide"])))
    all_hlas = list(set(df_train["hla_seq"]).union(set(df_val["hla_seq"])).union(set(df_test["hla_seq"])))
    print(f"  Unique peptides: {len(all_peptides):,} | Unique HLA sequences: {len(all_hlas):,}")

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
                pooled = (token_embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).float().cpu()
                for seq, emb in zip(batch, pooled):
                    embeddings[seq] = emb
        elapsed = time.time() - t_start
        rate = len(seq_list) / max(elapsed, 0.001)
        print(f"  -> Embedded {len(seq_list):,} {desc} in {elapsed:.2f}s ({rate:.1f} seq/s)")
        return embeddings

    # HLA: 182 residues, Peptide: 9 residues
    hla_batch = 16 if "6B" in model_name else 64
    pep_batch = 64 if "6B" in model_name else 256

    hla_embeddings = embed_sequences(all_hlas, batch_sz=hla_batch, desc="HLAs (182-aa)")
    pep_embeddings = embed_sequences(all_peptides, batch_sz=pep_batch, desc="Peptides (9-mer)")

    # Free ESMc transformer weights from GPU to maximize downstream memory & speed
    del esmc_model
    torch.cuda.empty_cache()
    print(f"  Freed transformer backbone from GPU. Current allocated VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # Assemble dataset tensors
    def df_to_tensors(df):
        p_tensors = torch.stack([pep_embeddings[p] for p in df["peptide"]])
        h_tensors = torch.stack([hla_embeddings[h] for h in df["hla_seq"]])
        y_reg = torch.tensor(np.log10(1.0 + df["thalf_hours"].values.astype(np.float32)))
        y_bin = torch.tensor((df["thalf_hours"].values >= 1.0).astype(np.float32))
        return p_tensors, h_tensors, y_reg, y_bin

    train_p, train_h, train_y_reg, train_y_bin = df_to_tensors(df_train)
    val_p, val_h, val_y_reg, val_y_bin = df_to_tensors(df_val)
    test_p, test_h, test_y_reg, test_y_bin = df_to_tensors(df_test)

    train_loader = DataLoader(
        TensorDataset(train_p, train_h, train_y_reg, train_y_bin),
        batch_size=batch_size,
        shuffle=True,
    )

    # 4. Deep Bilinear Multi-Task Interaction Head
    print("\n[Step 4/5] Initializing Deep Multi-Task Interaction Network...")
    class DeepInteractionNetwork(nn.Module):
        def __init__(self, d=d_model, h1=1024, h2=512, h3=128, dropout=0.25):
            super().__init__()
            in_dim = 4 * d
            self.input_norm = nn.LayerNorm(in_dim)
            self.proj = nn.Sequential(
                nn.Linear(in_dim, h1),
                nn.LayerNorm(h1),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.res_block = nn.Sequential(
                nn.Linear(h1, h1),
                nn.LayerNorm(h1),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.down = nn.Sequential(
                nn.Linear(h1, h2),
                nn.LayerNorm(h2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h2, h3),
                nn.GELU(),
            )
            self.reg_head = nn.Linear(h3, 1)
            self.cls_head = nn.Linear(h3, 1)

        def forward(self, e_pep, e_hla):
            diff = torch.abs(e_pep - e_hla)
            prod = e_pep * e_hla
            x = torch.cat([e_pep, e_hla, prod, diff], dim=-1)
            x = self.input_norm(x)
            h = self.proj(x)
            h = h + self.res_block(h)
            feats = self.down(h)
            reg_out = self.reg_head(feats).squeeze(-1)
            cls_out = self.cls_head(feats).squeeze(-1)
            return reg_out, cls_out

    model = DeepInteractionNetwork(d=d_model, h1=1024, h2=512, h3=128, dropout=0.25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    reg_criterion = nn.SmoothL1Loss(beta=0.1)  # Huber loss for robustness against stability outliers
    cls_criterion = nn.BCEWithLogitsLoss()

    print(f"  Interaction Head parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Full Training Loop with Checkpointing
    print(f"\n[Step 5/5] Training for {epochs} epochs with Multi-Task Loss...")
    print("-" * 76)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val MSE':<10} | {'Pearson r':<10} | {'Spearman ρ':<11} | {'AUROC (1h)':<10}")
    print("-" * 76)

    best_val_score = -1.0
    best_weights = None
    train_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bp, bh, by_reg, by_bin in train_loader:
            bp, bh = bp.to(device), bh.to(device)
            by_reg, by_bin = by_reg.to(device), by_bin.to(device)
            optimizer.zero_grad()
            pred_reg, pred_cls = model(bp, bh)
            loss_reg = reg_criterion(pred_reg, by_reg)
            loss_cls = cls_criterion(pred_cls, by_bin)
            loss = loss_reg + 0.3 * loss_cls
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(by_reg)
        total_loss /= len(train_p)
        scheduler.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_preds_list = []
            val_cls_list = []
            for i in range(0, len(val_p), 512):
                bp = val_p[i : i + 512].to(device)
                bh = val_h[i : i + 512].to(device)
                pr, pc = model(bp, bh)
                val_preds_list.append(pr.cpu())
                val_cls_list.append(torch.sigmoid(pc).cpu())
            val_preds = torch.cat(val_preds_list).numpy()
            val_cls = torch.cat(val_cls_list).numpy()

        val_true = val_y_reg.numpy()
        val_mse = float(np.mean((val_preds - val_true) ** 2))
        val_pearson, _ = pearsonr(val_preds, val_true)
        val_spearman, _ = spearmanr(val_preds, val_true)
        val_auroc_1h = float(roc_auc_score(val_y_bin.numpy(), val_preds))
        val_pr_auc_1h = float(average_precision_score(val_y_bin.numpy(), val_preds))

        # Checkpoint based on combined ranking & correlation metric
        val_score = val_spearman + val_pearson
        if val_score > best_val_score:
            best_val_score = val_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"{epoch:<6} | {total_loss:<10.4f} | {val_mse:<10.4f} | {val_pearson:<10.4f} | {val_spearman:<11.4f} | {val_auroc_1h:<10.4f}")

        train_history.append({
            "epoch": epoch,
            "train_loss": total_loss,
            "val_mse": val_mse,
            "val_pearson": float(val_pearson),
            "val_spearman": float(val_spearman),
            "val_auroc_1h": val_auroc_1h,
            "val_pr_auc_1h": val_pr_auc_1h,
        })

    # Restore best checkpoint for final test evaluation
    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()

    # Final Evaluation on Held-Out Test Set
    print("\n" + "=" * 76)
    print(f"FINAL EVALUATION ON HELD-OUT TEST SET ({split_name.upper()})")
    print("=" * 76)
    with torch.no_grad():
        test_preds_list = []
        test_cls_list = []
        for i in range(0, len(test_p), 512):
            bp = test_p[i : i + 512].to(device)
            bh = test_h[i : i + 512].to(device)
            pr, pc = model(bp, bh)
            test_preds_list.append(pr.cpu())
            test_cls_list.append(torch.sigmoid(pc).cpu())
        test_preds = torch.cat(test_preds_list).numpy()
        test_cls = torch.cat(test_cls_list).numpy()

    test_true_log = test_y_reg.numpy()
    test_true_raw = df_test["thalf_hours"].values.astype(np.float32)
    # Convert log-predicted back to hours: 10^(log_pred) - 1
    test_pred_raw = np.maximum(0.0, (10.0 ** test_preds) - 1.0)

    # 1h and 2h binary ground truths
    test_bin_1h = (test_true_raw >= 1.0).astype(int)
    test_bin_2h = (test_true_raw >= 2.0).astype(int)

    test_pearson_log, _ = pearsonr(test_preds, test_true_log)
    test_pearson_raw, _ = pearsonr(test_pred_raw, test_true_raw)
    test_spearman, _ = spearmanr(test_preds, test_true_log)
    test_rmse_log = float(np.sqrt(mean_squared_error(test_true_log, test_preds)))
    test_rmse_raw = float(np.sqrt(mean_squared_error(test_true_raw, test_pred_raw)))
    test_mae_raw = float(mean_absolute_error(test_true_raw, test_pred_raw))
    test_auroc_1h = float(roc_auc_score(test_bin_1h, test_preds))
    test_prauc_1h = float(average_precision_score(test_bin_1h, test_preds))
    test_auroc_2h = float(roc_auc_score(test_bin_2h, test_preds))
    test_prauc_2h = float(average_precision_score(test_bin_2h, test_preds))

    print(f"Pearson r (log scale):  {test_pearson_log:.4f}")
    print(f"Pearson r (raw scale):  {test_pearson_raw:.4f}")
    print(f"Spearman ρ:             {test_spearman:.4f}")
    print(f"RMSE (log scale):       {test_rmse_log:.4f}")
    print(f"RMSE (hours):           {test_rmse_raw:.4f}")
    print(f"MAE (hours):            {test_mae_raw:.4f}")
    print(f"AUROC (>= 1.0h binder): {test_auroc_1h:.4f}")
    print(f"PR-AUC (>= 1.0h):       {test_prauc_1h:.4f}")
    print(f"AUROC (>= 2.0h binder): {test_auroc_2h:.4f}")
    print(f"PR-AUC (>= 2.0h):       {test_prauc_2h:.4f}")
    print("=" * 76)

    # Build predictions dataframe records
    pred_records = []
    for idx, row in df_test.iterrows():
        pred_records.append({
            "allele": row["allele"],
            "peptide": row["peptide"],
            "thalf_hours": float(row["thalf_hours"]),
            "log_thalf_actual": float(test_true_log[idx]),
            "log_thalf_pred": float(test_preds[idx]),
            "thalf_hours_pred": float(test_pred_raw[idx]),
            "pred_binder_prob": float(test_cls[idx]),
            "abs_error_hours": float(abs(test_pred_raw[idx] - float(row["thalf_hours"]))),
        })

    test_metrics = {
        "pearson_r_log": float(test_pearson_log),
        "pearson_r_raw": float(test_pearson_raw),
        "spearman_rho": float(test_spearman),
        "rmse_log": test_rmse_log,
        "rmse_raw": test_rmse_raw,
        "mae_raw": test_mae_raw,
        "auc_roc_1h": test_auroc_1h,
        "pr_auc_1h": test_prauc_1h,
        "auc_roc_2h": test_auroc_2h,
        "pr_auc_2h": test_prauc_2h,
    }

    return {
        "model_name": model_name,
        "split_name": split_name,
        "d_model": d_model,
        "epochs": epochs,
        "train_history": train_history,
        "test_metrics": test_metrics,
        "pred_records": pred_records,
    }


# -----------------------------------------------------------------------------
# 3. CLI ENTRYPOINT: ORCHESTRATION & PERSISTENCE
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    model: str = "biohub/ESMC-6B",
    split: str = "both",
    epochs: int = 15,
):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "esmc_metrics_summary.json"

    # Load existing summary if present
    summary = {}
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    splits_to_run = ["iid", "novel_allele"] if split == "both" else [split]
    short_model = "esmc_6b" if "6B" in model else "esmc_300m"

    for s in splits_to_run:
        print(f"\n>>> Running Modal A100 training for {model} on '{s}' split ({epochs} epochs)...")
        res = run_benchmark_on_split.remote(
            model_name=model,
            split_name=s,
            epochs=epochs,
        )

        # 1. Save predictions CSV
        preds_df = pd.DataFrame(res["pred_records"])
        preds_file = results_dir / f"predictions_{short_model}_{s}.csv"
        preds_df.to_csv(preds_file, index=False)
        print(f"Saved predictions to: {preds_file} ({len(preds_df):,} rows)")

        # 2. Update metrics summary JSON
        if s not in summary:
            summary[s] = {}
        summary[s][short_model] = {
            "model_name": model,
            "d_model": res["d_model"],
            "epochs": epochs,
            "metrics": res["test_metrics"],
            "train_history": res["train_history"],
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Updated metrics summary in: {summary_path}")

    print("\nModal benchmark run complete! All predictions and metrics saved locally in results/.")
    try:
        sys.path.append(str(Path(__file__).resolve().parent))
        from plot_esmc_results import main as generate_plots
        print("\n>>> Generating publication-quality plots...")
        generate_plots()
    except Exception as e:
        print(f"Plot generation notice: {e}")
