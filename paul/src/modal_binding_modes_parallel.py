"""
Parallel Modal A100 Training Pipeline for Peptide-HLA Stability across Binding Modes.

Trains 3 specialized foundation-head models in parallel on NVIDIA A100 GPUs:
1. Non-binding:     thalf_hours < 1.0h
2. Mildly binding:  1.0h <= thalf_hours < 2.0h
3. Strong binders:  thalf_hours >= 2.0h

All models are trained on strictly disjoint PEPTIDE-LEVEL splits (zero peptide leakage)
and evaluated on held-out peptides, as well as cross-evaluated across all binding modes.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any

import modal
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. MODAL APP & CONTAINER DEFINITION
# -----------------------------------------------------------------------------
app = modal.App("peptide-hla-binding-modes")

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
    timeout=1800,  # 30 minutes
)
def train_binding_mode_model(
    mode_name: str = "strong_binding",
    model_name: str = "biohub/ESMC-6B",
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    import gc
    import time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from esm.models.esmc import EsmcModel, EsmcTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_mem_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    print("=" * 78)
    print(f"MODAL WORKER: TRAINING MODEL FOR BINDING MODE '{mode_name.upper()}'")
    print("=" * 78)
    print(f"Device: {device} ({gpu_name}) | Total VRAM: {total_mem_gb:.2f} GB")
    print(f"Backbone Model: {model_name} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 78)

    t_start_total = time.time()

    # 1. Load Data Splits
    splits_dir = Path("/root/splits")
    train_file = splits_dir / f"{mode_name}_train.csv"
    val_file = splits_dir / f"{mode_name}_val.csv"
    test_file = splits_dir / f"{mode_name}_test.csv"

    if not train_file.exists():
        train_file = splits_dir / "binding_modes" / f"{mode_name}_train.csv"
        val_file = splits_dir / "binding_modes" / f"{mode_name}_val.csv"
        test_file = splits_dir / "binding_modes" / f"{mode_name}_test.csv"

    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    print(f"\n[Step 1/5] Loaded data splits for {mode_name}:")
    print(f"  Train: {len(df_train):,} pairs ({df_train['peptide'].nunique()} unique peptides, {df_train['allele'].nunique()} alleles)")
    print(f"  Val:   {len(df_val):,} pairs ({df_val['peptide'].nunique()} unique peptides, {df_val['allele'].nunique()} alleles)")
    print(f"  Test:  {len(df_test):,} pairs ({df_test['peptide'].nunique()} unique peptides, {df_test['allele'].nunique()} alleles)")

    # Also load the test sets for the other two modes for cross-evaluation
    all_modes = ["non_binding", "mild_binding", "strong_binding"]
    cross_test_dfs = {}
    for other_mode in all_modes:
        o_path = splits_dir / f"{other_mode}_test.csv"
        if not o_path.exists():
            o_path = splits_dir / "binding_modes" / f"{other_mode}_test.csv"
        cross_test_dfs[other_mode] = pd.read_csv(o_path)

    # 2. Collect All Unique Peptides & HLAs to Embed
    all_peptides = set(df_train["peptide"]).union(set(df_val["peptide"])).union(set(df_test["peptide"]))
    all_hlas = set(df_train["hla_seq"]).union(set(df_val["hla_seq"])).union(set(df_test["hla_seq"]))
    for o_df in cross_test_dfs.values():
        all_peptides.update(o_df["peptide"])
        all_hlas.update(o_df["hla_seq"])

    unique_peptides = sorted(list(all_peptides))
    unique_hlas = sorted(list(all_hlas))
    print(f"\n[Step 2/5] Sequence Cache Requirements:")
    print(f"  Total unique peptides to embed: {len(unique_peptides):,}")
    print(f"  Total unique HLAs to embed:     {len(unique_hlas):,}")

    # 3. Load Backbone Foundation Model & Extract Embeddings
    print(f"\n[Step 3/5] Loading '{model_name}' onto GPU...")
    t0 = time.time()
    tokenizer = EsmcTokenizer.from_pretrained(model_name)
    esmc_model = EsmcModel.from_pretrained(model_name, dtype=torch.bfloat16, device="cuda").eval()
    d_model = esmc_model.config.d_model if hasattr(esmc_model.config, "d_model") else 2560
    print(f"  Loaded {model_name} in {time.time() - t0:.2f}s | d_model={d_model}")

    def embed_seqs(seq_list, batch_sz=64, desc="sequences"):
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
                for s, emb in zip(batch, pooled):
                    embeddings[s] = emb
        elapsed = time.time() - t_start
        print(f"  -> Embedded {len(seq_list):,} {desc} in {elapsed:.2f}s ({len(seq_list)/max(elapsed, 0.001):.1f} seq/s)")
        return embeddings

    hla_batch = 16 if "6B" in model_name else 64
    pep_batch = 64 if "6B" in model_name else 256

    hla_embs = embed_seqs(unique_hlas, batch_sz=hla_batch, desc="HLAs")
    pep_embs = embed_seqs(unique_peptides, batch_sz=pep_batch, desc="Peptides")

    # Free Backbone from GPU memory
    del esmc_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Freed foundation model from GPU. Active VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # 4. Prepare PyTorch Datasets
    def df_to_tensors(df):
        p_tensors = torch.stack([pep_embs[p] for p in df["peptide"]])
        h_tensors = torch.stack([hla_embs[h] for h in df["hla_seq"]])
        # Target: log10(1 + thalf_hours)
        y_reg = torch.tensor(np.log10(1.0 + np.clip(df["thalf_hours"].values.astype(np.float32), 0, None)))
        return p_tensors, h_tensors, y_reg

    train_p, train_h, train_y = df_to_tensors(df_train)
    val_p, val_h, val_y = df_to_tensors(df_val)
    test_p, test_h, test_y = df_to_tensors(df_test)

    train_loader = DataLoader(
        TensorDataset(train_p, train_h, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    # 5. Define Model Architecture & Loss
    print(f"\n[Step 4/5] Initializing Deep Interaction Network for {mode_name}...")
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

        def forward(self, e_pep, e_hla):
            diff = torch.abs(e_pep - e_hla)
            prod = e_pep * e_hla
            x = torch.cat([e_pep, e_hla, prod, diff], dim=-1)
            x = self.input_norm(x)
            h = self.proj(x)
            h = h + self.res_block(h)
            feats = self.down(h)
            return self.reg_head(feats).squeeze(-1)

    model = DeepInteractionNetwork(d=d_model, h1=1024, h2=512, h3=128, dropout=0.25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    reg_criterion = nn.SmoothL1Loss(beta=0.05)

    def compute_pearson_loss(p, t):
        if len(p) < 4:
            return torch.tensor(0.0, device=device)
        p_c = p - torch.mean(p)
        t_c = t - torch.mean(t)
        var_p = torch.sum(p_c ** 2)
        var_t = torch.sum(t_c ** 2)
        if var_p < 1e-7 or var_t < 1e-7:
            return torch.tensor(0.0, device=device)
        r = torch.sum(p_c * t_c) / torch.sqrt(var_p * var_t + 1e-8)
        return 1.0 - torch.clamp(r, -1.0, 1.0)

    # 6. Training Loop with Epoch History
    print(f"\n[Step 5/5] Training for {epochs} epochs...")
    print("-" * 78)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val MSE':<10} | {'Pearson r':<12} | {'Spearman ρ':<12}")
    print("-" * 78)

    best_val_score = -999.0
    best_weights = None
    train_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for bp, bh, by in train_loader:
            bp, bh, by = bp.to(device), bh.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bp, bh)
            loss_reg = reg_criterion(preds, by)
            loss_pearson = compute_pearson_loss(preds, by)
            loss = loss_reg + 0.4 * loss_pearson
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(by)
        epoch_loss /= len(train_p)
        scheduler.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_preds_list = []
            for i in range(0, len(val_p), 512):
                bp = val_p[i : i + 512].to(device)
                bh = val_h[i : i + 512].to(device)
                val_preds_list.append(model(bp, bh).cpu())
            val_preds = torch.cat(val_preds_list).numpy()

        val_true = val_y.numpy()
        val_mse = float(np.mean((val_preds - val_true) ** 2))
        
        # Guard against zero variance
        if np.std(val_preds) > 1e-6 and np.std(val_true) > 1e-6:
            val_pearson = float(pearsonr(val_preds, val_true)[0])
            val_spearman = float(spearmanr(val_preds, val_true)[0])
        else:
            val_pearson = 0.0
            val_spearman = 0.0

        val_score = val_pearson + val_spearman
        if val_score > best_val_score:
            best_val_score = val_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"{epoch:<6} | {epoch_loss:<12.4f} | {val_mse:<10.4f} | {val_pearson:<12.4f} | {val_spearman:<12.4f}")

        train_history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_mse": val_mse,
            "val_pearson": val_pearson,
            "val_spearman": val_spearman,
        })

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()

    # 7. Helper Evaluation Function
    def evaluate_on_dataframe(df_eval, desc="Test"):
        e_p, e_h, e_y = df_to_tensors(df_eval)
        with torch.no_grad():
            preds_list = []
            for i in range(0, len(e_p), 512):
                bp = e_p[i : i + 512].to(device)
                bh = e_h[i : i + 512].to(device)
                preds_list.append(model(bp, bh).cpu())
            preds_log = torch.cat(preds_list).numpy()

        true_log = e_y.numpy()
        true_raw = df_eval["thalf_hours"].values
        # Invert log10(1 + thalf)
        preds_raw = np.clip(10.0 ** preds_log - 1.0, 0.0, None)

        mse_log = float(mean_squared_error(true_log, preds_log))
        rmse_log = float(np.sqrt(mse_log))
        mae_raw = float(mean_absolute_error(true_raw, preds_raw))
        rmse_raw = float(np.sqrt(mean_squared_error(true_raw, preds_raw)))

        if np.std(preds_log) > 1e-6 and np.std(true_log) > 1e-6:
            r_log = float(pearsonr(preds_log, true_log)[0])
            rho = float(spearmanr(preds_log, true_log)[0])
        else:
            r_log = 0.0
            rho = 0.0

        if np.std(preds_raw) > 1e-6 and np.std(true_raw) > 1e-6:
            r_raw = float(pearsonr(preds_raw, true_raw)[0])
        else:
            r_raw = 0.0

        return {
            "pearson_r_log": r_log,
            "spearman_rho": rho,
            "pearson_r_raw": r_raw,
            "rmse_log": rmse_log,
            "mse_log": mse_log,
            "mae_raw": mae_raw,
            "rmse_raw": rmse_raw,
            "mean_pred_thalf": float(np.mean(preds_raw)),
            "mean_true_thalf": float(np.mean(true_raw)),
            "preds_log": preds_log.tolist(),
            "preds_raw": preds_raw.tolist(),
        }

    # 8. Own Held-Out Test Set Evaluation
    print(f"\n[Evaluation] Evaluating best model on own test set ({mode_name})...")
    own_test_metrics = evaluate_on_dataframe(df_test, desc=f"{mode_name} Own Test")
    print(f"  Test Pearson r (log):  {own_test_metrics['pearson_r_log']:.4f}")
    print(f"  Test Spearman ρ:       {own_test_metrics['spearman_rho']:.4f}")
    print(f"  Test RMSE (log):       {own_test_metrics['rmse_log']:.4f}")
    print(f"  Test MAE (raw hours):  {own_test_metrics['mae_raw']:.4f}h")

    # 9. Cross-Mode Evaluations
    print(f"\n[Evaluation] Cross-evaluating across all binding modes...")
    cross_metrics = {}
    for other_mode, other_df in cross_test_dfs.items():
        res = evaluate_on_dataframe(other_df, desc=f"{other_mode} Test")
        cross_metrics[other_mode] = {
            "pearson_r_log": res["pearson_r_log"],
            "spearman_rho": res["spearman_rho"],
            "rmse_log": res["rmse_log"],
            "mae_raw": res["mae_raw"],
            "mean_pred_thalf": res["mean_pred_thalf"],
            "mean_true_thalf": res["mean_true_thalf"],
        }
        print(f"  Tested on {other_mode:<15} test set: Pearson r={res['pearson_r_log']:.4f} | Spearman ρ={res['spearman_rho']:.4f} | Mean Pred={res['mean_pred_thalf']:.2f}h (True: {res['mean_true_thalf']:.2f}h)")

    # Prepare return records for own test set
    df_test_out = df_test.copy()
    df_test_out["pred_log_thalf"] = own_test_metrics["preds_log"]
    df_test_out["pred_thalf_hours"] = own_test_metrics["preds_raw"]
    df_test_out["trained_on_mode"] = mode_name

    total_time = time.time() - t_start_total

    result_payload = {
        "mode_name": mode_name,
        "model_name": model_name,
        "device": str(gpu_name),
        "total_vram_gb": total_mem_gb,
        "epochs": epochs,
        "total_time_seconds": total_time,
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
        "own_test_metrics": {
            "pearson_r_log": own_test_metrics["pearson_r_log"],
            "spearman_rho": own_test_metrics["spearman_rho"],
            "pearson_r_raw": own_test_metrics["pearson_r_raw"],
            "rmse_log": own_test_metrics["rmse_log"],
            "mse_log": own_test_metrics["mse_log"],
            "mae_raw": own_test_metrics["mae_raw"],
            "rmse_raw": own_test_metrics["rmse_raw"],
            "mean_pred_thalf": own_test_metrics["mean_pred_thalf"],
            "mean_true_thalf": own_test_metrics["mean_true_thalf"],
        },
        "cross_eval_metrics": cross_metrics,
        "train_history": train_history,
    }

    # Backup to volume
    try:
        with open(f"/root/.cache/huggingface/binding_mode_{mode_name}_metrics.json", "w") as f:
            json.dump(result_payload, f, indent=2)
        df_test_out.to_csv(f"/root/.cache/huggingface/binding_mode_{mode_name}_preds.csv", index=False)
        hf_volume.commit()
    except Exception as e:
        print(f"  [Volume Backup Warning] {e}")

    print("=" * 78)
    print(f"Worker for '{mode_name}' completed in {total_time:.2f}s!")
    print("=" * 78)
    return result_payload, df_test_out.to_dict(orient="records")


# -----------------------------------------------------------------------------
# 3. LOCAL ENTRYPOINT: PARALLEL DISPATCH ACROSS ALL BINDING MODES
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 3e-4,
    model_name: str = "biohub/ESMC-6B",
):
    modes = ["non_binding", "mild_binding", "strong_binding"]
    t0_all = time.time()

    print("=" * 80)
    print("🚀 DISPATCHING PARALLEL MODAL A100 TRAINING FOR ALL 3 BINDING MODES")
    print("=" * 80)
    print(f"Target Modes: {modes}")
    print(f"Foundation Backbone: {model_name}")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print("=" * 80)

    # Launch all 3 workers concurrently in parallel via spawn()
    futures = []
    for mode in modes:
        print(f"  -> Spawning Modal A100 GPU worker for mode: '{mode}'...")
        future = train_binding_mode_model.spawn(
            mode_name=mode,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        futures.append((mode, future))

    print("\n⏳ All 3 workers spawned! Concurrently executing on NVIDIA A100 GPUs in Modal cloud...")
    
    # Collect results as they finish
    results_map = {}
    preds_dfs = {}

    for mode, future in futures:
        print(f"Waiting for worker '{mode}' to complete...")
        payload, records = future.get()
        results_map[mode] = payload
        preds_dfs[mode] = pd.DataFrame(records)
        print(f"  ✅ Completed '{mode}' in {payload['total_time_seconds']:.1f}s: Pearson r = {payload['own_test_metrics']['pearson_r_log']:.4f}, Spearman ρ = {payload['own_test_metrics']['spearman_rho']:.4f}")

    total_parallel_time = time.time() - t0_all
    print("\n" + "=" * 80)
    print(f"🎉 ALL PARALLEL TRAINING JOBS COMPLETED IN {total_parallel_time:.1f}s!")
    print("=" * 80)

    # 4. Save Artifacts Locally
    results_dir = PROJECT_ROOT / "results" / "binding_modes"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save individual prediction CSVs
    for mode, df_pred in preds_dfs.items():
        out_csv = results_dir / f"predictions_{mode}.csv"
        df_pred.to_csv(out_csv, index=False)
        print(f"Saved predictions to: {out_csv} ({len(df_pred):,} rows)")

    # Save summary metrics JSON
    summary_path = results_dir / "binding_modes_metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_map, f, indent=2)
    print(f"Saved comprehensive metrics summary to: {summary_path}")

    # Build and print Cross-Evaluation Matrix
    print("\n" + "=" * 80)
    print("CROSS-EVALUATION SUMMARY MATRIX (Pearson r [log10] / Spearman ρ)")
    print("=" * 80)
    header = f"{'Trained On Mode':<18} | {'Test: Non-Binding':<20} | {'Test: Mild-Binding':<20} | {'Test: Strong-Binding':<20}"
    print(header)
    print("-" * len(header))

    cross_matrix_data = []
    for train_mode in modes:
        row_str = f"{train_mode:<18} | "
        matrix_row = {"trained_on": train_mode}
        for eval_mode in modes:
            cm = results_map[train_mode]["cross_eval_metrics"][eval_mode]
            cell = f"r={cm['pearson_r_log']:.3f}, ρ={cm['spearman_rho']:.3f}"
            matrix_row[f"test_{eval_mode}_pearson_r"] = cm["pearson_r_log"]
            matrix_row[f"test_{eval_mode}_spearman_rho"] = cm["spearman_rho"]
            matrix_row[f"test_{eval_mode}_mean_pred"] = cm["mean_pred_thalf"]
            row_str += f"{cell:<20} | "
        print(row_str)
        cross_matrix_data.append(matrix_row)

    cross_df = pd.DataFrame(cross_matrix_data)
    cross_df.to_csv(results_dir / "cross_evaluation_matrix.csv", index=False)
    print(f"\nSaved cross-evaluation matrix to: {results_dir / 'cross_evaluation_matrix.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    pass
