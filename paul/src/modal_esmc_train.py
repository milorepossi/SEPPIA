"""
Modal A100-40GB Training Pipeline for ESMC-300M on Peptide-HLA Stability.
Leverages biohub/ESMC-300M foundation model embeddings on NVIDIA A100-40GB GPU.
"""

from pathlib import Path
import modal

# -----------------------------------------------------------------------------
# 1. MODAL APP & CONTAINER IMAGE DEFINITION
# -----------------------------------------------------------------------------
app = modal.App("esmc-300m-a100-train")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = str(PROJECT_ROOT / "splits")
DATA_DIR = str(PROJECT_ROOT / "data")

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
    timeout=900,  # 15 minutes
)
def run_training_experiment(
    split_name: str = "iid",
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    test_gradient_backprop: bool = True,
):
    import os
    import time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score
    from esm.models.esmc import EsmcModel, EsmcTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0

    print("=" * 70)
    print("MODAL A100-40GB ESMC-300M TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device: {device} ({gpu_name})")
    print(f"Total VRAM: {total_mem_gb:.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("=" * 70)

    # 1. Load ESMC-300M Model & Tokenizer
    print("\n[Step 1/5] Loading 'biohub/ESMC-300M' onto A100 GPU...")
    t0 = time.time()
    tokenizer = EsmcTokenizer.from_pretrained("biohub/ESMC-300M")
    esmc_model = EsmcModel.from_pretrained("biohub/ESMC-300M").to(device)
    print(f"ESMC-300M loaded successfully in {time.time() - t0:.2f}s")
    print(f"ESMC Parameters: {sum(p.numel() for p in esmc_model.parameters()):,}")

    # 2. Test Gradient Backprop (Verifying fine-tuning / backward capability)
    if test_gradient_backprop:
        print("\n[Step 2/5] Verifying gradient backprop through ESMC-300M on A100...")
        esmc_model.train()
        sample_tokens = tokenizer(["VTTEVAFGL", "GSHSMRYFFT"], return_tensors="pt", padding=True).to(device)
        out = esmc_model(**sample_tokens)
        dummy_loss = out.last_hidden_state.sum()
        dummy_loss.backward()
        grad_norm = torch.norm(torch.stack([p.grad.norm() for p in esmc_model.parameters() if p.grad is not None]))
        print(f"Gradient backward pass succeeded! Total Gradient Norm: {grad_norm.item():.4f}")
        esmc_model.zero_grad()
        esmc_model.eval()

    # 3. Load Splits Data
    print(f"\n[Step 3/5] Loading dataset splits for regime: '{split_name}'...")
    train_csv = f"/root/splits/{split_name}_train.csv"
    val_csv = f"/root/splits/{split_name}_val.csv"
    
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing {train_csv}")
    
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    print(f"Train samples: {len(df_train):,} (Unique peptides: {df_train['peptide'].nunique()}, Unique HLAs: {df_train['hla_seq'].nunique()})")
    print(f"Val samples:   {len(df_val):,} (Unique peptides: {df_val['peptide'].nunique()}, Unique HLAs: {df_val['hla_seq'].nunique()})")

    # 4. Two-Tower Feature Extraction (Massive Efficiency Cache)
    print("\n[Step 4/5] Extracting ESMC-300M embeddings for unique peptides & HLAs...")
    all_peptides = list(set(df_train["peptide"]).union(set(df_val["peptide"])))
    all_hlas = list(set(df_train["hla_seq"]).union(set(df_val["hla_seq"])))
    print(f"Total unique peptides to embed: {len(all_peptides):,}")
    print(f"Total unique HLAs to embed:     {len(all_hlas):,}")

    def embed_sequences(seq_list, batch_sz=128, desc="Sequences"):
        embeddings = {}
        t_start = time.time()
        with torch.no_grad():
            for i in range(0, len(seq_list), batch_sz):
                batch = seq_list[i : i + batch_sz]
                tokens = tokenizer(batch, return_tensors="pt", padding=True).to(device)
                out = esmc_model(**tokens)
                # Compute mean-pooled embedding across tokens
                mask = tokens.attention_mask.unsqueeze(-1)
                token_embs = out.last_hidden_state * mask
                pooled = (token_embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).cpu()
                for seq, emb in zip(batch, pooled):
                    embeddings[seq] = emb
        elapsed = time.time() - t_start
        print(f"  Embedded {len(seq_list):,} {desc} in {elapsed:.2f}s ({len(seq_list)/max(elapsed, 0.001):.1f} seq/s)")
        return embeddings

    hla_embeddings = embed_sequences(all_hlas, batch_sz=64, desc="HLA full-domain sequences")
    pep_embeddings = embed_sequences(all_peptides, batch_sz=256, desc="9-mer peptides")

    # Build Tensors for PyTorch DataLoader
    # Target: log10(1 + thalf_hours)
    def prepare_tensors(df):
        e_pep = torch.stack([pep_embeddings[p] for p in df["peptide"]])
        e_hla = torch.stack([hla_embeddings[h] for h in df["hla_seq"]])
        # Log-transformed continuous target
        y_reg = torch.tensor(np.log10(1.0 + df["thalf_hours"].values.astype(np.float32)))
        # Binary target (t_half >= 1.0 hr)
        y_bin = torch.tensor((df["thalf_hours"].values >= 1.0).astype(np.float32))
        return e_pep, e_hla, y_reg, y_bin

    train_p, train_h, train_y_reg, train_y_bin = prepare_tensors(df_train)
    val_p, val_h, val_y_reg, val_y_bin = prepare_tensors(df_val)

    train_loader = DataLoader(
        TensorDataset(train_p, train_h, train_y_reg),
        batch_size=batch_size,
        shuffle=True,
    )

    # 5. Train Downstream Interaction Head
    print("\n[Step 5/5] Training Bilinear Interaction Head on ESMC-300M representations...")
    class PeptideHLAInteractionHead(nn.Module):
        """
        Deep bilinear interaction network combining peptide and HLA embeddings.
        Input features: [e_pep, e_hla, e_pep * e_hla, |e_pep - e_hla|] (dim: 4 * d)
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

    model = PeptideHLAInteractionHead(d=960, h=512, p=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    metrics_history = []
    print("-" * 70)
    print(f"{'Epoch':<6} | {'Train MSE':<10} | {'Val MSE':<10} | {'Pearson r':<10} | {'Spearman ρ':<11} | {'AUROC (1h)':<10}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for b_pep, b_hla, b_y in train_loader:
            b_pep, b_hla, b_y = b_pep.to(device), b_hla.to(device), b_y.to(device)
            optimizer.zero_grad()
            preds = model(b_pep, b_hla)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_y)
        train_loss /= len(train_p)
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds_list = []
            val_batch_size = 512
            for i in range(0, len(val_p), val_batch_size):
                bp = val_p[i : i + val_batch_size].to(device)
                bh = val_h[i : i + val_batch_size].to(device)
                val_preds_list.append(model(bp, bh).cpu())
            val_preds = torch.cat(val_preds_list).numpy()

        val_true = val_y_reg.numpy()
        val_mse = float(np.mean((val_preds - val_true) ** 2))
        val_pearson, _ = pearsonr(val_preds, val_true)
        val_spearman, _ = spearmanr(val_preds, val_true)

        # Calculate AUROC for binary classification using continuous predicted stability
        try:
            val_auroc = float(roc_auc_score(val_y_bin.numpy(), val_preds))
        except Exception:
            val_auroc = float("nan")

        print(f"{epoch:<6} | {train_loss:<10.4f} | {val_mse:<10.4f} | {val_pearson:<10.4f} | {val_spearman:<11.4f} | {val_auroc:<10.4f}")

        metrics_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_mse,
            "val_pearson": float(val_pearson),
            "val_spearman": float(val_spearman),
            "val_auroc": val_auroc,
        })

    print("-" * 70)
    print("\nTraining completed successfully on Modal A100-40GB!")
    final_metrics = metrics_history[-1]
    print(f"Final Validation Pearson r:  {final_metrics['val_pearson']:.4f}")
    print(f"Final Validation Spearman ρ: {final_metrics['val_spearman']:.4f}")
    print(f"Final Validation AUROC (1h): {final_metrics['val_auroc']:.4f}")
    print("=" * 70)

    return {
        "status": "success",
        "device": str(gpu_name),
        "total_vram_gb": total_mem_gb,
        "epochs": epochs,
        "final_metrics": final_metrics,
        "history": metrics_history,
    }


# -----------------------------------------------------------------------------
# 4. CLI LOCAL ENTRYPOINT
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main(split: str = "iid", epochs: int = 5):
    print(f"Starting Modal A100 training job on split='{split}' for {epochs} epochs...")
    res = run_training_experiment.remote(split_name=split, epochs=epochs)
    print("Modal Remote Job Result:")
    print(res)
