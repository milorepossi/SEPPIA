"""
Modal Training Pipeline for End-to-End Fine-Tuning of ESMC-600M on Peptide-HLA Complex Stability.

Fine-tunes the 600-million parameter EvolutionaryScale ESM-Cambrian foundation model (biohub/ESMC-600M)
on the IID split of peptide-HLA half-life stability data using an NVIDIA A100-40GB GPU.

Key Architecture and Fine-Tuning Strategy:
1. Joint Complex Tokenization:
   Peptide (9-mer) and HLA extracellular domain (182-aa cleft) are joint-encoded using the native
   chain separator token '|' (f"{peptide}|{hla_seq}").
   Zero padding is required: every complex is exactly 194 tokens (<cls>, 9 pep, |, 182 hla, <eos>).
2. Deep Quaternary Self-Attention:
   All 36 transformer layers perform bidirectional self-attention between peptide residues and
   HLA groove positions, allowing dynamic binding-pocket adaptation and induced-fit modeling.
3. Progressive Layer Unfreezing:
   Unfreezes the top K transformer blocks (e.g. top 6 or 8 layers) and final LayerNorm for stable,
   sample-efficient adaptation without catastrophic forgetting, with support for full backbone fine-tuning.
4. Biophysical Stability Readout Head:
   Integrates <cls> global complex context, mean-pooled peptide, mean-pooled HLA groove,
   bilinear interaction terms (element-wise product and absolute difference), and explicit
   anchor residue states (P2 B-pocket anchor and P9 F-pocket anchor).
5. Multi-Objective Stability Loss:
   L = L_SmoothL1 (magnitude calibration) + 0.6 * L_Pearson (direct correlation optimization)
     + 0.2 * L_Rank (soft-margin pairwise order) + 0.3 * L_BCE (1-hour clinical stability).
6. Differential Learning Rates:
   Backbone lr = 2e-5, Head lr = 3e-4 with 1-epoch warmup and cosine decay.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import modal

# -----------------------------------------------------------------------------
# 1. MODAL APP & CONTAINER DEFINITION
# -----------------------------------------------------------------------------
app = modal.App("esmc-600m-finetune-iid")

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
# 2. MODAL REMOTE FINE-TUNING FUNCTION (NVIDIA A100-40GB)
# -----------------------------------------------------------------------------
@app.function(
    gpu="A100-40GB",
    image=esmc_image,
    volumes={"/root/.cache/huggingface": hf_volume},
    timeout=2400,  # 40 minutes
)
def run_esmc_finetuning(
    model_name: str = "biohub/ESMC-600M",
    epochs: int = 12,
    batch_size: int = 32,
    lr_backbone: float = 2e-5,
    lr_head: float = 3e-4,
    weight_decay: float = 1e-4,
    unfreeze_layers: int = 6,
    d_head: int = 512,
    dropout: float = 0.15,
):
    import time
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
    total_mem_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    print("=" * 80)
    print("END-TO-END FINE-TUNING: ESMC-600M ON PEPTIDE-HLA STABILITY (IID SPLIT)")
    print("=" * 80)
    print(f"Compute Device: {device} ({gpu_name}) | VRAM: {total_mem_gb:.2f} GB")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    print(f"Config: epochs={epochs}, batch_size={batch_size}, lr_backbone={lr_backbone}, lr_head={lr_head}")
    print(f"Fine-tuning regime: unfreeze_layers={unfreeze_layers} (top {unfreeze_layers} of 36 transformer blocks)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Load Dataset Splits
    # -------------------------------------------------------------------------
    splits_dir = Path("/root/splits")
    df_train = pd.read_csv(splits_dir / "iid_train.csv")
    df_val = pd.read_csv(splits_dir / "iid_val.csv")
    df_test = pd.read_csv(splits_dir / "iid_test.csv")

    print(f"\n[Step 1/5] Loaded IID Dataset Splits:")
    print(f"  Train: {len(df_train):,} pairs ({df_train['peptide'].nunique()} peptides, {df_train['allele'].nunique()} alleles)")
    print(f"  Val:   {len(df_val):,} pairs ({df_val['peptide'].nunique()} peptides, {df_val['allele'].nunique()} alleles)")
    print(f"  Test:  {len(df_test):,} pairs ({df_test['peptide'].nunique()} peptides, {df_test['allele'].nunique()} alleles)")

    # -------------------------------------------------------------------------
    # Step 2: Load ESMC-600M Foundation Model & Tokenizer
    # -------------------------------------------------------------------------
    print(f"\n[Step 2/5] Loading '{model_name}' onto {gpu_name}...")
    t0 = time.time()
    tokenizer = EsmcTokenizer.from_pretrained(model_name)
    esmc_backbone = EsmcModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device="cuda",
    )
    load_time = time.time() - t0
    total_backbone_params = sum(p.numel() for p in esmc_backbone.parameters())
    d_model = esmc_backbone.config.hidden_size if hasattr(esmc_backbone.config, "hidden_size") else 1152
    num_layers = esmc_backbone.config.num_hidden_layers if hasattr(esmc_backbone.config, "num_hidden_layers") else 36

    print(f"  Loaded {model_name} in {load_time:.2f}s")
    print(f"  Backbone Parameters: {total_backbone_params:,} | Layers: {num_layers} | d_model: {d_model}")
    print(f"  GPU VRAM allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # -------------------------------------------------------------------------
    # Step 3: Fast Pre-Tokenization of Joint Sequences (Zero-Padding)
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Pre-tokenizing joint complex sequences (peptide | hla_seq)...")
    t_tok = time.time()

    def tokenize_df(df, desc="dataset"):
        # Format: "<cls> PEPTIDE | HLA <eos>"
        # Using pipe '|' separator standard in ESM3/ESMC for multimeric complexes
        joint_seqs = [f"{p}|{h}" for p, h in zip(df["peptide"], df["hla_seq"])]
        tok_out = tokenizer(joint_seqs, return_tensors="pt", padding=False)
        input_ids = tok_out["input_ids"]  # Shape: (N, 194)
        
        y_raw = df["thalf_hours"].values.astype(np.float32)
        y_log = np.log10(1.0 + np.clip(y_raw, 0, None))
        y_clf = (y_raw >= 1.0).astype(np.float32)
        y_clf_2h = (y_raw >= 2.0).astype(np.float32)

        print(f"  Tokenized {len(df):,} {desc} pairs -> tensor shape {input_ids.shape}")
        return (
            input_ids,
            torch.tensor(y_log, dtype=torch.float32),
            torch.tensor(y_clf, dtype=torch.float32),
            torch.tensor(y_clf_2h, dtype=torch.float32),
            torch.tensor(y_raw, dtype=torch.float32),
        )

    train_ids, train_ylog, train_yclf, train_yclf2h, train_yraw = tokenize_df(df_train, "Train")
    val_ids, val_ylog, val_yclf, val_yclf2h, val_yraw = tokenize_df(df_val, "Val")
    test_ids, test_ylog, test_yclf, test_yclf2h, test_yraw = tokenize_df(df_test, "Test")
    print(f"  All sequences tokenized in {time.time() - t_tok:.2f}s (exact length = 194 tokens, 0 padding)")

    train_dataset = TensorDataset(train_ids, train_ylog, train_yclf, train_yraw)
    val_dataset = TensorDataset(val_ids, val_ylog, val_yclf, val_yraw)
    test_dataset = TensorDataset(test_ids, test_ylog, test_yclf, test_yraw)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, pin_memory=True)

    # -------------------------------------------------------------------------
    # Step 4: Fine-Tuning Model Architecture
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Building Fine-Tuning Model Architecture...")

    class ESMC600MStabilityFineTuner(nn.Module):
        """
        End-to-End Fine-Tuning Model for ESM-Cambrian 600M on Peptide-MHC Stability.
        Passes joint sequence through transformer blocks, pools quaternary structural features,
        and predicts continuous half-life and classification stability.
        """
        def __init__(
            self,
            backbone: EsmcModel,
            unfreeze_k_layers: int = 6,
            d_head: int = 512,
            p_drop: float = 0.15,
        ):
            super().__init__()
            self.backbone = backbone
            d_m = backbone.config.hidden_size if hasattr(backbone.config, "hidden_size") else 1152
            total_blocks = len(backbone.transformer.blocks)

            # Parameter freezing / unfreezing
            for p in self.backbone.parameters():
                p.requires_grad = False

            if unfreeze_k_layers == -1 or unfreeze_k_layers >= total_blocks:
                for p in self.backbone.parameters():
                    p.requires_grad = True
                print(f"  [Backbone] Fully unfrozen: all {total_blocks} transformer blocks trainable.")
            elif unfreeze_k_layers > 0:
                for block in self.backbone.transformer.blocks[-unfreeze_k_layers:]:
                    for p in block.parameters():
                        p.requires_grad = True
                for p in self.backbone.transformer.norm.parameters():
                    p.requires_grad = True
                print(
                    f"  [Backbone] Partially unfrozen: top {unfreeze_k_layers} blocks "
                    f"(blocks {total_blocks - unfreeze_k_layers}..{total_blocks - 1}) + final LayerNorm trainable."
                )
            else:
                print("  [Backbone] Frozen: only downstream interaction head trainable.")

            # Biophysical Readout Feature Concatenation:
            # 1. <cls> token (global complex state): d_m
            # 2. Mean-pooled peptide (bound conformation): d_m
            # 3. Mean-pooled HLA cleft (induced groove conformation): d_m
            # 4. Element-wise product interaction (pep * hla): d_m
            # 5. Absolute difference interaction (|pep - hla|): d_m
            # 6. P2 anchor residue (B-pocket anchor, residue 2): d_m
            # 7. P9 anchor residue (F-pocket anchor, residue 9): d_m
            # Total input feature dim: 7 * d_m = 8,064
            in_dim = 7 * d_m
            self.head_norm = nn.LayerNorm(in_dim)

            # Continuous Half-Life Regressor (log10(1 + thalf_hours))
            self.regressor = nn.Sequential(
                nn.Linear(in_dim, d_head),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(d_head, d_head // 2),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(d_head // 2, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

            # Binary Binder Classifier (thalf >= 1.0 hr)
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, d_head),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(d_head, d_head // 2),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(d_head // 2, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(self, input_ids):
            # Backbone forward pass
            out = self.backbone(input_ids)
            h = out.last_hidden_state.to(torch.float32)  # [B, 194, 1152]

            # Decomposition of the 194 tokens:
            # 0: <cls>
            # 1..9: Peptide residues P1..P9
            # 10: '|' chain delimiter
            # 11..192: HLA cleft domain (182 residues)
            # 193: <eos>
            cls_rep = h[:, 0, :]               # [B, 1152]
            pep_tokens = h[:, 1:10, :]         # [B, 9, 1152]
            hla_tokens = h[:, 11:193, :]       # [B, 182, 1152]

            pep_mean = pep_tokens.mean(dim=1)  # [B, 1152]
            hla_mean = hla_tokens.mean(dim=1)  # [B, 1152]

            prod = pep_mean * hla_mean         # [B, 1152]
            diff = torch.abs(pep_mean - hla_mean)  # [B, 1152]

            p2_anchor = pep_tokens[:, 1, :]    # [B, 1152] (P2 anchor)
            p9_anchor = pep_tokens[:, 8, :]    # [B, 1152] (P9 anchor)

            features = torch.cat(
                [cls_rep, pep_mean, hla_mean, prod, diff, p2_anchor, p9_anchor],
                dim=-1,
            )  # [B, 8064]
            features = self.head_norm(features)

            pred_log = self.regressor(features).squeeze(-1)
            pred_logits = self.classifier(features).squeeze(-1)

            return pred_log, pred_logits

    model = ESMC600MStabilityFineTuner(
        backbone=esmc_backbone,
        unfreeze_k_layers=unfreeze_layers,
        d_head=d_head,
        p_drop=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_trainable = sum(p.numel() for n, p in model.named_parameters() if "backbone" in n and p.requires_grad)
    head_trainable = sum(p.numel() for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad)

    print(f"  Total Parameters:             {total_params:,}")
    print(f"  Trainable Parameters:         {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"    - Backbone Trainable:       {backbone_trainable:,}")
    print(f"    - Head Trainable:           {head_trainable:,}")
    print(f"  GPU VRAM after model wrap:    {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # Multi-Objective Loss Formulation
    def compute_loss(pred_log, pred_logits, target_log, target_clf, eps=1e-8):
        # 1. Smooth L1 for continuous stability calibration
        loss_reg = F.smooth_l1_loss(pred_log, target_log)

        # 2. Pearson correlation loss (1 - r)
        p_c = pred_log - pred_log.mean()
        t_c = target_log - target_log.mean()
        cov = (p_c * t_c).sum()
        p_std = torch.sqrt((p_c**2).sum() + eps)
        t_std = torch.sqrt((t_c**2).sum() + eps)
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

        # 4. Binary Classification Loss (>= 1.0h threshold)
        loss_clf = F.binary_cross_entropy_with_logits(pred_logits, target_clf)

        total_loss = loss_reg + 0.6 * loss_pearson + 0.2 * loss_rank + 0.3 * loss_clf
        return total_loss, loss_reg, loss_pearson, loss_rank, loss_clf

    # Optimizer with Parameter Groups (Differential Learning Rates)
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})

    optimizer = torch.optim.AdamW(param_groups)

    # Warmup + Cosine Decay Learning Rate Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * 1  # 1 epoch warmup

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Evaluation function
    def evaluate(loader, desc="Val"):
        model.eval()
        preds_log = []
        preds_clf = []
        targets_log = []
        targets_clf = []
        targets_raw = []

        with torch.no_grad():
            for b_ids, b_ylog, b_yclf, b_yraw in loader:
                b_ids = b_ids.to(device, non_blocking=True)
                p_log, p_logits = model(b_ids)

                preds_log.extend(p_log.cpu().numpy())
                preds_clf.extend(torch.sigmoid(p_logits).cpu().numpy())
                targets_log.extend(b_ylog.numpy())
                targets_clf.extend(b_yclf.numpy())
                targets_raw.extend(b_yraw.numpy())

        preds_log = np.array(preds_log)
        preds_clf = np.array(preds_clf)
        targets_log = np.array(targets_log)
        targets_clf = np.array(targets_clf)
        targets_raw = np.array(targets_raw)

        preds_raw = np.clip(10.0**preds_log - 1.0, 0, None)

        p_r_log, _ = pearsonr(preds_log, targets_log)
        s_rho, _ = spearmanr(preds_log, targets_log)
        p_r_raw, _ = pearsonr(preds_raw, targets_raw)
        mse_log = float(np.mean((preds_log - targets_log) ** 2))
        rmse_log = float(np.sqrt(mse_log))
        mae_raw = float(np.mean(np.abs(preds_raw - targets_raw)))
        rmse_raw = float(np.sqrt(np.mean((preds_raw - targets_raw) ** 2)))

        try:
            auc_1h = float(roc_auc_score(targets_clf, preds_log))
            pr_1h = float(average_precision_score(targets_clf, preds_log))
        except Exception:
            auc_1h, pr_1h = float("nan"), float("nan")

        targets_2h = (targets_raw >= 2.0).astype(np.float32)
        try:
            auc_2h = float(roc_auc_score(targets_2h, preds_log))
            pr_2h = float(average_precision_score(targets_2h, preds_log))
        except Exception:
            auc_2h, pr_2h = float("nan"), float("nan")

        return {
            "pearson_r_log": float(p_r_log),
            "spearman_rho": float(s_rho),
            "pearson_r_raw": float(p_r_raw),
            "mse_log": mse_log,
            "rmse_log": rmse_log,
            "mae_raw": mae_raw,
            "rmse_raw": rmse_raw,
            "auc_roc_1h": auc_1h,
            "pr_auc_1h": pr_1h,
            "auc_roc_2h": auc_2h,
            "pr_auc_2h": pr_2h,
            "preds_log": preds_log.tolist(),
            "preds_clf": preds_clf.tolist(),
            "preds_raw": preds_raw.tolist(),
        }

    # -------------------------------------------------------------------------
    # Step 5: Execute Training Loop
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"[Step 5/5] Training ESMC-600M for {epochs} Epochs on NVIDIA A100-40GB...")
    print("=" * 80)

    epoch_history = []
    best_val_r = -1.0
    best_model_state = None
    train_start_t = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss_total = 0.0
        train_loss_reg = 0.0
        train_loss_pear = 0.0
        train_loss_rank = 0.0
        train_loss_clf = 0.0
        n_samples = 0

        for step, (b_ids, b_ylog, b_yclf, b_yraw) in enumerate(train_loader):
            b_ids = b_ids.to(device, non_blocking=True)
            b_ylog = b_ylog.to(device, non_blocking=True)
            b_yclf = b_yclf.to(device, non_blocking=True)

            optimizer.zero_grad()
            p_log, p_logits = model(b_ids)

            loss, l_reg, l_pear, l_rank, l_clf = compute_loss(p_log, p_logits, b_ylog, b_yclf)
            loss.backward()

            # Gradient clipping across all trainable parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            bs = len(b_ids)
            train_loss_total += loss.item() * bs
            train_loss_reg += l_reg.item() * bs
            train_loss_pear += l_pear.item() * bs
            train_loss_rank += l_rank.item() * bs
            train_loss_clf += l_clf.item() * bs
            n_samples += bs

        # Epoch training metrics
        train_loss_total /= n_samples
        train_loss_reg /= n_samples
        train_loss_pear /= n_samples
        train_loss_rank /= n_samples
        train_loss_clf /= n_samples

        # Validation evaluation
        val_res = evaluate(val_loader, desc="Val")
        ep_elapsed = time.time() - ep_t0

        is_best = val_res["pearson_r_log"] > best_val_r
        star = " ★ BEST" if is_best else ""
        if is_best:
            best_val_r = val_res["pearson_r_log"]
            # Save best weights in memory (CPU)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        current_lr_backbone = optimizer.param_groups[0]["lr"] if backbone_params else 0.0
        current_lr_head = optimizer.param_groups[-1]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} ({ep_elapsed:.1f}s) | "
            f"Loss: {train_loss_total:.4f} (Reg:{train_loss_reg:.4f}, Pear:{train_loss_pear:.4f}, Clf:{train_loss_clf:.4f}) | "
            f"Val r: {val_res['pearson_r_log']:.4f} | Val ρ: {val_res['spearman_rho']:.4f} | "
            f"Val AUC: {val_res['auc_roc_1h']:.4f}{star}"
        )

        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_loss_total,
            "train_loss_reg": train_loss_reg,
            "train_loss_pear": train_loss_pear,
            "train_loss_rank": train_loss_rank,
            "train_loss_clf": train_loss_clf,
            "val_pearson_r_log": val_res["pearson_r_log"],
            "val_spearman_rho": val_res["spearman_rho"],
            "val_pearson_r_raw": val_res["pearson_r_raw"],
            "val_rmse_log": val_res["rmse_log"],
            "val_rmse_raw": val_res["rmse_raw"],
            "val_mae_raw": val_res["mae_raw"],
            "val_auc_roc_1h": val_res["auc_roc_1h"],
            "val_pr_auc_1h": val_res["pr_auc_1h"],
            "val_auc_roc_2h": val_res["auc_roc_2h"],
            "val_pr_auc_2h": val_res["pr_auc_2h"],
            "epoch_time_seconds": ep_elapsed,
            "lr_backbone": current_lr_backbone,
            "lr_head": current_lr_head,
        })

    # Restore best validation checkpoint for final test evaluation
    if best_model_state is not None:
        print(f"\nRestoring best model checkpoint (Val Pearson r: {best_val_r:.4f})...")
        model.load_state_dict(best_model_state)

    # -------------------------------------------------------------------------
    # Final Evaluation on Held-Out IID Test Set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON HELD-OUT IID TEST SET (4,225 SAMPLES)")
    print("=" * 80)
    test_res = evaluate(test_loader, desc="Test")

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
    print("=" * 80)

    # Prepare return records
    df_test_out = df_test.copy()
    df_test_out["pred_log_thalf"] = test_res["preds_log"]
    df_test_out["pred_thalf_hours"] = test_res["preds_raw"]
    df_test_out["pred_prob_binding"] = test_res["preds_clf"]

    results_payload = {
        "model_name": "ESMC-600M-FineTuned",
        "foundation_checkpoint": model_name,
        "split": "iid",
        "unfreeze_layers": unfreeze_layers,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_backbone": lr_backbone,
        "lr_head": lr_head,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
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
        "total_training_time_seconds": time.time() - train_start_t,
    }

    return results_payload, df_test_out.to_dict(orient="records")


# -----------------------------------------------------------------------------
# 3. CLI LOCAL ENTRYPOINT
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    epochs: int = 12,
    batch_size: int = 32,
    lr_backbone: float = 2e-5,
    lr_head: float = 3e-4,
    unfreeze_layers: int = 6,
):
    import pandas as pd
    import json
    from pathlib import Path

    print("=" * 80)
    print("DISPATCHING MODAL JOB: ESMC-600M FINE-TUNING ON IID SPLIT")
    print(f"  Epochs: {epochs} | Batch size: {batch_size} | Unfrozen layers: {unfreeze_layers}")
    print(f"  LR Backbone: {lr_backbone} | LR Head: {lr_head}")
    print("=" * 80)

    results_payload, test_records = run_esmc_finetuning.remote(
        model_name="biohub/ESMC-600M",
        epochs=epochs,
        batch_size=batch_size,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        unfreeze_layers=unfreeze_layers,
    )

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Test Predictions
    df_test_preds = pd.DataFrame(test_records)
    pred_path = results_dir / "predictions_esmc_600m_finetuned_iid.csv"
    df_test_preds.to_csv(pred_path, index=False)
    print(f"\n[Saved] Test predictions to: {pred_path} ({len(df_test_preds):,} rows)")

    # 2. Save Comprehensive Metrics & History
    metrics_path = results_dir / "esmc_600m_finetuned_iid_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results_payload, f, indent=4)
    print(f"[Saved] Full metrics & training history to: {metrics_path}")

    print("\nFine-tuning run completed successfully!")


if __name__ == "__main__":
    pass
