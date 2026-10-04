"""
Plotting Pipeline for ESMC-6B Cross-Attention Matrix Experiment (IID Split).
Generates publication-quality figures:
1. Detailed Training Loss & Validation Dynamics (Step-level loss, Reg vs Clf loss, Val MSE/Corr)
2. Stability Parity Plots (Log10 scale hexbin density & raw hours)
3. Discriminative Power: ROC & Precision-Recall curves
4. Cross-Attention Matrix Contact Heatmap (Residue 9x182 interaction map)
5. Model Comparison: Classical Baselines vs ESMc-6B Pooled vs ESMc-6B Cross-Attention
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from scipy.stats import pearsonr, spearmanr

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.titlesize": 15,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_loss_and_convergence(metrics_data, save_path):
    """Plot detailed step-level and epoch-level training losses + validation trajectories."""
    step_history = metrics_data.get("step_loss_history", [])
    epoch_history = metrics_data.get("epoch_history", [])

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: Step-level training loss
    if step_history:
        df_steps = pd.DataFrame(step_history)
        steps = df_steps["step"].values
        train_loss = df_steps["train_loss"].values
        reg_loss = df_steps["reg_loss"].values
        clf_loss = df_steps["clf_loss"].values

        # Rolling smooth
        window = 5
        smooth_loss = pd.Series(train_loss).rolling(window, min_periods=1).mean()

        axes[0, 0].plot(steps, train_loss, alpha=0.25, color="#1f77b4", label="Total Loss (raw steps)")
        axes[0, 0].plot(steps, smooth_loss, color="#0d47a1", linewidth=2.0, label="Total Loss (smoothed)")
        axes[0, 0].plot(steps, reg_loss, color="#d32f2f", linestyle="--", alpha=0.7, label="Smooth L1 (Reg)")
        axes[0, 0].plot(steps, 0.3 * clf_loss, color="#388e3c", linestyle=":", alpha=0.7, label="0.3 * BCE (Clf)")
        axes[0, 0].set_title("Step-by-Step Multi-Task Training Loss", fontweight="bold")
        axes[0, 0].set_xlabel("Optimization Step")
        axes[0, 0].set_ylabel("Batch Loss")
        axes[0, 0].legend(loc="upper right", frameon=True)
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Epoch Training Loss vs Validation MSE
    if epoch_history:
        epochs = [e["epoch"] for e in epoch_history]
        train_losses = [e["train_loss"] for e in epoch_history]
        val_mses = [e["val_mse"] for e in epoch_history]

        ax2 = axes[0, 1]
        line1 = ax2.plot(epochs, train_losses, "o-", color="#1976d2", linewidth=2.2, label="Train Loss (Multi-Task)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Train Loss", color="#1976d2")
        ax2.tick_params(axis="y", labelcolor="#1976d2")

        ax2_right = ax2.twinx()
        line2 = ax2_right.plot(epochs, val_mses, "s-", color="#e53935", linewidth=2.2, label="Val MSE (log10 scale)")
        ax2_right.set_ylabel("Validation MSE", color="#e53935")
        ax2_right.tick_params(axis="y", labelcolor="#e53935")

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="upper right", frameon=True)
        ax2.set_title("Epoch Train Loss vs Validation MSE", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Validation Correlation Trajectory
    if epoch_history:
        pearsons = [e["val_pearson"] for e in epoch_history]
        spearmans = [e["val_spearman"] for e in epoch_history]

        axes[1, 0].plot(epochs, pearsons, "o-", color="#7b1fa2", linewidth=2.2, label="Pearson r")
        axes[1, 0].plot(epochs, spearmans, "v-", color="#00897b", linewidth=2.2, label="Spearman ρ")
        axes[1, 0].set_title("Validation Correlation Trajectory", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Correlation Coefficient")
        axes[1, 0].legend(loc="lower right", frameon=True)
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Validation Classification Performance
    if epoch_history:
        aurocs = [e["val_auroc_1h"] for e in epoch_history]
        praucs = [e["val_pr_auc_1h"] for e in epoch_history]

        axes[1, 1].plot(epochs, aurocs, "D-", color="#f57c00", linewidth=2.2, label="Val AUROC (≥1.0h)")
        axes[1, 1].plot(epochs, praucs, "p-", color="#2e7d32", linewidth=2.2, label="Val PR-AUC (≥1.0h)")
        axes[1, 1].set_title("Validation Binding Classification Accuracy", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Area Under Curve")
        axes[1, 1].legend(loc="lower right", frameon=True)
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("ESMC-6B Cross-Attention Model: Detailed Training Dynamics (IID Split)", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] Training dynamics plot to: {save_path}")


def plot_predicted_vs_actual(df_preds, save_path):
    """Parity plots comparing predicted vs actual stability."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    actual_log = df_preds["actual_log"].values
    pred_log = df_preds["pred_stability_log"].values
    actual_raw = df_preds["thalf_hours"].values
    pred_raw = df_preds["pred_thalf_hours"].values

    r_log, _ = pearsonr(actual_log, pred_log)
    rho_log, _ = spearmanr(actual_log, pred_log)
    r_raw, _ = pearsonr(actual_raw, pred_raw)

    # Panel 1: Log scale Hexbin
    hb = axes[0].hexbin(
        actual_log,
        pred_log,
        gridsize=50,
        cmap="YlGnBu",
        mincnt=1,
        bins="log",
    )
    cb = fig.colorbar(hb, ax=axes[0])
    cb.set_label("Sample Count (log10)", rotation=270, labelpad=15)

    max_val_log = max(actual_log.max(), pred_log.max()) * 1.05
    axes[0].plot([0, max_val_log], [0, max_val_log], "r--", linewidth=2.0, label="Ideal Parity (y = x)")

    m_log, b_log = np.polyfit(actual_log, pred_log, 1)
    x_line = np.linspace(0, max_val_log, 100)
    axes[0].plot(x_line, m_log * x_line + b_log, "k-", linewidth=1.8, label=f"Fit (y = {m_log:.2f}x + {b_log:.2f})")

    axes[0].set_title("Stability Parity: Log Scale [log10(1 + t_1/2)]", fontweight="bold")
    axes[0].set_xlabel("Experimental log10(1 + t_1/2 [hours])")
    axes[0].set_ylabel("Predicted log10(1 + t_1/2 [hours])")
    axes[0].text(
        0.05, 0.92,
        f"Pearson r = {r_log:.3f}\nSpearman ρ = {rho_log:.3f}\nN = {len(df_preds):,}",
        transform=axes[0].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#ccc"),
    )
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Raw Hours Scale
    axes[1].scatter(actual_raw, pred_raw, alpha=0.25, color="#5c6bc0", s=18, edgecolors="none")
    max_val_raw = 26.0
    axes[1].plot([0, max_val_raw], [0, max_val_raw], "r--", linewidth=2.0, label="Ideal Parity (y = x)")

    m_raw, b_raw = np.polyfit(actual_raw, pred_raw, 1)
    x_raw_line = np.linspace(0, max_val_raw, 100)
    axes[1].plot(x_raw_line, m_raw * x_raw_line + b_raw, "k-", linewidth=1.8, label=f"Fit (y = {m_raw:.2f}x + {b_raw:.2f})")

    axes[1].set_xlim(-0.5, max_val_raw)
    axes[1].set_ylim(-0.5, max_val_raw)
    axes[1].set_title("Stability Parity: Raw Hours Scale (t_1/2)", fontweight="bold")
    axes[1].set_xlabel("Experimental Half-Life (hours)")
    axes[1].set_ylabel("Predicted Half-Life (hours)")

    mae_raw = np.mean(np.abs(actual_raw - pred_raw))
    axes[1].text(
        0.05, 0.92,
        f"Raw Pearson r = {r_raw:.3f}\nMAE = {mae_raw:.2f}h\nN = {len(df_preds):,}",
        transform=axes[1].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#ccc"),
    )
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("ESMC-6B Cross-Attention Stability Parity (IID Test Set)", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] Parity plot to: {save_path}")


def plot_roc_pr_curves(df_preds, save_path):
    """Plot ROC and PR curves for binding classification."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    actual_raw = df_preds["thalf_hours"].values
    pred_prob_1h = df_preds["pred_prob_binding_1h"].values
    pred_log = df_preds["pred_stability_log"].values

    thresholds = [
        ("Moderate Binder (≥1.0h)", actual_raw >= 1.0, pred_prob_1h, "#2979ff"),
        ("Strong Binder (≥2.0h)", actual_raw >= 2.0, pred_log, "#7c4dff"),
    ]

    for label, y_bin, score, color in thresholds:
        fpr, tpr, _ = roc_curve(y_bin, score)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})", color=color, linewidth=2.4)

        prec, rec, _ = precision_recall_curve(y_bin, score)
        pr_auc = average_precision_score(y_bin, score)
        base_rate = np.mean(y_bin)
        axes[1].plot(rec, prec, label=f"{label} (AP = {pr_auc:.3f}, Base: {base_rate:.2f})", color=color, linewidth=2.4)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.500)")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("Precision-Recall (PR) Curves", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Cross-Attention Binding Classification Power (IID Test Set)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] ROC & PR curves to: {save_path}")


def plot_cross_attention_contact_map(metrics_data, save_path):
    """Plot the learned 9x182 residue cross-attention contact matrix."""
    sample_maps = metrics_data.get("sample_attn_maps", [])
    if not sample_maps:
        print("[Warning] No sample attention maps found in metrics.")
        return

    # Average attention map across sample pairs
    avg_map = np.mean(np.array(sample_maps), axis=0) # shape: (9, 182)

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(avg_map, aspect="auto", cmap="viridis", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Mean Attention Weight", rotation=270, labelpad=15)

    ax.set_yticks(range(9))
    ax.set_yticklabels([f"P{i+1}" for i in range(9)], fontweight="bold")
    ax.set_xlabel("HLA Class I Heavy Chain Residue Index (1 to 182, Alpha-1 & Alpha-2 Grooves)")
    ax.set_ylabel("Peptide Position")
    ax.set_title("Residue-Level Cross-Attention Contact Matrix: Peptide (P1-P9) vs HLA Groove", fontweight="bold", fontsize=14)

    # Highlight canonical anchor positions P2 and P9 (PΩ)
    ax.axhline(1, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="Anchor P2 (B-pocket)")
    ax.axhline(8, color="cyan", linestyle="--", alpha=0.6, linewidth=1.5, label="Anchor P9/PΩ (F-pocket)")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] Cross-attention contact map to: {save_path}")


def plot_architecture_comparison(save_path):
    """Compare Classical Baselines vs ESMc-6B Pooled vs ESMc-6B Cross-Attention."""
    results_path = RESULTS_DIR / "esmc_cross_attention_iid_metrics.json"
    if not results_path.exists():
        return

    with open(results_path) as f:
        attn_metrics = json.load(f)["metrics"]

    with open(RESULTS_DIR / "esmc_metrics_summary.json") as f:
        summary = json.load(f)
        pooled_metrics = summary["iid"]["esmc_6b"]["metrics"]

    with open(RESULTS_DIR / "baseline_metrics_summary.json") as f:
        base_summary = json.load(f)
        pmhc_metrics = base_summary["iid"]["metrics"]["neural_network"]
        xgb_metrics = base_summary["iid"]["metrics"]["onehot_xgboost"]

    models = ["One-Hot XGBoost", "PMHC-Net (Learned)", "ESMC-6B (Pooled)", "ESMC-6B (Cross-Attn)"]
    pearsons = [xgb_metrics["pearson_r_log"], pmhc_metrics["pearson_r_log"], pooled_metrics["pearson_r_log"], attn_metrics["pearson_r_log"]]
    spearmans = [xgb_metrics["spearman_rho"], pmhc_metrics["spearman_rho"], pooled_metrics["spearman_rho"], attn_metrics["spearman_rho"]]
    aurocs = [xgb_metrics["auc_roc_1h"], pmhc_metrics["auc_roc_1h"], pooled_metrics["auc_roc_1h"], attn_metrics["auc_roc_1h"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.25

    rects1 = ax.bar(x - width, pearsons, width, label="Pearson r (log10)", color="#1976d2", edgecolor="black", alpha=0.88)
    rects2 = ax.bar(x, spearmans, width, label="Spearman ρ", color="#388e3c", edgecolor="black", alpha=0.88)
    rects3 = ax.bar(x + width, aurocs, width, label="AUROC (≥1.0h)", color="#f57c00", edgecolor="black", alpha=0.88)

    ax.set_ylabel("Score")
    ax.set_title("Architectural Comparison on In-Distribution (IID) Split: Baselines vs ESMC-6B", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] Architecture comparison to: {save_path}")


def main():
    metrics_path = RESULTS_DIR / "esmc_cross_attention_iid_metrics.json"
    preds_path = RESULTS_DIR / "predictions_esmc_6b_cross_attention_iid.csv"

    if not metrics_path.exists() or not preds_path.exists():
        print(f"Waiting for results files to exist: {metrics_path}")
        return

    with open(metrics_path) as f:
        metrics_data = json.load(f)

    df_preds = pd.read_csv(preds_path)

    plot_training_loss_and_convergence(metrics_data, PLOTS_DIR / "19_esmc_cross_attn_training_loss.png")
    plot_predicted_vs_actual(df_preds, PLOTS_DIR / "20_esmc_cross_attn_predicted_vs_actual.png")
    plot_roc_pr_curves(df_preds, PLOTS_DIR / "21_esmc_cross_attn_roc_and_pr.png")
    plot_cross_attention_contact_map(metrics_data, PLOTS_DIR / "22_esmc_cross_attn_contact_map.png")
    plot_architecture_comparison(PLOTS_DIR / "23_esmc_cross_attn_vs_baselines.png")
    print("\nAll cross-attention plots generated successfully!")


if __name__ == "__main__":
    main()
