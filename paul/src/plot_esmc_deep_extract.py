"""Plotting script for Deep ESMC-6B Extraction and Training Dynamics Optimization.

Generates:
1. plots/24_esmc_deep_training_dynamics.png - Comprehensive 6-panel training dynamics & decomposed losses
2. plots/25_esmc_deep_parity.png - Hexbin parity plot (log & raw hours)
3. plots/26_esmc_deep_roc_pr.png - ROC and PR curves (1.0h and 2.0h thresholds)
4. plots/27_esmc_deep_contact_attention.png - Learned 2D cross-attention contact matrix
5. plots/28_esmc_deep_vs_all_baselines.png - Benchmark bar chart comparing all 5 models
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_dynamics(save_path: Path):
    with open(RESULTS_DIR / "esmc_deep_extract_iid_metrics.json") as f:
        data = json.load(f)

    step_history = pd.DataFrame(data["step_history"])
    epoch_history = pd.DataFrame(data["epoch_history"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)

    # Panel 1: Step Total Loss (Raw & Smoothed)
    ax1 = axes[0, 0]
    ax1.plot(step_history["step"], step_history["total_loss"], color="#a0c4ff", alpha=0.4, label="Raw Batch Loss")
    smooth_loss = step_history["total_loss"].rolling(window=10, min_periods=1).mean()
    ax1.plot(step_history["step"], smooth_loss, color="#1d3557", linewidth=2.2, label="Smoothed Trend")
    ax1.set_title("Step-by-Step Total Multi-Task Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Optimization Step", fontsize=10)
    ax1.set_ylabel("Total Loss", fontsize=10)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Decomposed Loss Components
    ax2 = axes[0, 1]
    ax2.plot(step_history["step"], step_history["reg_loss"].rolling(8, min_periods=1).mean(),
             color="#e63946", linewidth=1.8, label=r"Smooth L1 ($\mathcal{L}_{\mathrm{reg}}$)")
    ax2.plot(step_history["step"], step_history["pearson_loss"].rolling(8, min_periods=1).mean(),
             color="#2a9d8f", linewidth=1.8, label=r"Pearson ($1 - r$)")
    ax2.plot(step_history["step"], step_history["rank_loss"].rolling(8, min_periods=1).mean(),
             color="#e76f51", linewidth=1.8, linestyle="--", label=r"Margin Rank ($\mathcal{L}_{\mathrm{rank}}$)")
    ax2.plot(step_history["step"], step_history["clf_loss"].rolling(8, min_periods=1).mean(),
             color="#457b9d", linewidth=1.8, linestyle=":", label=r"BCE Clf ($\mathcal{L}_{\mathrm{clf}}$)")
    ax2.set_title("Decomposed Multi-Task Loss Components", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Optimization Step", fontsize=10)
    ax2.set_ylabel("Loss Component Value", fontsize=10)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Epoch Train Loss vs Val MSE
    ax3 = axes[0, 2]
    ax3_twin = ax3.twinx()
    l1 = ax3.plot(epoch_history["epoch"], epoch_history["train_loss"], color="#1f77b4", marker="o",
                  linewidth=2.0, label="Train Loss (Multi-Task)")
    l2 = ax3_twin.plot(epoch_history["epoch"], epoch_history["val_mse_log"], color="#d62728", marker="s",
                       linewidth=2.0, label="Val MSE (log10)")
    ax3.set_title("Epoch Train Loss vs Validation MSE", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=10)
    ax3.set_ylabel("Train Loss", color="#1f77b4", fontsize=10)
    ax3_twin.set_ylabel("Val MSE (log10 scale)", color="#d62728", fontsize=10)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper right", frameon=True, fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Validation Correlation Trajectory
    ax4 = axes[1, 0]
    ax4.plot(epoch_history["epoch"], epoch_history["val_pearson_r_log"], color="#7209b7", marker="o",
             linewidth=2.2, label=r"Pearson $r$ ($\log_{10}$)")
    ax4.plot(epoch_history["epoch"], epoch_history["val_spearman_rho"], color="#007f5f", marker="v",
             linewidth=2.2, label=r"Spearman $\rho$")
    ax4.set_title("Validation Correlation Trajectory", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Epoch", fontsize=10)
    ax4.set_ylabel("Correlation Coefficient", fontsize=10)
    best_idx = epoch_history["val_pearson_r_log"].idxmax()
    best_r = epoch_history.loc[best_idx, "val_pearson_r_log"]
    best_ep = epoch_history.loc[best_idx, "epoch"]
    ax4.annotate(f"Peak r = {best_r:.4f}\n(Epoch {int(best_ep)})",
                 xy=(best_ep, best_r), xytext=(best_ep - 5, best_r - 0.05),
                 arrowprops=dict(facecolor="#7209b7", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#7209b7"))
    ax4.legend(loc="lower right", frameon=True, fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # Panel 5: Validation Binding Classification Accuracy
    ax5 = axes[1, 1]
    ax5.plot(epoch_history["epoch"], epoch_history["val_auc_roc_1h"], color="#f77f00", marker="D",
             linewidth=2.0, label=r"Val AUROC ($\geq 1.0$h)")
    ax5.plot(epoch_history["epoch"], epoch_history["val_pr_auc_1h"], color="#2b9348", marker="^",
             linewidth=2.0, label=r"Val PR-AUC ($\geq 1.0$h)")
    ax5.set_title("Validation Binding Classification Power", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Epoch", fontsize=10)
    ax5.set_ylabel("Area Under Curve", fontsize=10)
    ax5.legend(loc="lower right", frameon=True, fontsize=9)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # Panel 6: Learning Rate Schedule
    ax6 = axes[1, 2]
    ax6.plot(step_history["step"], step_history["lr"] * 1e4, color="#0096c7", linewidth=2.0)
    ax6.set_title("Learning Rate Schedule (Warmup + Cosine)", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Optimization Step", fontsize=10)
    ax6.set_ylabel(r"Learning Rate ($\times 10^{-4}$)", fontsize=10)
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("ESMC-6B DeepExtract: Multi-Objective Training Dynamics & Convergence",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Training dynamics plot to: {save_path}")


def plot_parity(save_path: Path):
    df = pd.read_csv(RESULTS_DIR / "predictions_esmc_6b_deep_extract_iid.csv")
    with open(RESULTS_DIR / "esmc_deep_extract_iid_metrics.json") as f:
        metrics = json.load(f)["test_metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    # Panel A: Log scale parity
    ax1 = axes[0]
    exp_log = np.log10(1.0 + np.clip(df["thalf_hours"].values, 0, None))
    pred_log = df["pred_log_thalf"].values

    hb = ax1.hexbin(exp_log, pred_log, gridsize=50, cmap="YlGnBu", mincnt=1, bins="log")
    cb = fig.colorbar(hb, ax=ax1, pad=0.02)
    cb.set_label("Sample Count (log10)", fontsize=10)

    m, b = np.polyfit(exp_log, pred_log, 1)
    x_line = np.linspace(exp_log.min(), exp_log.max(), 100)
    ax1.plot(x_line, x_line, "r--", linewidth=2.0, label="Ideal Parity (y = x)")
    ax1.plot(x_line, m * x_line + b, "k-", linewidth=2.0, label=f"Fit (y = {m:.2f}x + {b:.2f})")

    ax1.set_title(r"Stability Parity: Log Scale [$\log_{10}(1 + t_{1/2})$]", fontsize=12, fontweight="bold")
    ax1.set_xlabel(r"Experimental $\log_{10}(1 + t_{1/2}\ \mathrm{[hours]})$", fontsize=11)
    ax1.set_ylabel(r"Predicted $\log_{10}(1 + t_{1/2}\ \mathrm{[hours]})$", fontsize=11)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    stats_text = (
        f"Pearson r = {metrics['pearson_r_log']:.3f}\n"
        f"Spearman ρ = {metrics['spearman_rho']:.3f}\n"
        f"RMSE = {metrics['rmse_log']:.3f}\n"
        f"N = {len(df):,}"
    )
    ax1.text(0.04, 0.94, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc", alpha=0.9))

    # Panel B: Raw Hours Parity
    ax2 = axes[1]
    exp_raw = df["thalf_hours"].values
    pred_raw = df["pred_thalf_hours"].values

    ax2.scatter(exp_raw, pred_raw, alpha=0.25, s=16, color="#1d3557", edgecolors="none")
    m_r, b_r = np.polyfit(exp_raw, pred_raw, 1)
    max_h = 26
    ax2.plot([0, max_h], [0, max_h], "r--", linewidth=2.0, label="Ideal Parity (y = x)")
    ax2.plot([0, max_h], [b_r, m_r * max_h + b_r], "k-", linewidth=2.0, label=f"Fit (y = {m_r:.2f}x + {b_r:.2f})")

    ax2.set_xlim(-0.5, max_h)
    ax2.set_ylim(-0.5, max_h)
    ax2.set_title(r"Stability Parity: Raw Hours Scale ($t_{1/2}$)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Experimental Half-Life (hours)", fontsize=11)
    ax2.set_ylabel("Predicted Half-Life (hours)", fontsize=11)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    stats_raw = (
        f"Raw Pearson r = {metrics['pearson_r_raw']:.3f}\n"
        f"MAE = {metrics['mae_raw']:.2f}h\n"
        f"N = {len(df):,}"
    )
    ax2.text(0.04, 0.94, stats_raw, transform=ax2.transAxes, fontsize=10,
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc", alpha=0.9))

    plt.suptitle("ESMC-6B DeepExtract: Stability Predictions vs Experimental Half-Life (IID Test)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Parity plot to: {save_path}")


def plot_roc_pr(save_path: Path):
    df = pd.read_csv(RESULTS_DIR / "predictions_esmc_6b_deep_extract_iid.csv")
    targets_raw = df["thalf_hours"].values
    preds_clf = df["pred_prob_binding"].values
    preds_log = df["pred_log_thalf"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

    # ROC
    ax1 = axes[0]
    t_1h = (targets_raw >= 1.0).astype(int)
    fpr_1h, tpr_1h, _ = roc_curve(t_1h, preds_clf)
    auc_1h = auc(fpr_1h, tpr_1h)

    t_2h = (targets_raw >= 2.0).astype(int)
    fpr_2h, tpr_2h, _ = roc_curve(t_2h, preds_log)
    auc_2h = auc(fpr_2h, tpr_2h)

    ax1.plot(fpr_1h, tpr_1h, color="#0077b6", linewidth=2.5, label=rf"Moderate Binder ($\geq 1.0$h) (AUC = {auc_1h:.3f})")
    ax1.plot(fpr_2h, tpr_2h, color="#7209b7", linewidth=2.5, label=rf"Strong Binder ($\geq 2.0$h) (AUC = {auc_2h:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.500)")

    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # PR
    ax2 = axes[1]
    p_1h, r_1h, _ = precision_recall_curve(t_1h, preds_clf)
    ap_1h = average_precision_score(t_1h, preds_clf)
    base_1h = t_1h.mean()

    p_2h, r_2h, _ = precision_recall_curve(t_2h, preds_log)
    ap_2h = average_precision_score(t_2h, preds_log)
    base_2h = t_2h.mean()

    ax2.plot(r_1h, p_1h, color="#0077b6", linewidth=2.5, label=rf"Moderate Binder ($\geq 1.0$h) (AP = {ap_1h:.3f}, Base: {base_1h:.2f})")
    ax2.plot(r_2h, p_2h, color="#7209b7", linewidth=2.5, label=rf"Strong Binder ($\geq 2.0$h) (AP = {ap_2h:.3f}, Base: {base_2h:.2f})")

    ax2.set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("ESMC-6B DeepExtract: Binding Classification Discrimination (IID Test Set)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] ROC & PR curves to: {save_path}")


def plot_contact_map(save_path: Path):
    with open(RESULTS_DIR / "esmc_deep_extract_iid_metrics.json") as f:
        metrics = json.load(f)

    attn_map = np.array(metrics["sample_attn_map"])  # [9, 182]

    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
    im = ax.imshow(attn_map, aspect="auto", cmap="viridis", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Cross-Attention Interaction Weight", rotation=270, labelpad=15)

    ax.set_yticks(range(9))
    ax.set_yticklabels([f"P{i+1}" for i in range(9)], fontweight="bold")
    ax.axhline(1, color="red", linestyle="--", linewidth=1.5, alpha=0.9, label="Anchor P2 (B-pocket)")
    ax.axhline(8, color="cyan", linestyle="--", linewidth=1.5, alpha=0.9, label="Anchor P9/PΩ (F-pocket)")

    ax.set_title("Learned Residue-Level Cross-Attention Matrix: Peptide (P1-P9) vs HLA Groove",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("HLA Class I Heavy Chain Residue Index (1 to 182, Alpha-1 & Alpha-2 Grooves)", fontsize=11)
    ax.set_ylabel("Peptide Position", fontsize=11)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Cross-attention contact map to: {save_path}")


def plot_architecture_comparison(save_path: Path):
    with open(RESULTS_DIR / "esmc_deep_extract_iid_metrics.json") as f:
        deep_metrics = json.load(f)["test_metrics"]

    with open(RESULTS_DIR / "esmc_cross_attention_iid_metrics.json") as f:
        attn_metrics = json.load(f)["metrics"]

    with open(RESULTS_DIR / "esmc_metrics_summary.json") as f:
        pooled_metrics = json.load(f)["iid"]["esmc_6b"]["metrics"]

    with open(RESULTS_DIR / "baseline_metrics_summary.json") as f:
        base_summary = json.load(f)
        pmhc_metrics = base_summary["iid"]["metrics"]["neural_network"]
        xgb_metrics = base_summary["iid"]["metrics"]["onehot_xgboost"]

    models = [
        "One-Hot XGBoost",
        "ESMC-6B (Pooled)",
        "ESMC-6B (Cross-Attn)",
        "PMHC-Net (Learned)",
        "ESMC-6B (DeepExtract)",
    ]

    pearsons = [
        xgb_metrics["pearson_r_log"],
        pooled_metrics["pearson_r_log"],
        attn_metrics["pearson_r_log"],
        pmhc_metrics["pearson_r_log"],
        deep_metrics["pearson_r_log"],
    ]

    spearmans = [
        xgb_metrics["spearman_rho"],
        pooled_metrics["spearman_rho"],
        attn_metrics["spearman_rho"],
        pmhc_metrics["spearman_rho"],
        deep_metrics["spearman_rho"],
    ]

    aurocs = [
        xgb_metrics["auc_roc_1h"],
        pooled_metrics["auc_roc_1h"],
        attn_metrics["auc_roc_1h"],
        pmhc_metrics["auc_roc_1h"],
        deep_metrics["auc_roc_1h"],
    ]

    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)

    rects1 = ax.bar(x - width, pearsons, width, label=r"Pearson $r$ ($\log_{10}$)", color="#1976d2", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x, spearmans, width, label=r"Spearman $\rho$", color="#388e3c", edgecolor="black", linewidth=0.8)
    rects3 = ax.bar(x + width, aurocs, width, label=r"AUROC ($\geq 1.0$h)", color="#f57c00", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Architectural & Optimization Evolution on IID Split: Baselines vs ESMC-6B DeepExtract",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Architecture comparison to: {save_path}")


def main():
    print("Generating comprehensive visualization suite for ESMC-6B DeepExtract...")
    plot_training_dynamics(PLOTS_DIR / "24_esmc_deep_training_dynamics.png")
    plot_parity(PLOTS_DIR / "25_esmc_deep_parity.png")
    plot_roc_pr(PLOTS_DIR / "26_esmc_deep_roc_pr.png")
    plot_contact_map(PLOTS_DIR / "27_esmc_deep_contact_attention.png")
    plot_architecture_comparison(PLOTS_DIR / "28_esmc_deep_vs_all_baselines.png")
    print("\nAll DeepExtract plots generated successfully!")


if __name__ == "__main__":
    main()
