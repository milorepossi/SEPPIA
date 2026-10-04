#!/usr/bin/env python3
"""
Publication-Quality Visualization Suite for ESMC Foundation Models vs Baselines.
Produces:
1. plots/14_esmc_training_convergence.png: Multi-metric training curves across epochs
2. plots/15_esmc_predicted_vs_actual.png: Parity plots with hexbin density and regression lines
3. plots/16_esmc_roc_and_pr_curves.png: ROC and PR curves at 1h & 2h binding thresholds
4. plots/17_baseline_vs_foundation_model.png: Comprehensive comparison across IID and Novel Allele
5. plots/18_esmc_error_by_stability_tier.png: Residual error across stability tiers & HLA loci
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Styling
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#2d3748"
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["grid.color"] = "#e2e8f0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7
plt.rcParams["figure.autolayout"] = False

COLORS = {
    "primary": "#3b82f6",     # Blue
    "secondary": "#8b5cf6",   # Purple
    "accent": "#10b981",      # Emerald
    "danger": "#ef4444",      # Red
    "warning": "#f59e0b",     # Amber
    "dark": "#1e293b",        # Slate dark
    "gray": "#64748b",
}


def plot_convergence(esmc_summary_path="results/esmc_metrics_summary.json", out_dir="plots"):
    path = Path(esmc_summary_path)
    if not path.exists():
        print(f"Skipping convergence plot: {path} not found.")
        return

    with open(path) as f:
        data = json.load(f)

    # Pick the best run history available
    hist = None
    model_label = "ESMC"
    for s in ["iid", "novel_allele"]:
        if s in data:
            for m in ["esmc_6b", "esmc_300m"]:
                if m in data[s] and "train_history" in data[s][m]:
                    hist = data[s][m]["train_history"]
                    model_label = f"{data[s][m]['model_name']} ({s.upper()})"
                    break
        if hist is not None:
            break

    if hist is None or len(hist) == 0:
        print("No training history found in summary.")
        return

    df_hist = pd.DataFrame(hist)
    epochs = df_hist["epoch"].values

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # 1. Loss Curve
    ax = axes[0, 0]
    ax.plot(epochs, df_hist["train_loss"], marker="o", color=COLORS["primary"], lw=2.2, label="Train Multi-Task Loss")
    ax.set_title("Training Loss (Smooth L1 + BCE)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="#f8fafc")

    # 2. Validation MSE
    ax = axes[0, 1]
    ax.plot(epochs, df_hist["val_mse"], marker="s", color=COLORS["danger"], lw=2.2, label="Val MSE (log10 scale)")
    ax.set_title("Validation Mean Squared Error", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("MSE", fontsize=10)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="#f8fafc")

    # 3. Pearson r and Spearman rho
    ax = axes[1, 0]
    ax.plot(epochs, df_hist["val_pearson"], marker="^", color=COLORS["secondary"], lw=2.2, label="Pearson r")
    ax.plot(epochs, df_hist["val_spearman"], marker="v", color=COLORS["accent"], lw=2.2, label="Spearman ρ")
    ax.set_title("Validation Correlation Trajectory", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Correlation Coefficient", fontsize=10)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="#f8fafc")

    # 4. AUROC and PR-AUC
    ax = axes[1, 1]
    ax.plot(epochs, df_hist["val_auroc_1h"], marker="d", color=COLORS["warning"], lw=2.2, label="AUROC (>=1.0h)")
    if "val_pr_auc_1h" in df_hist.columns:
        ax.plot(epochs, df_hist["val_pr_auc_1h"], marker="o", color=COLORS["dark"], lw=2.2, label="PR-AUC (>=1.0h)")
    ax.set_title("Validation Binding Classification Accuracy", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Area Under Curve", fontsize=10)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="#f8fafc")

    fig.suptitle(f"ESMC Foundation Model Training Convergence: {model_label}", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_file = Path(out_dir) / "14_esmc_training_convergence.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def plot_parity(preds_csv_path="results/predictions_esmc_6b_iid.csv", out_dir="plots"):
    path = Path(preds_csv_path)
    if not path.exists():
        # Fallback to 300m
        path = Path("results/predictions_esmc_300m_iid.csv")
        if not path.exists():
            print(f"Skipping parity plot: {preds_csv_path} not found.")
            return

    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    y_true_log = df["log_thalf_actual"].values
    y_pred_log = df["log_thalf_pred"].values

    r_log, _ = pearsonr(y_true_log, y_pred_log)
    rho, _ = spearmanr(y_true_log, y_pred_log)

    # Subplot 1: Log Scale Parity with Hexbin
    ax = axes[0]
    hb = ax.hexbin(y_true_log, y_pred_log, gridsize=45, cmap="YlGnBu", mincnt=1, edgecolors="none")
    cb = fig.colorbar(hb, ax=ax, shrink=0.85)
    cb.set_label("Sample Count", fontsize=9)

    min_val = min(y_true_log.min(), y_pred_log.min())
    max_val = max(y_true_log.max(), y_pred_log.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.8, label="Ideal Parity (y = x)")

    m, b = np.polyfit(y_true_log, y_pred_log, 1)
    ax.plot(y_true_log, m * y_true_log + b, color="#0f172a", lw=1.5, label=f"Fit (y = {m:.2f}x + {b:.2f})")

    ax.set_title("Stability Parity: Log Scale [log10(1 + t_1/2)]", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Experimental log10(1 + t_1/2 [hours])", fontsize=10)
    ax.set_ylabel("Predicted log10(1 + t_1/2 [hours])", fontsize=10)
    ax.grid(True)
    ax.text(
        0.05, 0.85,
        f"Pearson r = {r_log:.3f}\nSpearman ρ = {rho:.3f}\nN = {len(df):,}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9),
    )
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff")

    # Subplot 2: Raw Hours Scale (Clamped to 24h for visual clarity)
    ax = axes[1]
    y_true_raw = df["thalf_hours"].values
    y_pred_raw = df["thalf_hours_pred"].values
    r_raw, _ = pearsonr(y_true_raw, y_pred_raw)

    ax.scatter(y_true_raw, y_pred_raw, alpha=0.25, color=COLORS["secondary"], s=16, edgecolors="none")
    ax.plot([0, 25], [0, 25], "r--", lw=1.8, label="Ideal Parity (y = x)")

    m_raw, b_raw = np.polyfit(y_true_raw, y_pred_raw, 1)
    ax.plot(y_true_raw, m_raw * y_true_raw + b_raw, color="#0f172a", lw=1.5, label=f"Fit (y = {m_raw:.2f}x + {b_raw:.2f})")

    ax.set_xlim(-0.5, 26)
    ax.set_ylim(-0.5, 26)
    ax.set_title("Stability Parity: Raw Hours Scale (t_1/2)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Experimental Half-Life (hours)", fontsize=10)
    ax.set_ylabel("Predicted Half-Life (hours)", fontsize=10)
    ax.grid(True)
    ax.text(
        0.05, 0.85,
        f"Raw Pearson r = {r_raw:.3f}\nMAE = {df['abs_error_hours'].mean():.2f}h\nN = {len(df):,}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9),
    )
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff")

    fig.suptitle(f"ESMC Stability Prediction vs Ground Truth ({path.stem})", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    out_file = Path(out_dir) / "15_esmc_predicted_vs_actual.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def plot_roc_pr(preds_csv_path="results/predictions_esmc_6b_iid.csv", out_dir="plots"):
    path = Path(preds_csv_path)
    if not path.exists():
        path = Path("results/predictions_esmc_300m_iid.csv")
        if not path.exists():
            print(f"Skipping ROC/PR plot: {preds_csv_path} not found.")
            return

    df = pd.read_csv(path)
    scores = df["log_thalf_pred"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # ROC Curves
    ax = axes[0]
    for thresh, color, label in [(1.0, COLORS["primary"], "Moderate Binder (>=1.0h)"), (2.0, COLORS["secondary"], "Strong Binder (>=2.0h)")]:
        y_bin = (df["thalf_hours"].values >= thresh).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Guess (AUC = 0.500)")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff")

    # PR Curves
    ax = axes[1]
    for thresh, color, label in [(1.0, COLORS["primary"], "Moderate Binder (>=1.0h)"), (2.0, COLORS["secondary"], "Strong Binder (>=2.0h)")]:
        y_bin = (df["thalf_hours"].values >= thresh).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, scores)
        pr_auc = average_precision_score(y_bin, scores)
        base_rate = np.mean(y_bin)
        ax.plot(rec, prec, color=color, lw=2.2, label=f"{label} (AP = {pr_auc:.3f}, Base: {base_rate:.2f})")

    ax.set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff")

    fig.suptitle(f"Binding Classification Discriminative Power ({path.stem})", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    out_file = Path(out_dir) / "16_esmc_roc_and_pr_curves.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def plot_baseline_vs_foundation(
    baseline_summary_path="results/baseline_metrics_summary.json",
    esmc_summary_path="results/esmc_metrics_summary.json",
    out_dir="plots",
):
    b_path = Path(baseline_summary_path)
    e_path = Path(esmc_summary_path)

    if not b_path.exists() or not e_path.exists():
        print(f"Skipping model comparison: {b_path} or {e_path} missing.")
        return

    with open(b_path) as f:
        b_data = json.load(f)
    with open(e_path) as f:
        e_data = json.load(f)

    # Compare on IID and Novel Allele
    splits = [s for s in ["iid", "novel_allele"] if s in b_data and s in e_data]
    if not splits:
        print("No overlapping splits between baseline and ESMC summaries.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    for idx, split in enumerate(splits):
        ax = axes[idx]
        split_title = "In-Distribution (IID Split)" if split == "iid" else "Zero-Shot Novel Allele Generalization"

        models = []
        pearson_scores = []
        spearman_scores = []
        auroc_scores = []

        # Baselines
        b_metrics = b_data[split]["metrics"]
        if "onehot_xgboost" in b_metrics:
            models.append("One-Hot XGB")
            pearson_scores.append(b_metrics["onehot_xgboost"]["pearson_r_log"])
            spearman_scores.append(b_metrics["onehot_xgboost"]["spearman_rho"])
            auroc_scores.append(b_metrics["onehot_xgboost"]["auc_roc_1h"])

        if "neural_network" in b_metrics:
            models.append("PMHC-Net\n(Learned Emb)")
            pearson_scores.append(b_metrics["neural_network"]["pearson_r_log"])
            spearman_scores.append(b_metrics["neural_network"]["spearman_rho"])
            auroc_scores.append(b_metrics["neural_network"]["auc_roc_1h"])

        # ESMC
        if split in e_data:
            for m_key, m_label in [("esmc_300m", "ESMC-300M\n(Zero-Shot Emb)"), ("esmc_6b", "ESMC-6B\n(Foundation Model)")]:
                if m_key in e_data[split]:
                    m = e_data[split][m_key]["metrics"]
                    models.append(m_label)
                    pearson_scores.append(m["pearson_r_log"])
                    spearman_scores.append(m["spearman_rho"])
                    auroc_scores.append(m["auc_roc_1h"])

        x = np.arange(len(models))
        width = 0.25

        rects1 = ax.bar(x - width, pearson_scores, width, label="Pearson r", color=COLORS["primary"])
        rects2 = ax.bar(x, spearman_scores, width, label="Spearman ρ", color=COLORS["accent"])
        rects3 = ax.bar(x + width, auroc_scores, width, label="AUROC (>=1h)", color=COLORS["secondary"])

        ax.set_title(split_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Metric Score", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend(loc="upper left", frameon=True, facecolor="#ffffff")

        # Value annotations
        for rects in [rects1, rects2, rects3]:
            for r in rects:
                h = r.get_height()
                if h > 0:
                    ax.annotate(
                        f"{h:.2f}",
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8,
                    )

    fig.suptitle("Foundation Protein Language Models vs Classical Baselines", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    out_file = Path(out_dir) / "17_baseline_vs_foundation_model.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def plot_error_by_tier(preds_csv_path="results/predictions_esmc_6b_iid.csv", out_dir="plots"):
    path = Path(preds_csv_path)
    if not path.exists():
        path = Path("results/predictions_esmc_300m_iid.csv")
        if not path.exists():
            print(f"Skipping tier plot: {preds_csv_path} not found.")
            return

    df = pd.read_csv(path)

    # Assign stability tier
    def get_tier(t):
        if t == 0:
            return "Unbound (0h)"
        elif t < 1.0:
            return "Weak (<1h)"
        elif t < 2.0:
            return "Moderate (1-2h)"
        elif t < 5.0:
            return "Strong (2-5h)"
        else:
            return "Very Strong (>=5h)"

    df["tier"] = df["thalf_hours"].apply(get_tier)
    tier_order = ["Unbound (0h)", "Weak (<1h)", "Moderate (1-2h)", "Strong (2-5h)", "Very Strong (>=5h)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # MAE by Tier
    tier_mae = df.groupby("tier")["abs_error_hours"].mean().reindex(tier_order)
    tier_counts = df["tier"].value_counts().reindex(tier_order)

    ax = axes[0]
    bars = ax.bar(tier_order, tier_mae.values, color=COLORS["primary"], width=0.55, edgecolor="#1e293b")
    ax.set_title("Mean Absolute Error (hours) by Stability Tier", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("MAE (hours)", fontsize=10)
    ax.set_xticklabels(tier_order, rotation=20, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar, count in zip(bars, tier_counts.values):
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}h\n(n={count:,})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    # Distribution of Errors by HLA Locus (A, B, C)
    df["locus"] = df["allele"].str[:5]  # HLA-A, HLA-B, HLA-C
    top_loci = [l for l in ["HLA-A", "HLA-B", "HLA-C"] if l in df["locus"].unique()]

    ax = axes[1]
    locus_data = [df[df["locus"] == loc]["abs_error_hours"].values for loc in top_loci]
    ax.boxplot(locus_data, tick_labels=top_loci, patch_artist=True,
               boxprops=dict(facecolor="#e0e7ff", color="#4338ca"),
               medianprops=dict(color="#1e1b4b", lw=2),
               whiskerprops=dict(color="#4338ca"),
               capprops=dict(color="#4338ca"),
               showfliers=False)

    ax.set_title("Absolute Error Distribution by HLA Locus", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Absolute Error (hours, outlier-clipped)", fontsize=10)
    ax.set_xlabel("MHC Class I Locus", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig.suptitle(f"Error Analysis Across Stability Tiers & HLA Loci ({path.stem})", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    out_file = Path(out_dir) / "18_esmc_error_by_stability_tier.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def main():
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    print("Generating comprehensive visualization suite...")
    plot_convergence()
    plot_parity()
    plot_roc_pr()
    plot_baseline_vs_foundation()
    plot_error_by_tier()
    print("All plots generated successfully in plots/!")


if __name__ == "__main__":
    main()
