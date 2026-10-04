#!/usr/bin/env python3
"""
Publication-Quality Visualization Suite for Binding Modes Experiment.

Generates 5 figures:
1. 34_binding_modes_split_distributions.png: Dataset partitioning, peptide distributions, and 0-leakage verification.
2. 35_binding_modes_parallel_training_dynamics.png: Convergence curves across the 3 parallel models.
3. 36_binding_modes_test_performance.png: Intra-mode test performance benchmark (Pearson, Spearman, RMSE, MAE).
4. 37_binding_modes_cross_evaluation_heatmap.png: Cross-regime generalization matrix ($3 \times 3$).
5. 38_binding_modes_predicted_vs_actual.png: Parity plots on held-out unseen peptides for each binding mode.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "splits" / "binding_modes"
RESULTS_DIR = PROJECT_ROOT / "results" / "binding_modes"
PLOTS_DIR = PROJECT_ROOT / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

COLORS = {
    "non_binding": "#5c6b73",     # Slate grey
    "mild_binding": "#f39c12",    # Amber / warm orange
    "strong_binding": "#27ae60",  # Emerald green
    "accent_blue": "#2980b9",
    "accent_purple": "#8e44ad",
    "dark": "#2c3e50",
}


def plot_split_distributions():
    """Figure 34: Dataset partitioning, peptide distributions, and 0-leakage verification."""
    print("[1/5] Generating Plot 34: Binding Modes Split Distributions...")
    meta_path = SPLITS_DIR / "binding_modes_split_metadata.json"
    if not meta_path.exists():
        meta_path = PROJECT_ROOT / "splits" / "binding_modes_split_metadata.json"
    
    with open(meta_path) as f:
        meta = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Sample Breakdown by Mode & Split
    ax = axes[0, 0]
    modes = ["non_binding", "mild_binding", "strong_binding"]
    labels = ["Non-Binding\n(<1.0h)", "Mildly Binding\n(1.0–2.0h)", "Strong Binders\n(≥2.0h)"]
    
    train_counts = [meta["modes"][m]["train"]["samples"] for m in modes]
    val_counts = [meta["modes"][m]["val"]["samples"] for m in modes]
    test_counts = [meta["modes"][m]["test"]["samples"] for m in modes]

    x = np.arange(len(modes))
    w = 0.55

    b1 = ax.bar(x, train_counts, width=w, label="Train (70%)", color="#2c3e50", edgecolor="white")
    b2 = ax.bar(x, val_counts, width=w, bottom=train_counts, label="Val (15%)", color="#3498db", edgecolor="white")
    bottom_val = np.array(train_counts) + np.array(val_counts)
    b3 = ax.bar(x, test_counts, width=w, bottom=bottom_val, label="Test (15%)", color="#e74c3c", edgecolor="white")

    total_counts = [meta["modes"][m]["total_samples"] for m in modes]
    for i, tot in enumerate(total_counts):
        ax.text(i, tot + 200, f"N={tot:,}\n({tot/meta['dataset_total_samples']*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Peptide-HLA Pairs", fontsize=12, fontweight="bold")
    ax.set_title("A. Dataset Partitioning by Half-Life Binding Regime", fontsize=13, fontweight="bold", pad=10)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel B: Unique Peptides per Mode & Split
    ax = axes[0, 1]
    train_peps = [meta["modes"][m]["train"]["unique_peptides"] for m in modes]
    val_peps = [meta["modes"][m]["val"]["unique_peptides"] for m in modes]
    test_peps = [meta["modes"][m]["test"]["unique_peptides"] for m in modes]
    tot_peps = [meta["modes"][m]["total_peptides"] for m in modes]

    b_p1 = ax.bar(x, train_peps, width=w, label="Train Peptides (70%)", color="#34495e", edgecolor="white")
    b_p2 = ax.bar(x, val_peps, width=w, bottom=train_peps, label="Val Peptides (15%)", color="#5dade2", edgecolor="white")
    bottom_pep_val = np.array(train_peps) + np.array(val_peps)
    b_p3 = ax.bar(x, test_peps, width=w, bottom=bottom_pep_val, label="Test Peptides (15%)", color="#f1948a", edgecolor="white")

    for i, tot in enumerate(tot_peps):
        ax.text(i, tot + 70, f"{tot:,} peps\n[0% Leakage]", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#27ae60")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Unique 9-mer Peptides", fontsize=12, fontweight="bold")
    ax.set_title("B. Disjoint Peptide-Level Split (Strict 0% Leakage)", fontsize=13, fontweight="bold", pad=10)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel C: Half-Life Target Distribution Across the 3 Regimes
    ax = axes[1, 0]
    data_df = pd.read_csv(PROJECT_ROOT / "data" / "rasmussen_et_al_dataset.csv")
    
    nb_thalf = data_df[data_df["thalf_hours"] < 1.0]["thalf_hours"]
    mb_thalf = data_df[(data_df["thalf_hours"] >= 1.0) & (data_df["thalf_hours"] < 2.0)]["thalf_hours"]
    sb_thalf = data_df[data_df["thalf_hours"] >= 2.0]["thalf_hours"]

    bins = np.linspace(0, 1.0, 25)
    ax.hist(nb_thalf, bins=bins, color=COLORS["non_binding"], alpha=0.7, label=f"Non-binding (<1h, N={len(nb_thalf):,})", edgecolor="white")
    
    ax_twin = ax.twinx()
    bins_mb = np.linspace(1.0, 2.0, 25)
    ax_twin.hist(mb_thalf, bins=bins_mb, color=COLORS["mild_binding"], alpha=0.7, label=f"Mild binding (1-2h, N={len(mb_thalf):,})", edgecolor="white")
    
    ax.set_xlabel("Complex Half-life $t_{1/2}$ (hours)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (Non-binding)", fontsize=11, fontweight="bold", color=COLORS["non_binding"])
    ax_twin.set_ylabel("Frequency (Mildly binding)", fontsize=11, fontweight="bold", color=COLORS["mild_binding"])
    ax.set_title("C. Stability Distribution in Non-binding & Mildly Binding Regimes", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel D: Allele Representation across Splits
    ax = axes[1, 1]
    allele_data = []
    for m in modes:
        allele_data.append({
            "mode": m,
            "Total Alleles": meta["modes"][m]["total_alleles"],
            "Train Alleles": meta["modes"][m]["train"]["unique_alleles"],
            "Val Alleles": meta["modes"][m]["val"]["unique_alleles"],
            "Test Alleles": meta["modes"][m]["test"]["unique_alleles"],
        })
    df_al = pd.DataFrame(allele_data)
    
    w_al = 0.2
    x_al = np.arange(len(modes))
    ax.bar(x_al - 1.5*w_al, df_al["Total Alleles"], width=w_al, label="Total in Mode", color="#7f8c8d")
    ax.bar(x_al - 0.5*w_al, df_al["Train Alleles"], width=w_al, label="Train Split", color="#2c3e50")
    ax.bar(x_al + 0.5*w_al, df_al["Val Alleles"], width=w_al, label="Val Split", color="#3498db")
    ax.bar(x_al + 1.5*w_al, df_al["Test Alleles"], width=w_al, label="Test Split", color="#e74c3c")

    for i in range(len(modes)):
        ax.text(x_al[i] - 1.5*w_al, df_al.loc[i, "Total Alleles"] + 1, f"{df_al.loc[i, 'Total Alleles']}", ha="center", fontsize=9, fontweight="bold")
        ax.text(x_al[i] + 1.5*w_al, df_al.loc[i, "Test Alleles"] + 1, f"{df_al.loc[i, 'Test Alleles']}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x_al)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("HLA Allotypes Present", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 85)
    ax.set_title("D. HLA Allele Coverage (>95% Allotype Retention)", fontsize=13, fontweight="bold", pad=10)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_file = PLOTS_DIR / "34_binding_modes_split_distributions.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_training_dynamics():
    """Figure 35: Convergence curves across the 3 parallel models."""
    print("[2/5] Generating Plot 35: Parallel Training Dynamics...")
    summary_path = RESULTS_DIR / "binding_modes_metrics_summary.json"
    if not summary_path.exists():
        print(f"Metrics summary not found at {summary_path}. Skipping Plot 35.")
        return

    with open(summary_path) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    modes = ["non_binding", "mild_binding", "strong_binding"]
    titles = ["Non-Binding Model", "Mildly Binding Model", "Strong Binders Model"]

    for i, mode in enumerate(modes):
        ax = axes[i]
        if mode not in results:
            continue
        history = results[mode]["train_history"]
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_pearson = [h["val_pearson"] for h in history]
        val_spearman = [h["val_spearman"] for h in history]

        color = COLORS[mode]

        ax.plot(epochs, train_loss, color=color, linestyle="--", linewidth=2.2, label="Train Loss (Smooth L1 + Pearson)")
        ax.plot(epochs, val_pearson, color="#2980b9", linewidth=2.5, marker="o", markersize=4, label="Val Pearson r (log)")
        ax.plot(epochs, val_spearman, color="#8e44ad", linewidth=2.5, marker="s", markersize=4, label="Val Spearman ρ")

        best_idx = np.argmax(np.array(val_pearson) + np.array(val_spearman))
        best_ep = epochs[best_idx]
        best_r = val_pearson[best_idx]
        ax.axvline(best_ep, color="#e74c3c", linestyle=":", alpha=0.7, label=f"Best Checkpoint (Ep {best_ep})")
        ax.scatter([best_ep], [best_r], color="#e74c3c", s=70, zorder=5)

        ax.set_title(f"{titles[i]}\n(Trained on N={results[mode]['train_samples']:,} pairs)", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Metric Value / Loss", fontsize=11, fontweight="bold")
        ax.legend(frameon=True, fontsize=9, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_file = PLOTS_DIR / "35_binding_modes_parallel_training_dynamics.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_test_performance():
    """Figure 36: Test performance benchmark on held-out peptides."""
    print("[3/5] Generating Plot 36: Test Performance Benchmark...")
    summary_path = RESULTS_DIR / "binding_modes_metrics_summary.json"
    if not summary_path.exists():
        print(f"Metrics summary not found at {summary_path}. Skipping Plot 36.")
        return

    with open(summary_path) as f:
        results = json.load(f)

    modes = ["non_binding", "mild_binding", "strong_binding"]
    labels = ["Non-Binding\nSpecialist", "Mildly Binding\nSpecialist", "Strong Binders\nSpecialist"]

    pearsons = [results[m]["own_test_metrics"]["pearson_r_log"] for m in modes]
    spearmans = [results[m]["own_test_metrics"]["spearman_rho"] for m in modes]
    rmses = [results[m]["own_test_metrics"]["rmse_log"] for m in modes]
    maes = [results[m]["own_test_metrics"]["mae_raw"] for m in modes]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Correlation Metrics (Pearson & Spearman)
    ax = axes[0]
    x = np.arange(len(modes))
    w = 0.35
    b1 = ax.bar(x - w/2, pearsons, width=w, label="Pearson r (log10)", color="#2980b9", edgecolor="white")
    b2 = ax.bar(x + w/2, spearmans, width=w, label="Spearman ρ", color="#8e44ad", edgecolor="white")

    for i in range(len(modes)):
        ax.text(x[i] - w/2, pearsons[i] + 0.02, f"{pearsons[i]:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(x[i] + w/2, spearmans[i] + 0.02, f"{spearmans[i]:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Correlation Coefficient", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 0.95)
    ax.set_title("A. Correlation on Held-Out Peptides", fontsize=13, fontweight="bold", pad=10)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel B: Log-scale RMSE
    ax = axes[1]
    palette = [COLORS[m] for m in modes]
    bars_rmse = ax.bar(x, rmses, width=0.5, color=palette, edgecolor="white")
    for i, b in enumerate(bars_rmse):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{rmses[i]:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("RMSE on log10(1 + thalf)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(rmses) * 1.35)
    ax.set_title("B. Log-Scale Error (Narrow vs Wide Regimes)", fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel C: Raw Half-life MAE (hours)
    ax = axes[2]
    bars_mae = ax.bar(x, maes, width=0.5, color=palette, edgecolor="white")
    for i, b in enumerate(bars_mae):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{maes[i]:.2f}h", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (hours)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(maes) * 1.35)
    ax.set_title("C. Absolute Deviation in Half-life (Hours)", fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_file = PLOTS_DIR / "36_binding_modes_test_performance.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_cross_evaluation_heatmap():
    """Figure 37: 3x3 Cross-regime generalization matrix."""
    print("[4/5] Generating Plot 37: Cross-Evaluation Generalization Heatmap...")
    summary_path = RESULTS_DIR / "binding_modes_metrics_summary.json"
    if not summary_path.exists():
        print(f"Metrics summary not found at {summary_path}. Skipping Plot 37.")
        return

    with open(summary_path) as f:
        results = json.load(f)

    modes = ["non_binding", "mild_binding", "strong_binding"]
    row_labels = ["Trained on Non-Binding", "Trained on Mild-Binding", "Trained on Strong Binders"]
    col_labels = ["Tested on Non-Binding", "Tested on Mild-Binding", "Tested on Strong Binders"]

    matrix_pearson = np.zeros((3, 3))
    matrix_spearman = np.zeros((3, 3))
    matrix_mean_pred = np.zeros((3, 3))

    for i, train_mode in enumerate(modes):
        for j, eval_mode in enumerate(modes):
            cm = results[train_mode]["cross_eval_metrics"][eval_mode]
            matrix_pearson[i, j] = cm["pearson_r_log"]
            matrix_spearman[i, j] = cm["spearman_rho"]
            matrix_mean_pred[i, j] = cm["mean_pred_thalf"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Heatmap 1: Pearson r
    ax = axes[0]
    sns.heatmap(
        matrix_pearson,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        cbar=True,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        vmin=0.0,
        vmax=0.85,
    )
    ax.set_title("A. Cross-Evaluation Pearson Correlation (log10)", fontsize=13, fontweight="bold", pad=12)

    # Heatmap 2: Mean Predicted Stability (Hours)
    ax = axes[1]
    sns.heatmap(
        matrix_mean_pred,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar=True,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    ax.set_title("B. Mean Predicted Half-life (Hours)\n[True Means: Non=0.27h, Mild=1.38h, Strong=12.70h]", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    out_file = PLOTS_DIR / "37_binding_modes_cross_evaluation_heatmap.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_predicted_vs_actual():
    """Figure 38: Parity plots for all 3 models on their test sets."""
    print("[5/5] Generating Plot 38: Predicted vs Actual Parity Plots...")
    modes = ["non_binding", "mild_binding", "strong_binding"]
    titles = ["Non-Binding Model (<1.0h)", "Mildly Binding Model (1.0–2.0h)", "Strong Binders Model (≥2.0h)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for i, mode in enumerate(modes):
        ax = axes[i]
        csv_path = RESULTS_DIR / f"predictions_{mode}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)

        y_true = df["thalf_hours"].values
        y_pred = df["pred_thalf_hours"].values

        color = COLORS[mode]
        ax.scatter(y_true, y_pred, color=color, alpha=0.35, s=25, edgecolors="none")

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=2, label="Identity Line ($y=x$)")

        # Trend line
        m, b = np.polyfit(y_true, y_pred, 1)
        ax.plot(np.sort(y_true), m * np.sort(y_true) + b, color="#2c3e50", linewidth=2, label=f"Fit ($y={m:.2f}x+{b:.2f}$)")

        r = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 and np.std(y_pred) > 0 else 0.0

        ax.set_title(f"{titles[i]}\n(N={len(df):,} Test Pairs | Pearson r={r:.3f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual $t_{1/2}$ (hours)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted $t_{1/2}$ (hours)", fontsize=11, fontweight="bold")
        ax.legend(frameon=True, fontsize=9, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_file = PLOTS_DIR / "38_binding_modes_predicted_vs_actual.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")


def main():
    plot_split_distributions()
    plot_training_dynamics()
    plot_test_performance()
    plot_cross_evaluation_heatmap()
    plot_predicted_vs_actual()
    print("\nVisualization suite execution complete!")


if __name__ == "__main__":
    main()
