"""Generate publication-grade benchmark visualization comparing all models across Paul's new splits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.4
plt.rcParams["grid.linestyle"] = "--"

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    paul_json = REPO_ROOT / "paul" / "results" / "baseline_metrics_summary.json"
    a0_json = REPO_ROOT / "RESULTS" / "onehot_paul_splits.json"
    bzzu_pca_json = REPO_ROOT / "RESULTS" / "BZZU_pca20_paul_splits.json"
    bzzu_flat_json = REPO_ROOT / "RESULTS" / "BZZU_flatten_paul_splits.json"

    with open(paul_json) as f:
        paul_data = json.load(f)
    with open(a0_json) as f:
        a0_data = json.load(f)
    with open(bzzu_pca_json) as f:
        bzzu_pca_data = json.load(f)

    bzzu_flat_data = None
    if bzzu_flat_json.exists():
        with open(bzzu_flat_json) as f:
            bzzu_flat_data = json.load(f)

    splits = ["iid", "h2", "aw_blosum", "novel_pep", "novel_allele"]
    split_labels = [
        "IID\n(Random)",
        "Hamming H=2\n(Seq Clustered)",
        "AW-BLOSUM\n(\u03c4=7.0 Clustered)",
        "Novel Peptide\n(Unseen Pep)",
        "Novel Allele\n(Unseen HLA)",
    ]

    models = [
        ("One-Hot XGBoost (Paul)", "#7f8c8d", "paul_xgb"),
        ("PMHCEmbeddingNet (Paul)", "#3498db", "paul_nn"),
        ("XGBoost on Embeddings (Paul)", "#2980b9", "paul_embed_xgb"),
        ("Baseline A0 MLP (One-Hot)", "#e67e22", "a0_mlp"),
        ("Boltz-2 BZZU (pca:20, 860-dim)", "#27ae60", "boltz_pca"),
    ]
    if bzzu_flat_data:
        models.append(("Boltz-2 BZZU (flatten, 5504-dim)", "#16a085", "boltz_flat"))

    # Build metric matrices
    spearman_matrix = []
    pearson_matrix = []

    for model_name, color, key in models:
        sp_vals = []
        pr_vals = []
        for s in splits:
            if key == "paul_xgb":
                sp = paul_data[s]["metrics"]["onehot_xgboost"]["spearman_rho"]
                pr = paul_data[s]["metrics"]["onehot_xgboost"]["pearson_r_log"]
            elif key == "paul_nn":
                sp = paul_data[s]["metrics"]["neural_network"]["spearman_rho"]
                pr = paul_data[s]["metrics"]["neural_network"]["pearson_r_log"]
            elif key == "paul_embed_xgb":
                sp = paul_data[s]["metrics"]["xgboost_on_embeddings"]["spearman_rho"]
                pr = paul_data[s]["metrics"]["xgboost_on_embeddings"]["pearson_r_log"]
            elif key == "a0_mlp":
                sp = a0_data[s]["spearman_rho"]
                pr = a0_data[s]["pearson_r_log"]
            elif key == "boltz_pca":
                sp = bzzu_pca_data[s]["spearman"]
                pr = bzzu_pca_data[s]["pearson"]
            elif key == "boltz_flat":
                sp = bzzu_flat_data[s]["spearman"]
                pr = bzzu_flat_data[s]["pearson"]
            sp_vals.append(sp)
            pr_vals.append(pr)
        spearman_matrix.append(sp_vals)
        pearson_matrix.append(pr_vals)

    # Figure 1: 2-Panel Comprehensive Benchmark Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

    x = np.arange(len(splits))
    n_models = len(models)
    width = 0.8 / n_models

    for i, ((m_name, color, _), sp_vals) in enumerate(zip(models, spearman_matrix)):
        pos = x - 0.4 + width / 2 + i * width
        rects = ax1.bar(pos, sp_vals, width, label=m_name, color=color, alpha=0.9, edgecolor="black", linewidth=0.6)
        for rect, val in zip(rects, sp_vals):
            y_pos = rect.get_height() + 0.01 if val >= 0 else rect.get_height() - 0.03
            ax1.text(rect.get_x() + rect.get_width() / 2, y_pos, f"{val:.2f}",
                     ha="center", va="bottom" if val >= 0 else "top", fontsize=7.5, fontweight="bold")

    ax1.set_title("A. Test Spearman Rank Correlation (\u03c1) Across Splitting Regimes", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Spearman Rank Correlation (\u03c1)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(split_labels, fontsize=10, fontweight="bold")
    ax1.set_ylim(-0.1, 0.95)
    ax1.axhline(0, color="#888888", linestyle="-", linewidth=0.8)
    ax1.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Panel 2: Retained Performance vs IID (% of IID Spearman)
    for i, ((m_name, color, key), sp_vals) in enumerate(zip(models, spearman_matrix)):
        if key == "paul_xgb":
            continue  # Near 0 on IID, percentage is noisy
        iid_val = sp_vals[0]
        pct_retained = [(val / iid_val) * 100 for val in sp_vals]
        ax2.plot(range(len(splits)), pct_retained, "o-", color=color, linewidth=2.4, markersize=8, label=m_name)
        for j, p in enumerate(pct_retained):
            ax2.annotate(f"{p:.1f}%", (j, p), textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=8.5, fontweight="bold", color=color)

    ax2.set_title("B. Robustness to Generalization Shift (% of IID Correlation Retained)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylabel("Retained Spearman \u03c1 Relative to IID (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(split_labels, fontsize=10, fontweight="bold")
    ax2.set_ylim(40, 105)
    ax2.legend(loc="lower left", frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Cross-Split Benchmark: Boltz-2 Structural Embeddings vs. Sequence Baselines\non GitHub paul/splits/ Regimes",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out1 = REPO_ROOT / "RESULTS" / "plots" / "splits_benchmark_comparison.png"
    out1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    out2 = REPO_ROOT / "paul" / "plots" / "splits_benchmark_comparison.png"
    out2.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1} and {out2}")


if __name__ == "__main__":
    main()
