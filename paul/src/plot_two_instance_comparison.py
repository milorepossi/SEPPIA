"""
Comparison script: DeepExtract (Single-Instance Shared Space) vs DeepExtract (Two Dedicated Instances)
Evaluates impact of dedicating specialized encoder instances for Peptide (9-mer) vs Allele (182 cleft).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def generate_two_instance_comparison():
    splits = ["iid", "novel_allele", "novel_pep"]
    split_labels = {"iid": "IID", "novel_allele": "Novel Allele", "novel_pep": "Novel Peptide"}

    data = []
    histories = {"1inst": {}, "2inst": {}}

    for s in splits:
        p1 = RESULTS_DIR / f"esmc_deep_extract_1inst_{s}_metrics.json"
        p2 = RESULTS_DIR / f"esmc_deep_extract_2inst_{s}_metrics.json"

        if not p1.exists() or not p2.exists():
            print(f"[Wait] Results for {s} not yet complete.")
            return

        with open(p1) as f:
            m1 = json.load(f)
        with open(p2) as f:
            m2 = json.load(f)

        histories["1inst"][s] = m1.get("epoch_history", [])
        histories["2inst"][s] = m2.get("epoch_history", [])

        tm1 = m1["test_metrics"]
        tm2 = m2["test_metrics"]

        data.append({
            "split": split_labels[s],
            "split_code": s,
            "architecture": "Single-Instance DeepExtract",
            "pearson_r": tm1["pearson_r_log"],
            "spearman_rho": tm1["spearman_rho"],
            "auroc_1h": tm1["auc_roc_1h"],
            "pr_auc_1h": tm1["pr_auc_1h"],
            "rmse_log": tm1["rmse_log"],
        })
        data.append({
            "split": split_labels[s],
            "split_code": s,
            "architecture": "Two-Instance DeepExtract (Ours)",
            "pearson_r": tm2["pearson_r_log"],
            "spearman_rho": tm2["spearman_rho"],
            "auroc_1h": tm2["auc_roc_1h"],
            "pr_auc_1h": tm2["pr_auc_1h"],
            "rmse_log": tm2["rmse_log"],
        })

    df = pd.DataFrame(data)
    print("\n" + "=" * 70)
    print("COMPARATIVE EVALUATION: SINGLE-INSTANCE VS TWO-INSTANCE DEEPEXTRACT")
    print("=" * 70)
    print(df.to_string(index=False))

    # --- Plot 1: 4-Panel Metric Comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    metrics = [
        ("pearson_r", "Pearson r (log10 t1/2) ↑", "Linear Correlation"),
        ("spearman_rho", "Spearman Rank ρ ↑", "Monotonic Rank Ordering"),
        ("auroc_1h", "AUROC (t1/2 >= 1h) ↑", "Long-Binder Classification"),
        ("pr_auc_1h", "PR-AUC (t1/2 >= 1h) ↑", "Precision-Recall Retention"),
    ]

    palette = {"Single-Instance DeepExtract": "#3b82f6", "Two-Instance DeepExtract (Ours)": "#8b5cf6"}

    for ax, (m_col, m_title, m_sub) in zip(axes.flat, metrics):
        sns.barplot(
            data=df,
            x="split",
            y=m_col,
            hue="architecture",
            palette=palette,
            ax=ax,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.92,
        )
        ax.set_title(f"{m_title}\n({m_sub})", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("")
        ax.set_ylabel(m_title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Annotate values
        for p in ax.patches:
            val = p.get_height()
            if val > 0:
                ax.annotate(
                    f"{val:.4f}",
                    (p.get_x() + p.get_width() / 2.0, val),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    xytext=(0, 3),
                    textcoords="offset points",
                )

        ax.legend(title="", fontsize=10, loc="lower right")

    plt.suptitle(
        "Architectural Ablation: Single-Instance Shared Space vs Dedicated Two-Instance DeepExtract\nESMc-6B Backbone Across All 3 Evaluation Splits",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    p_out = PLOTS_DIR / "34_two_instance_vs_single_instance_deepextract.png"
    plt.savefig(p_out, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] Plot to: {p_out}")

    # --- Plot 2: Epoch Learning Dynamics Comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    for i, s in enumerate(splits):
        ax = axes[i]
        h1 = histories["1inst"].get(s, [])
        h2 = histories["2inst"].get(s, [])

        if h1 and h2:
            epochs_1 = [e["epoch"] for e in h1]
            val_r_1 = [e["val_pearson_r_log"] for e in h1]
            epochs_2 = [e["epoch"] for e in h2]
            val_r_2 = [e["val_pearson_r_log"] for e in h2]

            ax.plot(epochs_1, val_r_1, color="#3b82f6", linewidth=2.2, linestyle="--", label="Single Instance (Val r)")
            ax.plot(epochs_2, val_r_2, color="#8b5cf6", linewidth=2.5, label="Two Dedicated Instances (Val r)")

            best_1 = max(val_r_1)
            best_2 = max(val_r_2)
            ax.axhline(best_1, color="#3b82f6", linestyle=":", alpha=0.7, label=f"Best 1-Inst: {best_1:.4f}")
            ax.axhline(best_2, color="#8b5cf6", linestyle=":", alpha=0.7, label=f"Best 2-Inst: {best_2:.4f}")

        ax.set_title(f"{split_labels[s]} Split Dynamics", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Validation Pearson r", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Validation Pearson r Convergence: Single-Instance vs Two-Instance Encoders", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    p_dyn = PLOTS_DIR / "35_two_instance_learning_dynamics.png"
    plt.savefig(p_dyn, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Plot to: {p_dyn}")


if __name__ == "__main__":
    generate_two_instance_comparison()
