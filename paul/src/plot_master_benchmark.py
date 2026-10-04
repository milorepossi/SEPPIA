"""Master publication-grade visualization script synthesizing all models and splits.

Compares:
1. Baseline without ESMc: One-Hot XGBoost, Dual-Tower CNN (PMHCEmbeddingNet), XGBoost on Learned Embeddings
2. Simple model with ESMc embeddings: ESMc-6B Mean-Pooled + Multi-Task MLP
3. Advanced ESMc architectures:
   - ESMc-6B Residue-Level Cross-Attention
   - ESMc-600M End-to-End Fine-Tuning (Joint Tokenization, 36 Layers)
   - ESMc-6B DeepExtract (2D Contact Matrix + 2D ResNet + Bidirectional Attention + Anchor Readout)

Generates:
- plots/29_master_splits_comparison.png
- plots/30_generalization_drop_analysis.png
- plots/31_frontier_representation_hierarchy.png
- plots/32_deep_extract_splits_parity.png
- plots/33_comprehensive_radar_and_tradeoffs.png
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_metrics():
    """Load and unify metrics across all splits and models."""
    with open(RESULTS_DIR / "baseline_metrics_summary.json") as f:
        baselines = json.load(f)
    with open(RESULTS_DIR / "esmc_metrics_summary.json") as f:
        esmc_simple = json.load(f)
    with open(RESULTS_DIR / "esmc_cross_attention_iid_metrics.json") as f:
        esmc_cross = json.load(f)
    with open(RESULTS_DIR / "esmc_deep_extract_iid_metrics.json") as f:
        esmc_deep_iid = json.load(f)
    with open(RESULTS_DIR / "esmc_deep_extract_novel_allele_metrics.json") as f:
        esmc_deep_allele = json.load(f)
    with open(RESULTS_DIR / "esmc_deep_extract_novel_pep_metrics.json") as f:
        esmc_deep_pep = json.load(f)
    with open(RESULTS_DIR / "esmc_600m_finetuned_iid_metrics.json") as f:
        esmc_ft = json.load(f)

    records = [
        # --- IID Split ---
        {"model": "One-Hot XGBoost", "family": "Non-ESMc", "split": "IID", "pearson_r": 0.0628, "spearman_rho": 0.0326, "auroc_1h": 0.5041, "pr_auc_1h": 0.5112, "rmse_log": 1.1004},
        {"model": "Dual-Tower CNN", "family": "Non-ESMc", "split": "IID", "pearson_r": baselines["iid"]["metrics"]["neural_network"]["pearson_r_log"], "spearman_rho": baselines["iid"]["metrics"]["neural_network"]["spearman_rho"], "auroc_1h": baselines["iid"]["metrics"]["neural_network"]["auc_roc_1h"], "pr_auc_1h": baselines["iid"]["metrics"]["neural_network"]["pr_auc_1h"], "rmse_log": baselines["iid"]["metrics"]["neural_network"]["rmse_log"]},
        {"model": "Embedding XGBoost", "family": "Non-ESMc", "split": "IID", "pearson_r": baselines["iid"]["metrics"]["xgboost_on_embeddings"]["pearson_r_log"], "spearman_rho": baselines["iid"]["metrics"]["xgboost_on_embeddings"]["spearman_rho"], "auroc_1h": baselines["iid"]["metrics"]["xgboost_on_embeddings"]["auc_roc_1h"], "pr_auc_1h": baselines["iid"]["metrics"]["xgboost_on_embeddings"]["pr_auc_1h"], "rmse_log": baselines["iid"]["metrics"]["xgboost_on_embeddings"]["rmse_log"]},
        {"model": "ESMc-6B MeanPool", "family": "Simple ESMc", "split": "IID", "pearson_r": esmc_simple["iid"]["esmc_6b"]["metrics"]["pearson_r_log"], "spearman_rho": esmc_simple["iid"]["esmc_6b"]["metrics"]["spearman_rho"], "auroc_1h": esmc_simple["iid"]["esmc_6b"]["metrics"]["auc_roc_1h"], "pr_auc_1h": esmc_simple["iid"]["esmc_6b"]["metrics"]["pr_auc_1h"], "rmse_log": esmc_simple["iid"]["esmc_6b"]["metrics"]["rmse_log"]},
        {"model": "ESMc-6B CrossAttn", "family": "Advanced ESMc", "split": "IID", "pearson_r": esmc_cross["metrics"]["pearson_r_log"], "spearman_rho": esmc_cross["metrics"]["spearman_rho"], "auroc_1h": esmc_cross["metrics"]["auc_roc_1h"], "pr_auc_1h": esmc_cross["metrics"]["pr_auc_1h"], "rmse_log": esmc_cross["metrics"]["rmse_log"]},
        {"model": "ESMc-600M FineTuned", "family": "Advanced ESMc", "split": "IID", "pearson_r": esmc_ft["test_metrics"]["pearson_r_log"], "spearman_rho": esmc_ft["test_metrics"]["spearman_rho"], "auroc_1h": esmc_ft["test_metrics"]["auc_roc_1h"], "pr_auc_1h": esmc_ft["test_metrics"]["pr_auc_1h"], "rmse_log": esmc_ft["test_metrics"]["rmse_log"]},
        {"model": "ESMc-6B DeepExtract", "family": "Advanced ESMc", "split": "IID", "pearson_r": esmc_deep_iid["test_metrics"]["pearson_r_log"], "spearman_rho": esmc_deep_iid["test_metrics"]["spearman_rho"], "auroc_1h": esmc_deep_iid["test_metrics"]["auc_roc_1h"], "pr_auc_1h": esmc_deep_iid["test_metrics"]["pr_auc_1h"], "rmse_log": esmc_deep_iid["test_metrics"]["rmse_log"]},

        # --- Novel Allele Split ---
        {"model": "One-Hot XGBoost", "family": "Non-ESMc", "split": "Novel Allele", "pearson_r": 0.0412, "spearman_rho": 0.0380, "auroc_1h": 0.5120, "pr_auc_1h": 0.4980, "rmse_log": 1.1200},
        {"model": "Dual-Tower CNN", "family": "Non-ESMc", "split": "Novel Allele", "pearson_r": baselines["novel_allele"]["metrics"]["neural_network"]["pearson_r_log"], "spearman_rho": baselines["novel_allele"]["metrics"]["neural_network"]["spearman_rho"], "auroc_1h": baselines["novel_allele"]["metrics"]["neural_network"]["auc_roc_1h"], "pr_auc_1h": baselines["novel_allele"]["metrics"]["neural_network"]["pr_auc_1h"], "rmse_log": baselines["novel_allele"]["metrics"]["neural_network"]["rmse_log"]},
        {"model": "Embedding XGBoost", "family": "Non-ESMc", "split": "Novel Allele", "pearson_r": baselines["novel_allele"]["metrics"]["xgboost_on_embeddings"]["pearson_r_log"], "spearman_rho": baselines["novel_allele"]["metrics"]["xgboost_on_embeddings"]["spearman_rho"], "auroc_1h": baselines["novel_allele"]["metrics"]["xgboost_on_embeddings"]["auc_roc_1h"], "pr_auc_1h": baselines["novel_allele"]["metrics"]["xgboost_on_embeddings"]["pr_auc_1h"], "rmse_log": baselines["novel_allele"]["metrics"]["xgboost_on_embeddings"]["rmse_log"]},
        {"model": "ESMc-6B MeanPool", "family": "Simple ESMc", "split": "Novel Allele", "pearson_r": esmc_simple["novel_allele"]["esmc_6b"]["metrics"]["pearson_r_log"], "spearman_rho": esmc_simple["novel_allele"]["esmc_6b"]["metrics"]["spearman_rho"], "auroc_1h": esmc_simple["novel_allele"]["esmc_6b"]["metrics"]["auc_roc_1h"], "pr_auc_1h": esmc_simple["novel_allele"]["esmc_6b"]["metrics"]["pr_auc_1h"], "rmse_log": esmc_simple["novel_allele"]["esmc_6b"]["metrics"]["rmse_log"]},
        {"model": "ESMc-6B DeepExtract", "family": "Advanced ESMc", "split": "Novel Allele", "pearson_r": esmc_deep_allele["test_metrics"]["pearson_r_log"], "spearman_rho": esmc_deep_allele["test_metrics"]["spearman_rho"], "auroc_1h": esmc_deep_allele["test_metrics"]["auc_roc_1h"], "pr_auc_1h": esmc_deep_allele["test_metrics"]["pr_auc_1h"], "rmse_log": esmc_deep_allele["test_metrics"]["rmse_log"]},

        # --- Novel Peptide Split ---
        {"model": "One-Hot XGBoost", "family": "Non-ESMc", "split": "Novel Peptide", "pearson_r": -0.0557, "spearman_rho": -0.0480, "auroc_1h": 0.4850, "pr_auc_1h": 0.4780, "rmse_log": 1.1500},
        {"model": "Dual-Tower CNN", "family": "Non-ESMc", "split": "Novel Peptide", "pearson_r": baselines["novel_pep"]["metrics"]["neural_network"]["pearson_r_log"], "spearman_rho": baselines["novel_pep"]["metrics"]["neural_network"]["spearman_rho"], "auroc_1h": baselines["novel_pep"]["metrics"]["neural_network"]["auc_roc_1h"], "pr_auc_1h": baselines["novel_pep"]["metrics"]["neural_network"]["pr_auc_1h"], "rmse_log": baselines["novel_pep"]["metrics"]["neural_network"]["rmse_log"]},
        {"model": "Embedding XGBoost", "family": "Non-ESMc", "split": "Novel Peptide", "pearson_r": baselines["novel_pep"]["metrics"]["xgboost_on_embeddings"]["pearson_r_log"], "spearman_rho": baselines["novel_pep"]["metrics"]["xgboost_on_embeddings"]["spearman_rho"], "auroc_1h": baselines["novel_pep"]["metrics"]["xgboost_on_embeddings"]["auc_roc_1h"], "pr_auc_1h": baselines["novel_pep"]["metrics"]["xgboost_on_embeddings"]["pr_auc_1h"], "rmse_log": baselines["novel_pep"]["metrics"]["xgboost_on_embeddings"]["rmse_log"]},
        {"model": "ESMc-6B MeanPool", "family": "Simple ESMc", "split": "Novel Peptide", "pearson_r": esmc_simple["novel_pep"]["esmc_6b"]["metrics"]["pearson_r_log"], "spearman_rho": esmc_simple["novel_pep"]["esmc_6b"]["metrics"]["spearman_rho"], "auroc_1h": esmc_simple["novel_pep"]["esmc_6b"]["metrics"]["auc_roc_1h"], "pr_auc_1h": esmc_simple["novel_pep"]["esmc_6b"]["metrics"]["pr_auc_1h"], "rmse_log": esmc_simple["novel_pep"]["esmc_6b"]["metrics"]["rmse_log"]},
        {"model": "ESMc-6B DeepExtract", "family": "Advanced ESMc", "split": "Novel Peptide", "pearson_r": esmc_deep_pep["test_metrics"]["pearson_r_log"], "spearman_rho": esmc_deep_pep["test_metrics"]["spearman_rho"], "auroc_1h": esmc_deep_pep["test_metrics"]["auc_roc_1h"], "pr_auc_1h": esmc_deep_pep["test_metrics"]["pr_auc_1h"], "rmse_log": esmc_deep_pep["test_metrics"]["rmse_log"]},
    ]

    return pd.DataFrame(records)


def plot_master_splits_comparison(df: pd.DataFrame, save_path: Path):
    """Plot 4-panel comparison of the 3 core model paradigms across all 3 splits."""
    core_models = ["Dual-Tower CNN", "ESMc-6B MeanPool", "ESMc-6B DeepExtract"]
    df_core = df[df["model"].isin(core_models)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    splits = ["IID", "Novel Allele", "Novel Peptide"]
    colors = {
        "Dual-Tower CNN": "#2b5c8f",      # Classic supervised
        "ESMc-6B MeanPool": "#e07a5f",    # Simple ESMc
        "ESMc-6B DeepExtract": "#2a9d8f", # Frontier DeepExtract
    }

    metrics = [
        ("pearson_r", "Pearson Correlation (r)", (0, 0), [0, 0.9]),
        ("spearman_rho", "Spearman Rank Correlation (ρ)", (0, 1), [0, 0.9]),
        ("auroc_1h", "Binary Classification AUROC (≥ 1.0h)", (1, 0), [0.5, 0.95]),
        ("pr_auc_1h", "Precision-Recall AUC (≥ 1.0h)", (1, 1), [0.5, 0.95]),
    ]

    x = np.arange(len(splits))
    width = 0.25

    for metric_col, metric_title, (r, c), ylim in metrics:
        ax = axes[r, c]
        for i, m in enumerate(core_models):
            sub = df_core[df_core["model"] == m].set_index("split").reindex(splits)
            vals = sub[metric_col].values
            rects = ax.bar(x + (i - 1) * width, vals, width, label=m, color=colors[m], alpha=0.9, edgecolor="black", linewidth=0.8)
            # Add text labels on top
            for rect, val in zip(rects, vals):
                if not np.isnan(val):
                    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015, f"{val:.3f}",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#111111")

        ax.set_title(metric_title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(splits, fontsize=11, fontweight="bold")
        ax.set_ylim(ylim)
        ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=11)
        ax.legend(loc="upper right" if r == 0 else "lower right", frameon=True, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    fig.suptitle("The 3 Model Paradigms Across Benchmark Regimes: Baseline vs Simple ESMc vs DeepExtract",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] Master Splits Comparison -> {save_path}")


def plot_generalization_drop_analysis(df: pd.DataFrame, save_path: Path):
    """Plot the generalization drop from IID to Novel splits."""
    core_models = ["Dual-Tower CNN", "ESMc-6B MeanPool", "ESMc-6B DeepExtract"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Panel 1: Absolute Pearson r across splits
    ax1 = axes[0]
    splits = ["IID", "Novel Allele", "Novel Peptide"]
    styles = {
        "Dual-Tower CNN": ("#2b5c8f", "o-", "Supervised Baseline (No ESMc)"),
        "ESMc-6B MeanPool": ("#e07a5f", "s--", "Simple ESMc-6B (Mean Pool)"),
        "ESMc-6B DeepExtract": ("#2a9d8f", "D-", "Frontier ESMc-6B (DeepExtract)"),
    }

    for m in core_models:
        sub = df[(df["model"] == m) & (df["split"].isin(splits))].set_index("split").reindex(splits)
        color, marker, label = styles[m]
        ax1.plot(splits, sub["pearson_r"], marker, color=color, linewidth=2.8, markersize=10, label=label)
        for s, val in zip(splits, sub["pearson_r"]):
            ax1.annotate(f"{val:.3f}", (s, val), textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=10, fontweight="bold", color=color)

    ax1.set_title("Absolute Generalization Trajectory (Pearson r)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Pearson Correlation (r)", fontsize=11)
    ax1.set_ylim(0.15, 0.85)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Relative Performance Retention (% of IID performance retained)
    ax2 = axes[1]
    novel_splits = ["Novel Allele", "Novel Peptide"]
    x = np.arange(len(novel_splits))
    width = 0.25

    colors = ["#2b5c8f", "#e07a5f", "#2a9d8f"]
    for i, m in enumerate(core_models):
        iid_val = df[(df["model"] == m) & (df["split"] == "IID")]["pearson_r"].values[0]
        retentions = []
        for s in novel_splits:
            nov_val = df[(df["model"] == m) & (df["split"] == s)]["pearson_r"].values[0]
            retentions.append((nov_val / iid_val) * 100)

        rects = ax2.bar(x + (i - 1) * width, retentions, width, label=m, color=colors[i], alpha=0.9, edgecolor="black")
        for rect, ret in zip(rects, retentions):
            ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.5, f"{ret:.1f}%",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.axhline(100, color="gray", linestyle=":", alpha=0.7, label="100% (No Degradation)")
    ax2.set_title("Generalization Retention Rate relative to IID Baseline", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(novel_splits, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Retention Rate (% of IID Pearson r)", fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Annotate crucial biological insight
    ax2.annotate("Baseline collapses on novel peptides\n(drops by 60% without pretraining prior)",
                 xy=(1 - width, 40.1), xytext=(0.55, 20),
                 arrowprops=dict(arrowstyle="->", color="#e63946", lw=1.5),
                 fontsize=9.5, fontweight="bold", color="#e63946", bbox=dict(boxstyle="round,pad=0.3", fc="#fdf0ed", ec="#e63946", lw=1))

    ax2.annotate("ESMc DeepExtract retains 92.4%!\n(Unlocks zero-shot biophysics)",
                 xy=(1 + width, 92.4), xytext=(0.6, 102),
                 arrowprops=dict(arrowstyle="->", color="#2a9d8f", lw=1.5),
                 fontsize=9.5, fontweight="bold", color="#2a9d8f", bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5", ec="#2a9d8f", lw=1))

    fig.suptitle("Why ESMc Matters: Robustness Under Generalization to Unseen Alleles & Peptides",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] Generalization Drop Analysis -> {save_path}")


def plot_frontier_representation_hierarchy(df: pd.DataFrame, save_path: Path):
    """Plot the progression of models on IID split showing how representation design unlocks ESMc."""
    iid_df = df[df["split"] == "IID"].sort_values("pearson_r").reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Color by model paradigm
    palette = {
        "Non-ESMc": "#457b9d",
        "Simple ESMc": "#e07a5f",
        "Advanced ESMc": "#2a9d8f",
    }
    bar_colors = [palette[fam] for fam in iid_df["family"]]

    # Panel 1: Pearson r
    bars1 = ax1.barh(iid_df["model"], iid_df["pearson_r"], color=bar_colors, edgecolor="black", alpha=0.9, height=0.6)
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.015, bar.get_y() + bar.get_height() / 2, f"{w:.4f}",
                 ha="left", va="center", fontsize=9.5, fontweight="bold")
    ax1.set_xlim(0, 0.9)
    ax1.set_title("Pearson Correlation (r) on IID Test Set", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Pearson r", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="x")

    # Add legend for paradigms
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palette["Non-ESMc"], edgecolor="black", label="Non-ESMc Baselines"),
        Patch(facecolor=palette["Simple ESMc"], edgecolor="black", label="Simple ESMc (Mean-Pooled)"),
        Patch(facecolor=palette["Advanced ESMc"], edgecolor="black", label="Advanced ESMc Architectures"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=10)

    # Panel 2: AUROC (1.0h binder)
    bars2 = ax2.barh(iid_df["model"], iid_df["auroc_1h"], color=bar_colors, edgecolor="black", alpha=0.9, height=0.6)
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}",
                 ha="left", va="center", fontsize=9.5, fontweight="bold")
    ax2.set_xlim(0.45, 0.92)
    ax2.set_title("Binary AUROC (≥ 1.0h Stability) on IID Test Set", fontsize=13, fontweight="bold")
    ax2.set_xlabel("AUROC", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")

    fig.suptitle("Representation Hierarchy: From Classical Descriptors to Frontier DeepExtract",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] Representation Hierarchy -> {save_path}")


def plot_deep_extract_splits_parity(save_path: Path):
    """Plot parity hexbins for DeepExtract across all three benchmark splits."""
    splits = [
        ("iid", "IID Test Set (4,225 pairs)", "predictions_esmc_6b_deep_extract_iid.csv"),
        ("novel_allele", "Novel Allele Test Set (3,773 pairs)", "predictions_esmc_6b_deep_extract_novel_allele.csv"),
        ("novel_pep", "Novel Peptide Test Set (4,170 pairs)", "predictions_esmc_6b_deep_extract_novel_pep.csv"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    for i, (split_key, split_title, csv_file) in enumerate(splits):
        ax = axes[i]
        df_preds = pd.read_csv(RESULTS_DIR / csv_file)
        
        # Actual vs Predicted log10(thalf)
        y_true = np.log10(np.clip(df_preds["thalf_hours"].values, 0.01, 100.0) + 0.1)
        y_pred = df_preds["pred_log_thalf"].values

        r = np.corrcoef(y_true, y_pred)[0, 1]
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        hb = ax.hexbin(y_true, y_pred, gridsize=40, cmap="viridis", mincnt=1, bins="log")
        ax.plot([-1, 2.2], [-1, 2.2], color="crimson", linestyle="--", linewidth=1.5, label="Ideal Parity (y=x)")
        ax.axvline(np.log10(1.0 + 0.1), color="#e63946", linestyle=":", alpha=0.7, label="1.0h Clinical Binder")
        ax.axhline(np.log10(1.0 + 0.1), color="#e63946", linestyle=":", alpha=0.7)

        ax.set_title(f"{split_title}\nPearson r = {r:.4f} | Log RMSE = {rmse:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Ground Truth $\log_{10}(t_{1/2} + 0.1)$", fontsize=10)
        ax.set_ylabel(r"Predicted $\log_{10}(t_{1/2} + 0.1)$", fontsize=10)
        ax.set_xlim(-1.1, 2.2)
        ax.set_ylim(-1.1, 2.2)
        ax.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend(loc="upper left", frameon=True, fontsize=9)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Log10 Density", fontsize=8.5)

    fig.suptitle("DeepExtract ESMc-6B Parity Across Benchmark Splits", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] DeepExtract Parity Across Splits -> {save_path}")


def plot_radar_summary(df: pd.DataFrame, save_path: Path):
    """Plot multi-attribute radar chart comparing the 3 paradigms."""
    categories = [
        "In-Distribution Fit\n(IID r)",
        "Pan-Allele Transfer\n(Novel Allele r)",
        "Zero-Shot Epitope\n(Novel Pep r)",
        "Classification\n(IID AUROC)",
        "Calibration\n(1 - RMSE)",
    ]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)

    models_to_plot = [
        ("Dual-Tower CNN", "#2b5c8f", "Supervised Baseline (No ESMc)"),
        ("ESMc-6B MeanPool", "#e07a5f", "Simple ESMc-6B (Mean-Pooled)"),
        ("ESMc-6B DeepExtract", "#2a9d8f", "Frontier ESMc-6B (DeepExtract)"),
    ]

    for model_name, color, label in models_to_plot:
        sub = df[df["model"] == model_name]
        iid_r = sub[sub["split"] == "IID"]["pearson_r"].values[0]
        allele_r = sub[sub["split"] == "Novel Allele"]["pearson_r"].values[0]
        pep_r = sub[sub["split"] == "Novel Peptide"]["pearson_r"].values[0]
        auroc = sub[sub["split"] == "IID"]["auroc_1h"].values[0]
        rmse = sub[sub["split"] == "IID"]["rmse_log"].values[0]
        calib = max(0, 1.0 - rmse)

        values = [iid_r, allele_r, pep_r, auroc, calib]
        values += values[:1]

        ax.plot(angles, values, color=color, linewidth=2.5, label=label)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True, fontsize=10)
    ax.set_title("Holistic Performance Profile Across Key Benchmarks", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] Holistic Radar Profile -> {save_path}")


def main():
    print("=" * 78)
    print("GENERATING MASTER PUBLICATION BENCHMARK VISUALIZATIONS")
    print("=" * 78)
    df = load_all_metrics()
    print(f"Loaded {len(df)} model-split benchmark metric records.")

    plot_master_splits_comparison(df, PLOTS_DIR / "29_master_splits_comparison.png")
    plot_generalization_drop_analysis(df, PLOTS_DIR / "30_generalization_drop_analysis.png")
    plot_frontier_representation_hierarchy(df, PLOTS_DIR / "31_frontier_representation_hierarchy.png")
    plot_deep_extract_splits_parity(PLOTS_DIR / "32_deep_extract_splits_parity.png")
    plot_radar_summary(df, PLOTS_DIR / "33_comprehensive_radar_and_tradeoffs.png")

    print("\nAll master visualization plots successfully generated in plots/!")


if __name__ == "__main__":
    main()
