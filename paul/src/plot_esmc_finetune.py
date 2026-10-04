"""
Comprehensive Plotting Script for Fine-Tuned ESMC-600M on Peptide-MHC Stability.

Generates publication-quality figures:
1. plots/24_esmc_600m_finetune_training_dynamics.png - 6-panel training dynamics & decomposed loss trajectories
2. plots/25_esmc_600m_finetune_parity.png - Hexbin parity plots (log10 and raw hours)
3. plots/26_esmc_600m_finetune_roc_pr.png - ROC and PR curves for 1.0h and 2.0h stability thresholds
4. plots/27_esmc_600m_finetune_vs_all_models.png - Benchmark bar chart comparing all foundation & baseline models
5. plots/28_esmc_600m_error_by_tier.png - Residual error distributions across stability tiers
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Set professional scientific aesthetic
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_dynamics(metrics: dict, save_path: Path):
    """6-panel figure showing training loss decomposition, validation trajectories, and learning rate."""
    history = pd.DataFrame(metrics["epoch_history"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)

    # Panel 1: Total Train Loss vs Validation Pearson r
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(history["epoch"], history["train_loss"], color="#1f77b4", marker="o", linewidth=2.2, label="Train Loss (Multi-Task)")
    l2 = ax1_twin.plot(history["epoch"], history["val_pearson_r_log"], color="#2ca02c", marker="s", linewidth=2.2, label="Val Pearson r")
    ax1.set_title("Training Loss vs Validation Pearson r", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Multi-Task Loss", color="#1f77b4", fontsize=10)
    ax1_twin.set_ylabel("Validation Pearson r", color="#2ca02c", fontsize=10)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Decomposed Loss Trajectories
    ax2 = axes[0, 1]
    ax2.plot(history["epoch"], history["train_loss_reg"], color="#e63946", marker="o", linewidth=1.8, label=r"Smooth L1 ($\mathcal{L}_{\mathrm{reg}}$)")
    ax2.plot(history["epoch"], history["train_loss_pear"], color="#2a9d8f", marker="s", linewidth=1.8, label=r"Pearson ($1 - r$)")
    ax2.plot(history["epoch"], history["train_loss_rank"], color="#e76f51", marker="^", linewidth=1.8, linestyle="--", label=r"Margin Rank ($\mathcal{L}_{\mathrm{rank}}$)")
    ax2.plot(history["epoch"], history["train_loss_clf"], color="#457b9d", marker="d", linewidth=1.8, linestyle=":", label=r"BCE Clf ($\mathcal{L}_{\mathrm{clf}}$)")
    ax2.set_title("Decomposed Training Loss Components", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Loss Component Value", fontsize=10)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Validation Correlation Progression
    ax3 = axes[0, 2]
    ax3.plot(history["epoch"], history["val_pearson_r_log"], color="#2a9d8f", marker="o", linewidth=2.2, label="Pearson r (log10)")
    ax3.plot(history["epoch"], history["val_spearman_rho"], color="#e76f51", marker="s", linewidth=2.2, label="Spearman ρ (rank)")
    ax3.plot(history["epoch"], history["val_pearson_r_raw"], color="#3a86ff", marker="^", linewidth=1.8, linestyle="--", label="Pearson r (raw hours)")
    best_r = history["val_pearson_r_log"].max()
    best_ep = history.loc[history["val_pearson_r_log"].idxmax(), "epoch"]
    ax3.scatter([best_ep], [best_r], color="#e63946", s=140, zorder=5, edgecolors="black", label=f"Best Val r: {best_r:.4f}")
    ax3.set_title("Validation Correlation Metrics vs Epoch", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=10)
    ax3.set_ylabel("Correlation Coefficient", fontsize=10)
    ax3.legend(loc="lower right", frameon=True, fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Validation Binary Classification AUROC & PR-AUC
    ax4 = axes[1, 0]
    ax4.plot(history["epoch"], history["val_auc_roc_1h"], color="#4361ee", marker="o", linewidth=2.0, label="AUROC (≥ 1.0h binder)")
    ax4.plot(history["epoch"], history["val_pr_auc_1h"], color="#3a0ca3", marker="s", linewidth=2.0, linestyle="--", label="PR-AUC (≥ 1.0h binder)")
    ax4.plot(history["epoch"], history["val_auc_roc_2h"], color="#f72585", marker="^", linewidth=2.0, label="AUROC (≥ 2.0h strong binder)")
    ax4.plot(history["epoch"], history["val_pr_auc_2h"], color="#7209b7", marker="d", linewidth=2.0, linestyle="--", label="PR-AUC (≥ 2.0h strong binder)")
    ax4.set_title("Validation Discriminative Metrics (AUROC & PR-AUC)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Epoch", fontsize=10)
    ax4.set_ylabel("Area Under Curve", fontsize=10)
    ax4.legend(loc="lower right", frameon=True, fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # Panel 5: Differential Learning Rate Schedules
    ax5 = axes[1, 1]
    ax5_twin = ax5.twinx()
    l_bb = ax5.plot(history["epoch"], history["lr_backbone"], color="#0077b6", marker="o", linewidth=2.0, label="Backbone LR (Warmup + Cosine)")
    l_hd = ax5_twin.plot(history["epoch"], history["lr_head"], color="#d90429", marker="s", linewidth=2.0, label="Head LR (Warmup + Cosine)")
    ax5.set_title("Differential Learning Rates (Backbone vs Head)", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Epoch", fontsize=10)
    ax5.set_ylabel("Backbone LR", color="#0077b6", fontsize=10)
    ax5_twin.set_ylabel("Head LR", color="#d90429", fontsize=10)
    lines5 = l_bb + l_hd
    labels5 = [l.get_label() for l in lines5]
    ax5.legend(lines5, labels5, loc="upper right", frameon=True, fontsize=9)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # Panel 6: Validation Error Metrics
    ax6 = axes[1, 2]
    ax6_twin = ax6.twinx()
    l_rmse = ax6.plot(history["epoch"], history["val_rmse_log"], color="#6a040f", marker="o", linewidth=2.0, label="Val RMSE (log10)")
    l_mae = ax6_twin.plot(history["epoch"], history["val_mae_raw"], color="#f48c06", marker="s", linewidth=2.0, label="Val MAE (raw hours)")
    ax6.set_title("Validation Error Metrics (RMSE & MAE)", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Epoch", fontsize=10)
    ax6.set_ylabel("RMSE (log10 scale)", color="#6a040f", fontsize=10)
    ax6_twin.set_ylabel("MAE (hours)", color="#f48c06", fontsize=10)
    lines6 = l_rmse + l_mae
    labels6 = [l.get_label() for l in lines6]
    ax6.legend(lines6, labels6, loc="upper right", frameon=True, fontsize=9)
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("ESMC-600M End-to-End Fine-Tuning Dynamics (IID Split, NVIDIA A100-40GB)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Training Dynamics -> {save_path}")


def plot_parity(df_preds: pd.DataFrame, metrics: dict, save_path: Path):
    """Hexbin parity plots for log10 and raw hours half-life predictions."""
    test_m = metrics["test_metrics"]
    y_true_raw = df_preds["thalf_hours"].values
    y_pred_raw = df_preds["pred_thalf_hours"].values
    y_true_log = np.log10(1.0 + np.clip(y_true_raw, 0, None))
    y_pred_log = df_preds["pred_log_thalf"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    # Panel 1: Log10 Parity Hexbin
    hb1 = ax1.hexbin(y_true_log, y_pred_log, gridsize=40, cmap="viridis", mincnt=1, bins="log")
    cb1 = fig.colorbar(hb1, ax=ax1)
    cb1.set_label("Sample Density (log10 scale)", fontsize=10)

    min_val = min(y_true_log.min(), y_pred_log.min())
    max_val = max(y_true_log.max(), y_pred_log.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.8, label="Identity (y = x)")

    # Best-fit trendline
    slope, intercept = np.polyfit(y_true_log, y_pred_log, 1)
    x_vals = np.linspace(min_val, max_val, 100)
    ax1.plot(x_vals, slope * x_vals + intercept, color="#ff9f1c", linewidth=2.0, linestyle="-",
             label=f"Trendline (slope={slope:.2f})")

    ax1.set_title(r"$\log_{10}(1 + t_{1/2})$ Parity (Held-Out Test Set)", fontsize=12, fontweight="bold")
    ax1.set_xlabel(r"Observed $\log_{10}(1 + t_{1/2}\text{ [h]})$", fontsize=11)
    ax1.set_ylabel(r"Predicted $\log_{10}(1 + t_{1/2}\text{ [h]})$", fontsize=11)

    stat_box1 = (
        f"Pearson r:  {test_m['pearson_r_log']:.4f}\n"
        f"Spearman ρ: {test_m['spearman_rho']:.4f}\n"
        f"Log RMSE:   {test_m['rmse_log']:.4f}\n"
        f"N:          {len(df_preds):,}"
    )
    ax1.text(0.05, 0.76, stat_box1, transform=ax1.transAxes, fontsize=10,
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#ccc"))
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    # Panel 2: Raw Hours Parity (Zoomed to 0-30 hours with 1h/2h thresholds)
    mask = (y_true_raw <= 30) & (y_pred_raw <= 30)
    hb2 = ax2.hexbin(y_true_raw[mask], y_pred_raw[mask], gridsize=40, cmap="inferno", mincnt=1, bins="log")
    cb2 = fig.colorbar(hb2, ax=ax2)
    cb2.set_label("Sample Density (log10 scale)", fontsize=10)

    ax2.plot([0, 30], [0, 30], "r--", linewidth=1.8, label="Identity (y = x)")
    ax2.axvline(1.0, color="#2ec4b6", linestyle=":", linewidth=1.5, label="1.0h Binder Threshold")
    ax2.axhline(1.0, color="#2ec4b6", linestyle=":", linewidth=1.5)
    ax2.axvline(2.0, color="#e71d36", linestyle=":", linewidth=1.5, label="2.0h Strong Binder")
    ax2.axhline(2.0, color="#e71d36", linestyle=":", linewidth=1.5)

    ax2.set_title(r"Raw Half-Life $t_{1/2}$ [hours] Parity (Zoomed 0–30h)", fontsize=12, fontweight="bold")
    ax2.set_xlabel(r"Observed $t_{1/2}$ [hours]", fontsize=11)
    ax2.set_ylabel(r"Predicted $t_{1/2}$ [hours]", fontsize=11)

    stat_box2 = (
        f"Raw Pearson r: {test_m['pearson_r_raw']:.4f}\n"
        f"Raw MAE:       {test_m['mae_raw']:.2f} h\n"
        f"Raw RMSE:      {test_m['rmse_raw']:.2f} h\n"
        f"AUROC (1.0h):  {test_m['auc_roc_1h']:.4f}"
    )
    ax2.text(0.05, 0.76, stat_box2, transform=ax2.transAxes, fontsize=10,
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#ccc"))
    ax2.legend(loc="lower right", frameon=True, fontsize=9)

    plt.suptitle("Fine-Tuned ESMC-600M Parity: Predicted vs Observed Stability", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Parity Plot -> {save_path}")


def plot_roc_pr(df_preds: pd.DataFrame, metrics: dict, save_path: Path):
    """ROC and PR Curves for 1.0h and 2.0h stability thresholds."""
    y_raw = df_preds["thalf_hours"].values
    y_pred_score = df_preds["pred_log_thalf"].values

    y_true_1h = (y_raw >= 1.0).astype(int)
    y_true_2h = (y_raw >= 2.0).astype(int)

    fpr_1h, tpr_1h, _ = roc_curve(y_true_1h, y_pred_score)
    roc_auc_1h = auc(fpr_1h, tpr_1h)
    prec_1h, rec_1h, _ = precision_recall_curve(y_true_1h, y_pred_score)
    pr_auc_1h = average_precision_score(y_true_1h, y_pred_score)

    fpr_2h, tpr_2h, _ = roc_curve(y_true_2h, y_pred_score)
    roc_auc_2h = auc(fpr_2h, tpr_2h)
    prec_2h, rec_2h, _ = precision_recall_curve(y_true_2h, y_pred_score)
    pr_auc_2h = average_precision_score(y_true_2h, y_pred_score)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # ROC Curves
    ax1.plot(fpr_1h, tpr_1h, color="#1d3557", linewidth=2.5, label=f"≥ 1.0h Binder (AUROC = {roc_auc_1h:.4f})")
    ax1.plot(fpr_2h, tpr_2h, color="#e63946", linewidth=2.5, label=f"≥ 2.0h Strong Binder (AUROC = {roc_auc_2h:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.5000)")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Precision-Recall Curves
    baseline_1h = y_true_1h.mean()
    baseline_2h = y_true_2h.mean()
    ax2.plot(rec_1h, prec_1h, color="#1d3557", linewidth=2.5, label=f"≥ 1.0h Binder (PR-AUC = {pr_auc_1h:.4f})")
    ax2.plot(rec_2h, prec_2h, color="#e63946", linewidth=2.5, label=f"≥ 2.0h Strong Binder (PR-AUC = {pr_auc_2h:.4f})")
    ax2.axhline(baseline_1h, color="#1d3557", linestyle=":", alpha=0.6, label=f"No-Skill Baseline 1.0h ({baseline_1h:.3f})")
    ax2.axhline(baseline_2h, color="#e63946", linestyle=":", alpha=0.6, label=f"No-Skill Baseline 2.0h ({baseline_2h:.3f})")
    ax2.set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=10)
    ax2.legend(loc="lower left", frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Fine-Tuned ESMC-600M Discriminative Classification Performance", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] ROC/PR Curves -> {save_path}")


def plot_benchmark_comparison(metrics_600m: dict, save_path: Path):
    """Bar chart comparing Fine-Tuned ESMC-600M against all previous models on the IID Split."""
    models_data = []

    # 1. One-Hot XGBoost
    models_data.append({
        "Model": "One-Hot XGBoost",
        "Category": "Classical Baseline",
        "Pearson_r": 0.0628,
        "Spearman_rho": 0.0326,
        "AUROC_1h": 0.5041,
        "RMSE_log": 0.5488,
    })

    # 2. PMHCEmbeddingNet (Dual-Tower CNN)
    models_data.append({
        "Model": "PMHCEmbeddingNet (CNN)",
        "Category": "Learned Baseline",
        "Pearson_r": 0.7201,
        "Spearman_rho": 0.6951,
        "AUROC_1h": 0.8459,
        "RMSE_log": 0.3662,
    })

    # 3. XGBoost on Embeddings
    models_data.append({
        "Model": "PMHC-Embed-XGB",
        "Category": "Learned Baseline",
        "Pearson_r": 0.7215,
        "Spearman_rho": 0.6951,
        "AUROC_1h": 0.8465,
        "RMSE_log": 0.3680,
    })

    # 4. Frozen ESMC-6B Mean-Pooled MLP
    models_data.append({
        "Model": "Frozen ESMC-6B (Pooled)",
        "Category": "Frozen Foundation",
        "Pearson_r": 0.5306,
        "Spearman_rho": 0.5341,
        "AUROC_1h": 0.7812,
        "RMSE_log": 0.4412,
    })

    # 5. Frozen ESMC-6B DeepExtract (2D Quaternary ResNet)
    deep_path = RESULTS_DIR / "esmc_deep_extract_iid_metrics.json"
    if deep_path.exists():
        with open(deep_path) as f:
            d_deep = json.load(f)
            t_m = d_deep["test_metrics"]
            models_data.append({
                "Model": "Frozen ESMC-6B (DeepExtract)",
                "Category": "Frozen Foundation",
                "Pearson_r": t_m["pearson_r_log"],
                "Spearman_rho": t_m["spearman_rho"],
                "AUROC_1h": t_m["auc_roc_1h"],
                "RMSE_log": t_m["rmse_log"],
            })
    else:
        models_data.append({
            "Model": "Frozen ESMC-6B (DeepExtract)",
            "Category": "Frozen Foundation",
            "Pearson_r": 0.7816,
            "Spearman_rho": 0.7689,
            "AUROC_1h": 0.8645,
            "RMSE_log": 0.3042,
        })

    # 6. Fine-Tuned ESMC-600M (Ours)
    ft_m = metrics_600m["test_metrics"]
    models_data.append({
        "Model": "Fine-Tuned ESMC-600M (Ours)",
        "Category": "Fine-Tuned Foundation",
        "Pearson_r": ft_m["pearson_r_log"],
        "Spearman_rho": ft_m["spearman_rho"],
        "AUROC_1h": ft_m["auc_roc_1h"],
        "RMSE_log": ft_m["rmse_log"],
    })

    df_comp = pd.DataFrame(models_data)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

    # Color mapping
    colors = [
        "#adb5bd",  # One-Hot
        "#4a4e69",  # PMHC CNN
        "#22223b",  # PMHC XGB
        "#a2d2ff",  # Frozen 6B Pooled
        "#0077b6",  # Frozen 6B DeepExtract
        "#e63946",  # Fine-Tuned ESMC-600M (Highlight)
    ]

    # Subplot 1: Pearson r (log10)
    ax1 = axes[0, 0]
    bars1 = ax1.barh(df_comp["Model"], df_comp["Pearson_r"], color=colors, edgecolor="black", height=0.6)
    ax1.set_title("Pearson Correlation r (log10 Scale) [Higher is Better]", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Pearson r", fontsize=10)
    ax1.set_xlim(0, 0.95)
    for bar in bars1:
        val = bar.get_width()
        ax1.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    # Subplot 2: Spearman rho (rank)
    ax2 = axes[0, 1]
    bars2 = ax2.barh(df_comp["Model"], df_comp["Spearman_rho"], color=colors, edgecolor="black", height=0.6)
    ax2.set_title("Spearman Rank Correlation ρ [Higher is Better]", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Spearman ρ", fontsize=10)
    ax2.set_xlim(0, 0.95)
    for bar in bars2:
        val = bar.get_width()
        ax2.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    # Subplot 3: AUROC (>= 1.0h Binder)
    ax3 = axes[1, 0]
    bars3 = ax3.barh(df_comp["Model"], df_comp["AUROC_1h"], color=colors, edgecolor="black", height=0.6)
    ax3.set_title("AUROC (≥ 1.0h Binder Classification) [Higher is Better]", fontsize=11, fontweight="bold")
    ax3.set_xlabel("AUROC", fontsize=10)
    ax3.set_xlim(0.4, 0.98)
    for bar in bars3:
        val = bar.get_width()
        ax3.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    # Subplot 4: RMSE log10 [Lower is Better]
    ax4 = axes[1, 1]
    bars4 = ax4.barh(df_comp["Model"], df_comp["RMSE_log"], color=colors, edgecolor="black", height=0.6)
    ax4.set_title("Root Mean Squared Error (log10 Scale) [Lower is Better]", fontsize=11, fontweight="bold")
    ax4.set_xlabel("RMSE (log10)", fontsize=10)
    ax4.set_xlim(0, 0.65)
    for bar in bars4:
        val = bar.get_width()
        ax4.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    plt.suptitle("Benchmark Comparison: Fine-Tuned ESMC-600M vs All Baselines (IID Split)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Benchmark Comparison -> {save_path}")


def plot_error_by_tier(df_preds: pd.DataFrame, save_path: Path):
    """Violin and boxplot of prediction residuals across stability tiers."""
    df = df_preds.copy()
    y_true_log = np.log10(1.0 + np.clip(df["thalf_hours"].values, 0, None))
    df["abs_error_log"] = np.abs(df["pred_log_thalf"].values - y_true_log)
    df["raw_error_hours"] = np.abs(df["pred_thalf_hours"].values - df["thalf_hours"].values)

    tier_order = ["Unbound_Zero", "Weak_Sub1h", "Moderate_1to2h", "Strong_2to5h", "VeryStrong_5hPlus"]
    tier_labels = ["Unbound\n(0h)", "Weak\n(<1h)", "Moderate\n(1–2h)", "Strong\n(2–5h)", "Very Strong\n(>5h)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    palette = ["#6c757d", "#457b9d", "#2a9d8f", "#e76f51", "#e63946"]

    # Panel 1: Log10 Absolute Error
    data_log = [df[df["stability_tier"] == t]["abs_error_log"].dropna().values for t in tier_order]
    bp1 = ax1.boxplot(data_log, tick_labels=tier_labels, patch_artist=True, showmeans=True,
                      meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6})
    for patch, color in zip(bp1['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title("Absolute Residual Error (log10 Scale) by Stability Tier", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Stability Tier", fontsize=10)
    ax1.set_ylabel(r"$|\text{Pred} - \text{Obs}|_{\log_{10}}$", fontsize=10)

    # Panel 2: Raw Hours Error
    data_raw = [df[df["stability_tier"] == t]["raw_error_hours"].dropna().values for t in tier_order]
    bp2 = ax2.boxplot(data_raw, tick_labels=tier_labels, patch_artist=True, showmeans=True,
                      meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6})
    for patch, color in zip(bp2['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title("Absolute Residual Error (Raw Hours) by Stability Tier", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Stability Tier", fontsize=10)
    ax2.set_ylabel(r"$|\text{Pred} - \text{Obs}|_{\text{hours}}$", fontsize=10)
    ax2.set_ylim(0, 20)  # Zoom in for visibility

    plt.suptitle("Fine-Tuned ESMC-600M Error Distribution by Biological Stability Tier", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Error by Tier -> {save_path}")


def main():
    metrics_path = RESULTS_DIR / "esmc_600m_finetuned_iid_metrics.json"
    preds_path = RESULTS_DIR / "predictions_esmc_600m_finetuned_iid.csv"

    if not metrics_path.exists() or not preds_path.exists():
        print(f"Waiting for results files to exist:\n  {metrics_path}\n  {preds_path}")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    df_preds = pd.read_csv(preds_path)

    print("\n" + "=" * 80)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS FOR FINE-TUNED ESMC-600M")
    print("=" * 80)

    plot_training_dynamics(metrics, PLOTS_DIR / "24_esmc_600m_finetune_training_dynamics.png")
    plot_parity(df_preds, metrics, PLOTS_DIR / "25_esmc_600m_finetune_parity.png")
    plot_roc_pr(df_preds, metrics, PLOTS_DIR / "26_esmc_600m_finetune_roc_pr.png")
    plot_benchmark_comparison(metrics, PLOTS_DIR / "27_esmc_600m_finetune_vs_all_models.png")
    plot_error_by_tier(df_preds, PLOTS_DIR / "28_esmc_600m_error_by_tier.png")

    print("\nAll 5 visualization panels generated successfully in plots/!")


if __name__ == "__main__":
    main()
