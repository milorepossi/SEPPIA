#!/usr/bin/env python3
"""
Publication-Quality Visualization Suite for Baseline Peptide-HLA Models.
Generates:
1. plots/09_baseline_model_comparison.png: Side-by-side metric comparison
   (One-Hot XGBoost vs Pure Neural Net vs Hybrid XGBoost on Learned Embeddings)
2. plots/10_baseline_predicted_vs_actual.png: Parity plots with density contours
3. plots/11_baseline_error_by_stability_tier.png: Error distribution across stability tiers
4. plots/12_baseline_generalization_drop.png: Performance across split regimes (if available)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#e0e0e0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.6


def plot_model_comparison(metrics_json_path: str = "results/baseline_metrics_summary.json", out_dir: str = "plots"):
    metrics_path = Path(metrics_json_path)
    if not metrics_path.exists():
        print(f"Metrics file {metrics_json_path} does not exist yet.")
        return

    with open(metrics_path) as f:
        data = json.load(f)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use 'iid' split for core comparison
    if "iid" not in data:
        split_keys = list(data.keys())
        if not split_keys:
            return
        split_key = split_keys[0]
    else:
        split_key = "iid"

    metrics = data[split_key]["metrics"]
    models = [
        ("1. One-Hot + XGBoost", metrics["onehot_xgboost"], "#3498db"),
        ("2. Pure Neural Net\n(Learned Embeddings)", metrics["neural_network"], "#e67e22"),
        ("3. Hybrid: XGBoost on\nLearned Embeddings", metrics["xgboost_on_embeddings"], "#2ecc71"),
    ]

    metric_names = [
        ("Spearman Rank (\u03c1)", "spearman_rho", (0.0, 1.0)),
        ("Pearson r (log)", "pearson_r_log", (0.0, 1.0)),
        ("RMSE Hours (Lower is better)", "rmse_raw", None),
        ("AUC-ROC (\u2265 1.0h Binder)", "auc_roc_1h", (0.5, 1.0)),
        ("AUC-ROC (\u2265 2.0h Strong)", "auc_roc_2h", (0.5, 1.0)),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)
    fig.suptitle(
        f"Baseline Model Comparison on Peptide-HLA Complex Stability ({split_key.upper()} Test Set, n={data[split_key]['n_test']:,})",
        fontsize=16,
        fontweight="bold",
        y=1.03,
    )

    for ax, (title, key, ylim) in zip(axes, metric_names):
        values = [m[1][key] for m in models]
        labels = [m[0] for m in models]
        colors = [m[2] for m in models]

        bars = ax.bar(labels, values, color=colors, alpha=0.88, edgecolor="#222222", linewidth=1.0, width=0.6)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.grid(axis="y")

        if ylim:
            ax.set_ylim(ylim)

        # Annotate bar values
        for bar, val in zip(bars, values):
            y_pos = bar.get_height()
            offset = 0.02 if ylim else (max(values) * 0.02)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos + offset,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    out_path = out_dir / "09_baseline_model_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_predicted_vs_actual(pred_csv_path: str = "results/predictions_iid.csv", out_dir: str = "plots"):
    csv_path = Path(pred_csv_path)
    if not csv_path.exists():
        print(f"Predictions file {pred_csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
    models_to_plot = [
        ("1. One-Hot + XGBoost", "pred_hours_xgb_onehot", "#2980b9"),
        ("2. Pure Neural Net (Learned Embeddings)", "pred_hours_nn", "#d35400"),
        ("3. Hybrid: XGBoost on Learned Embeddings", "pred_hours_xgb_embed", "#27ae60"),
    ]

    min_val, max_val = 0.05, 120.0  # Log-scale limits

    for ax, (name, col, color) in zip(axes, models_to_plot):
        y_true = np.clip(df["thalf_hours"].values, min_val, max_val)
        y_pred = np.clip(df[col].values, min_val, max_val)

        # Hexbin density plot for clarity with ~4,200 points
        hb = ax.hexbin(
            y_true,
            y_pred,
            xscale="log",
            yscale="log",
            gridsize=45,
            cmap="viridis",
            mincnt=1,
            alpha=0.85,
        )

        # Identity diagonal
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Alignment (y = x)")

        # Threshold guide lines
        ax.axvline(1.0, color="#888888", linestyle=":", alpha=0.7)
        ax.axhline(1.0, color="#888888", linestyle=":", alpha=0.7)

        # Compute stats
        sp_r = df[["thalf_hours", col]].corr(method="spearman").iloc[0, 1]
        log_corr = np.corrcoef(np.log1p(df["thalf_hours"]), np.log1p(np.maximum(0, df[col])))[0, 1]
        rmse = np.sqrt(np.mean((df["thalf_hours"] - np.maximum(0, df[col])) ** 2))

        ax.set_title(f"{name}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Measured Half-Life (hours, log scale)", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Predicted Half-Life (hours, log scale)", fontsize=11)

        # Stats box
        stats_text = f"Spearman \u03c1: {sp_r:.3f}\nPearson r (log): {log_corr:.3f}\nRMSE: {rmse:.2f}h"
        ax.text(
            0.05,
            0.93,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
        )
        ax.grid(True, which="both", alpha=0.4)

    fig.suptitle(
        "Measured vs. Predicted Peptide-HLA Complex Half-Life (IID Test Set, n=4,225)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = out_dir / "10_baseline_predicted_vs_actual.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_error_by_stability_tier(pred_csv_path: str = "results/predictions_iid.csv", out_dir: str = "plots"):
    csv_path = Path(pred_csv_path)
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)

    def categorize_tier(t):
        if t == 0:
            return "Unbound (0h)"
        elif t < 1.0:
            return "Weak (<1h)"
        elif t < 2.0:
            return "Moderate (1-2h)"
        elif t < 5.0:
            return "Strong (2-5h)"
        else:
            return "Very Strong (5h+)"

    df["tier"] = df["thalf_hours"].apply(categorize_tier)
    tier_order = ["Unbound (0h)", "Weak (<1h)", "Moderate (1-2h)", "Strong (2-5h)", "Very Strong (5h+)"]

    # Compute Absolute Error for each model
    df["err_onehot"] = np.abs(df["thalf_hours"] - np.maximum(0, df["pred_hours_xgb_onehot"]))
    df["err_nn"] = np.abs(df["thalf_hours"] - np.maximum(0, df["pred_hours_nn"]))
    df["err_embed_xgb"] = np.abs(df["thalf_hours"] - np.maximum(0, df["pred_hours_xgb_embed"]))

    mae_summary = []
    for tier in tier_order:
        sub = df[df["tier"] == tier]
        mae_summary.append({
            "tier": tier,
            "count": len(sub),
            "One-Hot + XGBoost": sub["err_onehot"].median(),
            "Pure Neural Net": sub["err_nn"].median(),
            "XGBoost on Learned Embeddings": sub["err_embed_xgb"].median(),
        })

    mdf = pd.DataFrame(mae_summary)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tier_order))
    width = 0.25

    rects1 = ax.bar(x - width, mdf["One-Hot + XGBoost"], width, label="1. One-Hot + XGBoost", color="#3498db")
    rects2 = ax.bar(x, mdf["Pure Neural Net"], width, label="2. Pure Neural Net", color="#e67e22")
    rects3 = ax.bar(x + width, mdf["XGBoost on Learned Embeddings"], width, label="3. XGBoost on Embeddings", color="#2ecc71")

    ax.set_ylabel("Median Absolute Error (hours)", fontsize=12, fontweight="bold")
    ax.set_title("Prediction Error (MAE) by Stability Tier across Baseline Models", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={c:,})" for t, c in zip(mdf["tier"], mdf["count"])], fontsize=11)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis="y")

    plt.tight_layout()
    out_path = out_dir / "11_baseline_error_by_stability_tier.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_split_generalization(metrics_json_path: str = "results/baseline_metrics_summary.json", out_dir: str = "plots"):
    metrics_path = Path(metrics_json_path)
    if not metrics_path.exists():
        return

    with open(metrics_path) as f:
        data = json.load(f)

    if len(data) <= 1:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [k for k in ["iid", "novel_pep", "novel_allele"] if k in data]
    split_labels = {
        "iid": "IID Random Split\n(Seen HLA, Seen Pep)",
        "novel_pep": "Novel Peptide Split\n(Unseen Pep, Seen HLA)",
        "novel_allele": "Novel Allele Split\n(Seen Pep, Unseen HLA)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    models = [
        ("onehot_xgboost", "1. One-Hot + XGBoost", "#3498db"),
        ("neural_network", "2. Pure Neural Net (Learned Embeddings)", "#e67e22"),
        ("xgboost_on_embeddings", "3. Hybrid: XGBoost on Embeddings", "#2ecc71"),
    ]

    for ax, (m_key, m_title, m_color) in zip(axes, models):
        rhos = [data[s]["metrics"][m_key]["spearman_rho"] for s in splits]
        labels = [split_labels.get(s, s) for s in splits]

        bars = ax.bar(labels, rhos, color=m_color, alpha=0.85, edgecolor="#222222", width=0.55)
        ax.set_title(m_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Spearman Rank Correlation (\u03c1)", fontsize=11)
        ax.set_ylim(-0.1, 0.85)
        ax.grid(axis="y")
        ax.axhline(0, color="#888888", linestyle="-", linewidth=0.8)

        for bar, val in zip(bars, rhos):
            y_pos = max(val, 0)
            offset = 0.02
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos + offset,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    fig.suptitle(
        "Generalization Performance Across Biophysical Evaluation Regimes (Spearman \u03c1)",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    out_path = out_dir / "12_baseline_generalization_drop.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_learned_amino_acid_space(ckpt_path: str = "results/pmhc_embedding_net_iid.pt", out_dir: str = "plots"):
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        return

    import torch
    from sklearn.decomposition import PCA

    state_dict = torch.load(ckpt, map_location="cpu")
    if "embedding.weight" not in state_dict:
        return

    weights = state_dict["embedding.weight"].numpy()
    # Indices 1 to 20 correspond to AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    aa_weights = weights[1 : len(AMINO_ACIDS) + 1]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(aa_weights)

    # Biochemical classifications
    groups = {
        "Hydrophobic/Aliphatic (A, V, I, L, M)": (["A", "V", "I", "L", "M"], "#2ecc71"),
        "Aromatic (F, W, Y)": (["F", "W", "Y"], "#9b59b6"),
        "Positively Charged / Basic (K, R, H)": (["K", "R", "H"], "#3498db"),
        "Negatively Charged / Acidic (D, E)": (["D", "E"], "#e74c3c"),
        "Polar Uncharged (S, T, N, Q)": (["S", "T", "N", "Q"], "#f39c12"),
        "Conformational / Special (G, P, C)": (["G", "P", "C"], "#95a5a6"),
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    for grp_name, (aa_list, color) in groups.items():
        sub_coords = [coords[AMINO_ACIDS.index(aa)] for aa in aa_list if aa in AMINO_ACIDS]
        if not sub_coords:
            continue
        xs, ys = zip(*sub_coords)
        ax.scatter(xs, ys, color=color, s=280, label=grp_name, alpha=0.9, edgecolor="#222222", linewidth=1.2)
        for aa in aa_list:
            if aa in AMINO_ACIDS:
                idx = AMINO_ACIDS.index(aa)
                ax.annotate(
                    aa,
                    (coords[idx, 0], coords[idx, 1]),
                    fontsize=12,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    color="white",
                )

    ax.set_title(
        f"Learned Amino Acid Embeddings (PCA Projection, d={weights.shape[1]}\u21922)\n"
        f"Explained Variance: {pca.explained_variance_ratio_.sum() * 100:.1f}%",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.legend(frameon=True, fontsize=10, loc="best")
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "13_learned_amino_acid_embedding_pca.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_all_baseline_plots():
    plot_model_comparison()
    plot_predicted_vs_actual()
    plot_error_by_stability_tier()
    plot_split_generalization()
    plot_learned_amino_acid_space()


if __name__ == "__main__":
    generate_all_baseline_plots()
