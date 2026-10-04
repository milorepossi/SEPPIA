#!/usr/bin/env python3
"""Generate composite predicted vs true comparison plots across Baseline and Boltz-2 models.

Reads:
  - RESULTS/baseline_test_predictions.npz
  - RESULTS/boltz/BZZU_pca20_test_predictions.npz
  - RESULTS/boltz/BZZU_flatten_test_predictions.npz

Writes:
  - RESULTS/boltz/predicted_vs_true_comparison.png (3-column, 2-row layout)
  - RESULTS/boltz/decile_shrinkage_overlay.png
"""
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

COLORS = {
    'surface': '#fcfcfb',
    'ink': '#1b1b1a',
    'secondary': '#5c5c58',
    'muted': '#8c8c85',
    'hair': '#e8e8e3',
    'axis': '#dcdcd5',
    'grid': '#ebebe6',
    'train': '#2a6f97',
    'validation': '#e07a5f',
    'baseline': '#a8dadc',
    'target': '#457b9d',
}

def style_axes(ax):
    ax.set_facecolor(COLORS['surface'])
    ax.grid(axis='both', color=COLORS['grid'], linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(COLORS['axis'])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=COLORS['muted'], labelsize=8, length=3)

def compute_deciles(truth, predicted):
    edges = np.quantile(truth, np.linspace(0, 1, 11))
    centers, medians = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (truth >= low) & (truth <= high)
        if mask.sum() > 20:
            centers.append(float(np.median(truth[mask])))
            medians.append(float(np.median(predicted[mask])))
    return np.array(centers), np.array(medians)

def generate_comparison_grid(models, out_path, epsilon=0.1):
    fig = Figure(figsize=(15, 9.5), layout='constrained', facecolor=COLORS['surface'])
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, len(models), sharex=True)

    all_truth = np.concatenate([m['truth'] for m in models])
    all_pred = np.concatenate([m['predicted'] for m in models])
    all_resid = all_pred - all_truth
    limits = (min(all_truth.min(), all_pred.min()) - 0.2, max(all_truth.max(), all_pred.max()) + 0.2)
    resid_limits = (-4.0, 4.0)

    for col, m in enumerate(models):
        truth, predicted = m['truth'], m['predicted']
        resid = predicted - truth
        centers, medians = compute_deciles(truth, predicted)

        # Row 0: Predicted vs Measured
        ax_top = axes[0, col]
        hexes = ax_top.hexbin(truth, predicted, gridsize=50, bins='log', cmap='Blues',
                              mincnt=1, linewidths=0.2, extent=(*limits, *limits))
        ax_top.plot(limits, limits, color=COLORS['ink'], linewidth=1.2,
                    linestyle=(0, (4, 3)), label='Perfect prediction', zorder=3)
        ax_top.plot(centers, medians, color=COLORS['validation'], linewidth=2.0,
                    marker='o', markersize=4.5, label='Median / decile', zorder=4)
        ax_top.set(xlim=limits, ylim=limits)
        ax_top.set_aspect('equal')
        ax_top.set_ylabel(f'Predicted ln(half-life + {epsilon:g})', color=COLORS['secondary'], fontsize=9)
        title_text = f"{m['title']}\nRMSE: {m['rmse']:.3f} | Spearman: {m['spearman']:.3f}"
        ax_top.set_title(title_text, color=COLORS['ink'], fontsize=10.5, fontweight='bold')
        if col == 0:
            ax_top.legend(frameon=False, fontsize=8, labelcolor=COLORS['secondary'], loc='upper left')
        style_axes(ax_top)

        # Row 1: Residual vs Measured
        ax_bot = axes[1, col]
        ax_bot.hexbin(truth, resid, gridsize=50, bins='log', cmap='Blues', mincnt=1, linewidths=0.2)
        ax_bot.axhline(0, color=COLORS['ink'], linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
        ax_bot.plot(centers, medians - centers, color=COLORS['validation'],
                    linewidth=2.0, marker='o', markersize=4.5, zorder=4)
        ax_bot.set_ylim(resid_limits)
        ax_bot.set_xlabel(f'Measured ln(half-life + {epsilon:g})', color=COLORS['secondary'], fontsize=9)
        ax_bot.set_ylabel('Residual (predicted \u2212 measured)', color=COLORS['secondary'], fontsize=9)
        bias_text = f"Mean Bias: {m['bias']:+.4f}"
        ax_bot.set_title(bias_text, color=COLORS['ink'], fontsize=9.5)
        style_axes(ax_bot)

    fig.suptitle('Test Predictions & Residuals Pooled Over Five Splits (n = 42,250 complexes)',
                 color=COLORS['ink'], fontsize=13, fontweight='bold')
    fig.savefig(out_path, dpi=200, facecolor=COLORS['surface'])
    print(f"Saved {out_path}")

def generate_decile_overlay(models, out_path, epsilon=0.1):
    fig = Figure(figsize=(7.5, 6.5), layout='constrained', facecolor=COLORS['surface'])
    FigureCanvasAgg(fig)
    ax = fig.subplots(1, 1)

    all_truth = np.concatenate([m['truth'] for m in models])
    all_pred = np.concatenate([m['predicted'] for m in models])
    limits = (-2.6, 3.8)
    ax.plot([-3, 4], [-3, 4], color='#888888', linewidth=1.5, linestyle='--', label='Perfect prediction (identity)', zorder=2)

    model_colors = ['#e07a5f', '#2a6f97', '#1b4332']
    model_styles = ['-o', '-s', '-^']

    for i, m in enumerate(models):
        centers, medians = compute_deciles(m['truth'], m['predicted'])
        ax.plot(centers, medians, model_styles[i], color=model_colors[i], linewidth=2.0,
                markersize=6, label=f"{m['name']} (RMSE {m['rmse']:.3f}, \u03c1 {m['spearman']:.3f})", zorder=4+i)

    ax.set(xlim=limits, ylim=limits)
    ax.set_aspect('equal')
    ax.set_xlabel(f'Measured ln(half-life + {epsilon:g})', color=COLORS['secondary'], fontsize=10)
    ax.set_ylabel(f'Predicted ln(half-life + {epsilon:g})', color=COLORS['secondary'], fontsize=10)
    ax.set_title('Shrinkage Comparison: Median Prediction per Decile', color=COLORS['ink'], fontsize=11.5, fontweight='bold')
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor=COLORS['axis'], fontsize=8.5, loc='upper left')
    style_axes(ax)

    fig.savefig(out_path, dpi=200, facecolor=COLORS['surface'])
    print(f"Saved {out_path}")

def main():
    import scipy.stats
    base_dir = Path("RESULTS")
    boltz_dir = base_dir / "boltz"

    # Load baseline
    base_npz = np.load(base_dir / "baseline_test_predictions.npz")
    base_err = base_npz['predicted'] - base_npz['truth']
    base_rmse = float(np.sqrt(np.mean(base_err**2)))
    base_bias = float(np.mean(base_err))
    base_spearman = float(scipy.stats.spearmanr(base_npz['truth'], base_npz['predicted']).statistic)

    # Load BZZU pca:20
    pca_npz = np.load(boltz_dir / "BZZU_pca20_test_predictions.npz")
    pca_err = pca_npz['predicted'] - pca_npz['truth']
    pca_rmse = float(np.sqrt(np.mean(pca_err**2)))
    pca_bias = float(np.mean(pca_err))
    pca_spearman = float(scipy.stats.spearmanr(pca_npz['truth'], pca_npz['predicted']).statistic)

    # Load BZZU flatten
    flat_npz = np.load(boltz_dir / "BZZU_flatten_test_predictions.npz")
    flat_err = flat_npz['predicted'] - flat_npz['truth']
    flat_rmse = float(np.sqrt(np.mean(flat_err**2)))
    flat_bias = float(np.mean(flat_err))
    flat_spearman = float(scipy.stats.spearmanr(flat_npz['truth'], flat_npz['predicted']).statistic)

    models = [
        {
            'name': 'Baseline A0 One-Hot',
            'title': 'Baseline A0 One-Hot\n(860 features)',
            'truth': base_npz['truth'],
            'predicted': base_npz['predicted'],
            'rmse': base_rmse,
            'bias': base_bias,
            'spearman': base_spearman,
        },
        {
            'name': 'Boltz-2 BZZU (pca:20)',
            'title': 'Boltz-2 BZZU Pair Contraction\n(pca:20, 860 features)',
            'truth': pca_npz['truth'],
            'predicted': pca_npz['predicted'],
            'rmse': pca_rmse,
            'bias': pca_bias,
            'spearman': pca_spearman,
        },
        {
            'name': 'Boltz-2 BZZU (flatten)',
            'title': 'Boltz-2 BZZU Pair Contraction\n(flatten, 5,504 features)',
            'truth': flat_npz['truth'],
            'predicted': flat_npz['predicted'],
            'rmse': flat_rmse,
            'bias': flat_bias,
            'spearman': flat_spearman,
        },
    ]

    generate_comparison_grid(models, boltz_dir / "predicted_vs_true_comparison.png")
    generate_decile_overlay(models, boltz_dir / "decile_shrinkage_overlay.png")

if __name__ == '__main__':
    main()
