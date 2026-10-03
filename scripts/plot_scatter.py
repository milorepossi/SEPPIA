#!/usr/bin/env python3
"""Prediction scatter per arm, and how each one handles the zero half-life spike.

Example:
    python scripts/plot_scatter.py RESULTS/A0_onehot RESULTS/A1_L0_pca20 \
        --embeddings-dir /scratch/.../embeddings --out-dir RESULTS --tag recap

20.2% of rows have thalf_hours exactly 0. Those are left-censored -- the assay
reporting a value below its resolution, not a measured zero -- so they sit as a
spike at ln(0 + epsilon) far below everything else, and squared error alone
hides how an arm treats them. This reports them as a detection problem too: a
row is called unstable when its predicted half-life is at or under the assay's
0.1 h resolution.

Predictions are pooled over the five test sets, which together cover every row
exactly once.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_features
from train_mlp import COLORS, MLP, features_and_target, style_axes

RESOLUTION = 0.1  # hours; the assay's reporting resolution and the target's epsilon


def predictions(run_dir, splits_dir, embeddings_dir=None):
    """Pooled (truth, prediction, is_zero) over the five test sets, in log units."""
    run_dir = Path(run_dir)
    report = json.loads((run_dir/'metrics.json').read_text())
    arm = report.get('arm') or {}
    truth, predicted, zero = [], [], []
    for entry in report['splits']:
        index = entry['split_index']
        checkpoint = torch.load(run_dir/f'mlp_split_{index}.pt', map_location='cpu',
                                weights_only=True)
        with np.load(Path(splits_dir)/f'testing_{index}.npz', allow_pickle=True) as d:
            test = {k: d[k] for k in d.files}
        encoder, standardiser = arm_features.from_checkpoint(checkpoint, embeddings_dir)
        x, y, _ = features_and_target(test, checkpoint['epsilon'], encoder=encoder)
        if standardiser is not None:
            x = standardiser(x)
        model = MLP(checkpoint['features'], hidden=tuple(checkpoint['hidden']),
                    dropout=checkpoint['dropout'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                out.append(model(torch.from_numpy(np.ascontiguousarray(x[start:start+4096]))).numpy())
        truth.append(y)
        predicted.append(np.concatenate(out)*checkpoint['target_scale']+checkpoint['target_center'])
        zero.append(test['thalf_hours'].astype(np.float64) == 0)
    name, pooling = arm.get('name', 'onehot'), arm.get('pooling', '-')
    return dict(label=name if name == 'onehot' else f'{name} {pooling}',
                name=name, pooling=pooling, epsilon=float(checkpoint['epsilon']),
                truth=np.concatenate(truth), predicted=np.concatenate(predicted),
                zero=np.concatenate(zero))


def zero_metrics(run):
    """Treat the zero spike as a detection problem, in hours."""
    epsilon = run['epsilon']
    hours = np.maximum(np.exp(run['predicted'].astype(np.float64))-epsilon, 0.0)
    called = hours <= RESOLUTION          # predicted unstable
    zero = run['zero']                    # measured unstable (thalf == 0)
    tp = int((called & zero).sum()); fn = int((~called & zero).sum())
    tn = int((~called & ~zero).sum()); fp = int((called & ~zero).sum())
    recall = tp/max(tp+fn, 1)             # accuracy on the zero rows
    specificity = tn/max(tn+fp, 1)        # accuracy on the non-zero rows
    error = run['predicted']-run['truth']
    return dict(
        zero_rows=int(zero.sum()), nonzero_rows=int((~zero).sum()),
        recall_zero=recall, specificity_nonzero=specificity,
        balanced=0.5*(recall+specificity),
        precision=tp/max(tp+fp, 1), overall=(tp+tn)/len(zero),
        rmse_zero=float(np.sqrt(np.mean(error[zero]**2))),
        bias_zero=float(np.mean(error[zero])),
        rmse_nonzero=float(np.sqrt(np.mean(error[~zero]**2))),
        bias_nonzero=float(np.mean(error[~zero])))


def figure(runs, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n = len(runs)
    fig = Figure(figsize=(3.6*n, 4.0), layout='constrained', facecolor=COLORS['surface'])
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(1, n, sharex=True, sharey=True))
    low = min(min(r['truth'].min(), r['predicted'].min()) for r in runs)
    high = max(max(r['truth'].max(), r['predicted'].max()) for r in runs)
    limits = (low-0.2, high+0.2)
    for ax, run in zip(axes, runs):
        ax.hexbin(run['truth'], run['predicted'], gridsize=48, bins='log',
                  cmap='Blues', mincnt=1, linewidths=0.2, extent=(*limits, *limits))
        ax.plot(limits, limits, color=COLORS['ink'], linewidth=1.1,
                linestyle=(0, (4, 3)), zorder=3)
        # Median prediction per decile of truth, so curvature is visible.
        edges = np.quantile(run['truth'], np.linspace(0, 1, 11))
        centres, medians = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (run['truth'] >= lo) & (run['truth'] <= hi)
            if mask.sum() > 20:
                centres.append(float(np.median(run['truth'][mask])))
                medians.append(float(np.median(run['predicted'][mask])))
        ax.plot(centres, medians, color=COLORS['validation'], linewidth=2.0,
                marker='o', markersize=4, zorder=4)
        zero_truth = run['truth'][run['zero']][0]
        ax.axvline(zero_truth, color=COLORS['gradient'], linewidth=1.2,
                   linestyle=(0, (2, 2)), zorder=2)
        metrics = zero_metrics(run)
        ax.set_title(f"{run['label']}\nzero-row recall {metrics['recall_zero']:.2f}  "
                     f"bias {metrics['bias_zero']:+.2f}",
                     color=COLORS['ink'], fontsize=10)
        ax.set_xlabel(f"Measured ln(half-life + {run['epsilon']:g})",
                      color=COLORS['secondary'], fontsize=9)
        ax.set(xlim=limits, ylim=limits)
        ax.set_aspect('equal')
        style_axes(ax)
    axes[0].set_ylabel('Predicted', color=COLORS['secondary'], fontsize=9)
    axes[0].annotate('zero spike', (run['truth'][run['zero']][0], limits[1]-0.4),
                     fontsize=7, color=COLORS['gradient'], rotation=90, ha='right')
    fig.suptitle('Predicted against measured, pooled over the five test sets '
                 '(dashed: perfect; orange: median per decile)',
                 color=COLORS['ink'], fontsize=12)
    fig.savefig(path, dpi=180, facecolor=COLORS['surface'])
    return path


def table(runs, path):
    lines = ['# The zero half-life spike', '',
             f'20.2% of rows have `thalf_hours` exactly 0, which is left censoring: the '
             f'assay reporting below its {RESOLUTION:g} h resolution, not a measured '
             f'zero. A row is *called* unstable when its predicted half-life is at or '
             f'under {RESOLUTION:g} h. Predictions pooled over the five test sets.',
             '', '## Detection of the zero rows', '',
             '| Arm | Zero rows | Recall on zero rows | Accuracy on non-zero rows | Balanced | Precision |',
             '|---|---:|---:|---:|---:|---:|']
    for run in runs:
        m = zero_metrics(run)
        lines.append(f'| {run["label"]} | {m["zero_rows"]} | **{m["recall_zero"]:.3f}** | '
                     f'{m["specificity_nonzero"]:.3f} | {m["balanced"]:.3f} | {m["precision"]:.3f} |')
    lines += ['', '## Error on each stratum (log units)', '',
              '| Arm | RMSE on zero rows | Bias on zero rows | RMSE on non-zero | Bias on non-zero |',
              '|---|---:|---:|---:|---:|']
    for run in runs:
        m = zero_metrics(run)
        lines.append(f'| {run["label"]} | {m["rmse_zero"]:.3f} | {m["bias_zero"]:+.3f} | '
                     f'{m["rmse_nonzero"]:.3f} | {m["bias_nonzero"]:+.3f} |')
    lines += ['', 'A positive bias on the zero rows means they are predicted as more '
              'stable than measured, which is what range compression does to a censored '
              'spike: the model will not commit to the extreme.', '',
              'Generated by `scripts/plot_scatter.py`.']
    Path(path).write_text('\n'.join(lines)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--splits-dir', default='DATA')
    parser.add_argument('--embeddings-dir', default=None)
    parser.add_argument('--out-dir', type=Path, default=Path('RESULTS'))
    parser.add_argument('--tag', default='recap')
    args = parser.parse_args()

    runs = [predictions(p, args.splits_dir, args.embeddings_dir) for p in args.runs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'wrote {figure(runs, args.out_dir/f"scatter_{args.tag}.png")}')
    print(f'wrote {table(runs, args.out_dir/f"zero_{args.tag}.md")}')


if __name__ == '__main__':
    main()
