#!/usr/bin/env python3
"""Where the MLP test error comes from. Dependencies: numpy, pandas, torch, matplotlib.

Example:
    python analyze_errors.py --splits-dir DATA --results-dir RESULTS

Loads the weights written by train_mlp.py, re-scores each test set, and splits
the squared error by true half-life so the contribution of each stratum is
explicit. Also reads the regression as a stability call at a half-life
threshold, which turns signed error into false positives (predicted stable,
measured unstable) and false negatives (the reverse).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_mlp import COLORS, MLP, features_and_target, style_axes
import arm_features

# Upper edges in hours; the zero spike is kept apart from the measurable rows.
BINS = [('= 0', 0.0, 0.0), ('(0, 0.5]', 0.0, 0.5), ('(0.5, 2]', 0.5, 2.0),
        ('(2, 8]', 2.0, 8.0), ('> 8', 8.0, np.inf)]
THRESHOLDS = (0.5, 1.0, 2.0)


def load_model(results_dir, split_index):
    """Return the stored model and its checkpoint, whose epsilon fixes the target space."""
    checkpoint = torch.load(Path(results_dir)/f'mlp_split_{split_index}.pt',
                            map_location='cpu', weights_only=True)
    model = MLP(checkpoint['features'], hidden=tuple(checkpoint['hidden']),
                dropout=checkpoint['dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def predict(model, checkpoint, features):
    """Re-score a test set with the stored weights; return predictions in log units."""
    with torch.no_grad():
        standardized = model(torch.from_numpy(features)).numpy()
    return standardized*checkpoint['target_scale']+checkpoint['target_center']


def stratify(half_life, error):
    """Per-stratum share of total squared error, signed bias and RMSE."""
    total = float(np.sum(error**2))
    rows = []
    for label, low, high in BINS:
        mask = (half_life == 0) if high == 0 else (half_life > low) & (half_life <= high)
        if not mask.any():
            continue
        subset = error[mask]
        rows.append(dict(stratum=label, rows=int(mask.sum()),
                         share_of_rows=float(mask.mean()),
                         share_of_squared_error=float(np.sum(subset**2)/total),
                         rmse=float(np.sqrt(np.mean(subset**2))),
                         bias=float(np.mean(subset))))
    return rows


def confusion(half_life, predicted_half_life, threshold):
    """Stability call at a half-life threshold: counts and rates of each error."""
    truth = half_life > threshold
    call = predicted_half_life > threshold
    tp = int(np.sum(call & truth))
    fp = int(np.sum(call & ~truth))
    fn = int(np.sum(~call & truth))
    tn = int(np.sum(~call & ~truth))
    positives, negatives = tp+fn, tn+fp
    return dict(threshold=threshold, true_positive=tp, false_positive=fp,
                false_negative=fn, true_negative=tn,
                positive_rows=positives, negative_rows=negatives,
                false_positive_rate=fp/negatives if negatives else float('nan'),
                false_negative_rate=fn/positives if positives else float('nan'),
                precision=tp/(tp+fp) if tp+fp else float('nan'),
                recall=tp/positives if positives else float('nan'),
                accuracy=(tp+tn)/len(truth))


def analyze(splits_dir='DATA', results_dir='RESULTS', splits=5, embeddings_dir=None):
    per_split = []
    for index in range(splits):
        with np.load(Path(splits_dir)/f'testing_{index}.npz') as data:
            test = {key: data[key] for key in data.files}
        # Score every model in its own target space, set by the epsilon it was
        # trained with; log-unit errors are therefore not comparable across
        # epsilons, while the stability calls below are, being made in hours.
        model, checkpoint = load_model(results_dir, index)
        epsilon = checkpoint['epsilon']
        # Rebuild the features the model was trained on. Without this a PLM arm
        # would be scored on one-hot inputs, and under pca:20 that is 860 wide
        # just like arm A0, so it would not even raise.
        encoder, standardiser = arm_features.from_checkpoint(checkpoint, embeddings_dir)
        x_test, y_test, _ = features_and_target(test, epsilon, encoder=encoder)
        if standardiser is not None:
            x_test = standardiser(x_test)
        if x_test.shape[1] != checkpoint['features']:
            raise ValueError(
                f"split {index}: rebuilt {x_test.shape[1]} features but the checkpoint "
                f"was trained on {checkpoint['features']}")
        predicted = predict(model, checkpoint, x_test)
        error = predicted-y_test
        half_life = test['thalf_hours'].astype(np.float64)
        predicted_half_life = np.maximum(np.exp(predicted.astype(np.float64))-epsilon, 0.0)
        per_split.append(dict(split_index=index, epsilon=float(epsilon),
                              truth=y_test, predicted=predicted,
                              rmse=float(np.sqrt(np.mean(error**2))),
                              bias=float(np.mean(error)),
                              over_predicted=float(np.mean(error > 0)),
                              strata=stratify(half_life, error),
                              confusion=[confusion(half_life, predicted_half_life, t)
                                         for t in THRESHOLDS],
                              half_life=half_life, error=error))
    return per_split


def aggregate(per_split, key, field, label_key):
    """Mean and SD over splits of one field, keyed by stratum or threshold label."""
    labels = [row[label_key] for row in per_split[0][key]]
    values = np.array([[row[field] for row in split[key]] for split in per_split])
    return labels, values.mean(axis=0), values.std(axis=0, ddof=1), values


def figure(per_split, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(14, 4.6), layout='constrained', facecolor=COLORS['surface'])
    FigureCanvasAgg(fig)
    left, middle, right = fig.subplots(1, 3)

    labels, share, share_sd, _ = aggregate(per_split, 'strata', 'share_of_squared_error', 'stratum')
    _, rows_share, _, _ = aggregate(per_split, 'strata', 'share_of_rows', 'stratum')
    positions = np.arange(len(labels))
    left.bar(positions-0.19, rows_share*100, width=0.36, color=COLORS['baseline'],
             label='Share of test rows', zorder=2)
    left.bar(positions+0.19, share*100, width=0.36, color=COLORS['train'],
             label='Share of squared error', zorder=2)
    left.errorbar(positions+0.19, share*100, yerr=share_sd*100, fmt='none',
                  ecolor=COLORS['ink'], elinewidth=1.2, capsize=4, zorder=3)
    left.set_xticks(positions, labels)
    left.set_ylabel('Percent', color=COLORS['secondary'], fontsize=9)
    left.set_xlabel('Measured half-life, hours', color=COLORS['secondary'], fontsize=9)
    left.set_title('Where the squared error sits', color=COLORS['ink'], fontsize=11)
    left.legend(frameon=False, fontsize=8, labelcolor=COLORS['secondary'])

    _, bias, bias_sd, bias_values = aggregate(per_split, 'strata', 'bias', 'stratum')
    middle.axhline(0, color=COLORS['axis'], linewidth=1.0, zorder=1)
    middle.bar(positions, bias, width=0.5, color=COLORS['validation'], zorder=2)
    middle.errorbar(positions, bias, yerr=bias_sd, fmt='none', ecolor=COLORS['ink'],
                    elinewidth=1.2, capsize=4, zorder=3)
    for column in range(bias_values.shape[1]):
        middle.scatter(np.full(len(bias_values), positions[column]), bias_values[:, column],
                       s=14, color=COLORS['surface'], edgecolor=COLORS['ink'],
                       linewidth=0.8, zorder=4)
    middle.set_xticks(positions, labels)
    middle.set_ylabel('Mean signed error, log units', color=COLORS['secondary'], fontsize=9)
    middle.set_xlabel('Measured half-life, hours', color=COLORS['secondary'], fontsize=9)
    middle.set_title('Over- and under-prediction by stratum', color=COLORS['ink'], fontsize=11)

    thresholds, fpr, fpr_sd, _ = aggregate(per_split, 'confusion', 'false_positive_rate', 'threshold')
    _, fnr, fnr_sd, _ = aggregate(per_split, 'confusion', 'false_negative_rate', 'threshold')
    positions = np.arange(len(thresholds))
    right.bar(positions-0.19, fpr*100, width=0.36, color=COLORS['train'],
              label='False positive rate', zorder=2)
    right.bar(positions+0.19, fnr*100, width=0.36, color=COLORS['validation'],
              label='False negative rate', zorder=2)
    right.errorbar(positions-0.19, fpr*100, yerr=fpr_sd*100, fmt='none',
                   ecolor=COLORS['ink'], elinewidth=1.2, capsize=4, zorder=3)
    right.errorbar(positions+0.19, fnr*100, yerr=fnr_sd*100, fmt='none',
                   ecolor=COLORS['ink'], elinewidth=1.2, capsize=4, zorder=3)
    right.set_xticks(positions, [f'> {t:g} h' for t in thresholds])
    right.set_ylabel('Percent of class', color=COLORS['secondary'], fontsize=9)
    right.set_xlabel('Stability threshold', color=COLORS['secondary'], fontsize=9)
    right.set_title('Stability call: which error dominates', color=COLORS['ink'], fontsize=11)
    right.legend(frameon=False, fontsize=8, labelcolor=COLORS['secondary'])

    for ax in (left, middle, right):
        style_axes(ax)
        ax.tick_params(axis='x', colors=COLORS['secondary'], labelsize=9)
    fig.suptitle('Test error decomposition over the five splits (bars: mean, whiskers: SD)',
                 color=COLORS['ink'], fontsize=12)
    fig.savefig(path, dpi=180, facecolor=COLORS['surface'])


def scatter_figure(per_split, path, epsilon):
    """Predicted against measured log half-life, pooled over splits, plus residuals.

    42,250 points overplot badly, so density is binned into hexagons on a
    single-hue sequential ramp; the median prediction per decile of the truth
    makes the shrinkage visible on top of it.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    truth = np.concatenate([split['truth'] for split in per_split])
    predicted = np.concatenate([split['predicted'] for split in per_split])
    residual = predicted-truth
    fig = Figure(figsize=(11.5, 5.2), layout='constrained', facecolor=COLORS['surface'])
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2)

    limits = (min(truth.min(), predicted.min())-0.2, max(truth.max(), predicted.max())+0.2)
    hexes = left.hexbin(truth, predicted, gridsize=55, bins='log', cmap='Blues',
                        mincnt=1, linewidths=0.2, extent=(*limits, *limits))
    left.plot(limits, limits, color=COLORS['ink'], linewidth=1.2,
              linestyle=(0, (4, 3)), label='Perfect prediction', zorder=3)
    edges = np.quantile(truth, np.linspace(0, 1, 11))
    centers, medians = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (truth >= low) & (truth <= high)
        if mask.sum() > 20:
            centers.append(float(np.median(truth[mask])))
            medians.append(float(np.median(predicted[mask])))
    left.plot(centers, medians, color=COLORS['validation'], linewidth=2.0,
              marker='o', markersize=5, label='Median prediction per decile', zorder=4)
    left.set(xlim=limits, ylim=limits)
    left.set_aspect('equal')
    left.set_xlabel(f'Measured ln(half-life + {epsilon:g})', color=COLORS['secondary'], fontsize=9)
    left.set_ylabel(f'Predicted ln(half-life + {epsilon:g})', color=COLORS['secondary'], fontsize=9)
    left.set_title('Predicted against measured', color=COLORS['ink'], fontsize=11)
    left.legend(frameon=False, fontsize=8, labelcolor=COLORS['secondary'], loc='upper left')
    bar = fig.colorbar(hexes, ax=left, shrink=0.82, pad=0.02)
    bar.set_label('Test rows per bin, log scale', color=COLORS['secondary'], fontsize=8)
    bar.ax.tick_params(colors=COLORS['muted'], labelsize=7)
    bar.outline.set_edgecolor(COLORS['hair'] if 'hair' in COLORS else COLORS['grid'])

    right.hexbin(truth, residual, gridsize=55, bins='log', cmap='Blues', mincnt=1,
                 linewidths=0.2)
    right.axhline(0, color=COLORS['ink'], linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    right.plot(centers, np.array(medians)-np.array(centers), color=COLORS['validation'],
               linewidth=2.0, marker='o', markersize=5, zorder=4)
    right.set_xlabel(f'Measured ln(half-life + {epsilon:g})', color=COLORS['secondary'], fontsize=9)
    right.set_ylabel('Residual, predicted - measured', color=COLORS['secondary'], fontsize=9)
    right.set_title('Residual against measured', color=COLORS['ink'], fontsize=11)

    for ax in (left, right):
        style_axes(ax)
        ax.tick_params(axis='x', colors=COLORS['secondary'], labelsize=9)
    fig.suptitle(f'Test predictions pooled over the five splits (n = {len(truth):,})',
                 color=COLORS['ink'], fontsize=12)
    fig.savefig(path, dpi=180, facecolor=COLORS['surface'])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--splits-dir', default='DATA')
    parser.add_argument('--results-dir', default='RESULTS')
    parser.add_argument('--splits', type=int, default=5)
    parser.add_argument('--embeddings-dir', default=None,
                        help='Override the embedding cache recorded in the '
                             'checkpoint; needed only if it has moved')
    args = parser.parse_args()
    per_split = analyze(args.splits_dir, args.results_dir, args.splits,
                        args.embeddings_dir)
    out = Path(args.results_dir)
    figure(per_split, out/'error_decomposition.png')
    scatter_figure(per_split, out/'predicted_vs_true.png', per_split[0]['epsilon'])

    labels, share, share_sd, _ = aggregate(per_split, 'strata', 'share_of_squared_error', 'stratum')
    _, rows_share, _, _ = aggregate(per_split, 'strata', 'share_of_rows', 'stratum')
    _, rmse, rmse_sd, _ = aggregate(per_split, 'strata', 'rmse', 'stratum')
    _, bias, bias_sd, _ = aggregate(per_split, 'strata', 'bias', 'stratum')
    table = pd.DataFrame(dict(stratum=labels, rows_pct=rows_share*100,
                              error_pct=share*100, error_pct_sd=share_sd*100,
                              rmse=rmse, rmse_sd=rmse_sd, bias=bias, bias_sd=bias_sd))
    print('\nSquared-error decomposition by measured half-life')
    print(table.to_string(index=False, float_format=lambda v: f'{v:7.3f}'))

    rows = []
    for field in ('false_positive', 'false_negative', 'false_positive_rate',
                  'false_negative_rate', 'precision', 'recall', 'positive_rows'):
        thresholds, mean, sd, _ = aggregate(per_split, 'confusion', field, 'threshold')
        rows.append(dict(metric=field, **{f'> {t:g} h': m for t, m in zip(thresholds, mean)}))
    print('\nStability call, mean over splits')
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:9.3f}'))
    print('\nFraction of test rows over-predicted:',
          f"{np.mean([s['over_predicted'] for s in per_split]):.3f}")

    report = dict(strata=table.to_dict(orient='records'),
                  thresholds=[{k: v for k, v in c.items()}
                              for c in per_split[0]['confusion']],
                  per_split=[{k: v for k, v in s.items()
                              if k not in ('half_life', 'error', 'truth', 'predicted')}
                             for s in per_split])
    (out/'error_decomposition.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
