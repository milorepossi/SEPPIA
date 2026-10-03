#!/usr/bin/env python3
"""Figures and tables for the arm ladder. Dependencies: numpy, matplotlib.

Example:
    python scripts/plot_ladder.py RESULTS/A0_onehot RESULTS/A1_L0_pca20 \
        RESULTS/A2_L15_pca20 RESULTS/A3_L33_pca20 --out-dir RESULTS --tag pca20

Writes, for the runs given in ladder order with the reference arm first:
    ladder_<tag>.png   three panels: headline metric, paired per-split deltas,
                       and error by generalisation subset
    ladder_<tag>.md    the same numbers as a table to read or paste

Every arm is scored on the same five splits, so the comparison is paired and
the middle panel is the one to read: an arm's standard deviation across splits
is mostly split difficulty, which is shared, and so swamps the difference
between arms. A consistent sign across all five splits says more than two
overlapping error bars.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_mlp import COLORS, style_axes

SUBSETS = [('rmse_all', 'All'), ('rmse_seen_both', 'Seen\npeptide & HLA'),
           ('rmse_unseen_peptide', 'Unseen\npeptide'),
           ('rmse_unseen_hla', 'Unseen\nHLA')]
PALETTE = ['#898781', '#2a78d6', '#1baf7a', '#eb6834', '#8b5cf6', '#d946a0', '#0891b2']


def load(path):
    path = Path(path)
    report = json.loads((path/'metrics.json').read_text())
    arm = report.get('arm') or {}
    name = arm.get('name', 'onehot')
    pooling = arm.get('pooling', '-')
    label = name if name == 'onehot' else f'{name}\n{pooling}'
    return dict(label=label, name=name, pooling=pooling,
                features=report['encoding']['features'],
                summary=report['summary'],
                explained=arm.get('explained_variance'),
                baseline=float(np.mean([s['baseline']['rmse'] for s in report['splits']])),
                epochs=[s['best_epoch'] for s in report['splits']],
                n_splits=len(report['splits']))


def figure(runs, path, metric='spearman'):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(15, 5), layout='constrained', facecolor=COLORS['surface'])
    FigureCanvasAgg(fig)
    left, middle, right = fig.subplots(1, 3, width_ratios=[1, 1.15, 1.3])
    labels = [r['label'] for r in runs]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(runs))]
    x = np.arange(len(runs))

    # Panel 1: the headline metric, with every split shown as a dot.
    values = np.array([r['summary'][metric]['values'] for r in runs])
    means = values.mean(axis=1)
    sd = values.std(axis=1, ddof=1) if values.shape[1] > 1 else None
    left.bar(x, means, width=0.6, color=colors, zorder=2)
    if sd is not None:
        left.errorbar(x, means, yerr=sd, fmt='none', ecolor=COLORS['ink'],
                      elinewidth=1.2, capsize=4, zorder=3)
    for i in range(len(runs)):
        left.scatter(np.full(values.shape[1], x[i]), values[i], s=16,
                     color=COLORS['surface'], edgecolor=COLORS['ink'],
                     linewidth=0.8, zorder=4)
    low = min(values.min(), means.min())
    left.set_ylim(max(0.0, low-0.06), min(1.0, values.max()+0.03))
    left.axhline(means[0], color=COLORS['muted'], linewidth=1.0,
                 linestyle=(0, (4, 3)), zorder=1)
    left.set_xticks(x, labels)
    left.set_ylabel(f'Test {metric}', color=COLORS['secondary'], fontsize=9)
    left.set_title(f'{metric.capitalize()}, mean of {values.shape[1]} splits',
                   color=COLORS['ink'], fontsize=11)

    # Panel 2: paired differences against the reference arm, split by split.
    reference = values[0]
    offsets = np.linspace(-0.26, 0.26, max(len(runs)-1, 1))
    for i in range(1, len(runs)):
        delta = values[i]-reference
        positions = np.arange(values.shape[1])+offsets[i-1]
        middle.bar(positions, delta, width=0.5/max(len(runs)-1, 1),
                   color=colors[i], zorder=2, label=labels[i].replace('\n', ' '))
    middle.axhline(0, color=COLORS['ink'], linewidth=1.2, zorder=3)
    middle.set_xticks(np.arange(values.shape[1]),
                      [f'split {i}' for i in range(values.shape[1])])
    middle.set_ylabel(f'{metric} minus {labels[0]}', color=COLORS['secondary'], fontsize=9)
    middle.set_title('Paired difference per split\n(below zero: worse than the baseline arm)',
                     color=COLORS['ink'], fontsize=11)
    middle.legend(frameon=False, fontsize=8, labelcolor=COLORS['secondary'], ncol=1)

    # Panel 3: where the error sits, by how unfamiliar the test row is.
    width = 0.8/len(runs)
    for i, run in enumerate(runs):
        heights = [run['summary'][key]['mean'] for key, _ in SUBSETS]
        errors = [run['summary'][key]['sd'] for key, _ in SUBSETS]
        positions = np.arange(len(SUBSETS))+(i-(len(runs)-1)/2)*width
        right.bar(positions, heights, width=width*0.92, color=colors[i], zorder=2,
                  label=labels[i].replace('\n', ' '))
        if all(e is not None for e in errors):
            right.errorbar(positions, heights, yerr=errors, fmt='none',
                           ecolor=COLORS['ink'], elinewidth=0.9, capsize=2, zorder=3)
    right.axhline(runs[0]['baseline'], color=COLORS['ink'], linewidth=1.0,
                  linestyle=(0, (2, 2)), zorder=1)
    right.annotate('train-mean baseline', (len(SUBSETS)-0.5, runs[0]['baseline']),
                   fontsize=7, color=COLORS['secondary'], ha='right', va='bottom')
    right.set_xticks(np.arange(len(SUBSETS)), [label for _, label in SUBSETS])
    right.set_ylabel('Test RMSE, log units', color=COLORS['secondary'], fontsize=9)
    right.set_title('Error by how unfamiliar the test row is',
                    color=COLORS['ink'], fontsize=11)
    right.legend(frameon=False, fontsize=8, labelcolor=COLORS['secondary'])

    for ax in (left, middle, right):
        style_axes(ax)
        ax.tick_params(axis='x', colors=COLORS['secondary'], labelsize=8)
    fig.suptitle('Arm ladder: one MLP, swapped input representation '
                 '(bars mean, whiskers SD, dots splits)',
                 color=COLORS['ink'], fontsize=12)
    fig.savefig(path, dpi=180, facecolor=COLORS['surface'])
    return path


def table(runs, path, metric='spearman'):
    """Markdown summary: per-arm metrics, then the paired view."""
    reference = np.array(runs[0]['summary'][metric]['values'])
    lines = ['# Arm ladder results', '',
             f'One MLP block (256-128-1), identical hyperparameters, '
             f'{runs[0]["n_splits"]} splits. Only the input representation differs.',
             '', f'Train-mean baseline RMSE: **{runs[0]["baseline"]:.3f}** '
             '(the same target for every arm).', '',
             '## Per arm', '',
             '| Arm | Pooling | Features | Spearman | Pearson | RMSE | MAE | PCA variance kept |',
             '|---|---|---|---|---:|---:|---:|---:|']
    for run in runs:
        cells = []
        for key in ('spearman', 'pearson', 'rmse', 'mae'):
            entry = run['summary'][key]
            cells.append(f'{entry["mean"]:.3f} ± {entry["sd"]:.3f}'
                         if entry['sd'] is not None else f'{entry["mean"]:.3f}')
        kept = '—' if run['explained'] is None else f'{run["explained"]:.1%}'
        lines.append(f'| {run["name"]} | {run["pooling"]} | {run["features"]} | '
                     + ' | '.join(cells) + f' | {kept} |')

    lines += ['', f'## Paired {metric} per split, against {runs[0]["name"]}', '',
              '| Arm | ' + ' | '.join(f'split {i}' for i in range(len(reference)))
              + ' | mean Δ | splits won |',
              '|---|' + '---:|'*len(reference) + '---:|---:|']
    for run in runs:
        values = np.array(run['summary'][metric]['values'])
        row = ' | '.join(f'{v:.3f}' for v in values)
        if run is runs[0]:
            lines.append(f'| **{run["name"]} ({run["pooling"]})** | {row} | — | reference |')
        else:
            delta = values-reference
            better = delta > 0 if metric in ('spearman', 'pearson') else delta < 0
            lines.append(f'| {run["name"]} ({run["pooling"]}) | {row} | '
                         f'{delta.mean():+.4f} | {better.sum()}/{len(delta)} |')

    lines += ['', '## Error by generalisation subset (RMSE, log units)', '',
              '| Arm | ' + ' | '.join(label.replace("\n", " ") for _, label in SUBSETS) + ' |',
              '|---|' + '---:|'*len(SUBSETS)]
    for run in runs:
        cells = [f'{run["summary"][key]["mean"]:.3f}' for key, _ in SUBSETS]
        lines.append(f'| {run["name"]} ({run["pooling"]}) | ' + ' | '.join(cells) + ' |')

    lines += ['', '## Reading this', '',
              '- The comparison is **paired**: every arm sees the same five splits. An '
              'arm\'s SD across splits is mostly split difficulty, which is shared, so '
              'it swamps the difference between arms. The sign being consistent across '
              'all five splits carries the argument, not the means.',
              '- **Single seed per split.** Pooling and standardisation alone move '
              'Spearman by about 0.025 on identical information, so a gap smaller than '
              'that is not evidence about a representation.',
              '- `pca:20` is the budget-matched comparison: every arm gets exactly 860 '
              'features and a byte-identical head. `flatten` gives a PLM arm 55,040 '
              'features and 64x the first-layer parameters, so it is a capacity '
              'experiment, not a representation one.',
              '- Layer 0 is rank 20 on this alphabet, so `pca:20` is **lossless** for '
              'it and arm L0 carries the same information as one-hot. L0 against '
              'one-hot is therefore a pipeline check whose expected result is a tie.',
              '', f'Generated by `scripts/plot_ladder.py` from each run\'s `metrics.json`.']
    Path(path).write_text('\n'.join(lines)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='Run directories, reference arm first')
    parser.add_argument('--out-dir', type=Path, default=Path('RESULTS'))
    parser.add_argument('--tag', default='ladder')
    parser.add_argument('--metric', default='spearman')
    args = parser.parse_args()

    runs = [load(path) for path in args.runs]
    counts = {run['n_splits'] for run in runs}
    if len(counts) > 1:
        print(f'WARNING: runs differ in split count {sorted(counts)}; the paired '
              'panels assume the same splits', file=sys.stderr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = figure(runs, args.out_dir/f'ladder_{args.tag}.png', args.metric)
    md = table(runs, args.out_dir/f'ladder_{args.tag}.md', args.metric)
    print(f'wrote {png}')
    print(f'wrote {md}')


if __name__ == '__main__':
    main()
