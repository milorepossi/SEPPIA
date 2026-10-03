#!/usr/bin/env python3
"""Train against test error per arm: is a representation overfitting?

Example:
    python scripts/plot_fit.py RESULTS/A0_onehot RESULTS/A1_L0_pca20 \
        RESULTS/A2_L15_pca20 RESULTS/A3_L33_pca20 --out-dir RESULTS --tag fit

Re-scores each saved checkpoint on its own training rows, its validation rows
and the test set, and reports RMSE in log units with the generalisation gap.

The train_loss already in metrics.json is NOT usable for this. It is
accumulated during the epoch with dropout active, so it is inflated against a
validation loss measured in eval mode -- validation can even sit below it,
which looks like a bug and is not. Here every split is scored with the model in
eval mode, so train, validation and test are measured the same way and the gap
between them means something.

Features are rebuilt exactly as trained, from the arm, pooling, PCA projection
and standardiser stored in the checkpoint, and the train/validation division is
reconstructed from the split's recorded seed.
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

PALETTE = ['#898781', '#2a78d6', '#1baf7a', '#eb6834', '#8b5cf6', '#d946a0', '#0891b2']


def rmse(model, features, target, center, scale, batch=4096):
    """Eval-mode RMSE in log units."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(features), batch):
            chunk = torch.from_numpy(np.ascontiguousarray(features[start:start+batch]))
            out.append(model(chunk).numpy())
    prediction = np.concatenate(out)*scale+center
    return float(np.sqrt(np.mean((prediction-target)**2)))


def score_run(run_dir, splits_dir, embeddings_dir=None):
    run_dir = Path(run_dir)
    report = json.loads((run_dir/'metrics.json').read_text())
    arm = report.get('arm') or {}
    name, pooling = arm.get('name', 'onehot'), arm.get('pooling', '-')
    fraction = report['model']['validation_fraction']
    rows = []
    for entry in report['splits']:
        index = entry['split_index']
        checkpoint = torch.load(run_dir/f'mlp_split_{index}.pt', map_location='cpu',
                                weights_only=True)
        with np.load(Path(splits_dir)/f'training_{index}.npz', allow_pickle=True) as d:
            train_raw = {k: d[k] for k in d.files}
        with np.load(Path(splits_dir)/f'testing_{index}.npz', allow_pickle=True) as d:
            test_raw = {k: d[k] for k in d.files}

        encoder, standardiser = arm_features.from_checkpoint(checkpoint, embeddings_dir)
        epsilon = checkpoint['epsilon']
        x_all, y_all, _ = features_and_target(train_raw, epsilon, encoder=encoder)
        x_test, y_test, _ = features_and_target(test_raw, epsilon, encoder=encoder)
        if standardiser is not None:
            x_all, x_test = standardiser(x_all), standardiser(x_test)

        # Same division the run used: the seed it recorded, same fraction.
        order = np.random.default_rng(entry['seed']).permutation(len(x_all))
        n_validation = int(round(len(order)*fraction))
        validation_idx, train_idx = order[:n_validation], order[n_validation:]

        model = MLP(checkpoint['features'], hidden=tuple(checkpoint['hidden']),
                    dropout=checkpoint['dropout'])
        model.load_state_dict(checkpoint['state_dict'])
        center, scale = checkpoint['target_center'], checkpoint['target_scale']
        rows.append(dict(
            split=index,
            train=rmse(model, x_all[train_idx], y_all[train_idx], center, scale),
            validation=rmse(model, x_all[validation_idx], y_all[validation_idx], center, scale),
            test=rmse(model, x_test, y_test, center, scale),
            reported_test=entry['test']['rmse'],
            epochs=entry['best_epoch']))
    label = name if name == 'onehot' else f'{name}\n{pooling}'
    return dict(label=label, name=name, pooling=pooling,
                features=report['encoding']['features'], rows=rows)


def figure(runs, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(13, 5), layout='constrained', facecolor=COLORS['surface'])
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2, width_ratios=[1.4, 1])
    labels = [r['label'] for r in runs]
    x = np.arange(len(runs))

    stages = [('train', 'Train', '#2a78d6'), ('validation', 'Validation', '#1baf7a'),
              ('test', 'Test', '#eb6834')]
    width = 0.8/len(stages)
    for i, (key, label, color) in enumerate(stages):
        values = np.array([[row[key] for row in run['rows']] for run in runs])
        means, sd = values.mean(axis=1), values.std(axis=1, ddof=1)
        positions = x+(i-(len(stages)-1)/2)*width
        left.bar(positions, means, width=width*0.9, color=color, label=label, zorder=2)
        left.errorbar(positions, means, yerr=sd, fmt='none', ecolor=COLORS['ink'],
                      elinewidth=1.0, capsize=3, zorder=3)
    left.set_xticks(x, labels)
    left.set_ylabel('RMSE, log units (eval mode)', color=COLORS['secondary'], fontsize=9)
    left.set_title('Fit on train, validation and test\n(all measured the same way)',
                   color=COLORS['ink'], fontsize=11)
    left.legend(frameon=False, fontsize=8, labelcolor=COLORS['secondary'])

    gaps = np.array([[row['test']-row['train'] for row in run['rows']] for run in runs])
    means, sd = gaps.mean(axis=1), gaps.std(axis=1, ddof=1)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(runs))]
    right.bar(x, means, width=0.6, color=colors, zorder=2)
    right.errorbar(x, means, yerr=sd, fmt='none', ecolor=COLORS['ink'],
                   elinewidth=1.2, capsize=4, zorder=3)
    for i in range(len(runs)):
        right.scatter(np.full(gaps.shape[1], x[i]), gaps[i], s=16, color=COLORS['surface'],
                      edgecolor=COLORS['ink'], linewidth=0.8, zorder=4)
    right.set_xticks(x, labels)
    right.set_ylabel('Test RMSE minus train RMSE', color=COLORS['secondary'], fontsize=9)
    right.set_title('Generalisation gap\n(higher: more overfitting)',
                    color=COLORS['ink'], fontsize=11)

    for ax in (left, right):
        style_axes(ax)
        ax.tick_params(axis='x', colors=COLORS['secondary'], labelsize=8)
    fig.suptitle('Where each representation spends its capacity',
                 color=COLORS['ink'], fontsize=12)
    fig.savefig(path, dpi=180, facecolor=COLORS['surface'])
    return path


def table(runs, path):
    lines = ['# Train against test error', '',
             'RMSE in log units, every stage scored in eval mode so the numbers are '
             'comparable. The `train_loss` in `metrics.json` is not: it is accumulated '
             'with dropout active.', '',
             '| Arm | Pooling | Features | Train | Validation | Test | Gap (test − train) | Epochs |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        values = {k: np.array([row[k] for row in run['rows']])
                  for k in ('train', 'validation', 'test', 'epochs')}
        gap = values['test']-values['train']
        lines.append(
            f'| {run["name"]} | {run["pooling"]} | {run["features"]} | '
            f'{values["train"].mean():.3f} | {values["validation"].mean():.3f} | '
            f'{values["test"].mean():.3f} ± {values["test"].std(ddof=1):.3f} | '
            f'**{gap.mean():+.3f}** | {values["epochs"].mean():.0f} |')
    lines += ['', 'A larger gap at equal test error means the arm is fitting the '
              'training set harder for the same generalisation.', '',
              'Generated by `scripts/plot_fit.py`.']
    Path(path).write_text('\n'.join(lines)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--splits-dir', default='DATA')
    parser.add_argument('--embeddings-dir', default=None)
    parser.add_argument('--out-dir', type=Path, default=Path('RESULTS'))
    parser.add_argument('--tag', default='fit')
    args = parser.parse_args()

    runs = []
    for path in args.runs:
        run = score_run(path, args.splits_dir, args.embeddings_dir)
        drift = max(abs(r['test']-r['reported_test']) for r in run['rows'])
        print(f"  {run['label'].replace(chr(10), ' '):16s} rescored, max drift from the "
              f"reported test RMSE: {drift:.2e}")
        runs.append(run)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'wrote {figure(runs, args.out_dir/f"fit_{args.tag}.png")}')
    print(f'wrote {table(runs, args.out_dir/f"fit_{args.tag}.md")}')


if __name__ == '__main__':
    main()
