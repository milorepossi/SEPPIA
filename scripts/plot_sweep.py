#!/usr/bin/env python3
"""Width sweep: test error and overfitting against the parameter-to-data ratio.

Example:
    python scripts/plot_sweep.py RESULTS/sweep --out-dir RESULTS --tag width

Reads every RESULTS/sweep/w<width>_s<seed>/metrics.json, groups by width, and
plots against parameters per training row rather than against width, so the
underparametrised and overparametrised regimes are directly readable and the
ratio-1 line means something.

Train error is recomputed from each checkpoint in eval mode. The train_loss in
metrics.json is accumulated with dropout active and so is inflated against
anything measured in eval mode; the gap between train and test only means
something when both are measured the same way.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_mlp import COLORS, style_axes
from plot_fit import score_run

PATTERN = re.compile(r'^w(\d+)_s(\d+)$')


def collect(sweep_dir, splits_dir, fit_rows):
    runs = {}
    for path in sorted(Path(sweep_dir).iterdir()):
        match = PATTERN.match(path.name)
        if not match or not (path/'metrics.json').exists():
            continue
        width, seed = int(match.group(1)), int(match.group(2))
        report = json.loads((path/'metrics.json').read_text())
        entry = report['splits'][0]
        scored = score_run(path, splits_dir)['rows'][0]
        runs.setdefault(width, []).append(dict(
            seed=seed,
            params=sum(p.numel() for p in __import__('torch').load(
                path/'mlp_split_0.pt', map_location='cpu',
                weights_only=True)['state_dict'].values()),
            spearman=entry['test']['spearman'], test=scored['test'],
            train=scored['train'], validation=scored['validation'],
            epochs=entry['best_epoch'], hidden=tuple(report['model']['hidden'])))
    rows = []
    for width in sorted(runs):
        group = runs[width]
        params = group[0]['params']
        rows.append(dict(width=width, hidden=group[0]['hidden'], params=params,
                         ratio=params/fit_rows, n=len(group),
                         **{key: np.array([np.nan if g[key] is None else g[key]
                                           for g in group], dtype=float)
                            for key in ('spearman', 'test', 'train', 'validation', 'epochs')}))
    return rows


def figure(rows, path, baseline_width=256):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(15, 4.6), layout='constrained', facecolor=COLORS['surface'])
    FigureCanvasAgg(fig)
    left, middle, right = fig.subplots(1, 3)
    ratio = np.array([r['ratio'] for r in rows])

    def band(ax, key, color, label):
        mean = np.array([np.nanmean(r[key]) for r in rows])
        lo = np.array([np.nanmin(r[key]) for r in rows])
        hi = np.array([np.nanmax(r[key]) for r in rows])
        ax.fill_between(ratio, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(ratio, mean, color=color, linewidth=2.0, marker='o', markersize=5,
                label=label, zorder=3)
        return mean

    for ax in (left, middle, right):
        ax.set_xscale('log')
        ax.axvline(1.0, color=COLORS['ink'], linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlabel('Parameters per training row', color=COLORS['secondary'], fontsize=9)

    spearman = band(left, 'spearman', COLORS['train'], 'Test Spearman')
    best = int(np.argmax(spearman))
    left.scatter([ratio[best]], [spearman[best]], s=90, facecolor='none',
                 edgecolor=COLORS['validation'], linewidth=2.0, zorder=5)
    left.annotate(f"best: {rows[best]['hidden']}\nratio {ratio[best]:.1f}",
                  (ratio[best], spearman[best]), textcoords='offset points',
                  xytext=(6, -26), fontsize=8, color=COLORS['validation'])
    left.set_ylabel('Test Spearman', color=COLORS['secondary'], fontsize=9)
    left.set_title('Test rank correlation', color=COLORS['ink'], fontsize=11)

    band(middle, 'train', COLORS['train'], 'Train (eval mode)')
    band(middle, 'validation', COLORS['gradient'], 'Validation')
    band(middle, 'test', COLORS['validation'], 'Test')
    middle.set_ylabel('RMSE, log units', color=COLORS['secondary'], fontsize=9)
    middle.set_title('Fit on each partition', color=COLORS['ink'], fontsize=11)
    middle.legend(frameon=False, fontsize=8, labelcolor=COLORS['secondary'])

    gap = np.array([np.nanmean(r['test']-r['train']) for r in rows])
    glo = np.array([np.nanmin(r['test']-r['train']) for r in rows])
    ghi = np.array([np.nanmax(r['test']-r['train']) for r in rows])
    right.fill_between(ratio, glo, ghi, color=COLORS['baseline'], alpha=0.18, linewidth=0)
    right.plot(ratio, gap, color=COLORS['baseline'], linewidth=2.0, marker='o',
               markersize=5, zorder=3)
    right.set_ylabel('Test RMSE minus train RMSE', color=COLORS['secondary'], fontsize=9)
    right.set_title('Generalisation gap', color=COLORS['ink'], fontsize=11)

    for ax in (left, middle, right):
        for i, r in enumerate(rows):
            if r['width'] == baseline_width:
                ax.axvline(ratio[i], color=COLORS['muted'], linewidth=1.0,
                           linestyle=(0, (1, 2)), zorder=1)
        style_axes(ax)
    left.annotate('params = rows', (1.0, left.get_ylim()[0]), fontsize=7,
                  color=COLORS['secondary'], rotation=90, ha='right', va='bottom')
    fig.suptitle('One-hot baseline, split 0, 3 seeds per width '
                 '(line: mean, band: seed range; dashed: ratio 1, dotted: current baseline)',
                 color=COLORS['ink'], fontsize=12)
    fig.savefig(path, dpi=180, facecolor=COLORS['surface'])
    return path


def table(rows, path, baseline_width=256):
    lines = ['# Width sweep on the one-hot baseline', '',
             'Split 0 only, 3 seeds per width. Train error is recomputed in eval mode '
             'from each checkpoint, so train, validation and test are comparable.', '',
             '| Hidden | Params | Params/row | Train | Validation | Test RMSE | Test Spearman | Gap | Epochs |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    best = max(range(len(rows)), key=lambda i: np.nanmean(rows[i]['spearman']))
    for i, r in enumerate(rows):
        gap = np.nanmean(r['test']-r['train'])
        mark = ' **←best**' if i == best else (' *(baseline)*' if r['width'] == baseline_width else '')
        lines.append(
            f"| {r['hidden']}{mark} | {r['params']:,} | {r['ratio']:.2f} | "
            f"{np.nanmean(r['train']):.3f} | {np.nanmean(r['validation']):.3f} | "
            f"{np.nanmean(r['test']):.3f} ± {np.nanstd(r['test'],ddof=1):.3f} | "
            f"**{np.nanmean(r['spearman']):.3f} ± {np.nanstd(r['spearman'],ddof=1):.3f}** | "
            f"{gap:+.3f} | {np.nanmean(r['epochs']):.0f} |")
    lines += ['', 'Generated by `scripts/plot_sweep.py`.']
    Path(path).write_text('\n'.join(lines)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sweep_dir', nargs='?', default='RESULTS/sweep')
    parser.add_argument('--splits-dir', default='DATA')
    parser.add_argument('--fit-rows', type=int, default=17744,
                        help='Training rows after the validation carve-out')
    parser.add_argument('--out-dir', type=Path, default=Path('RESULTS'))
    parser.add_argument('--tag', default='width')
    args = parser.parse_args()

    rows = collect(args.sweep_dir, args.splits_dir, args.fit_rows)
    if not rows:
        sys.exit(f'no completed runs found in {args.sweep_dir}')
    print(f'  {len(rows)} widths, {sum(r["n"] for r in rows)} runs')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'wrote {figure(rows, args.out_dir/f"sweep_{args.tag}.png")}')
    print(f'wrote {table(rows, args.out_dir/f"sweep_{args.tag}.md")}')


if __name__ == '__main__':
    main()
