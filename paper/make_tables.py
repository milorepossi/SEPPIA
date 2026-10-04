#!/usr/bin/env python3
"""Generate paper/tables.tex from the committed per-run metrics.json files.

Every number in the document comes from here, so no figure in the PDF can drift
from RESULTS/boltz/. Nothing is hand-entered; the generator also asserts the
claims the prose makes, and fails loudly rather than emitting a wrong table.

    python paper/make_tables.py            # writes paper/tables.tex
    python paper/make_tables.py --check    # verify only, write nothing
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'RESULTS' / 'boltz'
BASELINE = 'onehot'

# (arm, pooling) -> label. Order is the order of the results table.
ARMS = [
    ('onehot', 'pca:20', 'A0 one-hot'),
    ('BZS', 'pca:20', r'\armbzs{} single $s$'),
    ('BZZ', 'pca:20', r'\armbzz{} pair, $1/d^2$'),
    ('BZSZ', 'pca:20', r'\armbzsz{} single $+$ pair'),
    ('BZZU', 'pca:20', r'\armbzzu{} pair, uniform'),
]
FLATTEN = [
    ('onehot', 'flatten', 'A0 one-hot'),
    ('BZS', 'flatten', r'\armbzs{} single $s$'),
    ('BZZ', 'flatten', r'\armbzz{} pair, $1/d^2$'),
    ('BZSZ', 'flatten', r'\armbzsz{} single $+$ pair'),
    ('BZZU', 'flatten', r'\armbzzu{} pair, uniform'),
]


def slug(arm: str, pooling: str) -> str:
    return f"{arm}_{'pca20' if pooling == 'pca:20' else pooling}"


def load(arm: str, pooling: str) -> dict:
    path = RESULTS / slug(arm, pooling) / 'metrics.json'
    with path.open() as handle:
        return json.load(handle)


def recompute(run: dict, metric: str) -> tuple[float, float]:
    """Re-derive mean and sd from the per-split values, never trusting the
    stored summary. A mismatch means the committed artifact is inconsistent."""
    values = run['summary'][metric]['values']
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    stored = run['summary'][metric]
    for name, ours, theirs in (('mean', mean, stored['mean']),
                               ('sd', sd, stored['sd'])):
        if abs(ours - theirs) > 1e-9:
            raise SystemExit(
                f"{run['arm']['name']}/{run['arm']['pooling']} {metric} {name}: "
                f"recomputed {ours!r} != stored {theirs!r}")
    return mean, sd


def paired(run: dict, base: dict, metric: str = 'spearman') -> tuple[float, int, int]:
    """Mean delta and splits won, paired split by split against the baseline."""
    ours = run['summary'][metric]['values']
    theirs = base['summary'][metric]['values']
    if len(ours) != len(theirs):
        raise SystemExit('split count differs between arm and baseline')
    deltas = [a - b for a, b in zip(ours, theirs)]
    return statistics.fmean(deltas), sum(d > 0 for d in deltas), len(deltas)


def assert_comparable(runs: list[dict]) -> dict:
    """The paired comparison is only meaningful if every arm saw the same rows,
    the same splits, the same seeds and the same target correction."""
    fields = ('train_rows', 'validation_rows', 'test_rows', 'target_fixes', 'seed')
    # Keyed by split index: arms must agree with each other at each split, but
    # splits legitimately differ from one another (and seed == split index).
    reference: dict[int, tuple] = {}
    for run in runs:
        for split in run['splits']:
            index = split['split_index']
            got = tuple(split[f] for f in fields)
            if index not in reference:
                reference[index] = got
            elif got != reference[index]:
                raise SystemExit(
                    f"{run['arm']['name']}/{run['arm']['pooling']} split "
                    f"{index} differs: {got} != {reference[index]}")
    rows = runs[0]['splits'][0]
    return {
        'train_rows': rows['train_rows'],
        'validation_rows': rows['validation_rows'],
        'test_rows': rows['test_rows'],
        'target_fixes': rows['target_fixes'],
        'seeds': sorted({s['seed'] for s in runs[0]['splits']}),
        'n_splits': len(runs[0]['splits']),
    }


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def results_table(rows: list[tuple[str, str, str]], base: dict, caption: str,
                  label: str, note: str) -> str:
    lines = [
        r'\begin{table}[t]', r'\centering', r'\small',
        r'\begin{tabular}{llrrrr}', r'\toprule',
        r'Representation & Feat. & Spearman $\rho$ & RMSE & '
        r'Unseen-HLA & $\Delta\rho$ (won) \\',
        r'\midrule',
    ]
    for arm, pooling, label_text in rows:
        run = load(arm, pooling)
        rho, sd = recompute(run, 'spearman')
        rmse, _ = recompute(run, 'rmse')
        unseen, _ = recompute(run, 'rmse_unseen_hla')
        feats = run['splits'][0]['features']
        if arm == BASELINE:
            delta = 'reference'
        else:
            mean_delta, won, total = paired(run, base)
            delta = rf'{signed(mean_delta)} ({won}/{total})'
        emph = r'\bfseries ' if arm == 'BZZU' else ''
        lines.append(
            rf'{emph}{label_text} & {feats} & ${fmt(rho)} \pm {fmt(sd)}$ & '
            rf'{fmt(rmse)} & {fmt(unseen)} & {delta} \\')
    lines += [
        r'\bottomrule', r'\end{tabular}',
        rf'\caption{{{caption}}}', rf'\label{{{label}}}',
        note, r'\end{table}', '',
    ]
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    base_pca = load(BASELINE, 'pca:20')
    base_flat = load(BASELINE, 'flatten')

    every = [load(a, p) for a, p, _ in ARMS + FLATTEN]
    every.append(load('BZP', 'flatten'))
    every.append(load('BZZU', 'mean'))
    shared = assert_comparable(every)

    # The one-hot arm ignores --pooling (train_mlp.py builds no encoder for it),
    # so its three runs must be identical. If they ever diverge, the "matched
    # budget" framing in the paper is wrong and we want to know immediately.
    if base_pca['summary']['spearman']['values'] != base_flat['summary']['spearman']['values']:
        raise SystemExit('one-hot pca:20 and flatten runs differ; '
                         'the baseline is no longer a single configuration')

    # Assertions mirroring the prose, so the claims cannot rot silently.
    bzzu_pca = load('BZZU', 'pca:20')
    delta_pca, won_pca, total = paired(bzzu_pca, base_pca)
    if not (won_pca == total and delta_pca > 0.05):
        raise SystemExit(f'BZZU pca:20 claim broken: {delta_pca} {won_pca}/{total}')
    bzs_pca = load('BZS', 'pca:20')
    if recompute(bzs_pca, 'spearman')[0] >= recompute(base_pca, 'spearman')[0]:
        raise SystemExit('BZS no longer underperforms one-hot; §4 prose is wrong')

    macros = [
        '% Generated by paper/make_tables.py -- do not edit by hand.',
        rf"\newcommand{{\ntest}}{{{shared['test_rows']}}}",
        rf"\newcommand{{\ntrain}}{{{shared['train_rows']}}}",
        rf"\newcommand{{\nval}}{{{shared['validation_rows']}}}",
        rf"\newcommand{{\ntotal}}{{{shared['train_rows'] + shared['validation_rows'] + shared['test_rows']}}}",
        rf"\newcommand{{\nsplits}}{{{shared['n_splits']}}}",
        rf"\newcommand{{\targetfixes}}{{{shared['target_fixes']}}}",
        rf"\newcommand{{\bzzupca}}{{{fmt(recompute(bzzu_pca, 'spearman')[0])}}}",
        rf"\newcommand{{\bzzupcasd}}{{{fmt(recompute(bzzu_pca, 'spearman')[1])}}}",
        rf"\newcommand{{\bzzupcadelta}}{{{signed(delta_pca)}}}",
        rf"\newcommand{{\onehotrho}}{{{fmt(recompute(base_pca, 'spearman')[0])}}}",
        rf"\newcommand{{\onehotsd}}{{{fmt(recompute(base_pca, 'spearman')[1])}}}",
        rf"\newcommand{{\onehotunseen}}{{{fmt(recompute(base_pca, 'rmse_unseen_hla')[0])}}}",
        rf"\newcommand{{\bzzuunseen}}{{{fmt(recompute(bzzu_pca, 'rmse_unseen_hla')[0])}}}",
        rf"\newcommand{{\bzsrho}}{{{fmt(recompute(bzs_pca, 'spearman')[0])}}}",
        rf"\newcommand{{\bzzrho}}{{{fmt(recompute(load('BZZ', 'pca:20'), 'spearman')[0])}}}",
        rf"\newcommand{{\bzpflat}}{{{fmt(recompute(load('BZP', 'flatten'), 'spearman')[0])}}}",
        rf"\newcommand{{\bzzuvariance}}{{{100 * bzzu_pca['splits'][0]['explained_variance']:.1f}}}",
        rf"\newcommand{{\bzsvariance}}{{{100 * bzs_pca['splits'][0]['explained_variance']:.1f}}}",
        '',
    ]

    flat_delta, flat_won, _ = paired(load('BZZU', 'flatten'), base_flat)
    macros.insert(-1, rf"\newcommand{{\bzzuflat}}"
                      rf"{{{fmt(recompute(load('BZZU', 'flatten'), 'spearman')[0])}}}")
    macros.insert(-1, rf"\newcommand{{\bzzuflatdelta}}{{{signed(flat_delta)}}}")

    numbers = '\n'.join(macros)
    body = [
        results_table(
            ARMS, base_pca,
            caption=(r'Budget-matched comparison (\texttt{pca:20}). Every arm is '
                     r'compressed to 860 features, so the regression head is '
                     r'byte-identical across rows and only the input '
                     r'representation differs. RMSE is in log units; '
                     r'Unseen-HLA is RMSE restricted to test rows whose allele '
                     r'does not appear in training. $\Delta\rho$ is paired '
                     r'split by split against A0.'),
            label='tab:pca20',
            note=(r'\vspace{2pt}\footnotesize Mean $\pm$ SD over '
                  r'\nsplits{} splits, one seed per split; the SD is split '
                  r'spread, not seed spread (\S\ref{sec:limits}).')),
        results_table(
            FLATTEN, base_flat,
            caption=(r'Uncompressed comparison (\texttt{flatten}). This row set '
                     r'is \emph{not} budget-matched: the Boltz arms carry more '
                     r'features, and hence more first-layer parameters, than '
                     r'A0. It bounds what the compression in '
                     r'Table~\ref{tab:pca20} costs, and is not evidence about '
                     r'representation quality on its own.'),
            label='tab:flatten',
            note=(r'\vspace{2pt}\footnotesize Mean $\pm$ SD over '
                  r'\nsplits{} splits, one seed per split.')),
    ]
    out = '\n'.join(body)

    written = {HERE / 'numbers.tex': numbers, HERE / 'tables.tex': out}
    if args.check:
        for target, want in written.items():
            if not target.exists() or target.read_text() != want:
                raise SystemExit(f'{target.name} is stale; rerun paper/make_tables.py')
        print('numbers.tex and tables.tex up to date; all assertions pass')
        return
    for target, want in written.items():
        target.write_text(want)
        print(f'wrote {target}')
    print(f"rows {shared['train_rows']}/{shared['validation_rows']}/"
          f"{shared['test_rows']}, target_fixes={shared['target_fixes']}, "
          f"seeds={shared['seeds']}")


if __name__ == '__main__':
    main()
