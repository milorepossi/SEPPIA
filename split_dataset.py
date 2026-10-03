#!/usr/bin/env python3
"""Constrained row split. Dependencies: numpy, pandas, openpyxl, matplotlib.

Example:
    python split_dataset.py DATA/rasmussen_et_al_dataset.xlsx --thresholds 0.05 0.1

Thresholds are absolute train/test differences of E[ln(x)] and E[ln(x)^2].
Zero becomes 0.001 for logarithms only. Excel dates recover as day.month,
as in the existing analysis notebook. Holdouts mean AT LEAST 5 distinct
HLA sequences and 100 peptide sequences are exclusive to test by default
(configurable with --unseen-hla and --unseen-peptides). Other
sequences may overlap. Search failure is not proof of infeasibility.
"""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


def half_life(value):
    if pd.isna(value):
        raise ValueError('Missing half-life')
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return float(f'{value.day}.{value.month}')
    return float(str(value).strip().replace(',', '.'))


def _split_dataset(dataset, thresholds, output_dir='DATA/split', *, seed=42,
                  seeds=100, seconds=30, train_fraction=0.7, fraction_tolerance=0.05,
                  hla_column='hla_seq', peptide_column='peptide',
                  target_column='thalf_hours', sheet=0, unseen_hla=5,
                  unseen_peptides=100, split_index=0):
    """Write numbered training/testing workbooks, a plot and metadata.

    unseen_hla and unseen_peptides specify minimum test-exclusive counts.
    Return metadata describing the split and saved outputs.

    Fixed holdout rows cannot move into training. Random row swaps improve
    moment matching, across independently sampled holdouts/seeds. All input
    rows are retained exactly once; source_row identifies their input position.
    """
    started = time.monotonic()
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (2,) or not np.isfinite(thresholds).all() or (thresholds < 0).any():
        raise ValueError('thresholds must contain two finite nonnegative numbers')
    if seeds < 1 or seconds <= 0 or not 0 < train_fraction < 1 or not 0 <= fraction_tolerance < min(train_fraction, 1-train_fraction):
        raise ValueError('Invalid search budget or split fraction')
    for name, count in (('unseen_hla', unseen_hla), ('unseen_peptides', unseen_peptides)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    # Use a headless backend so command-line runs can save plots without a display.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    path = Path(dataset)
    df = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_excel(path, sheet_name=sheet)
    for column in (hla_column, peptide_column, target_column):
        if column not in df:
            raise ValueError(f'Missing column: {column}')
    if 'source_row' in df:
        raise ValueError('Input already contains reserved column source_row')
    converted = int(df[target_column].map(lambda x: isinstance(x, (date, datetime, pd.Timestamp))).sum())
    df[target_column] = df[target_column].map(half_life).astype(float)
    y = df[target_column].to_numpy()
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Half-lives must be finite and nonnegative')
    for column in (hla_column, peptide_column):
        if df[column].isna().any() or not df[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f'{column} must contain nonempty strings')
    h, h_values = pd.factorize(df[hla_column])
    p, p_values = pd.factorize(df[peptide_column])
    if len(h_values) <= unseen_hla or len(p_values) <= unseen_peptides:
        raise ValueError(f'Need at least {unseen_hla + 1} HLA and {unseen_peptides + 1} peptides to retain training observations')
    n = len(df)
    low = max(1, int(np.ceil(n * (train_fraction-fraction_tolerance))))
    high = min(n-1, int(np.floor(n * (train_fraction+fraction_tolerance))))
    if low > high:
        raise ValueError('No integer split size within fraction tolerance')
    z = np.log(np.where(y == 0, 0.001, y))
    features = np.column_stack((z, z*z))
    total = features.sum(axis=0)
    scales = np.maximum(thresholds, 1e-12)
    best = None
    attempts = 0
    feasible = 0
    for offset in range(seeds):
        if offset and time.monotonic()-started >= seconds:
            break
        attempts += 1
        rng = np.random.default_rng(seed+offset)
        held_h = rng.choice(len(h_values), unseen_hla, replace=False)
        held_p = rng.choice(len(p_values), unseen_peptides, replace=False)
        forced = np.isin(h, held_h) | np.isin(p, held_p)
        eligible = np.flatnonzero(~forced)
        if len(eligible) < low:
            continue
        feasible += 1
        size = min(int(round(n*train_fraction)), len(eligible), high)
        size = max(size, low)
        rng.shuffle(eligible)
        train = eligible[:size].copy()
        movable_test = eligible[size:].copy()
        sums = features[train].sum(axis=0)
        def differences(s):
            return np.abs(s/size - (total-s)/(n-size))
        diff = differences(sums)
        score = float(np.max(diff/scales))
        # Vectorized batches of candidate swaps; each preserves row counts.
        for iteration in range(501):
            if best is None or score < best['score']:
                best = dict(score=score, train=train.copy(), diff=diff.copy(),
                            held_h=held_h.copy(), held_p=held_p.copy(), seed=seed+offset)
            if (diff <= thresholds).all() or not len(movable_test) or iteration == 500 or time.monotonic()-started >= seconds:
                break
            a = rng.integers(len(train), size=256)
            b = rng.integers(len(movable_test), size=256)
            candidates = sums + features[movable_test[b]] - features[train[a]]
            diffs = np.abs(candidates/size - (total-candidates)/(n-size))
            scores = np.max(diffs/scales, axis=1)
            j = int(scores.argmin())
            if scores[j] < score:
                ai, bi = a[j], b[j]
                train[ai], movable_test[bi] = movable_test[bi], train[ai]
                sums, diff, score = candidates[j], diffs[j], float(scores[j])
        if best is not None and (best['diff'] <= thresholds).all():
            break
    if best is None:
        raise RuntimeError(f'No holdout selection allowed the requested row fraction in {attempts} seeds. Try more seeds or a larger fraction tolerance. No files written.')
    mask = np.zeros(n, dtype=bool)
    mask[best['train']] = True
    train_idx, test_idx = np.flatnonzero(mask), np.flatnonzero(~mask)
    exclusive_h = sorted(set(df.iloc[test_idx][hla_column])-set(df.iloc[train_idx][hla_column]))
    exclusive_p = sorted(set(df.iloc[test_idx][peptide_column])-set(df.iloc[train_idx][peptide_column]))
    if len(exclusive_h) < unseen_hla or len(exclusive_p) < unseen_peptides:
        raise RuntimeError('Internal holdout validation failed')
    def stats(idx):
        return dict(rows=len(idx), fraction=len(idx)/n,
                    unique_hla=int(df.iloc[idx][hla_column].nunique()),
                    unique_peptides=int(df.iloc[idx][peptide_column].nunique()),
                    log_mean=float(features[idx, 0].mean()),
                    log_second_raw_moment=float(features[idx, 1].mean()),
                    log_variance=float(z[idx].var()))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    training_name = f'training_{split_index}.xlsx'
    testing_name = f'testing_{split_index}.xlsx'
    metadata_name = f'metadata_{split_index}.json'
    plot_name = f'log_distributions_{split_index}.png'
    df.insert(0, 'source_row', np.arange(2, n+2))
    metadata = dict(input=str(path.resolve()), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        created_at=datetime.now(timezone.utc).isoformat(), sheet=sheet,
        columns=dict(hla=hla_column, peptide=peptide_column, half_life=target_column),
        total_rows=n, train=stats(train_idx), test=stats(test_idx),
        thresholds=thresholds.tolist(), absolute_moment_differences=best['diff'].tolist(),
        constraints_met=bool((best['diff'] <= thresholds).all()),
        requested_train_fraction=train_fraction, fraction_tolerance=fraction_tolerance,
        requested_unseen_hla=int(unseen_hla), requested_unseen_peptides=int(unseen_peptides),
        selected_seed=best['seed'], initial_seed=seed, seeds_attempted=attempts,
        seeds_with_feasible_holdouts=feasible, max_seeds=seeds, time_budget_seconds=seconds,
        search_seconds=time.monotonic()-started,
        selected_holdout_hla=h_values[best['held_h']].tolist(),
        selected_holdout_peptides=p_values[best['held_p']].tolist(),
        all_test_exclusive_hla=exclusive_h, all_test_exclusive_peptides=exclusive_p,
        train_source_rows=df.iloc[train_idx]['source_row'].tolist(),
        test_source_rows=df.iloc[test_idx]['source_row'].tolist(),
        preprocessing=dict(log_base='natural', zero_epsilon=0.001, zero_count=int((y == 0).sum()),
                           epsilon_applied_to='log statistics only', date_conversion='float(day.month)',
                           dates_converted=converted),
        split_index=split_index,
        outputs={name: str((out/name).resolve()) for name in (training_name, testing_name, metadata_name, plot_name)},
        search_note='Heuristic search; an unmet result does not prove infeasibility. Holdout counts are minimums.')
    df.iloc[train_idx].to_excel(out/training_name, index=False)
    df.iloc[test_idx].to_excel(out/testing_name, index=False)
    fig = Figure(figsize=(9, 5), layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    bins = np.histogram_bin_edges(z, bins='auto')
    ax.hist(z[train_idx], bins=bins, density=True, alpha=0.55,
            label=f'Train (n={len(train_idx)})')
    ax.hist(z[test_idx], bins=bins, density=True, alpha=0.55,
            label=f'Test (n={len(test_idx)})')
    ax.set(xlabel='ln(half-life / hour)', ylabel='Density',
           title='Train and test log half-life distributions')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    fig.savefig(out/plot_name, dpi=180)
    (out/metadata_name).write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    return metadata


def split_dataset(dataset, thresholds, output_dir='DATA/split', *, seed=42,
                  seeds=100, **kwargs):
    """Create five splits numbered 0–4 and return their metadata.

    Each split searches a separate range of ``seeds`` random seeds, starting
    at ``seed + split_index * seeds``. Search budgets apply to each split.
    """
    return [_split_dataset(dataset, thresholds, output_dir,
                           seed=seed + index * seeds, seeds=seeds,
                           split_index=index, **kwargs)
            for index in range(5)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--thresholds', nargs=2, type=float, required=True, metavar=('MEAN', 'SECOND_MOMENT'))
    parser.add_argument('--output-dir', default='DATA/split')
    parser.add_argument('--seed', type=int, default=42, help='Starting seed for five separate seed ranges')
    parser.add_argument('--seeds', type=int, default=100, help='Maximum seed attempts per split')
    parser.add_argument('--seconds', type=float, default=30, help='Search time budget per split')
    parser.add_argument('--train-fraction', type=float, default=0.7)
    parser.add_argument('--fraction-tolerance', type=float, default=0.05)
    parser.add_argument('--unseen-hla', type=int, default=5,
                        help='Minimum number of HLA sequences exclusive to test (default: 5)')
    parser.add_argument('--unseen-peptides', type=int, default=100,
                        help='Minimum number of peptide sequences exclusive to test (default: 100)')
    parser.add_argument('--hla-column', default='hla_seq')
    parser.add_argument('--peptide-column', default='peptide')
    parser.add_argument('--target-column', default='thalf_hours')
    parser.add_argument('--sheet', default=0, help='Excel sheet name; first sheet by default')
    args = vars(parser.parse_args())
    metadata = split_dataset(**args)
    print(json.dumps([{k: result[k] for k in ('split_index', 'constraints_met', 'absolute_moment_differences', 'selected_seed', 'seeds_attempted', 'outputs')} for result in metadata], indent=2))
    if not all(result['constraints_met'] for result in metadata):
        print('WARNING: Best splits saved, but moment thresholds were NOT met for all splits.')
        raise SystemExit(2)


if __name__ == '__main__':
    main()
