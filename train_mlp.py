#!/usr/bin/env python3
"""MLP regression of peptide-HLA stability. Dependencies: numpy, pandas, torch, matplotlib.

Example:
    python train_mlp.py --splits-dir DATA --output-dir RESULTS
    python train_mlp.py --arm L33 --pooling pca:20 --output-dir RESULTS/L33

One MLP block serves every arm of the ladder; --arm changes only the input
representation, and every other hyperparameter is held fixed so the arms share
a budget:

    onehot  A0  one-hot of the 43 kept positions, no PLM    (the default)
    L0      A1  ESM-2 layer 0, context-free control
    L15     A2  ESM-2 layer 15, mid-stack
    L33     A3  ESM-2 layer 33, final

The PLM arms read scripts/extract_embeddings.py's cache and join to a split on
source_row; --pooling decides how each row's (43, 1280) block becomes a vector
and therefore whether the arms also share a first-layer width. See
scripts/arm_features.py for what pooling and standardisation do to the
comparison. Anything fitted to the features is fitted on the training rows
only.

Features for the default arm are the peptide (9 residues) stacked with the HLA
pseudosequence (34 residues), one-hot encoded over the 20 amino acids, so
43*20 = 860 inputs.
The target is ln(thalf_hours + epsilon); epsilon regularizes the 20% of rows
whose half-life is exactly zero. It defaults to 0.1, the reporting resolution
of the assay: smaller values push the zero rows far below the rest and let them
dominate the squared error. Targets are standardized with training statistics
for optimization and all reported errors are in log units, which depend on
epsilon and so are only comparable between runs that share it.

One model per split (0-4). A fraction of each training set is held out for
early stopping, so the testing sets stay untouched until final scoring.
Outputs per-epoch histories, test metrics, and two figures: convergence of
loss and gradient norm, and test error averaged over the five splits.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent/'scripts'))
import arm_features

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
# Reference data-viz palette: categorical slots, chart chrome and ink.
COLORS = dict(train='#2a78d6', validation='#eb6834', gradient='#1baf7a',
              baseline='#898781', surface='#fcfcfb', ink='#0b0b0b',
              secondary='#52514e', muted='#898781', grid='#e1e0d9',
              axis='#c3c2b7')


def one_hot(sequences, length):
    """Return (rows, length*20) float32 one-hot encoding of equal-length strings."""
    chars = np.array(sequences, dtype=f'U{length}').view('U1').reshape(len(sequences), -1)
    if chars.shape[1] != length:
        raise ValueError(f'Expected sequences of length {length}, got {chars.shape[1]}')
    lookup = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
    codes = np.vectorize(lookup.get, otypes=[object])(chars)
    if pd.isna(pd.Series(codes.ravel())).any():
        unknown = sorted(set(chars.ravel()) - set(AMINO_ACIDS))
        raise ValueError(f'Unknown residues: {unknown}')
    codes = codes.astype(np.int64)
    encoded = np.zeros((len(sequences), length, len(AMINO_ACIDS)), dtype=np.float32)
    rows = np.arange(len(sequences))[:, None]
    encoded[rows, np.arange(length)[None, :], codes] = 1.0
    return encoded.reshape(len(sequences), -1)


def features_and_target(dataset, epsilon, peptide_length=9, pseudoseq_length=34,
                        encoder=None):
    """Encode features; return features, log target, rows.

    With no encoder this is arm A0: the peptide stacked with the HLA
    pseudosequence, one-hot over the 20 amino acids. An encoder replaces the
    features with another representation of the same rows, which is the only
    thing that differs between arms of the ladder.
    """
    if encoder is None:
        features = np.hstack((one_hot(dataset['peptide'], peptide_length),
                              one_hot(dataset['hla_pseudoseq'], pseudoseq_length)))
    else:
        features = encoder(dataset)
    half_life = dataset['thalf_hours'].astype(np.float64)
    if not np.isfinite(half_life).all() or (half_life < 0).any():
        raise ValueError('Half-lives must be finite and nonnegative')
    return (features,
            np.log(half_life + epsilon).astype(np.float32),
            dataset['source_row'])


class MLP(nn.Module):
    def __init__(self, inputs, hidden=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        size = inputs
        for width in hidden:
            layers += [nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)]
            size = width
        layers.append(nn.Linear(size, 1))
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x).squeeze(-1)


def correlations(truth, prediction):
    """Pearson and Spearman correlation, the latter with average ranks for ties.

    Undefined when either side is constant, as for the train-mean baseline;
    None is reported in that case.
    """
    truth, prediction = pd.Series(truth), pd.Series(prediction)
    if truth.std() == 0 or prediction.std() == 0:
        return None, None
    return (float(truth.corr(prediction)),
            float(truth.rank().corr(prediction.rank())))


def evaluate(model, features, target, device, batch=4096):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(features), batch):
            chunk = torch.from_numpy(features[start:start+batch]).to(device)
            predictions.append(model(chunk).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.zeros(0, dtype=np.float32)


def metrics(truth, prediction):
    error = prediction - truth
    pearson, spearman = correlations(truth, prediction)
    return dict(rows=int(len(truth)), rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(np.abs(error))), bias=float(np.mean(error)),
                pearson=pearson, spearman=spearman)


def train_one_split(split_index, splits_dir, *, epsilon, seed, epochs, patience,
                    batch_size, learning_rate, weight_decay, dropout, hidden,
                    validation_fraction, device, arm=arm_features.ONEHOT_ARM,
                    embeddings_dir='embeddings', pooling='flatten',
                    standardise='auto'):
    """Train, early-stop and score one split. Return history and metrics."""
    started = time.monotonic()
    train_file = Path(splits_dir)/f'training_{split_index}.npz'
    test_file = Path(splits_dir)/f'testing_{split_index}.npz'
    with np.load(train_file) as data:
        train_raw = {key: data[key] for key in data.files}
    with np.load(test_file) as data:
        test_raw = {key: data[key] for key in data.files}

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_raw['source_row']))
    n_validation = int(round(len(order)*validation_fraction))
    if not 0 < n_validation < len(order):
        raise ValueError('validation_fraction leaves no training or validation rows')
    validation_idx, train_idx = order[:n_validation], order[n_validation:]

    # Any fitted feature transform sees the training rows only, never the
    # validation rows it early-stops on and never the test set.
    encoder = None
    explained_variance = None
    cache_model, cache_dim = None, None
    if arm != arm_features.ONEHOT_ARM:
        encoder = arm_features.CachedArmEncoder(embeddings_dir, arm, pooling)
        encoder.fit({key: value[train_idx] for key, value in train_raw.items()})
        explained_variance = encoder.explained_variance
        # Recorded and printed because a cache built with the 8M development
        # model would otherwise work silently at d=320 instead of d=1280.
        cache_model, cache_dim = encoder.index.get('model'), encoder.dim

    x_all, y_all, _ = features_and_target(train_raw, epsilon, encoder=encoder)
    x_test, y_test, _ = features_and_target(test_raw, epsilon, encoder=encoder)
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_validation, y_validation = x_all[validation_idx], y_all[validation_idx]

    standardiser = None
    if arm_features.should_standardise(arm, standardise):
        standardiser = arm_features.Standardiser().fit(x_train)
        x_train, x_validation, x_test = (standardiser(x_train),
                                         standardiser(x_validation),
                                         standardiser(x_test))

    # Standardize the target with training statistics only; errors are reported
    # back in log units.
    center, scale = float(y_train.mean()), float(y_train.std())
    if scale <= 0:
        raise ValueError('Training target has zero variance')
    def standardize(values):
        return ((values-center)/scale).astype(np.float32)

    torch.manual_seed(seed)
    model = MLP(x_train.shape[1], hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    loss_function = nn.MSELoss()
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(standardize(y_train))
    generator = torch.Generator().manual_seed(seed)

    history = dict(epoch=[], train_loss=[], validation_loss=[], gradient_norm=[])
    best = dict(loss=np.inf, epoch=0, state=None)
    for epoch in range(1, epochs+1):
        model.train()
        permutation = torch.randperm(len(x_train_t), generator=generator)
        total_loss = total_norm = batches = 0.0
        for start in range(0, len(permutation), batch_size):
            rows = permutation[start:start+batch_size]
            inputs = x_train_t[rows].to(device)
            targets = y_train_t[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            optimizer.step()
            total_loss += float(loss.detach())*len(rows)
            total_norm += float(norm)
            batches += 1
        predictions = evaluate(model, x_validation, y_validation, device)*scale+center
        validation_loss = float(np.mean((predictions-y_validation)**2))
        history['epoch'].append(epoch)
        history['train_loss'].append(total_loss/len(permutation)*scale**2)
        history['validation_loss'].append(validation_loss)
        history['gradient_norm'].append(total_norm/batches)
        if validation_loss < best['loss']:
            best = dict(loss=validation_loss, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        elif epoch-best['epoch'] >= patience:
            break
    model.load_state_dict(best['state'])

    predictions = evaluate(model, x_test, y_test, device)*scale+center
    test_metrics = metrics(y_test, predictions)
    # A train-mean predictor bounds what the features have to beat.
    baseline = metrics(y_test, np.full(len(y_test), center, dtype=np.float32))
    # Rows whose peptide or HLA never appears in training are the hard cases the
    # split was built around.
    train_peptides = set(train_raw['peptide'].tolist())
    train_hla = set(train_raw['hla_pseudoseq'].tolist())
    unseen_peptide = np.array([p not in train_peptides for p in test_raw['peptide']])
    unseen_hla = np.array([h not in train_hla for h in test_raw['hla_pseudoseq']])
    subsets = dict(all=np.ones(len(y_test), dtype=bool),
                   seen_both=~unseen_peptide & ~unseen_hla,
                   unseen_peptide=unseen_peptide, unseen_hla=unseen_hla)
    return dict(split_index=split_index, arm=arm, pooling=pooling,
                standardised=standardiser is not None,
                explained_variance=explained_variance,
                cache_model=cache_model, cache_dim=cache_dim,
                fitted_state=arm_features.fitted_state(encoder, standardiser),
                train_rows=int(len(x_train)), validation_rows=int(len(x_validation)),
                test_rows=int(len(x_test)), features=int(x_train.shape[1]),
                target_center=center, target_scale=scale,
                epochs_run=len(history['epoch']), best_epoch=best['epoch'],
                best_validation_mse=best['loss'],
                final_gradient_norm=history['gradient_norm'][-1],
                test=test_metrics, baseline=baseline,
                test_subsets={name: metrics(y_test[mask], predictions[mask])
                              for name, mask in subsets.items() if mask.any()},
                history=history, seed=seed, train_seconds=time.monotonic()-started), model


def error_bars(values):
    """Whisker heights for a per-split mean; None when one split leaves them undefined."""
    return values.std(axis=0, ddof=1) if len(values) > 1 else None


def style_axes(ax):
    ax.set_facecolor(COLORS['surface'])
    ax.grid(axis='y', color=COLORS['grid'], linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(COLORS['axis'])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=COLORS['muted'], labelsize=8, length=3)


def convergence_figure(results, path):
    """Small multiples: loss and gradient norm per epoch, one column per split."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(14, 6), layout='constrained', facecolor=COLORS['surface'])
    FigureCanvasAgg(fig)
    # Share both axes across splits so the panels are directly comparable.
    axes = np.asarray(fig.subplots(2, len(results), sharex=True, squeeze=False))
    for row in axes:
        for ax in row[1:]:
            ax.sharey(row[0])
    for column, result in enumerate(results):
        history = result['history']
        top, bottom = axes[0, column], axes[1, column]
        top.plot(history['epoch'], history['train_loss'], color=COLORS['train'],
                 linewidth=1.8, label='Train')
        top.plot(history['epoch'], history['validation_loss'], color=COLORS['validation'],
                 linewidth=1.8, label='Validation')
        top.axvline(result['best_epoch'], color=COLORS['muted'], linewidth=1.0,
                    linestyle=(0, (4, 3)))
        top.set_title(f"Split {result['split_index']}", color=COLORS['ink'],
                      fontsize=10, pad=6)
        bottom.plot(history['epoch'], history['gradient_norm'], color=COLORS['gradient'],
                    linewidth=1.8, label='Gradient norm')
        bottom.set_yscale('log')
        bottom.set_xlabel('Epoch', color=COLORS['secondary'], fontsize=9)
        for ax in (top, bottom):
            style_axes(ax)
        if column == 0:
            top.set_ylabel('MSE, log units', color=COLORS['secondary'], fontsize=9)
            bottom.set_ylabel('Mean gradient L2 norm', color=COLORS['secondary'], fontsize=9)
    shared = max(result['history']['train_loss'][0] for result in results)
    for ax in axes[0]:
        ax.set_ylim(0, shared*1.05)
    for ax in axes[:, 1:].ravel():
        ax.tick_params(which='both', labelleft=False)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=COLORS['secondary'])
    axes[1, 0].legend(frameon=False, fontsize=8, labelcolor=COLORS['secondary'])
    fig.suptitle('Training convergence per split (dashed line: early-stopping epoch)',
                 color=COLORS['ink'], fontsize=12)
    fig.savefig(path, dpi=180, facecolor=COLORS['surface'])


def test_error_figure(results, path):
    """Mean test error over splits with standard-deviation bars, plus per-split dots."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    error_metrics = [('rmse', 'RMSE'), ('mae', 'MAE')]
    subsets = [('all', 'All test rows'), ('seen_both', 'Seen peptide & HLA'),
               ('unseen_peptide', 'Unseen peptide'), ('unseen_hla', 'Unseen HLA')]
    fig = Figure(figsize=(13, 5), layout='constrained', facecolor=COLORS['surface'])
    FigureCanvasAgg(fig)
    left, middle, right = fig.subplots(1, 3)

    positions = np.arange(len(error_metrics))
    for offset, (source, color, label) in enumerate(
            [('test', COLORS['train'], 'MLP'), ('baseline', COLORS['baseline'], 'Train-mean baseline')]):
        values = np.array([[result[source][key] for key, _ in error_metrics] for result in results])
        means, deviations = values.mean(axis=0), error_bars(values)
        x = positions + (offset-0.5)*0.3
        left.bar(x, means, width=0.26, color=color, label=label, zorder=2)
        left.errorbar(x, means, yerr=deviations, fmt='none', ecolor=COLORS['ink'],
                      elinewidth=1.2, capsize=4, zorder=3)
        for column in range(values.shape[1]):
            left.scatter(np.full(len(values), x[column]), values[:, column], s=14,
                         color=COLORS['surface'], edgecolor=COLORS['ink'],
                         linewidth=0.8, zorder=4)
    left.set_xticks(positions, [label for _, label in error_metrics])
    left.set_ylabel('Error, log units', color=COLORS['secondary'], fontsize=9)
    left.set_title('Test error, mean of 5 splits', color=COLORS['ink'], fontsize=11)
    left.legend(frameon=False, fontsize=8, labelcolor=COLORS['secondary'])

    positions = np.arange(len(subsets))
    rmse = np.array([[result['test_subsets'][name]['rmse'] for name, _ in subsets]
                     for result in results])
    means, deviations = rmse.mean(axis=0), error_bars(rmse)
    middle.bar(positions, means, width=0.5, color=COLORS['train'], zorder=2)
    middle.errorbar(positions, means, yerr=deviations, fmt='none', ecolor=COLORS['ink'],
                    elinewidth=1.2, capsize=4, zorder=3)
    for column in range(rmse.shape[1]):
        middle.scatter(np.full(len(rmse), positions[column]), rmse[:, column], s=14,
                       color=COLORS['surface'], edgecolor=COLORS['ink'],
                       linewidth=0.8, zorder=4)
    middle.set_xticks(positions, [label for _, label in subsets])
    middle.tick_params(axis='x', labelrotation=20)
    middle.set_ylabel('RMSE, log units', color=COLORS['secondary'], fontsize=9)
    middle.set_title('Test RMSE by subset', color=COLORS['ink'], fontsize=11)

    names = [('pearson', 'Pearson r'), ('spearman', 'Spearman rho')]
    positions = np.arange(len(names))
    values = np.array([[result['test'][key] for key, _ in names] for result in results])
    means, deviations = values.mean(axis=0), error_bars(values)
    right.bar(positions, means, width=0.4, color=COLORS['gradient'], zorder=2)
    right.errorbar(positions, means, yerr=deviations, fmt='none', ecolor=COLORS['ink'],
                   elinewidth=1.2, capsize=4, zorder=3)
    for column in range(values.shape[1]):
        right.scatter(np.full(len(values), positions[column]), values[:, column], s=14,
                      color=COLORS['surface'], edgecolor=COLORS['ink'],
                      linewidth=0.8, zorder=4)
    right.set_xticks(positions, [label for _, label in names])
    right.set_ylim(0, 1)
    right.set_ylabel('Correlation with observed ln half-life',
                     color=COLORS['secondary'], fontsize=9)
    right.set_title('Test correlation', color=COLORS['ink'], fontsize=11)

    for ax in (left, middle, right):
        style_axes(ax)
        ax.tick_params(axis='x', colors=COLORS['secondary'], labelsize=9)
    fig.suptitle('Test performance across the five splits (bars: mean, whiskers: SD, dots: splits)',
                 color=COLORS['ink'], fontsize=12)
    fig.savefig(path, dpi=180, facecolor=COLORS['surface'])


def mean_sd(values):
    """Mean and sample standard deviation over splits.

    The sample deviation needs two splits, so a single-split run reports None
    rather than the nan that ddof=1 returns there, which metrics.json cannot
    hold: it is written with allow_nan=False.
    """
    array = np.array(values)
    return dict(mean=float(array.mean()),
                sd=float(array.std(ddof=1)) if array.size > 1 else None,
                values=array.tolist())


def summary_table(results):
    """Mean and standard deviation of each test metric over the splits."""
    keys = ('rmse', 'mae', 'bias', 'pearson', 'spearman')
    table = {key: mean_sd([result['test'][key] for result in results])
             for key in keys}
    for name in results[0]['test_subsets']:
        table[f'rmse_{name}'] = mean_sd(
            [result['test_subsets'][name]['rmse'] for result in results])
    return table


def run(splits_dir='DATA', output_dir='RESULTS', *, epsilon=0.1, seed=0, epochs=300,
        patience=25, batch_size=256, learning_rate=1e-3, weight_decay=1e-5,
        dropout=0.2, hidden=(256, 128), validation_fraction=0.1, splits=5,
        device=None, save_models=True, arm=arm_features.ONEHOT_ARM,
        embeddings_dir='embeddings', pooling='flatten', standardise='auto'):
    """Train one MLP per split, write metrics, figures and optional weights.

    arm selects the input representation; everything else is held fixed, so a
    run differs from another arm's run only in its features.
    """
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for index in range(splits):
        result, model = train_one_split(index, splits_dir, epsilon=epsilon,
            seed=seed+index, epochs=epochs, patience=patience, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout,
            hidden=tuple(hidden), validation_fraction=validation_fraction, device=device,
            arm=arm, embeddings_dir=embeddings_dir, pooling=pooling,
            standardise=standardise)
        # Tensors, so they belong in the checkpoint and not in metrics.json.
        fitted_state = result.pop('fitted_state')
        if save_models:
            torch.save(dict(state_dict=model.state_dict(), hidden=tuple(hidden),
                            dropout=dropout, features=result['features'],
                            target_center=result['target_center'],
                            target_scale=result['target_scale'], epsilon=epsilon,
                            arm=arm, pooling=pooling, standardise=standardise,
                            embeddings_dir=(None if arm == arm_features.ONEHOT_ARM
                                            else str(Path(embeddings_dir).resolve())),
                            # The fitted feature transforms, without which a
                            # reloaded PLM model cannot rebuild its own inputs.
                            **fitted_state),
                       out/f'mlp_split_{index}.pt')
        print(f"[{arm}] split {index}: {result['features']} features"
              + (f" from {result['cache_model']} d={result['cache_dim']}"
                 if result['cache_model'] else '') + '  '
              f"best epoch {result['best_epoch']}/{result['epochs_run']}  "
              f"test RMSE {result['test']['rmse']:.3f}  "
              f"Spearman {result['test']['spearman']:.3f}  "
              f"({result['train_seconds']:.1f}s)", flush=True)
        results.append(result)
    convergence_figure(results, out/'convergence.png')
    test_error_figure(results, out/'test_error.png')
    onehot_scheme = 'one-hot of peptide stacked with HLA pseudosequence'
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  splits_dir=str(Path(splits_dir).resolve()), device=str(device),
                  arm=dict(name=arm, pooling=pooling, standardise=standardise,
                           standardised=results[0]['standardised'],
                           embeddings_dir=(None if arm == arm_features.ONEHOT_ARM
                                           else str(Path(embeddings_dir).resolve())),
                           explained_variance=results[0]['explained_variance'],
                           cache_model=results[0]['cache_model'],
                           cache_dim=results[0]['cache_dim']),
                  encoding=dict(alphabet=AMINO_ACIDS, peptide_length=9,
                                pseudoseq_length=34, features=results[0]['features'],
                                scheme=(onehot_scheme if arm == arm_features.ONEHOT_ARM
                                        else f'ESM-2 {arm} of the same 43 positions, '
                                             f'pooled with {pooling}')),
                  target=dict(transform='ln(thalf_hours + epsilon)', epsilon=epsilon,
                              standardized_for_training=True, units='log hours'),
                  model=dict(hidden=list(hidden), dropout=dropout, optimizer='Adam',
                             learning_rate=learning_rate, weight_decay=weight_decay,
                             batch_size=batch_size, max_epochs=epochs,
                             early_stopping_patience=patience,
                             validation_fraction=validation_fraction, loss='MSE'),
                  summary=summary_table(results), splits=results,
                  outputs={name: str((out/name).resolve())
                           for name in ('convergence.png', 'test_error.png', 'metrics.json')})
    (out/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--splits-dir', default='DATA', help='Directory holding training_i.npz/testing_i.npz')
    parser.add_argument('--output-dir', default='RESULTS')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Added to half-life before the logarithm (default: 0.1, the '
                             'reporting resolution of the assay)')
    parser.add_argument('--seed', type=int, default=0, help='Base seed; split i uses seed+i')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--splits', type=int, default=5)
    parser.add_argument('--device', default=None, help='cpu, cuda or mps; autodetected by default')
    parser.add_argument('--no-save-models', dest='save_models', action='store_false')
    parser.add_argument('--arm', default=arm_features.ONEHOT_ARM, choices=arm_features.ARMS,
                        help='Input representation: onehot (A0, no cache needed) or '
                             'L0/L15/L33 (A1-A3, from the embedding cache)')
    parser.add_argument('--embeddings-dir', default='embeddings',
                        help='Directory holding concat_L*.npy and index.json')
    parser.add_argument('--pooling', default='flatten',
                        help='flatten (43*D features), mean (2*D, over the peptide and '
                             'HLA blocks) or pca:K (43*K, one D->K projection fitted on '
                             'the training rows; pca:20 matches A0 feature for feature)')
    parser.add_argument('--standardise', default='auto', choices=('auto', 'always', 'never'),
                        help='Standardise features with training statistics. auto skips '
                             'the already unit-scale one-hot arm and standardises the '
                             'PLM arms, whose layers differ in scale by ~16x')
    args = vars(parser.parse_args())
    report = run(args.pop('splits_dir'), args.pop('output_dir'), **args)
    print(json.dumps({key: dict(mean=value['mean'], sd=value['sd'])
                      for key, value in report['summary'].items()}, indent=2))


if __name__ == '__main__':
    main()
