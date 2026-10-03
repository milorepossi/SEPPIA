#!/usr/bin/env python3
"""Tabulate the ladder: one row per arm, from each run's metrics.json.

Example:
    python scripts/compare_arms.py RESULTS/A0_onehot RESULTS/A1_L0 \
        RESULTS/A2_L15 RESULTS/A3_L33 --csv RESULTS/ladder.csv

Takes run directories (or metrics.json paths) in ladder order and reports mean
and standard deviation over the splits, plus the per-split paired difference
against the first arm given.

Pairing matters: every arm is evaluated on the same five splits, so the
comparison is paired. The sd of each arm's metric mixes split difficulty with
the arm's own variance, and split difficulty is shared, so a per-split delta is
the sharper read. An arm whose mean is barely higher but which wins on all five
splits is more convincing than the two error bars suggest.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HEADLINE = "spearman"
METRICS = ("spearman", "pearson", "rmse", "mae")
SUBSETS = ("rmse_seen_both", "rmse_unseen_peptide", "rmse_unseen_hla")


def load_run(path):
    path = Path(path)
    if path.is_dir():
        path = path/"metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run train_mlp.py for that arm first")
    report = json.loads(path.read_text())
    arm = report.get("arm") or {}
    return dict(
        label=f"{arm.get('name', 'onehot')}/{arm.get('pooling', '-')}",
        name=arm.get("name", "onehot"),
        pooling=arm.get("pooling", "-"),
        standardised=arm.get("standardised"),
        explained_variance=arm.get("explained_variance"),
        features=report["encoding"]["features"],
        hidden=tuple(report["model"]["hidden"]),
        epsilon=report["target"]["epsilon"],
        summary=report["summary"],
        n_splits=len(report["splits"]),
        baseline_rmse=float(np.mean([s["baseline"]["rmse"] for s in report["splits"]])),
        path=str(path),
    )


def comparable(runs):
    """Refuse to tabulate runs that differ in anything but their features."""
    problems = []
    first = runs[0]
    for run in runs[1:]:
        for field in ("hidden", "epsilon", "n_splits"):
            if run[field] != first[field]:
                problems.append(f"{run['label']} has {field}={run[field]!r}, "
                                f"{first['label']} has {first[field]!r}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run directories or metrics.json paths, "
                                                "reference arm first")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--metric", default=HEADLINE, help=f"Paired metric (default {HEADLINE})")
    args = parser.parse_args()

    runs = [load_run(path) for path in args.runs]
    problems = comparable(runs)
    if problems:
        print("WARNING: these runs differ in more than their features, so the "
              "comparison is not budget-matched:")
        for problem in problems:
            print(f"  - {problem}")
        print()

    width = max(len(run["label"]) for run in runs)
    header = (f"{'arm/pooling':<{width}}  {'feats':>6}  " +
              "  ".join(f"{name:>16}" for name in METRICS))
    print(header)
    print("-"*len(header))
    for run in runs:
        cells = []
        for name in METRICS:
            entry = run["summary"][name]
            sd = entry["sd"]
            cells.append(f"{entry['mean']:7.3f}+-{sd:5.3f}" if sd is not None
                         else f"{entry['mean']:7.3f}       ")
        print(f"{run['label']:<{width}}  {run['features']:>6}  " + "  ".join(cells))

    print(f"\nTrain-mean baseline RMSE: {runs[0]['baseline_rmse']:.3f} "
          f"(same target for every arm)")

    # Paired per-split deltas against the reference arm.
    reference = runs[0]
    values = np.array(reference["summary"][args.metric]["values"])
    print(f"\nPaired {args.metric} per split, against {reference['label']}")
    print(f"  {'arm/pooling':<{width}}  " +
          "  ".join(f"s{i}" for i in range(len(values))) + "   mean delta   wins")
    print(f"  {reference['label']:<{width}}  " +
          "  ".join(f"{v:.3f}" for v in values) + "        (reference)")
    for run in runs[1:]:
        other = np.array(run["summary"][args.metric]["values"])
        if len(other) != len(values):
            print(f"  {run['label']:<{width}}  different split count, skipped")
            continue
        delta = other-values
        better = delta > 0 if args.metric in ("spearman", "pearson") else delta < 0
        print(f"  {run['label']:<{width}}  " +
              "  ".join(f"{v:.3f}" for v in other) +
              f"   {delta.mean():+.4f}      {better.sum()}/{len(delta)}")

    for run in runs:
        if run["explained_variance"] is not None:
            print(f"\n{run['label']}: pca retains "
                  f"{run['explained_variance']:.1%} of position-vector variance")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["arm", "pooling", "features", "standardised", "n_splits"]
        fields += [f"{name}_{stat}" for name in (*METRICS, *SUBSETS) for stat in ("mean", "sd")]
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for run in runs:
                row = dict(arm=run["name"], pooling=run["pooling"],
                           features=run["features"], standardised=run["standardised"],
                           n_splits=run["n_splits"])
                for name in (*METRICS, *SUBSETS):
                    entry = run["summary"].get(name)
                    row[f"{name}_mean"] = entry["mean"] if entry else ""
                    row[f"{name}_sd"] = entry["sd"] if entry else ""
                writer.writerow(row)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
