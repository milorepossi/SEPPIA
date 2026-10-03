#!/usr/bin/env python3
"""Choose the pilot subset to extract Boltz-2 features for, and fix the target.

Two jobs, both cheap and both CPU-only.

1. Target correction
--------------------
The committed `.xlsx` had 42% of its `thalf_hours` cells mangled by Excel into
datetimes. `split_dataset.py:half_life()` inverts that as `float(day.month)`,
and `docs/02_next_steps.md` states the inversion is lossless. It is not.

Values of the form `x.0Y` collapse: a true `1.05` was stored as `2026-05-01`
and recovers as `1.5`, because the leading zero of the hundredths place is lost.
26 rows across 19 distinct values are affected, every one of them inflated
roughly tenfold in the decimal part. The clean CSV holds 944 distinct values
against the recovery's 925 and agrees on every other row, so the CSV is the
ground truth and the recovery is not invertible after all.

The audit in `scripts/audit_target.py` missed this because it compared recovered
values against *surviving string* values and found no overlap. The collapse is
between two mangled values, not between a mangled and a surviving one: both
`1.5` and `1.05` parse to the same datetime.

Split *membership* is unaffected in any way that matters (26 of 28,166 rows will
not move a moment-matching constraint), so the committed splits stay as they
are. Only the target is re-read.

2. Pilot subset
---------------
Boltz-2 needs one forward pass per complex and cannot cache per entity, so a
$20 budget cannot cover all 28,166 rows. The pilot therefore selects for
*breadth of allele* rather than depth: the generalisation question the splits
were built for is per-allele, and with only ~10 alleles the confidence interval
on held-out-allele performance is too wide to conclude anything.

Selection rules, in order:
  - keep every allele the splits hold out as test-exclusive (otherwise the
    unseen-HLA stratum cannot be scored at all)
  - spread the remainder over both loci, preferring well-populated alleles
  - within an allele, stratify peptides across the half-life range rather than
    sampling uniformly, so each allele spans its own dynamic range
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_ROW_OFFSET = 2   # split_dataset.py: source_row = arange(2, n+2)


# --------------------------------------------------------------------------- #
# target
# --------------------------------------------------------------------------- #
def load_clean(csv_path) -> pd.DataFrame:
    """The uncorrupted CSV, with source_row attached to match the splits."""
    df = pd.read_csv(csv_path)
    df["source_row"] = np.arange(SOURCE_ROW_OFFSET, len(df) + SOURCE_ROW_OFFSET)
    return df


def audit_target(csv_path, xlsx_path) -> dict:
    """Quantify where the Excel recovery disagrees with the clean CSV."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from split_dataset import half_life

    clean = load_clean(csv_path)
    raw = pd.read_excel(xlsx_path)
    if not (raw["peptide"].to_numpy() == clean["peptide"].to_numpy()).all():
        raise RuntimeError("CSV and XLSX are not row-aligned; cannot audit")

    recovered = raw["thalf_hours"].map(half_life).astype(float).to_numpy()
    truth = clean["thalf_hours"].to_numpy(dtype=float)
    bad = np.flatnonzero(np.abs(recovered - truth) > 1e-9)
    return {
        "n_rows": int(len(clean)),
        "n_disagreeing": int(len(bad)),
        "fraction_disagreeing": round(float(len(bad) / len(clean)), 6),
        "distinct_values_csv": int(len(np.unique(truth))),
        "distinct_values_recovered": int(len(np.unique(recovered))),
        "max_abs_error": float(np.abs(recovered - truth).max()),
        "affected_source_rows": [int(v) for v in clean["source_row"].to_numpy()[bad]],
        "examples": [
            {"source_row": int(clean["source_row"].iloc[i]),
             "truth": float(truth[i]), "recovered": float(recovered[i])}
            for i in bad[:10]
        ],
        "pattern": "true values of the form x.0Y recover as x.Y (leading zero of "
                   "the hundredths place is lost)",
        "conclusion": "the CSV is ground truth; split membership is unaffected",
    }


# --------------------------------------------------------------------------- #
# subset
# --------------------------------------------------------------------------- #
def test_exclusive_alleles(splits_dir: Path, n_splits: int = 5) -> set[str]:
    """Every hla_seq that any split holds out, mapped back to allele names."""
    held = set()
    for i in range(n_splits):
        meta = json.loads((splits_dir / f"split/metadata_{i}.json").read_text())
        held.update(meta["all_test_exclusive_hla"])
    return held


def choose_subset(df: pd.DataFrame, held_hla: set[str], *, n_alleles: int,
                  per_allele: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["locus"] = df["allele"].str[4]

    held_alleles = sorted(df.loc[df["hla_seq"].isin(held_hla), "allele"].unique())
    counts = df["allele"].value_counts()
    remaining = [a for a in counts.index if a not in held_alleles]

    # alternate loci so neither dominates, preferring well-populated alleles
    by_locus = {L: [a for a in remaining if a.startswith(f"HLA-{L}")] for L in "AB"}
    picked = list(held_alleles)
    turn = "A"
    while len(picked) < n_alleles and (by_locus["A"] or by_locus["B"]):
        pool = by_locus[turn] or by_locus["B" if turn == "A" else "A"]
        if pool:
            picked.append(pool.pop(0))
        turn = "B" if turn == "A" else "A"

    chunks = []
    for allele in picked:
        sub = df[df["allele"] == allele]
        take = min(per_allele, len(sub))
        if take == len(sub):
            chunks.append(sub)
            continue
        # stratify across the half-life range: split into `take` quantile bins
        # and draw one peptide from each, so the allele spans its own range
        order = sub.sort_values("thalf_hours").reset_index(drop=True)
        edges = np.linspace(0, len(order), take + 1).astype(int)
        rows = [order.iloc[rng.integers(lo, hi)] for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]
        chunks.append(pd.DataFrame(rows))

    out = pd.concat(chunks).drop_duplicates("source_row").sort_values("source_row")
    return out.reset_index(drop=True)


def describe(subset: pd.DataFrame, full: pd.DataFrame) -> dict:
    return {
        "n_complexes": int(len(subset)),
        "n_alleles": int(subset["allele"].nunique()),
        "n_unique_hla_seq": int(subset["hla_seq"].nunique()),
        "n_unique_peptides": int(subset["peptide"].nunique()),
        "loci": subset["allele"].str[4].value_counts().to_dict(),
        "rows_per_allele": {
            "min": int(subset["allele"].value_counts().min()),
            "median": int(subset["allele"].value_counts().median()),
            "max": int(subset["allele"].value_counts().max()),
        },
        "target": {
            "zeros": int((subset["thalf_hours"] == 0).sum()),
            "zero_fraction": round(float((subset["thalf_hours"] == 0).mean()), 4),
            "median": float(subset["thalf_hours"].median()),
            "p90": float(subset["thalf_hours"].quantile(0.9)),
            "max": float(subset["thalf_hours"].max()),
        },
        "full_dataset_zero_fraction": round(float((full["thalf_hours"] == 0).mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="DATA/rasmussen_clean.csv")
    ap.add_argument("--xlsx", default="DATA/rasmussen_et_al_dataset.xlsx")
    ap.add_argument("--splits-dir", default="DATA", type=Path)
    ap.add_argument("--n-alleles", type=int, default=28)
    ap.add_argument("--per-allele", type=int, default=110)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-rows", default="DATA/pilot_source_rows.json")
    ap.add_argument("--out-report", default="DATA/pilot_subset.json")
    ap.add_argument("--skip-audit", action="store_true")
    args = ap.parse_args()

    df = load_clean(args.csv)
    report: dict = {}
    if not args.skip_audit and Path(args.xlsx).exists():
        report["target_audit"] = audit_target(args.csv, args.xlsx)

    held = test_exclusive_alleles(args.splits_dir)
    subset = choose_subset(df, held, n_alleles=args.n_alleles,
                           per_allele=args.per_allele, seed=args.seed)
    report["subset"] = describe(subset, df)
    report["subset"]["covers_all_test_exclusive_hla"] = bool(
        held <= set(subset["hla_seq"]))
    report["parameters"] = {"n_alleles": args.n_alleles,
                            "per_allele": args.per_allele, "seed": args.seed}

    Path(args.out_rows).write_text(
        json.dumps([int(v) for v in subset["source_row"]]) + "\n")
    Path(args.out_report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
