#!/usr/bin/env python3
"""Audit the thalf_hours column and prove the Excel-date recovery is lossless.

42% of thalf_hours cells were mangled by Excel into datetimes: a source value
like "1.8" was stored as 2026-08-01, i.e. day.month of the current year.
split_dataset.py:half_life() inverts this with f'{value.day}.{value.month}'.

This script verifies that inversion is unambiguous rather than assuming it:

  1. Every datetime cell carries one consistent parse (same year, midnight).
  2. The stored day holds the integer part and the month the fractional part,
     so day.month is the right orientation and not month.day.
  3. The set of values recovered from datetimes and the set of values that
     survived as strings are DISJOINT.

(3) is the load-bearing check. The corruption is a deterministic function of
the value, so a number readable as day.month was eaten in every row it
appeared in, and one that is not readable (1.15 has no month 15, 1.0 has no
month 0) survived in every row. Disjoint populations therefore mean the
inverse map is well defined; an overlap would mean two different source values
collapse onto one number and the column could not be trusted.

Usage:
    python scripts/audit_target.py rasmussen_et_al_dataset.xlsx
"""
import argparse
from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd

TARGET_COLUMN = "thalf_hours"


def half_life(value):
    """Identical to split_dataset.py:half_life, kept here only for the audit."""
    if pd.isna(value):
        raise ValueError("Missing half-life")
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return float(f"{value.day}.{value.month}")
    return float(str(value).strip().replace(",", "."))


def audit(dataset_path, target_column=TARGET_COLUMN):
    frame = pd.read_excel(dataset_path)
    column = frame[target_column]
    is_datetime = column.map(lambda v: isinstance(v, (date, datetime, pd.Timestamp)))
    datetimes = column[is_datetime]
    strings = column[~is_datetime]

    print(f"{target_column}: {len(column)} cells = "
          f"{len(datetimes)} datetime ({len(datetimes) / len(column):.1%}) + "
          f"{len(strings)} string")

    failures = []

    # 1. One consistent Excel parse.
    years = sorted({v.year for v in datetimes})
    midnight = all((v.hour, v.minute, v.second) == (0, 0, 0) for v in datetimes)
    print(f"  years={years} all_midnight={midnight}")
    if len(years) != 1 or not midnight:
        failures.append("datetime cells do not share one consistent Excel parse")

    # 2. Orientation: days carry the integer part, months the fractional part.
    days = {v.day for v in datetimes}
    months = {v.month for v in datetimes}
    print(f"  day range={min(days)}..{max(days)}  month range={min(months)}..{max(months)}")
    if max(months) > 12 or max(days) > 31:
        failures.append("day/month ranges are impossible")
    if max(days) <= 12:
        failures.append("day range never exceeds 12, so day.month vs month.day "
                        "cannot be distinguished from the data alone")

    # 3. The two populations must be disjoint.
    recovered = {half_life(v) for v in datetimes}
    survived = {half_life(v) for v in strings}
    overlap = sorted(recovered & survived)
    print(f"  {len(recovered)} distinct values recovered from datetimes, "
          f"{len(survived)} distinct surviving string values")
    print(f"  overlap = {len(overlap)} {overlap[:10] if overlap else ''}")
    if overlap:
        failures.append(f"recovered and surviving value sets overlap on {overlap[:10]}: "
                        "the inverse map is ambiguous")

    # Recovered values must sit inside the plausible range set by the clean cells.
    print(f"  recovered range  {min(recovered):.2f}..{max(recovered):.2f}")
    print(f"  string    range  {min(survived):.3f}..{max(survived):.1f}")
    if min(recovered) < min(survived) or max(recovered) > max(survived):
        failures.append("recovered values fall outside the range of the clean cells")

    # Distribution of the recovered column, for the modelling decisions.
    recovered_column = column.map(half_life)
    zeros = int((recovered_column == 0).sum())
    print(f"\nrecovered {target_column}: median={recovered_column.median():.2f} "
          f"mean={recovered_column.mean():.2f} max={recovered_column.max():.1f} "
          f"n_unique={recovered_column.nunique()}")
    print(f"  zeros (left-censored): {zeros} ({zeros / len(recovered_column):.1%})")
    print("  quantiles: " + "  ".join(
        f"p{int(q * 100)}={recovered_column.quantile(q):.2f}"
        for q in (0.25, 0.5, 0.75, 0.9, 0.99)))

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return False
    print("\nPASS: the day.month inversion is unambiguous and lossless. "
          "Import split_dataset.half_life rather than writing another parser.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    args = parser.parse_args()
    sys.exit(0 if audit(args.dataset, args.target_column) else 1)


if __name__ == "__main__":
    main()
