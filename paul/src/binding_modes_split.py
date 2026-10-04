#!/usr/bin/env python3
"""
Binding Modes Dataset Partition & Peptide-Level Split Generator.

Partitions the Rasmussen et al. Peptide-HLA dataset into 3 biological regimes:
1. Non-binding:     thalf_hours < 1.0 h   (N = 13,540)
2. Mildly binding:  1.0 h <= thalf < 2.0 h (N = 3,171)
3. Strong binders:  thalf_hours >= 2.0 h  (N = 11,455)

For each binding mode, produces a strictly disjoint PEPTIDE-LEVEL split
(70% Train, 15% Val, 15% Test) with 0 peptide leakage across sets,
stratified by peptide observation frequency.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_binding_modes_splits(
    data_path: str = "data/rasmussen_et_al_dataset.csv",
    output_dir: str = "splits",
    random_seed: int = 42,
) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
    data_file = project_root / data_path
    out_dir = project_root / output_dir
    modes_dir = out_dir / "binding_modes"
    out_dir.mkdir(parents=True, exist_ok=True)
    modes_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("PEPTIDE-HLA BINDING MODES PARTITION & PEPTIDE-LEVEL SPLITS")
    print("=" * 76)
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    n_total = len(df)
    print(f"Total dataset samples: {n_total:,}")

    # Define Binding Modes
    def assign_mode(t: float) -> str:
        if t < 1.0:
            return "non_binding"
        elif t < 2.0:
            return "mild_binding"
        else:
            return "strong_binding"

    df["binding_mode"] = df["thalf_hours"].apply(assign_mode)
    df["locus"] = df["allele"].str.extract(r"HLA-([ABC])")[0]

    modes = ["non_binding", "mild_binding", "strong_binding"]
    metadata = {
        "dataset_total_samples": n_total,
        "random_seed": random_seed,
        "modes": {},
    }

    print("\nGlobal Binding Mode Distribution:")
    for mode in modes:
        mode_count = (df["binding_mode"] == mode).sum()
        pct = (mode_count / n_total) * 100
        print(f"  • {mode:<15}: {mode_count:>6,} pairs ({pct:>5.1f}%)")

    # Generate splits for each mode
    for mode in modes:
        print("\n" + "-" * 76)
        print(f"PROCESSING BINDING MODE: {mode.upper()}")
        print("-" * 76)
        sub_df = df[df["binding_mode"] == mode].copy().reset_index(drop=True)

        # Count occurrences of each peptide to stratify
        pep_counts = sub_df["peptide"].value_counts().reset_index()
        pep_counts.columns = ["peptide", "obs_count"]

        def freq_tier(c: int) -> str:
            if c == 1:
                return "1_singleton"
            elif c <= 3:
                return "2_rare"
            elif c <= 8:
                return "3_mid"
            else:
                return "4_high"

        pep_counts["strata"] = pep_counts["obs_count"].apply(freq_tier)

        # 70% Train, 15% Val, 15% Test at the unique PEPTIDE level
        train_peps_df, test_val_peps_df = train_test_split(
            pep_counts,
            test_size=0.30,
            random_state=random_seed,
            stratify=pep_counts["strata"],
        )
        val_peps_df, test_peps_df = train_test_split(
            test_val_peps_df,
            test_size=0.50,
            random_state=random_seed,
            stratify=test_val_peps_df["strata"],
        )

        train_peps = set(train_peps_df["peptide"])
        val_peps = set(val_peps_df["peptide"])
        test_peps = set(test_peps_df["peptide"])

        # Strict validation: Zero peptide leakage
        assert len(train_peps & val_peps) == 0, f"Peptide leakage between train and val in {mode}!"
        assert len(train_peps & test_peps) == 0, f"Peptide leakage between train and test in {mode}!"
        assert len(val_peps & test_peps) == 0, f"Peptide leakage between val and test in {mode}!"

        # Assign split label
        sub_df["split"] = ""
        sub_df.loc[sub_df["peptide"].isin(train_peps), "split"] = "train"
        sub_df.loc[sub_df["peptide"].isin(val_peps), "split"] = "val"
        sub_df.loc[sub_df["peptide"].isin(test_peps), "split"] = "test"

        df_train = sub_df[sub_df["split"] == "train"].copy().reset_index(drop=True)
        df_val = sub_df[sub_df["split"] == "val"].copy().reset_index(drop=True)
        df_test = sub_df[sub_df["split"] == "test"].copy().reset_index(drop=True)

        # Save to both out_dir (for flat access) and modes_dir
        for target_dir in [out_dir, modes_dir]:
            df_train.to_csv(target_dir / f"{mode}_train.csv", index=False)
            df_val.to_csv(target_dir / f"{mode}_val.csv", index=False)
            df_test.to_csv(target_dir / f"{mode}_test.csv", index=False)
            sub_df.to_csv(target_dir / f"{mode}_all.csv", index=False)

        mode_stats = {
            "total_samples": len(sub_df),
            "total_peptides": len(pep_counts),
            "total_alleles": int(sub_df["allele"].nunique()),
            "thalf_min": float(sub_df["thalf_hours"].min()),
            "thalf_max": float(sub_df["thalf_hours"].max()),
            "thalf_mean": float(sub_df["thalf_hours"].mean()),
            "thalf_median": float(sub_df["thalf_hours"].median()),
            "thalf_std": float(sub_df["thalf_hours"].std()),
            "train": {
                "samples": len(df_train),
                "sample_pct": float(len(df_train) / len(sub_df) * 100),
                "unique_peptides": len(train_peps),
                "unique_alleles": int(df_train["allele"].nunique()),
                "thalf_mean": float(df_train["thalf_hours"].mean()),
            },
            "val": {
                "samples": len(df_val),
                "sample_pct": float(len(df_val) / len(sub_df) * 100),
                "unique_peptides": len(val_peps),
                "unique_alleles": int(df_val["allele"].nunique()),
                "thalf_mean": float(df_val["thalf_hours"].mean()),
            },
            "test": {
                "samples": len(df_test),
                "sample_pct": float(len(df_test) / len(sub_df) * 100),
                "unique_peptides": len(test_peps),
                "unique_alleles": int(df_test["allele"].nunique()),
                "thalf_mean": float(df_test["thalf_hours"].mean()),
            },
        }

        metadata["modes"][mode] = mode_stats

        print(f"  Total samples:  {len(sub_df):,} | Peptides: {len(pep_counts):,} | Alleles: {sub_df['allele'].nunique()}")
        print(f"  t_half range:   {mode_stats['thalf_min']:.2f}h to {mode_stats['thalf_max']:.2f}h (mean: {mode_stats['thalf_mean']:.2f}h, med: {mode_stats['thalf_median']:.2f}h)")
        print(f"  Train split:    {len(df_train):,} pairs ({mode_stats['train']['sample_pct']:.1f}%) | {len(train_peps):,} peptides | {df_train['allele'].nunique()} alleles")
        print(f"  Val split:      {len(df_val):,} pairs ({mode_stats['val']['sample_pct']:.1f}%) | {len(val_peps):,} peptides | {df_val['allele'].nunique()} alleles")
        print(f"  Test split:     {len(df_test):,} pairs ({mode_stats['test']['sample_pct']:.1f}%) | {len(test_peps):,} peptides | {df_test['allele'].nunique()} alleles")
        print(f"  Peptide leakage check: PASSED (0 overlap between train/val/test)")

    # Save summary metadata JSON
    meta_path = modes_dir / "binding_modes_split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(out_dir / "binding_modes_split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 76)
    print(f"All binding mode splits generated successfully and saved to:")
    print(f"  • {out_dir}/")
    print(f"  • {modes_dir}/")
    print(f"Metadata summary written to: {meta_path}")
    print("=" * 76)
    return metadata


if __name__ == "__main__":
    create_binding_modes_splits()
