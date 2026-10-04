#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis for the Rasmussen et al. Peptide-HLA Dataset.
Identifies distributions of HLAs, peptides, rarity tiers, binding characteristics,
and outputs structured statistical tables.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def analyze_dataset(data_path="data/rasmussen_et_al_dataset.csv", output_dir="splits"):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Basic properties
    n_rows = len(df)
    n_alleles = df["allele"].nunique()
    n_peptides = df["peptide"].nunique()
    n_pseudoseq = df["hla_pseudoseq"].nunique()
    n_fullseq = df["hla_seq"].nunique()

    # Classifications
    # Rasmussen et al. thresholding:
    # Standard binder threshold: t_half >= 1.0 hour
    # Stringent / strong binder threshold: t_half >= 2.0 hours
    # High stability: t_half >= 5.0 hours
    # Unbound / background zero: t_half == 0.0 hour
    df["is_zero"] = df["thalf_hours"] == 0.0
    df["is_binder_1h"] = df["thalf_hours"] >= 1.0
    df["is_binder_2h"] = df["thalf_hours"] >= 2.0
    df["is_binder_5h"] = df["thalf_hours"] >= 5.0

    df["locus"] = df["allele"].str.extract(r"HLA-([ABC])")[0]

    def categorize_stability(t):
        if t == 0:
            return "01_Unbound_Zero"
        elif t < 1.0:
            return "02_Weak_Sub1h"
        elif t < 2.0:
            return "03_Moderate_1to2h"
        elif t < 5.0:
            return "04_Strong_2to5h"
        else:
            return "05_VeryStrong_5hPlus"

    df["stability_tier"] = df["thalf_hours"].apply(categorize_stability)

    print("\n" + "=" * 60)
    print("DATASET GLOBAL SUMMARY")
    print("=" * 60)
    print(f"Total Rows: {n_rows:,}")
    print(f"Unique Alleles: {n_alleles}")
    print(f"Unique Peptides: {n_peptides:,}")
    print(f"Unique Pseudosequences: {n_pseudoseq}")
    print(f"Unique Full HLA Sequences: {n_fullseq}")
    print(f"Peptide Lengths: {df['peptide'].str.len().value_counts().to_dict()} (All 9-mers)")
    print(f"HLA Loci Breakdown: A={sum(df['locus']=='A'):,} ({sum(df['locus']=='A')/n_rows*100:.1f}%), B={sum(df['locus']=='B'):,} ({sum(df['locus']=='B')/n_rows*100:.1f}%)")

    print("\n" + "=" * 60)
    print("STABILITY / BINDING DISTRIBUTION")
    print("=" * 60)
    tier_counts = df["stability_tier"].value_counts().sort_index()
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count:,} ({count/n_rows*100:.2f}%)")

    print(f"  Overall Binders (>= 1.0h): {df['is_binder_1h'].sum():,} ({df['is_binder_1h'].mean()*100:.2f}%)")
    print(f"  Overall Non-binders (< 1.0h): {(~df['is_binder_1h']).sum():,} ({(~df['is_binder_1h']).mean()*100:.2f}%)")
    print(f"  Strict Binders (>= 2.0h): {df['is_binder_2h'].sum():,} ({df['is_binder_2h'].mean()*100:.2f}%)")

    # 2. Allele-level analysis
    allele_stats = df.groupby("allele").agg(
        locus=("locus", "first"),
        total_samples=("thalf_hours", "count"),
        unbound_zero_count=("is_zero", "sum"),
        binder_1h_count=("is_binder_1h", "sum"),
        binder_2h_count=("is_binder_2h", "sum"),
        binder_5h_count=("is_binder_5h", "sum"),
        mean_thalf=("thalf_hours", "mean"),
        median_thalf=("thalf_hours", "median"),
        std_thalf=("thalf_hours", "std"),
        max_thalf=("thalf_hours", "max"),
        pseudoseq=("hla_pseudoseq", "first"),
    )
    allele_stats["pct_binder_1h"] = (allele_stats["binder_1h_count"] / allele_stats["total_samples"] * 100).round(2)
    allele_stats["pct_binder_2h"] = (allele_stats["binder_2h_count"] / allele_stats["total_samples"] * 100).round(2)
    allele_stats["pct_unbound_zero"] = (allele_stats["unbound_zero_count"] / allele_stats["total_samples"] * 100).round(2)

    # Rarity tier definitions for HLA:
    # Ultra-rare: N < 50
    # Moderate-rare: 50 <= N < 300
    # Common: N >= 300
    def allele_rarity_tier(n):
        if n < 50:
            return "Ultra-Rare (<50)"
        elif n < 300:
            return "Moderate-Rare (50-299)"
        else:
            return "Common (>=300)"

    allele_stats["rarity_tier"] = allele_stats["total_samples"].apply(allele_rarity_tier)
    allele_stats = allele_stats.sort_values(by=["total_samples", "allele"], ascending=[True, True])

    # Save allele statistics
    allele_stats_file = output_dir / "allele_statistics.csv"
    allele_stats.to_csv(allele_stats_file)
    print(f"\nSaved allele statistics to {allele_stats_file}")

    rare_alleles = allele_stats[allele_stats["rarity_tier"] == "Ultra-Rare (<50)"]
    rare_alleles_file = output_dir / "rare_alleles.csv"
    rare_alleles.to_csv(rare_alleles_file)
    print(f"Saved {len(rare_alleles)} ultra-rare alleles to {rare_alleles_file}")

    print("\n" + "=" * 60)
    print("ULTRA-RARE HLA ALLELES (N < 50)")
    print("=" * 60)
    for allele, row in rare_alleles.iterrows():
        print(f"  {allele:<15} N={row['total_samples']:>2} | Binders(>=1h)={row['binder_1h_count']:>2} ({row['pct_binder_1h']:>5.1f}%) | Zeros={row['unbound_zero_count']:>2} | Median t1/2={row['median_thalf']:>4.1f}h | Max={row['max_thalf']:>5.1f}h")

    # 3. Peptide-level analysis
    pep_counts = df["peptide"].value_counts()
    pep_bind = df.groupby("peptide")["is_binder_1h"].agg(["count", "sum", "mean"]).reset_index()
    pep_bind.columns = ["peptide", "total_tests", "binder_count", "binder_ratio"]

    def pep_freq_tier(n):
        if n == 1:
            return "1_Singleton (N=1)"
        elif n <= 3:
            return "2_Rare (N=2-3)"
        elif n <= 8:
            return "3_Moderate (N=4-8)"
        else:
            return "4_Frequent (N>=9)"

    pep_bind["freq_tier"] = pep_bind["total_tests"].apply(pep_freq_tier)

    def pep_promiscuity_tier(row):
        if row["binder_ratio"] == 0:
            return "Universal Non-Binder"
        elif row["binder_ratio"] == 1.0:
            return "Universal Binder"
        else:
            return "Selective Binder"

    pep_bind["binding_profile"] = pep_bind.apply(pep_promiscuity_tier, axis=1)

    pep_summary_file = output_dir / "peptide_statistics.csv"
    pep_bind.to_csv(pep_summary_file, index=False)
    print(f"Saved peptide statistics to {pep_summary_file}")

    print("\n" + "=" * 60)
    print("PEPTIDE FREQUENCY & RARITY TIERS")
    print("=" * 60)
    p_tier_counts = pep_bind["freq_tier"].value_counts().sort_index()
    for tier, count in p_tier_counts.items():
        print(f"  {tier}: {count:,} unique peptides ({count/n_peptides*100:.2f}%)")

    print("\n" + "=" * 60)
    print("PEPTIDE BINDING PROMISCUITY PROFILES")
    print("=" * 60)
    p_bind_counts = pep_bind["binding_profile"].value_counts()
    for prof, count in p_bind_counts.items():
        print(f"  {prof}: {count:,} ({count/n_peptides*100:.2f}%)")

    # Return key objects for plotting and splitting
    return df, allele_stats, pep_bind


if __name__ == "__main__":
    analyze_dataset()
