#!/usr/bin/env python3
"""
Scientific Split Generator for Rasmussen et al. Peptide-HLA Dataset.
Produces 5 complementary splitting regimes designed for machine learning
and protein foundation model evaluation:
1. IID Random Split (Baseline benchmark)
2. Novel Peptide Held-out Split (Antigen generalization)
3. Novel HLA Allele Held-out Split (Pan-allele generalization)
4. Double Held-out 4-Quadrant Split (Strict biophysical interaction benchmark)
5. Rare Allele Ablation Benchmark (Zero-shot and few-shot evaluation on n < 50 alleles)

NOTE: This is a standalone, clean splitting suite and does NOT use split_dataset.py.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_all_splits(
    data_path="data/rasmussen_et_al_dataset.csv",
    output_dir="splits",
    random_seed=42,
):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    n_total = len(df)

    # Classifications
    df["locus"] = df["allele"].str.extract(r"HLA-([ABC])")[0]
    df["is_zero"] = df["thalf_hours"] == 0.0
    df["is_binder_1h"] = df["thalf_hours"] >= 1.0
    df["is_binder_2h"] = df["thalf_hours"] >= 2.0

    def categorize_stability(t):
        if t == 0:
            return "Unbound_Zero"
        elif t < 1.0:
            return "Weak_Sub1h"
        elif t < 2.0:
            return "Moderate_1to2h"
        elif t < 5.0:
            return "Strong_2to5h"
        else:
            return "VeryStrong_5hPlus"

    df["stability_tier"] = df["thalf_hours"].apply(categorize_stability)

    # =========================================================================
    # 1. IID RANDOM SPLIT (70% Train, 15% Val, 15% Test)
    # Stratified by locus and binder_1h
    # =========================================================================
    print("\n[1/5] Generating IID Random Split...")
    strat_col = df["locus"] + "_" + df["is_binder_1h"].astype(str)
    train_idx, test_val_idx = train_test_split(
        df.index, test_size=0.30, random_state=random_seed, stratify=strat_col
    )
    val_idx, test_idx = train_test_split(
        test_val_idx,
        test_size=0.50,
        random_state=random_seed,
        stratify=strat_col.loc[test_val_idx],
    )

    df["iid_split"] = ""
    df.loc[train_idx, "iid_split"] = "train"
    df.loc[val_idx, "iid_split"] = "val"
    df.loc[test_idx, "iid_split"] = "test"

    # =========================================================================
    # 2. NOVEL PEPTIDE HELD-OUT SPLIT
    # Disjoint peptide sets: Train Peptides (70%), Val Peptides (15%), Test Peptides (15%)
    # Stratified by frequency tier and binder profile
    # =========================================================================
    print("[2/5] Generating Novel Peptide Held-out Split...")
    pep_df = (
        df.groupby("peptide")
        .agg(total=("thalf_hours", "count"), binder_ratio=("is_binder_1h", "mean"))
        .reset_index()
    )

    def pep_strata(row):
        if row["total"] == 1:
            freq = "singleton"
        elif row["total"] <= 3:
            freq = "rare"
        elif row["total"] <= 8:
            freq = "mid"
        else:
            freq = "high"

        if row["binder_ratio"] == 0:
            b = "non"
        elif row["binder_ratio"] == 1.0:
            b = "binder"
        else:
            b = "selective"
        return f"{freq}_{b}"

    pep_df["strata"] = pep_df.apply(pep_strata, axis=1)

    tr_p, tv_p = train_test_split(
        pep_df, test_size=0.30, random_state=random_seed, stratify=pep_df["strata"]
    )
    va_p, te_p = train_test_split(
        tv_p, test_size=0.50, random_state=random_seed, stratify=tv_p["strata"]
    )

    pep_map = {}
    for p in tr_p["peptide"]:
        pep_map[p] = "train"
    for p in va_p["peptide"]:
        pep_map[p] = "val"
    for p in te_p["peptide"]:
        pep_map[p] = "test"

    df["peptide_split"] = df["peptide"].map(pep_map)

    # Verification: check 0 peptide leakage
    train_peps = set(df[df["peptide_split"] == "train"]["peptide"])
    val_peps = set(df[df["peptide_split"] == "val"]["peptide"])
    test_peps = set(df[df["peptide_split"] == "test"]["peptide"])
    assert len(train_peps & test_peps) == 0, "Peptide leakage in test!"
    assert len(train_peps & val_peps) == 0, "Peptide leakage in val!"
    assert len(val_peps & test_peps) == 0, "Peptide leakage between val and test!"

    # =========================================================================
    # 3. NOVEL HLA ALLELE HELD-OUT SPLIT
    # Disjoint allele sets: Train Alleles (49), Val Alleles (10), Test Alleles (9)
    # Plus dedicated Rare Alleles holdout (7 alleles with N < 50)
    # Stratified by locus and binder ratio
    # =========================================================================
    print("[3/5] Generating Novel HLA Allele Held-out Split...")
    allele_df = (
        df.groupby("allele")
        .agg(
            total=("thalf_hours", "count"),
            binder_ratio=("is_binder_1h", "mean"),
            locus=("locus", "first"),
        )
        .reset_index()
    )

    ultra_rare_alleles = allele_df[allele_df["total"] < 50]["allele"].tolist()
    regular = allele_df[allele_df["total"] >= 50].copy()
    regular = regular.sort_values(by=["locus", "binder_ratio"]).reset_index(drop=True)

    val_idx_a, test_idx_a, train_idx_a = [], [], []
    for locus in ["A", "B"]:
        loc_sub = regular[regular["locus"] == locus].index.tolist()
        for i, idx in enumerate(loc_sub):
            if i % 7 == 1:
                val_idx_a.append(idx)
            elif i % 7 == 4:
                test_idx_a.append(idx)
            else:
                train_idx_a.append(idx)

    train_alleles = regular.loc[train_idx_a, "allele"].tolist()
    val_alleles = regular.loc[val_idx_a, "allele"].tolist()
    test_alleles = regular.loc[test_idx_a, "allele"].tolist()

    allele_map = {}
    for a in train_alleles:
        allele_map[a] = "train"
    for a in val_alleles:
        allele_map[a] = "val"
    for a in test_alleles:
        allele_map[a] = "test"
    for a in ultra_rare_alleles:
        allele_map[a] = "rare_holdout"

    df["allele_split"] = df["allele"].map(allele_map)

    # Verification: check 0 allele leakage
    train_a_set = set(train_alleles)
    test_a_set = set(test_alleles)
    val_a_set = set(val_alleles)
    rare_a_set = set(ultra_rare_alleles)
    assert len(train_a_set & test_a_set) == 0, "Allele leakage in test!"
    assert len(train_a_set & val_a_set) == 0, "Allele leakage in val!"
    assert len(train_a_set & rare_a_set) == 0, "Rare allele leakage in train!"

    # =========================================================================
    # 4. DOUBLE HELD-OUT 4-QUADRANT SPLIT
    # Unified Cartesian decomposition of Allele Partition x Peptide Partition
    # =========================================================================
    print("[4/5] Generating 4-Quadrant Double Held-out Split...")

    def assign_quadrant(row):
        a_sp = row["allele_split"]
        p_sp = row["peptide_split"]

        if a_sp == "rare_holdout":
            return "rare_allele_holdout"
        elif a_sp == "train" and p_sp == "train":
            return "train_both_seen"
        elif a_sp == "train" and p_sp == "test":
            return "test_unseen_pep"
        elif a_sp == "test" and p_sp == "train":
            return "test_unseen_allele"
        elif a_sp == "test" and p_sp == "test":
            return "test_double_unseen"
        elif a_sp == "train" and p_sp == "val":
            return "val_unseen_pep"
        elif a_sp == "val" and p_sp == "train":
            return "val_unseen_allele"
        elif a_sp == "val" and p_sp == "val":
            return "val_double_unseen"
        else:
            return "other_val_test_mix"

    df["quadrant_split"] = df.apply(assign_quadrant, axis=1)

    # =========================================================================
    # 5. RARE ALLELE ABLATION BENCHMARK (N < 50 alleles)
    # Zero-shot holdout vs Few-shot (3-shot per rare allele)
    # =========================================================================
    print("[5/5] Generating Rare Allele Ablation Benchmark...")
    df["rare_allele_flag"] = df["allele"].isin(ultra_rare_alleles)

    # Create 3-shot support set and query set for rare alleles
    rng = np.random.default_rng(random_seed)
    rare_support_idx = []
    rare_query_idx = []

    for r_allele in ultra_rare_alleles:
        allele_indices = df[df["allele"] == r_allele].index.to_numpy().copy()
        rng.shuffle(allele_indices)
        # Take 3 shots for few-shot adaptation support, rest for query/evaluation
        k_shot = min(3, len(allele_indices) // 2)
        rare_support_idx.extend(allele_indices[:k_shot])
        rare_query_idx.extend(allele_indices[k_shot:])

    df["rare_ablation_mode"] = "common_pool"
    df.loc[rare_support_idx, "rare_ablation_mode"] = "rare_3shot_support"
    df.loc[rare_query_idx, "rare_ablation_mode"] = "rare_heldout_query"

    # =========================================================================
    # SAVE MASTER ANNOTATED DATASET AND DEDICATED SPLIT FILES
    # =========================================================================
    master_path = output_dir / "dataset_with_splits.csv"
    df.to_csv(master_path, index=False)
    print(f"\nSaved master dataset with all split annotations to {master_path}")

    # Dedicated CSV files for easy plug-and-play machine learning
    # Regime 1: IID Random
    df[df["iid_split"] == "train"].to_csv(output_dir / "iid_train.csv", index=False)
    df[df["iid_split"] == "val"].to_csv(output_dir / "iid_val.csv", index=False)
    df[df["iid_split"] == "test"].to_csv(output_dir / "iid_test.csv", index=False)

    # Regime 2: Novel Peptide Generalization
    df[df["peptide_split"] == "train"].to_csv(output_dir / "novel_pep_train.csv", index=False)
    df[df["peptide_split"] == "val"].to_csv(output_dir / "novel_pep_val.csv", index=False)
    df[df["peptide_split"] == "test"].to_csv(output_dir / "novel_pep_test.csv", index=False)

    # Regime 3: Novel Allele Generalization
    df[df["allele_split"] == "train"].to_csv(output_dir / "novel_allele_train.csv", index=False)
    df[df["allele_split"] == "val"].to_csv(output_dir / "novel_allele_val.csv", index=False)
    df[df["allele_split"] == "test"].to_csv(output_dir / "novel_allele_test.csv", index=False)
    df[df["allele_split"] == "rare_holdout"].to_csv(output_dir / "rare_allele_zero_shot.csv", index=False)

    # Regime 4: 4-Quadrant Double Held-out
    df[df["quadrant_split"] == "train_both_seen"].to_csv(output_dir / "quadrant_train_both_seen.csv", index=False)
    df[df["quadrant_split"] == "test_unseen_pep"].to_csv(output_dir / "quadrant_test_unseen_pep.csv", index=False)
    df[df["quadrant_split"] == "test_unseen_allele"].to_csv(output_dir / "quadrant_test_unseen_allele.csv", index=False)
    df[df["quadrant_split"] == "test_double_unseen"].to_csv(output_dir / "quadrant_test_double_unseen.csv", index=False)

    # Regime 5: Rare Allele Few-shot
    df[df["rare_ablation_mode"] == "rare_3shot_support"].to_csv(output_dir / "rare_3shot_support.csv", index=False)
    df[df["rare_ablation_mode"] == "rare_heldout_query"].to_csv(output_dir / "rare_heldout_query.csv", index=False)

    # =========================================================================
    # SUMMARY METADATA JSON
    # =========================================================================
    def get_set_stats(subset):
        return {
            "n_samples": len(subset),
            "fraction_of_total": round(len(subset) / n_total, 4),
            "unique_alleles": int(subset["allele"].nunique()),
            "unique_peptides": int(subset["peptide"].nunique()),
            "n_binders_1h": int(subset["is_binder_1h"].sum()),
            "pct_binders_1h": round(float(subset["is_binder_1h"].mean() * 100), 2),
            "n_binders_2h": int(subset["is_binder_2h"].sum()),
            "pct_binders_2h": round(float(subset["is_binder_2h"].mean() * 100), 2),
            "n_unbound_zero": int(subset["is_zero"].sum()),
            "pct_unbound_zero": round(float(subset["is_zero"].mean() * 100), 2),
            "mean_thalf": round(float(subset["thalf_hours"].mean()), 2),
            "median_thalf": round(float(subset["thalf_hours"].median()), 2),
        }

    summary_metadata = {
        "dataset": {
            "total_samples": n_total,
            "unique_alleles": int(df["allele"].nunique()),
            "unique_peptides": int(df["peptide"].nunique()),
            "unique_pseudosequences": int(df["hla_pseudoseq"].nunique()),
            "overall_binder_1h_pct": round(float(df["is_binder_1h"].mean() * 100), 2),
            "overall_binder_2h_pct": round(float(df["is_binder_2h"].mean() * 100), 2),
        },
        "allele_breakdown": {
            "train_alleles": sorted(train_alleles),
            "val_alleles": sorted(val_alleles),
            "test_alleles": sorted(test_alleles),
            "ultra_rare_alleles": sorted(ultra_rare_alleles),
        },
        "regime_1_iid_random": {
            "train": get_set_stats(df[df["iid_split"] == "train"]),
            "val": get_set_stats(df[df["iid_split"] == "val"]),
            "test": get_set_stats(df[df["iid_split"] == "test"]),
        },
        "regime_2_novel_peptide": {
            "train": get_set_stats(df[df["peptide_split"] == "train"]),
            "val": get_set_stats(df[df["peptide_split"] == "val"]),
            "test": get_set_stats(df[df["peptide_split"] == "test"]),
            "peptide_leakage_train_test": len(train_peps & test_peps),
        },
        "regime_3_novel_allele": {
            "train": get_set_stats(df[df["allele_split"] == "train"]),
            "val": get_set_stats(df[df["allele_split"] == "val"]),
            "test": get_set_stats(df[df["allele_split"] == "test"]),
            "rare_holdout": get_set_stats(df[df["allele_split"] == "rare_holdout"]),
            "allele_leakage_train_test": len(train_a_set & test_a_set),
        },
        "regime_4_double_heldout_quadrants": {
            "quadrant_1_train_both_seen": get_set_stats(df[df["quadrant_split"] == "train_both_seen"]),
            "quadrant_2_test_unseen_pep": get_set_stats(df[df["quadrant_split"] == "test_unseen_pep"]),
            "quadrant_3_test_unseen_allele": get_set_stats(df[df["quadrant_split"] == "test_unseen_allele"]),
            "quadrant_4_test_double_unseen": get_set_stats(df[df["quadrant_split"] == "test_double_unseen"]),
            "val_unseen_pep": get_set_stats(df[df["quadrant_split"] == "val_unseen_pep"]),
            "val_unseen_allele": get_set_stats(df[df["quadrant_split"] == "val_unseen_allele"]),
            "val_double_unseen": get_set_stats(df[df["quadrant_split"] == "val_double_unseen"]),
        },
        "regime_5_rare_ablation": {
            "rare_zero_shot_total": get_set_stats(df[df["rare_allele_flag"]]),
            "rare_3shot_support": get_set_stats(df[df["rare_ablation_mode"] == "rare_3shot_support"]),
            "rare_heldout_query": get_set_stats(df[df["rare_ablation_mode"] == "rare_heldout_query"]),
        },
    }

    meta_path = output_dir / "split_metadata.json"
    meta_path.write_text(json.dumps(summary_metadata, indent=2))
    print(f"Saved split metadata to {meta_path}")

    # Print summary table
    print("\n" + "=" * 75)
    print("SUMMARY OF SPLIT REGIMES")
    print("=" * 75)
    print(f"{'Split Regime / Subset':<35} | {'Samples':>7} | {'Alleles':>7} | {'Peptides':>8} | {'Binder %':>8}")
    print("-" * 75)
    for name, stats in [
        ("IID Random Train", summary_metadata["regime_1_iid_random"]["train"]),
        ("IID Random Val", summary_metadata["regime_1_iid_random"]["val"]),
        ("IID Random Test", summary_metadata["regime_1_iid_random"]["test"]),
        ("Novel Peptide Train", summary_metadata["regime_2_novel_peptide"]["train"]),
        ("Novel Peptide Val", summary_metadata["regime_2_novel_peptide"]["val"]),
        ("Novel Peptide Test", summary_metadata["regime_2_novel_peptide"]["test"]),
        ("Novel Allele Train", summary_metadata["regime_3_novel_allele"]["train"]),
        ("Novel Allele Val", summary_metadata["regime_3_novel_allele"]["val"]),
        ("Novel Allele Test", summary_metadata["regime_3_novel_allele"]["test"]),
        ("Novel Allele Rare Holdout", summary_metadata["regime_3_novel_allele"]["rare_holdout"]),
        ("Quadrant 1: Train (Seen A + Seen P)", summary_metadata["regime_4_double_heldout_quadrants"]["quadrant_1_train_both_seen"]),
        ("Quadrant 2: Test (Seen A + Unseen P)", summary_metadata["regime_4_double_heldout_quadrants"]["quadrant_2_test_unseen_pep"]),
        ("Quadrant 3: Test (Unseen A + Seen P)", summary_metadata["regime_4_double_heldout_quadrants"]["quadrant_3_test_unseen_allele"]),
        ("Quadrant 4: Test (Double Unseen)", summary_metadata["regime_4_double_heldout_quadrants"]["quadrant_4_test_double_unseen"]),
        ("Rare Alleles: 3-Shot Support", summary_metadata["regime_5_rare_ablation"]["rare_3shot_support"]),
        ("Rare Alleles: Heldout Query", summary_metadata["regime_5_rare_ablation"]["rare_heldout_query"]),
    ]:
        print(f"{name:<35} | {stats['n_samples']:>7,d} | {stats['unique_alleles']:>7d} | {stats['unique_peptides']:>8,d} | {stats['pct_binders_1h']:>7.1f}%")
    print("=" * 75)

    return df, summary_metadata


if __name__ == "__main__":
    create_all_splits()
