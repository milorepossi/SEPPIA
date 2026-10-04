#!/usr/bin/env python3
"""
Publication-Quality Visualization Suite for Rasmussen et al. Dataset.
Generates comprehensive plots covering:
1. HLA Allele distribution and Rarity tiers (01_hla_distribution_and_rarity.png)
2. Peptide frequency distribution and Singletons (02_peptide_distribution_and_rarity.png)
3. Half-life stability and Binding distributions (03_stability_and_binding_distribution.png)
4. Allele-specific Binding Proportions across all 75 alleles (04_allele_binding_proportions.png)
5. Deep-dive into Ultra-Rare Alleles and Rare Peptides (05_rare_entities_deepdive.png)
6. Peptide Anchor Motifs and Sequence Specificity (06_peptide_anchor_motifs.png)
7. 4-Quadrant Generalization Split Matrix (07_split_regimes_matrix.png)
8. Comprehensive ML Evaluation & Ablation Framework (08_ablation_and_holdout_framework.png)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import json

# Set global matplotlib styles
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#e0e0e0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.6


def generate_all_plots(
    splits_csv="splits/dataset_with_splits.csv",
    allele_csv="splits/allele_statistics.csv",
    peptide_csv="splits/peptide_statistics.csv",
    meta_json="splits/split_metadata.json",
    out_dir="plots",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data for visualization...")
    df = pd.read_csv(splits_csv)
    allele_df = pd.read_csv(allele_csv)
    pep_df = pd.read_csv(peptide_csv)
    with open(meta_json) as f:
        meta = json.load(f)

    # Color palettes
    c_blue = "#1f77b4"
    c_orange = "#ff7f0e"
    c_green = "#2ca02c"
    c_red = "#d62728"
    c_purple = "#9467bd"
    c_gray = "#7f7f7f"
    c_cyan = "#17becf"

    # =========================================================================
    # PLOT 1: HLA Distribution and Rarity Tiers
    # =========================================================================
    print("[1/8] Generating Plot 1: HLA Distribution & Rarity...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={"height_ratios": [1.4, 1.0]})

    # Top panel: All 75 alleles sorted by count, spanning top row
    gs = axes[0, 0].get_gridspec()
    for ax in axes[0, :]:
        ax.remove()
    ax_top = fig.add_subplot(gs[0, :])

    # Sort alleles descending
    sorted_alleles = allele_df.sort_values("total_samples", ascending=False).reset_index(drop=True)
    colors = [c_blue if "HLA-A" in a else c_orange for a in sorted_alleles["allele"]]
    
    bars = ax_top.bar(range(len(sorted_alleles)), sorted_alleles["total_samples"], color=colors, width=0.75, edgecolor="none")
    
    # Highlight ultra-rare alleles with red outline / labels and a shaded region
    ax_top.axvspan(67.5, 74.5, color="#f8d7da", alpha=0.5, zorder=0, label="Ultra-Rare Region (N < 50)")
    rare_mask = sorted_alleles["total_samples"] < 50
    stagger_offsets = [
        (68, 220, "HLA-B*13:02 (7)"),
        (69, 360, "HLA-A*69:01 (15)"),
        (70, 500, "HLA-A*68:02 (16)"),
        (71, 640, "HLA-B*40:02 (19)"),
        (72, 780, "HLA-A*02:05 (21)"),
        (73, 920, "HLA-B*35:08 (27)"),
        (74, 1060, "HLA-A*32:01 (32)"),
    ]
    for idx in np.flatnonzero(rare_mask):
        bars[idx].set_edgecolor(c_red)
        bars[idx].set_linewidth(2.0)
        bars[idx].set_color("#f8d7da")
        bars[idx].set_zorder(3)

    # Clean callout box for the ultra-rare cluster
    ax_top.annotate(
        "7 Ultra-Rare Alleles (N < 50, Total=137)\n• HLA-B*13:02 (n=7, 57% bind)\n• HLA-A*69:01 (n=15, 33% bind)\n• HLA-A*68:02 (n=16, 50% bind)\n• HLA-B*40:02 (n=19, 79% bind)\n• HLA-A*02:05 (n=21, 86% bind)\n• HLA-B*35:08 (n=27, 30% bind)\n• HLA-A*32:01 (n=32, 34% bind)",
        xy=(71, 25),
        xytext=(56, 450),
        arrowprops=dict(arrowstyle="->", color=c_red, lw=2.0),
        fontsize=9,
        fontweight="bold",
        color="#721c24",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8d7da", edgecolor=c_red, lw=1.5, alpha=0.95),
        zorder=5
    )

    ax_top.set_xlim(-1, len(sorted_alleles))
    ax_top.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax_top.set_title("Distribution of Sample Counts Across All 75 HLA Alleles (Rarity Tiers Highlighted)", fontsize=14, fontweight="bold", pad=12)
    ax_top.set_xticks(range(0, len(sorted_alleles), 5))
    ax_top.set_xticklabels([sorted_alleles.loc[i, "allele"] for i in range(0, len(sorted_alleles), 5)], rotation=45, ha="right", fontsize=9)
    ax_top.grid(axis="y")
    
    # Legend for top panel
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c_blue, label="HLA-A Alleles (36 alleles, 13,335 samples)"),
        Patch(facecolor=c_orange, label="HLA-B Alleles (39 alleles, 14,831 samples)"),
        Patch(facecolor="#f8d7da", edgecolor=c_red, linewidth=1.5, label="Ultra-Rare Alleles (7 alleles, N < 50 samples)")
    ]
    ax_top.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=10)

    # Bottom Left: Ultra-Rare Alleles Detailed Bar
    ax_bl = axes[1, 0]
    rare_sub = allele_df[allele_df["total_samples"] < 50].sort_values("total_samples", ascending=True)
    y_pos = range(len(rare_sub))
    b_binders = ax_bl.barh(y_pos, rare_sub["binder_1h_count"], color=c_green, alpha=0.85, label="Binders (t1/2 >= 1h)", edgecolor="#1b611b")
    b_non = ax_bl.barh(y_pos, rare_sub["total_samples"] - rare_sub["binder_1h_count"], left=rare_sub["binder_1h_count"], color=c_gray, alpha=0.6, label="Non-binders (t1/2 < 1h)", edgecolor="#444")
    
    for i, (_, row) in enumerate(rare_sub.iterrows()):
        ax_bl.text(row["total_samples"] + 0.8, i, f"Total: {row['total_samples']} ({row['pct_binder_1h']:.0f}% bind)", va="center", fontsize=9, fontweight="bold")

    ax_bl.set_yticks(y_pos)
    ax_bl.set_yticklabels(rare_sub["allele"], fontsize=10, fontweight="bold")
    ax_bl.set_xlabel("Sample Count", fontsize=11, fontweight="bold")
    ax_bl.set_title("Deep Dive: The 7 Ultra-Rare HLA Alleles (N < 50)", fontsize=12, fontweight="bold")
    ax_bl.legend(loc="lower right", fontsize=9)
    ax_bl.set_xlim(0, 45)
    ax_bl.grid(axis="x")

    # Bottom Right: Allele Frequency Histogram & Rarity Gap
    ax_br = axes[1, 1]
    counts = allele_df["total_samples"]
    bins = [0, 50, 150, 300, 450, 600, 800, 1100]
    n_counts, _, patches = ax_br.hist(counts, bins=bins, color=c_blue, edgecolor="black", rwidth=0.85)
    patches[0].set_facecolor(c_red)  # Ultra rare bin
    patches[1].set_facecolor(c_purple) # Moderate rare
    patches[2].set_facecolor(c_blue)
    
    ax_br.annotate(
        "Bimodal Separation:\nNo alleles exist between\nN=32 and N=220!",
        xy=(100, 4), xytext=(120, 12),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.5)
    )

    ax_br.set_xlabel("Number of Samples per Allele", fontsize=11, fontweight="bold")
    ax_br.set_ylabel("Number of HLA Alleles", fontsize=11, fontweight="bold")
    ax_br.set_title("Allele Sample Count Bimodality & The 'Rarity Gap'", fontsize=12, fontweight="bold")
    ax_br.grid(axis="y")
    
    rarity_legend = [
        Patch(facecolor=c_red, label="Ultra-Rare (N < 50): 7 alleles (137 samples)"),
        Patch(facecolor=c_purple, label="Moderately-Rare (50-300): 2 alleles (492 samples)"),
        Patch(facecolor=c_blue, label="Common (N >= 300): 66 alleles (27,537 samples)")
    ]
    ax_br.legend(handles=rarity_legend, loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "01_hla_distribution_and_rarity.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # PLOT 2: Peptide Distribution, Singletons & Promiscuity
    # =========================================================================
    print("[2/8] Generating Plot 2: Peptide Distribution & Rarity...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel A: Peptide Frequency Distribution
    ax_a = axes[0, 0]
    freq_series = pep_df["total_tests"].value_counts().sort_index()
    x_freq = freq_series.index
    y_freq = freq_series.values
    bar_cols = [c_red if x == 1 else (c_purple if x <= 3 else c_blue) for x in x_freq]
    
    ax_a.bar(x_freq, y_freq, color=bar_cols, edgecolor="none", width=0.8)
    ax_a.set_xlabel("Number of HLA Tests per Peptide", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Number of Unique Peptides", fontsize=11, fontweight="bold")
    ax_a.set_title("Peptide Occurrence Frequency (Singletons in Red)", fontsize=12, fontweight="bold")
    ax_a.set_yscale("log")
    ax_a.grid(axis="y")
    
    ax_a.annotate(
        f"Singletons (Tested 1x):\n{y_freq[0]:,} peptides ({y_freq[0]/len(pep_df)*100:.1f}%)",
        xy=(1, y_freq[0]), xytext=(4, y_freq[0]*0.7),
        arrowprops=dict(arrowstyle="->", color=c_red, lw=1.5),
        fontsize=10, fontweight="bold", color=c_red
    )

    # Panel B: Cumulative Distribution Function (CDF)
    ax_b = axes[0, 1]
    sorted_tests = np.sort(pep_df["total_tests"].values)
    cdf = np.arange(1, len(sorted_tests) + 1) / len(sorted_tests) * 100
    ax_b.plot(sorted_tests, cdf, color=c_blue, lw=2.5)
    ax_b.set_xlabel("Number of Tests per Peptide", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Cumulative % of Peptides", fontsize=11, fontweight="bold")
    ax_b.set_title("Cumulative Distribution of Peptide Observations", fontsize=12, fontweight="bold")
    ax_b.grid(True)
    
    # Mark 50% and 90%
    med_val = np.median(sorted_tests)
    p90_val = np.percentile(sorted_tests, 90)
    ax_b.axhline(50, color=c_orange, linestyle=":", lw=1.5)
    ax_b.axhline(90, color=c_green, linestyle=":", lw=1.5)
    ax_b.annotate(f"Median = {med_val:.0f} tests", xy=(med_val, 50), xytext=(med_val + 3, 45), fontsize=10, fontweight="bold", color=c_orange)
    ax_b.annotate(f"90th Percentile = {p90_val:.0f} tests", xy=(p90_val, 90), xytext=(p90_val - 12, 85), fontsize=10, fontweight="bold", color=c_green)

    # Panel C: Peptide Promiscuity (Universal Binder, Selective, Non-binder)
    ax_c = axes[1, 0]
    promisc_counts = pep_df["binding_profile"].value_counts()
    colors_promisc = [c_blue, c_green, c_gray]
    wedges, texts, autotexts = ax_c.pie(
        promisc_counts.values,
        labels=promisc_counts.index,
        autopct="%1.1f%%",
        pctdistance=0.72,
        startangle=140,
        colors=colors_promisc,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10, fontweight="bold")
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_color("white")
        at.set_weight("bold")
    for t in texts:
        t.set_fontsize(9.5)
    ax_c.set_title("Peptide Binding Promiscuity Profiles\nAcross Evaluated HLAs", fontsize=12, fontweight="bold")

    # Panel D: Top Promiscuous Peptides Table / Bar
    ax_d = axes[1, 1]
    top_pep = pep_df.sort_values(by=["binder_count", "total_tests"], ascending=[False, False]).head(10)
    y_p = range(len(top_pep))
    ax_d.barh(y_p, top_pep["binder_count"], color=c_green, label="Alleles Bound (t1/2 >= 1h)")
    ax_d.barh(y_p, top_pep["total_tests"] - top_pep["binder_count"], left=top_pep["binder_count"], color=c_gray, alpha=0.5, label="Alleles Unbound")
    
    for i, (_, row) in enumerate(top_pep.iterrows()):
        ax_d.text(row["total_tests"] + 0.5, i, f"{row['binder_count']}/{row['total_tests']} ({row['binder_ratio']*100:.0f}%)", va="center", fontsize=9, fontweight="bold")

    ax_d.set_yticks(y_p)
    ax_d.set_yticklabels(top_pep["peptide"], fontfamily="monospace", fontsize=10, fontweight="bold")
    ax_d.set_xlabel("Number of Alleles Tested", fontsize=11, fontweight="bold")
    ax_d.set_title("Top 10 Most Promiscuous Binders\n(Cross-Allele Binders)", fontsize=12, fontweight="bold")
    ax_d.set_xlim(0, 45)
    ax_d.legend(loc="lower right", fontsize=9)
    ax_d.grid(axis="x")

    plt.tight_layout()
    fig.savefig(out_dir / "02_peptide_distribution_and_rarity.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # PLOT 3: Half-Life Stability & Binding Distribution
    # =========================================================================
    print("[3/8] Generating Plot 3: Stability & Binding Distribution...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel A: Raw Half-life Histogram
    ax_a = axes[0, 0]
    thalf = df["thalf_hours"].values
    ax_a.hist(thalf, bins=50, color=c_blue, edgecolor="black", alpha=0.75)
    ax_a.axvline(1.0, color=c_red, linestyle="--", lw=2, label="Standard Binder Threshold (1.0h)")
    ax_a.axvline(2.0, color=c_orange, linestyle=":", lw=2, label="Stringent Binder Threshold (2.0h)")
    ax_a.set_xlabel("Half-life $t_{1/2}$ (hours)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax_a.set_title("Raw Half-Life Distribution (Max: 256.7h)", fontsize=12, fontweight="bold")
    ax_a.set_yscale("log")
    ax_a.grid(axis="y")
    ax_a.legend(loc="upper right", fontsize=9)

    # Panel B: Log-scaled Density Plot
    ax_b = axes[0, 1]
    # Log10(t + 0.01) to cleanly display 0h
    log_thalf = np.log10(np.where(thalf == 0, 0.001, thalf))
    ax_b.hist(log_thalf, bins=50, color=c_purple, edgecolor="black", alpha=0.75, density=True)
    ax_b.axvline(np.log10(1.0), color=c_red, linestyle="--", lw=2, label="1.0h cutoff ($10^0$)")
    ax_b.axvline(np.log10(2.0), color=c_orange, linestyle=":", lw=2, label="2.0h cutoff ($10^{0.3}$)")
    
    ax_b.annotate(
        "Zero-Spike (20.2%)\nUnbound background\n($t_{1/2} = 0$)",
        xy=(-3, 0.5), xytext=(-2.5, 0.7),
        arrowprops=dict(arrowstyle="->", color=c_red, lw=1.5),
        fontsize=10, fontweight="bold", color=c_red
    )
    
    ax_b.set_xlabel(r"$\log_{10}(t_{1/2})$ [zeros mapped to -3]", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax_b.set_title("Log-Scaled Stability Distribution (Bimodal Regime)", fontsize=12, fontweight="bold")
    ax_b.grid(axis="y")
    ax_b.legend(loc="upper right", fontsize=9)

    # Panel C: 5-Tier Stability Breakdown Bar
    ax_c = axes[1, 0]
    tier_order = ["Unbound_Zero", "Weak_Sub1h", "Moderate_1to2h", "Strong_2to5h", "VeryStrong_5hPlus"]
    tier_labels = ["Unbound\n(0h)", "Weak\n(<1h)", "Moderate\n(1-2h)", "Strong\n(2-5h)", "Very Strong\n(>=5h)"]
    tier_vals = [sum(df["stability_tier"] == t) for t in tier_order]
    tier_pcts = [v / len(df) * 100 for v in tier_vals]
    tier_cols = [c_gray, "#aec7e8", "#ffbb78", c_orange, c_green]
    
    bars_c = ax_c.bar(range(5), tier_vals, color=tier_cols, edgecolor="black", width=0.65)
    for i, bar in enumerate(bars_c):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150, f"{tier_vals[i]:,}\n({tier_pcts[i]:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax_c.set_xticks(range(5))
    ax_c.set_xticklabels(tier_labels, fontsize=10, fontweight="bold")
    ax_c.set_ylabel("Interaction Pairs", fontsize=11, fontweight="bold")
    ax_c.set_title("Classification of Interactions into 5 Stability Tiers", fontsize=12, fontweight="bold")
    ax_c.set_ylim(0, max(tier_vals) * 1.25)
    ax_c.grid(axis="y")

    # Panel D: Balanced Binary Classification (>= 1.0h vs < 1.0h)
    ax_d = axes[1, 1]
    bind_counts = [sum(~df["is_binder_1h"]), sum(df["is_binder_1h"])]
    labels_bind = [f"Non-binders (<1.0h)\n{bind_counts[0]:,} ({bind_counts[0]/len(df)*100:.1f}%)", 
                   f"Binders (>=1.0h)\n{bind_counts[1]:,} ({bind_counts[1]/len(df)*100:.1f}%)"]
    
    wedges, texts, autotexts = ax_d.pie(
        bind_counts,
        labels=labels_bind,
        autopct="%1.1f%%",
        pctdistance=0.72,
        startangle=90,
        colors=[c_gray, c_green],
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10, fontweight="bold")
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_color("white")
        at.set_weight("bold")
    ax_d.set_title("Standard Binary Classification Split (t1/2 >= 1.0h)\nNear-Perfect 52% / 48% Balance", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "03_stability_and_binding_distribution.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # PLOT 4: Allele-Specific Binding Proportions (All 75 Alleles)
    # =========================================================================
    print("[4/8] Generating Plot 4: Allele-Specific Binding Proportions...")
    fig, ax = plt.subplots(figsize=(14, 22))

    # Calculate proportions for each of the 75 alleles
    allele_prop = df.groupby("allele").agg(
        total=("thalf_hours", "count"),
        unbound=("is_zero", lambda x: sum(x)),
        weak=("thalf_hours", lambda x: sum((x > 0) & (x < 1.0))),
        moderate=("thalf_hours", lambda x: sum((x >= 1.0) & (x < 2.0))),
        strong=("thalf_hours", lambda x: sum(x >= 2.0))
    )
    allele_prop["pct_unbound"] = allele_prop["unbound"] / allele_prop["total"] * 100
    allele_prop["pct_weak"] = allele_prop["weak"] / allele_prop["total"] * 100
    allele_prop["pct_moderate"] = allele_prop["moderate"] / allele_prop["total"] * 100
    allele_prop["pct_strong"] = allele_prop["strong"] / allele_prop["total"] * 100
    allele_prop["pct_binders"] = (allele_prop["moderate"] + allele_prop["strong"]) / allele_prop["total"] * 100

    # Sort ascending by pct_binders
    allele_prop = allele_prop.sort_values("pct_binders", ascending=True)

    y_pos = np.arange(len(allele_prop))
    p1 = ax.barh(y_pos, allele_prop["pct_unbound"], color="#7f7f7f", edgecolor="none", height=0.75, label="Unbound (0h)")
    p2 = ax.barh(y_pos, allele_prop["pct_weak"], left=allele_prop["pct_unbound"], color="#aec7e8", edgecolor="none", height=0.75, label="Weak (<1h)")
    p3 = ax.barh(y_pos, allele_prop["pct_moderate"], left=allele_prop["pct_unbound"] + allele_prop["pct_weak"], color="#ffbb78", edgecolor="none", height=0.75, label="Moderate (1-2h)")
    p4 = ax.barh(y_pos, allele_prop["pct_strong"], left=allele_prop["pct_unbound"] + allele_prop["pct_weak"] + allele_prop["pct_moderate"], color="#2ca02c", edgecolor="none", height=0.75, label="Strong (>=2h)")

    ax.set_yticks(y_pos)
    # Color allele label in red if rare
    labels = []
    for a in allele_prop.index:
        n = allele_prop.loc[a, "total"]
        if n < 50:
            labels.append(f"{a} [RARE, N={n}]")
        else:
            labels.append(f"{a} (N={n})")
    ax.set_yticklabels(labels, fontsize=8)
    for tick_label, a in zip(ax.get_yticklabels(), allele_prop.index):
        if allele_prop.loc[a, "total"] < 50:
            tick_label.set_color("red")
            tick_label.set_weight("bold")

    ax.axvline(50, color="black", linestyle="--", lw=1.2, alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of Interactions (%)", fontsize=11, fontweight="bold")
    ax.set_title("Allele Stability Profiles across All 75 HLA Allotypes\n(Extreme variation from 3.2% binders in B*14:01 to 92.6% in A*02:11; Ultra-rare in Red)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02), fontsize=10, frameon=True)
    ax.grid(axis="x")

    plt.tight_layout()
    fig.savefig(out_dir / "04_allele_binding_proportions.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # PLOT 5: Rare Entities Deep Dive (Ultra-rare HLAs & Rare Peptides)
    # =========================================================================
    print("[5/8] Generating Plot 5: Rare Entities Deep-Dive...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel A: Ultra-rare HLAs binding distribution
    ax_a = axes[0, 0]
    rare_df = df[df["rare_allele_flag"]].copy()
    rare_counts = rare_df.groupby(["allele", "is_binder_1h"]).size().unstack(fill_value=0)
    rare_counts = rare_counts.loc[rare_sub["allele"]]
    
    x_idx = np.arange(len(rare_counts))
    w = 0.35
    ax_a.bar(x_idx - w/2, rare_counts[False], width=w, color=c_gray, edgecolor="black", label="Non-binders (<1h)")
    ax_a.bar(x_idx + w/2, rare_counts[True], width=w, color=c_green, edgecolor="black", label="Binders (>=1h)")
    
    for i, a in enumerate(rare_counts.index):
        tot = rare_counts.loc[a].sum()
        bind = rare_counts.loc[a, True]
        ax_a.text(i, max(rare_counts.loc[a]) + 0.8, f"{bind}/{tot}\n({bind/tot*100:.0f}%)", ha="center", fontsize=8, fontweight="bold")

    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(rare_counts.index, rotation=35, ha="right", fontsize=9, fontweight="bold")
    ax_a.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax_a.set_title("Ultra-Rare Alleles (N < 50): Binder vs Non-Binder Split", fontsize=12, fontweight="bold")
    ax_a.set_ylim(0, 24)
    ax_a.legend(loc="upper left", fontsize=9)
    ax_a.grid(axis="y")

    # Panel B: Half-Life Comparison (Rare vs Common Alleles)
    ax_b = axes[0, 1]
    rare_thalf = df[df["rare_allele_flag"]]["thalf_hours"].values
    common_thalf = df[~df["rare_allele_flag"]]["thalf_hours"].values
    
    # Log values for comparison
    log_rare = np.log10(np.where(rare_thalf == 0, 0.001, rare_thalf))
    log_common = np.log10(np.where(common_thalf == 0, 0.001, common_thalf))
    
    box_data = [log_common, log_rare]
    bp = ax_b.boxplot(box_data, patch_artist=True, tick_labels=["Common Alleles\n(N=28,029)", "Ultra-Rare Alleles\n(N=137)"], widths=0.5)
    bp["boxes"][0].set_facecolor(c_blue)
    bp["boxes"][1].set_facecolor(c_red)
    for median in bp["medians"]:
        median.set(color="yellow", linewidth=2.5)
        
    ax_b.set_ylabel(r"$\log_{10}(t_{1/2})$", fontsize=11, fontweight="bold")
    ax_b.set_title("Stability Distribution: Common vs Ultra-Rare Alleles\n(Similar Medians: 1.1h vs 0.9h)", fontsize=12, fontweight="bold")
    ax_b.grid(axis="y")

    # Panel C: Peptide Frequency Tiers vs Binder Fraction
    ax_c = axes[1, 0]
    p_tier_agg = pep_df.groupby("freq_tier").agg(
        count=("peptide", "count"),
        mean_binder_ratio=("binder_ratio", "mean")
    )
    tier_names = ["Singletons\n(N=1)", "Rare\n(N=2-3)", "Moderate\n(N=4-8)", "Frequent\n(N>=9)"]
    x_c = np.arange(len(p_tier_agg))
    
    ax_c.plot(x_c, p_tier_agg["mean_binder_ratio"] * 100, marker="o", markersize=10, color=c_purple, lw=3)
    for i, txt in enumerate(p_tier_agg["mean_binder_ratio"] * 100):
        ax_c.annotate(f"{txt:.1f}% binders\n({p_tier_agg['count'].iloc[i]:,} peps)", xy=(i, txt), xytext=(i, txt + 3.5), ha="center", fontsize=9, fontweight="bold")

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(tier_names, fontsize=10, fontweight="bold")
    ax_c.set_ylabel("Average Binder Ratio (%)", fontsize=11, fontweight="bold")
    ax_c.set_title("Binder Percentage Across Peptide Frequency Tiers", fontsize=12, fontweight="bold")
    ax_c.set_ylim(40, 75)
    ax_c.grid(True)

    # Panel D: Overlap of Peptides Bound by Rare Alleles with Common Alleles
    ax_d = axes[1, 1]
    rare_peptides = set(df[df["rare_allele_flag"]]["peptide"])
    common_peptides = set(df[~df["rare_allele_flag"]]["peptide"])
    shared = len(rare_peptides & common_peptides)
    private_rare = len(rare_peptides - common_peptides)
    
    ax_d.pie(
        [shared, private_rare],
        labels=[f"Shared with Common HLAs\n({shared} peptides, {shared/len(rare_peptides)*100:.1f}%)", 
                f"Private to Rare HLAs\n({private_rare} peptides, {private_rare/len(rare_peptides)*100:.1f}%)"],
        autopct="%1.1f%%",
        startangle=140,
        colors=[c_blue, c_orange],
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10, fontweight="bold")
    )
    ax_d.set_title("Epitope Sharing: Peptides in Rare Alleles\nvs Peptides in Common Alleles", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "05_rare_entities_deepdive.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # PLOT 6: Peptide Anchor Motifs (P2 and P9)
    # =========================================================================
    print("[6/8] Generating Plot 6: Anchor Motifs & Sequence Specificity...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    binders_pep = df[df["is_binder_1h"]]["peptide"]
    non_binders_pep = df[~df["is_binder_1h"]]["peptide"]
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")

    # Pos 2 (P2)
    p2_b = pd.Series([p[1] for p in binders_pep]).value_counts(normalize=True).reindex(aa_order, fill_value=0) * 100
    p2_nb = pd.Series([p[1] for p in non_binders_pep]).value_counts(normalize=True).reindex(aa_order, fill_value=0) * 100
    diff_p2 = p2_b - p2_nb

    # Pos 9 (P9)
    p9_b = pd.Series([p[8] for p in binders_pep]).value_counts(normalize=True).reindex(aa_order, fill_value=0) * 100
    p9_nb = pd.Series([p[8] for p in non_binders_pep]).value_counts(normalize=True).reindex(aa_order, fill_value=0) * 100
    diff_p9 = p9_b - p9_nb

    # Panel A: P2 Comparison
    ax_a = axes[0, 0]
    x = np.arange(len(aa_order))
    w = 0.38
    ax_a.bar(x - w/2, p2_b, width=w, color=c_green, label="Binders (t1/2 >= 1h)", edgecolor="none")
    ax_a.bar(x + w/2, p2_nb, width=w, color=c_gray, label="Non-binders (t1/2 < 1h)", edgecolor="none")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(aa_order, fontweight="bold")
    ax_a.set_ylabel("Frequency (%)", fontsize=11, fontweight="bold")
    ax_a.set_title("Anchor Position 2 (P2, Pocket B) Amino Acid Frequency", fontsize=12, fontweight="bold")
    ax_a.legend(loc="upper right", fontsize=9)
    ax_a.grid(axis="y")

    # Panel B: P2 Enrichment Delta
    ax_b = axes[0, 1]
    cols_p2 = [c_green if v > 0 else c_red for v in diff_p2]
    ax_b.bar(x, diff_p2, color=cols_p2, edgecolor="black", width=0.6)
    ax_b.axhline(0, color="black", lw=1)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(aa_order, fontweight="bold")
    ax_b.set_ylabel(r"Enrichment $\Delta$ (%) [Binder - NonBinder]", fontsize=11, fontweight="bold")
    ax_b.set_title("P2 Enrichment in Binders (Leu, Tyr, Thr Favored)", fontsize=12, fontweight="bold")
    ax_b.grid(axis="y")

    # Panel C: P9 Comparison
    ax_c = axes[1, 0]
    ax_c.bar(x - w/2, p9_b, width=w, color=c_green, label="Binders (t1/2 >= 1h)", edgecolor="none")
    ax_c.bar(x + w/2, p9_nb, width=w, color=c_gray, label="Non-binders (t1/2 < 1h)", edgecolor="none")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(aa_order, fontweight="bold")
    ax_c.set_ylabel("Frequency (%)", fontsize=11, fontweight="bold")
    ax_c.set_title("Anchor Position 9 (P9, Pocket F, C-term) Amino Acid Frequency", fontsize=12, fontweight="bold")
    ax_c.legend(loc="upper right", fontsize=9)
    ax_c.grid(axis="y")

    # Panel D: P9 Enrichment Delta
    ax_d = axes[1, 1]
    cols_p9 = [c_green if v > 0 else c_red for v in diff_p9]
    ax_d.bar(x, diff_p9, color=cols_p9, edgecolor="black", width=0.6)
    ax_d.axhline(0, color="black", lw=1)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(aa_order, fontweight="bold")
    ax_d.set_ylabel(r"Enrichment $\Delta$ (%) [Binder - NonBinder]", fontsize=11, fontweight="bold")
    ax_d.set_title("P9 Enrichment in Binders (Val, Trp, Arg Favored vs Leu/Tyr)", fontsize=12, fontweight="bold")
    ax_d.grid(axis="y")

    plt.tight_layout()
    fig.savefig(out_dir / "06_peptide_anchor_motifs.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # PLOT 7: 4-Quadrant Generalization Split Matrix
    # =========================================================================
    print("[7/8] Generating Plot 7: 4-Quadrant Split Matrix...")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")

    # Draw a 2x2 grid diagram with annotations
    grid_rects = [
        # (x, y, w, h, title, subtitle, color, text)
        (0.08, 0.52, 0.40, 0.38, "QUADRANT 1: TRAIN (CORE)", 
         "Seen Allele x Seen Peptide", "#d4edda", 
         f"• Samples: 14,238 ({14238/28166*100:.1f}%)\n• Unique Alleles: 49\n• Unique Peptides: 3,572\n• Binder Rate: 53.2%\n• Core training corpus for foundation models\n  (e.g., ESM-C, Boltz-2, MLP heads)"),
        
        (0.52, 0.52, 0.40, 0.38, "QUADRANT 2: HELD-OUT PEPTIDES", 
         "Seen Allele x Unseen Peptide", "#cce5ff", 
         f"• Samples: 2,990 ({2990/28166*100:.1f}%)\n• Unique Alleles: 49\n• Unique Peptides: 767 (0% in Train)\n• Binder Rate: 52.1%\n• Evaluates Novel Epitope / Antigen generalization\n  (e.g., neoepitope discovery for known HLA allotypes)"),
        
        (0.08, 0.08, 0.40, 0.38, "QUADRANT 3: HELD-OUT HLA ALLELES", 
         "Unseen Allele x Seen Peptide", "#fff3cd", 
         f"• Samples: 2,669 ({2669/28166*100:.1f}%)\n• Unique Alleles: 9 (0% in Train)\n• Unique Peptides: 2,013\n• Binder Rate: 50.6%\n• Evaluates Pan-Allele / Allotype transfer\n  (predicting stability for uncharacterized HLA alleles)"),
        
        (0.52, 0.08, 0.40, 0.38, "QUADRANT 4: STRICT DOUBLE HELD-OUT", 
         "Unseen Allele x Unseen Peptide", "#f8d7da", 
         f"• Samples: 555 ({555/28166*100:.1f}%)\n• Unique Alleles: 9 (0% in Train)\n• Unique Peptides: 408 (0% in Train)\n• Binder Rate: 50.1%\n• True Biophysical Generalization benchmark\n  (Neither partner seen during model training!)")
    ]

    for (gx, gy, gw, gh, title, sub, col, desc) in grid_rects:
        rect = plt.Rectangle((gx, gy), gw, gh, facecolor=col, edgecolor="#333", linewidth=2, transform=ax.transAxes, zorder=1)
        ax.add_patch(rect)
        ax.text(gx + 0.02, gy + gh - 0.04, title, fontsize=12, fontweight="bold", transform=ax.transAxes)
        ax.text(gx + 0.02, gy + gh - 0.08, sub, fontsize=10, fontstyle="italic", color="#555", transform=ax.transAxes)
        ax.text(gx + 0.02, gy + 0.03, desc, fontsize=9.5, linespacing=1.6, transform=ax.transAxes)

    # Add axis banners
    ax.text(0.28, 0.94, "Seen Peptides (Train Pool: 3,572 peptides)", fontsize=13, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.72, 0.94, "Unseen Peptides (Test Pool: 767 peptides)", fontsize=13, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.02, 0.71, "Seen HLA\nAlleles\n(49 alleles)", fontsize=12, fontweight="bold", ha="center", va="center", rotation=90, transform=ax.transAxes)
    ax.text(0.02, 0.27, "Unseen HLA\nAlleles\n(9 alleles)", fontsize=12, fontweight="bold", ha="center", va="center", rotation=90, transform=ax.transAxes)

    ax.text(0.5, 0.98, "The 4-Quadrant Generalization Matrix for Peptide-HLA Stability", fontsize=15, fontweight="bold", ha="center", transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(out_dir / "07_split_regimes_matrix.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # PLOT 8: Comprehensive ML Evaluation & Ablation Framework
    # =========================================================================
    print("[8/8] Generating Plot 8: ML Ablation & Evaluation Framework...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left Panel: Generalization Hierarchy Comparison
    ax_l = axes[0]
    regimes = [
        "1. IID Random\n(In-Distribution)",
        "2. Novel Peptide\n(Antigen Holdout)",
        "3. Novel Allele\n(Pan-Allele Transfer)",
        "4. Double Unseen\n(Strict Generalization)",
        "5. Rare Allele\nZero-Shot (N < 50)"
    ]
    test_sizes = [4225, 4170, 3773, 555, 137]
    binder_pcts = [51.9, 50.8, 50.6, 50.1, 50.4]
    
    y_r = np.arange(len(regimes))
    bars = ax_l.barh(y_r, test_sizes, color=[c_blue, c_cyan, c_orange, c_red, c_purple], edgecolor="black", height=0.6)
    for i, b in enumerate(bars):
        ax_l.text(b.get_width() + 60, i, f"N={test_sizes[i]:,} | {binder_pcts[i]:.1f}% binders", va="center", fontsize=9.5, fontweight="bold")

    ax_l.set_yticks(y_r)
    ax_l.set_yticklabels(regimes, fontsize=10, fontweight="bold")
    ax_l.set_xlabel("Test Set Sample Count", fontsize=11, fontweight="bold")
    ax_l.set_title("Hierarchy of Evaluation Regimes & Benchmark Set Sizes", fontsize=12, fontweight="bold")
    ax_l.set_xlim(0, 5200)
    ax_l.grid(axis="x")

    # Right Panel: Expected Performance Profile across Regimes (Conceptual AUROC/Spearman)
    ax_r = axes[1]
    # Illustrative benchmarks showing expected decay from IID to Double Unseen
    x_pos = np.arange(len(regimes))
    expected_baseline_nn = [0.88, 0.76, 0.68, 0.58, 0.55]
    expected_foundation_esm = [0.91, 0.84, 0.79, 0.71, 0.69]
    
    ax_r.plot(x_pos, expected_baseline_nn, marker="o", color=c_gray, lw=2.5, markersize=8, label="Standard Neural Net Baseline (NetMHC-like)")
    ax_r.plot(x_pos, expected_foundation_esm, marker="s", color=c_blue, lw=2.5, markersize=8, label="Protein Foundation Model (e.g. ESMC / Boltz-2)")
    
    for i in range(len(regimes)):
        ax_r.text(i, expected_foundation_esm[i] + 0.02, f"{expected_foundation_esm[i]:.2f}", ha="center", fontsize=9, fontweight="bold", color=c_blue)
        ax_r.text(i, expected_baseline_nn[i] - 0.03, f"{expected_baseline_nn[i]:.2f}", ha="center", fontsize=9, fontweight="bold", color=c_gray)

    ax_r.set_xticks(x_pos)
    ax_r.set_xticklabels(["IID", "Novel\nPeptide", "Novel\nAllele", "Double\nUnseen", "Rare\nAllele"], fontsize=10, fontweight="bold")
    ax_r.set_ylabel("Expected Test AUROC / Generalization", fontsize=11, fontweight="bold")
    ax_r.set_title("Hypothesis: Foundation Model Generalization Advantage\n(Greatest delta on Novel Alleles & Double Unseen)", fontsize=12, fontweight="bold")
    ax_r.set_ylim(0.45, 1.0)
    ax_r.grid(True)
    ax_r.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    fig.savefig(out_dir / "08_ablation_and_holdout_framework.png", dpi=300)
    plt.close(fig)

    print("\nAll 8 plots successfully generated in plots/ directory!")


if __name__ == "__main__":
    generate_all_plots()
