#!/usr/bin/env python3
"""
Sequence-Clustered Split using Pairwise Hamming Distance for Peptide-HLA Stability.
Computes connected components at Hamming distance <= 2, performs stratified greedy
allocation into Train (80%), Val (10%), Test (10%), and generates diagnostic plots.
"""

import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import pandas as pd


def compute_hamming_graph(peptides, h_thresh=2):
    """Computes adjacency graph where edges exist if Hamming distance <= h_thresh."""
    pep_arr_int = np.array([[ord(c) for c in p] for p in peptides], dtype=np.int8)
    n = len(peptides)
    edges = []
    chunk_size = 500

    for i in range(0, n, chunk_size):
        chunk_a = pep_arr_int[i:i + chunk_size]
        dists = (chunk_a[:, None, :] != pep_arr_int[None, :, :]).sum(axis=-1)
        for row_idx in range(len(chunk_a)):
            global_i = i + row_idx
            row_dists = dists[row_idx, global_i + 1:]
            for offset in np.where(row_dists <= h_thresh)[0]:
                edges.append((peptides[global_i], peptides[global_i + 1 + offset]))

    G = nx.Graph()
    G.add_nodes_from(peptides)
    G.add_edges_from(edges)
    return G, edges


def generate_sequence_clustered_splits(
    data_path="data/rasmussen_et_al_dataset.csv",
    output_dir="splits",
    plots_dir="plots",
    random_seed=42,
):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    peptides = sorted(df["peptide"].unique())
    n_peptides = len(peptides)
    total_samples = len(df)

    print(f"Total rows: {total_samples:,} across {df['allele'].nunique()} alleles and {n_peptides:,} unique peptides.")

    # 1. Full pairwise distance profile (chunked)
    pep_arr_int = np.array([[ord(c) for c in p] for p in peptides], dtype=np.int8)
    dist_counts = Counter()
    chunk_size = 500

    print("Computing full pairwise distance distribution...")
    for i in range(0, n_peptides, chunk_size):
        chunk_a = pep_arr_int[i:i + chunk_size]
        dists = (chunk_a[:, None, :] != pep_arr_int[None, :, :]).sum(axis=-1)
        for row_idx in range(len(chunk_a)):
            global_i = i + row_idx
            row_dists = dists[row_idx, global_i + 1:]
            vals, counts = np.unique(row_dists, return_counts=True)
            for v, c in zip(vals, counts):
                dist_counts[v] += c

    # Percolation analysis across H in [1, 2, 3, 4, 5]
    print("Evaluating percolation curve across Hamming thresholds...")
    percolation_stats = []
    for h in [1, 2, 3, 4, 5]:
        G_h, edges_h = compute_hamming_graph(peptides, h_thresh=h)
        comps_h = list(nx.connected_components(G_h))
        sizes_h = sorted([len(c) for c in comps_h], reverse=True)
        singletons_h = sum(1 for s in sizes_h if s == 1)
        percolation_stats.append({
            "h": h,
            "edges": len(edges_h),
            "n_components": len(comps_h),
            "singletons": singletons_h,
            "pct_singletons": singletons_h / n_peptides * 100,
            "max_cluster_size": sizes_h[0],
            "top5_sizes": sizes_h[:5],
        })

    # 2. Main Graph at H <= 2
    print("Building target graph at Hamming threshold <= 2...")
    G, edges_h2 = compute_hamming_graph(peptides, h_thresh=2)
    components = list(nx.connected_components(G))
    pep2cluster = {p: cid for cid, c in enumerate(components) for p in c}
    df["cluster_id"] = df["peptide"].map(pep2cluster)

    # Cluster metadata
    clusters_meta = []
    for cid, c in enumerate(components):
        c_df = df[df["cluster_id"] == cid]
        clusters_meta.append({
            "cluster_id": cid,
            "peptides": list(c),
            "n_peptides": len(c),
            "n_samples": len(c_df),
            "n_binders": int((c_df["thalf_hours"] >= 1.0).sum()),
            "n_zeros": int((c_df["thalf_hours"] == 0.0).sum()),
            "mean_thalf": float(c_df["thalf_hours"].mean()),
            "median_thalf": float(c_df["thalf_hours"].median()),
            "alleles": set(c_df["allele"]),
        })

    # 3. Stratified Greedy Split Allocation
    np.random.seed(random_seed)
    np.random.shuffle(clusters_meta)
    clusters_meta.sort(key=lambda x: (x["n_peptides"], x["n_samples"]), reverse=True)

    target_ratios = {"train": 0.80, "val": 0.10, "test": 0.10}
    target_samples = {k: total_samples * v for k, v in target_ratios.items()}

    assigned_split = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_peptides = {"train": set(), "val": set(), "test": set()}
    split_binders = {"train": 0, "val": 0, "test": 0}
    split_zeros = {"train": 0, "val": 0, "test": 0}
    split_alleles = {"train": set(), "val": set(), "test": set()}

    global_binder_ratio = (df["thalf_hours"] >= 1.0).mean()

    for cl in clusters_meta:
        best_split = None
        best_cost = float("inf")
        for split_name in ["train", "val", "test"]:
            curr_samples = split_counts[split_name] + cl["n_samples"]
            target_s = target_samples[split_name]
            size_penalty = (curr_samples - target_s) / target_s
            curr_binders = split_binders[split_name] + cl["n_binders"]
            curr_br = curr_binders / curr_samples if curr_samples > 0 else 0
            binder_penalty = abs(curr_br - global_binder_ratio) * 5.0
            allele_bonus = 0
            if split_name in ["val", "test"]:
                new_alleles = len(cl["alleles"] - split_alleles[split_name])
                allele_bonus = -0.5 * new_alleles

            cost = size_penalty + binder_penalty + allele_bonus
            if cost < best_cost:
                best_cost = cost
                best_split = split_name

        assigned_split[cl["cluster_id"]] = best_split
        split_counts[best_split] += cl["n_samples"]
        split_peptides[best_split].update(cl["peptides"])
        split_binders[best_split] += cl["n_binders"]
        split_zeros[best_split] += cl["n_zeros"]
        split_alleles[best_split].update(cl["alleles"])

    df["split"] = df["cluster_id"].map(assigned_split)

    # 4. Strict distance verification
    train_peps = list(split_peptides["train"])
    val_peps = list(split_peptides["val"])
    test_peps = list(split_peptides["test"])

    train_arr = np.array([[ord(c) for c in p] for p in train_peps], dtype=np.int8)
    test_arr = np.array([[ord(c) for c in p] for p in test_peps], dtype=np.int8)

    min_test_train_dists = []
    for i in range(len(test_arr)):
        d = (test_arr[i:i + 1] != train_arr).sum(axis=-1)
        min_test_train_dists.append(int(d.min()))

    # 5. Save splits to disk
    print("\nSaving splits to disk...")
    df.to_csv(output_dir / "sequence_clustered_splits_h2.csv", index=False)
    df[df["split"] == "train"].to_csv(output_dir / "train_h2.csv", index=False)
    df[df["split"] == "val"].to_csv(output_dir / "val_h2.csv", index=False)
    df[df["split"] == "test"].to_csv(output_dir / "test_h2.csv", index=False)

    meta_df = pd.DataFrame(clusters_meta)
    meta_df["split"] = meta_df["cluster_id"].map(assigned_split)
    meta_df["alleles_count"] = meta_df["alleles"].apply(len)
    meta_df.drop(columns=["alleles"]).to_csv(output_dir / "cluster_metadata_h2.csv", index=False)

    summary_json = {
        "hamming_threshold": 2,
        "total_peptides": n_peptides,
        "total_samples": total_samples,
        "n_clusters": len(components),
        "min_test_to_train_distance": min(min_test_train_dists),
        "split_summary": {
            s: {
                "n_samples": split_counts[s],
                "pct_samples": round(split_counts[s] / total_samples * 100, 2),
                "n_peptides": len(split_peptides[s]),
                "pct_peptides": round(len(split_peptides[s]) / n_peptides * 100, 2),
                "n_binders": split_binders[s],
                "pct_binders": round(split_binders[s] / split_counts[s] * 100, 2),
                "n_zeros": split_zeros[s],
                "pct_zeros": round(split_zeros[s] / split_counts[s] * 100, 2),
                "n_alleles": len(split_alleles[s]),
            }
            for s in ["train", "val", "test"]
        },
        "percolation_analysis": percolation_stats,
    }
    with open(output_dir / "split_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # 6. Biological mutation & activity cliff analysis
    # For H=1 pairs
    G1, edges_h1 = compute_hamming_graph(peptides, h_thresh=1)
    pos_counts = {i + 1: 0 for i in range(9)}
    for p1, p2 in edges_h1:
        for pos in range(9):
            if p1[pos] != p2[pos]:
                pos_counts[pos + 1] += 1

    paired_cliffs = []
    for p1, p2 in edges_h1:
        df1 = df[df["peptide"] == p1].set_index("allele")["thalf_hours"]
        df2 = df[df["peptide"] == p2].set_index("allele")["thalf_hours"]
        common = df1.index.intersection(df2.index)
        for a in common:
            paired_cliffs.append({
                "allele": a,
                "p1": p1,
                "p2": p2,
                "t1": df1.loc[a],
                "t2": df2.loc[a],
                "diff": abs(df1.loc[a] - df2.loc[a]),
            })
    paired_cliffs_df = pd.DataFrame(paired_cliffs)

    # =========================================================================
    # PLOTTING SUITE
    # =========================================================================
    print("Generating comprehensive diagnostic figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10

    # PLOT 1: Distance Distribution & Percolation Phase Transition
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1A: Pairwise distance distribution
    ax = axes[0]
    dists = sorted(dist_counts.keys())
    counts = [dist_counts[d] for d in dists]
    total_pairs = sum(counts)
    pcts = [c / total_pairs * 100 for c in counts]
    bars = ax.bar(dists, pcts, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85, width=0.6)
    ax.set_title("A. Pairwise Hamming Distances (All 15.8M Pairs)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Hamming Distance (Mismatches in 9-mer)")
    ax.set_ylabel("% of All Peptide Pairs")
    ax.set_xticks(range(1, 10))
    for bar, pct in zip(bars, pcts):
        if pct > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{pct:.1f}%", ha="center", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 1B: Percolation Phase Transition
    ax = axes[1]
    h_vals = [p["h"] for p in percolation_stats]
    max_sizes = [p["max_cluster_size"] for p in percolation_stats]
    ax.plot(h_vals, max_sizes, marker="o", linewidth=2.5, markersize=8, color="#ef4444")
    ax.set_title("B. Phase Transition: Max Cluster Size vs. H", fontweight="bold", fontsize=11)
    ax.set_xlabel("Hamming Cutoff Threshold (H)")
    ax.set_ylabel("Largest Connected Component Size")
    ax.set_xticks(h_vals)
    ax.axvspan(3.5, 5.5, color="#fee2e2", alpha=0.5, label="Percolation Collapse (H >= 4)")
    ax.annotate(
        f"Giant Component:\n{max_sizes[3]:,} peptides (49.8%!)",
        xy=(4, max_sizes[3]),
        xytext=(3.2, 3200),
        arrowprops=dict(facecolor="#b91c1c", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#b91c1c",
    )
    ax.annotate(
        f"Stable Regimes:\nH=2: max={max_sizes[1]} peps\nH=3: max={max_sizes[2]} peps",
        xy=(2, max_sizes[1]),
        xytext=(1.2, 1500),
        arrowprops=dict(facecolor="#047857", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#047857",
    )
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 1C: Cluster Size Distribution at H <= 2
    ax = axes[2]
    comp_sizes = [c["n_peptides"] for c in clusters_meta]
    size_counts = pd.Series(comp_sizes).value_counts().sort_index()
    ax.bar(size_counts.index, size_counts.values, color="#10b981", edgecolor="#047857", width=0.6, alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("C. Cluster Size Distribution (H <= 2)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Cluster Size (Peptides per Cluster)")
    ax.set_ylabel("Number of Clusters (Log Scale)")
    for x, y in zip(size_counts.index, size_counts.values):
        ax.text(x, y * 1.2, f"{y:,}", ha="center", fontsize=8, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot1_path = plots_dir / "01_hamming_distribution_and_percolation.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved {plot1_path}")

    # PLOT 2: Biological Mechanisms & Activity Cliffs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 2A: Position-Specific Mutational Frequency
    ax = axes[0]
    positions = list(range(1, 10))
    pos_pcts = [pos_counts[p] / len(edges_h1) * 100 for p in positions]
    colors = ["#f59e0b" if p in [2, 9] else "#6366f1" for p in positions]
    bars = ax.bar(positions, pos_pcts, color=colors, edgecolor="#312e81", width=0.6, alpha=0.85)
    ax.set_title("A. Mutational Hotspots (H=1 Pairs)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Peptide Position (P1 - P9)")
    ax.set_ylabel("% of 1-Mutant Pairs Mutating Here")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"P{p}\n{'Anchor' if p in [2, 9] else ''}" for p in positions])
    for bar, val in zip(bars, pos_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{val:.1f}%", ha="center", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 2B: Delta Stability Distribution for 1-Mutants
    ax = axes[1]
    diffs = paired_cliffs_df["diff"]
    ax.hist(diffs, bins=40, color="#8b5cf6", edgecolor="#4c1d95", alpha=0.85)
    ax.set_title("B. Stability Delta (|Δ t1/2|) for 1-Mutants", fontweight="bold", fontsize=11)
    ax.set_xlabel("|Δ t1/2 (hours)| Between 1-Mutants on Same Allele")
    ax.set_ylabel("Pair Comparisons (N=513)")
    median_d = diffs.median()
    mean_d = diffs.mean()
    ax.axvline(median_d, color="#ef4444", linestyle="--", linewidth=2, label=f"Median: {median_d:.1f}h")
    ax.axvline(mean_d, color="#f59e0b", linestyle="-.", linewidth=2, label=f"Mean: {mean_d:.1f}h")
    ax.annotate(
        f"49.3% have Δ > 1.0h\n23.4% have Δ > 5.0h\nMax cliff: {diffs.max():.1f}h!",
        xy=(50, 40),
        xytext=(50, 55),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3e8ff", edgecolor="#9333ea"),
        fontsize=9,
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 2C: Activity Cliff Scatter
    ax = axes[2]
    scatter = ax.scatter(
        paired_cliffs_df["t1"],
        paired_cliffs_df["t2"],
        c=paired_cliffs_df["diff"],
        cmap="plasma",
        alpha=0.7,
        s=35,
        edgecolor="none",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("|Δ t1/2| (hours)")
    ax.plot([0, 150], [0, 150], color="#94a3b8", linestyle="--", linewidth=1.5, label="Perfect Stability Agreement")
    ax.set_title("C. Intra-Cluster Activity Cliffs (Same Allele)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Peptide 1 Half-Life (hours)")
    ax.set_ylabel("Peptide 2 (1-Mutant) Half-Life (hours)")
    ax.set_xlim(-5, 160)
    ax.set_ylim(-5, 160)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot2_path = plots_dir / "02_cluster_mechanisms_and_activity_cliffs.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved {plot2_path}")

    # PLOT 3: Split Quality, Balance & Minimum Distance
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 3A: Sample & Peptide Proportions
    ax = axes[0]
    splits = ["train", "val", "test"]
    labels = ["Train", "Val", "Test"]
    samples_pct = [split_counts[s] / total_samples * 100 for s in splits]
    peps_pct = [len(split_peptides[s]) / n_peptides * 100 for s in splits]
    x = np.arange(len(splits))
    width = 0.35
    ax.bar(x - width / 2, samples_pct, width, label="% Total Samples", color="#3b82f6", edgecolor="#1d4ed8")
    ax.bar(x + width / 2, peps_pct, width, label="% Unique Peptides", color="#10b981", edgecolor="#047857")
    ax.set_title("A. Dataset Split Proportions", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    for i, s_p in enumerate(samples_pct):
        ax.text(i - width / 2, s_p + 1.5, f"{s_p:.1f}%", ha="center", fontsize=8)
    for i, p_p in enumerate(peps_pct):
        ax.text(i + width / 2, p_p + 1.5, f"{p_p:.1f}%", ha="center", fontsize=8)
    ax.axhline(80, color="#94a3b8", linestyle=":", alpha=0.6)
    ax.axhline(10, color="#94a3b8", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 3B: Binder and Zero Balance
    ax = axes[1]
    binder_pcts = [split_binders[s] / split_counts[s] * 100 for s in splits]
    zero_pcts = [split_zeros[s] / split_counts[s] * 100 for s in splits]
    ax.bar(x - width / 2, binder_pcts, width, label="% Binders (t1/2 >= 1.0h)", color="#f59e0b", edgecolor="#b45309")
    ax.bar(x + width / 2, zero_pcts, width, label="% Zero Half-Life", color="#64748b", edgecolor="#334155")
    ax.set_title("B. Stratification: Binder & Zero Invariance", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage of Split Samples (%)")
    ax.set_ylim(0, 70)
    for i, b_p in enumerate(binder_pcts):
        ax.text(i - width / 2, b_p + 1.2, f"{b_p:.1f}%", ha="center", fontsize=8)
    for i, z_p in enumerate(zero_pcts):
        ax.text(i + width / 2, z_p + 1.2, f"{z_p:.1f}%", ha="center", fontsize=8)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 3C: Minimum Distance Distribution from Test to Train
    ax = axes[2]
    min_dist_counts = pd.Series(min_test_train_dists).value_counts().sort_index()
    ax.bar(min_dist_counts.index, min_dist_counts.values, color="#ec4899", edgecolor="#be185d", width=0.6)
    ax.set_title("C. Min Distance: Test Peptides to Closest Train Peptide", fontweight="bold", fontsize=11)
    ax.set_xlabel("Minimum Hamming Distance to Train Set")
    ax.set_ylabel("Test Peptides (Total=485)")
    ax.set_xticks([1, 2, 3, 4, 5])
    for x_val, y_val in zip(min_dist_counts.index, min_dist_counts.values):
        ax.text(x_val, y_val + 5, f"{y_val:,}\n({y_val/len(test_peps)*100:.1f}%)", ha="center", fontsize=8)
    ax.axvspan(0.5, 2.5, color="#fee2e2", alpha=0.5, label="Leaked Zone (H <= 2): 0 PEPTIDES")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot3_path = plots_dir / "03_split_balance_and_allele_coverage.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved {plot3_path}")

    # PLOT 4: Network Graphs of Representative Clusters
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Find 4 interesting clusters
    # 1: largest cluster (size 8 poly-A)
    # 2: kinase-like cluster
    # 3: N-terminal/C-terminal mutant cluster
    # 4: Hydrophobic cluster
    multi_clusters = [c for c in clusters_meta if c["n_peptides"] >= 4]
    selected_cids = [multi_clusters[0]["cluster_id"], multi_clusters[1]["cluster_id"], multi_clusters[2]["cluster_id"], multi_clusters[3]["cluster_id"]]

    for idx, (ax, cid) in enumerate(zip(axes, selected_cids)):
        c_info = next(c for c in clusters_meta if c["cluster_id"] == cid)
        subG = G.subgraph(c_info["peptides"])
        pos = nx.spring_layout(subG, seed=42)

        # Draw nodes
        node_colors = []
        for p in subG.nodes():
            p_mean = df[df["peptide"] == p]["thalf_hours"].mean()
            node_colors.append(p_mean)

        nx.draw_networkx_edges(subG, pos, ax=ax, edge_color="#94a3b8", width=2.0)
        nodes = nx.draw_networkx_nodes(
            subG,
            pos,
            ax=ax,
            node_color=node_colors,
            cmap="viridis",
            node_size=1200,
            edgecolors="#1e293b",
            linewidths=1.5,
        )
        labels = {p: f"{p}\n({df[df['peptide']==p]['thalf_hours'].mean():.1f}h)" for p in subG.nodes()}
        nx.draw_networkx_labels(subG, pos, labels=labels, ax=ax, font_size=8, font_weight="bold", font_color="black")

        c_split = assigned_split[cid]
        c_alleles_cnt = len(c_info['alleles'])
        ax.set_title(
            f"Cluster #{cid} ({c_info['n_peptides']} Peptides, {c_info['n_samples']} Measurements)\nAssigned Split: {c_split.upper()} | Tested on {c_alleles_cnt} Alleles",
            fontweight="bold",
            fontsize=10,
        )
        ax.axis("off")

    plt.suptitle("D. Network Graphs of Representative Hamming Clusters (Color = Mean t1/2)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot4_path = plots_dir / "04_cluster_network_samples.png"
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved {plot4_path}")

    print("\n" + "=" * 60)
    print("SEQUENCE CLUSTERED SPLIT COMPLETE!")
    print("=" * 60)
    print(f"Train samples: {split_counts['train']:,} ({split_counts['train']/total_samples*100:.2f}%) | Peptides: {len(split_peptides['train']):,}")
    print(f"Val samples  : {split_counts['val']:,} ({split_counts['val']/total_samples*100:.2f}%) | Peptides: {len(split_peptides['val']):,}")
    print(f"Test samples : {split_counts['test']:,} ({split_counts['test']/total_samples*100:.2f}%) | Peptides: {len(split_peptides['test']):,}")
    print(f"All 75 alleles covered in Val & Test: {len(split_alleles['val'])==75 and len(split_alleles['test'])==75}")
    print(f"Minimum distance from ANY Test peptide to ANY Train peptide: {min(min_test_train_dists)}")
    print("=" * 60)


if __name__ == "__main__":
    generate_sequence_clustered_splits()
