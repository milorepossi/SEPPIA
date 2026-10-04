#!/usr/bin/env python3
"""Cross-pooling and cross-arm analysis for Boltz-2 vs baselines."""
import json
import pathlib
import sys

def main():
    boltz_dir = pathlib.Path("RESULTS/boltz")
    records = []
    
    for p in sorted(boltz_dir.glob("*.json")):
        if p.name.startswith("ladder_"):
            continue
        stem = p.stem
        if "_" not in stem:
            continue
        arm, pooling = stem.rsplit("_", 1)
        if pooling == "pca20":
            pooling = "pca:20"
            
        data = json.loads(p.read_text())
        summ = data.get("summary", {})
        encoding = data.get("encoding", {})
        
        records.append({
            "arm": arm,
            "pooling": pooling,
            "features": encoding.get("features", 0),
            "spearman_mean": summ.get("spearman", {}).get("mean", 0.0),
            "spearman_sd": summ.get("spearman", {}).get("sd", 0.0),
            "spearman_vals": summ.get("spearman", {}).get("values", []),
            "pearson_mean": summ.get("pearson", {}).get("mean", 0.0),
            "rmse_mean": summ.get("rmse", {}).get("mean", 0.0),
            "mae_mean": summ.get("mae", {}).get("mean", 0.0),
            "unseen_hla_rmse": summ.get("rmse_unseen_hla", {}).get("mean", 0.0),
            "unseen_pep_rmse": summ.get("rmse_unseen_peptide", {}).get("mean", 0.0),
            "seen_both_rmse": summ.get("rmse_seen_both", {}).get("mean", 0.0),
            "pca_variance": (data.get("arm") or {}).get("explained_variance")
        })

    # Find onehot reference for each pooling (or fallback to onehot general)
    onehot_refs = {r["pooling"]: r["spearman_vals"] for r in records if r["arm"] == "onehot"}
    default_ref = onehot_refs.get("pca:20") or onehot_refs.get("flatten")

    print("=" * 115)
    print(f"{'Arm':10s} | {'Pooling':8s} | {'Feats':6s} | {'Spearman':17s} | {'Pearson':9s} | {'RMSE':8s} | {'Unseen HLA':11s} | {'Unseen Pep':11s} | {'Delta vs A0':11s}")
    print("=" * 115)
    
    # Sort by pooling then spearman descending
    pool_order = ["pca:20", "flatten", "mean"]
    records.sort(key=lambda r: (pool_order.index(r["pooling"]) if r["pooling"] in pool_order else 99, -r["spearman_mean"]))
    
    curr_pool = None
    for r in records:
        if r["pooling"] != curr_pool:
            if curr_pool is not None:
                print("-" * 115)
            curr_pool = r["pooling"]
            
        ref_vals = onehot_refs.get(r["pooling"], default_ref)
        if ref_vals and len(ref_vals) == len(r["spearman_vals"]) and r["arm"] != "onehot":
            deltas = [v - ref for v, ref in zip(r["spearman_vals"], ref_vals)]
            mean_delta = sum(deltas) / len(deltas)
            wins = sum(1 for d in deltas if d > 0)
            delta_str = f"{mean_delta:+.4f} ({wins}/5)"
        elif r["arm"] == "onehot":
            delta_str = "reference"
        else:
            delta_str = "—"
            
        print(f"{r['arm']:10s} | {r['pooling']:8s} | {r['features']:6d} | {r['spearman_mean']:.4f} +/- {r['spearman_sd']:.4f} | {r['pearson_mean']:.4f}   | {r['rmse_mean']:.4f}   | {r['unseen_hla_rmse']:.4f}     | {r['unseen_pep_rmse']:.4f}     | {delta_str:11s}")
    print("=" * 115)

if __name__ == "__main__":
    main()
