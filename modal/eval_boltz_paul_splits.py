"""Evaluate Boltz-2 and Baseline models on Paul's new splits using Modal."""
import json
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pandas", "scikit-learn", "matplotlib", "scipy")
    .add_local_file(REPO / "train_mlp.py", "/app/train_mlp.py")
    .add_local_file(REPO / "scripts" / "arm_features.py", "/app/arm_features.py")
    .add_local_file(REPO / "scripts" / "extract_embeddings.py", "/app/extract_embeddings.py")
    .add_local_file(REPO / "scripts" / "build_combined.py", "/app/build_combined.py")
    .add_local_file(REPO / "DATA" / "rasmussen_clean.csv", "/app/DATA/rasmussen_clean.csv")
    .add_local_dir(REPO / "paul" / "splits_npz", "/app/paul/splits_npz")
)

app = modal.App("boltz-paul-splits-eval", image=image)
cache = modal.Volume.from_name("boltz-pmhc-cache", create_if_missing=True)


@app.function(image=image, volumes={"/cache": cache}, timeout=3600, cpu=8.0, memory=65536)
def evaluate_arm_on_splits(arm: str, pooling: str, cache_dir: str = "/cache/boltz_full") -> dict:
    import sys
    sys.path.insert(0, "/app")
    import train_mlp

    splits = ["iid", "h2", "aw_blosum", "novel_pep", "novel_allele"]
    results = {}
    for s in splits:
        print(f"[{arm}_{pooling}] Training split {s}...", flush=True)
        res, model = train_mlp.train_one_split(
            s, "/app/paul/splits_npz", epsilon=0.1, seed=42,
            epochs=300, patience=25, batch_size=256, learning_rate=1e-3,
            weight_decay=1e-5, dropout=0.2, hidden=(256, 128),
            validation_fraction=0.1, device="cpu",
            arm=arm, embeddings_dir=cache_dir, pooling=pooling,
            clean_target_csv="/app/DATA/rasmussen_clean.csv"
        )
        t = res["test"]
        results[s] = {
            "spearman": float(t["spearman"]),
            "pearson": float(t["pearson"]),
            "rmse": float(t["rmse"]),
            "mae": float(t["mae"]),
            "best_epoch": int(res["best_epoch"]),
        }
        print(f"[{arm}_{pooling}] {s}: Spearman={t['spearman']:.4f}, Pearson={t['pearson']:.4f}, RMSE={t['rmse']:.4f}", flush=True)
    return results


@app.local_entrypoint()
def main(arm: str = "BZZU", pooling: str = "pca:20"):
    print(f"Launching Modal evaluation for arm={arm}, pooling={pooling}...")
    res = evaluate_arm_on_splits.remote(arm=arm, pooling=pooling)
    print(f"\nFinal Results for {arm} ({pooling}):")
    print(json.dumps(res, indent=2))
    out_path = REPO / "RESULTS" / f"{arm}_{pooling.replace(':', '')}_paul_splits.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved to: {out_path}")
