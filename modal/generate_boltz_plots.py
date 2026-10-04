"""Generate predicted vs true scatter and error decomposition on Modal for Boltz-2 arms."""
import io
import json
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pandas", "scikit-learn", "matplotlib")
    .add_local_file(REPO / "train_mlp.py", "/app/train_mlp.py")
    .add_local_file(REPO / "analyze_errors.py", "/app/analyze_errors.py")
    .add_local_file(REPO / "scripts" / "arm_features.py", "/app/arm_features.py")
    .add_local_file(REPO / "scripts" / "extract_embeddings.py", "/app/extract_embeddings.py")
    .add_local_file(REPO / "scripts" / "build_combined.py", "/app/build_combined.py")
    .add_local_file(REPO / "DATA" / "rasmussen_clean.csv", "/app/DATA/rasmussen_clean.csv")
    .add_local_dir(REPO / "DATA", "/app/DATA", ignore=["*.xlsx", "split", "*.json"])
)

app = modal.App("boltz-plots-generator", image=image)
cache = modal.Volume.from_name("boltz-pmhc-cache", create_if_missing=True)


@app.function(image=image, volumes={"/cache": cache}, timeout=3600, cpu=8.0, memory=65536)
def run_arm_and_analyze(arm: str, pooling: str, cache_dir: str = "/cache/boltz_full") -> dict:
    import sys
    sys.path.insert(0, "/app")
    import train_mlp
    import analyze_errors
    import numpy as np
    import io
    import pandas as pd

    splits_dir = "/app/DATA"
    clean_target_csv = "/app/DATA/rasmussen_clean.csv"
    epsilon = 0.1
    seed = 0
    splits = 5

    per_split = []
    for index in range(splits):
        print(f"[{arm}_{pooling}] Training split {index}...", flush=True)
        result, model = train_mlp.train_one_split(
            index, splits_dir, epsilon=epsilon, seed=seed + index,
            epochs=300, patience=25, batch_size=256, learning_rate=1e-3,
            weight_decay=1e-5, dropout=0.2, hidden=(256, 128),
            validation_fraction=0.1, device="cpu",
            arm=arm, embeddings_dir=cache_dir, pooling=pooling,
            standardise="auto", prescale=True,
            clean_target_csv=clean_target_csv,
        )
        predicted = result['predictions']
        truth = result['y_test']
        thalf_hours = result['thalf_hours']
        error = predicted - truth
        predicted_half_life = np.maximum(np.exp(predicted.astype(np.float64)) - epsilon, 0.0)

        per_split.append(dict(
            split_index=index,
            epsilon=float(epsilon),
            truth=truth,
            predicted=predicted,
            rmse=float(np.sqrt(np.mean(error**2))),
            bias=float(np.mean(error)),
            over_predicted=float(np.mean(error > 0)),
            strata=analyze_errors.stratify(thalf_hours, error),
            confusion=[analyze_errors.confusion(thalf_hours, predicted_half_life, t)
                       for t in analyze_errors.THRESHOLDS],
            half_life=thalf_hours,
            error=error,
        ))

    # Generate scatter figure
    scatter_path = f"/tmp/predicted_vs_true_{arm}_{pooling.replace(':', '')}.png"
    analyze_errors.scatter_figure(per_split, scatter_path, epsilon)
    with open(scatter_path, "rb") as f:
        scatter_bytes = f.read()

    # Generate error decomposition figure
    decomp_path = f"/tmp/error_decomposition_{arm}_{pooling.replace(':', '')}.png"
    analyze_errors.figure(per_split, decomp_path)
    with open(decomp_path, "rb") as f:
        decomp_bytes = f.read()

    # Calculate strata and confusion tables
    labels, share, share_sd, _ = analyze_errors.aggregate(per_split, 'strata', 'share_of_squared_error', 'stratum')
    _, rows_share, _, _ = analyze_errors.aggregate(per_split, 'strata', 'share_of_rows', 'stratum')
    _, rmse, rmse_sd, _ = analyze_errors.aggregate(per_split, 'strata', 'rmse', 'stratum')
    _, bias, bias_sd, _ = analyze_errors.aggregate(per_split, 'strata', 'bias', 'stratum')
    table = pd.DataFrame(dict(stratum=labels, rows_pct=rows_share*100,
                              error_pct=share*100, error_pct_sd=share_sd*100,
                              rmse=rmse, rmse_sd=rmse_sd, bias=bias, bias_sd=bias_sd))

    rows = []
    for field in ('false_positive', 'false_negative', 'false_positive_rate',
                  'false_negative_rate', 'precision', 'recall', 'positive_rows'):
        thresholds, mean, sd, _ = analyze_errors.aggregate(per_split, 'confusion', field, 'threshold')
        rows.append(dict(metric=field, **{f'> {t:g} h': m for t, m in zip(thresholds, mean)}))

    report = dict(
        strata=table.to_dict(orient='records'),
        thresholds=[{k: v for k, v in c.items()} for c in per_split[0]['confusion']],
        per_split=[{k: v for k, v in s.items() if k not in ('half_life', 'error', 'truth', 'predicted')} for s in per_split],
        confusion_table=rows,
        over_predicted=float(np.mean([s['over_predicted'] for s in per_split])),
    )

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        truth=np.concatenate([s['truth'] for s in per_split]),
        predicted=np.concatenate([s['predicted'] for s in per_split]),
        half_life=np.concatenate([s['half_life'] for s in per_split]),
        split_indices=np.concatenate([np.full(len(s['truth']), s['split_index'], dtype=np.int32) for s in per_split])
    )
    npz_bytes = buf.getvalue()

    return {
        "arm": arm,
        "pooling": pooling,
        "scatter_bytes": scatter_bytes,
        "decomp_bytes": decomp_bytes,
        "npz_bytes": npz_bytes,
        "report": report,
    }


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    out_dir = REPO / "RESULTS" / "boltz"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [("BZZU", "pca:20"), ("BZZU", "flatten")]
    print(f"Launching parallel training and error analysis for {tasks} on Modal...")
    results = list(run_arm_and_analyze.starmap(tasks))

    for res in results:
        arm = res["arm"]
        pool_clean = res["pooling"].replace(":", "")
        prefix = f"{arm}_{pool_clean}"
        
        scatter_file = out_dir / f"predicted_vs_true_{prefix}.png"
        scatter_file.write_bytes(res["scatter_bytes"])
        print(f"Wrote {scatter_file} ({len(res['scatter_bytes'])} bytes)")

        decomp_file = out_dir / f"error_decomposition_{prefix}.png"
        decomp_file.write_bytes(res["decomp_bytes"])
        print(f"Wrote {decomp_file} ({len(res['decomp_bytes'])} bytes)")

        pred_file = out_dir / f"{prefix}_test_predictions.npz"
        pred_file.write_bytes(res["npz_bytes"])
        print(f"Wrote {pred_file} ({len(res['npz_bytes'])} bytes)")

        json_file = out_dir / f"error_decomposition_{prefix}.json"
        json_file.write_text(json.dumps(res["report"], indent=2) + "\n")
        print(f"Wrote {json_file}")
