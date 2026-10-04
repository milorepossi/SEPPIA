"""Train and evaluate Baseline A0 (One-Hot MLP) on Paul's 5 splits with comprehensive metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import train_mlp


def compute_comprehensive_metrics(y_true_raw, y_pred_raw, y_true_log, y_pred_log):
    sp_rho, _ = stats.spearmanr(y_true_raw, y_pred_raw)
    pr_r_log, _ = stats.pearsonr(y_true_log, y_pred_log)
    rmse_raw = float(np.sqrt(np.mean((y_true_raw - y_pred_raw) ** 2)))
    mae_raw = float(np.mean(np.abs(y_true_raw - y_pred_raw)))
    rmse_log = float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))

    # Classification metrics
    b1_true = (y_true_raw >= 1.0).astype(int)
    b2_true = (y_true_raw >= 2.0).astype(int)

    def get_aucs(y_true_bin, scores):
        if len(np.unique(y_true_bin)) < 2:
            return 0.5, 0.5
        fpr, tpr, _ = roc_curve(y_true_bin, scores)
        roc = float(auc(fpr, tpr))
        pr = float(average_precision_score(y_true_bin, scores))
        return roc, pr

    auc_roc_1h, pr_auc_1h = get_aucs(b1_true, y_pred_log)
    auc_roc_2h, pr_auc_2h = get_aucs(b2_true, y_pred_log)

    return {
        "spearman_rho": float(sp_rho),
        "pearson_r_log": float(pr_r_log),
        "rmse_raw": rmse_raw,
        "mae_raw": mae_raw,
        "rmse_log": rmse_log,
        "auc_roc_1h": auc_roc_1h,
        "pr_auc_1h": pr_auc_1h,
        "auc_roc_2h": auc_roc_2h,
        "pr_auc_2h": pr_auc_2h,
    }


def main():
    splits = ["iid", "h2", "aw_blosum", "novel_pep", "novel_allele"]
    all_metrics = {}

    for s in splits:
        print(f"\n--- Training Baseline A0 on split: {s.upper()} ---")
        res, model = train_mlp.train_one_split(
            s,
            "paul/splits_npz",
            epsilon=0.1,
            seed=42,
            epochs=30,
            patience=8,
            batch_size=256,
            learning_rate=1e-3,
            weight_decay=1e-5,
            dropout=0.2,
            hidden=(256, 128),
            validation_fraction=0.1,
            device="mps",
            arm="onehot",
        )

        y_true_raw = res["thalf_hours"]
        y_true_log = res["y_test"]
        y_pred_log = res["predictions"]
        y_pred_raw = np.maximum(0.0, np.exp(y_pred_log) - 0.1)

        m = compute_comprehensive_metrics(y_true_raw, y_pred_raw, y_true_log, y_pred_log)
        m["best_epoch"] = int(res["best_epoch"])
        m["n_train"] = int(res["train_rows"])
        m["n_test"] = int(res["test_rows"])
        all_metrics[s] = m

        print(f"[{s.upper()}] Spearman: {m['spearman_rho']:.4f} | Pearson: {m['pearson_r_log']:.4f} | RMSE raw: {m['rmse_raw']:.2f}h | AUROC 1h: {m['auc_roc_1h']:.4f}")

    out_file = Path("RESULTS/onehot_paul_splits.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved comprehensive Baseline A0 metrics to: {out_file}")


if __name__ == "__main__":
    main()
