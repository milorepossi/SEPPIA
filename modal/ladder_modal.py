"""Run the arm ladder on Modal CPU, against the extracted Boltz-2 cache.

Training runs here rather than locally for two reasons: the 3.5 GB cache lives
on the Modal volume, and this laptop has no torch installed at all.

The head is tiny, so this is CPU work. No GPU is requested and none is needed —
the whole point of the frozen-embedding design is that the expensive part was
the one-off extraction.

What gets scored
----------------
Every arm on the same five committed splits, with the same head and the same
hyperparameter budget, so only the input representation differs.

  onehot  A0, the incumbent to beat at 0.771 Spearman
  BZS     Boltz-2 trunk single representation
  BZZ     interface pair contraction, 1/d^2 -- the arm carrying the hypothesis
  BZZU    same contraction, uniform weights -- is the weighting earning its keep?
  BZP     per-token pLDDT -- zero-training, does confidence predict stability?
  BZSZ    BZS and BZZ concatenated -- closest to PreFold-dG's combination

Two poolings per arm, because `pca:20` alone is not trustworthy: across the
ESM-2 ladder Spearman tracked the variance each arm retained at r = 0.94, so
the ranking may measure compression rather than representation. `flatten`
is the capacity-unconstrained check. Both are reported; disagreement between
them is itself the result.

  modal run modal/ladder_modal.py::ladder
"""
from __future__ import annotations

import json
import pathlib

import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pandas", "scikit-learn", "matplotlib")
    .add_local_file(REPO / "train_mlp.py", "/app/train_mlp.py")
    .add_local_file(REPO / "scripts" / "arm_features.py", "/app/arm_features.py")
    .add_local_file(REPO / "scripts" / "extract_embeddings.py", "/app/extract_embeddings.py")
    .add_local_file(REPO / "scripts" / "build_combined.py", "/app/build_combined.py")
    .add_local_file(REPO / "DATA" / "rasmussen_clean.csv", "/app/DATA/rasmussen_clean.csv")
    .add_local_dir(REPO / "DATA", "/app/DATA", ignore=["*.xlsx", "split", "*.json"])
)

app = modal.App("boltz-pmhc-ladder", image=image)
cache = modal.Volume.from_name("boltz-pmhc-cache", create_if_missing=True)


@app.function(image=image, volumes={"/cache": cache}, timeout=1800,
              cpu=4.0, memory=32768)
def build_combined(cache_dir: str) -> str:
    """Concatenate boltz_S and boltz_Z into the BZSZ arm. CPU only."""
    import sys
    sys.path.insert(0, "/app")
    import build_combined as bc

    summary = bc.build(cache_dir)
    summary["verification"] = bc.verify(cache_dir)
    cache.commit()
    return json.dumps(summary)


@app.function(image=image, volumes={"/cache": cache}, timeout=7200,
              cpu=8.0, memory=65536)
def score_arm(arm: str, pooling: str, cache_dir: str, prescale: bool,
              clean_target: bool, seed: int) -> str:
    """Train one arm on all five splits and return its metrics."""
    import sys
    import traceback
    sys.path.insert(0, "/app")
    import train_mlp

    out_dir = f"/cache/RESULTS/{arm}_{pooling.replace(':', '')}"
    try:
        report = train_mlp.run(
            splits_dir="/app/DATA", output_dir=out_dir, arm=arm,
            embeddings_dir=cache_dir, pooling=pooling, prescale=prescale,
            clean_target_csv="/app/DATA/rasmussen_clean.csv" if clean_target else None,
            seed=seed, save_models=False)
    except Exception as exc:
        return json.dumps({"arm": arm, "pooling": pooling,
                           "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()[-2500:]})
    finally:
        cache.commit()

    per_split = report.get("splits", report.get("results", []))
    return json.dumps({"arm": arm, "pooling": pooling, "report": report,
                       "n_splits": len(per_split) if hasattr(per_split, "__len__") else None})


def _spearman(report: dict):
    """Pull mean/sd test Spearman out of whatever shape the report uses."""
    table = report.get("table") or report.get("summary") or {}
    for key in ("spearman", "test_spearman"):
        if key in table:
            entry = table[key]
            if isinstance(entry, dict):
                return entry.get("mean"), entry.get("sd")
    rows = report.get("splits") or report.get("results") or []
    values = [r["test"]["spearman"] for r in rows
              if isinstance(r, dict) and r.get("test", {}).get("spearman") is not None]
    if not values:
        return None, None
    import statistics
    return (statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0)


@app.function(image=image, volumes={"/cache": cache}, timeout=10800, cpu=1.0)
def ladder_all(cache_dir: str, arms: list[str], poolings: list[str],
               prescale: bool, clean_target: bool, seed: int,
               skip_combined: bool) -> str:
    """Orchestrate the whole ladder **server-side**, fanning out with .map.

    This has to be a remote function rather than a local entrypoint. A local
    entrypoint holds the app open with a heartbeat, so when the client
    disconnects -- which a long tool call does -- Modal stops the app mid-run
    with `App state is APP_STATE_STOPPED`. Orchestrating from inside means the
    run survives the client going away, and every arm's metrics.json is written
    to the volume as it finishes, so results are recoverable even if this
    function is itself interrupted.
    """
    summary = {"cache_dir": cache_dir, "arms": arms, "poolings": poolings,
               "prescale": prescale, "clean_target": clean_target, "seed": seed}
    if not skip_combined:
        summary["combined"] = json.loads(build_combined.local(cache_dir))

    jobs = [(a, p, cache_dir, prescale, clean_target, seed)
            for p in poolings for a in arms]
    results = {}
    for (a, p, *_), raw in zip(jobs, score_arm.map(*zip(*jobs))):
        results[f"{a}|{p}"] = json.loads(raw)
    summary["results"] = results

    out = pathlib.Path("/cache/RESULTS/ladder_boltz.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    cache.commit()
    return json.dumps(summary)


@app.local_entrypoint()
def ladder(cache_dir: str = "/cache/boltz_full",
           arms: str = "onehot,BZP,BZZ,BZZU,BZS,BZSZ",
           poolings: str = "pca:20,flatten",
           prescale: bool = True, clean_target: bool = True, seed: int = 0,
           skip_combined: bool = False,
           out: str = "RESULTS/ladder_boltz.json"):
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    pool_list = [p.strip() for p in poolings.split(",") if p.strip()]
    print(f"orchestrating server-side: {len(arm_list)} arms x {len(pool_list)} poolings",
          flush=True)
    summary = json.loads(ladder_all.remote(cache_dir, arm_list, pool_list,
                                           prescale, clean_target, seed,
                                           skip_combined))
    if "combined" in summary:
        print(json.dumps(summary["combined"], indent=2), flush=True)
    results = summary["results"]
    for key, payload in results.items():
        a, p = key.split("|")
        if payload.get("error"):
            print(f"  {a:7s} {p:9s} ERROR {payload['error']}", flush=True)
        else:
            mean, sd = _spearman(payload["report"])
            shown = "n/a" if mean is None else f"{mean:.4f} +/- {sd:.4f}"
            print(f"  {a:7s} {p:9s} Spearman {shown}", flush=True)

    print("\n=== ladder, test Spearman (mean +/- sd over 5 splits) ===", flush=True)
    for p in pool_list:
        print(f"\n  pooling = {p}", flush=True)
        rows = []
        for a in arm_list:
            payload = results.get(f"{a}|{p}", {})
            if payload.get("error"):
                rows.append((a, None, None))
                continue
            rows.append((a, *_spearman(payload["report"])))
        base = next((m for a, m, _ in rows if a == "onehot" and m is not None), None)
        for a, mean, sd in sorted(rows, key=lambda r: (r[1] is None, -(r[1] or 0))):
            if mean is None:
                print(f"    {a:7s}      failed", flush=True)
                continue
            delta = "" if base is None or a == "onehot" else f"   vs A0 {mean - base:+.4f}"
            print(f"    {a:7s} {mean:.4f} +/- {sd:.4f}{delta}", flush=True)

    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}", flush=True)
