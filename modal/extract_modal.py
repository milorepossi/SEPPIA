"""Modal extraction of Boltz-2 trunk features, sharded across containers.

`scripts/extract_boltz.py:extract()` takes every path as an argument, so this
wrapper only mounts storage and hands it paths and a row list.

Sharding is the whole point of running here rather than on a single Hugging
Face Job: Modal bills per second and fans out, so N containers finish N times
sooner for the **same total cost**. Wall clock becomes free; only GPU-seconds
are charged. Each shard writes its own cache directory and the shards are
merged afterwards, which keeps the resume checkpoint per-shard and means one
failed container only costs its own slice.

  modal run modal/extract_modal.py::run --rows DATA/pilot_source_rows.json --shards 8
  modal run modal/extract_modal.py::merge --out-dir /cache/boltz_pilot
"""
from __future__ import annotations

import json
import pathlib

import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
GPU = "L4"            # set from the sweep's $/complex winner
SHARD_TIMEOUT = 8 * 3600

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("boltz==2.2.1", "numpy", "pandas", "openpyxl")
    .run_commands("pip install --no-cache-dir cuequivariance-torch "
                  "cuequivariance-ops-torch-cu12 || echo CUEQ_INSTALL_FAILED")
    .env({"BOLTZ_CACHE": "/cache/boltz"})
    .add_local_file(REPO / "scripts" / "extract_boltz.py", "/app/extract_boltz.py")
    .add_local_file(REPO / "scripts" / "extract_embeddings.py", "/app/extract_embeddings.py")
    .add_local_file(REPO / "DATA" / "rasmussen_clean.csv", "/app/rasmussen_clean.csv")
)

app = modal.App("boltz-pmhc-extract", image=image)
cache = modal.Volume.from_name("boltz-pmhc-cache", create_if_missing=True)


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=SHARD_TIMEOUT,
              retries=modal.Retries(max_retries=2, backoff_coefficient=1.0))
def extract_shard(shard: int, n_shards: int, source_rows: list[int],
                  out_root: str, batch_size: int = 256,
                  sampling_steps: int = 10, recycling_steps: int = 3,
                  max_seconds: float | None = None) -> dict:
    import sys
    import time

    sys.path.insert(0, "/app")
    import extract_boltz

    mine = source_rows[shard::n_shards]          # strided, so shards are balanced
    out_dir = f"{out_root}/shard_{shard:03d}"
    started = time.monotonic()

    def guard(_stat: dict) -> None:
        if max_seconds and time.monotonic() - started > max_seconds:
            raise KeyboardInterrupt("max-seconds reached")

    try:
        summary = extract_boltz.extract(
            "/app/rasmussen_clean.csv", out_dir, source_rows=mine,
            boltz_cache="/cache/boltz", work_dir=f"/tmp/wk{shard}",
            batch_size=batch_size, sampling_steps=sampling_steps,
            recycling_steps=recycling_steps, resume=True, progress_callback=guard)
    except KeyboardInterrupt:
        summary = {"stopped_on_budget": True}
    finally:
        cache.commit()

    summary |= {"shard": shard, "n_assigned": len(mine),
                "elapsed_seconds": round(time.monotonic() - started, 1)}
    return {k: v for k, v in summary.items() if k != "batches"}


@app.function(volumes={"/cache": cache}, timeout=3600)
def merge_shards(out_root: str, dest: str) -> dict:
    """Concatenate shard caches into one cache ordered by source_row.

    Shards were assigned by stride, so the merged order is restored by sorting
    on source_row -- which is also the key `arm_features.row_positions` joins a
    split on, so the merged cache is a drop-in for the ladder.
    """
    import numpy as np

    root = pathlib.Path(out_root)
    shards = sorted(p for p in root.glob("shard_*") if (p / "index.json").exists())
    if not shards:
        raise FileNotFoundError(f"no completed shards under {out_root}")

    indices = [json.loads((p / "index.json").read_text()) for p in shards]
    names = list(indices[0]["arrays"])
    rows: list[tuple[int, int, int]] = []          # (source_row, shard_i, position)
    pairs: dict[int, list] = {}
    for si, idx in enumerate(indices):
        done = json.loads((shards[si] / "progress.json").read_text())["rows_done"]
        for pos, sr in enumerate(idx["source_row"][:done]):
            rows.append((int(sr), si, pos))
            pairs[int(sr)] = idx["pairs"][pos]
    rows.sort()

    dest_dir = pathlib.Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    opened = [{n: np.load(p / n, mmap_mode="r") for n in names} for p in shards]
    out = {}
    for n in names:
        dim = opened[0][n].shape[-1]
        out[n] = np.lib.format.open_memmap(
            dest_dir / n, mode="w+", dtype=np.float16,
            shape=(len(rows), indices[0]["n_slots"], dim))
    for k, (_sr, si, pos) in enumerate(rows):
        for n in names:
            out[n][k] = opened[si][n][pos]
    for n in names:
        out[n].flush()

    merged = dict(indices[0])
    merged |= {"n_rows": len(rows),
               "source_row": [sr for sr, _, _ in rows],
               "pairs": [pairs[sr] for sr, _, _ in rows],
               "merged_from": [p.name for p in shards]}
    (dest_dir / "index.json").write_text(json.dumps(merged, indent=2) + "\n")
    (dest_dir / "progress.json").write_text(
        json.dumps({"rows_done": len(rows), "fingerprint": merged["fingerprint"]}) + "\n")
    cache.commit()
    return {"merged_rows": len(rows), "shards": len(shards), "dest": dest,
            "arrays": {n: list(out[n].shape) for n in names}}


@app.local_entrypoint()
def run(rows: str = "DATA/pilot_source_rows.json", shards: int = 8,
        out_root: str = "/cache/boltz_pilot_shards", batch_size: int = 256,
        max_seconds: float = 0.0, limit: int = 0):
    source_rows = json.loads(pathlib.Path(rows).read_text())
    if limit:
        source_rows = source_rows[:limit]
    print(f"{len(source_rows)} complexes over {shards} shards on {GPU}", flush=True)

    args = [(s, shards, source_rows, out_root, batch_size, 10, 3,
             max_seconds or None) for s in range(shards)]
    total_gpu_seconds = 0.0
    for res in extract_shard.starmap(args):
        total_gpu_seconds += res.get("elapsed_seconds", 0.0)
        print(json.dumps(res), flush=True)

    rate = {"L4": 0.000222, "A10G": 0.000306, "L40S": 0.000542,
            "A100-40GB": 0.000583, "A100-80GB": 0.000694, "H100": 0.001097,
            "H200": 0.001261, "B200": 0.001736}[GPU]
    print(json.dumps({"total_gpu_seconds": round(total_gpu_seconds, 1),
                      "total_gpu_hours": round(total_gpu_seconds / 3600, 2),
                      "estimated_usd": round(total_gpu_seconds * rate, 2)}, indent=2))


@app.local_entrypoint()
def merge(out_root: str = "/cache/boltz_pilot_shards",
          dest: str = "/cache/boltz_pilot"):
    print(json.dumps(merge_shards.remote(out_root, dest), indent=2))
