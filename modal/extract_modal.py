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
# From modal/sweep_results.json: L4 is cheapest per complex ($0.001364 at
# concurrency 4) even though L40S/H100/B200 are 2-3x faster, because Modal
# fans out and wall clock is bought with shards rather than a bigger card.
GPU = "L4"
WORKERS = 4           # concurrent boltz processes per GPU; ~3 GiB each
CPU = 4.0 * 2.5       # CPU in proportion to workers: each spawns dataloader
                      # threads, and starving them was measured to cost more
                      # than the GPU does
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
    .add_local_file(REPO / "scripts" / "merge_shards.py", "/app/merge_shards.py")
    .add_local_file(REPO / "DATA" / "rasmussen_clean.csv", "/app/rasmussen_clean.csv")
)

app = modal.App("boltz-pmhc-extract", image=image)
cache = modal.Volume.from_name("boltz-pmhc-cache", create_if_missing=True)


@app.function(gpu=GPU, cpu=CPU, memory=24576, volumes={"/cache": cache},
              timeout=SHARD_TIMEOUT,
              retries=modal.Retries(max_retries=2, backoff_coefficient=1.0))
def extract_shard(shard: int, n_shards: int, source_rows: list[int],
                  out_root: str, batch_size: int = 256,
                  sampling_steps: int = 10, recycling_steps: int = 3,
                  workers: int = WORKERS,
                  max_seconds: float | None = None) -> str:
    import json
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
            recycling_steps=recycling_steps, workers=workers,
            resume=True, progress_callback=guard)
    except KeyboardInterrupt:
        summary = {"stopped_on_budget": True}
    finally:
        cache.commit()

    summary |= {"shard": shard, "n_assigned": len(mine),
                "elapsed_seconds": round(time.monotonic() - started, 1)}
    return json.dumps({k: v for k, v in summary.items() if k != "batches"})


@app.function(volumes={"/cache": cache}, timeout=3600)
def merge_shards(out_root: str, dest: str) -> str:
    """Reassemble shard caches. Logic lives in scripts/merge_shards.py so it is
    testable locally with synthetic shards rather than on a paid run."""
    import json
    import sys

    sys.path.insert(0, "/app")
    import merge_shards as ms

    summary = ms.merge(out_root, dest)
    cache.commit()
    return json.dumps(summary)


@app.local_entrypoint()
def run(rows: str = "DATA/pilot_source_rows.json", shards: int = 8,
        out_root: str = "/cache/boltz_pilot_shards", batch_size: int = 256,
        max_seconds: float = 0.0, limit: int = 0):
    source_rows = json.loads(pathlib.Path(rows).read_text())
    if limit:
        source_rows = source_rows[:limit]
    print(f"{len(source_rows)} complexes over {shards} shards on {GPU}, "
          f"{WORKERS} concurrent boltz workers each", flush=True)

    args = [(s, shards, source_rows, out_root, batch_size, 10, 3, WORKERS,
             max_seconds or None) for s in range(shards)]
    total_gpu_seconds = 0.0
    for raw in extract_shard.starmap(args):
        res = json.loads(raw)
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
    print(json.dumps(json.loads(merge_shards.remote(out_root, dest)), indent=2))
