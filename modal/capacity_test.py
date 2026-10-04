"""Can Modal actually give us S concurrent L4 containers?

The overnight extraction plan rests entirely on sharding: 48 GPU-hours on an L4
becomes 6 hours of wall clock only if 8 containers genuinely run at the same
time. If the workspace or the region caps us below that, the job silently takes
proportionally longer and misses the window.

This checks it directly and cheaply, with a minimal image and no Boltz: every
container records when it started and stopped, and overlap is computed from
those timestamps. A cap shows up as staggered start times rather than an error.

  modal run modal/capacity_test.py::check --shards 8
"""
from __future__ import annotations

import json
import pathlib
import time

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")
app = modal.App("boltz-pmhc-capacity", image=image)


def _hold_body(shard: int, seconds: float) -> str:
    """Occupy one L4 for `seconds`, confirming the GPU is real and usable."""
    import torch

    t0 = time.time()
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA"
    # a real allocation and a real matmul, so we know the GPU is not a stub
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(40):
        x = (x @ x.T).div_(4096).to(torch.bfloat16)
    torch.cuda.synchronize()
    busy = time.time() - t0
    time.sleep(max(0.0, seconds - busy))
    return json.dumps({
        "shard": shard, "device": device,
        "mem_GiB": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
        "start": t0, "end": time.time(),
    })


@app.function(gpu="L4", cpu=10.0, timeout=600, max_containers=64)
def hold_l4(shard: int, seconds: float) -> str:
    return _hold_body(shard, seconds)


@app.function(gpu="L40S", cpu=10.0, timeout=600, max_containers=64)
def hold_l40s(shard: int, seconds: float) -> str:
    return _hold_body(shard, seconds)


@app.function(gpu="H100", cpu=10.0, timeout=600, max_containers=64)
def hold_h100(shard: int, seconds: float) -> str:
    return _hold_body(shard, seconds)


HOLDS = {"L4": hold_l4, "L40S": hold_l40s, "H100": hold_h100}


@app.local_entrypoint()
def check(shards: int = 8, seconds: float = 45.0, gpu: str = "L4",
          out: str = "modal/capacity_results.json"):
    launched = time.time()
    hold = HOLDS[gpu]
    rows = [json.loads(r) for r in hold.starmap([(s, seconds) for s in range(shards)])]
    rows.sort(key=lambda r: r["start"])

    # peak overlap: how many containers were ever running at the same instant
    events = [(r["start"], 1) for r in rows] + [(r["end"], -1) for r in rows]
    events.sort()
    live = peak = 0
    for _, delta in events:
        live += delta
        peak = max(peak, live)

    report = {
        "gpu": gpu,
        "requested_shards": shards,
        "containers_returned": len(rows),
        "peak_concurrent": peak,
        "first_start_offset_s": round(rows[0]["start"] - launched, 1),
        "last_start_offset_s": round(rows[-1]["start"] - launched, 1),
        "start_spread_s": round(rows[-1]["start"] - rows[0]["start"], 1),
        "devices": sorted({r["device"] for r in rows}),
        "total_wall_s": round(max(r["end"] for r in rows) - launched, 1),
    }
    report["verdict"] = (
        "full parallelism" if peak >= shards else
        f"capped at {peak} of {shards} concurrent containers")
    print(json.dumps(report, indent=2), flush=True)
    for r in rows:
        print("  shard %2d  start +%6.1fs  end +%6.1fs  %s"
              % (r["shard"], r["start"] - launched, r["end"] - launched, r["device"]),
              flush=True)
    pathlib.Path(out).write_text(json.dumps({"report": report, "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {out}", flush=True)
