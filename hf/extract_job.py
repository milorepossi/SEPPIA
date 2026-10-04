# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["boltz==2.2.1", "numpy", "pandas", "openpyxl"]
# ///
"""Hugging Face Job entrypoint for the Boltz-2 trunk extraction.

`scripts/extract_boltz.py:extract()` is a plain function taking every path as an
argument, so this wrapper only has to mount storage and hand it paths.

Budget discipline
-----------------
The job stops itself when `--max-seconds` is reached, flushing the memmap cache
and the resume checkpoint first. The cache is resumable and fingerprinted, so a
follow-up job continues from exactly where this one stopped. That makes the $20
budget enforceable from the outside rather than hoped for: launch, let it stop,
read how far it got, decide whether to buy more.

Usage (one shard of the pilot subset):

    hf jobs uv run hf/extract_job.py --flavor l4x1 --timeout 2h \\
      --volume hf://buckets/<user>/boltz-cache:/bcache:rw \\
      --volume hf://datasets/<user>/phla-inputs:/inputs:ro \\
      -- --dataset /inputs/rasmussen_clean.csv \\
         --source-rows /inputs/pilot_source_rows.json \\
         --out-dir /bcache/boltz_pilot --boltz-cache /bcache/boltz \\
         --max-seconds 5400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--source-rows", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--boltz-cache", required=True)
    ap.add_argument("--msa-dir", default=None)
    ap.add_argument("--scripts-dir", default=None,
                    help="directory holding extract_boltz.py; defaults to ./scripts")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop cleanly after this much wall time (budget guard)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    scripts = Path(args.scripts_dir) if args.scripts_dir else Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts))
    import extract_boltz

    source_rows = None
    if args.source_rows:
        text = Path(args.source_rows).read_text().strip()
        source_rows = json.loads(text) if text.startswith("[") else [
            int(v) for v in text.split() if v]

    started = time.monotonic()
    stop = {"hit": False}

    def guard(batch_stat: dict) -> None:
        """Called after each batch; raises to unwind once the budget is spent."""
        if args.max_seconds and time.monotonic() - started > args.max_seconds:
            stop["hit"] = True
            raise KeyboardInterrupt("max-seconds reached")

    summary: dict
    try:
        summary = extract_boltz.extract(
            args.dataset, args.out_dir, source_rows=source_rows,
            msa_dir=args.msa_dir, boltz_cache=args.boltz_cache,
            batch_size=args.batch_size, sampling_steps=args.sampling_steps,
            recycling_steps=args.recycling_steps, num_workers=args.num_workers,
            limit=args.limit, resume=True, progress_callback=guard)
    except KeyboardInterrupt:
        # extract() checkpoints after every CHECKPOINT_EVERY rows, so the cache
        # and progress.json on disk are already consistent. Report and exit 0:
        # hitting the budget is an expected outcome, not a failure.
        summary = {"stopped_on_budget": True,
                   "elapsed_seconds": round(time.monotonic() - started, 1)}

    summary["budget_stop"] = stop["hit"]
    terse = {k: v for k, v in summary.items() if k != "batches"}
    print(json.dumps(terse, indent=2), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
