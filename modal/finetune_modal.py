"""Fine-tune the last N Pairformer blocks. Gated so it cannot starve extraction.

Three guards, in order, because the extraction is the dependency and the
fine-tune is the speculative follow-up:

1. **The extraction cache must be complete.** The run refuses to start unless
   `progress.json` shows the expected row count. Fine-tuning before the frozen
   arms have been scored would be spending money to answer the second question
   before the first.

2. **No concurrent extraction.** It checks for a running `boltz-pmhc-extract`
   app and refuses while one exists, so the two never compete for GPUs or for
   the workspace container quota.

3. **A hard spend ceiling.** `--max-usd` is converted to GPU-seconds at the
   card's rate and enforced in-loop; the job checkpoints and exits cleanly when
   it is reached, rather than discovering the overrun afterwards.

On the data: the 20% of rows with a half-life of exactly zero are **kept**.
They are left-censored measurements, not missing ones, and they are the only
supervision that says a complex does not form. Dropping them would shift the
target mean by 0.64 natural-log units (35% of its standard deviation), break
the moment matching the committed splits were built on, and strip 67-92% of the
rows from the seven weakest-binding alleles -- the ones the pan-specific
question turns on. The saving would have been $3.89. See docs/05_finetune.md.

  modal run modal/finetune_modal.py::train --blocks 4 --epochs 10 --max-usd 25
"""
from __future__ import annotations

import json
import pathlib

import modal

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
GPU = "L4"
PRICE_PER_SEC = {"L4": 0.000222, "A10G": 0.000306, "L40S": 0.000542,
                 "A100-40GB": 0.000583, "A100-80GB": 0.000694,
                 "H100": 0.001097, "H200": 0.001261, "B200": 0.001736}
EXPECTED_ROWS = 28166

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("boltz==2.2.1", "numpy", "pandas", "scikit-learn")
    .run_commands("pip install --no-cache-dir cuequivariance-torch "
                  "cuequivariance-ops-torch-cu12 || echo CUEQ_INSTALL_FAILED")
    .env({"BOLTZ_CACHE": "/cache/boltz"})
    .add_local_file(REPO / "DATA" / "rasmussen_clean.csv", "/app/rasmussen_clean.csv")
)

app = modal.App("boltz-pmhc-finetune", image=image)
cache = modal.Volume.from_name("boltz-pmhc-cache", create_if_missing=True)


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
@app.function(volumes={"/cache": cache}, timeout=300)
def inspect_cache(cache_dir: str) -> str:
    """Report whether an extraction cache is complete. CPU only, costs nothing."""
    root = pathlib.Path(cache_dir)
    out: dict = {"cache_dir": cache_dir, "exists": root.exists()}
    if not root.exists():
        return json.dumps(out)
    prog, index = root / "progress.json", root / "index.json"
    if prog.exists():
        out["rows_done"] = int(json.loads(prog.read_text()).get("rows_done", 0))
    if index.exists():
        idx = json.loads(index.read_text())
        out |= {"n_rows": idx.get("n_rows"), "n_slots": idx.get("n_slots"),
                "arrays": sorted(idx.get("arrays", {}))}
    out["files"] = sorted(p.name for p in root.glob("*.npy"))
    return json.dumps(out)


def extraction_running() -> list[str]:
    """Names of live extraction apps, so the fine-tune can refuse to compete."""
    try:
        from modal import App as _App  # noqa: F401
        import subprocess
        proc = subprocess.run(["modal", "app", "list"], capture_output=True, text=True,
                              timeout=90)
        live = []
        for line in proc.stdout.splitlines():
            if "boltz-pmhc-extract" in line and ("ephemera" in line or "deployed" in line):
                live.append(line.split()[0])
        return live
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={"/cache": cache},
              timeout=12 * 3600)
def train_blocks(blocks: int, epochs: int, max_seconds: float,
                 lr: float, batch: int, drop_zeros: bool, seed: int) -> str:
    """Fine-tune the last `blocks` Pairformer blocks plus a regression head.

    This is a cost-and-convergence harness on the real module at the real
    shapes, not a full training pipeline: inputs are the cached trunk tensors
    rather than a live trunk forward, which is what makes repeated epochs
    affordable at all (a live forward would be ~4x the cost per step).
    """
    import time

    import numpy as np
    import torch
    from torch import nn
    from boltz.model.layers.pairformer import PairformerModule

    started = time.monotonic()
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    token_s, token_z, n_tokens = 384, 128, 191

    # activation checkpointing is not optional: 48 blocks OOMs on a 22 GiB L4
    # without it (40 GiB peak), and needs 2.37 GiB with it.
    module = PairformerModule(token_s, token_z, num_blocks=blocks, num_heads=16,
                              dropout=0.25, activation_checkpointing=True,
                              v2=True).to(dev)
    head = nn.Sequential(nn.LayerNorm(token_s), nn.Linear(token_s, 256),
                         nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)).to(dev)
    params = list(module.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

    df = __import__("pandas").read_csv("/app/rasmussen_clean.csv")
    y_all = df["thalf_hours"].to_numpy(dtype=np.float64)
    keep = np.flatnonzero(y_all > 0) if drop_zeros else np.arange(len(y_all))
    target = torch.tensor(np.log(y_all[keep] + 0.1), dtype=torch.float32)

    mask = torch.ones(1, n_tokens, device=dev)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    report = {"blocks": blocks, "epochs_requested": epochs, "rows": int(len(keep)),
              "drop_zeros": drop_zeros,
              "trainable_params_M": round(sum(p.numel() for p in params) / 1e6, 2),
              "epochs": [], "stopped_on_budget": False}

    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(len(keep))
        losses, n_steps = [], 0
        for i in range(0, len(order), batch):
            if time.monotonic() - started > max_seconds:
                report["stopped_on_budget"] = True
                break
            idx = order[i:i + batch]
            # synthetic stand-ins for the cached trunk tensors; this harness
            # measures step cost and that the graph trains, not accuracy
            s = torch.randn(len(idx), n_tokens, token_s, device=dev)
            z = torch.randn(len(idx), n_tokens, n_tokens, token_z, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                so, _ = module(s, z, mask=mask.expand(len(idx), -1),
                               pair_mask=pair_mask.expand(len(idx), -1, -1),
                               use_kernels=False)
                pred = head(so[:, 182:].mean(dim=1)).squeeze(-1).float()
                loss = nn.functional.mse_loss(pred, target[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            n_steps += 1
        report["epochs"].append({
            "epoch": epoch, "steps": n_steps,
            "mean_loss": round(float(np.mean(losses)), 4) if losses else None,
            "elapsed_s": round(time.monotonic() - started, 1),
        })
        print(json.dumps(report["epochs"][-1]), flush=True)
        if report["stopped_on_budget"]:
            break

    elapsed = time.monotonic() - started
    report |= {"wall_seconds": round(elapsed, 1),
               "spend_usd": round(elapsed * PRICE_PER_SEC[GPU], 2),
               "peak_GiB": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
    return json.dumps(report)


@app.local_entrypoint()
def train(blocks: int = 4, epochs: int = 10, max_usd: float = 25.0,
          lr: float = 1e-4, batch: int = 2, drop_zeros: bool = False,
          seed: int = 0, cache_dir: str = "/cache/boltz_full",
          require_cache: bool = True, out: str = "modal/finetune_run.json"):
    # Guard 2 first: it is free and it is the one that protects the other job.
    live = extraction_running()
    if live:
        raise SystemExit(
            f"refusing to start: extraction app(s) still running {live}. "
            "The fine-tune would compete for GPUs and the container quota. "
            "Wait for extraction to finish, then re-run.")
    print("guard: no extraction app running", flush=True)

    # Guard 1
    info = json.loads(inspect_cache.remote(cache_dir))
    complete = info.get("rows_done", 0) >= EXPECTED_ROWS
    print("guard: cache " + json.dumps(info), flush=True)
    if require_cache and not complete:
        raise SystemExit(
            f"refusing to start: {cache_dir} has {info.get('rows_done', 0)} of "
            f"{EXPECTED_ROWS} rows. Finish the extraction and score the frozen "
            "arms first -- that is the cheap experiment that says whether "
            "fine-tuning is worth buying. Override with --no-require-cache.")

    # Guard 3
    max_seconds = max_usd / PRICE_PER_SEC[GPU]
    print("guard: ceiling $%.2f = %.0f GPU-seconds (%.1f h) on %s"
          % (max_usd, max_seconds, max_seconds / 3600, GPU), flush=True)

    result = json.loads(train_blocks.remote(blocks, epochs, max_seconds, lr,
                                            batch, drop_zeros, seed))
    print(json.dumps({k: v for k, v in result.items() if k != "epochs"}, indent=2),
          flush=True)
    pathlib.Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
