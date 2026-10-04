import json, sys
import numpy as np
import modal

app = modal.App("boltz-verify")
cache = modal.Volume.from_name("boltz-pmhc-cache")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "pandas")

@app.function(image=image, volumes={"/cache": cache}, timeout=1800, cpu=4.0,
              memory=16384)
def verify(cache_dir: str) -> str:
    import pathlib
    root = pathlib.Path(cache_dir)
    idx = json.loads((root / "index.json").read_text())
    out = {"n_rows": idx["n_rows"], "n_slots": idx["n_slots"],
           "shards_merged": len(idx.get("merged_from", []))}
    sr = np.array(idx["source_row"])
    out["source_rows_sorted"] = bool((np.diff(sr) > 0).all())
    out["source_rows_complete"] = bool(sr.min() == 2 and sr.max() == 28167
                                       and len(sr) == 28166 and len(set(sr.tolist())) == 28166)
    arrays = {}
    for name in idx["arrays"]:
        a = np.load(root / name, mmap_mode="r")
        # an unwritten row stays exactly zero everywhere
        zero_rows = 0
        peak = 0.0
        nonfinite = 0
        for s in range(0, a.shape[0], 4096):
            blk = np.asarray(a[s:s + 4096], dtype=np.float32)
            zero_rows += int((np.abs(blk).sum(axis=(1, 2)) == 0).sum())
            peak = max(peak, float(np.abs(blk[np.isfinite(blk)]).max()))
            nonfinite += int((~np.isfinite(blk)).sum())
        arrays[name] = {"shape": list(a.shape), "dtype": str(a.dtype),
                        "all_zero_rows": zero_rows, "peak_abs": round(peak, 2),
                        "nonfinite": nonfinite}
    out["arrays"] = arrays
    return json.dumps(out)

@app.local_entrypoint()
def main(cache_dir: str = "/cache/boltz_full"):
    print(json.dumps(json.loads(verify.remote(cache_dir)), indent=2))
