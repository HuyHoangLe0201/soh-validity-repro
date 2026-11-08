"""Step 12 - noise of the hyper-parameter search under P3 (leave-one-group-out).

For inner_seed in 1..N the inner search uses a different subsample phase (every 3rd row, offset seed % 3)
and a different random assignment of whole inner groups (cells for `tuned`, condition groups for
`tuned_grp`) to the 3 inner folds.  Seed 0 is the run already stored in step 7 / step 9.

Usage: python -u 12_tuning_repeats.py <dataset> <out_dir> [n_seeds=3] [featset=F1]
"""
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import predict  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")

if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    d, feats = m7.load(name)
    out.mkdir(parents=True, exist_ok=True)
    store = {}
    for seed in range(1, n_seeds + 1):
        for regime in ("tuned", "tuned_grp"):
            for model in m7.FAST4:
                pr, ch = predict(d, feats, "SOH_rated", "P3", model, regime, 0, inner_seed=seed)
                store[f"SOH_rated|{regime}|P3|{model}|{seed}"] = pr
                m = ~np.isnan(pr)
                rm = float(np.sqrt(np.mean((pr[m] - d["SOH_rated"].values[m]) ** 2)))
                print(f"[{name}] seed={seed} {regime:9s} {model:12s} rmse={rm:.3f}", flush=True)
            np.savez_compressed(out / f"{name}_tuningrepeats_preds.npz", **store)
