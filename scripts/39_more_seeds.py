"""Step 39 - seeds 1 and 2 for the remaining networks (LSTM, TCN, window MLP, CNN-GRU, GRU with attention), questions P2, P3, PT, same partitions as step 29.
Usage: python -u 39_more_seeds.py <dataset> <out_dir>
"""
import importlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import folds  # noqa: E402

m26 = importlib.import_module("26_neural")
m29 = importlib.import_module("29_architectures")
m31 = importlib.import_module("31_tuned_arch")
m7 = m26.m7
CLS = {"LSTM": m29.LSTMNet, "TCN": m29.TCNNet, "WinMLP": m29.WinMLP, "CNNGRU": m31.CNNGRU, "AttnGRU": m31.AttnGRU}

if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    out.mkdir(parents=True, exist_ok=True)
    d0, feats = m7.load(name)
    m26.STRIDE = 1 if len(d0) < 30000 else 2 if len(d0) < 70000 else 4
    m26.L = 10
    R = m26.Runner(d0, feats)
    d = R.d; n = len(d)
    rk = d.groupby("cell")["cycle"].rank(method="first").values
    ncell = d.groupby("cell")["cycle"].transform("size").values
    late = rk > np.floor(0.6 * ncell)
    store = {"late": late, "y": R.y0}
    t0 = time.time()
    for q in ("P2", "P3", "PT"):
        splits = [(tr, te) for tr, te in folds(d, q, 0)] if q != "PT" else [(np.where(~late)[0], np.where(late)[0])]
        for v, cls in CLS.items():
            for seed in (1, 2):
                p = np.full(n, np.nan, np.float32)
                for tr, te in splits:
                    p[te] = m29.tp(R, cls, tr, te, seed)
                store[f"{q}|{v}|{seed}"] = p
                print(f"[{name}] {v:8s} s{seed} {q} rmse={np.sqrt(np.nanmean((p[late]-R.y0[late])**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(out / f"{name}_seeds.npz", **store)
