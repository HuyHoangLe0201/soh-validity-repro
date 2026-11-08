"""Step 45 - the pooled multilayer perceptron (window of one cycle) tuned as the other networks (hidden {16,32,64} x learning rate {1e-3,4e-3}, inner validation as in step 31).
Questions P2, P3, PT; compare with the fixed-setting MLP of step 7 (P2, P3) and the window MLP.  Usage: python -u 45_tuned_mlp.py <dataset> <out_dir>
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
m36 = importlib.import_module("36_tuned_more")
m7 = m26.m7

if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    out.mkdir(parents=True, exist_ok=True)
    d0, feats = m7.load(name)
    m26.STRIDE = 1 if len(d0) < 30000 else 2 if len(d0) < 70000 else 4
    m26.L = 1
    R = m26.Runner(d0, feats); d = R.d; n = len(d)
    rk = d.groupby("cell")["cycle"].rank(method="first").values
    ncell = d.groupby("cell")["cycle"].transform("size").values
    late = rk > np.floor(0.6 * ncell)
    store = {"late": late, "y": R.y0}; t0 = time.time()
    for q in ("P2", "P3", "PT"):
        splits = [(tr, te) for tr, te in folds(d, q, 0)] if q != "PT" else [(np.where(~late)[0], np.where(late)[0])]
        p = np.full(n, np.nan, np.float32); ch = []
        for tr, te in splits:
            h, lr = m36.tune_cls(R, m29.WinMLP, tr, m31.inner_val(d, tr, q, rk, ncell)); ch.append(f"{h}/{lr}")
            p[te] = m31.fit_pred(R, lambda f, h=h: m29.WinMLP(f, h), tr, te, 0, lr)
        store[f"{q}|MLP_tuned"] = p; store[f"chosen|{q}"] = np.array(ch)
        print(f"[{name}] MLP_tuned {q} rmse={np.sqrt(np.nanmean((p[late]-R.y0[late])**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(out / f"{name}_mlp_tuned.npz", **store)
