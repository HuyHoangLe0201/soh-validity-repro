"""Step 48 - tune the two remaining networks (CNN-GRU, GRU with attention) as in steps 31/36: hidden size {16,32,64} x learning rate {1e-3,4e-3}
chosen on an inner validation set (P2/P3: 20 % of the training cells; PT: the last 25 % of each training cell).  Questions P2, P3, PT.
Usage: python -u 48_tuned_cnngru_attn.py <dataset> <out_dir>"""
import importlib, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import folds
m26 = importlib.import_module("26_neural"); m31 = importlib.import_module("31_tuned_arch"); m36 = importlib.import_module("36_tuned_more")
m7 = m26.m7
CLS = {"CNNGRU_tuned": m31.CNNGRU, "AttnGRU_tuned": m31.AttnGRU}
if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]; out.mkdir(parents=True, exist_ok=True)
    d0, feats = m7.load(name)
    m26.STRIDE = 1 if len(d0) < 30000 else 2 if len(d0) < 70000 else 4
    m26.L = 10
    R = m26.Runner(d0, feats); d = R.d; n = len(d)
    rk = d.groupby("cell")["cycle"].rank(method="first").values
    ncell = d.groupby("cell")["cycle"].transform("size").values
    late = rk > np.floor(0.6 * ncell)
    store = {"late": late, "y": R.y0}; t0 = time.time()
    for q in ("P2", "P3", "PT"):
        splits = [(tr, te) for tr, te in folds(d, q, 0)] if q != "PT" else [(np.where(~late)[0], np.where(late)[0])]
        for v, cls in CLS.items():
            p = np.full(n, np.nan, np.float32); ch = []
            for tr, te in splits:
                h, lr = m36.tune_cls(R, cls, tr, m31.inner_val(d, tr, q, rk, ncell))
                ch.append(f"{h}/{lr}")
                p[te] = m31.fit_pred(R, lambda f, h=h, c=cls: c(f, h), tr, te, 0, lr)
            store[f"{q}|{v}"] = p; store[f"chosen|{q}|{v}"] = np.array(ch)
            print(f"[{name}] {v:14s} {q} rmse={np.sqrt(np.nanmean((p[late]-R.y0[late])**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(out / f"{name}_tuned_cnngru_attn.npz", **store)
