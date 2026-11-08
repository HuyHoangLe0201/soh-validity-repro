"""Step 31 - tuned GRU, two more architectures, and charge-curve (F2) inputs for the networks.
Variants (questions P2, P3, PT; same partitions and rows as steps 26/29):
  GRU_tuned  GRU whose hidden size {16,32,64} and learning rate {1e-3,4e-3} are chosen on an inner validation set of the training rows:
             P2/P3: 20 % of the training cells (random, seed 0) held out; PT: the last 25 % of each training cell (temporal inner split).
  CNNGRU     one conv layer (kernel 3) before a GRU;   AttnGRU  GRU with attention pooling over the window.
  GRU_F2     GRU on windows of the F2 features (voltage at fixed charge-capacity checkpoints), MATR, HUST, XJTU only; with LightGBM and ridge on
             the same rows and folds for comparison.
Usage: python -u 31_tuned_arch.py <dataset> <out_dir>
"""
import importlib
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import DEFAULTS, build, folds  # noqa: E402

m26 = importlib.import_module("26_neural")
m13 = importlib.import_module("13_waveform_experiment")
m7 = m26.m7
GRID = list(itertools.product((16, 32, 64), (1e-3, 4e-3)))


class CNNGRU(nn.Module):
    def __init__(self, f, h=32):
        super().__init__()
        self.conv = nn.Conv1d(f, h, 3, padding=1)
        self.rnn = nn.GRU(h, h, batch_first=True)
        self.out = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        z = nn.functional.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        o, _ = self.rnn(z)
        return self.out(o[:, -1]).squeeze(-1)


class AttnGRU(nn.Module):
    def __init__(self, f, h=32):
        super().__init__()
        self.rnn = nn.GRU(f, h, batch_first=True)
        self.att = nn.Linear(h, 1)
        self.out = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.rnn(x)
        a = nn.functional.softmax(self.att(o).squeeze(-1), dim=1).unsqueeze(-1)
        return self.out((a * o).sum(1)).squeeze(-1)


def fit_pred(R, make, tr, te, seed, lr=2e-3, rows=None):
    X, y, ym, ys = R.prep(tr)
    net = make(X.shape[1])
    m26.fit(net, X, R.W, y, np.asarray(tr)[::m26.STRIDE], m26.EPOCHS, lr, seed)
    return m26.predict(net, X, R.W, np.asarray(te)) * ys + ym


def tune(R, tr, d, inner):
    """inner: array of row indices (subset of tr) to validate on; training part = tr minus inner."""
    tr = np.asarray(tr)
    val = np.asarray(inner)
    fit_rows = np.setdiff1d(tr, val)
    best, best_s = None, np.inf
    X, y, ym, ys = R.prep(fit_rows)
    for h, lr in GRID:
        net = m26.GRUNet(X.shape[1], h)
        m26.fit(net, X, R.W, y, fit_rows[::m26.STRIDE], m26.EPOCHS, lr, 0)
        p = m26.predict(net, X, R.W, val[::2]) * ys + ym
        s = float(np.mean((p - R.y0[val[::2]]) ** 2))
        if s < best_s:
            best, best_s = (h, lr), s
    return best


def inner_val(d, tr, protocol, rk, ncell):
    tr = np.asarray(tr)
    if protocol == "PT":
        return tr[rk[tr] > np.floor(0.75 * 0.6 * ncell[tr])]  # last 25 % of the training part of each cell
    cells = d["cell"].values[tr]
    u = np.unique(cells); rng = np.random.default_rng(0); rng.shuffle(u)
    vc = set(u[: max(1, int(0.2 * len(u)))])
    return tr[np.isin(cells, list(vc))]


def run_variants(R, d, feats_tag, variants, store, extra_tab=None):
    n = len(d)
    rk = d.groupby("cell")["cycle"].rank(method="first").values
    ncell = d.groupby("cell")["cycle"].transform("size").values
    late = rk > np.floor(0.6 * ncell)
    store["late"], store["y"] = late, R.y0
    t0 = time.time()
    for q in ("P2", "P3", "PT"):
        splits = [(tr, te) for tr, te in folds(d, q, 0)] if q != "PT" else [(np.where(~late)[0], np.where(late)[0])]
        for v in variants:
            p = np.full(n, np.nan, np.float32)
            for tr, te in splits:
                if v == "GRU_tuned":
                    h, lr = tune(R, tr, d, inner_val(d, tr, q, rk, ncell))
                    p[te] = fit_pred(R, lambda f, h=h: m26.GRUNet(f, h), tr, te, 0, lr)
                    store.setdefault(f"chosen|{q}", []).append(f"{h}/{lr}")
                else:
                    cls = {"CNNGRU": CNNGRU, "AttnGRU": AttnGRU, "GRU": m26.GRUNet}[v]
                    p[te] = fit_pred(R, cls, tr, te, 0)
            store[f"{q}|{v}"] = p
            print(f"[{feats_tag}] {v:10s} {q} rmse={np.sqrt(np.nanmean((p[late]-R.y0[late])**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
        if extra_tab:
            for m in ("Ridge", "LightGBM"):
                p = np.full(n, np.nan, np.float32)
                for tr, te in splits:
                    p[te] = build(m, DEFAULTS[m]).fit(R.X0[tr], R.y0[tr]).predict(R.X0[te])
                store[f"{q}|{m}"] = p
                print(f"[{feats_tag}] {m:10s} {q} rmse={np.sqrt(np.nanmean((p[late]-R.y0[late])**2)):.3f}", flush=True)


if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    out.mkdir(parents=True, exist_ok=True)
    d0, feats = m7.load(name)
    m26.STRIDE = 1 if len(d0) < 30000 else 2 if len(d0) < 70000 else 4
    R = m26.Runner(d0, feats)
    store = {}
    run_variants(R, R.d, name, ["GRU_tuned", "CNNGRU", "AttnGRU"], store)
    store = {k: (np.array(v) if isinstance(v, list) else v) for k, v in store.items()}
    np.savez_compressed(out / f"{name}_tuned.npz", **store)
    wcsv = {"matr": "matr_waveform.csv.gz", "hust": "hust_waveform.csv.gz", "xjtu": "xjtu_waveform.csv.gz"}.get(name)
    if wcsv:
        d2, f2 = m13.merge(d0, pd.read_csv(Path(__file__).resolve().parent.parent / "results" / wcsv), name)
        m26.STRIDE = 1 if len(d2) < 30000 else 2 if len(d2) < 70000 else 4
        R2 = m26.Runner(d2, f2)
        s2 = {"grp": R2.d["grp"].values, "cell": R2.d["cell"].values}
        run_variants(R2, R2.d, name + "_F2", ["GRU"], s2, extra_tab=True)
        np.savez_compressed(out / f"{name}_f2.npz", **{k: (np.array(v) if isinstance(v, list) else v) for k, v in s2.items()})
