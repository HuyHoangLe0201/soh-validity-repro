"""Step 28 - fine-tuning and calibration schedules for Q2 (early rows of the target cell are labeled).

Cell-disjoint 5-fold (partition seed 100, as in steps 22/26).  For each fold a base network (GRU, as in step 26) is trained on the cells
outside the fold only; for f in {0.2, 0.6} the first fraction f of the rows of every fold cell is then used by one of
  concat   train from scratch on outside cells + early rows (step 26 design)
  head     freeze the recurrent layers, train the output head 5 epochs, lr 1e-3, on the early rows
  lowlr    all weights, 3 epochs, lr 5e-5, on the early rows
  replay   all weights, 3 epochs, lr 3e-4, on the early rows plus an equally large random sample of the outside rows
  offset   no retraining: add to every prediction of a cell the median residual of the base model on that cell's early rows
Offset calibration is also applied to the base LightGBM and ridge models (trained on the outside cells).  Seeds 0-2 for the networks.
Scored on the last 40 % of every fold cell.   Usage: python -u 28_finetune.py <dataset> <out_dir>
"""
import copy
import importlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import DEFAULTS, build  # noqa: E402

m26 = importlib.import_module("26_neural")
m7 = m26.m7
SEEDS = (0, 1, 2)
FRACS = (0.2, 0.6)


def offsets(pred_early, y_early, cells_early, cells_all):
    off = {c: float(np.median(y_early[cells_early == c] - pred_early[cells_early == c])) for c in np.unique(cells_early)}
    return np.array([off.get(c, 0.0) for c in cells_all])


if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    out.mkdir(parents=True, exist_ok=True)
    d, feats = m7.load(name)
    R = m26.Runner(d, feats)
    d = R.d
    m26.STRIDE = 1 if len(d) < 30000 else 2 if len(d) < 70000 else 4
    cells = d["cell"].values
    rk = d.groupby("cell")["cycle"].rank(method="first").values
    ncell = d.groupby("cell")["cycle"].transform("size").values
    late = rk > np.floor(0.6 * ncell)
    y = R.y0
    rng0 = np.random.default_rng(100)
    u = d["cell"].unique(); rng0.shuffle(u)
    fold = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values
    store = {"late": late, "y": y}
    t0 = time.time()
    n = len(d)

    def new(): return np.full(n, np.nan, np.float32)

    for k in range(5):
        inf = fold == k
        out_rows = np.where(~inf)[0]
        te = np.where(inf & late)[0]
        # tabular base models
        Xb = R.X0
        for m in ("Ridge", "LightGBM"):
            mod = build(m, DEFAULTS[m]).fit(Xb[out_rows], y[out_rows])
            pe, pl = mod.predict(Xb), None
            store.setdefault(f"{m}_base", new())[te] = pe[te]
            for f in FRACS:
                early = np.where(inf & (rk <= np.floor(f * ncell)))[0]
                off = offsets(pe[early], y[early], cells[early], cells[te])
                store.setdefault(f"{m}_offset{f}", new())[te] = pe[te] + off
        for seed in SEEDS:
            X, yt, ym, ys = R.prep(out_rows)
            net = (m26.GRUNet)(X.shape[1])
            m26.fit(net, X, R.W, yt, out_rows[::m26.STRIDE], m26.EPOCHS, 2e-3, seed)
            allp = m26.predict(net, X, R.W, np.arange(n)) * ys + ym
            store.setdefault(f"GRU_base|{seed}", new())[te] = allp[te]
            for f in FRACS:
                early = np.where(inf & (rk <= np.floor(f * ncell)))[0]
                store.setdefault(f"GRU_offset{f}|{seed}", new())[te] = allp[te] + offsets(allp[early], y[early], cells[early], cells[te])
                # concat
                store.setdefault(f"GRU_concat{f}|{seed}", new())[te] = R.train_predict("GRU", np.concatenate([out_rows, early]), te, seed)
                # head-only
                nh = copy.deepcopy(net); nh.rnn.requires_grad_(False)
                m26.fit(nh, X, R.W, yt, early, 5, 1e-3, seed)
                store.setdefault(f"GRU_head{f}|{seed}", new())[te] = m26.predict(nh, X, R.W, te) * ys + ym
                # low-lr all weights
                nl = copy.deepcopy(net)
                m26.fit(nl, X, R.W, yt, early, 3, 5e-5, seed)
                store.setdefault(f"GRU_lowlr{f}|{seed}", new())[te] = m26.predict(nl, X, R.W, te) * ys + ym
                # replay
                nr = copy.deepcopy(net)
                rs = np.random.default_rng(seed).choice(out_rows, size=min(len(early), len(out_rows)), replace=False)
                m26.fit(nr, X, R.W, yt, np.concatenate([early, rs]), 3, 3e-4, seed)
                store.setdefault(f"GRU_replay{f}|{seed}", new())[te] = m26.predict(nr, X, R.W, te) * ys + ym
        print(f"[{name}] fold {k} done ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(out / f"{name}_finetune.npz", **store)
    for key in sorted({k.split("|")[0] for k in store if k not in ("late", "y")}):
        ps = [store[k] for k in store if k.split("|")[0] == key]
        e = np.nanmean([np.sqrt(np.nanmean((p[late] - y[late]) ** 2)) for p in ps])
        print(f"[{name}] {key:20s} rmse={e:.3f}", flush=True)
