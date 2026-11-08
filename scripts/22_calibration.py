"""Step 22 - calibration with early data of the target cell (the 'fine-tune on a share of the test cell' design).

Cell-disjoint 5-fold (2 random partitions).  For each fold the model is trained on ALL rows of the cells outside the fold plus the
FIRST fraction f of the rows (by cycle) of every cell inside the fold; it is scored on the LAST 40 % of the rows of the fold cells.
f = 0 is a new cell without any own data (P2 scored on the same late rows).  f = 0.6 uses all non-test rows of the target cell.
Models: Ridge, LightGBM (default settings); *_w give the target rows weight 10; 'TargetRidge' is a per-cell ridge model trained on the
early rows of that cell only (needs >= 10 early rows).  F1 features, rated label.
Usage: python -u 22_calibration.py <dataset> <out_dir>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import DEFAULTS, build  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")
FRACS = [0.0, 0.1, 0.2, 0.4, 0.6]
REPS = 2

if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    out.mkdir(parents=True, exist_ok=True)
    d, feats = m7.load(name)
    d = d.reset_index(drop=True)
    X, y = d[feats].values.astype(float), d["SOH_rated"].values
    cells = d["cell"].values
    rk = d.groupby("cell")["cycle"].rank(method="first").values
    n = d.groupby("cell")["cycle"].transform("size").values
    late = rk > np.floor(0.6 * n)
    store = {"late": late, "y": y}
    for rep in range(REPS):
        rng = np.random.default_rng(100 + rep)
        u = d["cell"].unique(); rng.shuffle(u)
        fold = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values
        for f in FRACS:
            early = rk <= np.floor(f * n)
            preds = {k: np.full(len(d), np.nan, np.float32) for k in ("Ridge", "LightGBM", "Ridge_w", "LightGBM_w", "TargetRidge")}
            for k in range(5):
                te_cells = fold == k
                tr = (~te_cells) | (te_cells & early)
                test = te_cells & late
                w = np.where(te_cells[tr], 10.0, 1.0)
                for m in ("Ridge", "LightGBM"):
                    mod = build(m, DEFAULTS[m]).fit(X[tr], y[tr])
                    preds[m][test] = mod.predict(X[test])
                    if f > 0:
                        if m == "Ridge":
                            mod = build(m, DEFAULTS[m]); mod.fit(X[tr], y[tr], ridge__sample_weight=w)
                        else:
                            mod = build(m, DEFAULTS[m]); mod.fit(X[tr], y[tr], sample_weight=w)
                        preds[m + "_w"][test] = mod.predict(X[test])
                if f > 0:
                    for c in np.unique(cells[te_cells]):
                        ic = (cells == c) & early
                        if ic.sum() >= 10:
                            mod = build("Ridge", DEFAULTS["Ridge"]).fit(X[ic], y[ic])
                            t_ = (cells == c) & late
                            preds["TargetRidge"][t_] = mod.predict(X[t_])
            for m, p in preds.items():
                store[f"{rep}|{f}|{m}"] = p
                msk = late & ~np.isnan(p)
                if msk.any():
                    print(f"[{name}] rep{rep} f={f:.1f} {m:12s} rmse={np.sqrt(np.mean((p[msk]-y[msk])**2)):.3f} n={msk.sum()}", flush=True)
    np.savez_compressed(out / f"{name}_calibration.npz", **store)
