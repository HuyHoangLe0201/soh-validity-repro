"""Step 20 - temporal versus random within-cell splits (the 'share of one cell's data' design of Table practice).

For every cell the rows are ordered by cycle.  PT: the first 60 % of a cell's rows (all cells pooled) train the model, the last 40 %
are the test rows.  PR: the same 60/40 proportion, but the rows of a cell are assigned at random (3 seeds).
Default hyper-parameters, F1 features, rated label, models Ridge / LightGBM / XGBoost / RandomForest.
Predictions are saved with a train mask so that step 21 can score all protocols on the same (late) rows.

Usage: python -u 20_temporal.py <dataset> <out_dir>   (dataset in matr, hust, xjtu)
"""
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import DEFAULTS, build  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")
MODELS = ["Ridge", "LightGBM", "XGBoost", "RandomForest"]
FRAC = 0.6

if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    out.mkdir(parents=True, exist_ok=True)
    d, feats = m7.load(name)
    d = d.reset_index(drop=True)
    X, y = d[feats].values.astype(float), d["SOH_rated"].values
    rank = d.groupby("cell")["cycle"].rank(method="first").values
    n = d.groupby("cell")["cycle"].transform("size").values
    late = rank > np.floor(FRAC * n)            # last 40 % of each cell
    print(f"{name}: cells={d.cell.nunique()} rows={len(d)} late rows={late.sum()}", flush=True)
    store = {"late": late}
    split = {"PT": ~late}
    for s in range(3):
        rng = np.random.default_rng(s)
        r = np.zeros(len(d), bool)
        for c, idx in d.groupby("cell").indices.items():
            k = int(np.floor(FRAC * len(idx)))
            r[rng.permutation(idx)[:k]] = True
        split[f"PR{s}"] = r
    for key, tr in split.items():
        for m in MODELS:
            p = np.full(len(d), np.nan, dtype=np.float32)
            te = ~tr
            p[te] = build(m, DEFAULTS[m]).fit(X[tr], y[tr]).predict(X[te])
            store[f"{key}|{m}"] = p
            store[f"{key}|mask"] = te
            print(f"[{name}] {key} {m:12s} rmse={np.sqrt(np.mean((p[te]-y[te])**2)):.3f}", flush=True)
    store["y"] = y
    store["y_train_mean_PT"] = np.array([y[~late].mean(), y[late].mean()])
    np.savez_compressed(out / f"{name}_temporal.npz", **store)
