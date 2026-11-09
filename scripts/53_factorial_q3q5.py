"""Step 53 - factorial separation of the two differences between a new-cell question (Q3) and temporal forecasting (Q5).

Both factors are varied independently on the same cell-disjoint 5-fold partitions (2 random partitions, seeds 100, 101 as in step 22):
  c : share of the life (first c x n rows by cycle) of every NON-target cell that is available for training, c in {0.6, 0.8, 1.0}
  f : share of the life (first f x n rows) of the TARGET cell that is available for training, f in {0, 0.6}
Test rows are the last 40 % of every cell of the held-out fold.  Corners of the 2x2 design (c in {0.6, 1.0}, f in {0, 0.6}):
  A  c=1.0 f=0    new cell, other cells observed over the whole life            (Q3)
  B  c=0.6 f=0    new cell, other cells observed over the first 60 % only       (new cell without late-life training data)
  C  c=1.0 f=0.6  target's early rows + other cells over the whole life         (Q2, calibration with a late-life pool)
  D  c=0.6 f=0.6  every cell observed over the first 60 % only                  (Q5, temporal forecasting)
Default hyper-parameters, F1 features, rated label; models Ridge, LightGBM, XGBoost, RandomForest.
Usage: python -u 53_factorial_q3q5.py <dataset> <out_dir>
"""
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import DEFAULTS, build  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")
MODELS = ["Ridge", "LightGBM", "XGBoost", "RandomForest"]
CS = [0.6, 0.8, 1.0]
FS = [0.0, 0.6]
REPS = 2

if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    out.mkdir(parents=True, exist_ok=True)
    d, feats = m7.load(name)
    d = d.reset_index(drop=True)
    X, y = d[feats].values.astype(float), d["SOH_rated"].values
    rk = d.groupby("cell")["cycle"].rank(method="first").values
    n = d.groupby("cell")["cycle"].transform("size").values
    late = rk > np.floor(0.6 * n)
    store = {"late": late, "y": y}
    for rep in range(REPS):
        rng = np.random.default_rng(100 + rep)
        u = d["cell"].unique(); rng.shuffle(u)
        fold = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values
        for c in CS:
            for f in FS:
                tag = f"{rep}|{c}|{f}"
                for m in MODELS:
                    p = np.full(len(d), np.nan, np.float32)
                    if c == f:   # every cell observed over the same first share: training set does not depend on the fold
                        tr = rk <= np.floor(c * n)
                        p[late] = build(m, DEFAULTS[m]).fit(X[tr], y[tr]).predict(X[late])
                    else:
                        for k in range(5):
                            te_cells = fold == k
                            tr = (~te_cells & (rk <= np.floor(c * n))) | (te_cells & (rk <= np.floor(f * n)))
                            test = te_cells & late
                            p[test] = build(m, DEFAULTS[m]).fit(X[tr], y[tr]).predict(X[test])
                    store[f"{tag}|{m}"] = p
                    print(f"[{name}] rep{rep} c={c} f={f} {m:12s} rmse={np.sqrt(np.mean((p[late]-y[late])**2)):.3f}", flush=True)
        np.savez_compressed(out / f"{name}_factorial.npz", **store)
