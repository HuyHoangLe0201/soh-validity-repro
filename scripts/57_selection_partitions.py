"""Step 57 - optimism of selecting hyper-parameters on the reported folds, over many outer group partitions.

Step 37 used one leave-one-group-out partition for P3.  Here the natural groups (MATR: 70 charging policies, HUST: 10 identifier groups, XJTU: 6 batches,
TJU: 11 type x condition groups) are assigned at random to K outer folds, R = 8 times (5 for HUST and MATR, which are much larger; fold counts: MATR 5, HUST 5, XJTU 3, TJU 4).  For every partition, model
(Ridge, LightGBM, XGBoost) and dataset:
  default       fixed default hyper-parameters
  nested        configuration chosen by an inner 3-fold search whose folds hold out whole groups (training part of the outer fold only)
  best_global   configuration with the lowest pooled outer-fold error over the grid (selection on the reported folds)
Optimism = nested - best_global (how much lower the reported error looks when the configuration is picked on the reported folds).
Usage: python -u 57_selection_partitions.py <dataset> <out_dir>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import rankexp  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")
K = {"matr": 5, "hust": 5, "xjtu": 3, "tju": 4}
R = 8
MODELS = ["Ridge", "LightGBM", "XGBoost"]

if __name__ == "__main__":
    ds, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    d, feats = m7.load(ds)
    if ds == "matr":
        d["grp"] = pd.factorize(d["policy"])[0]
    d = d.reset_index(drop=True)
    X, y, grp = d[feats].values.astype(float), d["SOH_rated"].values, d["grp"].values
    cells = d["cell"].values
    ug = np.unique(grp)
    rows = []
    R = R if ds in ("xjtu", "tju") else 5     # HUST and MATR are 3-5 times larger: five partitions
    for rep in range(R):
        rng = np.random.default_rng(700 + rep)
        perm = rng.permutation(ug)
        fmap = {g: i % K[ds] for i, g in enumerate(perm)}
        fold = np.array([fmap[g] for g in grp])
        fl = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(K[ds])]
        n = len(y)
        for model in MODELS:
            grid = rankexp.GRIDS[model]
            sse = np.zeros(len(grid)); dflt = 0.0; nest = 0.0
            for tr, te in fl:
                for ci, p in enumerate(grid):
                    sse[ci] += float(np.sum((rankexp.build(model, p).fit(X[tr], y[tr]).predict(X[te]) - y[te]) ** 2))
                dflt += float(np.sum((rankexp.build(model, rankexp.DEFAULTS[model]).fit(X[tr], y[tr]).predict(X[te]) - y[te]) ** 2))
                p = rankexp.inner_select(model, X[tr], y[tr], cells[tr], "P3", grp[tr], rep)
                nest += float(np.sum((rankexp.build(model, p).fit(X[tr], y[tr]).predict(X[te]) - y[te]) ** 2))
            rows.append({"dataset": ds, "rep": rep, "model": model, "folds": K[ds], "default": np.sqrt(dflt / n), "nested": np.sqrt(nest / n),
                         "best_global": np.sqrt(sse.min() / n), "best_config": int(sse.argmin())})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out / f"{ds}_selection_partitions.csv", index=False)
