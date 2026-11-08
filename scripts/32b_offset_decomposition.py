"""Step 32b - why does the per-cell offset help on some datasets and hurt on others?
For ridge and LightGBM trained outside cell-disjoint folds (partition seed 100) we compute, on the last 40 % of the rows (pooled over rows, as in step 28):
  RMSE of the base model; RMSE after adding the per-cell MEDIAN early residual (step 28 recipe) and the per-cell MEAN early residual;
  RMSE after an oracle offset (the cell's own mean late residual: the best a constant per-cell shift could do);
  the share of the late MSE that is between-cell (constant per cell) rather than within-cell.
f = 0.6.  Usage: python 32b_offset_decomposition.py <out_csv>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import DEFAULTS, build  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")
rows = []
for ds in ("matr", "hust", "xjtu", "tju"):
    d, feats = m7.load(ds)
    d = d.reset_index(drop=True)
    X, y = d[feats].values.astype(float), d["SOH_rated"].values
    cells = d["cell"].values
    rk = d.groupby("cell")["cycle"].rank(method="first").values
    n = d.groupby("cell")["cycle"].transform("size").values
    late = rk > np.floor(0.6 * n); early = ~late
    rng = np.random.default_rng(100)
    u = d["cell"].unique(); rng.shuffle(u)
    fold = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values
    for m in ("Ridge", "LightGBM"):
        res = np.full(len(d), np.nan)
        for k in range(5):
            inf = fold == k
            res[inf] = y[inf] - build(m, DEFAULTS[m]).fit(X[~inf], y[~inf]).predict(X[inf])
        s = pd.DataFrame({"c": cells, "r": res, "late": late})
        med = s[~s.late].groupby("c").r.median(); mean = s[~s.late].groupby("c").r.mean(); lmean = s[s.late].groupby("c").r.mean()
        rl = res[late]; cl = cells[late]
        rm = lambda v: float(np.sqrt(np.mean(v ** 2)))  # noqa: E731
        base = rm(rl)
        r_med = rm(rl - pd.Series(cl).map(med).fillna(0).values)
        r_mean = rm(rl - pd.Series(cl).map(mean).fillna(0).values)
        r_orc = rm(rl - pd.Series(cl).map(lmean).fillna(0).values)
        between = float(1 - r_orc ** 2 / base ** 2)
        rows.append({"dataset": ds, "model": m, "rmse_base": base, "rmse_median_offset": r_med, "rmse_mean_offset": r_mean, "rmse_oracle_offset": r_orc,
                     "between_cell_share_of_mse": between})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
