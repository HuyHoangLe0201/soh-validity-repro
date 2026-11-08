"""Step 46 - two further tests of open explanations.
A. Shrinkage: the sign change of the LightGBM residual over the life of long-lived MATR cells is what regression toward the mean of the SOH range would produce.
   (i) slope of the residual on the true SOH (pooled, cell-disjoint folds, partition seed 100); (ii) a model-level recalibration: y ~ a + b * prediction fitted on inner
   out-of-fold predictions of the training cells (3 inner folds by cell) and applied to the held-out cells; late-row RMSE with and without.
B. Fade rate: is the temporal error of LightGBM concentrated in cells whose SOH falls fastest or accelerates (knee)?  Per cell, on the late rows: slope of the true SOH versus cycle
   (points per 100 cycles), acceleration (slope of the second half minus slope of the first half of the late rows), cell RMSE of the step-20 temporal LightGBM predictions;
   Spearman correlations and the share of the squared error carried by the fastest-fading third.
Usage: python 46_shrinkage_fade.py <step20> <out_dir>
"""
import importlib
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import DEFAULTS  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")


def lg():
    return lgb.LGBMRegressor(num_leaves=31, random_state=42, verbose=-1, n_jobs=1, **DEFAULTS["LightGBM"])


def main(d20, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    A, B = [], []
    for ds in ("matr", "hust", "xjtu", "tju"):
        d, feats = m7.load(ds); d = d.reset_index(drop=True)
        X, y = d[feats].values.astype(float), d["SOH_rated"].values
        cells = d["cell"].values
        rk = d.groupby("cell")["cycle"].rank(method="first").values
        n = d.groupby("cell")["cycle"].transform("size").values
        late = rk > np.floor(0.6 * n)
        rng = np.random.default_rng(100)
        u = d["cell"].unique(); rng.shuffle(u)
        fold = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values
        pred, rec = np.full(len(d), np.nan), np.full(len(d), np.nan)
        for k in range(5):
            inf = fold == k
            tr = np.where(~inf)[0]
            pred[inf] = lg().fit(X[tr], y[tr]).predict(X[inf])
            # inner out-of-fold predictions by cell
            tc = np.unique(cells[tr]); r2 = np.random.default_rng(k); r2.shuffle(tc)
            inner = pd.Series(cells[tr]).map({c: i % 3 for i, c in enumerate(tc)}).values
            oof = np.zeros(len(tr))
            for j in range(3):
                oof[inner == j] = lg().fit(X[tr[inner != j]], y[tr[inner != j]]).predict(X[tr[inner == j]])
            b, a = np.polyfit(oof, y[tr], 1)
            rec[inf] = a + b * pred[inf]
        res = y - pred
        slope = float(np.polyfit(y - y.mean(), res, 1)[0])
        rm = lambda v: float(np.sqrt(np.mean(v ** 2)))  # noqa: E731
        A.append({"dataset": ds, "residual_on_soh_slope": slope, "rmse_late_base": rm(res[late]), "rmse_late_recalibrated": rm((y - rec)[late]),
                  "rmse_all_base": rm(res), "rmse_all_recalibrated": rm(y - rec)})
        print(f"[{ds}] A", A[-1], flush=True)
        # B
        T = np.load(Path(d20) / ds / f"{ds}_temporal.npz"); ept = (T["PT|LightGBM"].astype(float) - y) ** 2
        rows = []
        for c, idx in d.groupby("cell").indices.items():
            ii = idx[late[idx]]
            if len(ii) < 8: continue
            o = ii[np.argsort(d["cycle"].values[ii])]
            cyc, s = d["cycle"].values[o].astype(float), y[o]
            h = len(o) // 2
            sl = np.polyfit(cyc, s, 1)[0] * 100
            acc = (np.polyfit(cyc[h:], s[h:], 1)[0] - np.polyfit(cyc[:h], s[:h], 1)[0]) * 100 if h >= 4 else np.nan
            rows.append({"c": c, "slope": sl, "acc": acc, "sse": float(ept[o].sum()), "n": len(o)})
        g = pd.DataFrame(rows).dropna()
        g["rmse"] = np.sqrt(g.sse / g.n)
        fast = g.slope <= g.slope.quantile(1 / 3)
        B.append({"dataset": ds, "cells": len(g), "spearman_rmse_vs_abs_slope": float(spearmanr(g.rmse, g.slope.abs())[0]), "spearman_rmse_vs_acceleration": float(spearmanr(g.rmse, -g.acc)[0]),
                  "fastest_third_share_of_sse": float(g.sse[fast].sum() / g.sse.sum()), "median_slope_fast": float(g.slope[fast].median()), "median_slope_rest": float(g.slope[~fast].median())})
        print(f"[{ds}] B", B[-1], flush=True)
    pd.DataFrame(A).to_csv(out / "shrinkage_recalibration.csv", index=False); pd.DataFrame(B).to_csv(out / "fade_rate.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
