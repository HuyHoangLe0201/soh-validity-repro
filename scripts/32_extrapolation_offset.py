"""Step 32 - tests of two explanations that were stated as untested.
A. Temporal split: is the error of the forecast concentrated on late rows whose SOH lies below everything seen in training?
   RMSE of each model on in-range / out-of-range late rows (PT, step 20/26/29 predictions), and an intervention: LightGBM with linear leaves
   (linear_tree=True) and ridge + LightGBM on the ridge residuals, under P2, P3 and PT (same rows, partitions).
B. Offset calibration: does the early-life residual of a cell predict its late-life residual?  For ridge and LightGBM trained outside the
   cell-disjoint folds (partition seed 100) the mean residual on the first 60 % of each cell is correlated (Spearman) with that on the last 40 %.
Usage: python 32_extrapolation_offset.py <step7> <step20> <step26> <step29> <out>
"""
import importlib
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import DEFAULTS, build, folds  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")


def lin_tree():
    return lgb.LGBMRegressor(num_leaves=31, random_state=42, verbose=-1, n_jobs=1, linear_tree=True, **DEFAULTS["LightGBM"])


class RidgeLGB:
    def __init__(self):
        self.r, self.g = build("Ridge", DEFAULTS["Ridge"]), build("LightGBM", DEFAULTS["LightGBM"])

    def fit(self, X, y):
        self.r.fit(X, y)
        self.g.fit(X, y - self.r.predict(X))
        return self

    def predict(self, X):
        return self.r.predict(X) + self.g.predict(X)


def main(d7, d20, d26, d29, out):
    inr, interv, persist = [], [], []
    for ds in ("matr", "hust", "xjtu", "tju"):
        d, feats = m7.load(ds)
        d = d.reset_index(drop=True)
        X, y = d[feats].values.astype(float), d["SOH_rated"].values
        cells = d["cell"].values
        rk = d.groupby("cell")["cycle"].rank(method="first").values
        n = d.groupby("cell")["cycle"].transform("size").values
        late = rk > np.floor(0.6 * n)
        T = np.load(Path(d20) / ds / f"{ds}_temporal.npz"); N = np.load(Path(d26) / ds / f"{ds}_neural.npz"); A = np.load(Path(d29) / ds / f"{ds}_arch.npz")
        assert np.allclose(y, T["y"])
        ymin = y[~late].min()
        inrange = late & (y >= ymin); outrange = late & (y < ymin)
        preds = {m: T[f"PT|{m}"].astype(float) for m in ("Ridge", "LightGBM", "XGBoost", "RandomForest")}
        preds["GRU"], preds["TRF"] = N["PT|GRU"].astype(float), N["PT|TRF"].astype(float)
        for v in ("LSTM", "TCN", "WinMLP", "GRU_L30"):
            preds[v] = A[f"PT|{v}"].astype(float)
        for m, p in preds.items():
            r = {"dataset": ds, "model": m, "share_out_of_range": float(outrange.sum() / late.sum()), "train_min": float(ymin)}
            r["rmse_in"] = float(np.sqrt(np.mean((p[inrange] - y[inrange]) ** 2))) if inrange.any() else np.nan
            r["rmse_out"] = float(np.sqrt(np.mean((p[outrange] - y[outrange]) ** 2))) if outrange.any() else np.nan
            r["rmse_all_late"] = float(np.sqrt(np.mean((p[late] - y[late]) ** 2)))
            inr.append(r)
        # intervention
        for q in ("P2", "P3", "PT"):
            sp = [(tr, te) for tr, te in folds(d, q, 0)] if q != "PT" else [(np.where(~late)[0], np.where(late)[0])]
            for name, mk in (("LightGBM_linear_tree", lin_tree), ("Ridge_plus_LightGBM", RidgeLGB)):
                p = np.full(len(d), np.nan)
                for tr, te in sp:
                    p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
                interv.append({"dataset": ds, "model": name, "question": q, "rmse_late": float(np.sqrt(np.nanmean((p[late] - y[late]) ** 2)))})
            print(f"[{ds}] intervention {q} done", flush=True)
        # offset persistence
        rng = np.random.default_rng(100)
        u = d["cell"].unique(); rng.shuffle(u)
        fold = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values
        for m in ("Ridge", "LightGBM"):
            res = np.full(len(d), np.nan)
            for k in range(5):
                inf = fold == k
                mod = build(m, DEFAULTS[m]).fit(X[~inf], y[~inf])
                res[inf] = y[inf] - mod.predict(X[inf])
            e = pd.Series(res[~late]).groupby(cells[~late]).mean(); l = pd.Series(res[late]).groupby(cells[late]).mean()
            e, l = e.align(l, join="inner")
            rho, pv = spearmanr(e, l)
            slope = float(np.polyfit(e.values, l.values, 1)[0])
            persist.append({"dataset": ds, "model": m, "cells": len(e), "spearman": float(rho), "p": float(pv), "slope": slope,
                            "sd_early": float(e.std()), "sd_late": float(l.std())})
    Path(out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inr).to_csv(Path(out) / "range_split.csv", index=False)
    pd.DataFrame(interv).to_csv(Path(out) / "extrapolation_intervention.csv", index=False)
    pd.DataFrame(persist).to_csv(Path(out) / "offset_persistence.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(inr).round(2).to_string(index=False)); print(pd.DataFrame(interv).round(2).to_string(index=False)); print(pd.DataFrame(persist).round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:6])
