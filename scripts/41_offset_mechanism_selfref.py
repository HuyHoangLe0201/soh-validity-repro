"""Step 41 - (A) mechanism of the offset results on MATR; (B) self-referenced inputs as an intervention for the temporal question.
A. LightGBM trained outside cell-disjoint folds (partition seed 100).  Per cell: mean residual on the first 60 % of the rows (early) and on the last 40 % (late).
   By terciles of the cycle life of the cell (MATR): Spearman correlation early-late, regression slope, share of cells whose early and late residual have the same sign,
   and the Spearman correlation between the late residual and log cycle life.
B. Self-referenced inputs: every F1 feature minus its median over the first 10 cycles of the same cell, added to F1 (F1+SR); ridge and LightGBM under P2, P3, PT.
Usage: python 41_offset_mechanism_selfref.py <out_dir>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lightgbm as lgb  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from rankexp import DEFAULTS, folds  # noqa: E402


def build(name, p):  # single-threaded versions so that the script can run next to a torch job
    if name == "Ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=p["alpha"]))
    return lgb.LGBMRegressor(num_leaves=31, random_state=42, verbose=-1, n_jobs=1, **p)


m7 = importlib.import_module("07_rank_experiment")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
mech, sr = [], []
for ds in ("matr", "hust", "xjtu", "tju"):
    d, feats = m7.load(ds)
    d = d.reset_index(drop=True)
    X, y = d[feats].values.astype(float), d["SOH_rated"].values
    cells = d["cell"].values
    rk = d.groupby("cell")["cycle"].rank(method="first").values
    n = d.groupby("cell")["cycle"].transform("size").values
    late = rk > np.floor(0.6 * n)
    # A
    rng = np.random.default_rng(100)
    u = d["cell"].unique(); rng.shuffle(u)
    fold = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values
    res = np.full(len(d), np.nan)
    for k in range(5):
        inf = fold == k
        res[inf] = y[inf] - build("LightGBM", DEFAULTS["LightGBM"]).fit(X[~inf], y[~inf]).predict(X[inf])
    s = pd.DataFrame({"c": cells, "r": res, "late": late, "n": n})
    e = s[~s.late].groupby("c").r.mean(); l = s[s.late].groupby("c").r.mean(); ln = s.groupby("c").n.first()
    life = ln if ds != "matr" else d.groupby("cell").cycle_life.first()
    g = pd.DataFrame({"e": e, "l": l, "life": life}).dropna()
    terc = pd.qcut(g.life, 3, labels=["short", "middle", "long"])
    for t_, gg in g.groupby(terc, observed=True):
        rho = spearmanr(gg.e, gg.l)[0]
        mech.append({"dataset": ds, "life_tercile": str(t_), "cells": len(gg), "spearman_early_late": float(rho), "slope": float(np.polyfit(gg.e, gg.l, 1)[0]),
                     "same_sign_share": float((np.sign(gg.e) == np.sign(gg.l)).mean()), "mean_early": float(gg.e.mean()), "mean_late": float(gg.l.mean())})
    mech.append({"dataset": ds, "life_tercile": "all: late residual vs log life", "cells": len(g), "spearman_early_late": float(spearmanr(g.l, np.log(g.life))[0]),
                 "slope": np.nan, "same_sign_share": np.nan, "mean_early": np.nan, "mean_late": np.nan})
    # B
    ref = np.zeros_like(X)
    for c, idx in d.groupby("cell").indices.items():
        o = idx[np.argsort(d["cycle"].values[idx])][:10]
        ref[idx] = np.nanmedian(X[o], axis=0)
    Xs = np.column_stack([X, X - ref])
    for q in ("P2", "P3", "PT"):
        sp = [(tr, te) for tr, te in folds(d, q, 0)] if q != "PT" else [(np.where(~late)[0], np.where(late)[0])]
        for m in ("Ridge", "LightGBM"):
            p0, p1 = np.full(len(d), np.nan), np.full(len(d), np.nan)
            for tr, te in sp:
                p0[te] = build(m, DEFAULTS[m]).fit(X[tr], y[tr]).predict(X[te])
                p1[te] = build(m, DEFAULTS[m]).fit(Xs[tr], y[tr]).predict(Xs[te])
            sr.append({"dataset": ds, "model": m, "question": q, "rmse_F1": float(np.sqrt(np.nanmean((p0[late] - y[late]) ** 2))),
                       "rmse_F1_plus_selfref": float(np.sqrt(np.nanmean((p1[late] - y[late]) ** 2)))})
    print(f"[{ds}] done", flush=True)
    pd.DataFrame(mech).to_csv(out / "offset_mechanism.csv", index=False); pd.DataFrame(sr).to_csv(out / "selfref_intervention.csv", index=False)
pd.DataFrame(mech).to_csv(out / "offset_mechanism.csv", index=False)
pd.DataFrame(sr).to_csv(out / "selfref_intervention.csv", index=False)
pd.set_option("display.width", 200)
print(pd.DataFrame(mech).round(2).to_string(index=False)); print(pd.DataFrame(sr).round(2).to_string(index=False))
