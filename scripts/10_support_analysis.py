"""Step 10 - is the protocol effect carried by how far held-out rows lie from the training data?

Outcome-blind support measure (computed from the FEATURES only):
  r(row) = mean distance of the row to its 10 nearest training rows (standardised, median-imputed
           features) / median of the same quantity for reference training rows (which have same-cell
           neighbours in the training subsample).  r ~ 1  => the row looks like the training data;
           r >> 1 => it lies away from the training data.
Rows are binned by TERCILES of r computed over the P3 (leave-one-group-out) rows of the same dataset
(thresholds fixed from the feature-based r only), then applied to P0 / P2 / P3 alike.
Per bin, the pooled RMSE of each model and the paired difference between two models are reported with a
cell-cluster bootstrap.  Secondary flag: label outside the training label range [q0.5, q99.5].

Usage: python -u 10_support_analysis.py <step7_dir> <step9_dir> <out_dir>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import folds  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")
RNG = np.random.default_rng(0)
K, N_S, N_Q, N_BOOT = 10, 20000, 3000, 1000
MODELS = ["Ridge", "LightGBM", "XGBoost", "RandomForest"]
LABEL = "SOH_rated"


def support_ratio(d, feats, protocol):
    X = d[feats].values.astype(float)
    r = np.full(len(d), np.nan)
    out_lab = np.zeros(len(d), bool)
    for tr, te in folds(d, protocol, 0):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        Z = lambda A: sc.transform(imp.transform(A))  # noqa: E731
        perm = RNG.permutation(tr)
        S, Q = perm[:min(N_S, len(perm) - N_Q)], perm[-N_Q:]
        nn = NearestNeighbors(n_neighbors=K).fit(Z(X[S]))
        d_ref = np.median(nn.kneighbors(Z(X[Q]))[0].mean(axis=1))
        r[te] = nn.kneighbors(Z(X[te]))[0].mean(axis=1) / d_ref
        y = d[LABEL].values
        lo, hi = np.percentile(y[tr], [0.5, 99.5])
        out_lab[te] = (y[te] < lo) | (y[te] > hi)
    return r, out_lab


def boot_delta(e2A, e2B, cells, n_boot=N_BOOT):
    df = pd.DataFrame({"c": cells, "a": e2A, "b": e2B}).groupby("c").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    a, b, n = df["a"].values, df["b"].values, df["n"].values
    point = np.sqrt(a.sum() / n.sum()) - np.sqrt(b.sum() / n.sum())
    k = len(n)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        j = RNG.integers(0, k, k)
        bs[i] = np.sqrt(a[j].sum() / n[j].sum()) - np.sqrt(b[j].sum() / n[j].sum())
    return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5), k


def main(dir7, dir9, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, strata = [], []
    for ds in ("matr", "matr_crate", "hust", "xjtu"):
        d, feats = m7.load(ds)
        y = d[LABEL].values
        base = ds if ds != "matr_crate" else "matr"
        store = dict(np.load(Path(dir7) / base / f"{base}_preds.npz"))
        if ds == "matr_crate":
            store.update(dict(np.load(Path(dir7) / ds / f"{ds}_preds.npz")))
        g9 = Path(dir9) / ds / f"{ds}_tunedgrp_preds.npz"
        store9 = dict(np.load(g9)) if g9.exists() else {}
        R, L = {}, {}
        for p in ("P0", "P2", "P3"):
            if ds == "matr_crate" and p != "P3":
                R[p], L[p] = R_base[p], L_base[p]  # noqa: F821  (same rows/folds as matr)
            else:
                R[p], L[p] = support_ratio(d, feats, p)
        if ds == "matr":
            R_base, L_base = dict(R), dict(L)
        q1, q2 = np.nanpercentile(R["P3"], [33.3, 66.7])
        bins = {p: np.digitize(R[p], [q1, q2]) for p in R}
        print(f"{ds}: tercile thresholds of r (P3): {q1:.2f}, {q2:.2f}", flush=True)
        for p in R:
            share = np.bincount(bins[p], minlength=3) / len(d)
            strata.append({"dataset": ds, "protocol": p, "q33": q1, "q67": q2,
                           "median_r": float(np.nanmedian(R[p])), "share_low": share[0],
                           "share_mid": share[1], "share_high": share[2],
                           "share_label_outside": float(L[p].mean())})
        for regime, st in (("default", store), ("tuned", store), ("tuned_grp", store9)):
            for p in R:
                E = {}
                for m in MODELS:
                    k = f"{LABEL}|{regime}|{p}|{m}|0"
                    if k in st:
                        E[m] = (st[k].astype(float) - y) ** 2
                if len(E) < 2:
                    continue
                for b in (0, 1, 2, "label_out", "label_in"):
                    mask = (bins[p] == b) if isinstance(b, int) else (L[p] if b == "label_out" else ~L[p])
                    if mask.sum() < 500 or d.cell[mask].nunique() < 5:
                        continue
                    rec = {"dataset": ds, "regime": regime, "protocol": p, "stratum": str(b),
                           "rows": int(mask.sum()), "cells": int(d.cell[mask].nunique())}
                    for m, e in E.items():
                        rec[f"rmse_{m}"] = float(np.sqrt(e[mask].mean()))
                    for A, B in (("Ridge", "LightGBM"), ("XGBoost", "RandomForest")):
                        if A in E and B in E:
                            dl, lo, hi, _ = boot_delta(E[A][mask], E[B][mask], d.cell.values[mask])
                            rec.update({f"d_{A}_minus_{B}": dl, f"lo_{A}_{B}": lo, f"hi_{A}_{B}": hi})
                    rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(out / "support_strata.csv", index=False)
    pd.DataFrame(strata).to_csv(out / "support_distribution.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n== Where do the rows of each protocol sit? (share in tercile bins of r; P3 terciles)")
    print(pd.DataFrame(strata).round(2).to_string(index=False))
    cols = ["dataset", "regime", "protocol", "stratum", "rows", "cells", "rmse_Ridge", "rmse_LightGBM",
            "d_Ridge_minus_LightGBM", "lo_Ridge_LightGBM", "hi_Ridge_LightGBM"]
    print("\n== Ridge vs LightGBM by support stratum (default regime); delta<0 => Ridge better")
    print(res[res.regime == "default"][cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
