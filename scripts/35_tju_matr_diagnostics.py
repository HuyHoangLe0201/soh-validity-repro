"""Step 35 - two open explanations.
A. Why is the temporal forecast poor on TJU although no test row lies below the training range?
   (i) per-group and per-cell concentration of the LightGBM temporal error, (ii) standardized shift of each F1 feature between the training
   rows and the test rows, (iii) intervention: add the cycle number as an input (LightGBM and ridge), temporal question and P2/P3.
B. Why does the per-cell offset hurt on MATR?  Per cell: late RMSE of the base model and after the median-early-residual offset (cell-disjoint folds,
   partition seed 100, LightGBM); share of cells improved, gain by cell length tercile, and share of the pooled squared error carried by the longest cells.
Usage: python 35_tju_matr_diagnostics.py <step20> <out_dir>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import DEFAULTS, build, folds  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")


def main(d20, out):
    conc, shift, cyc, offc = [], [], [], []
    for ds in ("matr", "hust", "xjtu", "tju"):
        d, feats = m7.load(ds)
        d = d.reset_index(drop=True)
        X, y = d[feats].values.astype(float), d["SOH_rated"].values
        cells, grp = d["cell"].values, d["grp"].values
        rk = d.groupby("cell")["cycle"].rank(method="first").values
        n = d.groupby("cell")["cycle"].transform("size").values
        late = rk > np.floor(0.6 * n)
        T = np.load(Path(d20) / ds / f"{ds}_temporal.npz")
        # A(i) concentration
        e = np.where(late, (T["PT|LightGBM"].astype(float) - y) ** 2, 0)
        pc = pd.DataFrame({"c": cells, "g": grp, "e": e, "l": late.astype(int)}).groupby("c").agg(e=("e", "sum"), n=("l", "sum"), g=("g", "first"))
        pc["rmse"] = np.sqrt(pc.e / pc.n)
        top = pc.e.sort_values(ascending=False)
        conc.append({"dataset": ds, "cells": len(pc), "top10pct_cells_share_of_sse": float(top.iloc[:max(1, len(pc) // 10)].sum() / top.sum()),
                     "median_cell_rmse": float(pc.rmse.median()), "p90_cell_rmse": float(pc.rmse.quantile(.9))})
        if ds == "tju":
            pg = pc.groupby("g").agg(e=("e", "sum"), n=("n", "sum"), cells=("e", "size")); pg["rmse"] = np.sqrt(pg.e / pg.n)
            pg.reset_index().to_csv(Path(out) / "tju_group_temporal_error.csv", index=False)
        # A(ii) feature shift
        Xtr, Xte = X[~late], X[late]
        sd = np.nanstd(Xtr, axis=0) + 1e-9
        for j, f in enumerate(feats):
            shift.append({"dataset": ds, "feature": f, "std_shift": float((np.nanmean(Xte[:, j]) - np.nanmean(Xtr[:, j])) / sd[j])})
        # A(iii) intervention: cycle number as input
        Xc = np.column_stack([X, d["cycle"].values.astype(float)])
        for q in ("P2", "P3", "PT"):
            sp = [(tr, te) for tr, te in folds(d, q, 0)] if q != "PT" else [(np.where(~late)[0], np.where(late)[0])]
            for m in ("Ridge", "LightGBM"):
                p0, p1 = np.full(len(d), np.nan), np.full(len(d), np.nan)
                for tr, te in sp:
                    p0[te] = build(m, DEFAULTS[m]).fit(X[tr], y[tr]).predict(X[te])
                    p1[te] = build(m, DEFAULTS[m]).fit(Xc[tr], y[tr]).predict(Xc[te])
                cyc.append({"dataset": ds, "model": m, "question": q, "rmse_F1": float(np.sqrt(np.nanmean((p0[late] - y[late]) ** 2))),
                            "rmse_F1_plus_cycle": float(np.sqrt(np.nanmean((p1[late] - y[late]) ** 2)))})
        print(f"[{ds}] A done", flush=True)
        # B: offset by cell
        rng = np.random.default_rng(100)
        u = d["cell"].unique(); rng.shuffle(u)
        fold = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values
        res = np.full(len(d), np.nan)
        for k in range(5):
            inf = fold == k
            res[inf] = y[inf] - build("LightGBM", DEFAULTS["LightGBM"]).fit(X[~inf], y[~inf]).predict(X[inf])
        s = pd.DataFrame({"c": cells, "r": res, "late": late})
        med = s[~s.late].groupby("c").r.median()
        rl = s[s.late].copy(); rl["adj"] = rl.r - rl.c.map(med).fillna(0)
        g = rl.groupby("c").agg(base=("r", lambda x: float(np.sqrt(np.mean(x ** 2)))), adj=("adj", lambda x: float(np.sqrt(np.mean(x ** 2)))), nlate=("r", "size"))
        g["improved"] = g.adj < g.base
        terc = pd.qcut(g.nlate, 3, labels=["short", "middle", "long"])
        for t_, gg in g.groupby(terc, observed=True):
            offc.append({"dataset": ds, "length_tercile": str(t_), "cells": len(gg), "share_improved": float(gg.improved.mean()),
                         "mse_base": float((gg.base ** 2 * gg.nlate).sum() / gg.nlate.sum()), "mse_adj": float((gg.adj ** 2 * gg.nlate).sum() / gg.nlate.sum()),
                         "share_of_late_rows": float(gg.nlate.sum() / g.nlate.sum())})
        print(f"[{ds}] B done", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(conc).to_csv(Path(out) / "temporal_concentration.csv", index=False)
    pd.DataFrame(shift).to_csv(Path(out) / "feature_shift.csv", index=False)
    pd.DataFrame(cyc).to_csv(Path(out) / "cycle_number_intervention.csv", index=False)
    pd.DataFrame(offc).to_csv(Path(out) / "offset_by_cell_length.csv", index=False)
    pd.set_option("display.width", 220)
    for n_ in ("temporal_concentration", "cycle_number_intervention", "offset_by_cell_length"):
        print(pd.read_csv(Path(out) / f"{n_}.csv").round(3).to_string(index=False))
    fs = pd.DataFrame(shift); print(fs[fs.dataset == "tju"].round(2).to_string(index=False))


if __name__ == "__main__":
    Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
    main(*sys.argv[1:3])
