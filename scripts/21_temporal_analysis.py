"""Step 21 - score temporal and random within-cell splits and compare them with P0 / P2 / P3 on the same late rows.

Rows scored: the last 40 % of each cell (the PT test rows).  P0, P2, P3 (step 7, default regime, F1, rated label) are
re-scored on exactly these rows; PR is the random 60/40 split within cells (its test rows are random, so its RMSE is over its own test rows).
Uncertainty: paired cluster bootstrap over cells (4000 resamples); difference Ridge - LightGBM and PT/other ratios.
Usage: python 21_temporal_analysis.py <step7_dir> <step20_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 4000
MODELS = ["Ridge", "LightGBM", "XGBoost", "RandomForest"]
rng = np.random.default_rng(20)


def prot_err(store, label, protocol, m, y):
    keys = [k for k in store if k.startswith(f"{label}|default|{protocol}|{m}|")]
    return np.nanmean(np.vstack([(store[k].astype(float) - y) ** 2 for k in keys]), axis=0)


def boot(cells_idx, num, den, fn):
    """cells_idx: array of unit ids per row; num/den: dict name->per-row sq err / row weight; fn maps pooled rmse dict -> stat."""
    u, inv = np.unique(cells_idx, return_inverse=True)
    S = {k: np.bincount(inv, weights=v, minlength=len(u)) for k, v in num.items()}
    N = {k: np.bincount(inv, weights=v, minlength=len(u)) for k, v in den.items()}
    pt = fn({k: np.sqrt(S[k].sum() / N[k].sum()) for k in S})
    bs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, len(u), len(u))
        bs[b] = fn({k: np.sqrt(S[k][i].sum() / N[k][i].sum()) for k in S})
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(u)


def main(d7, d20, out):
    rows, diffs = [], []
    for ds in ("matr", "hust", "xjtu", "tju"):
        meta = pd.read_csv(Path(d7) / ds / f"{ds}_meta.csv.gz")
        T = np.load(Path(d20) / ds / f"{ds}_temporal.npz")
        st = dict(np.load(Path(d7) / ds / f"{ds}_preds.npz"))
        y, late, cells = T["y"], T["late"], meta["cell"].values
        assert np.allclose(y, meta["SOH_rated"].values)
        rng_note = (T["y_train_mean_PT"][0], T["y_train_mean_PT"][1])
        sq = {}
        for m in MODELS:
            sq[("PT", m)] = np.where(late, (T[f"PT|{m}"].astype(float) - y) ** 2, 0.0)
            for p in ("P0", "P2", "P3"):
                sq[(p, m)] = np.where(late, prot_err(st, "SOH_rated", p, m, y), 0.0)
            # random 60/40 within cells: its own test rows
            prs = []
            for s in range(3):
                te = T[f"PR{s}|mask"]
                prs.append(np.where(te, (T[f"PR{s}|{m}"].astype(float) - y) ** 2, 0.0))
            sq[("PR", m)] = np.mean(prs, axis=0)
        w_late = late.astype(float)
        w_pr = np.mean([T[f"PR{s}|mask"].astype(float) for s in range(3)], axis=0)
        for m in MODELS:
            for p in ("PR", "PT", "P0", "P2", "P3"):
                w = w_pr if p == "PR" else w_late
                pt, lo, hi, k = boot(cells, {"a": sq[(p, m)]}, {"a": w}, lambda r: r["a"])
                rows.append({"dataset": ds, "model": m, "protocol": p, "rmse": pt, "lo": lo, "hi": hi, "cells": k,
                             "soh_train_mean": rng_note[0], "soh_late_mean": rng_note[1]})
        # model contrasts: Ridge - LightGBM per protocol
        for p in ("PR", "PT", "P0", "P2", "P3"):
            w = w_pr if p == "PR" else w_late
            pt, lo, hi, k = boot(cells, {"R": sq[(p, "Ridge")], "L": sq[(p, "LightGBM")]}, {"R": w, "L": w}, lambda r: r["R"] - r["L"])
            diffs.append({"dataset": ds, "protocol": p, "ridge_minus_lgbm": pt, "lo": lo, "hi": hi, "cells": k})
        # temporal / random ratio for LightGBM and Ridge
        for m in ("Ridge", "LightGBM"):
            pt, lo, hi, k = boot(cells, {"T": sq[("PT", m)], "R": sq[("PR", m)]}, {"T": w_late, "R": w_pr}, lambda r: r["T"] / r["R"])
            diffs.append({"dataset": ds, "protocol": f"PT/PR {m}", "ridge_minus_lgbm": pt, "lo": lo, "hi": hi, "cells": k})
    Path(out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(out) / "temporal_rmse.csv", index=False)
    pd.DataFrame(diffs).to_csv(Path(out) / "temporal_contrasts.csv", index=False)
    pd.set_option("display.width", 200)
    r = pd.DataFrame(rows)
    print(r.pivot_table(index=["dataset", "model"], columns="protocol", values="rmse")[["PR", "PT", "P0", "P2", "P3"]].round(2))
    print(pd.DataFrame(diffs).round(2).to_string(index=False))
    print(r.groupby("dataset")[["soh_train_mean", "soh_late_mean"]].first())


if __name__ == "__main__":
    main(*sys.argv[1:4])


def range_table(d20, out):
    """Share of late rows whose true SOH is below the lowest SOH seen in training (PT), and mean true / predicted SOH."""
    rows = []
    for ds in ("matr", "hust", "xjtu", "tju"):
        T = np.load(Path(d20) / ds / f"{ds}_temporal.npz")
        y, late = T["y"], T["late"]
        rows.append({"dataset": ds, "late_below_train_min": float((y[late] < y[~late].min()).mean()),
                     "train_min": float(y[~late].min()), "late_mean": float(y[late].mean()),
                     "lgbm_pred_mean": float(np.nanmean(T["PT|LightGBM"][late])), "ridge_pred_mean": float(np.nanmean(T["PT|Ridge"][late])),
                     "lgbm_pred_min": float(np.nanmin(T["PT|LightGBM"][late]))})
    pd.DataFrame(rows).to_csv(Path(out) / "temporal_range.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__" and len(sys.argv) > 3:
    range_table(sys.argv[2], sys.argv[3])
