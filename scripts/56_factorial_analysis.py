"""Step 56 - analysis of the factorial experiment of step 53.
Per dataset and model, on the last 40 % of every cell: pooled RMSE and macro-cell RMSE of the four corners A (c=1, f=0), B (c=0.6, f=0), C (c=1, f=0.6), D (c=0.6, f=0.6)
and of the intermediate coverage c = 0.8; contrasts with a cell-cluster bootstrap (4000 resamples, 95 % percentile intervals):
  coverage_alone   B - A   (loss of late-life training data from other cells, target cell absent)
  target_alone     C - A   (early rows of the target cell, other cells observed over the whole life)
  coverage_with_target  D - C
  target_at_short_pool  D - B
  total            D - A   (new cell -> temporal forecasting; Q3 -> Q5)
  interaction      (B - A) - (D - C)
  share_coverage   0.5 [(B - A) + (D - C)] / (D - A)   (average of the two orders of adding the factors)
Also the share of test rows whose true SOH lies below the lowest SOH of the training rows ('uncovered rows').
Usage: python 56_factorial_analysis.py <step53> <out>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = 4000
rng = np.random.default_rng(56)
MODELS = ["Ridge", "LightGBM", "XGBoost", "RandomForest"]


def main(d53, out):
    rows, con, unc = [], [], []
    for ds in ("matr", "hust", "xjtu", "tju"):
        Z = np.load(Path(d53) / ds / f"{ds}_factorial.npz"); y, late = Z["y"], Z["late"]
        import importlib
        m7 = importlib.import_module("07_rank_experiment")
        d, _ = m7.load(ds); d = d.reset_index(drop=True)
        cu, cinv = np.unique(d["cell"].values, return_inverse=True)
        n_c = np.bincount(cinv, weights=late.astype(float), minlength=len(cu))
        idx = rng.integers(0, len(cu), (B, len(cu)))
        corner = {"A": (1.0, 0.0), "B": (0.6, 0.0), "C": (1.0, 0.6), "D": (0.6, 0.6)}
        for m in MODELS:
            S = {}
            for key in [(c, f) for c in (0.6, 0.8, 1.0) for f in (0.0, 0.6)]:
                e = np.nanmean([(Z[f"{r}|{key[0]}|{key[1]}|{m}"].astype(float) - y) ** 2 for r in (0, 1)], axis=0)
                S[key] = np.bincount(cinv, weights=np.where(late, e, 0.0), minlength=len(cu))
                rows.append({"dataset": ds, "model": m, "c": key[0], "f": key[1],
                             "pooled": float(np.sqrt(S[key].sum() / n_c.sum())), "macro_cell": float(np.mean(np.sqrt(S[key] / np.maximum(n_c, 1))))})
            R = {k: np.sqrt(S[v].sum() / n_c.sum()) for k, v in corner.items()}
            Rb = {k: np.sqrt(S[v][idx].sum(1) / n_c[idx].sum(1)) for k, v in corner.items()}
            Mb = {k: np.mean(np.sqrt(S[v] / np.maximum(n_c, 1))[idx], axis=1) for k, v in corner.items()}
            M0 = {k: float(np.mean(np.sqrt(S[v] / np.maximum(n_c, 1)))) for k, v in corner.items()}
            for scale, pt, bt in (("pooled", R, Rb), ("macro_cell", M0, Mb)):
                fns = {"coverage_alone": lambda x: x["B"] - x["A"], "target_alone": lambda x: x["C"] - x["A"],
                       "coverage_with_target": lambda x: x["D"] - x["C"], "target_at_short_pool": lambda x: x["D"] - x["B"],
                       "total": lambda x: x["D"] - x["A"], "interaction": lambda x: (x["B"] - x["A"]) - (x["D"] - x["C"]),
                       "share_coverage": lambda x: 0.5 * ((x["B"] - x["A"]) + (x["D"] - x["C"])) / (x["D"] - x["A"])}
                for nm, fn in fns.items():
                    b = fn(bt)
                    con.append({"dataset": ds, "model": m, "scale": scale, "contrast": nm, "estimate": float(fn(pt)), "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)})
        # uncovered rows: true SOH of a test row below the minimum SOH of the training rows
        rk = d.groupby("cell")["cycle"].rank(method="first").values; n = d.groupby("cell")["cycle"].transform("size").values
        for rep in (0, 1):
            rng2 = np.random.default_rng(100 + rep); u = d["cell"].unique(); rng2.shuffle(u)
            fold = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values
            for name, (c, f) in {"A": (1.0, 0.0), "B": (0.6, 0.0), "C": (1.0, 0.6), "D": (0.6, 0.6)}.items():
                k_unc = k_n = 0
                for k in range(5):
                    te = (fold == k)
                    tr = (~te & (rk <= np.floor(c * n))) | (te & (rk <= np.floor(f * n)))
                    test = te & late
                    k_unc += int((y[test] < y[tr].min()).sum()); k_n += int(test.sum())
                unc.append({"dataset": ds, "rep": rep, "corner": name, "uncovered_share": k_unc / k_n})
        print(ds, "done", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(out) / "factorial_rmse.csv", index=False)
    C = pd.DataFrame(con); C.to_csv(Path(out) / "factorial_contrasts.csv", index=False)
    pd.DataFrame(unc).groupby(["dataset", "corner"]).uncovered_share.mean().reset_index().to_csv(Path(out) / "factorial_uncovered.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).query("model=='LightGBM' and c in [0.6, 1.0]").pivot_table(index="dataset", columns=["c", "f"], values="pooled").round(2).to_string())
    print(C[(C.model.isin(["LightGBM", "Ridge"])) & (C.scale == "pooled")].round(2).to_string(index=False))
    print(pd.DataFrame(unc).groupby(["dataset", "corner"]).uncovered_share.mean().unstack().round(3))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    main(*sys.argv[1:3])
