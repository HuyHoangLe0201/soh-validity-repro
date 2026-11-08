"""Step 25 - analyse step 22 (calibration with early rows of the target cell).
RMSE on the last 40 % of the fold cells for every model and early fraction f (mean squared error over the 2 partitions first),
cluster bootstrap over cells (4000 resamples) for: LightGBM(f) - LightGBM(0), Ridge(f) - Ridge(0), Ridge - LightGBM at each f,
TargetRidge - LightGBM at each f.   Usage: python 25_calibration_analysis.py <step22_dir> <step7_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 4000
FRACS = [0.0, 0.1, 0.2, 0.4, 0.6]
rng = np.random.default_rng(22)


def stat(cells, sq, w, fn):
    u, inv = np.unique(cells, return_inverse=True)
    S = {k: np.bincount(inv, weights=np.nan_to_num(v) * w[k], minlength=len(u)) for k, v in sq.items()}
    N = {k: np.bincount(inv, weights=w[k], minlength=len(u)) for k in sq}
    pt = fn({k: np.sqrt(S[k].sum() / N[k].sum()) for k in S})
    bs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, len(u), len(u))
        bs[b] = fn({k: np.sqrt(S[k][i].sum() / max(N[k][i].sum(), 1e-9)) for k in S})
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main(d22, d7, out):
    rows, con = [], []
    for ds in sys.argv[4:] or ("matr", "hust", "xjtu", "tju"):
        T = np.load(Path(d22) / ds / f"{ds}_calibration.npz")
        meta = pd.read_csv(Path(d7) / ds / f"{ds}_meta.csv.gz")
        cells, y, late = meta["cell"].values, T["y"], T["late"]
        sq, w = {}, {}
        for f in FRACS:
            for m in ("Ridge", "LightGBM", "Ridge_w", "LightGBM_w", "TargetRidge"):
                k = f"0|{f}|{m}"
                if k not in T.files:
                    continue
                e = np.nanmean([(T[f"{r}|{f}|{m}"].astype(float) - y) ** 2 for r in (0, 1)], axis=0)
                ok = late & ~np.isnan(e)
                sq[(f, m)], w[(f, m)] = np.where(ok, e, 0.0), ok.astype(float)
        for (f, m) in sq:
            pt, lo, hi = stat(cells, {"a": sq[(f, m)]}, {"a": w[(f, m)]}, lambda r: r["a"])
            rows.append({"dataset": ds, "frac": f, "model": m, "rmse": pt, "lo": lo, "hi": hi})

        def diff(a, b, label):
            if a not in sq or b not in sq:
                return
            ww = {"a": w[a] * w[b], "b": w[a] * w[b]}
            pt, lo, hi = stat(cells, {"a": sq[a], "b": sq[b]}, ww, lambda r: r["a"] - r["b"])
            con.append({"dataset": ds, "contrast": label, "diff": pt, "lo": lo, "hi": hi})
        for f in FRACS[1:]:
            diff((f, "LightGBM"), (0.0, "LightGBM"), f"LightGBM f={f} minus f=0")
            diff((f, "Ridge"), (0.0, "Ridge"), f"Ridge f={f} minus f=0")
        for f in FRACS:
            diff((f, "Ridge"), (f, "LightGBM"), f"Ridge minus LightGBM f={f}")
        for f in FRACS[1:]:
            diff((f, "TargetRidge"), (f, "LightGBM"), f"TargetRidge minus LightGBM f={f}")
            diff((f, "LightGBM_w"), (f, "LightGBM"), f"LightGBM_w minus LightGBM f={f}")
    Path(out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(out) / "calibration_rmse.csv", index=False)
    pd.DataFrame(con).to_csv(Path(out) / "calibration_contrasts.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).pivot_table(index=["dataset", "model"], columns="frac", values="rmse").round(2))
    print(pd.DataFrame(con).round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
