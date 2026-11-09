"""Step 54 - macro-averaged errors: every cell (or held-out group) weighs the same, instead of every cycle.

Pooled RMSE weights a cell by its number of cycles.  Here, per question (P0, P2, P3, PT), model and dataset, the following are computed on the last 40 % of the
rows of each cell (the rows scored for all four questions, as in step 33):
  pooled        RMSE over all scored rows
  macro_cell    mean over cells of the per-cell RMSE
  median_cell   median over cells of the per-cell RMSE
  macro_group   (P3 only) mean over the held-out groups of the per-group RMSE
Also computed: agreement of the model order (Spearman) between pooled and macro_cell RMSE, pairwise sign agreement for model pairs, and paired
cluster-bootstrap (unit = cell, 4000 resamples) intervals for key contrasts under macro_cell RMSE:
  PT - P3 per model, P3 / P0 ratio per model, Ridge - LightGBM per question.
Usage: python 54_macro_rmse.py <step7> <step20> <step26> <step29> <step31> <out> <step36>
"""
import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
g33 = importlib.import_module("33_group_inference")
QS = g33.QS
B = 4000
rng = np.random.default_rng(54)


def per_unit_rmse(E, mask, inv, k):
    s = np.bincount(inv, weights=np.where(mask, E, 0.0), minlength=k)
    n = np.bincount(inv, weights=mask.astype(float), minlength=k)
    ok = n > 0
    return np.where(ok, np.sqrt(s / np.maximum(n, 1)), np.nan)


def main(d7, d20, d26, d29, d31, out, d36):
    rows, agree, contr = [], [], []
    for ds in ("matr", "hust", "xjtu", "tju"):
        meta, late, E = g33.errors(ds, d7, d20, d26, d29, d31, d36)
        cu, cinv = np.unique(meta["cell"].values, return_inverse=True)
        gu, ginv = np.unique(meta["grp"].values, return_inverse=True)
        R, RA = {}, {}   # (q, model) -> per-cell rmse on late rows / on all rows
        for q in QS:
            models = [m for m in E[q] if not np.isnan(E[q][m][late]).any()]
            for m in models:
                rc = per_unit_rmse(E[q][m], late, cinv, len(cu)); R[(q, m)] = rc
                if q != "PT":
                    RA[(q, m)] = per_unit_rmse(E[q][m], np.ones(len(late), bool), cinv, len(cu))
                rg = per_unit_rmse(E[q][m], late, ginv, len(gu))
                rows.append({"dataset": ds, "question": q, "model": m, "pooled": float(np.sqrt(np.mean(E[q][m][late]))),
                             "macro_cell": float(np.nanmean(rc)), "median_cell": float(np.nanmedian(rc)),
                             "macro_group": float(np.nanmean(rg)) if q == "P3" else np.nan,
                             "pooled_all_rows": float(np.sqrt(np.nanmean(E[q][m]))) if q != "PT" else np.nan,
                             "macro_cell_all_rows": float(np.nanmean(RA[(q, m)])) if q != "PT" else np.nan})
        D = pd.DataFrame([r for r in rows if r["dataset"] == ds])
        for q in QS:
            s = D[D.question == q]
            rho = spearmanr(s.pooled, s.macro_cell)[0]
            n_pairs = flips = 0
            for a, b in itertools.combinations(s.index, 2):
                n_pairs += 1
                flips += int((s.pooled[a] - s.pooled[b]) * (s.macro_cell[a] - s.macro_cell[b]) < 0)
            agree.append({"dataset": ds, "question": q, "models": len(s), "spearman": rho, "pairs": n_pairs, "pair_flips": flips})
        # bootstrap contrasts over cells
        idx = rng.integers(0, len(cu), (B, len(cu)))
        def boot(vec):
            return np.nanmean(vec[idx], axis=1)
        for m in sorted({m for (_, m) in R}):
            if all((q, m) in R for q in ("P3", "PT")):
                dl = R[("PT", m)] - R[("P3", m)]
                bs = boot(dl)
                contr.append({"dataset": ds, "contrast": "PT-P3", "model": m, "estimate": float(np.nanmean(dl)), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
            if all((q, m) in R for q in ("P0", "P3")):
                b0, b3 = boot(R[("P0", m)]), boot(R[("P3", m)])
                ratio = b3 / b0
                contr.append({"dataset": ds, "contrast": "P3/P0", "model": m, "estimate": float(np.nanmean(R[("P3", m)]) / np.nanmean(R[("P0", m)])), "lo": np.percentile(ratio, 2.5), "hi": np.percentile(ratio, 97.5)})
        for m in sorted({m for (_, m) in RA}):
            if ("P0", m) in RA and ("P3", m) in RA:
                ratio = boot(RA[("P3", m)]) / boot(RA[("P0", m)])
                contr.append({"dataset": ds, "contrast": "P3/P0 all rows", "model": m, "estimate": float(np.nanmean(RA[("P3", m)]) / np.nanmean(RA[("P0", m)])), "lo": np.percentile(ratio, 2.5), "hi": np.percentile(ratio, 97.5)})
        for q in QS:
            for a, b in itertools.combinations([m for m in ("Ridge", "LightGBM", "GRU", "TRF") if (q, m) in R], 2):
                dl = R[(q, a)] - R[(q, b)]
                bs = boot(dl)
                contr.append({"dataset": ds, "contrast": f"{q}: {a}-{b}", "model": f"{a}-{b}", "estimate": float(np.nanmean(dl)), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
        print(f"[{ds}] done", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(out) / "macro_rmse.csv", index=False)
    pd.DataFrame(agree).to_csv(Path(out) / "macro_agreement.csv", index=False)
    pd.DataFrame(contr).to_csv(Path(out) / "macro_contrasts.csv", index=False)
    print(pd.DataFrame(agree).round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:8])
