"""Step 17 - optimism of selecting hyperparameters on the folds that are reported.

For one repetition of P0 and P2 (seed 0) and for P3, every grid configuration is trained on the training part of each
outer fold (all training rows) and evaluated on the outer test part.  Reported are the pooled RMSE of
  default      : fixed default hyperparameters
  nested       : inner-CV selection on the training part only (step 7 / step 9 predictions, rep 0)
  best_global  : the single configuration with the lowest pooled test RMSE over all folds (selection on the reported folds)
  best_fold    : the configuration chosen separately for each fold by its own test error (oracle)
Usage: python -u 17_selection_optimism.py <out_dir>"""
import importlib, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import rankexp
m7 = importlib.import_module("07_rank_experiment")
R = Path(__file__).resolve().parent.parent / "results"
def pooled(store, d, label, regime, proto, model):
    k = f"{label}|{regime}|{proto}|{model}|0"
    return float(np.sqrt(np.nanmean((store[k].astype(float) - d[label].values) ** 2))) if k in store else np.nan
if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); rows = []
    for ds in ("xjtu", "hust", "matr"):
        d, feats = m7.load(ds); y = d["SOH_rated"].values; X = d[feats].values.astype(float)
        s7 = dict(np.load(R / "step7" / ds / f"{ds}_preds.npz")); s9 = dict(np.load(R / "step9" / ds / f"{ds}_tunedgrp_preds.npz"))
        for proto in ("P0", "P2", "P3"):
            folds = list(rankexp.folds(d, proto, 0))
            for model in m7.FAST4:
                grid = rankexp.GRIDS[model]; sse = np.zeros((len(folds), len(grid))); n = np.array([len(te) for _, te in folds]); dflt = 0.0
                for fi, (tr, te) in enumerate(folds):
                    for ci, p in enumerate(grid):
                        pr = rankexp.build(model, p).fit(X[tr], y[tr]).predict(X[te]); sse[fi, ci] = float(np.sum((pr - y[te]) ** 2))
                    pr = rankexp.build(model, rankexp.DEFAULTS[model]).fit(X[tr], y[tr]).predict(X[te]); dflt += float(np.sum((pr - y[te]) ** 2))
                rows.append({"dataset": ds, "protocol": proto, "model": model, "folds": len(folds), "n_configs": len(grid),
                             "default": np.sqrt(dflt / n.sum()), "nested_cell_inner": pooled(s7, d, "SOH_rated", "tuned", proto, model),
                             "nested_group_inner": pooled(s9, d, "SOH_rated", "tuned_grp", proto, model) if proto == "P3" else np.nan,
                             "best_global": np.sqrt(sse.sum(0).min() / n.sum()), "best_fold": np.sqrt(sse.min(1).sum() / n.sum()),
                             "grid_median": np.sqrt(np.median(sse.sum(0)) / n.sum()), "grid_worst": np.sqrt(sse.sum(0).max() / n.sum())})
                print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(out / "selection_optimism.csv", index=False)
