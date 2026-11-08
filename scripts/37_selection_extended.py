"""Step 37 - selection on the reported folds (cf. step 17) with two repetitions of P0 and P2 and with TJU.
For P0 and P2 repetitions 0 and 1 (the repetitions with nested predictions in step 7) and for P3, every grid configuration is trained on every outer fold;
reported: default, nested (cell-held-out inner folds; group-held-out inner folds for P3), best_global (selection on the reported folds).
Usage: python -u 37_selection_extended.py <out_dir>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import rankexp  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")
R = Path(__file__).resolve().parent.parent / "results"


def pooled(store, d, regime, proto, model, rep):
    k = f"SOH_rated|{regime}|{proto}|{model}|{rep}"
    return float(np.sqrt(np.nanmean((store[k].astype(float) - d["SOH_rated"].values) ** 2))) if k in store else np.nan


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); rows = []
    for ds in ("tju", "xjtu", "hust", "matr"):
        d, feats = m7.load(ds); y = d["SOH_rated"].values; X = d[feats].values.astype(float)
        s7 = dict(np.load(R / "step7" / ds / f"{ds}_preds.npz")); f9 = R / "step9" / ds / f"{ds}_tunedgrp_preds.npz"
        s9 = dict(np.load(f9)) if f9.exists() else {}
        for proto, reps in (("P0", (0, 1)), ("P2", (0, 1)), ("P3", (0,))):
            for rep in reps:
                fl = list(rankexp.folds(d, proto, rep))
                for model in m7.FAST4:
                    grid = rankexp.GRIDS[model]; sse = np.zeros((len(fl), len(grid))); n = np.array([len(te) for _, te in fl]); dflt = 0.0
                    for fi, (tr, te) in enumerate(fl):
                        for ci, p in enumerate(grid):
                            sse[fi, ci] = float(np.sum((rankexp.build(model, p).fit(X[tr], y[tr]).predict(X[te]) - y[te]) ** 2))
                        dflt += float(np.sum((rankexp.build(model, rankexp.DEFAULTS[model]).fit(X[tr], y[tr]).predict(X[te]) - y[te]) ** 2))
                    rows.append({"dataset": ds, "protocol": proto, "rep": rep, "model": model, "default": np.sqrt(dflt / n.sum()),
                                 "nested_cell_inner": pooled(s7, d, "tuned", proto, model, rep),
                                 "nested_group_inner": pooled(s9, d, "tuned_grp", proto, model, 0) if proto == "P3" else np.nan,
                                 "best_global": np.sqrt(sse.sum(0).min() / n.sum())})
                    print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(out / "selection_extended.csv", index=False)
