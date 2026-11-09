"""Step 61 - sensitivity of the temporal result to the training proportion.
For the training share s in {0.4, 0.5, 0.6, 0.7, 0.8} of every cell's life (first s x n rows of each cell pooled), LightGBM and ridge regression (default settings, F1, rated label) are
trained and scored on the remaining rows of all cells (PT).  P0, P2 and P3 predictions of step 7 (every row is predicted out of fold) are scored on the same rows.
Usage: python -u 61_temporal_proportions.py <step7_dir> <out_dir>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import DEFAULTS, build  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")
S = [0.4, 0.5, 0.6, 0.7, 0.8]
MODELS = ["Ridge", "LightGBM"]

if __name__ == "__main__":
    d7, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); rows = []
    for ds in ("xjtu", "tju", "hust", "matr"):
        d, feats = m7.load(ds); d = d.reset_index(drop=True)
        X, y = d[feats].values.astype(float), d["SOH_rated"].values
        rk = d.groupby("cell")["cycle"].rank(method="first").values; n = d.groupby("cell")["cycle"].transform("size").values
        st = dict(np.load(d7 / ds / f"{ds}_preds.npz"))
        for s in S:
            tr = rk <= np.floor(s * n); te = ~tr
            for m in MODELS:
                pt = build(m, DEFAULTS[m]).fit(X[tr], y[tr]).predict(X[te])
                row = {"dataset": ds, "share": s, "model": m, "test_rows": int(te.sum()), "PT": float(np.sqrt(np.mean((pt - y[te]) ** 2)))}
                for q in ("P0", "P2", "P3"):
                    ks = [k for k in st if k.startswith(f"SOH_rated|default|{q}|{m}|")]
                    e = np.nanmean([(st[k].astype(float) - y) ** 2 for k in ks], axis=0)
                    row[q] = float(np.sqrt(np.mean(e[te])))
                rows.append(row); print(row, flush=True)
        pd.DataFrame(rows).to_csv(out / "temporal_proportions.csv", index=False)
