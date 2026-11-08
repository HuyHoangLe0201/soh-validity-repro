"""Step 18 - preprocessing fitted before splitting (imputer + standardiser fitted on ALL rows) versus on the training part only.
Ridge regression (alpha = 1) is the model for which scaling matters; tree models are invariant to monotone feature scaling.
Usage: python -u 18_preprocessing_leakage.py <out_dir>"""
import importlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import rankexp
m7 = importlib.import_module("07_rank_experiment")
if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); rows = []
    for ds in ("matr", "hust", "xjtu"):
        d, feats = m7.load(ds); y = d["SOH_rated"].values; X = d[feats].values.astype(float)
        imp_all = SimpleImputer(strategy="median").fit(X); Xall = StandardScaler().fit(imp_all.transform(X)).transform(imp_all.transform(X))
        for proto in ("P0", "P2", "P3"):
            pl = np.full(len(d), np.nan); pt = np.full(len(d), np.nan)
            for tr, te in rankexp.folds(d, proto, 0):
                pl[te] = Ridge(alpha=1.0).fit(Xall[tr], y[tr]).predict(Xall[te])
                imp = SimpleImputer(strategy="median").fit(X[tr]); sc = StandardScaler().fit(imp.transform(X[tr]))
                pt[te] = Ridge(alpha=1.0).fit(sc.transform(imp.transform(X[tr])), y[tr]).predict(sc.transform(imp.transform(X[te])))
            r = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
            rows.append({"dataset": ds, "protocol": proto, "ridge_fit_on_all": r(pl), "ridge_fit_on_train": r(pt), "diff": r(pl) - r(pt)}); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "preprocessing_leakage.csv", index=False)
