"""Step 42 - (A) a second measure of distance from the training data and (B) the leave-one-policy-out comparison restricted to policies with several cells.
A. Mahalanobis distance of each held-out row to the training rows (imputed, standardized F1 features; covariance of the training rows with a small ridge),
   divided by the median of the same quantity over training rows.  Terciles over the P3 rows of each dataset; per tercile the RMSE difference ridge - LightGBM
   (step 7, default regime, repetitions averaged) with a cluster bootstrap over cells (1000 resamples), for P0, P2, P3.
B. MATR leave-one-charging-policy-out (step 15): per-policy RMSE of ridge and LightGBM for policies with >= 3 cells (the others are close to leave-one-cell-out);
   pooled RMSE, median per-policy RMSE and number of policies where ridge is better; cluster bootstrap over policies.
Usage: python 42_support2_multicell.py <out_dir>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import folds  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")
R = Path(__file__).resolve().parent.parent / "results"
rng = np.random.default_rng(42)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)


def maha_ratio(d, feats, proto):
    X = d[feats].values.astype(float); r = np.full(len(d), np.nan)
    for tr, te in folds(d, proto, 0):
        imp = SimpleImputer(strategy="median").fit(X[tr]); sc = StandardScaler().fit(imp.transform(X[tr]))
        Z = sc.transform(imp.transform(X)); mu = Z[tr].mean(0); C = np.cov(Z[tr].T) + 1e-3 * np.eye(Z.shape[1]); Ci = np.linalg.inv(C)
        dist = np.sqrt(np.einsum("ij,jk,ik->i", Z - mu, Ci, Z - mu))
        r[te] = dist[te] / np.median(dist[tr[:20000]])
    return r


def cluster_diff(ea, eb, cells, n=1000):
    df = pd.DataFrame({"c": cells, "a": ea, "b": eb}).groupby("c").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    a, b, m = df.a.values, df.b.values, df.n.values
    pt = np.sqrt(a.sum() / m.sum()) - np.sqrt(b.sum() / m.sum())
    bs = np.empty(n)
    for i in range(n):
        j = rng.integers(0, len(m), len(m)); bs[i] = np.sqrt(a[j].sum() / m[j].sum()) - np.sqrt(b[j].sum() / m[j].sum())
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


rowsA = []
for ds in ("matr", "hust", "xjtu", "tju"):
    d, feats = m7.load(ds); y = d["SOH_rated"].values; cells = d["cell"].values
    st = dict(np.load(R / "step7" / ds / f"{ds}_preds.npz"))
    r3 = maha_ratio(d, feats, "P3"); edges = np.nanpercentile(r3, [33.3, 66.7])
    for proto in ("P0", "P2", "P3"):
        r = r3 if proto == "P3" else maha_ratio(d, feats, proto)
        bins = np.digitize(r, edges)
        E = {}
        for m in ("Ridge", "LightGBM"):
            ks = [k for k in st if k.startswith(f"SOH_rated|default|{proto}|{m}|")]
            E[m] = np.nanmean([(st[k].astype(float) - y) ** 2 for k in ks], axis=0)
        for b in range(3):
            msk = bins == b
            pt, lo, hi = cluster_diff(E["Ridge"][msk], E["LightGBM"][msk], cells[msk])
            rowsA.append({"dataset": ds, "protocol": proto, "tercile": b, "ridge_minus_lgbm": pt, "lo": lo, "hi": hi, "rows": int(msk.sum())})
    print(f"[{ds}] A done", flush=True)
    pd.DataFrame(rowsA).to_csv(out / "support_mahalanobis.csv", index=False)

# B
meta = pd.read_csv(R / "step15/matr_policy/matr_policy_meta.csv.gz"); st = dict(np.load(R / "step15/matr_policy/matr_policy_preds.npz"))
y = meta.SOH_rated.values; g = meta.grp.values
ncell = meta.groupby("grp").cell.nunique()
E = {m: (st[f"SOH_rated|default|P3|{m}|0"].astype(float) - y) ** 2 for m in ("Ridge", "LightGBM")}
rowsB = []
for lab, keep in (("all policies", ncell.index), ("policies with >=3 cells", ncell[ncell >= 3].index), ("policies with 1 cell", ncell[ncell == 1].index)):
    msk = np.isin(g, keep)
    pol = pd.DataFrame({"g": g[msk], "R": E["Ridge"][msk], "L": E["LightGBM"][msk]}).groupby("g").agg(R=("R", "mean"), L=("L", "mean"), n=("R", "size"))
    pt, lo, hi = cluster_diff(E["Ridge"][msk], E["LightGBM"][msk], g[msk])
    rowsB.append({"subset": lab, "policies": len(pol), "rows": int(msk.sum()), "rmse_ridge": float(np.sqrt(np.average(pol.R, weights=pol.n))), "rmse_lgbm": float(np.sqrt(np.average(pol.L, weights=pol.n))),
                  "median_ridge": float(np.median(np.sqrt(pol.R))), "median_lgbm": float(np.median(np.sqrt(pol.L))), "ridge_better_policies": int((pol.R < pol.L).sum()),
                  "diff": pt, "lo": lo, "hi": hi})
pd.DataFrame(rowsB).to_csv(out / "policy_multicell.csv", index=False)
pd.set_option("display.width", 200)
print(pd.DataFrame(rowsA).round(2).to_string(index=False)); print(pd.DataFrame(rowsB).round(2).to_string(index=False))
