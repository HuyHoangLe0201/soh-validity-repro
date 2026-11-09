"""Step 19b - distribution of the leave-one-charging-policy-out error over the held-out policies (the table columns 'median per-policy RMSE' and 'share of the squared error
of the five policies with the largest error').  Usage: python 19b_policy_tail.py <results_dir> <out_csv>   (writes the file that make_tables.py reads as step19/policy_tail.csv)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1])
meta = pd.read_csv(R / "step15/matr_policy/matr_policy_meta.csv.gz"); y = meta.SOH_rated.values
pr = np.load(R / "step15/matr_policy/matr_policy_preds.npz")
rows = {}
for m in ("Ridge", "LightGBM", "XGBoost", "RandomForest"):
    e2 = (pr[f"SOH_rated|default|P3|{m}|0"].astype(float) - y) ** 2
    g = pd.DataFrame({"g": meta.grp.values, "e": e2}).groupby("g").e
    sse, n = g.sum(), g.size()
    rmse = np.sqrt(sse / n)
    rows[m] = {"median_group_rmse": float(rmse.median()), "top5_share": float(sse.sort_values(ascending=False).iloc[:5].sum() / sse.sum()), "max_group_rmse": float(rmse.max())}
out = pd.DataFrame(rows).T
Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(sys.argv[2])
print(out)
