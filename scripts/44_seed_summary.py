"""Step 44 - three seeds for every network of steps 29, 31, 39: spread of the late-row RMSE per question, agreement of the sign of (network - LightGBM),
and agreement of 'temporal error above held-out-group error' across seeds.  Seed 0: step 29 (LSTM, TCN, WinMLP), step 26 (GRU, TRF), step 31 (CNNGRU, AttnGRU);
seeds 1-2: step 29 (GRU, TRF), step 39 (the others).  Usage: python 44_seed_summary.py <out_dir>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rows = []
for ds in ("matr", "hust", "xjtu", "tju"):
    A = np.load(R / f"step29/{ds}/{ds}_arch.npz"); N = np.load(R / f"step26/{ds}/{ds}_neural.npz"); Z31 = np.load(R / f"step31/{ds}/{ds}_tuned.npz")
    Z39 = np.load(R / f"step39/{ds}/{ds}_seeds.npz"); T = np.load(R / f"step20/{ds}/{ds}_temporal.npz"); st = dict(np.load(R / f"step7/{ds}/{ds}_preds.npz"))
    y, late = A["y"], A["late"]
    def rm(p): return float(np.sqrt(np.nanmean((p.astype(float)[late] - y[late]) ** 2)))
    lg = {}
    for q in ("P2", "P3"):
        ks = [k for k in st if k.startswith(f"SOH_rated|default|{q}|LightGBM|")]
        lg[q] = float(np.sqrt(np.nanmean(np.nanmean([(st[k].astype(float) - y) ** 2 for k in ks], axis=0)[late])))
    lg["PT"] = rm(T["PT|LightGBM"])
    for arch in ("GRU", "TRF", "LSTM", "TCN", "WinMLP", "CNNGRU", "AttnGRU"):
        per = {}
        for q in ("P2", "P3", "PT"):
            if arch == "GRU": v = [rm(N[f"{q}|GRU"])] + [rm(A[f"{q}|GRU_s{s}"]) for s in (1, 2)]
            elif arch == "TRF": v = [rm(N[f"{q}|TRF"])] + [rm(A[f"{q}|TRF_s{s}"]) for s in (1, 2)]
            elif arch in ("LSTM", "TCN", "WinMLP"): v = [rm(A[f"{q}|{arch}"])] + [rm(Z39[f"{q}|{arch}|{s}"]) for s in (1, 2)]
            else: v = [rm(Z31[f"{q}|{arch}"])] + [rm(Z39[f"{q}|{arch}|{s}"]) for s in (1, 2)]
            per[q] = v
            rows.append({"dataset": ds, "arch": arch, "question": q, "mean": float(np.mean(v)), "min": min(v), "max": max(v),
                         "sign_vs_lgbm_agree": int(len({np.sign(x - lg[q]) for x in v}) == 1), "lgbm": lg[q]})
        rows.append({"dataset": ds, "arch": arch, "question": "PT>P3 in seeds", "mean": int(sum(per["PT"][i] > per["P3"][i] for i in range(3))), "min": np.nan, "max": np.nan, "sign_vs_lgbm_agree": np.nan, "lgbm": np.nan})
pd.DataFrame(rows).to_csv(out / "seed_summary.csv", index=False)
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(d[d.question.isin(["P2", "P3", "PT"])].groupby("question").agg(cases=("sign_vs_lgbm_agree", "size"), agree=("sign_vs_lgbm_agree", "sum"), max_range=("max", "max")).to_string())
x = d[d.question == "PT>P3 in seeds"]; print(x.pivot(index="arch", columns="dataset", values="mean").to_string())
r = d[d.question.isin(["P2", "P3", "PT"])].copy(); r["range"] = r["max"] - r["min"]; print(r.groupby("question")["range"].describe().round(2).to_string())
