"""Step 50 - three seeds for the eight tuned variants (seed 0: tuning runs of steps 31, 36, 40, 45, 48; seeds 1-2: step 49; hyperparameters kept):
range of the late-row RMSE per question, agreement of the sign of (variant - LightGBM) and of 'temporal error above P3 error' across seeds.
Usage: python 50_tuned_seed_summary.py <out_dir>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parent.parent / "results"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
SRC = {"GRU_tuned": "step31/{n}/{n}_tuned.npz", "CNNGRU_tuned": "step48/{n}/{n}_tuned_cnngru_attn.npz", "AttnGRU_tuned": "step48/{n}/{n}_tuned_cnngru_attn.npz",
       "LSTM_tuned": "step36/{n}/{n}_tuned_more.npz", "TCN_tuned": "step36/{n}/{n}_tuned_more.npz", "WinMLP_tuned": "step36/{n}/{n}_tuned_more.npz",
       "TRF_tuned": "step40/{n}/{n}_trf_tuned.npz", "MLP_tuned": "step45/{n}/{n}_mlp_tuned.npz"}
rows = []
for ds in ("matr", "hust", "xjtu", "tju"):
    Z49 = np.load(R / f"step49/{ds}/{ds}_tuned_seeds.npz"); T = np.load(R / f"step20/{ds}/{ds}_temporal.npz"); st = dict(np.load(R / f"step7/{ds}/{ds}_preds.npz"))
    y, late = Z49["y"], Z49["late"]
    def rm(p): return float(np.sqrt(np.nanmean((p.astype(float)[late] - y[late]) ** 2)))
    lg = {}
    for q in ("P2", "P3"):
        ks = [k for k in st if k.startswith(f"SOH_rated|default|{q}|LightGBM|")]
        lg[q] = float(np.sqrt(np.nanmean(np.nanmean([(st[k].astype(float) - y) ** 2 for k in ks], axis=0)[late])))
    lg["PT"] = rm(T["PT|LightGBM"])
    for v, f in SRC.items():
        Z0 = np.load(R / f.format(n=ds)); per = {}
        for q in ("P2", "P3", "PT"):
            vals = [rm(Z0[f"{q}|{v}"])] + [rm(Z49[f"{q}|{v}|{s}"]) for s in (1, 2)]; per[q] = vals
            rows.append({"dataset": ds, "arch": v, "question": q, "mean": float(np.mean(vals)), "min": min(vals), "max": max(vals),
                         "sign_vs_lgbm_agree": int(len({np.sign(x - lg[q]) for x in vals}) == 1), "lgbm": lg[q]})
        rows.append({"dataset": ds, "arch": v, "question": "PT>P3 in seeds", "mean": int(sum(per["PT"][i] > per["P3"][i] for i in range(3))), "min": np.nan, "max": np.nan, "sign_vs_lgbm_agree": np.nan, "lgbm": np.nan})
d = pd.DataFrame(rows); d.to_csv(out / "tuned_seed_summary.csv", index=False)
pd.set_option("display.width", 200)
print(d[d.question.isin(["P2", "P3", "PT"])].groupby("question").agg(cases=("sign_vs_lgbm_agree", "size"), agree=("sign_vs_lgbm_agree", "sum"), max_range=("max", "max")).to_string())
x = d[d.question == "PT>P3 in seeds"]; print(x.pivot(index="arch", columns="dataset", values="mean").to_string())
r = d[d.question.isin(["P2", "P3", "PT"])].copy(); r["range"] = r["max"] - r["min"]; print(r.groupby("question")["range"].describe().round(2).to_string())
