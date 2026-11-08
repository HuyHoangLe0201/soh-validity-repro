"""Step 51 - TJU entries that were missing from the tables: (a) capacity-derived input F3 = F1 + charge capacity (as step 16 for XJTU),
(b) the distance-to-training-data analysis of step 10 restricted to TJU (default regime).
Usage: python -u 51_tju_f3_support.py <out_dir>"""
import importlib, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import run
m7 = importlib.import_module("07_rank_experiment"); m10 = importlib.import_module("10_support_analysis")
RES = Path(__file__).resolve().parent.parent / "results"
if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    d, feats = m7.load("tju")
    # (b) support analysis
    y = d[m10.LABEL].values
    store = dict(np.load(RES / "step7/tju/tju_preds.npz"))
    R, L = {}, {}
    for p in ("P0", "P2", "P3"): R[p], L[p] = m10.support_ratio(d, feats, p)
    q1, q2 = np.nanpercentile(R["P3"], [33.3, 66.7]); bins = {p: np.digitize(R[p], [q1, q2]) for p in R}
    rows = []
    for p in R:
        E = {m: (store[f"{m10.LABEL}|default|{p}|{m}|0"].astype(float) - y) ** 2 for m in m10.MODELS if f"{m10.LABEL}|default|{p}|{m}|0" in store}
        for b in (0, 1, 2):
            mask = bins[p] == b
            if mask.sum() < 500 or d.cell[mask].nunique() < 5: continue
            rec = {"dataset": "tju", "regime": "default", "protocol": p, "stratum": str(b), "rows": int(mask.sum()), "cells": int(d.cell[mask].nunique())}
            for m, e in E.items(): rec[f"rmse_{m}"] = float(np.sqrt(e[mask].mean()))
            dl, lo, hi, _ = m10.boot_delta(E["Ridge"][mask], E["LightGBM"][mask], d.cell.values[mask])
            rec.update({"d_Ridge_minus_LightGBM": dl, "lo_Ridge_LightGBM": lo, "hi_Ridge_LightGBM": hi}); rows.append(rec)
    pd.DataFrame(rows).to_csv(out / "support_strata_tju.csv", index=False); print(pd.DataFrame(rows).round(2).to_string(index=False), flush=True)
    # (a) F3
    g = d; diff = g.q_chg - g.QD
    pd.DataFrame([{"corr_qchg_QD": float(np.corrcoef(g.q_chg, g.QD)[0, 1]), "mean_diff_Ah": float(diff.mean()), "sd_diff_Ah": float(diff.std())}]).to_csv(out / "tju_qchg_vs_label.csv", index=False)
    print("corr", np.corrcoef(g.q_chg, g.QD)[0, 1], flush=True)
    run(d, feats + ["q_chg"], "SOH_rated", [{"regime": "default", "protocols": {"P0": 3, "P2": 3, "P3": 1}, "models": m7.FAST4}], out, "tju_F3")
