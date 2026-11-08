"""Step 19 - inference for the MATR leave-one-policy-out experiment (70 held-out groups).
Family: all pairwise model tests of regime `default`, protocols P0, P2 (units = cells) and P3 = leave-one-policy-out (units = policies).
Usage: python -u 19_policy_inference.py <results_dir> <out_dir>"""
import importlib, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
m11 = importlib.import_module("11_multiplicity")
M = ["Ridge", "LightGBM", "XGBoost", "RandomForest"]
def main(R, out):
    R, out = Path(R), Path(out); out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(R / "step15/matr_policy/matr_policy_meta.csv.gz"); y = meta.SOH_rated.values
    s7 = dict(np.load(R / "step7/matr/matr_preds.npz")); s15 = dict(np.load(R / "step15/matr_policy/matr_policy_preds.npz"))
    s15g = dict(np.load(R / "step15/matr_policy/matr_policy_tunedgrp_preds.npz"))
    def e2(st, regime, p, m):
        ks = [k for k in st if k.startswith(f"SOH_rated|{regime}|{p}|{m}|")]
        return np.nanmean([(st[k].astype(float) - y) ** 2 for k in ks], axis=0)
    E = {}
    for p in ("P0", "P2"):
        for m in M: E[(p, m)] = e2(s7, "default", p, m)
    for m in M: E[("P3", m)] = e2(s15, "default", "P3", m)
    rows = [{"protocol": p, "model": m, "rmse": float(np.sqrt(np.nanmean(E[(p, m)])))} for p in ("P0", "P2", "P3") for m in M]
    rm = pd.DataFrame(rows); rm.to_csv(out / "policy_rmse.csv", index=False)
    tg = pd.DataFrame([{"model": m, "default_P3": float(np.sqrt(np.nanmean(E[("P3", m)]))),
                        "tuned_group_inner_P3": float(np.sqrt(np.nanmean(e2(s15g, "tuned_grp", "P3", m))))} for m in M]); tg.to_csv(out / "policy_tuning.csv", index=False)
    fam = []
    for p in ("P0", "P2", "P3"):
        units = meta.grp.values if p == "P3" else meta.cell.values
        for A, B in itertools.combinations(M, 2):
            d, pv, k = m11.boot_p(E[(p, A)], E[(p, B)], units)
            fam.append({"protocol": p, "A": A, "B": B, "delta": d, "p": pv, "units": k})
    f = pd.DataFrame(fam); f["p_holm"], f["p_bh"] = m11.holm(f.p), m11.bh(f.p); f.to_csv(out / "policy_tests.csv", index=False)
    rev = []
    for (A, B), g in f.groupby(["A", "B"]):
        g = g.set_index("protocol")
        for p1, p2 in itertools.combinations(g.index, 2):
            if np.sign(g.loc[p1, "delta"]) * np.sign(g.loc[p2, "delta"]) < 0:
                rev.append({"A": A, "B": B, "p1": p1, "p2": p2, "delta1": g.loc[p1, "delta"], "delta2": g.loc[p2, "delta"],
                            "raw": bool(g.loc[p1, "p"] < .05 and g.loc[p2, "p"] < .05), "holm": bool(g.loc[p1, "p_holm"] < .05 and g.loc[p2, "p_holm"] < .05),
                            "bh": bool(g.loc[p1, "p_bh"] < .05 and g.loc[p2, "p_bh"] < .05)})
    rv = pd.DataFrame(rev); rv.to_csv(out / "policy_reversals.csv", index=False)
    # group-level wins: in how many held-out policies is ridge better than LightGBM?
    g = pd.DataFrame({"g": meta.grp.values, "r": E[("P3", "Ridge")], "l": E[("P3", "LightGBM")], "x": E[("P3", "XGBoost")]}).groupby("g").mean()
    wins = int((g.r < g.l).sum()); n = len(g)
    pd.set_option("display.width", 200)
    print(rm.pivot(index="model", columns="protocol", values="rmse").round(2).to_string()); print(tg.round(2).to_string(index=False))
    print(f"\nsign flips: {len(rv)}, raw {int(rv.raw.sum()) if len(rv) else 0}, Holm {int(rv.holm.sum()) if len(rv) else 0}, BH {int(rv.bh.sum()) if len(rv) else 0}")
    if len(rv): print(rv.round(3).to_string(index=False))
    print(f"\nridge better than LightGBM in {wins} of {n} held-out policies")
    P3 = f[f.protocol == "P3"].round(3); print(P3.to_string(index=False))
    pd.DataFrame([{"ridge_wins": wins, "groups": n}]).to_csv(out / "policy_group_wins.csv", index=False)
if __name__ == "__main__": main(sys.argv[1], sys.argv[2])
