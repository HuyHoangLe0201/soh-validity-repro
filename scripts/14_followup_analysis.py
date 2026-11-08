"""Step 14 - analysis of (a) tuning repeats (step 12) and (b) charge-curve features (step 13).

(a) P3 pooled RMSE for 4 inner-search seeds (0 = steps 7/9, 1..3 = step 12): mean, sd, min-max per model
    and regime; pooled ranking per seed; stability of the top-1 model.
(b) F2 (charge-curve) features: pooled RMSE by protocol/regime/model, and multiplicity-corrected (Holm, BH)
    sign reversals computed with the same procedure as step 11.

Usage: python 14_followup_analysis.py <results_dir> <out_dir>
"""
import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

m11 = importlib.import_module("11_multiplicity")
MODELS = ["Ridge", "LightGBM", "XGBoost", "RandomForest"]


def load_npz(p):
    return dict(np.load(p)) if Path(p).exists() else {}


def tuning_repeats(R, out):
    rows = []
    for ds in ("xjtu", "matr_crate", "matr", "hust"):
        meta = pd.read_csv(R / "step7" / ds / f"{ds}_meta.csv.gz")
        y = meta.SOH_rated.values
        s7 = load_npz(R / "step7" / ds / f"{ds}_preds.npz")
        s9 = load_npz(R / "step9" / ds / f"{ds}_tunedgrp_preds.npz")
        s12 = load_npz(R / "step12" / ds / f"{ds}_tuningrepeats_preds.npz")
        for regime, src0 in (("tuned", s7), ("tuned_grp", s9)):
            for m in MODELS:
                vals = {}
                k0 = f"SOH_rated|{regime}|P3|{m}|0"
                if k0 in src0:
                    vals[0] = src0[k0]
                for seed in (1, 2, 3):
                    k = f"SOH_rated|{regime}|P3|{m}|{seed}"
                    if k in s12:
                        vals[seed] = s12[k]
                r = {s: float(np.sqrt(np.mean((v.astype(float) - y) ** 2))) for s, v in vals.items()}
                if r:
                    dflt = s7.get(f"SOH_rated|default|P3|{m}|0")
                    rows.append({"dataset": ds, "regime": regime, "model": m, "n_seeds": len(r),
                                 "mean": np.mean(list(r.values())), "sd": np.std(list(r.values())),
                                 "min": min(r.values()), "max": max(r.values()),
                                 "default": float(np.sqrt(np.mean((dflt.astype(float) - y) ** 2))) if dflt is not None else np.nan,
                                 **{f"seed{s}": v for s, v in r.items()}})
    t = pd.DataFrame(rows)
    t.to_csv(out / "tuning_repeats_P3.csv", index=False)
    pd.set_option("display.width", 220)
    print("== P3 pooled RMSE over inner-search seeds (mean, sd, range) vs default")
    print(t[["dataset", "regime", "model", "n_seeds", "default", "mean", "sd", "min", "max"]].round(2).to_string(index=False))
    print("\n== top-1 model per seed")
    for (ds, rg), g in t.groupby(["dataset", "regime"]):
        seeds = [c for c in g.columns if c.startswith("seed")]
        tops = {s: g.loc[g[s].idxmin(), "model"] for s in seeds if g[s].notna().all()}
        print(ds, rg, tops)


def waveform(R, out):
    rows, fams = [], []
    for ds in ("matr", "hust", "xjtu"):
        mp = R / "step13" / ds / f"{ds}_preds.npz"
        if not mp.exists():
            continue
        meta = pd.read_csv(R / "step13" / ds / f"{ds}_meta.csv.gz")
        y = meta.SOH_rated.values
        st = load_npz(mp)
        for regime in ("default", "tuned"):
            E = {}
            for p in ("P0", "P2", "P3"):
                for m in MODELS:
                    ks = [k for k in st if k.startswith(f"SOH_rated|{regime}|{p}|{m}|")]
                    if ks:
                        E[(p, m)] = np.nanmean([(st[k].astype(float) - y) ** 2 for k in ks], axis=0)
                        rows.append({"dataset": ds, "regime": regime, "protocol": p, "model": m,
                                     "rmse": float(np.sqrt(np.nanmean(E[(p, m)])))})
            fam = []
            for p in ("P0", "P2", "P3"):
                units = meta.grp.values if p == "P3" else meta.cell.values
                for A, B in itertools.combinations([m for m in MODELS if (p, m) in E], 2):
                    d, pv, k = m11.boot_p(E[(p, A)], E[(p, B)], units)
                    fam.append({"dataset": ds, "regime": regime, "protocol": p, "A": A, "B": B,
                                "delta": d, "p": pv, "units": k})
            if not fam:
                continue
            f = pd.DataFrame(fam)
            f["p_holm"], f["p_bh"] = m11.holm(f.p), m11.bh(f.p)
            fams.append(f)
    r = pd.DataFrame(rows)
    r.to_csv(out / "waveform_rmse.csv", index=False)
    pd.set_option("display.width", 220)
    for (ds, rg), g in r.groupby(["dataset", "regime"]):
        print(f"\n== F2 curve features: {ds} / {rg}  pooled RMSE")
        print(g.pivot(index="model", columns="protocol", values="rmse").round(2).to_string())
    allp = pd.concat(fams, ignore_index=True)
    allp.to_csv(out / "waveform_tests.csv", index=False)
    rev = []
    for (ds, rg, A, B), g in allp.groupby(["dataset", "regime", "A", "B"]):
        g = g[(g.protocol != "P3") | (g.units >= 5)].set_index("protocol")
        for p1, p2 in itertools.combinations(g.index, 2):
            if np.sign(g.loc[p1, "delta"]) * np.sign(g.loc[p2, "delta"]) < 0:
                rev.append({"dataset": ds, "regime": rg, "A": A, "B": B, "p1": p1, "p2": p2,
                            "delta1": g.loc[p1, "delta"], "delta2": g.loc[p2, "delta"],
                            "raw": bool(g.loc[p1, "p"] < .05 and g.loc[p2, "p"] < .05),
                            "holm": bool(g.loc[p1, "p_holm"] < .05 and g.loc[p2, "p_holm"] < .05),
                            "bh": bool(g.loc[p1, "p_bh"] < .05 and g.loc[p2, "p_bh"] < .05)})
    rv = pd.DataFrame(rev)
    rv.to_csv(out / "waveform_reversals.csv", index=False)
    if len(rv):
        print("\n== F2 sign flips between protocols and survivors")
        print(rv.groupby(["dataset", "regime"])[["raw", "holm", "bh"]].sum().join(
            rv.groupby(["dataset", "regime"]).size().rename("sign_flips")).to_string())
        print(rv[rv.holm].round(3).to_string(index=False))


if __name__ == "__main__":
    R, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    tuning_repeats(R, out)
    waveform(R, out)
