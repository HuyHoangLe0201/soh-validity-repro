"""Step 7 - rank-reversal experiment (default and nested-tuned regimes), predictions saved.

Usage: python -u 07_rank_experiment.py <dataset> <out_dir>
  dataset in {matr, matr_crate, hust, xjtu, tju}
    matr        : P0, P2, P3 with group = MATR batch (3)           [P0/P2 shared with matr_crate]
    matr_crate  : P3 only, group = first-stage charge C-rate bin (5 bins)
    hust        : P0, P2, P3 with group = cell-identifier prefix (10 groups of 7-8 cells)
    xjtu        : P0, P2, P3 with group = XJTU batch (6)
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from harness import add_labels  # noqa: E402
from rankexp import run  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
M_F1 = ["IR", "Tavg", "Tmin", "Tmax", "chargetime"]
H_F1 = ["t_cc", "t_cccv", "t_ratio", "v_cc_start", "v_cc_end", "v_cccv_start", "i_cccv_mean"]
ALL5 = ["Ridge", "LightGBM", "XGBoost", "RandomForest", "MLP"]
FAST4 = ["Ridge", "LightGBM", "XGBoost", "RandomForest"]


def crate_bin(policy):
    c1 = float(re.match(r"\s*([\d.]+)C", policy.replace("new", ""))[1])
    return 0 if c1 <= 4.0 else 1 if c1 <= 5.0 else 2 if c1 <= 5.4 else 3 if c1 <= 6.0 else 4


def load(name):
    if name in ("matr", "matr_crate"):
        d = pd.read_csv(RESULTS / "step4" / "matr_summary_clean.csv.gz")
        d["grp"] = d["batch"] if name == "matr" else d["policy"].map(crate_bin)
        return d, M_F1
    if name == "hust":
        h = pd.read_csv(RESULTS / "hust_summary.csv.gz")
        h = h[(h["QD"] > 0.3) & (h["QD"] < 1.5) & (h["cycle"] >= 1)].copy()
        h["t_ratio"] = h["t_ratio"].where(h["t_ratio"] < 20)
        d = add_labels(h, nominal=1.1)
        d["grp"] = d["batch"]
        return d, H_F1
    if name == "xjtu":
        d = pd.read_csv(RESULTS / "xjtu_features.csv.gz")
        d["f_ratio"] = d["f_ratio"].where(d["f_ratio"] < 20)  # degenerate CC segments (as for HUST)
        d["grp"] = d["batch"]
        return d, [c for c in d.columns if c.startswith("f_")]
    if name == "tju":  # Zhu et al. 2022 (Zenodo 6405084); groups = cell type x cycling condition (11 groups)
        h = pd.read_csv(RESULTS / "tju_summary.csv.gz")
        h["t_ratio"] = h["t_ratio"].where(h["t_ratio"] < 20)
        d = add_labels(h, nominal=1.0)
        nom = d[d["cycle"].between(2, 6)].groupby(["ctype", "cell"])["QD"].median().groupby("ctype").median()  # cell-type reference
        d["SOH_rated"] = 100 * d["QD"] / d["ctype"].map(nom)
        d = d[d["SOH_ref"].between(55, 105)].reset_index(drop=True)  # removes <0.2 % of rows with implausible capacities (partial cycles)
        d["grp"] = d["ctype"] + "|" + d["cond"]
        return d, H_F1
    raise ValueError(name)


def main(name, out):
    d, feats = load(name)
    out = Path(out) / name
    out.mkdir(parents=True, exist_ok=True)
    d[["cell", "grp", "cycle", "SOH_rated", "SOH_ref"]].to_csv(out / f"{name}_meta.csv.gz", index=False)
    print(f"{name}: cells={d.cell.nunique()} rows={len(d)} groups={sorted(d.grp.unique())}", flush=True)
    if name == "matr_crate":
        plan = [{"regime": "default", "protocols": {"P3": 1}, "models": ALL5},
                {"regime": "tuned", "protocols": {"P3": 1}, "models": FAST4}]
        run(d, feats, "SOH_rated", plan, out, name)
        run(d, feats, "SOH_ref", [{"regime": "default", "protocols": {"P3": 1}, "models": FAST4}],
            out, name + "_ref")
        return
    plan = [{"regime": "default", "protocols": {"P0": 3, "P2": 3, "P3": 1}, "models": ALL5},
            {"regime": "tuned", "protocols": {"P0": 2, "P2": 2, "P3": 1}, "models": FAST4}]
    run(d, feats, "SOH_rated", plan, out, name)
    run(d, feats, "SOH_ref", [{"regime": "default", "protocols": {"P0": 2, "P2": 2, "P3": 1},
                               "models": FAST4}], out, name + "_ref")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
