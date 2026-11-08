"""Step 52 - P3 with groups = charge C-rate bins for XJTU and TJU (the MATR column 'P3 (C-rate bins)' of the protocol table had no counterpart).
XJTU: charge rate 1C / 2C / 3C from the cycle description (3 groups).  TJU: charge rate 0.25C / 0.5C / 1C from the condition label CY<T>-<charge>_<discharge> (3 groups,
7 / 118 / 9 cells).  HUST has a single charging protocol, so no such grouping exists.  Default hyperparameters, F1 features, rated label, models of step 7.
Usage: python -u 52_crate_groups_other_datasets.py <out_dir>"""
import importlib, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import run
m7 = importlib.import_module("07_rank_experiment")
if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for name, base in (("xjtu_crate", "xjtu"), ("tju_crate", "tju")):
        d, feats = m7.load(base)
        if base == "xjtu":
            rate = d["desc"].str.extract(r"([\d.]+)C charge")[0].astype(float)
            d["grp"] = rate.groupby(d["cell"]).transform(lambda s: s.mode().iloc[0]).map(lambda v: f"{v:g}C")
        else:
            ch = d["cond"].str.extract(r"CY\d+-(\d+)_")[0].map({"025": "0.25C", "05": "0.5C", "1": "1C"})
            d["grp"] = ch
        print(name, d.groupby("grp").cell.nunique().to_dict(), flush=True)
        o = out / name; o.mkdir(exist_ok=True)
        d[["cell", "grp", "cycle", "SOH_rated", "SOH_ref"]].to_csv(o / f"{name}_meta.csv.gz", index=False)
        run(d, feats, "SOH_rated", [{"regime": "default", "protocols": {"P3": 1}, "models": m7.ALL5}], o, name)
