"""Step 55 - what does 'a held-out group' mean?  Alternative, comparable definitions of the held-out group in all four datasets.

natural groups used so far:  MATR batch (3) / charging policy (70) / C-rate bin (5), HUST cell-identifier group (10), XJTU batch (6), TJU type x condition (11).
Added here (leave-one-group-out, default hyper-parameters, F1 features, rated label, Ridge / LightGBM / XGBoost / RandomForest, one run):
  cluster  all datasets: k-means (k = 5, seed 0) on cell descriptors = median of the standardized F1 features over cycles 2-30 of each cell;
           the held-out group is a region of the input space that the other cells cover only partly (the same definition for every dataset)
  ctype    TJU: leave one cell type out (NCA / NCM / NCM+NCA; 3 groups)
  temp     TJU: leave one cycling temperature out (25 / 35 / 45 degC; 3 groups)
  rate     TJU: leave one charging rate out (0.25 / 0.5 / 1 C; 3 groups)
Usage: python -u 55_alt_groupings.py <dataset> <grouping> <out_dir>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import run  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")


def regroup(d, feats, kind):
    if kind == "cluster":
        z = (d[feats] - d[feats].mean()) / d[feats].std()
        e = z[d["cycle"].between(2, 30)].groupby(d["cell"]).median().dropna()
        lab = pd.Series(KMeans(5, n_init=10, random_state=0).fit_predict(e.values), index=e.index)
        d = d[d["cell"].isin(lab.index)].copy(); d["grp"] = d["cell"].map(lab)
    elif kind == "ctype":
        d["grp"] = d["ctype"]
    elif kind == "temp":
        d["grp"] = d["cond"].str.extract(r"CY(\d+)-")[0]
    elif kind == "rate":
        d["grp"] = d["cond"].str.extract(r"CY\d+-([\d]+)_")[0]
    return d.reset_index(drop=True)


if __name__ == "__main__":
    ds, kind, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]) / f"{sys.argv[1]}_{sys.argv[2]}"
    out.mkdir(parents=True, exist_ok=True)
    d, feats = m7.load(ds)
    d = regroup(d, feats, kind)
    print(ds, kind, "groups:", d.groupby("grp").cell.nunique().to_dict(), "rows:", len(d), flush=True)
    d[["cell", "grp", "cycle", "SOH_rated", "SOH_ref"]].to_csv(out / f"{ds}_{kind}_meta.csv.gz", index=False)
    run(d, feats, "SOH_rated", [{"regime": "default", "protocols": {"P3": 1}, "models": m7.FAST4}], out, f"{ds}_{kind}")
