"""TJU dataset (Zhu et al., Nat. Commun. 2022; Zenodo 10.5281/zenodo.6405084, CC BY 4.0): per-cycle features from the csv time series.

One csv per cell, named CY<temperature>-<charge rate>_<discharge rate>-#<tag>; three folders = three cell types
(1: NCA, 2: NCM, 3: NCM+NCA).  Columns: time, control V/mA, Ecell, <I>, Q discharge, Q charge (mAh, reset every step), control V, control mA, cycle.
Charging is CC then CV at 4.2 V (control voltage > 0 in the CV phase).  No rated capacity is given in the record: the label uses a
cell-type reference (median first-cycles capacity over the cells of that type) in place of 'rated' (see build_tju.py).

Features (charge stage, not capacity-derived; same logic as HUST):  t_cc, t_cccv, t_ratio, v_cc_start, v_cc_end, v_cccv_start, i_cccv_mean.
q_chg (charge capacity) is kept for the capacity-derived feature set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["t", "ctrlVmA", "V", "I", "Qd", "Qc", "cV", "cmA", "cyc"]
TYPE = {"Dataset_1_NCA_battery": "NCA", "Dataset_2_NCM_battery": "NCM", "Dataset_3_NCM_NCA_battery": "NCMNCA"}


def cell_table(path, ctype):
    cell = Path(path).stem
    d = pd.read_csv(path, header=0, names=COLS)
    rows = []
    for c, g in d.groupby("cyc"):
        qd = g["Qd"].max() / 1000.0
        chg = g[(g["I"] > 50)]
        cv = chg[chg["cV"] > 0]
        cc = chg[chg["cV"] == 0]
        if len(cc) < 20 or len(cv) < 5 or qd < 0.3:
            continue
        t_cc = cc["t"].iloc[-1] - cc["t"].iloc[0]
        t_cv = cv["t"].iloc[-1] - cv["t"].iloc[0]
        rows.append({"cell": f"{ctype}|{cell}", "ctype": ctype, "cond": cell.split("-#")[0], "cycle": int(c),
                     "t_cc": t_cc, "t_cccv": t_cv, "t_ratio": t_cv / t_cc if t_cc > 0 else np.nan,
                     "v_cc_start": cc["V"].iloc[0], "v_cc_end": cc["V"].iloc[-1], "v_cccv_start": cv["V"].iloc[0],
                     "i_cccv_mean": cv["I"].abs().mean() / 1000.0, "q_chg": g["Qc"].max() / 1000.0, "QD": qd})
    return pd.DataFrame(rows)


def load_tju(root):
    parts = []
    for folder, ctype in TYPE.items():
        for f in sorted((Path(root) / folder).rglob("*.csv")):
            t = cell_table(f, ctype)
            if len(t):
                parts.append(t)
    return pd.concat(parts, ignore_index=True)
