"""XJTU battery dataset (Wang et al., Nat. Commun. 2024; Zenodo 10963339): per-cycle charge features.

55 LISHEN 18650 NCM cells (2.0 Ah nominal) in 6 batches with different charge/discharge strategies:
  B1 2C charge, 1C discharge (full)           B2 3C charge, 1C discharge (full)
  B3 2C charge, discharge rate cycled {0.5,1,2,3,5}C (full)
  B4 variable discharge to 3 V (partial) + periodic 'test capacity' cycles
  B5 random-walk discharge (partial) + periodic 'test capacity' cycles
  B6 simulated GEO-satellite profile + 'test capacity' cycles
Capacity is directly measured only in full-discharge cycles, so labels are kept for
  * every cycle >= 2 in batches 1-3, and
  * cycles flagged '[test capacity]' in batches 4-6.
(The data providers interpolate capacity for the other cycles; those cycles are NOT used.)

Features come from the charging stage of the same cycle (stage 1; relative_time_min restarts at 0
at the start of each stage).  Columns named f_* are model inputs; `q_chg` (charge capacity) is the
capacity-derived coupling feature; `dis_c` (discharge C-rate parsed from the description) is kept
only for label-sensitivity analysis.
"""
from __future__ import annotations

import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

NOMINAL = 2.0
V_CV = 4.199


def cell_table(path):
    path = Path(path)
    batch = int(path.parent.name.split("-")[1])
    cell = f"B{batch}_{path.stem}"
    m = loadmat(str(path))
    data, summ = m["data"], m["summary"]
    cap = summ[0][0][1].reshape(-1)
    n = data.shape[1]
    rows = []
    for i in range(1, n):  # index 0 is the initial 0.2C capacity test: not used
        desc = str(data[0][i][7][0])
        labelled = batch in (1, 2, 3) or "test capacity" in desc
        if not labelled:
            continue
        rt = data[0][i][1].reshape(-1)
        starts = np.where(rt == 0)[0]
        if len(starts) < 2:
            continue
        sl = slice(starts[0], starts[1])
        v = data[0][i][2].reshape(-1)[sl]
        cur = data[0][i][3].reshape(-1)[sl]
        q = data[0][i][4].reshape(-1)[sl]
        temp = data[0][i][6].reshape(-1)[sl]
        t = rt[sl]
        hi = np.where(v >= V_CV)[0]
        if len(hi) == 0 or len(t) < 10:
            continue
        k = int(hi[0])
        t_cc, t_all = float(t[k]), float(t[-1])
        t_cv = t_all - t_cc
        mdis = re.search(r"([\d.]+)C discharge", desc)
        rows.append({
            "cell": cell, "batch": batch, "cycle": i + 1, "desc": desc,
            "f_t_cc": t_cc, "f_t_cv": t_cv, "f_ratio": t_cv / t_cc if t_cc > 0 else np.nan,
            "f_i_cc": float(np.abs(cur[:k]).mean()) if k > 0 else np.nan,
            "f_i_cv": float(np.abs(cur[k:]).mean()),
            "f_v_start": float(v[0]), "f_temp": float(np.nanmean(temp)),
            "q_chg": float(np.nanmax(q)), "dis_c": float(mdis[1]) if mdis else np.nan,
            "QD": float(cap[i]),
        })
    return pd.DataFrame(rows)


def build(root, workers=2):
    files = sorted(Path(root).glob("Batch-*/*.mat"))
    with Pool(workers) as p:
        tabs = p.map(cell_table, files)
    d = pd.concat(tabs, ignore_index=True)
    ref = (d[d.groupby("cell").cumcount() < 5].groupby("cell")["QD"].median().rename("QD_ref"))
    d = d.join(ref, on="cell")
    d["SOH_rated"] = 100 * d["QD"] / NOMINAL
    d["SOH_ref"] = 100 * d["QD"] / d["QD_ref"]
    d["rank_in_cell"] = d.groupby("cell").cumcount().astype(float)
    d["n_in_cell"] = d.groupby("cell")["cycle"].transform("count")
    return d


if __name__ == "__main__":
    out = build(sys.argv[1], workers=int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    out.to_csv(sys.argv[2], index=False)
    print(out.shape, out.cell.nunique(), out.groupby("batch").cell.nunique().to_dict())
