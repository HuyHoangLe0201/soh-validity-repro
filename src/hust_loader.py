"""HUST LFP dataset (Ma et al. 2022; Mendeley doi:10.17632/nsc7hnsg4s): per-cycle features.

Each cell pickle holds {'rul': {cycle: int}, 'dq': {cycle: discharge capacity in mAh},
'data': {cycle: DataFrame(Status, Cycle number, Current (mA), Voltage (V), Capacity (mAh), Time (s))}}.
Charging is identical for all cells (5.5 A CC, then CC-CV to 3.6 V). Every cell has its own four-stage discharge protocol (checked on 2026-11-01:
77 distinct patterns of stage currents). `batch` below is the cell-identifier prefix (7-8 cells per group); it is NOT a discharge-protocol group:
groups overlap in their discharge-current patterns, so a held-out identifier group is a cell-disjoint fold, not an unseen discharge protocol.

Pickles are read with a restricted unpickler (see safe_pickle.py).

Features per cycle (no temperature or resistance exist in these files):
  t_cc, t_cccv, t_ratio, v_cc_start, v_cc_end, v_cccv_start, i_cccv_mean  -> F1 (not capacity-derived)
  q_chg (charge capacity at the end of charging)                          -> capacity-derived (F3)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from safe_pickle import safe_load

CC = "Constant current charge"
CCCV = "Constant current-constant voltage charge"


def cell_table(path):
    cid = Path(path).stem
    d = safe_load(path)[cid]
    rows = []
    for c, df in d["data"].items():
        st = df["Status"].values
        t, v, i, q = (df["Time (s)"].values, df["Voltage (V)"].values,
                      df["Current (mA)"].values, df["Capacity (mAh)"].values)
        m_cc, m_cv = st == CC, st == CCCV
        if not m_cc.any() or not m_cv.any() or c not in d["dq"]:
            continue
        t_cc = t[m_cc][-1] - t[m_cc][0]
        t_cv = t[m_cv][-1] - t[m_cv][0]
        rows.append({
            "cell": cid, "batch": int(cid.split("-")[0]), "cycle": int(c),
            "t_cc": t_cc, "t_cccv": t_cv, "t_ratio": t_cv / t_cc if t_cc > 0 else np.nan,
            "v_cc_start": v[m_cc][0], "v_cc_end": v[m_cc][-1], "v_cccv_start": v[m_cv][0],
            "i_cccv_mean": np.abs(i[m_cv]).mean() / 1000.0,
            "q_chg": max(q[m_cc].max(), q[m_cv].max()) / 1000.0,
            "QD": d["dq"][c] / 1000.0,
        })
    return pd.DataFrame(rows)


def load_hust(data_dir):
    files = sorted(Path(data_dir).glob("*.pkl"))
    return pd.concat([cell_table(f) for f in files], ignore_index=True)
