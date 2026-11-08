"""Charge-curve (waveform) features: voltage at fixed charge-capacity checkpoints, V(Q = f * nominal).

Only the CHARGING part of the same cycle is used (never the discharge curve, which would contain the
label).  The checkpoints are fractions of the rated capacity and are reached by every cycle of a healthy
or aged cell (capacity >= 80 %), so missing values are rare.
  MATR : nominal 1.1 Ah, f = .1 ... .7  (7 points)       -> w_v1..w_v7
  HUST : nominal 1.1 Ah, f = .1 ... .5  (5 points)       -> w_v1..w_v5
  XJTU : nominal 2.0 Ah, f = .1 ... .6  (6 points)       -> w_v1..w_v6
MATR cells that continue from batch 1 into batch 2 are concatenated exactly as in matr_loader.py.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matr_loader import B2_TO_B1, BATCH_FILES  # noqa: E402

MATR_FR = np.array([.1, .2, .3, .4, .5, .6, .7]) * 1.1
HUST_FR = np.array([.1, .2, .3, .4, .5]) * 1.1
XJTU_FR = np.array([.1, .2, .3, .4, .5, .6]) * 2.0


def v_at(V, Q, grid):
    """Voltage at the capacity checkpoints (NaN where the charge never reaches them)."""
    if len(Q) < 5:
        return np.full(len(grid), np.nan)
    Qm = np.maximum.accumulate(Q)
    keep = np.concatenate([[True], np.diff(Qm) > 1e-9])
    Qk, Vk = Qm[keep], V[keep]
    out = np.interp(grid, Qk, Vk)
    out[grid > Qk[-1]] = np.nan
    return out


# ------------------------------------------------------------------------------------------ MATR
def matr_cell(f, k, i):
    cyc = f[f["batch"]["cycles"][i, 0]]
    n = cyc["V"].shape[0]
    rows = []
    for j in range(n):
        V = f[cyc["V"][j, 0]][:].ravel()
        Qc = f[cyc["Qc"][j, 0]][:].ravel()
        I = f[cyc["I"][j, 0]][:].ravel()  # noqa: E741
        Qd = f[cyc["Qd"][j, 0]][:].ravel()
        if len(V) < 5:
            rows.append([j, np.nan] + [np.nan] * len(MATR_FR))
            continue
        act = np.where(I > 0.05)[0]  # batches 1-2 start with I = 0 samples: begin at the first charging sample
        if len(act) == 0:
            rows.append([j, float(np.nanmax(Qd))] + [np.nan] * len(MATR_FR))
            continue
        s0 = int(act[0])
        rest = np.where(I[s0:] <= 0.01)[0]
        stop = s0 + (int(rest[0]) if len(rest) else len(I) - s0)
        rows.append([j, float(np.nanmax(Qd))] + list(v_at(V[s0:stop], Qc[s0:stop], MATR_FR)))
    return pd.DataFrame(rows, columns=["idx", "qd_wave"] + [f"w_v{m+1}" for m in range(len(MATR_FR))])


def matr_table(data_dir):
    data_dir = Path(data_dir)
    out = {}
    for k, fn in BATCH_FILES.items():
        with h5py.File(data_dir / fn, "r") as f:
            for i in range(f["batch"]["summary"].shape[0]):
                out[f"b{k}c{i}"] = matr_cell(f, k, i)
                print("matr", k, i, flush=True)
    for b2, b1 in B2_TO_B1.items():  # same merge as matr_loader.load_matr_summary
        a, b = out[b1], out.pop(b2)
        b = b.copy()
        b["idx"] = b["idx"] + len(a)
        out[b1] = pd.concat([a, b], ignore_index=True)
    return pd.concat([t.assign(cell=c) for c, t in out.items()], ignore_index=True)


# ------------------------------------------------------------------------------------------ HUST
def hust_cell(path):
    from safe_pickle import safe_load
    path = Path(path)
    cid = path.stem
    d = safe_load(path)[cid]
    rows = []
    for c, df in d["data"].items():
        st = df["Status"].values
        m = np.isin(st, ["Constant current charge", "Constant current-constant voltage charge"])
        if not m.any():
            continue
        v, q = df["Voltage (V)"].values[m], df["Capacity (mAh)"].values[m] / 1000.0
        rows.append([cid, int(c)] + list(v_at(v, q, HUST_FR)))
    return pd.DataFrame(rows, columns=["cell", "cycle"] + [f"w_v{m+1}" for m in range(len(HUST_FR))])


def hust_table(data_dir, workers=2):
    files = sorted(Path(data_dir).glob("*.pkl"))
    with Pool(workers) as p:
        return pd.concat(p.map(hust_cell, files), ignore_index=True)


# ------------------------------------------------------------------------------------------ XJTU
def xjtu_cell(path):
    from scipy.io import loadmat
    path = Path(path)
    batch = int(path.parent.name.split("-")[1])
    cell = f"B{batch}_{path.stem}"
    m = loadmat(str(path))
    data = m["data"]
    rows = []
    for i in range(1, data.shape[1]):
        desc = str(data[0][i][7][0])
        if not (batch in (1, 2, 3) or "test capacity" in desc):
            continue
        rt = data[0][i][1].reshape(-1)
        starts = np.where(rt == 0)[0]
        if len(starts) < 2:
            continue
        sl = slice(starts[0], starts[1])
        v = data[0][i][2].reshape(-1)[sl]
        q = data[0][i][4].reshape(-1)[sl]
        rows.append([cell, i + 1] + list(v_at(v, q, XJTU_FR)))
    return pd.DataFrame(rows, columns=["cell", "cycle"] + [f"w_v{k+1}" for k in range(len(XJTU_FR))])


def xjtu_table(root, workers=2):
    files = sorted(Path(root).glob("Batch-*/*.mat"))
    with Pool(workers) as p:
        return pd.concat(p.map(xjtu_cell, files), ignore_index=True)


if __name__ == "__main__":
    which, src, dst = sys.argv[1:4]
    t = {"matr": matr_table, "hust": hust_table, "xjtu": xjtu_table}[which](src)
    t.to_csv(dst, index=False)
    print(which, t.shape, flush=True)
