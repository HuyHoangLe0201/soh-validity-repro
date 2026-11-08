"""Read the *summary* group of the Severson/MATR batch files (HDF5, MATLAB v7.3).

Only per-cycle summary scalars are read (IR, QCharge, QDischarge, Tavg/Tmin/Tmax,
chargetime), so no multi-GB per-cycle waveform arrays are touched.

Cell identity handling copied from BatteryML (`preprocess_MATR.py`): five cells of batch 1
(b1c0..b1c4) continued their cycling in batch 2 (b2c7, b2c8, b2c9, b2c15, b2c16).  They are
the SAME physical cells.  Their batch-2 records are appended to the batch-1 cell and the
batch-2 ids are dropped; otherwise a "cell-disjoint" split would leak the cell.
"""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

BATCH_FILES = {
    1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
B2_TO_B1 = {"b2c7": "b1c0", "b2c8": "b1c1", "b2c9": "b1c2", "b2c15": "b1c3", "b2c16": "b1c4"}
SUMMARY_KEYS = {"IR": "IR", "QC": "QCharge", "QD": "QDischarge", "Tavg": "Tavg",
                "Tmin": "Tmin", "Tmax": "Tmax", "chargetime": "chargetime", "cycle": "cycle"}


def _read_batch(path, k):
    cells = {}
    with h5py.File(path, "r") as f:
        batch = f["batch"]
        n = batch["summary"].shape[0]
        for i in range(n):
            s = f[batch["summary"][i, 0]]
            rec = {name: np.hstack(s[key][0, :].tolist()).astype(float)
                   for name, key in SUMMARY_KEYS.items()}
            cl = float(np.ravel(f[batch["cycle_life"][i, 0]][:])[0])
            policy = f[batch["policy_readable"][i, 0]][:].tobytes()[::2].decode()
            rec["cycle_life"], rec["policy"] = cl, policy
            cells[f"b{k}c{i}"] = rec
    return cells


def load_matr_summary(data_dir, batches=(1, 2, 3)):
    from pathlib import Path
    data_dir = Path(data_dir)
    raw = {k: _read_batch(data_dir / BATCH_FILES[k], k) for k in batches}
    if 1 in raw and 2 in raw:
        for b2, b1 in B2_TO_B1.items():
            a, b = raw[1][b1], raw[2].pop(b2)
            n1 = len(a["cycle"])
            for key in SUMMARY_KEYS:
                a[key] = np.hstack([a[key], b[key] + (n1 if key == "cycle" else 0)])
            a["cycle_life"] = a["cycle_life"] + len(b["cycle"])
    rows = []
    for k, cells in raw.items():
        for cid, r in cells.items():
            n = len(r["cycle"])
            df = pd.DataFrame({key: r[key] for key in SUMMARY_KEYS})
            df.insert(0, "cell", cid)
            df.insert(1, "batch", k)
            df["cycle_life"] = r["cycle_life"]
            df["policy"] = r["policy"]
            rows.append(df)
    return pd.concat(rows, ignore_index=True)
