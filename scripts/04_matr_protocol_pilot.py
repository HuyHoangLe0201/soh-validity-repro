"""Step 4 (pilot) - protocol ladder, feature-label coupling and label definition on MATR.

Per-cycle SOH regression from *summary-level* cycle features (no waveforms).
The evaluation logic lives in src/harness.py (shared with the HUST pilot).

Feature sets
  F1  IR, Tavg, Tmin, Tmax, chargetime            (no capacity-derived input)
  F3  F1 + QCharge                                 (charge capacity ~ discharge capacity: label proxy)
  F4  F1 + cycle number                            (a deployment-known "age" shortcut)
Label variants
  rated : 100 * QDischarge / 1.1 Ah
  ref   : 100 * QDischarge / median(QDischarge of cycles 2..6 of the same cell)

Usage: python -u 04_matr_protocol_pilot.py <matr_dir> <out_dir> [--quick]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from harness import add_labels, run_pilot  # noqa: E402
from matr_loader import load_matr_summary  # noqa: E402

F1 = ["IR", "Tavg", "Tmin", "Tmax", "chargetime"]
FEATSETS = {"F1": F1, "F3": F1 + ["QC"], "F4": F1 + ["cycle"]}
PROTOCOLS = {"P0": 3, "P0b": 3, "P1": 1, "P2": 3, "P3a": 1, "P3b": 1}


def prepare(df):
    d = df.copy()
    for c in ["IR", "Tavg", "Tmin", "Tmax"]:  # zeros are missing-value codes in the raw summaries
        d[c] = d[c].replace(0, np.nan)
    d.loc[d["chargetime"] > 60, "chargetime"] = np.nan  # 31 implausible values (max 1290 min)
    d.loc[(d["QC"] < 0.3) | (d["QC"] > 1.3), "QC"] = np.nan
    d = d[(d["QD"] > 0.3) & (d["QD"] < 1.3) & (d["cycle"] >= 1)].copy()
    return add_labels(d, nominal=1.1)


def main(matr_dir, out_dir, quick=False, table=None):
    """table: optional path of the cleaned per-cycle table of a previous run (skips the read of the raw MATR files; used on machines without the raw data)"""
    import pandas as pd
    if table:
        d = pd.read_csv(table)
    else:
        from matr_loader import load_matr_summary
        d = prepare(load_matr_summary(matr_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if not table:
        d.to_csv(Path(out_dir) / "matr_summary_clean.csv.gz", index=False)
    print(f"cells={d.cell.nunique()} rows={len(d)} batches={d.groupby('batch').cell.nunique().to_dict()}",
          flush=True)
    runs = [("F1", "SOH_rated")]
    if not quick:
        runs += [("F3", "SOH_rated"), ("F4", "SOH_rated"), ("F1", "SOH_ref")]
    run_pilot(d, FEATSETS, runs, PROTOCOLS, out_dir, quick=quick, holdout_batch=3, tag="matr_")


if __name__ == "__main__":
    tab = sys.argv[sys.argv.index("--table") + 1] if "--table" in sys.argv else None
    main(sys.argv[1], sys.argv[2], quick="--quick" in sys.argv, table=tab)
