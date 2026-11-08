"""Step 6 - fast protocol ladder (Ridge, LightGBM, XGBoost) on MATR and HUST.

Why a separate fast run: the first MATR pilot with Random Forest and an MLP took ~95 min for a
single feature/label configuration (two processes oversubscribing 4 cores).  The slow models are
kept for the base configuration only (results/step4/matr_F1_all5models_protocol_pilot.csv);
this run adds all feature/label configurations, the fraction-of-life block protocol (P0f) and the
cell-cluster bootstrap for the three fast models.

Usage: python -u 06_fast_ladder_both.py <matr_clean_csv_gz> <hust_features_csv_gz> <out_dir>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from harness import add_labels, run_pilot  # noqa: E402

FAST = ["Ridge", "LightGBM", "XGBoost"]

M_F1 = ["IR", "Tavg", "Tmin", "Tmax", "chargetime"]
MATR_FS = {"F1": M_F1, "F3": M_F1 + ["QC"], "F4": M_F1 + ["cycle"]}
MATR_PROT = {"P0": 3, "P0f": 3, "P0b": 3, "P1": 1, "P2": 3, "P3a": 1, "P3b": 1}

H_F1 = ["t_cc", "t_cccv", "t_ratio", "v_cc_start", "v_cc_end", "v_cccv_start", "i_cccv_mean"]
HUST_FS = {"F1": H_F1, "F3": H_F1 + ["q_chg"], "F4": H_F1 + ["cycle"]}
HUST_PROT = {"P0": 3, "P0f": 3, "P0b": 3, "P1": 1, "P2": 3, "P3a": 1}


def main(matr_csv, hust_csv, out):
    out = Path(out)
    m = pd.read_csv(matr_csv)
    print("== MATR", m.cell.nunique(), "cells", len(m), "rows", flush=True)
    run_pilot(m, MATR_FS, [("F1", "SOH_rated"), ("F3", "SOH_rated"), ("F4", "SOH_rated"),
                           ("F1", "SOH_ref")], MATR_PROT, out / "matr", holdout_batch=3,
              tag="matr_fast_", model_names=FAST)

    h = pd.read_csv(hust_csv)
    h = h[(h["QD"] > 0.3) & (h["QD"] < 1.5) & (h["cycle"] >= 1)].copy()
    h["t_ratio"] = h["t_ratio"].where(h["t_ratio"] < 20)
    h = add_labels(h, nominal=1.1)
    print("== HUST", h.cell.nunique(), "cells", len(h), "rows", flush=True)
    run_pilot(h, HUST_FS, [("F1", "SOH_rated"), ("F3", "SOH_rated"), ("F4", "SOH_rated"),
                           ("F1", "SOH_ref")], HUST_PROT, out / "hust", tag="hust_fast_",
              model_names=FAST)


if __name__ == "__main__":
    main(*sys.argv[1:4])
