"""Step 5 (pilot) - the MATR protocol pilot repeated on HUST (77 LFP cells, 10 cell-identifier groups).

Condition-disjoint protocol P3a = leave-one-identifier-group-out.
HUST charging is identical for all cells; F3 adds the charge capacity, which equals the
discharge capacity label to within ~1 mAh (std of q_chg - QD = 0.001 Ah) - an extreme
feature-label coupling case.

Usage: python 05_hust_protocol_pilot.py <hust_features_csv_gz> <out_dir> [--quick]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from harness import add_labels, run_pilot  # noqa: E402

F1 = ["t_cc", "t_cccv", "t_ratio", "v_cc_start", "v_cc_end", "v_cccv_start", "i_cccv_mean"]
FEATSETS = {"F1": F1, "F3": F1 + ["q_chg"], "F4": F1 + ["cycle"]}
PROTOCOLS = {"P0": 5, "P0b": 5, "P1": 1, "P2": 5, "P3a": 1}


def main(csv, out, quick=False):
    d = pd.read_csv(csv)
    d = d[(d["QD"] > 0.3) & (d["QD"] < 1.5) & (d["cycle"] >= 1)]
    d["t_ratio"] = d["t_ratio"].where(d["t_ratio"] < 20)  # a few degenerate CC segments
    d = add_labels(d, nominal=1.1)
    print(f"cells={d.cell.nunique()} rows={len(d)} groups={d.groupby('batch').cell.nunique().to_dict()}")
    runs = [("F1", "SOH_rated")]
    if not quick:
        runs += [("F3", "SOH_rated"), ("F1", "SOH_ref")]
    run_pilot(d, FEATSETS, runs, PROTOCOLS, out, quick=quick, tag="hust_")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], quick="--quick" in sys.argv)
