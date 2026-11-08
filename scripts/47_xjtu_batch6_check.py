"""Step 47 - check the explanation for the XJTU batch-6 coupling result on the charge records.
Hypothesis (dataset documentation): batch-6 cells are not fully discharged before the charge that precedes a
capacity test, so the charge capacity q_chg of that cycle is below the discharge-capacity label QD.
Prediction: in batch 6 the start voltage of the charge is higher than in batches 1-5, and the deficit QD - q_chg
rises with the start voltage.  Usage: python -u 47_xjtu_batch6_check.py <xjtu_root> <out_dir>"""
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from xjtu_loader import cell_table
if __name__ == "__main__":
    root, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("Batch-*/*.mat"))
    with Pool(2) as p: d = pd.concat(p.map(cell_table, files), ignore_index=True)
    d["deficit"] = d.QD - d.q_chg
    rows = []
    for b, g in d.groupby("batch"):
        rho = g[["deficit", "f_v_start"]].corr(method="spearman").iloc[0, 1]
        rows.append({"batch": int(b), "rows": len(g), "v_start_mean": g.f_v_start.mean(), "v_start_sd": g.f_v_start.std(),
                     "deficit_mean_Ah": g.deficit.mean(), "deficit_sd_Ah": g.deficit.std(), "spearman_deficit_vstart": rho})
    r = pd.DataFrame(rows); r.to_csv(out / "batch_start_voltage.csv", index=False); print(r.round(3).to_string(index=False))
    g = d[d.batch == 6]
    print("batch 6 cells:", g.cell.nunique())
    d[["cell","batch","cycle","f_v_start","q_chg","QD","deficit"]].to_csv(out / "xjtu_qchg_start_voltage.csv.gz", index=False)
