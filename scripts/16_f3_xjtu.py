"""Step 16 - capacity-derived input (F3 = F1 + charge capacity) on XJTU, default regime, P0/P2/P3 (rated label).
Also reports how closely the charge capacity of a labelled cycle tracks the discharge-capacity label, per batch.
Usage: python -u 16_f3_xjtu.py <out_dir>"""
import importlib, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import run
m7 = importlib.import_module("07_rank_experiment")
if __name__ == "__main__":
    out = Path(sys.argv[1]) / "xjtu_F3"; out.mkdir(parents=True, exist_ok=True)
    d, feats = m7.load("xjtu")
    d[["cell", "grp", "cycle", "SOH_rated", "SOH_ref"]].to_csv(out / "xjtu_F3_meta.csv.gz", index=False)
    rows = []
    for b, g in d.groupby("batch"):
        diff = g.q_chg - g.QD
        rows.append({"batch": int(b), "rows": len(g), "corr_qchg_QD": float(np.corrcoef(g.q_chg, g.QD)[0, 1]), "mean_diff_Ah": float(diff.mean()), "sd_diff_Ah": float(diff.std())})
    pd.DataFrame(rows).to_csv(out / "qchg_vs_label_by_batch.csv", index=False); print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
    run(d, feats + ["q_chg"], "SOH_rated", [{"regime": "default", "protocols": {"P0": 3, "P2": 3, "P3": 1}, "models": m7.FAST4}], out, "xjtu_F3")
