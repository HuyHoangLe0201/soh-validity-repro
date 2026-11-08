"""Step 34 - resolve the 135-versus-124 cell difference of the MATR files and repeat the protocol experiment on the 124 cells.
The starter code of the Severson data removes 5 cells of batch 1 (b1c8, b1c10, b1c12, b1c13, b1c22) and 6 cells of batch 3
(b3c37, b3c2, b3c23, b3c32, b3c42, b3c43); 135 - 11 = 124.  (List as implemented in the repository of Erdogan 2026, scripts/extract_mit_h5.py,
which cites the Severson starter code.)  Default regime, F1 features, rated label, P0 (3 repetitions), P2 (3), P3 (batches).
Usage: python -u 34_matr124.py <out_dir>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import run  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")
REMOVE = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22", "b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]
out = Path(sys.argv[1]) / "matr124"
out.mkdir(parents=True, exist_ok=True)
d, feats = m7.load("matr")
assert set(REMOVE) <= set(d.cell.unique())
d = d[~d.cell.isin(REMOVE)].reset_index(drop=True)
print("cells", d.cell.nunique(), "rows", len(d), flush=True)
d[["cell", "grp", "cycle", "SOH_rated", "SOH_ref"]].to_csv(out / "matr124_meta.csv.gz", index=False)
y = d["SOH_rated"].values
store = run(d, feats, "SOH_rated", [{"regime": "default", "protocols": {"P0": 3, "P2": 3, "P3": 1}, "models": m7.ALL5}], out, "matr124")
rows = []
for p in ("P0", "P2", "P3"):
    for m in m7.ALL5:
        ks = [k for k in store if k.startswith(f"SOH_rated|default|{p}|{m}|")]
        e = np.nanmean([(store[k].astype(float) - y) ** 2 for k in ks], axis=0)
        rows.append({"protocol": p, "model": m, "rmse": float(np.sqrt(e.mean()))})
pd.DataFrame(rows).to_csv(out / "matr124_rmse.csv", index=False)
print(pd.DataFrame(rows).pivot(index="model", columns="protocol", values="rmse").round(2))
