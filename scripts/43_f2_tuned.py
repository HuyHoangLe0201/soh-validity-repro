"""Step 43 - tuned hyperparameters with the charge-curve features F2 on MATR and HUST (only XJTU had them in step 13).
Ridge and LightGBM, regime 'tuned' (inner three-fold search holding out cells), P2 (1 partition) and P3, rows as in step 13.
Usage: python -u 43_f2_tuned.py <dataset> <waveform_csv> <out_dir>
"""
import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import run  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")
m13 = importlib.import_module("13_waveform_experiment")
ds, wcsv, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]) / sys.argv[1]
d, _ = m7.load(ds)
d, feats = m13.merge(d, pd.read_csv(wcsv), ds)
out.mkdir(parents=True, exist_ok=True)
d[["cell", "grp", "cycle", "SOH_rated", "SOH_ref"]].to_csv(out / f"{ds}_meta.csv.gz", index=False)
run(d, feats, "SOH_rated", [{"regime": "tuned", "protocols": {"P2": 1, "P3": 1}, "models": ["Ridge", "LightGBM"]}], out, ds)
