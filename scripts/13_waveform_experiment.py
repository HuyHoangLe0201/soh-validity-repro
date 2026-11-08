"""Step 13 - repeat the protocol / regime experiment with charge-curve features (F2).

F2 = voltage at fixed charge-capacity checkpoints (see src/waveform.py).  Same protocols (P0, P2, P3),
regimes (default, tuned), models (Ridge, LightGBM, XGBoost, RandomForest) as step 7 (no MLP, 2 repeats
for P0/P2).  The rows are those of the step-7 tables for which a waveform feature row exists.

Usage: python -u 13_waveform_experiment.py <dataset> <waveform_csv> <out_dir>   (matr | hust | xjtu)
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import run  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")


def merge(d, w, ds):
    if ds == "matr":
        # waveform rows are indexed by position within the cell; find the offset to the summary 'cycle'
        best = None
        for off in (0, 1):
            m = d[["cell", "cycle", "QD"]].merge(w.assign(cycle=w.idx + off)[["cell", "cycle", "qd_wave"]],
                                                 on=["cell", "cycle"])
            err = float(np.nanmedian(np.abs(m.QD - m.qd_wave)))
            if best is None or err < best[0]:
                best = (err, off)
        print(f"matr: alignment offset={best[1]}, median |QD - Qd_wave| = {best[0]:.4f} Ah", flush=True)
        assert best[0] < 0.01, "waveform rows are not aligned with the summary rows"
        w = w.assign(cycle=w.idx + best[1])
    wv = [c for c in w.columns if c.startswith("w_v")]
    out = d.merge(w[["cell", "cycle"] + wv], on=["cell", "cycle"], how="inner")
    out = out[out[wv].isna().mean(axis=1) <= 0.5].reset_index(drop=True)
    return out, wv


if __name__ == "__main__":
    ds, wcsv, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]) / sys.argv[1]
    d, _ = m7.load(ds)
    d, feats = merge(d, pd.read_csv(wcsv), ds)
    out.mkdir(parents=True, exist_ok=True)
    d[["cell", "grp", "cycle", "SOH_rated", "SOH_ref"]].to_csv(out / f"{ds}_meta.csv.gz", index=False)
    print(f"{ds}: cells={d.cell.nunique()} rows={len(d)} groups={sorted(d.grp.unique())} feats={feats} "
          f"NaN share={d[feats].isna().mean().mean():.3f}", flush=True)
    plan = [{"regime": "default", "protocols": {"P0": 2, "P2": 2, "P3": 1}, "models": m7.FAST4},
            {"regime": "tuned", "protocols": {"P0": 2, "P2": 2, "P3": 1}, "models": m7.FAST4}]
    run(d, feats, "SOH_rated", plan, out, ds)
