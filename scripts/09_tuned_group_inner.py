"""Step 9 - does tuning with a GROUP-held-out inner CV remove the 'tuned is worse under P3' effect?

Re-runs only P3 (leave-one-group-out) in a new regime `tuned_grp`: the inner 3-fold CV used for
hyper-parameter selection holds out whole condition groups (batch / C-rate bin / HUST cell-identifier group) of the
training part, instead of whole cells.  Everything else is identical to the `tuned` regime of step 7.

Usage: python -u 09_tuned_group_inner.py <dataset> <out_dir>   (dataset in matr, matr_crate, hust, xjtu)
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import run  # noqa: E402

m7 = importlib.import_module("07_rank_experiment")

if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    d, feats = m7.load(name)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{name}: cells={d.cell.nunique()} rows={len(d)} groups={sorted(d.grp.unique())}", flush=True)
    run(d, feats, "SOH_rated", [{"regime": "tuned_grp", "protocols": {"P3": 1}, "models": m7.FAST4}],
        out, name + "_tunedgrp")
