"""Step 15 - MATR leave-one-charging-policy-out (about 70 held-out groups instead of 3 batches or 5 rate bins).

Group = exact charging-policy string.  P3 therefore holds out every cell that was cycled with one policy, so the
held-out policy is unseen (neighbouring policies exist, as the policies are variations on multi-step fast charging).
P0/P2 predictions are those of step 7 (same rows); only P3 is run here.
Regimes: default, tuned_grp (inner folds hold out whole policies).
Usage: python -u 15_policy_experiment.py <out_dir>
"""
import importlib, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import run
m7 = importlib.import_module("07_rank_experiment")
if __name__ == "__main__":
    out = Path(sys.argv[1]) / "matr_policy"; out.mkdir(parents=True, exist_ok=True)
    d, feats = m7.load("matr")
    d["grp"] = pd.factorize(d["policy"])[0]
    print("policy groups:", d.grp.nunique(), "cells:", d.cell.nunique(), "rows:", len(d), flush=True)
    d[["cell", "grp", "cycle", "SOH_rated", "SOH_ref"]].to_csv(out / "matr_policy_meta.csv.gz", index=False)
    run(d, feats, "SOH_rated", [{"regime": "default", "protocols": {"P3": 1}, "models": m7.FAST4}], out, "matr_policy")
    run(d, feats, "SOH_rated", [{"regime": "tuned_grp", "protocols": {"P3": 1}, "models": m7.FAST4}], out, "matr_policy_tunedgrp")
