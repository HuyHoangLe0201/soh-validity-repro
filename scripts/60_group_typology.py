"""Step 60 - typology of the held-out groups: what kind of shift does each leave-one-group-out design create, and how large is it?
For every grouping (natural groupings of steps 7 and 15, alternative groupings of step 55) and the four tabular models (default settings, F1 features):
number of groups, cells per group, pooled RMSE, macro-group RMSE, ratio to the cell-held-out error (P2) on the same rows, and the median distance of the held-out rows to their
nearest training row in the standardized F1 feature space relative to the same distance under P2 (distance ratio; 1 = no more remote than a new cell of the same pool).
Usage: python 60_group_typology.py <results_dir> <out_dir>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
m7 = importlib.import_module("07_rank_experiment")
MODELS = ["Ridge", "LightGBM", "XGBoost", "RandomForest"]
rng = np.random.default_rng(60)

SPEC = [  # dataset, grouping label, shift type, preds file, meta file
    ("matr", "batch (3)", "manufacturing / test batch", "step7/matr/matr_preds.npz", "step7/matr/matr_meta.csv.gz"),
    ("matr", "C-rate bin (5)", "charging-protocol family", "step7/matr_crate/matr_crate_preds.npz", "step7/matr_crate/matr_crate_meta.csv.gz"),
    ("matr", "charging policy (70)", "individual charging policy", "step15/matr_policy/matr_policy_preds.npz", "step15/matr_policy/matr_policy_meta.csv.gz"),
    ("matr", "input-space cluster (5)", "region of input space", "step55/matr_cluster/matr_cluster_preds.npz", "step55/matr_cluster/matr_cluster_meta.csv.gz"),
    ("hust", "identifier group (10)", "cell identifier (no condition shift)", "step7/hust/hust_preds.npz", "step7/hust/hust_meta.csv.gz"),
    ("hust", "input-space cluster (5)", "region of input space", "step55/hust_cluster/hust_cluster_preds.npz", "step55/hust_cluster/hust_cluster_meta.csv.gz"),
    ("xjtu", "batch (6)", "charge / discharge strategy", "step7/xjtu/xjtu_preds.npz", "step7/xjtu/xjtu_meta.csv.gz"),
    ("xjtu", "input-space cluster (5)", "region of input space", "step55/xjtu_cluster/xjtu_cluster_preds.npz", "step55/xjtu_cluster/xjtu_cluster_meta.csv.gz"),
    ("tju", "type x condition (11)", "cell type and condition", "step7/tju/tju_preds.npz", "step7/tju/tju_meta.csv.gz"),
    ("tju", "cell type (3)", "cell chemistry", "step55/tju_ctype/tju_ctype_preds.npz", "step55/tju_ctype/tju_ctype_meta.csv.gz"),
    ("tju", "temperature (3)", "cycling temperature", "step55/tju_temp/tju_temp_preds.npz", "step55/tju_temp/tju_temp_meta.csv.gz"),
    ("tju", "charging rate (3)", "charging rate", "step55/tju_rate/tju_rate_preds.npz", "step55/tju_rate/tju_rate_meta.csv.gz"),
    ("tju", "input-space cluster (5)", "region of input space", "step55/tju_cluster/tju_cluster_preds.npz", "step55/tju_cluster/tju_cluster_meta.csv.gz"),
]


def nn_median(Xz, mask_train, mask_test, cap=20000):
    tr = np.where(mask_train)[0]; te = np.where(mask_test)[0]
    tr = rng.choice(tr, min(cap, len(tr)), replace=False); te = rng.choice(te, min(3000, len(te)), replace=False)
    return float(np.median(NearestNeighbors(n_neighbors=1).fit(Xz[tr]).kneighbors(Xz[te])[0]))


def main(R, out):
    R, out = Path(R), Path(out); out.mkdir(parents=True, exist_ok=True); rows = []
    cache = {}
    for ds, label, shift, pf, mf in SPEC:
        d, feats = m7.load(ds); meta = pd.read_csv(R / mf)
        if ds not in cache:
            d = d.reset_index(drop=True); Xz = ((d[feats] - d[feats].mean()) / d[feats].std()).fillna(0).values
            p2 = dict(np.load(R / f"step7/{ds}/{ds}_preds.npz")); y2 = d["SOH_rated"].values
            cache[ds] = (d, Xz, p2, y2)
            # distance under P2: five cell folds of repetition 0
            rg = np.random.default_rng(0); u = d["cell"].unique(); rg.shuffle(u); f = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values
            cache[ds] += (float(np.median([nn_median(Xz, f != k, f == k) for k in range(5)])),)
        d, Xz, p2, y2, dist_p2 = cache[ds]
        if len(meta) != len(d):   # alternative groupings may drop cells: align by row keys
            d = d.merge(meta[["cell", "cycle", "grp"]].rename(columns={"grp": "grp_alt"}), on=["cell", "cycle"], how="inner")
        d = d.reset_index(drop=True)
        y = meta["SOH_rated"].values
        P = np.load(R / pf)
        grp = meta["grp"].values; ug = np.unique(grp)
        cells_per = meta.groupby("grp").cell.nunique()
        # distance: median over held-out groups of the nn distance of the group's rows to the other groups' rows
        Xs = ((d[feats] - d[feats].mean()) / d[feats].std()).fillna(0).values if len(d) == len(meta) else Xz
        dist = float(np.median([nn_median(Xs, grp != g, grp == g) for g in ug[: 12]])) if len(ug) <= 12 else float(np.median([nn_median(Xs, grp != g, grp == g) for g in rng.choice(ug, 12, replace=False)]))
        for m in MODELS:
            k3 = f"SOH_rated|default|P3|{m}|0"
            e3 = (P[k3].astype(float) - y) ** 2
            ks = [k for k in p2 if k.startswith(f"SOH_rated|default|P2|{m}|")]
            e2 = np.nanmean([(p2[k].astype(float) - y2) ** 2 for k in ks], axis=0)
            if len(e2) != len(e3):   # alternative groupings over a subset of cells (HUST/others keep all rows; guard)
                e2 = e2[: len(e3)]
            gm = pd.Series(e3).groupby(grp).mean()
            rows.append({"dataset": ds, "grouping": label, "shift": shift, "groups": len(ug), "cells_min": int(cells_per.min()), "cells_max": int(cells_per.max()),
                         "model": m, "pooled_P3": float(np.sqrt(np.nanmean(e3))), "macro_group_P3": float(np.mean(np.sqrt(gm))),
                         "pooled_P2": float(np.sqrt(np.nanmean(e2))), "ratio_P3_P2": float(np.sqrt(np.nanmean(e3)) / np.sqrt(np.nanmean(e2))),
                         "dist_ratio": dist / dist_p2})
        print(ds, label, "done", flush=True)
    D = pd.DataFrame(rows); D.to_csv(out / "group_typology.csv", index=False)
    pd.set_option("display.width", 220)
    print(D[D.model.isin(["Ridge", "LightGBM"])].pivot_table(index=["dataset", "grouping", "groups", "dist_ratio"], columns="model", values=["pooled_P3", "ratio_P3_P2"]).round(2).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
