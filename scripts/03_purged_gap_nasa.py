"""Step 3 (pilot) - how much of the shuffled-CV -> LOBO gap is adjacent-cycle leakage?

Block-shuffled 5-fold: every cell is cut into contiguous blocks of b cycles; whole blocks
(not single cycles) are assigned at random to the 5 folds.  b = 1 is ordinary shuffled 5-fold
(every test cycle has same-cell neighbours at distance 1 in training).  Growing b removes
near neighbours from training while every cell is still present in training, so the
RMSE-vs-b curve isolates temporal-adjacency leakage from cross-cell shift.  LOBO (the cell
is never seen) is plotted/printed as the upper reference for the same model.

NOTE: an earlier draft that purged +-w cycles around single shuffled test cycles was
degenerate (test cycles are dense, so almost no same-cell training data survived) and
was replaced by this design.

Usage: python 03_purged_gap_nasa.py <dataset_csv> <out_dir>
"""
import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
ladder = import_module("02_protocol_ladder_nasa")
FEATS, SK = ladder.FEATS, ladder.SK
BLOCKS = [1, 2, 4, 8, 16, 32, 64]
N_REPEATS = 10
K = 5


def block_folds(df, b, seed):
    """Assign (cell, seq//b) blocks to K folds, balanced round-robin after a shuffle."""
    rng = np.random.default_rng(seed)
    key = df["battery"] + "_" + (df["cycle_seq"] // b).astype(str)
    uniq = key.unique()
    rng.shuffle(uniq)
    fold_of = {k: i % K for i, k in enumerate(uniq)}
    return key.map(fold_of).values


def rmse_block(df, model, b, seed):
    X, y = df[FEATS].values, df["SOH"].values
    fold = block_folds(df, b, seed)
    pred = np.full(len(df), np.nan)
    for k in range(K):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        sc = StandardScaler().fit(X[tr])
        pred[te] = clone(model).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def rmse_lobo(df, model):
    X, y, g = df[FEATS].values, df["SOH"].values, df["battery"].values
    pred = np.zeros(len(df))
    for c in np.unique(g):
        tr, te = np.where(g != c)[0], np.where(g == c)[0]
        sc = StandardScaler().fit(X[tr])
        pred[te] = clone(model).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def main(csv, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    rows = []
    for name in ["LightGBM", "XGBoost", "RandomForest", "CatBoost"]:
        lobo = rmse_lobo(df, SK[name])
        for b in BLOCKS:
            rs = [rmse_block(df, SK[name], b, s) for s in range(N_REPEATS)]
            rows.append({"model": name, "block": b, "rmse_mean": np.mean(rs),
                         "rmse_sd": np.std(rs), "lobo_ref": lobo})
    res = pd.DataFrame(rows)
    res.to_csv(out / "block_size_curve_nasa.csv", index=False)
    piv = res.pivot(index="block", columns="model", values="rmse_mean").round(3)
    sd = res.pivot(index="block", columns="model", values="rmse_sd").round(3)
    print(f"Mean RMSE (pct SOH) over {N_REPEATS} random block assignments")
    print(piv.to_string())
    print("\nSD across assignments\n", sd.to_string())
    print("\nLOBO reference:", res.groupby("model")["lobo_ref"].first().round(3).to_dict())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
