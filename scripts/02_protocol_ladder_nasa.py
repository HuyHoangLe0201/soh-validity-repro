"""Step 2 (pilot) - protocol ladder on the NASA 4-cell dataset.

Question (H1/H2/H3 of the study plan, NASA-only, therefore *indicative*):
  does reported error and model ranking depend on how the data are split?

Protocols (all scalers are fitted inside the training part - no preprocessing leakage):
  P0  shuffled 5-fold on pooled cycles              (literature-common, same-cell interpolation)
  P0b contiguous-block 5-fold within each cell      (same cell, adjacent-cycle interpolation removed)
  P1  forward split: first 70% of each cell -> train, last 30% -> test  (same-cell extrapolation)
  P2  leave-one-battery-out                         (cell-disjoint)

Models: Ridge, LightGBM, RandomForest, XGBoost, CatBoost,
        MLP_as_published (4-16-16-1, raw target in %SOH, 150 full-batch Adam steps, lr=0.01),
        MLP_std_target   (identical, but the target is standardised on the training part).

Usage: python 02_protocol_ladder_nasa.py <dataset_csv> <out_dir>
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import kendalltau
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
FEATS = ["t_CV", "t_CV_t_CC_ratio", "tau", "area_CV"]
N_MLP_SEEDS = 5


# ----------------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, d=4, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x)


def fit_predict_mlp(Xtr, ytr, Xte, seed, standardise_target):
    torch.manual_seed(seed)
    mu, sd = (ytr.mean(), ytr.std()) if standardise_target else (0.0, 1.0)
    m = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(((ytr - mu) / sd), dtype=torch.float32).reshape(-1, 1)
    m.train()
    for _ in range(150):
        opt.zero_grad()
        loss = nn.functional.mse_loss(m(xt), yt)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        p = m(torch.tensor(Xte, dtype=torch.float32)).numpy().ravel()
    return p * sd + mu


def sk_models():
    return {
        "Ridge": Ridge(alpha=1.0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                      num_leaves=31, random_state=42, verbose=-1),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10,
                                              random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                    random_state=42, verbosity=0),
        "CatBoost": CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1,
                                      random_state=42, verbose=0),
    }


# -------------------------------------------------------------------- protocols
def splits(df, protocol, seed=42):
    n = len(df)
    idx = np.arange(n)
    g = df["battery"].values
    if protocol == "P0_shuffled5":
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(idx):
            yield tr, te
    elif protocol == "P0b_block5":
        # rank of each cycle within its cell, split into 5 contiguous blocks
        rank = df.groupby("battery")["cycle_seq"].rank(method="first").values - 1
        size = df.groupby("battery")["cycle_seq"].transform("count").values
        block = np.minimum((rank / size * 5).astype(int), 4)
        for k in range(5):
            yield idx[block != k], idx[block == k]
    elif protocol == "P1_forward":
        rank = df.groupby("battery")["cycle_seq"].rank(method="first").values - 1
        size = df.groupby("battery")["cycle_seq"].transform("count").values
        late = rank / size >= 0.7
        yield idx[~late], idx[late]
    elif protocol == "P2_LOBO":
        for b in np.unique(g):
            yield idx[g != b], idx[g == b]
    else:
        raise ValueError(protocol)


def run(df, protocol, name, seed_for_model=None):
    X, y = df[FEATS].values, df["SOH"].values
    pred = np.full(len(df), np.nan)
    for tr, te in splits(df, protocol):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if name.startswith("MLP"):
            pred[te] = fit_predict_mlp(Xtr, y[tr], Xte, seed_for_model,
                                       standardise_target=(name == "MLP_std_target"))
        else:
            pred[te] = clone(SK[name]).fit(Xtr, y[tr]).predict(Xte)
    mask = ~np.isnan(pred)
    err = pred[mask] - y[mask]
    out = {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
           "n_test": int(mask.sum())}
    if protocol == "P2_LOBO":
        out["per_cell_rmse"] = {b: float(np.sqrt(np.mean((pred[(df.battery == b).values]
                                - y[(df.battery == b).values]) ** 2))) for b in df.battery.unique()}
    return out


SK = sk_models()


def main(csv, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    protocols = ["P0_shuffled5", "P0b_block5", "P1_forward", "P2_LOBO"]
    names = list(SK) + ["MLP_as_published", "MLP_std_target"]

    rows, per_cell = [], {}
    for p in protocols:
        for n in names:
            if n.startswith("MLP"):
                rs = [run(df, p, n, seed_for_model=s) for s in range(N_MLP_SEEDS)]
                rm = np.array([r["rmse"] for r in rs])
                rows.append({"protocol": p, "model": n, "rmse": rm.mean(), "rmse_sd": rm.std(),
                             "mae": np.mean([r["mae"] for r in rs]), "n_test": rs[0]["n_test"]})
            else:
                r = run(df, p, n)
                rows.append({"protocol": p, "model": n, "rmse": r["rmse"], "rmse_sd": 0.0,
                             "mae": r["mae"], "n_test": r["n_test"]})
                if "per_cell_rmse" in r:
                    per_cell[n] = r["per_cell_rmse"]
    res = pd.DataFrame(rows)
    res.to_csv(out / "protocol_ladder_nasa.csv", index=False)
    (out / "per_cell_rmse_lobo.json").write_text(json.dumps(per_cell, indent=2))

    piv = res.pivot(index="model", columns="protocol", values="rmse")[protocols]
    print("\nPooled RMSE (%SOH)\n", piv.round(3).to_string())
    print("\nRatio to P2_LOBO\n", (piv.div(piv["P2_LOBO"], axis=0)).round(2).to_string())

    # Ranking stability among the five non-MLP-as-published models + std-target MLP
    rank_models = [m for m in piv.index if m != "MLP_as_published"]
    ranks = piv.loc[rank_models].rank()
    print("\nRanks (1 = lowest RMSE), excluding MLP_as_published\n", ranks.to_string())
    taus = {}
    for i, a in enumerate(protocols):
        for b in protocols[i + 1:]:
            t, _ = kendalltau(piv.loc[rank_models, a], piv.loc[rank_models, b])
            taus[f"{a} vs {b}"] = round(float(t), 3)
    print("\nKendall tau between protocol rankings\n", json.dumps(taus, indent=2))
    print("\nTop-1 per protocol:", {p: piv.loc[rank_models, p].idxmin() for p in protocols})
    print("\nPer-cell LOBO RMSE:\n", json.dumps(per_cell, indent=1))
    (out / "kendall_tau.json").write_text(json.dumps(taus, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
