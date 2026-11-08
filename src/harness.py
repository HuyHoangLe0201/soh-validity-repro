"""Dataset-agnostic evaluation harness for per-cycle SOH regression.

A dataset is a DataFrame with columns
  cell, batch (condition group), cycle, <feature columns>, QD (discharge capacity, Ah),
  and the derived columns created by `add_labels` (SOH_rated, SOH_ref, rank_in_cell, n_in_cell).

Protocols (scalers / imputers are always fitted inside the training part):
  P0   shuffled 5-fold over pooled cycles                      (REPEATS random seeds)
  P0b  block-shuffled 5-fold, blocks of 64 consecutive cycles within a cell
  P1   forward split inside every cell (first 70% train, last 30% test)
  P2   cell-disjoint random 5-fold                             (REPEATS random cell partitions)
  P3a  leave-one-condition-group-out  (MATR: batch, HUST: cell-identifier group)
  P3b  train on all batches except `holdout_batch`, test on it (optional)
"""
from __future__ import annotations

import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPEATS = 5
MLP_SEEDS = 1  # pilot setting; seed variance (H3) is studied in the full run


def add_labels(d, nominal):
    d = d.copy()
    ref = d[d["cycle"].between(2, 6)].groupby("cell")["QD"].median().rename("QD_ref")
    d = d.join(ref, on="cell").dropna(subset=["QD_ref"])
    d["SOH_rated"] = 100 * d["QD"] / nominal
    d["SOH_ref"] = 100 * d["QD"] / d["QD_ref"]
    d["rank_in_cell"] = d.groupby("cell")["cycle"].rank(method="first") - 1
    d["n_in_cell"] = d.groupby("cell")["cycle"].transform("count")
    return d.reset_index(drop=True)


class MLP(nn.Module):
    def __init__(self, d, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x)


def mlp_fit_predict(Xtr, ytr, Xte, seed, steps=200):
    torch.manual_seed(seed)
    mu, sd = ytr.mean(), ytr.std()
    m = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor((ytr - mu) / sd, dtype=torch.float32).reshape(-1, 1)
    m.train()
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(m(xt), yt).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xte, dtype=torch.float32)).numpy().ravel() * sd + mu


def sk_models(n_rf=100):
    return {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(1.0)),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                      num_leaves=31, random_state=42, verbose=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                    random_state=42, verbosity=0),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"),
                                      RandomForestRegressor(n_estimators=n_rf, max_depth=10,
                                                            min_samples_leaf=5, random_state=42,
                                                            n_jobs=-1)),
    }


def folds(d, protocol, rep, holdout_batch=None):
    idx = np.arange(len(d))
    cells = d["cell"].values
    if protocol == "P0":
        yield from KFold(5, shuffle=True, random_state=rep).split(idx)
    elif protocol == "P0b":
        rng = np.random.default_rng(rep)
        key = d["cell"] + "_" + (d["rank_in_cell"] // 64).astype(int).astype(str)
        u = key.unique()
        rng.shuffle(u)
        f = key.map({k: i % 5 for i, k in enumerate(u)}).values
        for k in range(5):
            yield idx[f != k], idx[f == k]
    elif protocol == "P0f":  # 10 contiguous blocks per cell (block length = 10% of that cell's life)
        rng = np.random.default_rng(rep)
        key = d["cell"] + "_" + np.minimum((d["rank_in_cell"] / d["n_in_cell"] * 10).astype(int), 9).astype(str)
        u = key.unique()
        rng.shuffle(u)
        f = key.map({k: i % 5 for i, k in enumerate(u)}).values
        for k in range(5):
            yield idx[f != k], idx[f == k]
    elif protocol == "P1":
        late = (d["rank_in_cell"] / d["n_in_cell"]).values >= 0.7
        yield idx[~late], idx[late]
    elif protocol == "P2":
        rng = np.random.default_rng(rep)
        u = np.unique(cells)
        rng.shuffle(u)
        f = pd.Series(cells).map({c: i % 5 for i, c in enumerate(u)}).values
        for k in range(5):
            yield idx[f != k], idx[f == k]
    elif protocol == "P3a":
        b = d["batch"].values
        for g in sorted(np.unique(b)):
            yield idx[b != g], idx[b == g]
    elif protocol == "P3b":
        b = d["batch"].values
        yield idx[b != holdout_batch], idx[b == holdout_batch]


def predict(d, protocol, name, model, feats, label, rep, holdout_batch=None):
    X, y = d[feats].values.astype(float), d[label].values
    pred = np.full(len(d), np.nan)
    for tr, te in folds(d, protocol, rep, holdout_batch):
        if name == "MLP":
            imp = SimpleImputer(strategy="median").fit(X[tr])
            sc = StandardScaler().fit(imp.transform(X[tr]))
            f = lambda A: sc.transform(imp.transform(A))  # noqa: E731
            pred[te] = np.mean([mlp_fit_predict(f(X[tr]), y[tr], f(X[te]), s)
                                for s in range(MLP_SEEDS)], axis=0)
        else:
            pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return pred


def run_pilot(d, featsets, runs, protocols, out, quick=False, holdout_batch=None, tag="", model_names=None):
    """featsets: {name: [cols]}; runs: [(featset, label)]; protocols: {name: n_repeats}."""
    from pathlib import Path
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    models = sk_models(50 if quick else 100)
    names = list(models) + ["MLP"]
    if model_names:
        names = [n for n in names if n in model_names]
    rows, preds = [], {}
    for fs, label in runs:
        for p, reps in protocols.items():
            for n in names:
                rs = []
                for rep in range(reps):
                    pr = predict(d, p, n, models.get(n), featsets[fs], label, rep, holdout_batch)
                    preds[(fs, label, p, n, rep)] = pr
                    m = ~np.isnan(pr)
                    e = pr[m] - d[label].values[m]
                    ce = pd.DataFrame({"c": d.cell.values[m], "e2": e ** 2}).groupby("c").e2.mean() ** 0.5
                    rs.append((float(np.sqrt(np.mean(e ** 2))), float(ce.median()),
                               float(np.mean(np.abs(e)))))
                a = np.array(rs)
                rows.append({"featset": fs, "label": label, "protocol": p, "model": n,
                             "rmse": a[:, 0].mean(), "rmse_sd_over_reps": a[:, 0].std(),
                             "median_cell_rmse": a[:, 1].mean(), "mae": a[:, 2].mean(),
                             "n_test": int(m.sum())})
            print(f"[{fs}/{label}] {p} done ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out / f"{tag}protocol_pilot.csv", index=False)  # incremental save
    res = pd.DataFrame(rows)
    res.to_csv(out / f"{tag}protocol_pilot.csv", index=False)

    # cell-cluster bootstrap: how often is each model top-1 (base setting, strict protocols)
    rng = np.random.default_rng(0)
    base = (runs[0][0], runs[0][1])
    boot = []
    for p in [q for q in protocols if q in ("P0", "P2", "P3a", "P3b")]:
        tab = {}
        for n in names:
            pr = preds[(base[0], base[1], p, n, 0)]
            m = ~np.isnan(pr)
            tab[n] = (pd.Series((pr[m] - d[base[1]].values[m]) ** 2)
                      .groupby(d.cell.values[m]).agg(["sum", "count"]))
        cs = tab[names[0]].index.values
        wins = {n: 0 for n in names}
        for _ in range(1000):
            samp = rng.choice(cs, len(cs))
            sc_ = {n: np.sqrt(tab[n].loc[samp, "sum"].sum() / tab[n].loc[samp, "count"].sum())
                   for n in names}
            wins[min(sc_, key=sc_.get)] += 1
        boot.append({"protocol": p, **{f"top1_{n}": wins[n] / 1000 for n in names}})
    bt = pd.DataFrame(boot)
    bt.to_csv(out / f"{tag}top1_bootstrap.csv", index=False)

    for (fs, lb), g in res.groupby(["featset", "label"], sort=False):
        print(f"\nPooled RMSE (pct SOH): {fs} / {lb}")
        print(g.pivot(index="model", columns="protocol", values="rmse")[list(protocols)].round(2).to_string())
    print("\nTop-1 frequency under cell-cluster bootstrap (base setting):")
    print(bt.round(3).to_string(index=False))
    print(f"\ntotal time {time.time()-t0:.0f}s")
    return res, bt
