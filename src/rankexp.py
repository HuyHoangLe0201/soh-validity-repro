"""Rank-reversal experiment: protocols x models x regimes, predictions saved for post-hoc inference.

Protocols
  P0  shuffled 5-fold over pooled cycles (REPS random seeds)
  P2  cell-disjoint random 5-fold (REPS random cell partitions)
  P3  leave-one-group-out, group = column `grp` (batch / charging-rate bin / HUST cell-identifier group)
Regimes
  default : fixed hyper-parameters (same as earlier steps)
  tuned   : hyper-parameters chosen by an INNER 3-fold CV on the training part only
            (P0: shuffled inner folds, P2/P3: cell-grouped inner folds); the inner search uses every
            3rd training row for speed, the final model is refitted on all training rows.
Models : Ridge, LightGBM, XGBoost, RandomForest (60 trees, 30% bootstrap), MLP (default regime only)
"""
from __future__ import annotations

import itertools
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness import mlp_fit_predict

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

DEFAULTS = {
    "Ridge": {"alpha": 1.0},
    "LightGBM": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    "XGBoost": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    "RandomForest": {"max_depth": 10, "min_samples_leaf": 5},
}
GRIDS = {
    "Ridge": [{"alpha": a} for a in (0.01, 0.1, 1, 10, 100, 1000)],
    "LightGBM": [dict(max_depth=md, n_estimators=ne, learning_rate=lr)
                 for md, ne, lr in itertools.product((3, 6), (100, 300), (0.05, 0.1))],
    "XGBoost": [dict(max_depth=md, n_estimators=ne, learning_rate=lr)
                for md, ne, lr in itertools.product((3, 6), (100, 300), (0.05, 0.1))],
    "RandomForest": [dict(max_depth=md, min_samples_leaf=ml)
                     for md, ml in itertools.product((6, 12), (5, 50))],
}


def build(name, p):
    if name == "Ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=p["alpha"]))
    if name == "LightGBM":
        return lgb.LGBMRegressor(num_leaves=31, random_state=42, verbose=-1, n_jobs=4, **p)
    if name == "XGBoost":
        return xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=4, **p)
    if name == "RandomForest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(n_estimators=60, max_samples=0.3, random_state=42,
                                                   n_jobs=4, **p))
    raise ValueError(name)


def inner_select(name, X, y, cells, protocol, grp=None, seed=0):
    """grp given => inner folds hold out whole condition groups (batch / C-rate bin / HUST cell-identifier group)."""
    sub = np.arange(seed % 3, len(X), 3)
    X, y, cells = X[sub], y[sub], cells[sub]

    def rand_group_splits(g, k):  # random assignment of whole groups to k inner folds
        u = np.unique(g)
        np.random.default_rng(seed).shuffle(u)
        f = pd.Series(g).map({c: i % k for i, c in enumerate(u)}).values
        return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]

    if grp is not None:
        grp = grp[sub]
        k = min(3, len(np.unique(grp)))
        splits = rand_group_splits(grp, k) if seed else list(GroupKFold(k).split(X, y, grp))
    elif protocol == "P0":
        splits = list(KFold(3, shuffle=True, random_state=seed).split(X))
    else:
        splits = rand_group_splits(cells, 3) if seed else list(GroupKFold(3).split(X, y, cells))
    best, best_s = None, np.inf
    for p in GRIDS[name]:
        sse, n = 0.0, 0
        for tr, te in splits:
            pr = build(name, p).fit(X[tr], y[tr]).predict(X[te])
            sse += float(np.sum((pr - y[te]) ** 2))
            n += len(te)
        if sse / n < best_s:
            best, best_s = p, sse / n
    return best


def folds(d, protocol, rep):
    idx = np.arange(len(d))
    if protocol == "P0":
        yield from KFold(5, shuffle=True, random_state=rep).split(idx)
    elif protocol == "P2":
        rng = np.random.default_rng(rep)
        u = d["cell"].unique()
        rng.shuffle(u)
        f = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values
        for k in range(5):
            yield idx[f != k], idx[f == k]
    elif protocol == "P3":
        g = d["grp"].values
        for k in sorted(np.unique(g)):
            yield idx[g != k], idx[g == k]
    else:
        raise ValueError(protocol)


def predict(d, feats, label, protocol, name, regime, rep, inner_seed=0):
    X, y, cells = d[feats].values.astype(float), d[label].values, d["cell"].values
    grp_all = d["grp"].values
    pred = np.full(len(d), np.nan, dtype=np.float32)
    chosen = []
    for tr, te in folds(d, protocol, rep):
        if name == "MLP":
            imp = SimpleImputer(strategy="median").fit(X[tr])
            sc = StandardScaler().fit(imp.transform(X[tr]))
            f = lambda A: sc.transform(imp.transform(A))  # noqa: E731
            pred[te] = np.mean([mlp_fit_predict(f(X[tr]), y[tr], f(X[te]), s) for s in range(3)], axis=0)
            continue
        if regime == "default":
            p = DEFAULTS[name]
        else:
            g_in = grp_all[tr] if (regime == "tuned_grp" and protocol == "P3") else None
            p = inner_select(name, X[tr], y[tr], cells[tr], protocol, g_in, inner_seed)
        chosen.append(p)
        pred[te] = build(name, p).fit(X[tr], y[tr]).predict(X[te])
    return pred, chosen


def run(d, feats, label, plan, out, tag):
    """plan: list of dicts {regime, protocols:{P:reps}, models:[...]}.  Saves npz with all predictions."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    store, chosen_log = {}, []
    for blk in plan:
        for protocol, reps in blk["protocols"].items():
            for name in blk["models"]:
                for rep in range(reps):
                    pr, ch = predict(d, feats, label, protocol, name, blk["regime"], rep, blk.get("inner_seed", 0))
                    key = f"{label}|{blk['regime']}|{protocol}|{name}|{rep}"
                    store[key] = pr
                    for p in ch:
                        chosen_log.append({"key": key, **{k: str(v) for k, v in p.items()}})
                m = ~np.isnan(pr)
                rm = float(np.sqrt(np.mean((pr[m] - d[label].values[m]) ** 2)))
                print(f"[{tag}] {blk['regime']:7s} {protocol} {name:12s} rmse(rep{reps-1})={rm:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            np.savez_compressed(out / f"{tag}_preds.npz", **store)  # incremental save
    pd.DataFrame(chosen_log).to_csv(out / f"{tag}_chosen_params.csv", index=False)
    return store
