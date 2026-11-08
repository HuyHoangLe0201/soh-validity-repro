"""Step 1 - reproduce the published NASA LOBO result (Le & Nguyen 2026, Table 9).

Usage:  python 01_reproduce_lobo.py <nasa_mat_dir> <out_dir>

Reproduces, with the notebook configurations from paper_a_baselines_deployment.ipynb:
  * dataset size and per-battery counts (expect 623; 168/156/168/131)
  * 5-fold shuffled CV vs Leave-One-Battery-Out RMSE for LightGBM / RF / XGBoost / CatBoost
Reference values (published): 5-fold 2.300/2.123/2.215/2.225; LOBO 4.629/5.080/4.651/5.072.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402
from nasa_features import build_feature_dataset, clean_dataset, load_all_batteries  # noqa: E402

try:
    from catboost import CatBoostRegressor
except Exception:  # catboost optional
    CatBoostRegressor = None

FEATS = ["t_CV", "t_CV_t_CC_ratio", "tau", "area_CV"]
PUBLISHED = {
    "LightGBM": (2.300, 4.629), "Random Forest": (2.123, 5.080),
    "XGBoost": (2.215, 4.651), "CatBoost": (2.225, 5.072),
}


def models():
    m = {
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                      num_leaves=31, random_state=42, verbose=-1),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10,
                                               random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                    random_state=42, verbosity=0),
    }
    if CatBoostRegressor is not None:
        m["CatBoost"] = CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1,
                                          random_state=42, verbose=0)
    return m


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def main(mat_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = build_feature_dataset(load_all_batteries(mat_dir))
    df = clean_dataset(raw)
    df.to_csv(out / "cv_features_dataset_reproduced.csv", index=False)

    report = {"n_samples": int(len(df)),
              "per_battery": df.groupby("battery").size().to_dict(),
              "soh_min": float(df.SOH.min()), "soh_max": float(df.SOH.max()),
              "tau_fallback_share": float(df.tau_fallback.mean())}
    print(json.dumps(report, indent=2))

    X, y, g = df[FEATS].values, df["SOH"].values, df["battery"].values
    # The baselines notebook scales on the FULL data before any split (global scaler).
    # Tree models are invariant to this; it is kept here for fidelity.
    Xs = StandardScaler().fit_transform(X)

    rows = []
    for name, model in models().items():
        # 5-fold shuffled (cross_val_predict equivalent)
        pred5 = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=42).split(Xs):
            pred5[te] = clone(model).fit(Xs[tr], y[tr]).predict(Xs[te])
        # LOBO
        predl = np.zeros_like(y)
        for tr, te in LeaveOneGroupOut().split(Xs, y, g):
            predl[te] = clone(model).fit(Xs[tr], y[tr]).predict(Xs[te])
        r5, rl = rmse(y, pred5), rmse(y, predl)
        pub = PUBLISHED.get(name)
        rows.append({"model": name, "rmse_5fold": round(r5, 3), "rmse_lobo": round(rl, 3),
                     "increase_pct": round((rl / r5 - 1) * 100, 1),
                     "pub_5fold": pub[0] if pub else None, "pub_lobo": pub[1] if pub else None})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print("mean increase: %.1f%%" % res.increase_pct.mean())
    res.to_csv(out / "reproduction_table9.csv", index=False)
    (out / "reproduction_report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
