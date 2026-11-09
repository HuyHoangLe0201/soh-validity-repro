"""Step 38 - preprocessing fitted on all rows (imputer, standardization of inputs and target) instead of the training rows, for the GRU.
P2 and P3 with seed 0 on XJTU and TJU; compared with the same GRU with training-row preprocessing.  Usage: python -u 38_preproc_networks.py <out_csv>
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import folds  # noqa: E402

m26 = importlib.import_module("26_neural")


class RunnerAll(m26.Runner):
    def prep(self, tr):
        imp = SimpleImputer(strategy="median").fit(self.X0)
        Xi = imp.transform(self.X0)
        mu, sd = Xi.mean(0), Xi.std(0) + 1e-9
        X = torch.tensor(((Xi - mu) / sd), dtype=torch.float32).to(m26.DEV)
        ym, ys = self.y0.mean(), self.y0.std()
        return X, torch.tensor((self.y0 - ym) / ys, dtype=torch.float32).to(m26.DEV), ym, ys


rows = []
for ds in ("xjtu", "tju"):
    d, feats = m26.m7.load(ds)
    m26.STRIDE = 1 if len(d) < 30000 else 2
    runners = {"train_rows": m26.Runner(d, feats), "all_rows": RunnerAll(d, feats)}
    R0 = runners["train_rows"]
    rk = R0.d.groupby("cell")["cycle"].rank(method="first").values
    ncell = R0.d.groupby("cell")["cycle"].transform("size").values
    late = rk > np.floor(0.6 * ncell)
    for q in ("P2", "P3"):
        res = {}
        for tag, R in runners.items():
            p = np.full(len(R.d), np.nan, np.float32)
            for tr, te in folds(R.d, q, 0):
                p[te] = R.train_predict("GRU", tr, te, 0)
            res[tag] = float(np.sqrt(np.nanmean((p[late] - R.y0[late]) ** 2)))
        rows.append({"dataset": ds, "question": q, **res})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
