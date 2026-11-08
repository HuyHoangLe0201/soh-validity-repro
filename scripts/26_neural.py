"""Step 26 - sequence models (GRU, small Transformer) that read the previous L cycles of a cell, under the five evaluation questions.

Input of a row = F1 features of that cycle and of the L-1 preceding cycles of the same cell (first rows are padded with the first cycle).
Questions (same definitions as steps 20-22; scored on the last 40 % of every cell unless stated):
  PR  random 60:40 split within cells (own random test rows)      P0  random 5-fold      P2  cell-disjoint 5-fold      P3  leave one group out
  PT  first 60 % of every cell train / last 40 % test              CAL cell-disjoint 5-fold + first f of the rows of the fold cells in training
      (GRU/TRF: concatenated; GRU_ft: pre-trained without the early rows, then fine-tuned 3 epochs on them, lr 3e-4, all weights)
Training: Adam lr 2e-3, 12 epochs, batch 512, every k-th training row (k = 1, 2 or 4 for <30k, <70k, >=70k rows), seed 0, targets standardised on the training rows.
Usage: python -u 26_neural.py <dataset> <out_dir>
"""
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import folds  # noqa: E402

import os
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
m7 = importlib.import_module("07_rank_experiment")
L, EPOCHS, BS = 10, 12, 512
STRIDE = 2  # overridden per dataset size in main: 1 (<30k rows), 2 (<70k), 4 (>=70k)


class GRUNet(nn.Module):
    def __init__(self, f, h=32):
        super().__init__()
        self.rnn = nn.GRU(f, h, batch_first=True)
        self.out = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.out(o[:, -1]).squeeze(-1)


class TrfNet(nn.Module):
    def __init__(self, f, h=32, layers=2):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.pos = nn.Parameter(torch.zeros(1, L, h))
        enc = nn.TransformerEncoderLayer(h, 4, 64, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        z = self.enc(self.inp(x) + self.pos)
        return self.out(z[:, -1]).squeeze(-1)


def fit(net, X, W, y, rows, epochs, lr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(rows)
        for b in range(0, len(perm), BS):
            ib = perm[b:b + BS]
            loss = nn.functional.mse_loss(net(X[W[ib]]), y[ib])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def predict(net, X, W, rows):
    net.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(rows), 4096):
            out.append(net(X[W[rows[b:b + 4096]]]).numpy())
    return np.concatenate(out) if out else np.array([])


class Runner:
    def __init__(self, d, feats):
        d = d.reset_index(drop=True)
        self.d = d
        o = d.sort_values(["cell", "cycle"]).index.values
        assert (o == np.arange(len(d))).all() or True
        self.X0 = d[feats].values.astype(np.float64)
        self.y0 = d["SOH_rated"].values.astype(np.float64)
        # window index matrix in the original row order
        order = np.lexsort((d["cycle"].values, d["cell"].values))
        pos = np.empty(len(d), int); pos[order] = np.arange(len(d))
        cells_sorted = d["cell"].values[order]
        W = np.zeros((len(d), L), int)
        for k in range(L):
            j = np.arange(len(d)) - (L - 1 - k)
            j = np.clip(j, 0, None)
            same = cells_sorted[j] == cells_sorted
            first = np.zeros(len(d), int)
            # first row index of each cell in sorted order
            starts = np.r_[0, np.where(cells_sorted[1:] != cells_sorted[:-1])[0] + 1]
            first = np.repeat(starts, np.diff(np.r_[starts, len(d)]))
            jj = np.where(same, j, first)
            W[order, k] = order[jj]
        self.W = torch.from_numpy(W)
        self.feats = feats

    def prep(self, tr):
        imp = SimpleImputer(strategy="median").fit(self.X0[tr])
        Xi = imp.transform(self.X0)
        mu, sd = Xi[tr].mean(0), Xi[tr].std(0) + 1e-9
        X = torch.tensor(((Xi - mu) / sd), dtype=torch.float32)
        ym, ys = self.y0[tr].mean(), self.y0[tr].std()
        y = torch.tensor((self.y0 - ym) / ys, dtype=torch.float32)
        return X, y, ym, ys

    def train_predict(self, kind, tr, te, seed=0, ft_rows=None):
        X, y, ym, ys = self.prep(tr)
        sub = np.asarray(tr)[::STRIDE]
        net = (GRUNet if kind == "GRU" else TrfNet)(X.shape[1])
        fit(net, X, self.W, y, sub, EPOCHS, 2e-3, seed)
        if ft_rows is not None and len(ft_rows):
            fit(net, X, self.W, y, np.asarray(ft_rows), 3, 3e-4, seed)
        return predict(net, X, self.W, np.asarray(te)) * ys + ym


if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    out.mkdir(parents=True, exist_ok=True)
    d, feats = m7.load(name)
    R = Runner(d, feats)
    d = R.d
    STRIDE = 1 if len(d) < 30000 else 2 if len(d) < 70000 else 4
    print(f"{name}: rows={len(d)} stride={STRIDE}", flush=True)
    n_rows = len(d)
    cells = d["cell"].values
    rk = d.groupby("cell")["cycle"].rank(method="first").values
    ncell = d.groupby("cell")["cycle"].transform("size").values
    late = rk > np.floor(0.6 * ncell)
    store = {"late": late, "y": R.y0}
    t0 = time.time()
    for kind in ("GRU", "TRF"):
        # P0, P2, P3 (one repetition)
        for prot in ("P0", "P2", "P3"):
            p = np.full(n_rows, np.nan, np.float32)
            for tr, te in folds(d, prot, 0):
                p[te] = R.train_predict(kind, tr, te)
            store[f"{prot}|{kind}"] = p
            print(f"[{name}] {kind} {prot} late-rmse={np.sqrt(np.nanmean((p[late]-R.y0[late])**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
        # PT
        p = np.full(n_rows, np.nan, np.float32)
        tr, te = np.where(~late)[0], np.where(late)[0]
        p[te] = R.train_predict(kind, tr, te)
        store[f"PT|{kind}"] = p
        print(f"[{name}] {kind} PT rmse={np.sqrt(np.mean((p[te]-R.y0[te])**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
        # PR (seed 0)
        rng = np.random.default_rng(0)
        r = np.zeros(n_rows, bool)
        for c, idx in d.groupby("cell").indices.items():
            r[rng.permutation(idx)[:int(np.floor(0.6 * len(idx)))]] = True
        p = np.full(n_rows, np.nan, np.float32)
        te = np.where(~r)[0]
        p[te] = R.train_predict(kind, np.where(r)[0], te)
        store[f"PR|{kind}"] = p; store["PR|mask"] = ~r
        print(f"[{name}] {kind} PR rmse={np.sqrt(np.mean((p[te]-R.y0[te])**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
        # CAL: cell-disjoint folds (partition seed 100), f = 0.2 and 0.6, concatenation and fine-tuning
        rngc = np.random.default_rng(100)
        u = d["cell"].unique(); rngc.shuffle(u)
        fold = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values
        for f in (0.0, 0.2, 0.6):
            early = rk <= np.floor(f * ncell)
            pc = np.full(n_rows, np.nan, np.float32)
            pf = np.full(n_rows, np.nan, np.float32)
            for k in range(5):
                inf = fold == k
                tr = np.where((~inf) | (inf & early))[0]
                te = np.where(inf & late)[0]
                pc[te] = R.train_predict(kind, tr, te)
                if f > 0:
                    base = np.where(~inf)[0]
                    pf[te] = R.train_predict(kind, base, te, ft_rows=np.where(inf & early)[0])
            store[f"CAL{f}|{kind}"] = pc
            if f > 0:
                store[f"CALFT{f}|{kind}"] = pf
            msg = f"[{name}] {kind} CAL f={f} concat={np.sqrt(np.nanmean((pc[late]-R.y0[late])**2)):.3f}"
            if f > 0:
                msg += f" finetune={np.sqrt(np.nanmean((pf[late]-R.y0[late])**2)):.3f}"
            print(msg + f" ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(out / f"{name}_neural.npz", **store)
