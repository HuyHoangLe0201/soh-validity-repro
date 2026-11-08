"""Step 29 - other architectures, a longer window and seed variation for the five questions (cf. step 26).
Variants: LSTM (1 layer, 32), TCN (3 causal dilated conv layers, 32 channels), WinMLP (MLP on the flattened window), GRU with a window of 30
cycles (GRU_L30), GRU seeds 1 and 2, transformer seeds 1 and 2.  Questions PR, P0, P2, P3, PT as in step 26 (same partitions, same rows).
Usage: python -u 29_architectures.py <dataset> <out_dir>
"""
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import folds  # noqa: E402

m26 = importlib.import_module("26_neural")
m7 = m26.m7


class LSTMNet(nn.Module):
    def __init__(self, f, h=32):
        super().__init__()
        self.rnn = nn.LSTM(f, h, batch_first=True)
        self.out = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.out(o[:, -1]).squeeze(-1)


class TCNNet(nn.Module):
    def __init__(self, f, h=32):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(f if i == 0 else h, h, 3, dilation=2 ** i, padding=2 * 2 ** i) for i in range(3)])
        self.out = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        z = x.transpose(1, 2)
        T = z.shape[2]
        for c in self.convs:
            z = nn.functional.relu(c(z))[:, :, :T]  # causal: drop right padding
        return self.out(z[:, :, -1]).squeeze(-1)


class WinMLP(nn.Module):
    def __init__(self, f, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(f * m26.L, h), nn.ReLU(), nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


VARIANTS = {  # name: (class, window, seed)
    "LSTM": (LSTMNet, 10, 0), "TCN": (TCNNet, 10, 0), "WinMLP": (WinMLP, 10, 0), "GRU_L30": (m26.GRUNet, 30, 0),
    "GRU_s1": (m26.GRUNet, 10, 1), "GRU_s2": (m26.GRUNet, 10, 2), "TRF_s1": (m26.TrfNet, 10, 1), "TRF_s2": (m26.TrfNet, 10, 2),
}


def tp(R, make, tr, te, seed):
    X, y, ym, ys = R.prep(tr)
    net = make(X.shape[1])
    m26.fit(net, X, R.W, y, np.asarray(tr)[::m26.STRIDE], m26.EPOCHS, 2e-3, seed)
    return m26.predict(net, X, R.W, np.asarray(te)) * ys + ym


if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    out.mkdir(parents=True, exist_ok=True)
    d0, feats = m7.load(name)
    m26.STRIDE = 1 if len(d0) < 30000 else 2 if len(d0) < 70000 else 4
    t0 = time.time()
    store = {}
    runners = {}
    for vname, (cls, win, seed) in VARIANTS.items():
        m26.L = win
        if win not in runners:
            runners[win] = m26.Runner(d0, feats)
        R = runners[win]
        d = R.d
        n = len(d)
        rk = d.groupby("cell")["cycle"].rank(method="first").values
        ncell = d.groupby("cell")["cycle"].transform("size").values
        late = rk > np.floor(0.6 * ncell)
        store["late"], store["y"] = late, R.y0
        qs = ["P0", "P2", "P3", "PT", "PR"] if vname in ("LSTM", "TCN", "WinMLP", "GRU_L30") else ["P2", "P3", "PT"]
        for q in qs:
            p = np.full(n, np.nan, np.float32)
            if q in ("P0", "P2", "P3"):
                for tr, te in folds(d, q, 0):
                    p[te] = tp(R, cls, tr, te, seed)
            elif q == "PT":
                tr, te = np.where(~late)[0], np.where(late)[0]
                p[te] = tp(R, cls, tr, te, seed)
            else:
                rng = np.random.default_rng(0)
                r = np.zeros(n, bool)
                for c, idx in d.groupby("cell").indices.items():
                    r[rng.permutation(idx)[:int(np.floor(0.6 * len(idx)))]] = True
                te = np.where(~r)[0]
                p[te] = tp(R, cls, np.where(r)[0], te, seed); store["PR|mask"] = ~r
            store[f"{q}|{vname}"] = p
            msk = late if q != "PR" else store["PR|mask"]
            print(f"[{name}] {vname:8s} {q} rmse={np.sqrt(np.nanmean((p[msk]-R.y0[msk])**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(out / f"{name}_arch.npz", **store)
