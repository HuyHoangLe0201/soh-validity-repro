"""Step 49 - seeds 1 and 2 for the tuned variants.  The hyperparameters chosen in the tuning runs (seed 0; steps 31, 36, 40, 45, 48) are kept
and each variant is refitted with two further seeds on the same partitions (P2, P3, PT).  Usage: python -u 49_tuned_seeds.py <dataset> <out_dir>"""
import importlib, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankexp import folds
m26 = importlib.import_module("26_neural"); m29 = importlib.import_module("29_architectures"); m31 = importlib.import_module("31_tuned_arch")
m7 = m26.m7
RES = Path(__file__).resolve().parent.parent / "results"
# variant -> (class, window length, file, key template)
V = {"GRU_tuned": (m26.GRUNet, 10, "step31/{n}/{n}_tuned.npz", "chosen|{q}"),
     "CNNGRU_tuned": (m31.CNNGRU, 10, "step48/{n}/{n}_tuned_cnngru_attn.npz", "chosen|{q}|CNNGRU_tuned"),
     "AttnGRU_tuned": (m31.AttnGRU, 10, "step48/{n}/{n}_tuned_cnngru_attn.npz", "chosen|{q}|AttnGRU_tuned"),
     "LSTM_tuned": (m29.LSTMNet, 10, "step36/{n}/{n}_tuned_more.npz", "chosen|{q}|LSTM_tuned"),
     "TCN_tuned": (m29.TCNNet, 10, "step36/{n}/{n}_tuned_more.npz", "chosen|{q}|TCN_tuned"),
     "WinMLP_tuned": (m29.WinMLP, 10, "step36/{n}/{n}_tuned_more.npz", "chosen|{q}|WinMLP_tuned"),
     "TRF_tuned": (m26.TrfNet, 10, "step40/{n}/{n}_trf_tuned.npz", "chosen|{q}"),
     "MLP_tuned": (m29.WinMLP, 1, "step45/{n}/{n}_mlp_tuned.npz", "chosen|{q}")}
if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]; out.mkdir(parents=True, exist_ok=True)
    d0, feats = m7.load(name)
    m26.STRIDE = 1 if len(d0) < 30000 else 2 if len(d0) < 70000 else 4
    Rs = {}
    for L in (10, 1):
        m26.L = L; Rs[L] = m26.Runner(d0, feats)
    R0 = Rs[10]; d = R0.d; n = len(d)
    rk = d.groupby("cell")["cycle"].rank(method="first").values
    ncell = d.groupby("cell")["cycle"].transform("size").values
    late = rk > np.floor(0.6 * ncell)
    store = {"late": late, "y": R0.y0}; t0 = time.time()
    for q in ("P2", "P3", "PT"):
        splits = [(tr, te) for tr, te in folds(d, q, 0)] if q != "PT" else [(np.where(~late)[0], np.where(late)[0])]
        for v, (cls, L, f, k) in V.items():
            Z = np.load(RES / f.format(n=name)); ch = [str(c) for c in Z[k.format(q=q)]]
            assert len(ch) == len(splits), (v, q, len(ch), len(splits))
            m26.L = L; R = Rs[L]
            for seed in (1, 2):
                p = np.full(n, np.nan, np.float32)
                for (tr, te), c in zip(splits, ch):
                    h, lr = c.split("/"); h = int(h); lr = float(lr)
                    p[te] = m31.fit_pred(R, lambda f_, h=h, c_=cls: c_(f_, h), tr, te, seed, lr)
                store[f"{q}|{v}|{seed}"] = p
                print(f"[{name}] {v:13s} s{seed} {q} rmse={np.sqrt(np.nanmean((p[late]-R0.y0[late])**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(out / f"{name}_tuned_seeds.npz", **store)
