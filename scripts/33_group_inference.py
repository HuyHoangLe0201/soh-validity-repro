"""Step 33 - model comparisons with the independent unit and multiplicity control, for ridge, LightGBM and the networks.
Per dataset and question (P0, P2, P3, PT) every pair of models is compared on the last 40 % of each cell.  Unit of the cluster bootstrap:
the cell for P0, P2 and PT, the held-out group for P3 (MATR has 3 groups: no claims).  4000 resamples shared by all models; two-sided
bootstrap p-values; Holm and Benjamini-Hochberg within each (dataset, question) family.  A change of the sign of the RMSE difference
between two questions is 'supported' if both questions give Holm-adjusted p < 0.05 and opposite signs.
Usage: python 33_group_inference.py <step7> <step20> <step26> <step29> <step31> <out> [<step36>]
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = 4000
rng = np.random.default_rng(33)
QS = ["P0", "P2", "P3", "PT"]


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def bh(p):
    p = np.asarray(p); o = np.argsort(p)[::-1]; m = len(p); adj = np.empty(m); run = 1.0
    for r, i in enumerate(o):
        k = m - r; run = min(run, p[i] * m / k); adj[i] = run
    return adj


def errors(ds, d7, d20, d26, d29, d31, d36=None):
    meta = pd.read_csv(Path(d7) / ds / f"{ds}_meta.csv.gz")
    T = np.load(Path(d20) / ds / f"{ds}_temporal.npz"); N = np.load(Path(d26) / ds / f"{ds}_neural.npz"); A = np.load(Path(d29) / ds / f"{ds}_arch.npz")
    st = dict(np.load(Path(d7) / ds / f"{ds}_preds.npz"))
    y, late = T["y"], T["late"]
    E = {q: {} for q in QS}
    for m in ("Ridge", "LightGBM"):
        for q in ("P0", "P2", "P3"):
            ks = [k for k in st if k.startswith(f"SOH_rated|default|{q}|{m}|")]
            E[q][m] = np.nanmean(np.vstack([(st[k].astype(float) - y) ** 2 for k in ks]), axis=0)
        E["PT"][m] = (T[f"PT|{m}"].astype(float) - y) ** 2
    for m in ("GRU", "TRF"):
        for q in QS:
            E[q][m] = (N[f"{q}|{m}"].astype(float) - y) ** 2
    for v in ("LSTM", "TCN", "WinMLP", "GRU_L30"):
        for q in QS:
            E[q][v] = (A[f"{q}|{v}"].astype(float) - y) ** 2
    f31 = Path(d31) / ds / f"{ds}_tuned.npz"
    if f31.exists():
        Z = np.load(f31)
        for v in ("GRU_tuned", "CNNGRU", "AttnGRU"):
            for q in ("P2", "P3", "PT"):
                if f"{q}|{v}" in Z.files:
                    E[q][v] = (Z[f"{q}|{v}"].astype(float) - y) ** 2
    if d36:
        f36 = Path(d36) / ds / f"{ds}_tuned_more.npz"
        if f36.exists():
            Z = np.load(f36)
            for v in ("LSTM_tuned", "TCN_tuned", "WinMLP_tuned"):
                for q in ("P2", "P3", "PT"):
                    if f"{q}|{v}" in Z.files:
                        E[q][v] = (Z[f"{q}|{v}"].astype(float) - y) ** 2
    return meta, late, E


def main(d7, d20, d26, d29, d31, out, d36=None):
    pairs_all, rev_all = [], []
    for ds in ("matr", "hust", "xjtu", "tju"):
        meta, late, E = errors(ds, d7, d20, d26, d29, d31, d36)
        res = {}
        for q in QS:
            units = meta["grp"].values if q == "P3" else meta["cell"].values
            u, inv = np.unique(units, return_inverse=True)
            models = [m for m in E[q] if not np.isnan(E[q][m][late]).any()]
            S = {m: np.bincount(inv, weights=np.where(late, E[q][m], 0.0), minlength=len(u)) for m in models}
            Nn = np.bincount(inv, weights=late.astype(float), minlength=len(u))
            idx = rng.integers(0, len(u), (B, len(u)))
            br = {m: np.sqrt(S[m][idx].sum(1) / Nn[idx].sum(1)) for m in models}
            pt = {m: np.sqrt(S[m].sum() / Nn.sum()) for m in models}
            fam = []
            for a, b in itertools.combinations(models, 2):
                diff = br[a] - br[b]
                p = min(1.0, 2 * min((np.sum(diff <= 0) + 1) / (B + 1), (np.sum(diff >= 0) + 1) / (B + 1)))
                fam.append({"dataset": ds, "question": q, "A": a, "B": b, "delta": pt[a] - pt[b], "lo": np.percentile(diff, 2.5), "hi": np.percentile(diff, 97.5),
                            "p": p, "units": len(u)})
            f = pd.DataFrame(fam)
            f["p_holm"], f["p_bh"] = holm(f.p.values), bh(f.p.values)
            f["claimable"] = len(u) >= 5
            pairs_all.append(f)
            res[q] = f.set_index(["A", "B"])
        # supported sign changes between questions
        for (a, b) in res["P2"].index:
            for q1, q2 in itertools.combinations(QS, 2):
                if (a, b) in res[q1].index and (a, b) in res[q2].index:
                    r1, r2 = res[q1].loc[(a, b)], res[q2].loc[(a, b)]
                    if r1.claimable and r2.claimable and r1.p_holm < 0.05 and r2.p_holm < 0.05 and r1.delta * r2.delta < 0:
                        rev_all.append({"dataset": ds, "A": a, "B": b, "q1": q1, "delta1": r1.delta, "q2": q2, "delta2": r2.delta})
        print(f"[{ds}] done", flush=True)
    P = pd.concat(pairs_all); R = pd.DataFrame(rev_all)
    Path(out).mkdir(parents=True, exist_ok=True)
    P.to_csv(Path(out) / "model_pairs.csv", index=False); R.to_csv(Path(out) / "model_reversals.csv", index=False)
    pd.set_option("display.width", 220)
    s = P[P.claimable].groupby(["dataset", "question"]).agg(tests=("p", "size"), raw=("p", lambda x: int((x < .05).sum())), holm=("p_holm", lambda x: int((x < .05).sum())))
    print(s.to_string()); print("supported sign changes between questions:", len(R)); print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:7], *(sys.argv[7:8]))
