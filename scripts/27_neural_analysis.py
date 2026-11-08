"""Step 27 - sequence models (step 26) against LightGBM and ridge under the five questions, on the same late rows.
RMSE with cluster-bootstrap intervals over cells (4000 resamples) and paired differences (neural - LightGBM, neural - Ridge).
LightGBM/Ridge squared errors: steps 7 (P0, P2, P3; mean over repetitions), 20 (PT; PR over its own random rows), 22 (CAL).
P0/P2/CAL partitions differ between models (different random partitions); P3 and PT are identical across models.
Usage: python 27_neural_analysis.py <step7> <step20> <step22> <step26> <out>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 4000
rng = np.random.default_rng(27)
QS = ["PR", "P0", "CAL0.0", "CAL0.2", "CAL0.6", "P2", "P3", "PT"]


def prot_err(store, protocol, m, y):
    keys = [k for k in store if k.startswith(f"SOH_rated|default|{protocol}|{m}|")]
    return np.nanmean(np.vstack([(store[k].astype(float) - y) ** 2 for k in keys]), axis=0)


def boot(cells, sq, w, fn):
    u, inv = np.unique(cells, return_inverse=True)
    S = {k: np.bincount(inv, weights=np.nan_to_num(v), minlength=len(u)) for k, v in sq.items()}
    N = {k: np.bincount(inv, weights=w[k], minlength=len(u)) for k in sq}
    pt = fn({k: np.sqrt(S[k].sum() / N[k].sum()) for k in S})
    bs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, len(u), len(u))
        bs[b] = fn({k: np.sqrt(S[k][i].sum() / max(N[k][i].sum(), 1e-9)) for k in S})
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main(d7, d20, d22, d26, out):
    rows, con = [], []
    for ds in ("matr", "hust", "xjtu", "tju"):
        f26 = Path(d26) / ds / f"{ds}_neural.npz"
        if not f26.exists():
            continue
        N = np.load(f26); T = np.load(Path(d20) / ds / f"{ds}_temporal.npz"); C = np.load(Path(d22) / ds / f"{ds}_calibration.npz")
        st = dict(np.load(Path(d7) / ds / f"{ds}_preds.npz"))
        meta = pd.read_csv(Path(d7) / ds / f"{ds}_meta.csv.gz"); cells = meta["cell"].values
        y, late = N["y"], N["late"]
        # the neural runner uses the same row order as step 7 (both load via 07_rank_experiment.load)
        assert np.allclose(y, T["y"])
        sq, w = {}, {}
        one = late.astype(float)
        for m in ("Ridge", "LightGBM"):
            sq[(m, "PR")] = np.mean([np.where(T[f"PR{s}|mask"], (T[f"PR{s}|{m}"].astype(float) - y) ** 2, 0) for s in range(3)], axis=0)
            w[(m, "PR")] = np.mean([T[f"PR{s}|mask"].astype(float) for s in range(3)], axis=0)
            for p in ("P0", "P2", "P3"):
                sq[(m, p)], w[(m, p)] = np.where(late, prot_err(st, p, m, y), 0), one
            sq[(m, "PT")], w[(m, "PT")] = np.where(late, (T[f"PT|{m}"].astype(float) - y) ** 2, 0), one
            for f in (0.0, 0.2, 0.6):
                k0 = f"0|{f}|{m}"
                if k0 in C.files:
                    e = np.nanmean([(C[f"{r}|{f}|{m}"].astype(float) - y) ** 2 for r in (0, 1)], axis=0)
                    ok = late & ~np.isnan(e)
                    sq[(m, f"CAL{f}")], w[(m, f"CAL{f}")] = np.where(ok, e, 0), ok.astype(float)
        for kind in ("GRU", "TRF"):
            for q in QS:
                key = f"{q}|{kind}"
                if key not in N.files:
                    continue
                p = N[key].astype(float)
                if q == "PR":
                    msk = N["PR|mask"]
                    sq[(kind, q)], w[(kind, q)] = np.where(msk, (p - y) ** 2, 0), msk.astype(float)
                else:
                    ok = late & ~np.isnan(p)
                    sq[(kind, q)], w[(kind, q)] = np.where(ok, (p - y) ** 2, 0), ok.astype(float)
            for f in (0.2, 0.6):
                key = f"CALFT{f}|{kind}"
                if key in N.files:
                    p = N[key].astype(float); ok = late & ~np.isnan(p)
                    sq[(kind + "_ft", f"CAL{f}")], w[(kind + "_ft", f"CAL{f}")] = np.where(ok, (p - y) ** 2, 0), ok.astype(float)
        for (m, q) in sq:
            pt, lo, hi = boot(cells, {"a": sq[(m, q)]}, {"a": w[(m, q)]}, lambda r: r["a"])
            rows.append({"dataset": ds, "model": m, "question": q, "rmse": pt, "lo": lo, "hi": hi})
        for kind in ("GRU", "TRF"):
            for q in QS:
                if q == "PR":
                    continue  # test rows differ between models (random masks): no paired contrast
                for ref in ("LightGBM", "Ridge"):
                    if (kind, q) in sq and (ref, q) in sq:
                        ww = w[(kind, q)] * w[(ref, q)]
                        pt, lo, hi = boot(cells, {"a": sq[(kind, q)] * ww, "b": sq[(ref, q)] * ww}, {"a": ww, "b": ww}, lambda r: r["a"] - r["b"])
                        con.append({"dataset": ds, "contrast": f"{kind} minus {ref} {q}", "diff": pt, "lo": lo, "hi": hi})
            for f in (0.2, 0.6):
                a, b = (kind + "_ft", f"CAL{f}"), (kind, f"CAL{f}")
                if a in sq and b in sq:
                    ww = w[a] * w[b]
                    pt, lo, hi = boot(cells, {"a": sq[a] * ww, "b": sq[b] * ww}, {"a": ww, "b": ww}, lambda r: r["a"] - r["b"])
                    con.append({"dataset": ds, "contrast": f"{kind} finetune minus concat f={f}", "diff": pt, "lo": lo, "hi": hi})
    Path(out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(out) / "neural_rmse.csv", index=False)
    pd.DataFrame(con).to_csv(Path(out) / "neural_contrasts.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).pivot_table(index=["dataset", "model"], columns="question", values="rmse")[[q for q in QS if q in set(pd.DataFrame(rows).question)]].round(2))
    print(pd.DataFrame(con).round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:6])
