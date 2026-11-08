"""Step 30 - analysis of steps 28 (fine-tuning / calibration schedules) and 29 (architectures, window, seeds).
All scores on the last 40 % of the rows of every cell; pooled RMSE with cluster bootstrap over cells (4000 resamples).
Networks: squared errors averaged over seeds before pooling.  Usage: python 30_finetune_arch_analysis.py <step7> <step20> <step26> <step28> <step29> <out>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 4000
rng = np.random.default_rng(30)


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


def seed_mean_sq(store, prefix, y, seeds=(0, 1, 2)):
    arr = [(store[f"{prefix}|{s}"].astype(float) - y) ** 2 for s in seeds if f"{prefix}|{s}" in store]
    return np.nanmean(arr, axis=0) if arr else None


def main(d7, d20, d26, d28, d29, out):
    rows, con, arch, seedr = [], [], [], []
    for ds in ("matr", "hust", "xjtu", "tju"):
        meta = pd.read_csv(Path(d7) / ds / f"{ds}_meta.csv.gz"); cells = meta["cell"].values
        # ------------------------------------------------ A. fine-tuning schedules
        f28 = Path(d28) / ds / f"{ds}_finetune.npz"
        if f28.exists():
            S = np.load(f28); y, late = S["y"], S["late"]
            sq, w = {}, {}
            def add(name, e):
                ok = late & ~np.isnan(e)
                sq[name], w[name] = np.where(ok, e, 0.0), ok.astype(float)
            add("GRU_base", seed_mean_sq(S, "GRU_base", y))
            for m in ("Ridge", "LightGBM"):
                add(f"{m}_base", (S[f"{m}_base"].astype(float) - y) ** 2)
            for f in (0.2, 0.6):
                for v in ("concat", "head", "lowlr", "replay", "offset"):
                    add(f"GRU_{v}{f}", seed_mean_sq(S, f"GRU_{v}{f}", y))
                for m in ("Ridge", "LightGBM"):
                    add(f"{m}_offset{f}", (S[f"{m}_offset{f}"].astype(float) - y) ** 2)
            for k in sq:
                pt, lo, hi = boot(cells, {"a": sq[k]}, {"a": w[k]}, lambda r: r["a"])
                rows.append({"dataset": ds, "variant": k, "rmse": pt, "lo": lo, "hi": hi})
            def diff(a, b, label):
                ww = w[a] * w[b]
                pt, lo, hi = boot(cells, {"a": sq[a] * ww, "b": sq[b] * ww}, {"a": ww, "b": ww}, lambda r: r["a"] - r["b"])
                con.append({"dataset": ds, "contrast": label, "diff": pt, "lo": lo, "hi": hi})
            for f in (0.2, 0.6):
                for v in ("concat", "head", "lowlr", "replay", "offset"):
                    diff(f"GRU_{v}{f}", "GRU_base", f"GRU {v} f={f} minus base")
                diff(f"GRU_offset{f}", f"GRU_concat{f}", f"GRU offset minus concat f={f}")
                for m in ("Ridge", "LightGBM"):
                    diff(f"{m}_offset{f}", f"{m}_base", f"{m} offset f={f} minus base")
                diff(f"GRU_offset{f}", f"LightGBM_offset{f}", f"GRU offset minus LightGBM offset f={f}")
        # ------------------------------------------------ B. architectures, window, seeds
        f29 = Path(d29) / ds / f"{ds}_arch.npz"
        if f29.exists():
            A = np.load(f29); y, late = A["y"], A["late"]
            N26 = np.load(Path(d26) / ds / f"{ds}_neural.npz")
            T = np.load(Path(d20) / ds / f"{ds}_temporal.npz")
            st = dict(np.load(Path(d7) / ds / f"{ds}_preds.npz"))
            one = late.astype(float)
            def lg(q):
                if q == "PT":
                    return np.where(late, (T["PT|LightGBM"].astype(float) - y) ** 2, 0)
                keys = [k for k in st if k.startswith(f"SOH_rated|default|{q}|LightGBM|")]
                return np.where(late, np.nanmean(np.vstack([(st[k].astype(float) - y) ** 2 for k in keys]), axis=0), 0)
            variants = {}
            for v in ("LSTM", "TCN", "WinMLP", "GRU_L30"):
                variants[v] = {q: A[f"{q}|{v}"] for q in ("P0", "P2", "P3", "PT") if f"{q}|{v}" in A.files}
            for base, tag in (("GRU", "GRU"), ("TRF", "TRF")):
                srcs = {0: {q: N26[f"{q}|{base}"] for q in ("P0", "P2", "P3", "PT")}}
                for s in (1, 2):
                    srcs[s] = {q: A[f"{q}|{tag}_s{s}"] for q in ("P2", "P3", "PT") if f"{q}|{tag}_s{s}" in A.files}
                for q in ("P2", "P3", "PT"):
                    r = [np.sqrt(np.nanmean(((srcs[s][q].astype(float) - y)[late]) ** 2)) for s in (0, 1, 2) if q in srcs[s]]
                    seedr.append({"dataset": ds, "model": tag, "question": q, "min": min(r), "max": max(r), "n": len(r)})
                    variants.setdefault(f"{tag}_3seeds", {})[q] = np.nanmean([srcs[s][q].astype(float) for s in (0, 1, 2) if q in srcs[s]], axis=0)  # mean prediction
            # per-variant errors; ensembles (mean prediction over seeds) are reported separately from single runs
            for v, qd in variants.items():
                for q, p in qd.items():
                    e = np.where(late & ~np.isnan(p), (p.astype(float) - y) ** 2, 0)
                    ok = (late & ~np.isnan(p)).astype(float)
                    pt, lo, hi = boot(cells, {"a": e}, {"a": ok}, lambda r: r["a"])
                    arch.append({"dataset": ds, "variant": v, "question": q, "rmse": pt, "lo": lo, "hi": hi})
                    if q in ("P2", "P3", "PT", "P0"):
                        ref = lg(q)
                        ww = ok * one
                        pt2, lo2, hi2 = boot(cells, {"a": e * ww, "b": ref * ww}, {"a": ww, "b": ww}, lambda r: r["a"] - r["b"])
                        con.append({"dataset": ds, "contrast": f"{v} minus LightGBM {q}", "diff": pt2, "lo": lo2, "hi": hi2})
    Path(out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(out) / "finetune_rmse.csv", index=False)
    pd.DataFrame(con).to_csv(Path(out) / "finetune_arch_contrasts.csv", index=False)
    pd.DataFrame(arch).to_csv(Path(out) / "arch_rmse.csv", index=False)
    pd.DataFrame(seedr).to_csv(Path(out) / "arch_seeds.csv", index=False)
    pd.set_option("display.width", 220)
    r = pd.DataFrame(rows)
    if len(r): print(r.pivot_table(index=["dataset", "variant"], values="rmse").round(2).to_string())
    a = pd.DataFrame(arch)
    if len(a): print(a.pivot_table(index=["dataset", "variant"], columns="question", values="rmse").round(2).to_string())
    print(pd.DataFrame(seedr).round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:7])
