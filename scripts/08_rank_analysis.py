"""Step 8 - inference on rank reversals from the predictions saved by step 7.

For every dataset / regime / protocol the pooled RMSE of each model is computed (mean squared error
averaged over repetitions first).  For every ordered pair of models (A, B):
    delta = RMSE_A - RMSE_B     (negative => A is better)
is estimated with a paired bootstrap over INDEPENDENT UNITS:
    P0, P2 : cells (cluster bootstrap)       P3 : held-out groups (batch / C-rate bin / HUST cell-identifier group)
A reversal between protocols X and Y is "supported" when the 95% interval of delta excludes 0 under X
and under Y with opposite signs.  With few held-out groups (G < 5) the bootstrap is not informative;
those cases are flagged and an exact sign test over groups is reported as well.

Usage: python 08_rank_analysis.py <step7_dir> <out_dir>
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kendalltau

N_BOOT = 2000
RNG = np.random.default_rng(0)
DATASETS = {  # tag -> (meta tag for P0/P2, [files])
    "matr": ("matr", "matr"), "matr_crate": ("matr_crate", "matr"),
    "hust": ("hust", "hust"), "xjtu": ("xjtu", "xjtu"), "tju": ("tju", "tju"),
}


def load_preds(dir_, tag, label_suffix=""):
    f = Path(dir_) / tag / f"{tag}{label_suffix}_preds.npz"
    return dict(np.load(f)) if f.exists() else {}


def err2_by(store, y, label, regime, protocol, model):
    keys = [k for k in store if k.startswith(f"{label}|{regime}|{protocol}|{model}|")]
    if not keys:
        return None
    e = np.vstack([(store[k].astype(float) - y) ** 2 for k in keys])
    return np.nanmean(e, axis=0)


def unit_sums(e2, units):
    s = pd.DataFrame({"u": units, "e": e2}).groupby("u")["e"].agg(["sum", "count"])
    return s["sum"].values, s["count"].values


def paired_delta(eA, eB, units):
    sA, n = unit_sums(eA, units)
    sB, _ = unit_sums(eB, units)
    point = np.sqrt(sA.sum() / n.sum()) - np.sqrt(sB.sum() / n.sum())
    k = len(n)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = RNG.integers(0, k, k)
        boots[b] = np.sqrt(sA[i].sum() / n[i].sum()) - np.sqrt(sB[i].sum() / n[i].sum())
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5), k


def group_signs(eA, eB, units):
    sA, n = unit_sums(eA, units)
    sB, _ = unit_sums(eB, units)
    rA, rB = np.sqrt(sA / n), np.sqrt(sB / n)
    wins = int((rA < rB).sum())
    return wins, len(n), binomtest(wins, len(n), 0.5).pvalue


def analyse(dir_, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_rmse, rows_pair, rows_tau, rows_rev = [], [], [], []
    for tag, (_, base) in DATASETS.items():
        meta_f = Path(dir_) / base / f"{base}_meta.csv.gz"
        if not (Path(dir_) / tag).exists():
            continue
        meta_tag = pd.read_csv(Path(dir_) / tag / f"{tag}_meta.csv.gz")
        stores = {"SOH_rated": load_preds(dir_, tag), "SOH_ref": load_preds(dir_, tag, "_ref")}
        # P0 / P2 predictions of matr_crate come from the matr run (same rows, same order)
        if tag == "matr_crate":
            base_store = load_preds(dir_, "matr")
            base_ref = load_preds(dir_, "matr", "_ref")
            for lab, bs in (("SOH_rated", base_store), ("SOH_ref", base_ref)):
                stores[lab] = {**{k: v for k, v in bs.items() if "|P3|" not in k}, **stores[lab]}
        for label, store in stores.items():
            if not store:
                continue
            y = meta_tag[label].values
            cells, grp = meta_tag["cell"].values, meta_tag["grp"].values
            regimes = sorted({k.split("|")[1] for k in store})
            for regime in regimes:
                models = sorted({k.split("|")[3] for k in store if k.split("|")[1] == regime})
                E = {}
                for p in ("P0", "P2", "P3"):
                    for m in models:
                        e = err2_by(store, y, label, regime, p, m)
                        if e is not None:
                            E[(p, m)] = e
                            rows_rmse.append({"dataset": tag, "label": label, "regime": regime,
                                              "protocol": p, "model": m,
                                              "rmse": float(np.sqrt(np.nanmean(e)))})
                prots = [p for p in ("P0", "P2", "P3") if any((p, m) in E for m in models)]
                # rank agreement between protocols
                for pa, pb in itertools.combinations(prots, 2):
                    ms = [m for m in models if (pa, m) in E and (pb, m) in E]
                    if len(ms) >= 4:
                        ra = [np.sqrt(np.nanmean(E[(pa, m)])) for m in ms]
                        rb = [np.sqrt(np.nanmean(E[(pb, m)])) for m in ms]
                        tau = kendalltau(ra, rb)[0]
                        rows_tau.append({"dataset": tag, "label": label, "regime": regime,
                                         "protocols": f"{pa} vs {pb}", "n_models": len(ms), "tau": tau,
                                         "top1_a": ms[int(np.argmin(ra))], "top1_b": ms[int(np.argmin(rb))]})
                # pairwise deltas with CIs
                for p in prots:
                    units = grp if p == "P3" else cells
                    for A, B in itertools.combinations([m for m in models if (p, m) in E], 2):
                        d, lo, hi, k = paired_delta(E[(p, A)], E[(p, B)], units)
                        rec = {"dataset": tag, "label": label, "regime": regime, "protocol": p,
                               "A": A, "B": B, "delta": d, "lo": lo, "hi": hi, "n_units": k,
                               "sign": 0 if lo <= 0 <= hi else (-1 if hi < 0 else 1)}
                        if p == "P3":
                            w, n, pv = group_signs(E[(p, A)], E[(p, B)], grp)
                            rec.update({"groups_A_better": w, "groups": n, "sign_test_p": pv})
                        rows_pair.append(rec)
    rmse, pair, tau = map(pd.DataFrame, (rows_rmse, rows_pair, rows_tau))
    rmse.to_csv(out / "rank_rmse.csv", index=False)
    pair.to_csv(out / "rank_pairs.csv", index=False)
    tau.to_csv(out / "rank_tau.csv", index=False)

    # supported reversals: same (A, B), opposite non-zero signs under two protocols
    for (ds, lb, rg, A, B), g in pair.groupby(["dataset", "label", "regime", "A", "B"]):
        s = g.set_index("protocol")
        for pa, pb in itertools.combinations(s.index, 2):
            if s.loc[pa, "sign"] * s.loc[pb, "sign"] == -1:
                rows_rev.append({"dataset": ds, "label": lb, "regime": rg, "A": A, "B": B,
                                 "protocol_1": pa, "delta_1": s.loc[pa, "delta"],
                                 "ci_1": f"[{s.loc[pa,'lo']:.2f},{s.loc[pa,'hi']:.2f}]",
                                 "units_1": s.loc[pa, "n_units"],
                                 "protocol_2": pb, "delta_2": s.loc[pb, "delta"],
                                 "ci_2": f"[{s.loc[pb,'lo']:.2f},{s.loc[pb,'hi']:.2f}]",
                                 "units_2": s.loc[pb, "n_units"]})
    rev = pd.DataFrame(rows_rev)
    rev.to_csv(out / "rank_supported_reversals.csv", index=False)
    return rmse, pair, tau, rev


if __name__ == "__main__":
    rmse, pair, tau, rev = analyse(sys.argv[1], sys.argv[2])
    pd.set_option("display.width", 220)
    for (ds, lb, rg), g in rmse.groupby(["dataset", "label", "regime"]):
        print(f"\n== {ds} / {lb} / {rg}: pooled RMSE")
        print(g.pivot(index="model", columns="protocol", values="rmse").round(2).to_string())
    print("\n== Kendall tau between protocol rankings\n", tau.round(2).to_string(index=False))
    print("\n== Supported reversals (opposite significant signs)\n",
          rev.round(2).to_string(index=False) if len(rev) else "none")
