"""Step 11 - multiple-comparison control for the rank-reversal claims of step 8.

Tests: for every dataset / label / regime ("family") and protocol, every model pair (A, B):
H0: RMSE_A = RMSE_B.  Two-sided bootstrap p-value from the paired unit bootstrap (units = cells for
P0/P2, held-out groups for P3; B = 4000 resamples, p floor = 1/(B+1)).  Holm (FWER) and Benjamini-Hochberg
(FDR) adjustments are applied within each family.  A reversal for a pair is "supported after correction"
when the two protocols give opposite signs and BOTH adjusted p-values are < 0.05.  P3 tests with fewer
than 6 held-out groups are excluded from reversal claims (the smallest two-sided sign-flip p-value is 2^(1-G), which exceeds 0.05 for G < 6).
P-values are sign-flip randomization p-values (src/inference.py); the earlier percentile-bootstrap p-values are kept in results/_boot_inference_backup.

Usage: python 11_multiplicity.py <step7_dir> <out_dir>
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import perm_p  # noqa: E402

B = 4000
RNG = np.random.default_rng(1)
TAGS = {"matr": "matr", "matr_crate": "matr", "hust": "hust", "xjtu": "xjtu", "tju": "tju"}


def boot_p(eA, eB, units):
    df = pd.DataFrame({"u": units, "a": eA, "b": eB}).groupby("u").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    a, b, n = df.a.values, df.b.values, df.n.values
    k = len(n)
    point = np.sqrt(a.sum() / n.sum()) - np.sqrt(b.sum() / n.sum())
    idx = RNG.integers(0, k, (B, k))
    bs = np.sqrt(a[idx].sum(1) / n[idx].sum(1)) - np.sqrt(b[idx].sum(1) / n[idx].sum(1))
    p = perm_p(a - b)   # exact / Monte-Carlo sign-flip randomization test (src/inference.py); the bootstrap is kept for intervals only
    return point, min(p, 1.0), k


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def bh(p):
    p = np.asarray(p)
    m = len(p)
    o = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for r, i in enumerate(o):
        run = min(run, p[i] * m / (m - r))
        adj[i] = run
    return adj


def main(dir7, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in ("matr", "matr_crate", "hust", "xjtu", "tju"):
        meta = pd.read_csv(Path(dir7) / ds / f"{ds}_meta.csv.gz")
        base = np.load(Path(dir7) / TAGS[ds] / f"{TAGS[ds]}_preds.npz")
        store = dict(base)
        if ds == "matr_crate":
            store = {k: v for k, v in store.items() if "|P3|" not in k}
            store.update(dict(np.load(Path(dir7) / ds / f"{ds}_preds.npz")))
        for label in ("SOH_rated",):
            y = meta[label].values
            for regime in ("default", "tuned"):
                models = sorted({k.split("|")[3] for k in store if k.startswith(f"{label}|{regime}|")})
                fam = []
                E = {}
                for p in ("P0", "P2", "P3"):
                    for m in models:
                        ks = [k for k in store if k.startswith(f"{label}|{regime}|{p}|{m}|")]
                        if ks:
                            E[(p, m)] = np.nanmean([(store[k].astype(float) - y) ** 2 for k in ks], axis=0)
                for p in ("P0", "P2", "P3"):
                    units = meta.grp.values if p == "P3" else meta.cell.values
                    for A, Bm in itertools.combinations([m for m in models if (p, m) in E], 2):
                        d, pv, k = boot_p(E[(p, A)], E[(p, Bm)], units)
                        fam.append({"dataset": ds, "regime": regime, "protocol": p, "A": A, "B": Bm,
                                    "delta": d, "p": pv, "units": k})
                f = pd.DataFrame(fam)
                f["p_holm"], f["p_bh"] = holm(f.p), bh(f.p)
                rows.append(f)
    allp = pd.concat(rows, ignore_index=True)
    allp.to_csv(out / "multiplicity_tests.csv", index=False)

    rev = []
    for (ds, rg, A, Bm), g in allp.groupby(["dataset", "regime", "A", "B"]):
        g = g[(g.protocol != "P3") | (g.units >= 6)].set_index("protocol")
        for p1, p2 in itertools.combinations(g.index, 2):
            if np.sign(g.loc[p1, "delta"]) * np.sign(g.loc[p2, "delta"]) < 0:
                rev.append({"dataset": ds, "regime": rg, "A": A, "B": Bm, "p1": p1, "p2": p2,
                            "delta1": g.loc[p1, "delta"], "delta2": g.loc[p2, "delta"],
                            "raw": bool(g.loc[p1, "p"] < .05 and g.loc[p2, "p"] < .05),
                            "holm": bool(g.loc[p1, "p_holm"] < .05 and g.loc[p2, "p_holm"] < .05),
                            "bh": bool(g.loc[p1, "p_bh"] < .05 and g.loc[p2, "p_bh"] < .05)})
    rv = pd.DataFrame(rev)
    rv.to_csv(out / "multiplicity_reversals.csv", index=False)
    pd.set_option("display.width", 220)
    print("Sign reversals between two protocols (P3 only if >= 6 groups); survivors by criterion")
    print(rv.groupby(["dataset", "regime"])[["raw", "holm", "bh"]].sum().join(
        rv.groupby(["dataset", "regime"]).size().rename("sign_flips")).to_string())
    print("\nSurvivors after Holm:")
    print(rv[rv.holm].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
