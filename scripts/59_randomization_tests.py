"""Step 59 - calibration of the cluster bootstrap and exact paired randomization tests at the level of the independent unit.

Step 58 showed that the percentile cluster bootstrap rejects a true null too often when the error is concentrated in few units.  Here
(a) CALIBRATION: for the question/dataset settings of step 33 (unit = cell for P0, P2, PT; held-out group for P3) and the four main models, the two
    models of a pair are exchanged at random within every unit (a null with no difference between them), and the rejection rate of the 5 %
    bootstrap interval (B = 500, 300 draws) and of the sign-flip randomization test is recorded.
(b) TESTS: for every model pair of step 33 the paired sign-flip randomization test on the per-unit squared-error totals is computed
    (statistic: sum over units of s_u (SSE_a,u - SSE_b,u), s_u = +-1 at random; all 2^U sign patterns when U <= 14, 100 000 draws otherwise),
    with Holm and Benjamini-Hochberg adjustment inside every (dataset, question) family, and the supported sign changes between questions.
Usage: python 59_randomization_tests.py <step7> <step20> <step26> <step29> <step31> <out> <step36>
"""
import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
g33 = importlib.import_module("33_group_inference")
QS = g33.QS
rng = np.random.default_rng(59)
MAIN = ["Ridge", "LightGBM", "GRU", "TRF"]


def perm_p(dif, draws=100000):
    """two-sided exact/Monte-Carlo sign-flip p-value for the mean of the per-unit differences dif"""
    U = len(dif); t = abs(dif.sum())
    if U <= 14:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=U)))
        T = np.abs(signs @ dif)
        return float(np.mean(T >= t - 1e-12 * max(t, 1.0)))
    cnt = 0; done = 0; chunk = 5000
    while done < draws:
        s = rng.choice((-1.0, 1.0), size=(chunk, U)); T = np.abs(s @ dif)
        cnt += int(np.sum(T >= t - 1e-12 * max(t, 1.0))); done += chunk
    return (cnt + 1) / (done + 1)


def main(d7, d20, d26, d29, d31, out, d36):
    calib, pairs_all, rev_all = [], [], []
    for ds in ("matr", "hust", "xjtu", "tju"):
        meta, late, E = g33.errors(ds, d7, d20, d26, d29, d31, d36)
        res = {}
        for q in QS:
            units = meta["grp"].values if q == "P3" else meta["cell"].values
            u, inv = np.unique(units, return_inverse=True)
            models = [m for m in E[q] if not np.isnan(E[q][m][late]).any()]
            S = {m: np.bincount(inv, weights=np.where(late, E[q][m], 0.0), minlength=len(u)) for m in models}
            Nn = np.bincount(inv, weights=late.astype(float), minlength=len(u))
            fam = []
            for a, b in itertools.combinations(models, 2):
                dif = S[a] - S[b]
                p = perm_p(dif)
                fam.append({"dataset": ds, "question": q, "A": a, "B": b, "delta": float(np.sqrt(S[a].sum() / Nn.sum()) - np.sqrt(S[b].sum() / Nn.sum())),
                            "p_perm": p, "units": len(u)})
            f = pd.DataFrame(fam)
            f["p_holm"], f["p_bh"] = g33.holm(f.p_perm.values), g33.bh(f.p_perm.values)
            f["claimable"] = len(u) >= 6          # smallest unit count at which a two-sided sign-flip p-value can fall below 0.05 is 6
            pairs_all.append(f); res[q] = f.set_index(["A", "B"])
            # calibration on the four main models
            for a, b in itertools.combinations([m for m in MAIN if m in models], 2):
                fb = fp = 0
                for _ in range(300):
                    sw = rng.random(len(u)) < 0.5
                    Sa, Sb = np.where(sw, S[b], S[a]), np.where(sw, S[a], S[b])
                    bi = rng.integers(0, len(u), (500, len(u)))
                    d = np.sqrt(Sa[bi].sum(1) / Nn[bi].sum(1)) - np.sqrt(Sb[bi].sum(1) / Nn[bi].sum(1))
                    fb += int(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)
                    fp += int(perm_p(Sa - Sb, draws=2000) < 0.05)
                calib.append({"dataset": ds, "question": q, "A": a, "B": b, "units": len(u), "boot_fpr": fb / 300, "perm_fpr": fp / 300})
        for (a, b) in res["P2"].index:
            for q1, q2 in itertools.combinations(QS, 2):
                if (a, b) in res[q1].index and (a, b) in res[q2].index:
                    r1, r2 = res[q1].loc[(a, b)], res[q2].loc[(a, b)]
                    if r1.claimable and r2.claimable and r1.p_holm < 0.05 and r2.p_holm < 0.05 and r1.delta * r2.delta < 0:
                        rev_all.append({"dataset": ds, "A": a, "B": b, "q1": q1, "delta1": r1.delta, "q2": q2, "delta2": r2.delta})
        print(f"[{ds}] done", flush=True)
    P = pd.concat(pairs_all); R = pd.DataFrame(rev_all); C = pd.DataFrame(calib)
    Path(out).mkdir(parents=True, exist_ok=True)
    P.to_csv(Path(out) / "model_pairs_perm.csv", index=False); R.to_csv(Path(out) / "model_reversals_perm.csv", index=False); C.to_csv(Path(out) / "calibration_null.csv", index=False)
    old = pd.read_csv(Path(d7).parent / "step33" / "model_pairs.csv")
    M = P.merge(old[["dataset", "question", "A", "B", "p", "p_holm"]].rename(columns={"p": "p_boot", "p_holm": "p_holm_boot"}), on=["dataset", "question", "A", "B"])
    c = M[M.claimable]
    print("tests", len(c), "Holm perm", int((c.p_holm < .05).sum()), "Holm boot", int((c.p_holm_boot < .05).sum()))
    print(c.groupby(["dataset", "question"]).agg(tests=("p_perm", "size"), holm_perm=("p_holm", lambda x: int((x < .05).sum())), holm_boot=("p_holm_boot", lambda x: int((x < .05).sum()))).to_string())
    print("supported sign changes:", len(R), " among the four main models:", int(((R.A.isin(MAIN)) & (R.B.isin(MAIN))).sum()))
    print(C.groupby(["dataset", "question"]).agg(units=("units", "first"), boot_fpr=("boot_fpr", "mean"), perm_fpr=("perm_fpr", "mean")).round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:8])
