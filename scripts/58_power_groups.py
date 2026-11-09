"""Step 58 - how many independent groups are needed for a claim about a model contrast? (replaces the unsupported 'about five groups' rule)

Units: the 70 charging policies of MATR (step 15, default regime, leave-one-policy-out predictions) and the 11 groups of TJU (step 7, P3).
For G in a grid, G units are drawn without replacement, the contrast of pooled RMSE between two models is computed on them and a percentile
cluster bootstrap interval (B = 500, resampling the G units) is formed.  Per G and model pair:
  power_perm   share of draws in which the sign-flip randomization test (src/inference.py) gives p < 0.05 with the sign of the contrast on all units
  wrong_perm   share with p < 0.05 and the opposite sign
  power_boot, wrong_boot, null_fpr_boot   the same for the 95 % percentile cluster bootstrap interval, and its rejection rate when the two models are exchanged at
               random within every unit (a null with no true difference, same unit weights and error scales); the randomization test has size 0.05 under that null by construction
Usage: python 58_power_groups.py <step15> <step7> <out>
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import perm_p  # noqa: E402

R_SIM, B = 400, 500
rng = np.random.default_rng(58)
MODELS = ["Ridge", "LightGBM", "XGBoost", "RandomForest"]


def unit_sse(preds, meta, key_fmt):
    y = meta["SOH_rated"].values; g = meta["grp"].values
    u, inv = np.unique(g, return_inverse=True)
    out = {}
    for m in MODELS:
        e = (preds[key_fmt.format(m)].astype(float) - y) ** 2
        out[m] = np.bincount(inv, weights=e, minlength=len(u))
    return out, np.bincount(inv, minlength=len(u)).astype(float)


def sim(S, n, a, b, G):
    U = len(n); full = np.sqrt(S[a].sum() / n.sum()) - np.sqrt(S[b].sum() / n.sum()); sgn = np.sign(full)
    hit = wrong = fp = ph = pw = 0
    for _ in range(R_SIM):
        idx = rng.choice(U, G, replace=False)
        Sa, Sb, nn = S[a][idx], S[b][idx], n[idx]
        bi = rng.integers(0, G, (B, G))
        d = np.sqrt(Sa[bi].sum(1) / nn[bi].sum(1)) - np.sqrt(Sb[bi].sum(1) / nn[bi].sum(1))
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        if lo > 0 or hi < 0:
            hit += int(np.sign(d.mean()) == sgn); wrong += int(np.sign(d.mean()) != sgn)
        pv = perm_p(Sa - Sb, draws=2000)
        if pv < 0.05:
            ph += int(np.sign(Sa.sum() - Sb.sum()) == sgn); pw += int(np.sign(Sa.sum() - Sb.sum()) != sgn)
        sw = rng.random(G) < 0.5
        Sa0, Sb0 = np.where(sw, Sb, Sa), np.where(sw, Sa, Sb)
        d0 = np.sqrt(Sa0[bi].sum(1) / nn[bi].sum(1)) - np.sqrt(Sb0[bi].sum(1) / nn[bi].sum(1))
        fp += int(np.percentile(d0, 2.5) > 0 or np.percentile(d0, 97.5) < 0)
    return full, hit / R_SIM, wrong / R_SIM, fp / R_SIM, ph / R_SIM, pw / R_SIM


def main(d15, d7, out):
    rows = []
    meta = pd.read_csv(Path(d15) / "matr_policy" / "matr_policy_meta.csv.gz"); pr = np.load(Path(d15) / "matr_policy" / "matr_policy_preds.npz")
    sets = [("matr_policy", *unit_sse(pr, meta, "SOH_rated|default|P3|{}|0"), [3, 4, 5, 6, 8, 10, 15, 20, 35, 70])]
    meta = pd.read_csv(Path(d7) / "tju" / "tju_meta.csv.gz"); pr = np.load(Path(d7) / "tju" / "tju_preds.npz")
    sets.append(("tju", *unit_sse(pr, meta, "SOH_rated|default|P3|{}|0"), [3, 4, 5, 6, 8]))
    for name, S, n, grid in sets:
        for a, b in itertools.combinations(MODELS, 2):
            for G in grid:
                full, pw, wr, fp, ph, pwr = sim(S, n, a, b, G)
                rows.append({"units": name, "A": a, "B": b, "G": G, "full_delta": full, "power_boot": pw, "wrong_boot": wr, "null_fpr_boot": fp, "power_perm": ph, "wrong_perm": pwr})
        print(name, "done", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    D = pd.DataFrame(rows); D.to_csv(Path(out) / "power_groups.csv", index=False)
    print(D.groupby(["units", "G"])[["power_perm", "wrong_perm", "power_boot", "null_fpr_boot"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
