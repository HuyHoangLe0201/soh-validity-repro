"""Exact paired sign-flip randomization test on per-unit error totals (unit = cell or held-out group).

Under the null that the two models are exchangeable within every unit, the sign of the per-unit difference of squared-error totals is symmetric.
All 2^U sign patterns are enumerated when U <= 14 (smallest attainable two-sided p = 2^(1-U)); otherwise 100 000 random sign patterns are drawn.
This replaces the percentile cluster bootstrap for p-values (the bootstrap is retained for intervals): step 59 shows that the percentile bootstrap
rejects a true null in 10-27 % of draws at 3-11 groups and 5-9 % at 55-135 cells, whereas this test keeps the nominal 5 %.
"""
import itertools

import numpy as np

_RNG = np.random.default_rng(2026)


def perm_p(dif, draws=100000):
    dif = np.asarray(dif, float); U = len(dif); t = abs(dif.sum()); tol = 1e-12 * max(t, 1e-300)
    if U <= 14:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=U)))
        return float(np.mean(np.abs(signs @ dif) >= t - tol))
    cnt = done = 0; chunk = 5000
    while done < draws:
        s = _RNG.choice((-1.0, 1.0), size=(chunk, U))
        cnt += int(np.sum(np.abs(s @ dif) >= t - tol)); done += chunk
    return (cnt + 1) / (done + 1)
