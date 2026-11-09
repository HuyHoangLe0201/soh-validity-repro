"""Compare a fresh run of the pipeline (from raw data) with stored results.
Usage: python compare_reproduction.py <stored_results_dir> <new_results_dir> <out_csv> [<stored_numbers.json> <new_numbers.json>]
For every csv / csv.gz file of the new run the numeric columns are compared with the stored file of the same relative path (max absolute and relative
difference over all cells); for every npz file the arrays of every key are compared (NaN-aware).  Status: identical (all differences below 1e-9),
close (relative difference below 1e-3 or absolute below 1e-3), differs.  If the two numbers.json files are given, their entries are compared as well.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def spread(a, b):
    """median and 99th percentile of the absolute difference over the finite entries of two arrays"""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "USO" or b.dtype.kind in "USO":
        return np.nan, np.nan
    a, b = a.astype(float), b.astype(float)
    if a.shape != b.shape or a.size == 0:
        return np.nan, np.nan
    d = np.abs(a - b); d = d[np.isfinite(d)]
    return (float(np.median(d)), float(np.percentile(d, 99))) if d.size else (0.0, 0.0)


def cmp_arrays(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "USO" or b.dtype.kind in "USO":   # text arrays (for example the chosen configurations): equal or not
        return (0.0, 0.0) if a.shape == b.shape and (a.astype(str) == b.astype(str)).all() else (np.inf, np.inf)
    a, b = a.astype(float), b.astype(float)
    if a.shape != b.shape:
        return np.inf, np.inf
    m = ~(np.isnan(a) & np.isnan(b))
    if np.isnan(a[m]).any() != np.isnan(b[m]).any() or (np.isnan(a) != np.isnan(b)).any():
        return np.inf, np.inf
    m = ~np.isnan(a)
    if not m.any():
        return 0.0, 0.0
    d = np.abs(a[m] - b[m]); rel = d / np.maximum(np.abs(a[m]), 1e-12)
    return float(d.max()), float(rel.max())


def status(ad, rd):
    if ad < 1e-9:
        return "identical"
    if rd < 1e-3 or ad < 1e-3:
        return "close"
    return "differs"


def main(old, new, out):
    old, new = Path(old), Path(new); rows = []
    for f in sorted(new.rglob("*")):
        if not f.is_file() or "_boot_inference_backup" in f.parts or f.name.startswith("cache_"):
            continue
        rel = f.relative_to(new); g = old / rel
        if f.suffix == ".npz":
            if not g.exists():
                rows.append({"file": str(rel), "kind": "npz", "status": "missing in stored"}); continue
            A, B = np.load(f), np.load(g); ad = rd = 0.0; keys = set(A.files) | set(B.files); bad = 0; skipped = 0; med, p99 = [], []
            for k in keys:
                if k not in A.files or k not in B.files:
                    bad += 1; continue
                try:
                    xa, xb = A[k], B[k]
                except ValueError:   # object arrays (not loaded: pickles are not read)
                    skipped += 1; continue
                x, y = cmp_arrays(xa, xb); ad, rd = max(ad, x), max(rd, y)
                m_, p_ = spread(xa, xb); med.append(m_); p99.append(p_)
            rows.append({"file": str(rel), "kind": "npz", "keys": len(keys), "key_mismatch": bad, "object_arrays_skipped": skipped, "max_abs": ad, "max_rel": rd,
                         "median_abs": float(np.nanmedian(med)) if med else np.nan, "p99_abs": float(np.nanmax(p99)) if p99 else np.nan,
                         "status": status(ad, rd) if not bad else "differs"})
        elif f.name.endswith((".csv", ".csv.gz")):
            if not g.exists():
                rows.append({"file": str(rel), "kind": "csv", "status": "missing in stored"}); continue
            A, B = pd.read_csv(f), pd.read_csv(g)
            if A.shape != B.shape:
                rows.append({"file": str(rel), "kind": "csv", "shape_new": str(A.shape), "shape_old": str(B.shape), "status": "differs"}); continue
            ad = rd = 0.0; txt_bad = 0
            for c in A.columns:
                if c not in B.columns:
                    txt_bad += 1; continue
                if pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c]):
                    x, y = cmp_arrays(A[c].values, B[c].values); ad, rd = max(ad, x), max(rd, y)
                else:
                    txt_bad += int((A[c].astype(str).values != B[c].astype(str).values).sum())
            rows.append({"file": str(rel), "kind": "csv", "rows": len(A), "text_mismatch": txt_bad, "max_abs": ad, "max_rel": rd, "status": status(ad, rd) if not txt_bad else "differs"})
    D = pd.DataFrame(rows); D.to_csv(out, index=False)
    print(D.groupby(["kind", "status"]).size().to_string())
    return D


if __name__ == "__main__":
    D = main(*sys.argv[1:4])
    if len(sys.argv) > 5:
        a, b = json.load(open(sys.argv[4])), json.load(open(sys.argv[5])); keys = sorted(set(a) | set(b)); same = close = diff = 0; bad = []
        for k in keys:
            x, y = str(a.get(k)), str(b.get(k))
            if x == y:
                same += 1; continue
            try:
                u, v = float(x.replace("\\mbox{$-$}", "-")), float(y.replace("\\mbox{$-$}", "-"))
                if abs(u - v) <= max(0.011, 0.02 * abs(u)):
                    close += 1; continue
            except ValueError:
                pass
            diff += 1; bad.append((k, x, y))
        print(f"numbers.json: {len(keys)} entries, identical {same}, within 0.01 or 2 percent {close}, different {diff}")
        pd.DataFrame(bad, columns=["key", "stored", "new"]).to_csv(Path(sys.argv[3]).with_name("numbers_differences.csv"), index=False)
