"""Faithful port of the CV-feature pipeline from paper_a_cv_features_fixed.ipynb.

The goal of this module is *reproduction*, not improvement: logic, thresholds and
filters are kept identical to the original notebook so that the published
623-sample dataset (B0005/B0006/B0007/B0018) is recovered exactly.

Deliberate non-changes (documented so later experiments can vary them):
  * NOMINAL_CAPACITY = 1.86 Ah  (maximum observed capacity, not the rated 2.0 Ah)
  * label = last value of the 'Capacity' array of the *next* discharge cycle
  * CV start = first sample with V >= 4.17 V
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import curve_fit

NOMINAL_CAPACITY = 1.86  # Ah
CV_VOLTAGE_THRESHOLD = 4.17  # V
CV_CURRENT_CUTOFF = 0.02  # A
BATTERY_FILES = ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"]

_trapz = getattr(np, "trapezoid", None) or np.trapz


def load_nasa_battery(filepath):
    mat = loadmat(filepath)
    name = os.path.basename(filepath).replace(".mat", "")
    cycles = mat[name]["cycle"][0, 0][0]
    return cycles, name


def get_cycle_type(cycle):
    try:
        t = cycle["type"]
        if isinstance(t, np.ndarray):
            t = t.flatten()[0]
        if isinstance(t, np.ndarray):
            t = t[0]
        return str(t)
    except Exception:
        return None


def get_cycle_data(cycle):
    try:
        data = cycle["data"][0, 0]
        names = data.dtype.names
        time_field = "Time" if "Time" in names else "time"
        return {
            "voltage": data["Voltage_measured"].flatten(),
            "current": data["Current_measured"].flatten(),
            "time": data[time_field].flatten(),
            "temperature": data["Temperature_measured"].flatten()
            if "Temperature_measured" in names
            else None,
            "capacity": data["Capacity"].flatten() if "Capacity" in names else None,
        }
    except Exception:
        return None


def load_all_batteries(data_path, battery_files=BATTERY_FILES):
    out = {}
    for bf in battery_files:
        fp = os.path.join(data_path, bf)
        cycles, name = load_nasa_battery(fp)
        charge, discharge = [], []
        for i, c in enumerate(cycles):
            ctype, cdata = get_cycle_type(c), get_cycle_data(c)
            if cdata is None:
                continue
            if ctype == "charge":
                charge.append({"idx": i, "data": cdata})
            elif ctype == "discharge":
                discharge.append({"idx": i, "data": cdata})
        out[name] = {"charge_cycles": charge, "discharge_cycles": discharge}
    return out


def extract_capacity_sequence(battery_data, nominal=NOMINAL_CAPACITY):
    cyc, cap = [], []
    for dc in battery_data["discharge_cycles"]:
        arr = dc["data"]["capacity"]
        if arr is not None and len(arr) > 0:
            c = arr[-1]
            if c > 0.5:
                cyc.append(dc["idx"])
                cap.append(c)
    cyc, cap = np.array(cyc), np.array(cap)
    return cyc, cap, cap / nominal * 100


def detect_cv_phase(voltage, current, time,
                    voltage_threshold=CV_VOLTAGE_THRESHOLD,
                    current_cutoff=CV_CURRENT_CUTOFF):
    mask = voltage >= voltage_threshold
    if not np.any(mask):
        return None, None, False, "voltage never reaches threshold"
    cv_start = int(np.argmax(mask))
    post = current[cv_start:]
    if len(post) < 10:
        return None, None, False, "not enough data after CV start"
    low = np.abs(post) <= current_cutoff
    if np.any(low):
        cv_end = cv_start + int(np.argmax(low))
    else:
        target = max(current_cutoff, post[0] * 0.1)
        below = np.abs(post) <= target
        cv_end = cv_start + int(np.argmax(below)) if np.any(below) else len(current) - 1
    if cv_end - cv_start < 20:
        return None, None, False, "CV phase too short"
    cvc = current[cv_start:cv_end]
    if len(cvc) > 10 and cvc[-1] > cvc[0] * 1.1:
        return None, None, False, "current not decreasing"
    return cv_start, cv_end, True, "OK"


def _exp_decay(t, I0, tau, I_off):
    return I0 * np.exp(-t / tau) + I_off


def extract_cv_features(cd, voltage_threshold=CV_VOLTAGE_THRESHOLD):
    v, i, t = cd["voltage"], cd["current"], cd["time"]
    s, e, ok, msg = detect_cv_phase(v, i, t, voltage_threshold=voltage_threshold)
    if not ok:
        return None, msg
    cv_time = t[s:e + 1] - t[s]
    cv_cur = i[s:e + 1]
    t_CV = cv_time[-1]
    t_CC = t[s] - t[0]
    ratio = t_CV / t_CC if t_CC > 0 else np.nan
    area = _trapz(np.abs(cv_cur), cv_time)
    used_fallback = False
    try:
        p0 = [np.abs(cv_cur[0] - cv_cur[-1]), t_CV / 3, np.abs(cv_cur[-1])]
        popt, _ = curve_fit(_exp_decay, cv_time, np.abs(cv_cur), p0=p0,
                            bounds=([0, 1, 0], [10, 10000, 5]), maxfev=5000)
        tau = popt[1]
    except Exception:
        used_fallback = True
        i0, i1 = np.abs(cv_cur[0]), np.abs(cv_cur[-1])
        i632 = i0 - 0.632 * (i0 - i1)
        k = int(np.argmin(np.abs(np.abs(cv_cur) - i632)))
        tau = cv_time[k] if k > 0 else t_CV / 3
    return {
        "t_CV": t_CV, "t_CC": t_CC, "t_CV_t_CC_ratio": ratio, "tau": tau,
        "area_CV": area, "I_cv_start": np.abs(cv_cur[0]),
        "dI_dt": (np.abs(cv_cur[-1]) - np.abs(cv_cur[0])) / t_CV if t_CV > 0 else 0,
        "tau_fallback": used_fallback,
    }, "OK"


def build_feature_dataset(all_batteries, nominal=NOMINAL_CAPACITY,
                          voltage_threshold=CV_VOLTAGE_THRESHOLD):
    """One row per charge cycle; label = capacity of the nearest *following*
    discharge cycle (distance <= 10 in raw cycle index)."""
    rows = []
    for name, bd in all_batteries.items():
        cyc, cap, soh = extract_capacity_sequence(bd, nominal)
        lookup = {int(c): (k, cp, s) for k, (c, cp, s) in enumerate(zip(cyc, cap, soh))}
        for cc in bd["charge_cycles"]:
            feats, _ = extract_cv_features(cc["data"], voltage_threshold)
            if feats is None:
                continue
            ci = cc["idx"]
            best, dmin = None, float("inf")
            for d in lookup:
                dist = d - ci
                if 0 < dist < dmin:
                    dmin, best = dist, d
            if best is None:
                for d in lookup:
                    dist = abs(d - ci)
                    if dist < dmin:
                        dmin, best = dist, d
            if best is None or dmin > 10:
                continue
            seq, cp, s = lookup[best]
            rows.append({"battery": name, "charge_cycle_idx": ci,
                         "discharge_cycle_idx": best, "cycle_seq": seq,
                         **feats, "capacity": cp, "SOH": s})
    return pd.DataFrame(rows)


def clean_dataset(df):
    """Same filters, same order, as the notebook."""
    d = df.dropna().copy()
    d = d[(d["tau"] > 0) & (d["tau"] < 10000)]
    d = d[(d["t_CV"] > 0) & (d["t_CV"] < 15000)]
    d = d[(d["SOH"] > 50) & (d["SOH"] <= 105)]
    d = d[d["t_CV_t_CC_ratio"] < 100]
    return d.reset_index(drop=True)
