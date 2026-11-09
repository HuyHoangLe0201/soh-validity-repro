"""Generate every table, figure and in-text number of the manuscript directly from the result files."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent / "results"; T = Path("tables"); F = Path("figures"); NUM = {}
f2 = lambda x: f"{x:.2f}"
def tex_table(df, fmt, path, header=None):
    s = df.to_latex(index=False, escape=False, column_format=fmt, header=header if header else True)
    s = s.replace("\\toprule", "\\toprule").replace("\\begin{tabular}", "\\begin{tabular}")
    Path(path).write_text(s)

# ------------------------------------------------------------------ sizes
mat = pd.read_csv(R / "step4/matr_summary_clean.csv.gz", usecols=["cell", "batch"])
hus = pd.read_csv(R / "hust_summary.csv.gz", usecols=["cell"]); xj = pd.read_csv(R / "xjtu_features.csv.gz", usecols=["cell", "batch"])
tj = pd.read_csv(R / "step7/tju/tju_meta.csv.gz", usecols=["cell"])
nasa = pd.read_csv(R / "step1/cv_features_dataset_reproduced.csv", usecols=["battery"])
NUM.update(tju_cells=tj.cell.nunique(), tju_rows=len(tj), matr_cells=mat.cell.nunique(), matr_rows=len(mat), hust_cells=hus.cell.nunique(), xjtu_cells=xj.cell.nunique(), xjtu_rows=len(xj), nasa_rows=len(nasa))
sizes = pd.DataFrame([
 ["NASA PCoE (B0005/6/7/18)", "LCO", f"{nasa.battery.nunique()}", f"{len(nasa):,}", "-"],
 ["MATR (Severson batches 1--3)", "LFP/graphite", f"{mat.cell.nunique()}", f"{len(mat):,}", "3 batches; 5 C-rate bins"],
 ["HUST", "LFP/graphite", f"{hus.cell.nunique()}", "146,122", "10 identifier groups"],
 ["XJTU", "NCM", f"{xj.cell.nunique()}", f"{len(xj):,}", "6 batches"],
 ["TJU", "NCA, NCM, NCM+NCA", f"{tj.cell.nunique()}", f"{len(tj):,}", "11 type-by-condition groups"]], columns=["Dataset", "Chemistry", "Cells", "Rows", "Held-out groups (P3)"])
tex_table(sizes, "llrrl", T / "datasets.tex")

# ------------------------------------------------------------------ protocol table (default regime, F1, rated)
r = pd.read_csv(R / "step8/rank_rmse.csv"); r = r[(r.label == "SOH_rated") & (r.regime == "default")]
def val(ds, p, m):
    x = r[(r.dataset == ds) & (r.protocol == p) & (r.model == m)].rmse
    return f2(x.iloc[0]) if len(x) else "-"
def crate_val(ds, m):
    """P3 with charge-rate bins (step 52 for XJTU and TJU; step 7 for MATR)."""
    if ds == "matr": return val("matr_crate", "P3", m)
    if ds == "hust": return "n/a"
    mt = pd.read_csv(R / f"step52/{ds}_crate/{ds}_crate_meta.csv.gz"); st = np.load(R / f"step52/{ds}_crate/{ds}_crate_preds.npz")
    return f2(float(np.sqrt(np.nanmean((st[f"SOH_rated|default|P3|{m}|0"].astype(float) - mt.SOH_rated.values) ** 2))))
rows = []
for ds, name in [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]:
    for m in ["Ridge", "LightGBM", "XGBoost", "RandomForest", "MLP"]:
        rows.append([name if m == "Ridge" else "", m, val(ds, "P0", m), val(ds, "P2", m), val(ds, "P3", m), crate_val(ds, m)])
for _ds in ("xjtu", "tju"):
    for _m in ("LightGBM", "Ridge"): NUM[f"cr_{_ds}_{_m}"] = crate_val(_ds, _m)
tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "P0", "P2", "P3", "P3 (C-rate bins)"]), "llrrrr", T / "protocol_rmse.tex")
for ds in ("matr", "hust", "xjtu", "tju"):
    for m in ("LightGBM", "Ridge", "MLP"):
        a = {p: float(r[(r.dataset == ds) & (r.protocol == p) & (r.model == m)].rmse.iloc[0]) for p in ("P0", "P2", "P3")}
        NUM[f"{ds}_{m}_P0"], NUM[f"{ds}_{m}_P2"], NUM[f"{ds}_{m}_P3"] = [f"{a[p]:.2f}" for p in ("P0", "P2", "P3")]
        NUM[f"{ds}_{m}_ratio_P3_P0"] = round(a["P3"] / a["P0"], 1)

# ------------------------------------------------------------------ Holm survivors
h = pd.read_csv(R / "step11/multiplicity_reversals.csv"); hs = h[h.holm]
t = pd.DataFrame({"Dataset": hs.dataset.str.upper(), "Regime": hs.regime, "Pair (A, B)": hs.A + " -- " + hs.B,
                  "Protocols": hs.p1 + " vs " + hs.p2, "$\\Delta_1$": hs.delta1.map(f2), "$\\Delta_2$": hs.delta2.map(f2)})
tex_table(t, "lllllr", T / "holm.tex")
tab = h.groupby(["dataset", "regime"]).agg(flips=("raw", "size"), raw=("raw", "sum"), holm=("holm", "sum"), bh=("bh", "sum")).reset_index()
NUM["holm_total"] = int(h.holm.sum()); NUM["flips_total"] = int(len(h))
tex_table(tab.rename(columns={"dataset": "Dataset", "regime": "Regime", "flips": "Sign flips", "raw": "$p<.05$ (both)", "holm": "Holm", "bh": "BH"}).assign(Dataset=lambda d: d.Dataset.map({"hust":"HUST","matr":"MATR (batch)","matr_crate":"MATR (C-rate)","xjtu":"XJTU","tju":"TJU"})), "llrrrr", T / "holm_counts.tex")

# ------------------------------------------------------------------ tuning repeats
tr = pd.read_csv(R / "step14/tuning_repeats_P3.csv")
rows = []
for ds in ("xjtu", "matr_crate", "matr", "hust"):
    for m in ("Ridge", "LightGBM", "XGBoost", "RandomForest"):
        a = tr[(tr.dataset == ds) & (tr.model == m)].set_index("regime")
        if {"tuned", "tuned_grp"} <= set(a.index):
            rows.append([ds.replace("_", " ").upper() if m == "Ridge" else "", m, f2(a.loc["tuned", "default"]),
                         f"{a.loc['tuned','mean']:.2f} ({a.loc['tuned','sd']:.2f})", f"{a.loc['tuned_grp','mean']:.2f} ({a.loc['tuned_grp','sd']:.2f})"])
tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "Default", "Tuned, cell-held-out inner CV", "Tuned, group-held-out inner CV"]), "llrrr", T / "tuning.tex")

# ------------------------------------------------------------------ feature-label coupling and label definition (step 6 configuration)
rows = []
for ds, path in (("MATR", "step6/matr/matr_fast_protocol_pilot.csv"), ("HUST", "step6/hust/hust_fast_protocol_pilot.csv")):
    d = pd.read_csv(R / path); d = d[d.label == "SOH_rated"]
    for m in ("Ridge", "LightGBM"):
        row = [ds if m == "Ridge" else "", m]
        for fs in ("F1", "F3"):
            for p in ("P0", "P2", "P3a"):
                row.append(f2(d[(d.featset == fs) & (d.protocol == p) & (d.model == m)].rmse.iloc[0]))
        rows.append(row)
tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "P0", "P2", "P3a", "P0 ", "P2 ", "P3a "]), "llrrrrrr", T / "coupling.tex")
rows = []
for ds, name in [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]:
    for m in ("Ridge", "LightGBM"):
        row = [name if m == "Ridge" else "", m]
        for lab in ("SOH_rated", "SOH_ref"):
            q = pd.read_csv(R / "step8/rank_rmse.csv"); q = q[(q.label == lab) & (q.regime == "default") & (q.dataset == ds)]
            for p in ("P0", "P3"):
                x = q[(q.protocol == p) & (q.model == m)].rmse
                row.append(f2(x.iloc[0]) if len(x) else "-")
        rows.append(row)
tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "P0", "P3", "P0 ", "P3 "]), "llrrrr", T / "labels.tex")
_q = pd.read_csv(R / "step8/rank_rmse.csv"); _q = _q[(_q.regime == "default") & (_q.dataset == "tju")]
for _m in ("Ridge", "LightGBM"):
    for _lab, _nm in (("SOH_rated", "rated"), ("SOH_ref", "ref")):
        for _p in ("P0", "P3"):
            NUM[f"lab_tju_{_m}_{_nm}_{_p}"] = f2(_q[(_q.label == _lab) & (_q.protocol == _p) & (_q.model == _m)].rmse.iloc[0])

# ------------------------------------------------------------------ curve features
w = pd.read_csv(R / "step14/waveform_rmse.csv"); w = w[w.regime == "default"]
rows = []
for ds in ("matr", "hust", "xjtu"):
    for m in ("Ridge", "LightGBM", "XGBoost", "RandomForest"):
        x = w[(w.dataset == ds) & (w.model == m)].set_index("protocol").rmse
        rows.append([ds.upper() if m == "Ridge" else "", m] + [f2(x[p]) for p in ("P0", "P2", "P3")])
tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "P0", "P2", "P3"]), "llrrr", T / "curve.tex")
wr = pd.read_csv(R / "step14/waveform_reversals.csv"); NUM["curve_holm"] = int(wr.holm.sum())

# ------------------------------------------------------------------ support strata
s = pd.concat([pd.read_csv(R / "step10/support_strata.csv"), pd.read_csv(R / "step51/support_strata_tju.csv")], ignore_index=True); s["stratum"] = s["stratum"].astype(str); s = s[(s.regime == "default") & s.stratum.isin(["0", "1", "2"])]
rows = []
for ds, name in [("matr", "MATR (batch)"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]:
    for p in ("P0", "P2", "P3"):
        row = [name if p == "P0" else "", p]
        for st in ("0", "1", "2"):
            x = s[(s.dataset == ds) & (s.protocol == p) & (s.stratum == st)]
            row.append(f"{x.d_Ridge_minus_LightGBM.iloc[0]:+.2f} [{x.lo_Ridge_LightGBM.iloc[0]:+.2f}, {x.hi_Ridge_LightGBM.iloc[0]:+.2f}]" if len(x) else "-")
        rows.append(row)
tex_table(pd.DataFrame(rows, columns=["Dataset", "Protocol", "near", "intermediate", "far"]), "llrrr", T / "support.tex")
for p in ("P0", "P2", "P3"):
    for st, nm in (("0", "near"), ("2", "far")):
        x = s[(s.dataset == "tju") & (s.protocol == p) & (s.stratum == st)]; NUM[f"sup_tju_{p}_{nm}"] = f"{x.d_Ridge_minus_LightGBM.iloc[0]:+.2f}"

# ------------------------------------------------------------------ NASA pilot
rep = pd.read_csv(R / "step1/reproduction_table9.csv")
NUM["nasa_mean_increase"] = round(rep.increase_pct.mean(), 1)
rows = [[x.model, f"{x.pub_5fold:.3f}", f"{x.pub_lobo:.3f}", f"{x.rmse_5fold:.3f}", f"{x.rmse_lobo:.3f}", f"{x.increase_pct:.1f}\\%"] for x in rep.itertuples()]
tex_table(pd.DataFrame(rows, columns=["Model", "5-fold (pub.)", "LOBO (pub.)", "5-fold (repr.)", "LOBO (repr.)", "Increase"]), "lrrrrr", T / "nasa.tex")
b = pd.read_csv(R / "step3/block_size_curve_nasa.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for m, c in (("LightGBM", "tab:blue"), ("XGBoost", "tab:orange")):
    d = b[b.model == m]; ax.errorbar(d.block, d.rmse_mean, yerr=d.rmse_sd, marker="o", ms=3, lw=1, color=c, label=m)
    ax.axhline(d.lobo_ref.iloc[0], color=c, ls="--", lw=0.8)
ax.set_xscale("log", base=2); ax.set_xlabel("Test block length $b$ (cycles)"); ax.set_ylabel("RMSE (% SOH)"); ax.legend(fontsize=7, frameon=False)
ax.text(1.05, b[b.model=="XGBoost"].lobo_ref.iloc[0]+0.08, "LOBO", fontsize=6)
fig.tight_layout(); fig.savefig(F / "nasa_block.pdf"); fig.savefig(F / "nasa_block.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ protocol figure
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=False)
for ax, (ds, name) in zip(axs, [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU")]):
    for m, c in (("Ridge", "tab:gray"), ("LightGBM", "tab:blue"), ("XGBoost", "tab:orange"), ("RandomForest", "tab:green")):
        y = [float(r[(r.dataset == ds) & (r.protocol == p) & (r.model == m)].rmse.iloc[0]) for p in ("P0", "P2", "P3")]
        ax.plot(["P0", "P2", "P3"], y, marker="o", ms=3, lw=1, color=c, label=m)
    ax.set_title(name, fontsize=8); ax.tick_params(labelsize=7)
axs[0].set_ylabel("RMSE (% SOH)"); axs[0].legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(F / "protocols.pdf"); fig.savefig(F / "protocols.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ nested tuning (cell-held-out inner cross-validation) relative to fixed hyperparameters, by protocol
rr = pd.read_csv(R / "step8/rank_rmse.csv"); rr = rr[(rr.label == "SOH_rated") & rr.dataset.isin(["matr", "hust", "xjtu", "tju"])]
rows = []
for ds, name in [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]:
    for mdl in ("LightGBM", "XGBoost"):
        row = [name if mdl == "LightGBM" else "", mdl]
        for p in ("P0", "P2", "P3"):
            d0 = rr[(rr.dataset == ds) & (rr.regime == "default") & (rr.protocol == p) & (rr.model == mdl)].rmse.iloc[0]
            d1 = rr[(rr.dataset == ds) & (rr.regime == "tuned") & (rr.protocol == p) & (rr.model == mdl)].rmse.iloc[0]
            row.append(f"{100*(d1/d0-1):+.0f}\\%")
            NUM[f"tune_{ds}_{mdl}_{p}"] = f"{abs(100*(d1/d0-1)):.0f}"
        rows.append(row)
tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "P0", "P2", "P3"]), "llrrr", T / "tuning_by_protocol.tex")


# ------------------------------------------------------------------ experiments added on 2026-11-01 (steps 15-19)
S = Path(__file__).resolve().parent.parent / "results"
# (a) leave-one-policy-out on MATR (70 groups)
pr = pd.read_csv(S / "step19/policy_rmse.csv"); pt = pd.read_csv(S / "step19/policy_tuning.csv").set_index("model")
tail = pd.read_csv(S / "step19/policy_tail.csv", index_col=0); wins = pd.read_csv(S / "step19/policy_group_wins.csv").iloc[0]
tests = pd.read_csv(S / "step19/policy_tests.csv"); revp = pd.read_csv(S / "step19/policy_reversals.csv")
rows = []
for mdl in ("Ridge", "LightGBM", "XGBoost", "RandomForest"):
    g = lambda p: pr[(pr.protocol == p) & (pr.model == mdl)].rmse.iloc[0]
    rows.append([mdl, f2(g("P0")), f2(g("P2")), f2(g("P3")), f2(pt.loc[mdl, "tuned_group_inner_P3"]), f2(tail.loc[mdl, "median_group_rmse"]), f"{100*tail.loc[mdl,'top5_share']:.0f}\\%"])
tex_table(pd.DataFrame(rows, columns=["Model", "P0", "P2", "P3 (70 policies)", "P3, group-inner tuning", "Median per policy", "Top-5 share of squared error"]), "lrrrrrr", T / "policy.tex")
t_rl = tests[(tests.protocol == "P3") & (tests.A == "Ridge") & (tests.B == "LightGBM")].iloc[0]
NUM.update(policy_groups=70, policy_single=40, policy_ridge_wins=int(wins.ridge_wins), policy_ridge_p3=f2(pr[(pr.protocol=="P3")&(pr.model=="Ridge")].rmse.iloc[0]),
           policy_lgb_p3=f2(pr[(pr.protocol=="P3")&(pr.model=="LightGBM")].rmse.iloc[0]), policy_xgb_p3=f2(pr[(pr.protocol=="P3")&(pr.model=="XGBoost")].rmse.iloc[0]),
           policy_lgb_tg=f2(pt.loc["LightGBM","tuned_group_inner_P3"]), policy_xgb_tg=f2(pt.loc["XGBoost","tuned_group_inner_P3"]), policy_ridge_tg=f2(pt.loc["Ridge","tuned_group_inner_P3"]),
           policy_delta=f"{t_rl.delta:.2f}", policy_p=f"{t_rl.p:.2f}", policy_flips=len(revp), policy_raw=int(revp.raw.sum()),
           policy_med_lgb=f2(tail.loc["LightGBM","median_group_rmse"]), policy_med_ridge=f2(tail.loc["Ridge","median_group_rmse"]),
           policy_top5_lgb=f"{100*tail.loc['LightGBM','top5_share']:.0f}", policy_top5_ridge=f"{100*tail.loc['Ridge','top5_share']:.0f}",
           policy_trees_better=int(wins.groups - wins.ridge_wins))
# (b) selection on the reported folds
so = pd.read_csv(S / "step17/selection_optimism.csv"); so["opt"] = 100 * (so.nested_cell_inner - so.best_global) / so.nested_cell_inner
rows = []
for ds, name in [("matr", "MATR (batch)"), ("hust", "HUST"), ("xjtu", "XJTU")]:
    for mdl in ("Ridge", "LightGBM", "XGBoost", "RandomForest"):
        x = so[(so.dataset == ds) & (so.protocol == "P3") & (so.model == mdl)].iloc[0]
        rows.append([name if mdl == "Ridge" else "", mdl, f2(x.default), f2(x.nested_cell_inner), f2(x.nested_group_inner), f2(x.best_global), f"{x.opt:.0f}\\%"])
tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "Fixed", "Nested, cell inner", "Nested, group inner", "Selected on reported folds", "Optimism"]), "llrrrrr", T / "selection.tex")
for p in ("P0", "P2", "P3"):
    v = so[so.protocol == p].opt; NUM[f"opt_{p}_med"] = f"{v.median():.0f}"; NUM[f"opt_{p}_max"] = f"{v.max():.0f}"
x = so[(so.dataset == "xjtu") & (so.protocol == "P3") & (so.model == "XGBoost")].iloc[0]
NUM["opt_xjtu_xgb_nested"] = f2(x.nested_cell_inner); NUM["opt_xjtu_xgb_sel"] = f2(x.best_global); NUM["opt_xjtu_xgb_pct"] = f"{x.opt:.0f}"
mat = so[(so.dataset == "matr") & (so.protocol == "P3")]; NUM["opt_matr_P3_max"] = f"{mat.opt.max():.0f}"
hu = so[(so.dataset == "hust") & (so.protocol == "P3")]; NUM["opt_hust_P3_max"] = f"{hu.opt.max():.0f}"
# (c) preprocessing fitted on all rows (ridge)
pp = pd.read_csv(S / "step18/preprocessing_leakage.csv"); NUM["prep_max_abs"] = f"{pp['diff'].abs().max():.2f}"
NUM["prep_max_where"] = pp.loc[pp['diff'].abs().idxmax(), 'dataset'].upper() + " " + pp.loc[pp['diff'].abs().idxmax(), 'protocol']
NUM["prep_second"] = f"{pp['diff'].abs().sort_values().iloc[-2]:.5f}"
# (d) charge capacity as input on XJTU
f3 = S / "step16/xjtu_F3"; mt = pd.read_csv(f3 / "xjtu_F3_meta.csv.gz"); yy = mt.SOH_rated.values
s16 = dict(np.load(f3 / "xjtu_F3_preds.npz")); s7 = dict(np.load(S / "step7/xjtu/xjtu_preds.npz"))
def pooled(st, p, mdl):
    ks = [k for k in st if k.startswith(f"SOH_rated|default|{p}|{mdl}|")]
    return float(np.sqrt(np.nanmean(np.nanmean([(st[k].astype(float) - yy) ** 2 for k in ks], axis=0))))
rows = []
for mdl in ("Ridge", "LightGBM", "XGBoost", "RandomForest"):
    rows.append(["XJTU" if mdl == "Ridge" else "", mdl] + [f2(pooled(s7, p, mdl)) for p in ("P0", "P2", "P3")] + [f2(pooled(s16, p, mdl)) for p in ("P0", "P2", "P3")])
import importlib, sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts")); _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
_d, _ = importlib.import_module("07_rank_experiment").load("tju"); yy_t = _d["SOH_rated"].values
t7 = dict(np.load(S / "step7/tju/tju_preds.npz")); t51 = dict(np.load(S / "step51/tju_F3_preds.npz"))
def pooled_t(st, p, mdl):
    ks = [k for k in st if k.startswith(f"SOH_rated|default|{p}|{mdl}|")]
    return float(np.sqrt(np.nanmean(np.nanmean([(st[k].astype(float) - yy_t) ** 2 for k in ks], axis=0))))
for mdl in ("Ridge", "LightGBM", "XGBoost", "RandomForest"):
    rows.append(["TJU" if mdl == "Ridge" else "", mdl] + [f2(pooled_t(t7, p, mdl)) for p in ("P0", "P2", "P3")] + [f2(pooled_t(t51, p, mdl)) for p in ("P0", "P2", "P3")])
for mdl in ("LightGBM", "Ridge"):
    for p in ("P0", "P3"):
        NUM[f"f3t_{mdl}_F1_{p}"] = f2(pooled_t(t7, p, mdl)); NUM[f"f3t_{mdl}_F3_{p}"] = f2(pooled_t(t51, p, mdl))
NUM["f3t_corr"] = f"{pd.read_csv(S / 'step51/tju_qchg_vs_label.csv').corr_qchg_QD.iloc[0]:.3f}"
tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "P0", "P2", "P3", "P0 ", "P2 ", "P3 "]), "llrrrrrr", T / "f3xjtu.tex")
for mdl in ("LightGBM", "Ridge"):
    for p in ("P0", "P3"):
        NUM[f"f3x_{mdl}_F1_{p}"] = f2(pooled(s7, p, mdl)); NUM[f"f3x_{mdl}_F3_{p}"] = f2(pooled(s16, p, mdl))
qb = pd.read_csv(f3 / "qchg_vs_label_by_batch.csv").set_index("batch")
NUM["qb_corr_min_1to5"] = f"{qb.loc[1:5,'corr_qchg_QD'].min():.2f}"; NUM["qb_corr_6"] = f"{qb.loc[6,'corr_qchg_QD']:.2f}"; NUM["qb_diff_6"] = f"{abs(qb.loc[6,'mean_diff_Ah']):.2f}"

_b = pd.read_csv(R / "step47/batch_start_voltage.csv").set_index("batch")
NUM["bv_def6"] = f"{_b.loc[6,'deficit_mean_Ah']:.2f}"; NUM["bv_rho6"] = f"{_b.loc[6,'spearman_deficit_vstart']:.2f}"
NUM["bv_rho4"] = f"{_b.loc[4,'spearman_deficit_vstart']:.2f}"; NUM["bv_def4"] = f"{_b.loc[4,'deficit_mean_Ah']:.2f}"
NUM["bv_def13"] = f"{_b.loc[1:3,'deficit_mean_Ah'].abs().max():.3f}"
NUM["bv_sd6"] = f"{_b.loc[6,'v_start_sd']:.2f}"; NUM["bv_sd13"] = f"{_b.loc[1:3,'v_start_sd'].max():.2f}"

# MATR charging-policy overlap between batches (exact policy strings)
pol = pd.read_csv(R / "step4/matr_summary_clean.csv.gz", usecols=["cell", "batch", "policy"]).drop_duplicates("cell")
ps = {bb: set(g.policy) for bb, g in pol.groupby("batch")}
for bb in (1, 2, 3): NUM[f"matr_pol_b{bb}"] = len(ps[bb])
NUM["matr_pol_12"] = len(ps[1] & ps[2]); NUM["matr_pol_13"] = len(ps[1] & ps[3]); NUM["matr_pol_23"] = len(ps[2] & ps[3])
# ------------------------------------------------------------------ numbers for the prose
bk = b.set_index(["model", "block"])
for m, tag in (("XGBoost", "xgb"), ("LightGBM", "lgb")):
    NUM[f"block_{tag}_b1"] = f"{bk.loc[(m,1),'rmse_mean']:.2f}"; NUM[f"block_{tag}_b64"] = f"{bk.loc[(m,64),'rmse_mean']:.2f}"
    NUM[f"block_{tag}_lobo"] = f"{bk.loc[(m,1),'lobo_ref']:.2f}"
hs_ = pd.read_csv(R / "hust_summary.csv.gz", usecols=["q_chg", "QD"]); NUM["hust_qchg_sd_mAh"] = f"{(hs_.q_chg - hs_.QD).std()*1000:.1f}"
t_ = tr.set_index(["dataset", "model", "regime"])
def tv(ds, m, reg, col="mean"): return t_.loc[(ds, m, reg), col]
NUM["xjtu_xgb_default"] = f"{tv('xjtu','XGBoost','tuned','default'):.2f}"
NUM["xjtu_xgb_cell"] = f"{tv('xjtu','XGBoost','tuned'):.2f}"; NUM["xjtu_xgb_cell_sd"] = f"{tv('xjtu','XGBoost','tuned','sd'):.2f}"
NUM["xjtu_xgb_grp"] = f"{tv('xjtu','XGBoost','tuned_grp'):.2f}"; NUM["xjtu_xgb_grp_sd"] = f"{tv('xjtu','XGBoost','tuned_grp','sd'):.2f}"
NUM["crate_lgb_default"] = f"{tv('matr_crate','LightGBM','tuned','default'):.2f}"
NUM["crate_lgb_cell"] = f"{tv('matr_crate','LightGBM','tuned'):.2f}"; NUM["crate_lgb_grp"] = f"{tv('matr_crate','LightGBM','tuned_grp'):.2f}"; NUM["crate_ridge"] = f"{tv('matr_crate','Ridge','tuned_grp'):.2f}"
lab = pd.read_csv(R / "step8/rank_rmse.csv"); lab = lab[(lab.regime == "default") & (lab.protocol == "P0") & (lab.dataset == "hust")].set_index(["label", "model"]).rmse
NUM["hust_lgb_rated_p0"] = f"{lab[('SOH_rated','LightGBM')]:.2f}"; NUM["hust_lgb_ref_p0"] = f"{lab[('SOH_ref','LightGBM')]:.2f}"
NUM["hust_ridge_rated_p0"] = f"{lab[('SOH_rated','Ridge')]:.2f}"; NUM["hust_ridge_ref_p0"] = f"{lab[('SOH_ref','Ridge')]:.2f}"
cf = pd.read_csv(R / "step6/matr/matr_fast_protocol_pilot.csv"); cf = cf[(cf.label == "SOH_rated") & (cf.model == "LightGBM")].set_index(["featset", "protocol"]).rmse
NUM["matr_lgb_f1_p0"] = f"{cf[('F1','P0')]:.2f}"; NUM["matr_lgb_f3_p0"] = f"{cf[('F3','P0')]:.2f}"
wk = w.set_index(["dataset", "protocol", "model"]).rmse
for ds in ("matr", "xjtu", "hust"):
    for m in ("LightGBM", "Ridge"):
        for p in ("P0", "P3"): NUM[f"curve_{ds}_{m}_{p}"] = f"{wk[(ds,p,m)]:.2f}"

# ------------------------------------------------------------------ temporal vs random vs held-out (steps 20-21)
tr = pd.read_csv(R / "step21/temporal_rmse.csv"); tc = pd.read_csv(R / "step21/temporal_contrasts.csv"); tg = pd.read_csv(R / "step21/temporal_range.csv")
rows = []
for ds, name in [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]:
    for m in ["Ridge", "LightGBM", "XGBoost", "RandomForest"]:
        x = tr[(tr.dataset == ds) & (tr.model == m)].set_index("protocol")
        rows.append([name if m == "Ridge" else "", m] + [f2(x.loc[p, "rmse"]) for p in ("PR", "PT", "P0", "P2", "P3")])
        for p in ("PR", "PT", "P0", "P2", "P3"): NUM[f"tmp_{ds}_{m}_{p}"] = f2(x.loc[p, "rmse"])
tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "Random 60/40", "Temporal 60/40", "P0", "P2", "P3"]), "llrrrrr", T / "temporal.tex")
for _, r_ in tc.iterrows():
    k = r_.protocol.replace("/", "_").replace(" ", "_")
    NUM[f"tmpd_{r_.dataset}_{k}"] = f2(r_.ridge_minus_lgbm); NUM[f"tmpd_{r_.dataset}_{k}_lo"] = f2(r_.lo); NUM[f"tmpd_{r_.dataset}_{k}_hi"] = f2(r_.hi)
for _, r_ in tg.iterrows():
    NUM[f"tmprng_{r_.dataset}_below"] = f"{100 * r_.late_below_train_min:.0f}"
    NUM[f"tmprng_{r_.dataset}_lgbm_min"] = f"{r_.lgbm_pred_min:.1f}"; NUM[f"tmprng_{r_.dataset}_train_min"] = f"{r_.train_min:.1f}"
    NUM[f"tmprng_{r_.dataset}_true"] = f"{r_.late_mean:.1f}"; NUM[f"tmprng_{r_.dataset}_ridge"] = f"{r_.ridge_pred_mean:.1f}"; NUM[f"tmprng_{r_.dataset}_lgbm"] = f"{r_.lgbm_pred_mean:.1f}"

# ------------------------------------------------------------------ calibration with early target rows (steps 22, 25)
cr = pd.read_csv(R / "step25/calibration_rmse.csv"); cc = pd.read_csv(R / "step25/calibration_contrasts.csv")
rows = []
for ds, name in [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]:
    for m, lab in [("LightGBM", "LightGBM"), ("Ridge", "Ridge"), ("TargetRidge", "Target-only ridge")]:
        x = cr[(cr.dataset == ds) & (cr.model == m)].set_index("frac").rmse
        rows.append([name if m == "LightGBM" else "", lab] + [f2(x[f]) if f in x.index and x[f] == x[f] else "-" for f in (0.0, 0.1, 0.2, 0.4, 0.6)])
        for f in (0.0, 0.1, 0.2, 0.4, 0.6):
            if f in x.index and x[f] == x[f]: NUM[f"cal_{ds}_{m}_{f}".replace(".", "p")] = f2(x[f])
tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "$f=0$", "$f=0.1$", "$f=0.2$", "$f=0.4$", "$f=0.6$"]), "llrrrrr", T / "calibration.tex")
for _, r_ in cc.iterrows():
    k = r_.contrast.replace(" ", "_").replace("=", "").replace(".", "p")
    NUM[f"calc_{r_.dataset}_{k}"] = f2(r_["diff"]); NUM[f"calc_{r_.dataset}_{k}_lo"] = f2(r_.lo); NUM[f"calc_{r_.dataset}_{k}_hi"] = f2(r_.hi)
NUM["tju_groups"] = int(pd.read_csv(R / "step7/tju/tju_meta.csv.gz")["grp"].nunique())
NUM["tju_flips"] = int((h.dataset == "tju").sum())

# ------------------------------------------------------------------ figure: the same late rows under every evaluation question
LAD = [("PR", "Random split within cells"), ("P0", "Random folds"), ("CAL", "Early rows of target added"), ("P2", "New cell"), ("P3", "New group"), ("PT", "Forecast of later life")]
def lad(ds, m, k):
    if k == "CAL":
        x = cr[(cr.dataset == ds) & (cr.model == m) & (cr.frac == 0.6)].rmse; return float(x.iloc[0])
    x = tr[(tr.dataset == ds) & (tr.model == m) & (tr.protocol == k)].rmse; return float(x.iloc[0])
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.9), sharey=True)
cols = {"LightGBM": "#2a78d6", "Ridge": "#eb6834"}
for ax, (ds, name) in zip(axs, [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]):
    for j, (m, off) in enumerate((("Ridge", -0.2), ("LightGBM", 0.2))):
        v = [lad(ds, m, k) for k, _ in LAD]
        ax.bar(np.arange(len(LAD)) + off, v, width=0.36, color=cols[m], label=m, linewidth=0)
    ax.set_yscale("log"); ax.set_title(name, fontsize=8); ax.set_xticks(range(len(LAD)))
    ax.set_xticklabels([l for _, l in LAD], fontsize=6, rotation=45, ha="right", rotation_mode="anchor"); ax.tick_params(axis="y", labelsize=6.5)
    ax.set_yticks([0.1, 0.3, 1, 3, 10]); ax.set_yticklabels(["0.1", "0.3", "1", "3", "10"]); ax.minorticks_off(); ax.set_ylim(0.1, 12)
    ax.grid(axis="y", linewidth=0.3, color="#dddddd"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("RMSE (% SOH, log scale)", fontsize=7); h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=6.5, frameon=False, loc="upper right", ncol=2)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(F / "ladder.pdf"); fig.savefig(F / "ladder.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ sequence models under the five questions (steps 26-27)
if (R / "step27/neural_rmse.csv").exists():
    nr = pd.read_csv(R / "step27/neural_rmse.csv"); nc = pd.read_csv(R / "step27/neural_contrasts.csv")
    cols = [("PR", "Random split"), ("P0", "P0"), ("CAL0.6", "Early rows $f$=0.6"), ("P2", "P2"), ("P3", "P3"), ("PT", "Temporal")]
    rows = []
    for ds, name in [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]:
        for m, lab in [("Ridge", "Ridge"), ("LightGBM", "LightGBM"), ("GRU", "GRU"), ("TRF", "Transformer")]:
            x = nr[(nr.dataset == ds) & (nr.model == m)].set_index("question").rmse
            if not len(x): continue
            rows.append([name if m == "Ridge" else "", lab] + [f2(x[q]) if q in x.index else "-" for q, _ in cols])
            for q in x.index: NUM[f"nn_{ds}_{m}_{q}".replace(".", "p")] = f2(x[q])
        for m in ("GRU", "TRF"):
            x = nr[(nr.dataset == ds) & (nr.model == m + "_ft")].set_index("question").rmse
            for q in x.index: NUM[f"nn_{ds}_{m}_ft_{q}".replace(".", "p")] = f2(x[q])
    tex_table(pd.DataFrame(rows, columns=["Dataset", "Model"] + [c for _, c in cols]), "llrrrrrr", T / "neural.tex")
    for _, r_ in nc.iterrows():
        k = r_.contrast.replace(" ", "_").replace("=", "").replace(".", "p")
        NUM[f"nnd_{r_.dataset}_{k}"] = f2(r_["diff"]); NUM[f"nnd_{r_.dataset}_{k}_lo"] = f2(r_.lo); NUM[f"nnd_{r_.dataset}_{k}_hi"] = f2(r_.hi)

# ------------------------------------------------------------------ fine-tuning schedules and architectures (steps 28-30)
if (R / "step30/finetune_rmse.csv").exists():
    fr = pd.read_csv(R / "step30/finetune_rmse.csv"); fc = pd.read_csv(R / "step30/finetune_arch_contrasts.csv")
    ar = pd.read_csv(R / "step30/arch_rmse.csv"); sd = pd.read_csv(R / "step30/arch_seeds.csv")
    cols = [("GRU_base", "GRU base"), ("GRU_concat0.6", "concat"), ("GRU_head0.6", "head only"), ("GRU_lowlr0.6", "low lr"), ("GRU_replay0.6", "replay"),
            ("GRU_offset0.6", "offset"), ("LightGBM_base", "LGBM base"), ("LightGBM_offset0.6", "LGBM offset"), ("Ridge_base", "Ridge base"), ("Ridge_offset0.6", "Ridge offset")]
    rows = []
    for ds, name in [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]:
        x = fr[fr.dataset == ds].set_index("variant").rmse
        if not len(x): continue
        rows.append([name] + [f2(x[c]) for c, _ in cols])
        for v in x.index: NUM[f"ft_{ds}_{v}".replace(".", "p")] = f2(x[v])
    tex_table(pd.DataFrame(rows, columns=["Dataset"] + [c for _, c in cols]), "l" + "r" * len(cols), T / "finetune.tex")
    for _, r_ in fc.iterrows():
        k = r_.contrast.replace(" ", "_").replace("=", "").replace(".", "p")
        NUM[f"ftc_{r_.dataset}_{k}"] = f2(r_["diff"]); NUM[f"ftc_{r_.dataset}_{k}_lo"] = f2(r_.lo); NUM[f"ftc_{r_.dataset}_{k}_hi"] = f2(r_.hi)
    nr0 = pd.read_csv(R / "step27/neural_rmse.csv")
    order = [("Ridge", "Ridge"), ("LightGBM", "LightGBM"), ("GRU", "GRU (seed 0)"), ("TRF", "Transformer (seed 0)"), ("LSTM", "LSTM"), ("TCN", "TCN"), ("WinMLP", "Window MLP"), ("GRU_L30", "GRU, window 30")]
    rows = []
    for ds, name in [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]:
        for j, (m, lab) in enumerate(order):
            if m in ("Ridge", "LightGBM", "GRU", "TRF"):
                x = nr0[(nr0.dataset == ds) & (nr0.model == m)].set_index("question").rmse
            else:
                x = ar[(ar.dataset == ds) & (ar.variant == m)].set_index("question").rmse
            if not len(x): continue
            rows.append([name if j == 0 else "", lab] + [f2(x[q]) if q in x.index else "-" for q in ("P0", "P2", "P3", "PT")])
            for q in x.index: NUM[f"arch_{ds}_{m}_{q}"] = f2(x[q])
    tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "P0", "P2", "P3", "Temporal"]), "llrrrr", T / "arch.tex")
    for _, r_ in sd.iterrows():
        NUM[f"seed_{r_.dataset}_{r_.model}_{r_.question}_min"] = f2(r_["min"]); NUM[f"seed_{r_.dataset}_{r_.model}_{r_.question}_max"] = f2(r_["max"])

# ------------------------------------------------------------------ explanations tested (step 32)
if (R / "step32/range_split.csv").exists():
    rs = pd.read_csv(R / "step32/range_split.csv"); iv = pd.read_csv(R / "step32/extrapolation_intervention.csv")
    op = pd.read_csv(R / "step32/offset_persistence.csv"); od = pd.read_csv(R / "step32/offset_decomposition.csv")
    rows = []
    for ds, name in [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]:
        for m, lab in [("Ridge", "Ridge"), ("LightGBM", "LightGBM"), ("GRU", "GRU"), ("TCN", "TCN"), ("WinMLP", "Window MLP")]:
            x = rs[(rs.dataset == ds) & (rs.model == m)].iloc[0]
            f_ = lambda v: f2(v) if v == v else "n/a"
            rows.append([name if m == "Ridge" else "", lab, f"{100 * x.share_out_of_range:.0f}\\%", f_(x.rmse_in), f_(x.rmse_out), f_(x.rmse_all_late)])
        for _, x in rs[rs.dataset == ds].iterrows():
            NUM[f"rs_{ds}_{x.model}_in"] = f2(x.rmse_in) if x.rmse_in == x.rmse_in else "-"
            NUM[f"rs_{ds}_{x.model}_out"] = f2(x.rmse_out) if x.rmse_out == x.rmse_out else "-"
        NUM[f"rs_{ds}_share"] = f"{100 * rs[rs.dataset == ds].share_out_of_range.iloc[0]:.0f}"
    tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "Share below", "In range", "Out of range", "All rows"]), "llrrrr", T / "range_split.tex")
    for _, x in iv.iterrows(): NUM[f"iv_{x.dataset}_{x.model}_{x.question}"] = f2(x.rmse_late)
    for _, x in op.iterrows():
        NUM[f"op_{x.dataset}_{x.model}_rho"] = f2(x.spearman); NUM[f"op_{x.dataset}_{x.model}_slope"] = f2(x.slope)
    for _, x in od.iterrows():
        k = f"od_{x.dataset}_{x.model}"
        NUM[k + "_base"], NUM[k + "_median"], NUM[k + "_mean"], NUM[k + "_oracle"] = f2(x.rmse_base), f2(x.rmse_median_offset), f2(x.rmse_mean_offset), f2(x.rmse_oracle_offset)
        NUM[k + "_between"] = f"{100 * x.between_cell_share_of_mse:.0f}"

# ------------------------------------------------------------------ tuned GRU, further architectures, F2 inputs (step 31) and group-level inference (step 33)
if (R / "step31/xjtu/xjtu_tuned.npz").exists():
    def late_rmse(Z, key):
        y_, l_ = Z["y"], Z["late"]; p_ = Z[key].astype(float); return float(np.sqrt(np.nanmean((p_[l_] - y_[l_]) ** 2)))
    rows = []
    for ds, name in [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]:
        Z = np.load(R / f"step31/{ds}/{ds}_tuned.npz")
        Z6 = np.load(R / f"step36/{ds}/{ds}_tuned_more.npz") if (R / f"step36/{ds}/{ds}_tuned_more.npz").exists() else None
        for j, (m, lab) in enumerate([("GRU_tuned", "GRU, tuned"), ("CNNGRU", "CNN-GRU"), ("CNNGRU_tuned", "CNN-GRU, tuned"), ("AttnGRU", "GRU with attention"), ("AttnGRU_tuned", "GRU with attention, tuned"), ("LSTM_tuned", "LSTM, tuned"), ("TCN_tuned", "TCN, tuned"), ("WinMLP_tuned", "Window MLP, tuned"), ("TRF_tuned", "Transformer, tuned"), ("MLP_tuned", "Pooled MLP, tuned")]):
            if m == "TRF_tuned":
                f40 = R / f"step40/{ds}/{ds}_trf_tuned.npz"
                ZZ = np.load(f40) if f40.exists() else None
            elif m in ("CNNGRU_tuned", "AttnGRU_tuned"):
                f48 = R / f"step48/{ds}/{ds}_tuned_cnngru_attn.npz"
                ZZ = np.load(f48) if f48.exists() else None
            elif m == "MLP_tuned":
                f45 = R / f"step45/{ds}/{ds}_mlp_tuned.npz"
                ZZ = np.load(f45) if f45.exists() else None
            else:
                ZZ = Z if m in Z.files or f"P2|{m}" in Z.files else Z6
            if ZZ is None: continue
            vals = [late_rmse(ZZ, f"{q}|{m}") for q in ("P2", "P3", "PT")]
            rows.append([name if j == 0 else "", lab] + [f2(v) for v in vals])
            for q, v in zip(("P2", "P3", "PT"), vals): NUM[f"t31_{ds}_{m}_{q}"] = f2(v)
        ch = [str(c) for q in ("P2", "P3") for c in Z[f"chosen|{q}"]]
        NUM[f"t31_{ds}_chosen_n"] = len(ch); NUM[f"t31_{ds}_chosen_h64"] = sum(c.startswith("64") for c in ch)
    tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "P2", "P3", "Temporal"]), "llrrr", T / "tuned_arch.tex")
    rows = []
    for ds, name in [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU")]:
        Z = np.load(R / f"step31/{ds}/{ds}_f2.npz")
        y_, l_ = Z["y"], Z["late"]
        for j, (m, lab) in enumerate([("Ridge", "Ridge"), ("LightGBM", "LightGBM"), ("GRU", "GRU")]):
            vals = [late_rmse(Z, f"{q}|{m}") for q in ("P2", "P3", "PT")]
            rows.append([name if j == 0 else "", lab] + [f2(v) for v in vals])
            for q, v in zip(("P2", "P3", "PT"), vals): NUM[f"f2n_{ds}_{m}_{q}"] = f2(v)
    tex_table(pd.DataFrame(rows, columns=["Dataset", "Model (F2 inputs)", "P2", "P3", "Temporal"]), "llrrr", T / "f2_networks.tex")
if (R / "step33/model_pairs.csv").exists():
    mp = pd.read_csv(R / "step33/model_pairs.csv"); mr = pd.read_csv(R / "step33/model_reversals.csv")
    cl = mp[mp.claimable]
    NUM["g33_tests"] = int(len(cl)); NUM["g33_holm"] = int((cl.p_holm < .05).sum()); NUM["g33_rev_all"] = int(len(mr))
    core = ["Ridge", "LightGBM", "GRU", "TRF"]
    mc = mr[mr.A.isin(core) & mr.B.isin(core)]
    NUM["g33_rev_core"] = int(len(mc))
    for ds in ("matr", "hust", "xjtu", "tju"):
        NUM[f"g33_rev_core_{ds}"] = int((mc.dataset == ds).sum()); NUM[f"g33_rev_all_{ds}"] = int((mr.dataset == ds).sum())
    def pr(ds, q, a, b):
        x = mp[(mp.dataset == ds) & (mp.question == q) & (((mp.A == a) & (mp.B == b)) | ((mp.A == b) & (mp.B == a)))].iloc[0]
        s = 1 if x.A == a else -1
        return s * x.delta, (x.lo if s == 1 else -x.hi), (x.hi if s == 1 else -x.lo), x.p_holm, int(x.units)
    for q in ("P0", "P2", "P3", "PT"):
        d_, lo_, hi_, ph_, un_ = pr("xjtu", q, "TRF", "LightGBM")
        NUM[f"g33_xjtu_TRF_LGB_{q}"] = f2(d_); NUM[f"g33_xjtu_TRF_LGB_{q}_lo"] = f2(lo_); NUM[f"g33_xjtu_TRF_LGB_{q}_hi"] = f2(hi_); NUM[f"g33_xjtu_TRF_LGB_{q}_p"] = f"{ph_:.3f}"; NUM[f"g33_xjtu_TRF_LGB_{q}_n"] = un_
        d_, lo_, hi_, ph_, un_ = pr("hust", q, "GRU", "Ridge")
        NUM[f"g33_hust_GRU_Ridge_{q}"] = f2(d_); NUM[f"g33_hust_GRU_Ridge_{q}_lo"] = f2(lo_); NUM[f"g33_hust_GRU_Ridge_{q}_hi"] = f2(hi_); NUM[f"g33_hust_GRU_Ridge_{q}_p"] = f"{ph_:.3f}"

# ------------------------------------------------------------------ limitations round 3 (steps 34, 35, 37)
if (R / "step37/selection_extended.csv").exists():
    se = pd.read_csv(R / "step37/selection_extended.csv"); se["opt"] = 100 * (se.nested_cell_inner - se.best_global) / se.nested_cell_inner
    for pr in ("P0", "P2", "P3"):
        x = se[se.protocol == pr]
        NUM[f"sx_{pr}_med"] = f"{x.opt.median():.0f}"; NUM[f"sx_{pr}_max"] = f"{x.opt.max():.0f}"
    for ds in ("tju", "xjtu", "hust", "matr"):
        x = se[(se.dataset == ds) & (se.protocol == "P3")]
        NUM[f"sx_{ds}_P3_max"] = f"{x.opt.max():.0f}"
    for _, x in se[(se.dataset == "tju") & (se.protocol == "P3")].iterrows():
        NUM[f"sxt_tju_{x.model}_default"], NUM[f"sxt_tju_{x.model}_cell"], NUM[f"sxt_tju_{x.model}_grp"] = f2(x.default), f2(x.nested_cell_inner), f2(x.nested_group_inner)
    x = se[(se.dataset == "tju") & (se.protocol == "P3") & (se.model == "LightGBM")].iloc[0]
    NUM["sx_tju_lgb_nested"], NUM["sx_tju_lgb_sel"] = f2(x.nested_cell_inner), f2(x.best_global)
    NUM["sx_reps_P2_max_rep0"] = f"{se[(se.protocol == 'P2') & (se.rep == 0)].opt.max():.0f}"; NUM["sx_reps_P2_max_rep1"] = f"{se[(se.protocol == 'P2') & (se.rep == 1)].opt.max():.0f}"
if (R / "step34/matr124/matr124_rmse.csv").exists():
    m1 = pd.read_csv(R / "step34/matr124/matr124_rmse.csv")
    for _, x in m1.iterrows(): NUM[f"m124_{x.model}_{x.protocol}"] = f2(x.rmse)
if (R / "step35/temporal_concentration.csv").exists():
    tc2 = pd.read_csv(R / "step35/temporal_concentration.csv"); ci = pd.read_csv(R / "step35/cycle_number_intervention.csv")
    ob = pd.read_csv(R / "step35/offset_by_cell_length.csv"); fs2 = pd.read_csv(R / "step35/feature_shift.csv")
    for _, x in tc2.iterrows(): NUM[f"tc_{x.dataset}_top10"] = f"{100 * x.top10pct_cells_share_of_sse:.0f}"
    for _, x in ci.iterrows():
        NUM[f"cyc_{x.dataset}_{x.model}_{x.question}_F1"] = f2(x.rmse_F1); NUM[f"cyc_{x.dataset}_{x.model}_{x.question}_plus"] = f2(x.rmse_F1_plus_cycle)
    for _, x in ob.iterrows():
        k = f"ob_{x.dataset}_{x.length_tercile}"
        NUM[k + "_share"] = f"{100 * x.share_improved:.0f}"; NUM[k + "_base"] = f2(x.mse_base); NUM[k + "_adj"] = f2(x.mse_adj); NUM[k + "_rows"] = f"{100 * x.share_of_late_rows:.0f}"
    for _, x in fs2[fs2.dataset == "tju"].iterrows(): NUM[f"fs_tju_{x.feature}"] = f2(x.std_shift)
if (R / "step35/preproc_networks.csv").exists():
    pn = pd.read_csv(R / "step35/preproc_networks.csv")
    for _, x in pn.iterrows():
        NUM[f"pp_{x.dataset}_{x.question}_train"], NUM[f"pp_{x.dataset}_{x.question}_all"] = f2(x.train_rows), f2(x.all_rows)
    NUM["pp_min"] = f2((pn.all_rows - pn.train_rows).min()); NUM["pp_max"] = f2((pn.all_rows - pn.train_rows).max())

# ------------------------------------------------------------------ round 4 (steps 41, 42)
if (R / "step41/selfref_intervention.csv").exists():
    om = pd.read_csv(R / "step41/offset_mechanism.csv"); sf = pd.read_csv(R / "step41/selfref_intervention.csv")
    for _, x in om.iterrows():
        if x.life_tercile.startswith("all"):
            NUM[f"mm_{x.dataset}_lifecorr"] = f2(x.spearman_early_late); continue
        k = f"mm_{x.dataset}_{x.life_tercile}"
        NUM[k + "_rho"], NUM[k + "_slope"], NUM[k + "_same"] = f2(x.spearman_early_late), f2(x.slope), f"{100 * x.same_sign_share:.0f}"
        NUM[k + "_early"], NUM[k + "_late"] = f2(x.mean_early), f2(x.mean_late)
    for _, x in sf.iterrows():
        NUM[f"sr_{x.dataset}_{x.model}_{x.question}_F1"] = f2(x.rmse_F1); NUM[f"sr_{x.dataset}_{x.model}_{x.question}_plus"] = f2(x.rmse_F1_plus_selfref)
if (R / "step42/support_mahalanobis.csv").exists():
    s2 = pd.read_csv(R / "step42/support_mahalanobis.csv"); s1 = pd.read_csv(R / "step10/support_strata.csv")
    s1 = s1[(s1.regime == "default") & s1.stratum.isin(["0", "1", "2"])].copy(); s1["tercile"] = s1.stratum.astype(int)
    for _, x in s2.iterrows(): NUM[f"mh_{x.dataset}_{x.protocol}_{x.tercile}"] = f2(x.ridge_minus_lgbm)
    for _, x in s1.iterrows(): NUM[f"kn_{x.dataset}_{x.protocol}_{x.tercile}"] = f2(x.d_Ridge_minus_LightGBM)
    pm = pd.read_csv(R / "step42/policy_multicell.csv")
    for _, x in pm.iterrows():
        k = "pm_" + {"all policies": "all", "policies with >=3 cells": "multi", "policies with 1 cell": "single"}[x.subset]
        NUM[k + "_n"], NUM[k + "_R"], NUM[k + "_L"], NUM[k + "_d"], NUM[k + "_lo"], NUM[k + "_hi"], NUM[k + "_better"] = int(x.policies), f2(x.rmse_ridge), f2(x.rmse_lgbm), f2(x["diff"]), f2(x.lo), f2(x.hi), int(x.ridge_better_policies)

# ------------------------------------------------------------------ seeds for all networks (step 44)
if (R / "step44/seed_summary.csv").exists():
    ss = pd.read_csv(R / "step44/seed_summary.csv"); q3 = ss[ss.question.isin(["P2", "P3", "PT"])].copy(); q3["range"] = q3["max"] - q3["min"]
    for q in ("P2", "P3", "PT"):
        x = q3[q3.question == q]
        NUM[f"ss_{q}_cases"] = int(len(x)); NUM[f"ss_{q}_agree"] = int(x.sign_vs_lgbm_agree.sum()); NUM[f"ss_{q}_med"] = f2(x.range.median()); NUM[f"ss_{q}_max"] = f2(x.range.max())
    pt = ss[ss.question == "PT>P3 in seeds"]
    NUM["ss_ptp3_cases"] = int(len(pt)); NUM["ss_ptp3_consistent"] = int(((pt["mean"] == 0) | (pt["mean"] == 3)).sum())

# ------------------------------------------------------------------ seeds for the tuned variants (step 50)
if (R / "step50/tuned_seed_summary.csv").exists():
    ts = pd.read_csv(R / "step50/tuned_seed_summary.csv"); q3 = ts[ts.question.isin(["P2", "P3", "PT"])].copy(); q3["range"] = q3["max"] - q3["min"]
    for q in ("P2", "P3", "PT"):
        x = q3[q3.question == q]
        NUM[f"tss_{q}_cases"] = int(len(x)); NUM[f"tss_{q}_agree"] = int(x.sign_vs_lgbm_agree.sum()); NUM[f"tss_{q}_med"] = f2(x.range.median()); NUM[f"tss_{q}_max"] = f2(x.range.max())
    pt = ts[ts.question == "PT>P3 in seeds"]; cons = (pt["mean"] == 0) | (pt["mean"] == 3)
    NUM["tss_ptp3_cases"] = int(len(pt)); NUM["tss_ptp3_consistent"] = int(cons.sum())
    for ds in ("matr", "hust", "xjtu", "tju"):
        NUM[f"tss_ptp3_{ds}"] = int(cons[pt.dataset == ds].sum()); NUM[f"tss_ptp3_{ds}_n"] = int((pt.dataset == ds).sum())

# ------------------------------------------------------------------ F2 with tuned hyperparameters (step 43)
if (R / "step43/hust/hust_preds.npz").exists():
    for ds in ("matr", "hust"):
        S43 = np.load(R / f"step43/{ds}/{ds}_preds.npz"); M43 = pd.read_csv(R / f"step43/{ds}/{ds}_meta.csv.gz"); y43 = M43.SOH_rated.values
        D13 = np.load(R / f"step13/{ds}/{ds}_preds.npz")
        for p_ in ("P2", "P3"):
            for m_ in ("Ridge", "LightGBM"):
                kd = [k for k in D13.files if k.startswith(f"SOH_rated|default|{p_}|{m_}|")]
                NUM[f"f2t_{ds}_{m_}_{p_}_default"] = f2(float(np.sqrt(np.nanmean(np.nanmean([(D13[k].astype(float) - y43) ** 2 for k in kd], axis=0)))))
                NUM[f"f2t_{ds}_{m_}_{p_}_tuned"] = f2(float(np.sqrt(np.nanmean((S43[f"SOH_rated|tuned|{p_}|{m_}|0"].astype(float) - y43) ** 2))))

# ------------------------------------------------------------------ shrinkage and fade rate (step 46)
if (R / "step46/shrinkage_recalibration.csv").exists():
    for _, x in pd.read_csv(R / "step46/shrinkage_recalibration.csv").iterrows():
        NUM[f"shr_{x.dataset}_slope"] = f2(x.residual_on_soh_slope); NUM[f"shr_{x.dataset}_base"] = f2(x.rmse_late_base); NUM[f"shr_{x.dataset}_rec"] = f2(x.rmse_late_recalibrated)
    for _, x in pd.read_csv(R / "step46/fade_rate.csv").iterrows():
        NUM[f"fade_{x.dataset}_rho"] = f2(x.spearman_rmse_vs_abs_slope); NUM[f"fade_{x.dataset}_share"] = f"{100 * x.fastest_third_share_of_sse:.0f}"
# ------------------------------------------------------------------ round 5 (steps 53-60): factorial design, macro-averaged errors, group typology, selection over partitions, calibration of inference
DSN = {"matr": "MATR", "hust": "HUST", "xjtu": "XJTU", "tju": "TJU"}
if (R / "step56/factorial_contrasts.csv").exists():
    fr = pd.read_csv(R / "step56/factorial_rmse.csv"); fc = pd.read_csv(R / "step56/factorial_contrasts.csv"); fu = pd.read_csv(R / "step56/factorial_uncovered.csv")
    rows = []
    for ds in ("matr", "hust", "xjtu", "tju"):
        for m, ml in (("LightGBM", "LightGBM"), ("Ridge", "Ridge"), ("XGBoost", "XGBoost"), ("RandomForest", "Random forest")):
            g = fr[(fr.dataset == ds) & (fr.model == m)].set_index(["c", "f"]).pooled
            cc = fc[(fc.dataset == ds) & (fc.model == m) & (fc.scale == "pooled")].set_index("contrast")
            tot = cc.loc["total"]; sh = cc.loc["share_coverage"]
            share = f"{sh.estimate:.2f} ({sh.lo:.2f}, {sh.hi:.2f})" if tot.lo > 1.0 else "not defined"
            rows.append([DSN[ds] if m == "LightGBM" else "", ml, f2(g[(1.0, 0.0)]), f2(g[(0.6, 0.0)]), f2(g[(1.0, 0.6)]), f2(g[(0.6, 0.6)]), f"{tot.estimate:.2f} ({tot.lo:.2f}, {tot.hi:.2f})", share])
            for k in ("coverage_alone", "target_alone", "coverage_with_target", "target_at_short_pool", "total", "interaction"):
                NUM[f"fx_{ds}_{m}_{k}"] = f2(cc.loc[k].estimate); NUM[f"fx_{ds}_{m}_{k}_lo"] = f2(cc.loc[k].lo); NUM[f"fx_{ds}_{m}_{k}_hi"] = f2(cc.loc[k].hi)
            NUM[f"fx_{ds}_{m}_share"] = f2(sh.estimate); NUM[f"fx_{ds}_{m}_share_lo"] = f2(sh.lo); NUM[f"fx_{ds}_{m}_share_hi"] = f2(sh.hi)
            for cv in (0.6, 0.8, 1.0): NUM[f"fx_{ds}_{m}_c{int(cv * 10)}"] = f2(g[(cv, 0.0)])
            for k, (cv, ff) in {"A": (1.0, 0.0), "B": (0.6, 0.0), "C": (1.0, 0.6), "D": (0.6, 0.6)}.items(): NUM[f"fx_{ds}_{m}_{k}"] = f2(g[(cv, ff)])
        for _, u in fu[fu.dataset == ds].iterrows(): NUM[f"fx_{ds}_unc_{u.corner}"] = f"{100 * u.uncovered_share:.0f}"
    tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "A", "B", "C", "D", "D$-$A (95\\% CI)", "Coverage share"]), "llrrrrll", T / "factorial.tex")
    sh_all = fc[(fc.scale == "pooled") & (fc.contrast == "share_coverage") & (fc.model.isin(["LightGBM", "XGBoost", "RandomForest"]))]
    NUM["fx_share_min"] = f2(sh_all.estimate.min()); NUM["fx_share_max"] = f2(sh_all.estimate.max())
    ta = fc[(fc.scale == "pooled") & (fc.contrast == "target_alone") & (fc.model.isin(["LightGBM", "XGBoost", "RandomForest"]))]
    NUM["fx_target_abs_max"] = f2(ta.estimate.abs().max())
if (R / "step54/macro_rmse.csv").exists():
    mr_ = pd.read_csv(R / "step54/macro_rmse.csv"); ag = pd.read_csv(R / "step54/macro_agreement.csv"); mc_ = pd.read_csv(R / "step54/macro_contrasts.csv")
    qn = {"P0": "P0", "P2": "P2", "P3": "P3", "PT": "Temporal"}; rows = []
    for ds in ("matr", "hust", "xjtu", "tju"):
        for q in ("P0", "P2", "P3", "PT"):
            row = [DSN[ds] if q == "P0" else "", qn[q]]
            for m in ("LightGBM", "Ridge", "GRU"):
                x = mr_[(mr_.dataset == ds) & (mr_.question == q) & (mr_.model == m)].iloc[0]; row += [f2(x.pooled), f2(x.macro_cell)]
            a = ag[(ag.dataset == ds) & (ag.question == q)].iloc[0]; row += [f"{a.spearman:.2f}", f"{int(a.pair_flips)}/{int(a.pairs)}"]
            rows.append(row)
    tex_table(pd.DataFrame(rows, columns=["Dataset", "Question", "LGBM pooled", "LGBM macro", "Ridge pooled", "Ridge macro", "GRU pooled", "GRU macro", "Spearman", "Pairs flipped"]), "llrrrrrrrr", T / "macro.tex")
    for ds in ("matr", "hust", "xjtu", "tju"):
        for m in ("LightGBM", "Ridge"):
            x = mc_[(mc_.dataset == ds) & (mc_.contrast == "P3/P0 all rows") & (mc_.model == m)].iloc[0]; NUM[f"mac_{ds}_{m}_ratio"] = f"{x.estimate:.1f}"; NUM[f"mac_{ds}_{m}_ratio_lo"] = f"{x.lo:.1f}"; NUM[f"mac_{ds}_{m}_ratio_hi"] = f"{x.hi:.1f}"
        for m in ("LightGBM", "Ridge", "GRU", "TRF"):
            x = mc_[(mc_.dataset == ds) & (mc_.contrast == "PT-P3") & (mc_.model == m)]
            if len(x): x = x.iloc[0]; NUM[f"mac_{ds}_{m}_ptp3"] = f2(x.estimate); NUM[f"mac_{ds}_{m}_ptp3_lo"] = f2(x.lo); NUM[f"mac_{ds}_{m}_ptp3_hi"] = f2(x.hi)
        for q in ("P0", "P2", "P3", "PT"):
            for m in ("LightGBM", "Ridge"):
                x = mr_[(mr_.dataset == ds) & (mr_.question == q) & (mr_.model == m)].iloc[0]; NUM[f"mac_{ds}_{m}_{q}"] = f2(x.macro_cell); NUM[f"mac_{ds}_{m}_{q}_pooled"] = f2(x.pooled)
        a = ag[ag.dataset == ds]; NUM[f"mac_{ds}_flips"] = int(a.pair_flips.sum()); NUM[f"mac_{ds}_pairs"] = int(a.pairs.sum())
        x = mr_[(mr_.dataset == ds) & (mr_.question == "P3") & (mr_.model == "LightGBM")].iloc[0]; NUM[f"mac_{ds}_LightGBM_P3_group"] = f2(x.macro_group)
    NUM["mac_flips"] = int(ag.pair_flips.sum()); NUM["mac_pairs"] = int(ag.pairs.sum())
    NUM["mac_ptp3_ok"] = int(sum(float(NUM[f"mac_{ds}_LightGBM_ptp3_lo"]) > 0 for ds in ("matr", "hust", "xjtu", "tju")))
if (R / "step60/group_typology.csv").exists():
    gt = pd.read_csv(R / "step60/group_typology.csv"); rows = []
    for (ds, gl), g in gt.groupby(["dataset", "grouping"], sort=False):
        l = g[g.model == "LightGBM"].iloc[0]; rr = g[g.model == "Ridge"].iloc[0]
        cells = f"{l.cells_min}" if l.cells_min == l.cells_max else f"{l.cells_min}--{l.cells_max}"
        rows.append([DSN[ds], gl, l["shift"], int(l.groups), cells, f"{l.dist_ratio:.1f}", f2(l.pooled_P3), f"{l.ratio_P3_P2:.1f}", f2(rr.pooled_P3), f"{rr.ratio_P3_P2:.1f}"])
    d_ = pd.DataFrame(rows, columns=["Dataset", "Held-out group", "Type of shift", "Groups", "Cells per group", "Distance ratio", "LightGBM RMSE", "LightGBM P3/P2", "Ridge RMSE", "Ridge P3/P2"])
    d_["Dataset"] = d_["Dataset"].where(d_["Dataset"] != d_["Dataset"].shift(), "")
    tex_table(d_, "llllrrrrrr", T / "typology.tex")
    from scipy.stats import spearmanr
    l = gt[gt.model == "LightGBM"]; NUM["gt_rho_lgb"] = f2(spearmanr(l.dist_ratio, l.ratio_P3_P2)[0]); r_ = gt[gt.model == "Ridge"]; NUM["gt_rho_ridge"] = f2(spearmanr(r_.dist_ratio, r_.ratio_P3_P2)[0])
    NUM["gt_n"] = int(len(l))
    NUM["gt_max_ridge_ratio"] = f"{r_.ratio_P3_P2.max():.1f}"
    for _, x in l.iterrows(): NUM[f"gt_{x.dataset}_{x.grouping.split(' (')[0].replace(' ', '_').replace('-', '_')}_ratio"] = f"{x.ratio_P3_P2:.1f}"; NUM[f"gt_{x.dataset}_{x.grouping.split(' (')[0].replace(' ', '_').replace('-', '_')}_dist"] = f"{x.dist_ratio:.1f}"
if sorted(Path(R / "step57").glob("*_selection_partitions.csv")):
    sp = pd.concat([pd.read_csv(f) for f in sorted((R / "step57").glob("*_selection_partitions.csv"))], ignore_index=True)
    sp["opt"] = 100 * (sp.nested - sp.best_global) / sp.nested; sp["opt_def"] = 100 * (sp.default - sp.best_global) / sp.default; rows = []
    for ds in ("matr", "hust", "xjtu", "tju"):
        for m in ("Ridge", "LightGBM", "XGBoost"):
            x = sp[(sp.dataset == ds) & (sp.model == m)]
            if len(x):
                rows.append([DSN[ds] if m == "Ridge" else "", m, int(x.folds.iloc[0]), len(x), f"{x.default.mean():.2f} ({x.default.std():.2f})", f"{x.nested.mean():.2f} ({x.nested.std():.2f})", f"{x.best_global.mean():.2f} ({x.best_global.std():.2f})", f"{x.opt.mean():.0f} ({x.opt.min():.0f}, {x.opt.max():.0f})", f"{int((x.opt > 0).sum())}/{len(x)}"])
                NUM[f"sp_{ds}_{m}_opt"] = f"{x.opt.mean():.0f}"; NUM[f"sp_{ds}_{m}_min"] = f"{x.opt.min():.0f}"; NUM[f"sp_{ds}_{m}_max"] = f"{x.opt.max():.0f}"; NUM[f"sp_{ds}_{m}_pos"] = f"{int((x.opt > 0).sum())}"; NUM[f"sp_{ds}_{m}_n"] = len(x)
    tex_table(pd.DataFrame(rows, columns=["Dataset", "Model", "Folds", "Partitions", "Default", "Nested", "Selected on reported folds", "Optimism \\% (range)", "Partitions with optimism"]), "llrrllllr", T / "selection_partitions.tex")
    NUM["sp_reps"] = int(sp.groupby(["dataset", "model"]).size().max()); NUM["sp_med"] = f"{sp.opt.median():.0f}"; NUM["sp_max"] = f"{sp.opt.max():.0f}"; NUM["sp_pos"] = int((sp.opt > 0).sum()); NUM["sp_n"] = int(len(sp))
if (R / "step59/calibration_null.csv").exists():
    ca = pd.read_csv(R / "step59/calibration_null.csv"); g = ca.groupby(["dataset", "question"]).agg(units=("units", "first"), b=("boot_fpr", "mean"), p=("perm_fpr", "mean")).reset_index(); rows = []
    for ds in ("matr", "hust", "xjtu", "tju"):
        for q in ("P0", "P2", "P3", "PT"):
            x = g[(g.dataset == ds) & (g.question == q)]
            if len(x): x = x.iloc[0]; rows.append([DSN[ds] if q == "P0" else "", qn[q], int(x.units), f"{100 * x.b:.1f}", f"{100 * x.p:.1f}"])
    tex_table(pd.DataFrame(rows, columns=["Dataset", "Question", "Units", "Bootstrap (\\%)", "Randomization (\\%)"]), "llrrr", T / "calibration.tex")
    for q in ("P0", "P2", "PT"): NUM[f"cal_{q}_boot_max"] = f"{100 * g[g.question == q].b.max():.0f}"
    g3 = g[(g.question == "P3") & (g.units >= 5)]; NUM["cal_P3_boot_min"] = f"{100 * g3.b.min():.0f}"; NUM["cal_P3_boot_max"] = f"{100 * g3.b.max():.0f}"; NUM["cal_perm_max"] = f"{100 * g.p.max():.1f}"; NUM["cal_perm_min"] = f"{100 * g.p.min():.1f}"
    NUM["cal_cell_boot_min"] = f"{100 * g[g.question != 'P3'].b.min():.1f}"; NUM["cal_cell_boot_max"] = f"{100 * g[g.question != 'P3'].b.max():.1f}"
if (R / "step58/power_groups.csv").exists():
    pw = pd.read_csv(R / "step58/power_groups.csv"); rows = []
    for G in (3, 4, 5, 6, 8, 10, 15, 20, 35, 70):
        row = [G]
        for u in ("tju", "matr_policy"):
            x = pw[(pw.units == u) & (pw.G == G)]
            row += [f"{x.power_perm.mean():.2f}" if len(x) else "--", f"{x.power_boot.mean():.2f}" if len(x) else "--", f"{100 * x.null_fpr_boot.mean():.0f}" if len(x) else "--"]
        row.append(f"{2.0 ** (1 - G):.3f}")
        rows.append(row)
    tex_table(pd.DataFrame(rows, columns=["Groups", "TJU rand.", "TJU boot.", "TJU null (\\%)", "MATR rand.", "MATR boot.", "MATR null (\\%)", "Smallest $p$"]), "rrrrrrrr", T / "power.tex")
    x = pw[(pw.units == "tju")]
    for G in (5, 6, 8): NUM[f"pw_tju_{G}"] = f"{x[x.G == G].power_perm.mean():.2f}"
    NUM["pw_tju_boot6"] = f"{x[x.G == 6].power_boot.mean():.2f}"
    NUM["pw_null_g3"] = f"{100 * x[x.G == 3].null_fpr_boot.mean():.0f}"; NUM["pw_null_g8"] = f"{100 * x[x.G == 8].null_fpr_boot.mean():.0f}"
    y_ = pw[pw.units == "matr_policy"]; NUM["pw_matr_null_g70"] = f"{100 * y_[y_.G == 70].null_fpr_boot.mean():.0f}"; NUM["pw_matr_null_g3"] = f"{100 * y_[y_.G == 3].null_fpr_boot.mean():.0f}"
import math
for m_ in (1, 6, 91):
    NUM[f"gmin_{m_}"] = int(math.floor(1 + math.log2(m_ / 0.05))) + 1
if (R / "step33/model_reversals_main.csv").exists():
    rm_ = pd.read_csv(R / "step33/model_reversals_main.csv"); NUM["g33_main_rev"] = int(len(rm_))
    for ds in ("matr", "hust", "xjtu", "tju"): NUM[f"g33_main_rev_{ds}"] = int((rm_.dataset == ds).sum())
    NUM["g33_main_rev_p3"] = int(((rm_.q1 == "P3") | (rm_.q2 == "P3")).sum())
if (R / "step54/macro_agreement.csv").exists():
    _ag = pd.read_csv(R / "step54/macro_agreement.csv"); NUM["mac_pct"] = f"{100 * _ag.pair_flips.sum() / _ag.pairs.sum():.0f}"
    _m = _ag.loc[_ag.spearman.idxmin()]; NUM["mac_rho_min"] = f"{_m.spearman:.2f}"; NUM["mac_rho_min_where"] = f"{DSN[_m.dataset]} under {qn[_m.question]}"
    NUM["mac_rho_med"] = f"{_ag.spearman.median():.2f}"; NUM["mac_flips_max"] = f"{int(_ag.pair_flips.max())}"
    _m2 = _ag[_ag.question != "PT"].sort_values("spearman").iloc[1]; NUM["mac_rho_second"] = f"{_m2.spearman:.2f}"; NUM["mac_rho_second_where"] = f"{DSN[_m2.dataset]} under {qn[_m2.question]}"
if (R / "step33/model_pairs.csv").exists() and "p_holm_global" in pd.read_csv(R / "step33/model_pairs.csv", nrows=1).columns:
    _p = pd.read_csv(R / "step33/model_pairs.csv"); _p = _p[_p.claimable]
    NUM["gl_tests"] = int(len(_p)); NUM["gl_holm"] = int((_p.p_holm_global < .05).sum()); NUM["gl_bh"] = int((_p.p_bh_global < .05).sum()); NUM["gl_rev"] = int(len(pd.read_csv(R / "step33/model_reversals_global.csv")))
    NUM["fam_bh"] = int((_p.p_bh < .05).sum())
if (R / "step61/temporal_proportions.csv").exists():
    tp = pd.read_csv(R / "step61/temporal_proportions.csv"); rows = []
    for ds in ("matr", "hust", "xjtu", "tju"):
        for s in (0.4, 0.5, 0.6, 0.7, 0.8):
            a = tp[(tp.dataset == ds) & (tp.share == s) & (tp.model == "LightGBM")]; b = tp[(tp.dataset == ds) & (tp.share == s) & (tp.model == "Ridge")]
            if len(a) and len(b):
                a, b = a.iloc[0], b.iloc[0]; rows.append([DSN[ds] if s == 0.4 else "", f"{int(100 * s)}", f2(a.PT), f2(a.P3), f2(a.P2), f2(b.PT), f2(b.P3), f2(b.P2)])
    tex_table(pd.DataFrame(rows, columns=["Dataset", "Train share (\\%)", "LGBM PT", "LGBM P3", "LGBM P2", "Ridge PT", "Ridge P3", "Ridge P2"]), "llrrrrrr", T / "tprop.tex")
    _l = tp[tp.model == "LightGBM"]; NUM["tp_gt"] = int((_l.PT > _l.P3).sum()); NUM["tp_n"] = int(len(_l))
    NUM["tp_gt_p2"] = int((_l.PT > _l.P2).sum())
    for ds in ("matr", "hust", "xjtu", "tju"):
        x = _l[_l.dataset == ds]; NUM[f"tp_{ds}_gt"] = int((x.PT > x.P3).sum()); NUM[f"tp_{ds}_n"] = int(len(x)); NUM[f"tp_{ds}_min_ratio"] = f"{(x.PT / x.P3).min():.1f}"
NUM["nasa_pub_mean"] = "119"
NUM["dsg_a"] = f"{float(NUM['hust_lgb_ref_p0']) / float(NUM['hust_lgb_rated_p0']):.1f}"
NUM["dsg_c"] = f"{100 * (float(NUM['f3x_LightGBM_F3_P3']) / float(NUM['f3x_LightGBM_F1_P3']) - 1):.0f}"
NUM["dsg_e"] = f"{float(NUM['tmp_hust_LightGBM_PT']) / float(NUM['tmp_hust_LightGBM_P0']):.0f}"
Path("numbers.json").write_text(json.dumps(NUM, indent=1))
print(json.dumps(NUM, indent=1))
