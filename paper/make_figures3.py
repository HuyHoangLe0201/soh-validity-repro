"""Overview and mechanism figures built from the stored results (paper/figures/).  Usage (venv python, from paper/): python make_figures3.py [names...]
F1 fig_overview, F2 fig_design, F3 fig_reading, F4 fig_resultmap, F5 fig_distance, F6 fig_extrap, F7 fig_offset, F8 fig_schedules, F9 fig_ranking.
Schematic parts (partition grids, flow boxes) are marked as such in the captions; every plotted number comes from results/ or tables/."""
import os
import json, re, sys, importlib
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import ListedColormap
ROOT = Path(__file__).resolve().parent.parent; F = Path("figures"); R = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6})
BLUE, ORANGE, GREY, GREEN, RED, PURPLE = "#3b6ea5", "#e08a3c", "#d9d9d9", "#4f9d6b", "#b8433f", "#7a5c9e"
NUM = json.load(open("numbers.json"))
DS = [("matr", "MATR"), ("hust", "HUST"), ("xjtu", "XJTU"), ("tju", "TJU")]
want = set(sys.argv[1:])
def do(n): return not want or n in want
def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03); fig.savefig(F / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.03); plt.close(fig); print("wrote", name, flush=True)
def plabel(ax, s, dx=-0.02, dy=1.03): ax.text(dx, dy, f"({s.lower()})", transform=ax.transAxes, fontsize=7.5, fontweight="bold", ha="right", va="bottom")
def read_tab(path):
    rows = []
    for l in open(path).read().split("\n"):
        if "&" in l and "Dataset" not in l:
            rows.append([c.strip().replace("\\\\", "").strip() for c in l.split("&")])
    return rows
def neural():
    out, cur = {}, None
    for r in read_tab("tables/neural.tex"):
        if r[0]: cur = r[0]
        out[(cur, r[1])] = [float(x) for x in r[2:8]]
    return out
NQ = ["Random split", "Random folds (P0)", "Early rows (f=0.6)", "New cell (P2)", "New group (P3)", "Forecast (PT)"]

# ====================================================================== F1 overview: partitions on real curves + measured errors
if do("fig_overview"):
    m7 = importlib.import_module("07_rank_experiment")
    d, _ = m7.load("matr"); d = d.reset_index(drop=True)
    life = d.groupby("cell").cycle_life.first()
    pick = []
    for b in (1, 2, 3):
        cs = life[d.groupby("cell").batch.first() == b].sort_values()
        pick += [cs.index[len(cs) // 4], cs.index[(3 * len(cs)) // 4]]
    batches = d.groupby("cell").batch.first()
    sub = d[d.cell.isin(pick)].copy(); sub["rk"] = sub.groupby("cell").cycle.rank(method="first"); sub["n"] = sub.groupby("cell").cycle.transform("size")
    rng = np.random.default_rng(1); sub["u"] = rng.random(len(sub))
    tgt = pick[3]  # a target cell for Q2 / Q3
    held_batch = batches[pick[2]]
    def role(q):
        if q == 1: return np.where(sub.u < 0.35, 2, 1)
        if q == 2: return np.where((sub.cell == tgt) & (sub.rk > np.floor(0.6 * sub.n)), 2, 1)
        if q == 3: return np.where(sub.cell == tgt, 2, 1)
        if q == 4: return np.where(sub.cell.map(batches) == held_batch, 2, 1)
        if q == 5: return np.where(sub.rk > np.floor(0.6 * sub.n), 2, 1)
    nr = neural()[("MATR", "LightGBM")]; nrr = neural()[("MATR", "Ridge")]
    cols = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    titles = ["Q1  interpolation", "Q2  calibration", "Q3  new cell", "Q4  new group", "Q5  forecast"]
    desc = ["test cycles lie between\ntraining cycles of the same cell", "early cycles of the target cell\nare added to training", "the target cell is absent\nfrom training", "a whole condition (batch)\nis absent from training", "the later life of every cell\nis the test set"]
    proto = ["P0 (random folds)", "early rows, f = 0.6", "P2 (cell-disjoint folds)", "P3 (held-out batch)", "PT (temporal split)"]
    fig = plt.figure(figsize=(7.2, 5.5)); gs = fig.add_gridspec(3, 5, height_ratios=[0.95, 2.0, 1.55], hspace=0.5, wspace=0.28, left=0.065, right=0.995, top=0.94, bottom=0.075)
    cmap = ListedColormap([GREY, BLUE, ORANGE]); nc, nt = 6, 14; r_ = np.random.default_rng(3)
    for q in range(1, 6):
        M = np.ones((nc, nt))
        if q == 1: M[r_.random((nc, nt)) < .35] = 2
        if q == 2: M[5, 8:] = 2
        if q == 3: M[5, :] = 2
        if q == 4: M[2:4, :] = 2
        if q == 5: M[:, 8:] = 2
        ax = fig.add_subplot(gs[0, q - 1]); ax.imshow(M, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(-.5, nt, 1), minor=True); ax.set_yticks(np.arange(-.5, nc, 1), minor=True); ax.grid(which="minor", color="white", lw=.6); ax.tick_params(which="both", length=0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[q - 1], fontsize=7.5, fontweight="bold", loc="left", pad=3); ax.set_xlabel(["P0, PR", "early rows ($f=0.6$)", "P2", "P3", "PT"][q - 1], fontsize=6.4, labelpad=2)
        if q == 1: ax.set_ylabel("Cells", fontsize=6)
        # real curves
        ax2 = fig.add_subplot(gs[1, q - 1]); rl = role(q)
        for c in pick:
            m_ = (sub.cell == c).values; ax2.plot(sub.cycle[m_], sub.SOH_rated[m_], color="#bbbbbb", lw=0.5, zorder=1)
        tr = rl == 1; te = rl == 2
        ax2.scatter(sub.cycle[tr], sub.SOH_rated[tr], s=1.2, color=BLUE, lw=0, zorder=2, rasterized=True); ax2.scatter(sub.cycle[te], sub.SOH_rated[te], s=1.2, color=ORANGE, lw=0, zorder=3, rasterized=True)
        ax2.set_ylim(60, 104); ax2.set_xlabel("Cycle"); ax2.tick_params(labelsize=5.5)
        if q == 1: ax2.set_ylabel("SOH (%)")
        else: ax2.set_yticklabels([])
        
        # measured errors
        ax3 = fig.add_subplot(gs[2, q - 1]); idx = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}[q]
        vals = [nrr[idx], nr[idx]]; b = ax3.bar([0, 1], vals, color=[GREEN, ORANGE], width=0.65)
        for x_, v_ in zip([0, 1], vals): ax3.text(x_, v_ + 0.12, f"{v_:.2f}", ha="center", fontsize=6)
        ax3.set_ylim(0, 10.6); ax3.set_xticks([0, 1]); ax3.set_xticklabels(["Ridge", "LightGBM"], fontsize=5.8)
        ax3.text(0.5, 0.82, f"$\\times${nr[idx]/nr[1]:.1f}", transform=ax3.transAxes, ha="center", fontsize=7.5, fontweight="bold", color=ORANGE if q > 1 else "#888")
        if q == 1: ax3.set_ylabel("Pooled RMSE (% SOH)")
        else: ax3.set_yticklabels([])
        if q == 1: plabel(ax, "A", -0.12, 1.12); plabel(ax2, "B", -0.2, 1.02); plabel(ax3, "C", -0.2, 1.02)
    fig.legend(handles=[Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color=ORANGE)], labels=["Training rows", "Test rows"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    save(fig, "fig_overview")

# ====================================================================== F2 detailed study design
def box(ax, x, y, w, h, title, lines, fc, tfs=6.8, fs=5.9, ec="#555"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=0.7))
    ax.text(x + 0.008, y + h - 0.016, title, fontsize=tfs, fontweight="bold", va="top", ha="left")
    ax.text(x + 0.008, y + h - 0.056, "\n".join(lines), fontsize=fs, va="top", ha="left", linespacing=1.32)
def arr(ax, x0, y0, x1, y1, txt=None, rad=0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#444", connectionstyle=f"arc3,rad={rad}"))
    if txt: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.012, txt, fontsize=5.4, ha="center", style="italic", color="#444")
def chip(ax, x, y, w, h, text, fc, fs=6.2, tc="black", ec="none", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%.2f" % (min(w, h) * 0.3), fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")
def thumb(ax, x, y, w, h, M):
    nr, nc = M.shape; cw, ch = w / nc, h / nr
    for a in range(nr):
        for b in range(nc):
            ax.add_patch(Rectangle((x + b * cw, y + h - (a + 1) * ch), cw, ch, fc=[GREY, BLUE, ORANGE][int(M[a, b])], ec="white", lw=0.35))
def stage(ax, x, w, n, title, top):
    ax.add_patch(plt.Circle((x + 1.6, top - 1.6), 1.45, fc="#333", ec="none")); ax.text(x + 1.6, top - 1.6, str(n), color="white", ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.text(x + 3.6, top - 1.6, title, ha="left", va="center", fontsize=7.8, fontweight="bold")
def chevron(ax, x, y):
    ax.add_patch(FancyArrowPatch((x, y), (x + 1.7, y), arrowstyle="-|>", mutation_scale=9, lw=1.2, color="#777"))
# fig_design is drawn in TikZ inside manuscript_template.tex (copy in figures/fig_design.tikz)

# ====================================================================== F3 reading overview
if do("fig_reading") and os.path.exists("manuscript_template.tex"):   # needs the manuscript text, which is not distributed
    t = open("manuscript_template.tex", encoding="utf-8").read(); i = t.index("\\multicolumn{3}{l}{\\textit{Whole cells held out}}"); j = t.index("\\bottomrule", i)
    parts = re.split(r"\\multicolumn\{3\}\{l\}\{\\textit\{(.*?)\}\} \\\\", t[i:j]); cats, cnt = [], []
    for k in range(1, len(parts), 2):
        cats.append(parts[k]); cnt.append(sum(1 for l in parts[k + 1].split("\n") if " & " in l and l.strip()))
    assert sum(cnt) == 40
    fig = plt.figure(figsize=(7.2, 2.3)); gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.35], wspace=0.55, left=0.01, right=0.985, top=0.9, bottom=0.17)
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 60); ax.set_ylim(0, 30); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 21), 60, 8, boxstyle="round,pad=0,rounding_size=1.5", fc="#d9e4f0", ec="none")); ax.text(30, 25, "60 studies included", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0, 4), 34, 13, boxstyle="round,pad=0,rounding_size=1.5", fc="#cfe5d3", ec="none")); ax.text(17, 10.5, "40 studies\ncharacterized", ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.add_patch(FancyBboxPatch((35.5, 4), 24.5, 13, boxstyle="round,pad=0,rounding_size=1.5", fc="#e6e6e6", ec="none")); ax.text(47.75, 10.5, "20 not\ncharacterized", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#555")
    ax.add_patch(FancyArrowPatch((17, 21), (17, 17.4), arrowstyle="-|>", mutation_scale=8, lw=1, color="#666")); ax.add_patch(FancyArrowPatch((47.75, 21), (47.75, 17.4), arrowstyle="-|>", mutation_scale=8, lw=1, color="#666"))
    ax.text(47.75, 1.6, "3 outside scope", ha="center", fontsize=5.8, color="#666")
    ax2 = fig.add_subplot(gs[1]); order = np.argsort(cnt)
    short = {"Whole cells held out": "Whole cells held out", "One to a few cells in the test set": "One to a few cells\nas test set", "Part of one cell's data used for training": "Part of the test cell\nused for training", "Proportional split of pooled records": "Proportional split\nof pooled records", "Other cell types or conditions in the test set": "Other cell types\nor conditions"}
    ax2.barh(np.arange(5), np.array(cnt)[order], color=BLUE, height=0.62); ax2.set_yticks(np.arange(5)); ax2.set_yticklabels([short[cats[o]] for o in order], fontsize=6)
    for k, o in enumerate(order): ax2.text(cnt[o] + 0.2, k, str(cnt[o]), va="center", fontsize=7)
    ax2.set_xlim(0, 13); ax2.set_xlabel("Number of studies"); ax2.spines[["top", "right"]].set_visible(False); ax2.tick_params(axis="y", length=0)
    plabel(ax2, "B", -0.45, 1.0); ax.text(0.0, 30, "(a)", fontsize=7.5, fontweight="bold", va="bottom")
    save(fig, "fig_reading")

# ====================================================================== F4 result map
if do("fig_resultmap"):
    nn = neural(); models = ["Ridge", "LightGBM", "GRU", "Transformer"]; short = ["Random\nsplit", "P0", "Early\nrows", "P2", "P3", "PT"]
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-1.2, vcenter=0, vmax=5.4); cm = plt.get_cmap("RdBu_r")
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.35), gridspec_kw={"wspace": 0.08})
    for ax, (k, nm) in zip(axs, DS):
        M = np.array([nn[(nm, m)] for m in models]); L = np.log2(M / M[:, [1]])
        im = ax.imshow(L, cmap=cm, norm=norm, aspect="auto")
        for i in range(4):
            for j in range(6): ax.text(j, i, f"{M[i, j]:.2f}" if M[i, j] < 10 else f"{M[i, j]:.1f}", ha="center", va="center", fontsize=5.6, color="white" if abs(L[i, j]) > 3 else "black")
        ax.set_xticks(range(6)); ax.set_xticklabels(short, fontsize=5.0); ax.set_title(nm, fontsize=7.5, fontweight="bold")
        ax.set_yticks(range(4)); ax.set_yticklabels(models if k == "matr" else [], fontsize=6.3); ax.tick_params(length=0)
        for e in (1.5, 3.5): ax.axvline(e, color="white", lw=0.8)
    cb = fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.018, pad=0.012); cb.set_label("log$_2$ (RMSE / RMSE under P0)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    save(fig, "fig_resultmap")

# ====================================================================== F5 distance mechanism
if do("fig_distance"):
    m7 = importlib.import_module("07_rank_experiment"); m10 = importlib.import_module("10_support_analysis")
    cache = Path("/tmp/dist_r.npz")
    if cache.exists(): Z = dict(np.load(cache))
    else:
        Z = {}
        for ds in ("matr", "xjtu", "tju"):
            d, feats = m7.load(ds); d = d.reset_index(drop=True)
            for p in ("P0", "P2", "P3"):
                r, _ = m10.support_ratio(d, feats, p); Z[f"{ds}|{p}"] = r.astype(np.float32); print(ds, p, flush=True)
        np.savez(cache, **Z)
    sc = pd.concat([pd.read_csv(R / "step10/support_strata.csv"), pd.read_csv(R / "step51/support_strata_tju.csv")], ignore_index=True); sc["stratum"] = sc["stratum"].astype(str)
    sc = sc[(sc.regime == "default") & sc.stratum.isin(["0", "1", "2"])]
    pc = {"P0": BLUE, "P2": ORANGE, "P3": RED}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), gridspec_kw={"hspace": 0.5, "wspace": 0.32})
    for j, (ds, nm) in enumerate([("matr", "MATR"), ("xjtu", "XJTU"), ("tju", "TJU")]):
        ax = axs[0, j]
        for p in ("P0", "P2", "P3"):
            r = Z[f"{ds}|{p}"]; r = np.sort(r[~np.isnan(r)]); ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=pc[p], lw=1.2, label=p)
        ax.set_xscale("log"); ax.set_xlim(0.5, 60); ax.set_xlabel("Distance ratio $r$"); ax.set_ylabel("Cumulative share of held-out rows" if j == 0 else ""); ax.set_title(nm, fontsize=7.5, fontweight="bold", loc="left"); ax.axvline(1, color="k", lw=0.4, ls=":")
        ax = axs[1, j]
        for p in ("P0", "P2", "P3"):
            x = sc[(sc.dataset == ds) & (sc.protocol == p)].sort_values("stratum")
            if len(x) == 3:
                ax.plot(range(3), x.rmse_LightGBM, color=pc[p], marker="o", ms=3, lw=1.1); ax.plot(range(3), x.rmse_Ridge, color=pc[p], marker="s", ms=2.5, lw=0.9, ls="--", alpha=0.8)
            else:
                xs = [int(v) for v in x.stratum]; ax.plot(xs, x.rmse_LightGBM, color=pc[p], marker="o", ms=3, lw=1.1); ax.plot(xs, x.rmse_Ridge, color=pc[p], marker="s", ms=2.5, lw=0.9, ls="--", alpha=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(["near", "intermediate", "far"]); ax.set_xlabel("Tercile of distance ratio $r$"); ax.set_ylabel("RMSE (% SOH)" if j == 0 else ""); ax.set_yscale("log"); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.set_yticks([t for t in [0.3, 0.5, 1, 2, 3, 5, 10, 15] if ax.get_ylim()[0] * 0.9 <= t <= ax.get_ylim()[1] * 1.1])
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=pc[p], lw=1.6) for p in ("P0", "P2", "P3")] + [Line2D([0], [0], color="#444", marker="o", ms=3, lw=1.1), Line2D([0], [0], color="#444", marker="s", ms=2.5, lw=0.9, ls="--")]
    fig.legend(h, ["P0", "P2", "P3", "LightGBM", "Ridge regression"], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=6.5)
    for a, l in zip(axs.flat, "ABCDEF"): plabel(a, l, -0.13 if l in "AD" else -0.1, 1.02)
    save(fig, "fig_distance")

# ====================================================================== F6 extrapolation mechanism
if do("fig_extrap"):
    m7 = importlib.import_module("07_rank_experiment"); rs = pd.read_csv(R / "step32/range_split.csv")
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.4), gridspec_kw={"hspace": 0.5, "wspace": 0.32, "height_ratios": [1, 1, 0.9]})
    for j, (ds, nm) in enumerate([("hust", "HUST"), ("xjtu", "XJTU"), ("matr", "MATR")]):
        d, feats = m7.load(ds); d = d.reset_index(drop=True); z = np.load(R / f"step20/{ds}/{ds}_temporal.npz"); y = d["SOH_rated"].values; late = z["late"]
        pl, pr = z["PT|LightGBM"].astype(float), z["PT|Ridge"].astype(float); lo = y[~late].min()
        e = pd.DataFrame({"c": d.cell, "e": (pl - y) ** 2, "late": late}); e = e[e.late].groupby("c").e.mean().pow(0.5); c = (e - e.median()).abs().idxmin()
        idx = np.where(d.cell.values == c)[0]; cyc = d.cycle.values[idx]; lt = late[idx]; ax = axs[0, j]
        ax.plot(cyc[~lt], y[idx][~lt], color=BLUE, lw=1.3, label="Training rows"); ax.plot(cyc[lt], y[idx][lt], color="k", lw=1.3, label="True SOH (test rows)")
        ax.plot(cyc[lt], pl[idx][lt], color=ORANGE, lw=1.0, label="LightGBM"); ax.plot(cyc[lt], pr[idx][lt], color=GREEN, lw=1.0, ls="--", label="Ridge regression")
        ax.axhline(lo, color=RED, lw=0.8, ls=":"); ax.set_title(f"{nm}, cell of median error", fontsize=7, loc="left"); ax.set_xlabel("Cycle"); ax.set_ylabel("SOH (%)")
        ax.text(cyc.max(), lo, "minimum SOH of training rows ", color=RED, fontsize=5.4, va="bottom", ha="right", bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.4))
        if j == 0: ax.legend(frameon=False, loc="lower left", fontsize=5.5)
        ax = axs[1, j]; rng = np.random.default_rng(0); li = np.where(late)[0]; sel = rng.choice(li, min(5000, len(li)), replace=False)
        ax.scatter(y[sel], pl[sel], s=2, color=ORANGE, alpha=.35, lw=0, rasterized=True, label="LightGBM"); ax.scatter(y[sel], pr[sel], s=2, color=GREEN, alpha=.35, lw=0, rasterized=True, label="Ridge regression")
        lims = [min(y[li].min(), 70), max(y[li].max(), 100)]; ax.plot(lims, lims, color="k", lw=0.6); ax.axvline(lo, color=RED, lw=0.8, ls=":"); ax.axvspan(lims[0], lo, color=RED, alpha=0.07)
        ax.set_xlabel("True SOH of test rows (%)"); ax.set_ylabel("Predicted SOH (%)"); ax.set_xlim(lims); ax.set_ylim(lims[0], lims[1] + 5 if ds != "hust" else 112)
        if j == 0: ax.legend(frameon=False, loc="upper left", markerscale=3, fontsize=5.5)
        sh = rs[(rs.dataset == ds)].share_out_of_range.iloc[0]; ax.text(0.97, 0.05, f"{100*sh:.0f} % of test rows lie\nbelow the training range", transform=ax.transAxes, ha="right", fontsize=5.6, color=RED)
        ax = axs[2, j]; mods = [("Ridge", "ridge"), ("LightGBM", "LightGBM"), ("GRU", "GRU"), ("TCN", "TCN"), ("WinMLP", "window\nMLP")]
        x = rs[rs.dataset == ds].set_index("model"); w = 0.38
        ax.bar(np.arange(5) - w / 2, [x.loc[m, "rmse_in"] for m, _ in mods], w, color="#8fb3d9", label="Within training range"); ax.bar(np.arange(5) + w / 2, [x.loc[m, "rmse_out"] for m, _ in mods], w, color=RED, label="Below training range")
        for i, (m, _) in enumerate(mods): ax.text(i + w / 2, (x.loc[m, "rmse_out"] if x.loc[m, "rmse_out"] == x.loc[m, "rmse_out"] else 0) + 0.2, f"{x.loc[m, 'rmse_out']:.1f}" if x.loc[m, "rmse_out"] == x.loc[m, "rmse_out"] else "", ha="center", fontsize=5.4)
        ax.set_xticks(range(5)); ax.set_xticklabels(["Ridge", "LightGBM", "GRU", "TCN", "Window MLP"], fontsize=5.4, rotation=25, ha="right", rotation_mode="anchor"); ax.set_ylabel("RMSE (% SOH)"); ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        if j == 0: ax.legend(frameon=False, fontsize=5.3, loc="upper right")
    for a, l in zip(axs.flat, "ABCDEFGHI"): plabel(a, l, -0.13, 1.02)
    save(fig, "fig_extrap")

# ====================================================================== F7 offset mechanism
if do("fig_offset"):
    import lightgbm as lgb
    from rankexp import DEFAULTS
    m7 = importlib.import_module("07_rank_experiment"); cache = Path("/tmp/offset_res.npz")
    if cache.exists(): Z = np.load(cache, allow_pickle=True); RES = {k: Z[k].item() for k in Z.files}
    else:
        RES = {}
        for ds, _ in DS:
            d, feats = m7.load(ds); d = d.reset_index(drop=True); X, y = d[feats].values.astype(float), d["SOH_rated"].values
            rk = d.groupby("cell")["cycle"].rank(method="first").values; n = d.groupby("cell")["cycle"].transform("size").values; late = rk > np.floor(0.6 * n)
            rng = np.random.default_rng(100); u = d["cell"].unique(); rng.shuffle(u); fold = d["cell"].map({c: i % 5 for i, c in enumerate(u)}).values; res = np.full(len(d), np.nan)
            for k in range(5):
                inf = fold == k; res[inf] = y[inf] - lgb.LGBMRegressor(num_leaves=31, random_state=42, verbose=-1, n_jobs=4, **DEFAULTS["LightGBM"]).fit(X[~inf], y[~inf]).predict(X[inf])
            life = d.groupby("cell").cycle_life.first() if ds == "matr" else d.groupby("cell").cycle.transform("size").groupby(d.cell).first()
            RES[ds] = {"cell": d.cell.values, "res": res, "late": late, "y": y, "rk": rk, "n": n, "life": life.reindex(d.cell.values).values}; print(ds, flush=True)
        np.savez(cache, **{k: np.array(v, dtype=object) for k, v in RES.items()})
    def per_cell(ds):
        r = RES[ds]; df = pd.DataFrame({"c": r["cell"], "r": r["res"], "late": r["late"], "y": r["y"], "f": r["rk"] / r["n"], "life": r["life"]}); return df
    cols3 = {"short": GREEN, "middle": "#999999", "long": RED}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6), gridspec_kw={"hspace": 0.5, "wspace": 0.34})
    dfm = per_cell("matr"); g = pd.DataFrame({"e": dfm[~dfm.late].groupby("c").r.mean(), "l": dfm[dfm.late].groupby("c").r.mean(), "life": dfm.groupby("c").life.first()}).dropna(); terc = pd.qcut(g.life, 3, labels=["short", "middle", "long"])
    cell_t = terc.reindex(dfm.c.values).values; dfm["t"] = cell_t
    ax = axs[0, 0]; bins = np.linspace(0, 1, 11)
    for t_ in ("short", "middle", "long"):
        x = dfm[dfm.t == t_]; b = pd.cut(x.f, bins, include_lowest=True); m = x.groupby(b, observed=True).r.agg(["mean", "sem"]); xc = 0.5 * (bins[1:] + bins[:-1])[: len(m)]
        ax.plot(xc, m["mean"], color=cols3[t_], marker="o", ms=2.5, lw=1.1, label={"short": "shortest-lived tercile", "middle": "middle tercile", "long": "longest-lived tercile"}[t_]); ax.fill_between(xc, m["mean"] - 1.96 * m["sem"], m["mean"] + 1.96 * m["sem"], color=cols3[t_], alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0.6, color="k", lw=0.5, ls=":"); ax.set_xlabel("Fraction of cell life"); ax.set_ylabel("Mean residual (points)"); ax.set_title("MATR, bias over cell life", fontsize=6.8, loc="left"); ax.legend(frameon=False, fontsize=5.4, loc="lower left")
    for ax, t_ in ((axs[0, 1], "short"), (axs[0, 2], "long")):
        gg = g[terc == t_]; ax.scatter(gg.e, gg.l, s=9, color=cols3[t_], alpha=.8, lw=0); ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
        same = float((np.sign(gg.e) == np.sign(gg.l)).mean()); sl = np.polyfit(gg.e, gg.l, 1)[0]; xs = np.array([-4, 7]); ax.plot(xs, np.polyval(np.polyfit(gg.e, gg.l, 1), xs), color=cols3[t_], lw=0.8, ls="--")
        ax.set_xlim(-4.5, 7.5); ax.set_ylim(-7.5, 5); ax.set_xlabel("Early residual (first 60 %)"); ax.set_ylabel("Late residual (last 40 %)" if t_ == "short" else "")
        ax.set_title("Shortest-lived tercile" if t_ == "short" else "Longest-lived tercile", fontsize=6.8, loc="left"); ax.text(0.04, 0.05, f"equal sign: {100*same:.0f} % of cells\nslope: {sl:.2f}", transform=ax.transAxes, fontsize=5.8)
    ax = axs[1, 0]; dcol = {"matr": BLUE, "hust": GREEN, "xjtu": ORANGE, "tju": PURPLE}
    for ds, nm in DS:
        df = per_cell(ds); b = pd.qcut(df.y, 12, duplicates="drop"); m = df.groupby(b, observed=True).agg(y=("y", "mean"), r=("r", "mean")); ax.plot(m.y, m.r, color=dcol[ds], marker="o", ms=2.3, lw=1, label=nm)
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("True SOH (%), binned"); ax.set_ylabel("Mean residual (points)"); ax.set_title("Residual versus true SOH", fontsize=6.8, loc="left"); ax.legend(frameon=False, fontsize=5.4, ncol=2, loc="upper left")
    for ax, ds, nm in ((axs[1, 1], "matr", "MATR"), (axs[1, 2], "xjtu", "XJTU")):
        df = per_cell(ds); gdf = pd.DataFrame({"e": df[~df.late].groupby("c").r.median(), "life": df.groupby("c").life.first()}); gdf["t"] = pd.qcut(gdf.life, 3, labels=["short", "middle", "long"]); df["t"] = gdf.t.reindex(df.c.values).values; df["off"] = gdf.e.reindex(df.c.values).values
        L_ = df[df.late]; base, adj = [], []
        for t_ in ("short", "middle", "long"):
            x = L_[L_.t == t_]; base.append(np.sqrt((x.r ** 2).mean())); adj.append(np.sqrt(((x.r - x.off) ** 2).mean()))
        xx = np.arange(3); ax.bar(xx - 0.19, base, 0.38, color="#8fb3d9", label="LightGBM"); ax.bar(xx + 0.19, adj, 0.38, color=ORANGE, label="with per-cell offset")
        for i in range(3): ax.text(xx[i] + 0.19, adj[i] + 0.05, f"{adj[i]:.2f}", ha="center", fontsize=5.4); ax.text(xx[i] - 0.19, base[i] + 0.05, f"{base[i]:.2f}", ha="center", fontsize=5.4)
        ax.set_xticks(xx); ax.set_xticklabels(["Shortest", "Middle", "Longest"]); ax.set_xlabel("Tercile of cells by cycle life"); ax.set_ylabel("Late-row RMSE (% SOH)" if ds == "matr" else ""); ax.set_title(f"{nm}, effect of the offset", fontsize=6.8, loc="left"); ax.set_ylim(0, max(base + adj) * 1.25)
        if ds == "matr": ax.legend(frameon=False, fontsize=5.4, loc="upper left")
    for a, l in zip(axs.flat, "ABCDEF"): plabel(a, l, -0.14, 1.02)
    save(fig, "fig_offset")

# ====================================================================== F8 early data and schedules
if do("fig_schedules"):
    fr = [0.0, 0.1, 0.2, 0.4, 0.6]; ft = read_tab("tables/finetune.tex"); ftd = {r[0]: [float(x) for x in r[1:]] for r in ft}
    fig = plt.figure(figsize=(7.2, 2.9)); gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.35], wspace=0.28, left=0.07, right=0.985, top=0.9, bottom=0.17)
    ax = fig.add_subplot(gs[0]); dcol = {"matr": BLUE, "hust": GREEN, "xjtu": ORANGE, "tju": PURPLE}
    for ds, nm in DS:
        v = [float(NUM[f"cal_{ds}_LightGBM_{str(f).replace('.', 'p')}"]) for f in fr]; ax.plot(np.array(fr) * 100, 100 * (np.array(v) / v[0] - 1), color=dcol[ds], marker="o", ms=3, lw=1.1, label=f"{nm} ({v[0]:.2f})")
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("Share of the target cell's rows added to training (%)"); ax.set_ylabel("Change in LightGBM RMSE (%)"); ax.set_title("Early rows added by concatenation", fontsize=6.8, loc="left"); ax.legend(frameon=False, fontsize=5.4, title="RMSE without target rows", title_fontsize=5.4, loc="lower left")
    ax2 = fig.add_subplot(gs[1]); labs = ["GRU\npooled\ntraining", "GRU\nhead only", "GRU\nlow learning\nrate", "GRU\nreplay", "GRU\noffset", "LightGBM\noffset", "Ridge\noffset"]
    D = np.zeros((4, 7)); B = np.zeros((4, 7))
    for i, (ds, nm) in enumerate(DS):
        v = ftd[nm]; base = v[0]; D[i, :5] = np.array(v[1:6]) - base; D[i, 5] = v[7] - v[6]; D[i, 6] = v[9] - v[8]; B[i, :5] = base; B[i, 5] = v[6]; B[i, 6] = v[8]
    from matplotlib.colors import TwoSlopeNorm
    im = ax2.imshow(D, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=2.4), aspect="auto")
    for i in range(4):
        for j in range(7): ax2.text(j, i, f"{D[i, j]:+.2f}", ha="center", va="center", fontsize=6, color="white" if abs(D[i, j]) > 1.4 else "black")
    ax2.set_xticks(range(7)); ax2.set_xticklabels(labs, fontsize=5.6); ax2.set_yticks(range(4)); ax2.set_yticklabels([f"{nm} (base {B[i,0]:.2f})" for i, (_, nm) in enumerate(DS)], fontsize=5.8); ax2.tick_params(length=0)
    ax2.axvline(4.5, color="white", lw=1.5); ax2.set_title("Change in late-row RMSE (points) relative to the model without target rows", fontsize=6.8, loc="left")
    cb = fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.01); cb.ax.tick_params(labelsize=5.5); cb.set_label("worse $\\leftarrow$ 0 $\\rightarrow$ better" if False else "RMSE change (points)", fontsize=5.8)
    plabel(ax, "A", -0.14, 1.02); plabel(ax2, "B", -0.3, 1.02)
    save(fig, "fig_schedules")

# ====================================================================== F9 ranking stability
if do("fig_ranking"):
    hc = read_tab("tables/holm_counts.tex"); labs = [f"{r[0]}\n{r[1]}" for r in hc]; fl = [int(r[2]) for r in hc]; p05 = [int(r[3]) for r in hc]; hm = [int(r[4]) for r in hc]; bh = [int(r[5]) for r in hc]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.7, 0.9, 1.0], "wspace": 0.62})
    ax = axs[0]; x = np.arange(len(hc)); w = 0.2
    for k, (v, c, l) in enumerate(((fl, "#bbbbbb", "Sign changes"), (p05, "#8fb3d9", "Both $p<0.05$"), (bh, BLUE, "BH"), (hm, RED, "Holm"))):
        ax.bar(x + (k - 1.5) * w, v, w, color=c, label=l)
    ax.set_xticks(x); ax.set_xticklabels([l.replace("\n", ", ").replace("MATR (C-rate)", "MATR C-rate").replace("MATR (batch)", "MATR batch") for l in labs], fontsize=5.2, rotation=35, ha="right"); ax.set_ylim(0, 18); ax.set_ylabel("Model pairs"); ax.set_title("Tabular models", fontsize=6.8, loc="left"); ax.legend(frameon=False, fontsize=5.4, ncol=2, loc="upper right")
    ax = axs[1]; dsn = ["matr", "hust", "xjtu", "tju"]; ra = [NUM[f"g33_rev_all_{d}"] for d in dsn]; rc = [NUM[f"g33_rev_core_{d}"] for d in dsn]; x = np.arange(4)
    ax.bar(x - 0.19, ra, 0.38, color="#8fb3d9", label="All 14 models"); ax.bar(x + 0.19, rc, 0.38, color=RED, label="Four main models")
    for i in range(4): ax.text(x[i] - 0.19, ra[i] + 0.6, str(ra[i]), ha="center", fontsize=5.6); ax.text(x[i] + 0.19, rc[i] + 0.6, str(rc[i]), ha="center", fontsize=5.6)
    ax.set_xticks(x); ax.set_xticklabels(["MATR", "HUST", "XJTU", "TJU"]); ax.set_ylabel("Supported reversals"); ax.set_title("Networks included", fontsize=6.8, loc="left"); ax.legend(frameon=False, fontsize=5.0, loc="upper right"); ax.set_ylim(0, max(ra) * 1.35)
    ax = axs[2]; tot = int(NUM["g33_tests"]); sig = int(NUM["g33_holm"]); rv = int(NUM["g33_rev_all"]); co = int(NUM["g33_rev_core"])
    vals = [tot, sig, rv, co]; labels2 = ["Pairwise tests", "Significant after Holm", "Reversals between\ntwo questions", "Reversals among\nfour main models"]
    ax.barh(range(4)[::-1], vals, color=["#bbbbbb", "#8fb3d9", BLUE, RED], height=0.6); ax.set_xscale("log"); ax.set_yticks(range(4)[::-1]); ax.set_yticklabels(labels2, fontsize=5.4)
    for i, v in zip(range(4)[::-1], vals): ax.text(v * 1.15, i, str(v), va="center", fontsize=6)
    ax.set_xlim(5, 6000); ax.set_title("Family of 14 models", fontsize=6.8, loc="left")
    for a, l in zip(axs, "ABC"): plabel(a, l, -0.14 if l != "C" else -0.62, 1.02)
    save(fig, "fig_ranking")

# ====================================================================== F10 factorial separation of coverage and target-cell information (step 53/56)
if do("fig_factorial"):
    fr = pd.read_csv(R / "step56/factorial_rmse.csv")
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.15), gridspec_kw={"wspace": 0.42})
    for ax, (ds, nm) in zip(axs, DS):
        for m, ls, mk in (("LightGBM", "-", "o"), ("Ridge", "--", "s")):
            for f_, col, lab in ((0.0, BLUE, "target cell absent"), (0.6, ORANGE, "first 60 % of target cell added")):
                x = fr[(fr.dataset == ds) & (fr.model == m) & (fr.f == f_)].sort_values("c")
                ax.plot(x.c * 100, x.pooled, ls, color=col, marker=mk, ms=3, lw=1.0, label=f"{m}, {lab}")
        ax.set_title(nm, fontsize=7); ax.set_xticks([60, 80, 100]); ax.set_xlabel("Other cells: life used (%)")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.2, lw=0.4)
        if ds == "matr": ax.set_ylabel("RMSE on last 40 % of cell (% SOH)")
        g = fr[(fr.dataset == ds) & (fr.model == "LightGBM")].set_index(["c", "f"]).pooled
        ax.annotate("A, C", (100, g[(1.0, 0.0)]), xytext=(-2, 6), textcoords="offset points", fontsize=6, ha="right", fontweight="bold")
        ax.annotate("B, D", (60, g[(0.6, 0.0)]), xytext=(4, 6), textcoords="offset points", fontsize=6, ha="left", fontweight="bold")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.24), fontsize=6)
    for a, s in zip(axs, "abcd"): plabel(a, s, -0.02, 1.0)
    save(fig, "fig_factorial")

# ====================================================================== F11 shift magnitude versus error penalty of held-out groups (step 60)
if do("fig_typology"):
    gt = pd.read_csv(R / "step60/group_typology.csv"); col = {"matr": BLUE, "hust": GREEN, "xjtu": ORANGE, "tju": PURPLE}; mk = {"matr": "o", "hust": "s", "xjtu": "^", "tju": "D"}
    short = {"batch (3)": "batch", "C-rate bin (5)": "rate bin", "charging policy (70)": "policy", "input-space cluster (5)": "cluster", "identifier group (10)": "identifier", "batch (6)": "batch",
             "type x condition (11)": "type$\\times$cond.", "cell type (3)": "type", "temperature (3)": "temp.", "charging rate (3)": "rate"}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.28})
    for ax, m in zip(axs, ("LightGBM", "Ridge")):
        x = gt[gt.model == m]
        for ds, nm in DS:
            y_ = x[x.dataset == ds]; ax.scatter(y_.dist_ratio, y_.ratio_P3_P2, color=col[ds], marker=mk[ds], s=18, label=nm, zorder=3, lw=0.3, edgecolor="white")
            for _, r in y_.iterrows():
                lab, off = short[r.grouping], (3, 2)
                if r.grouping == "charging policy (70)": continue
                if r.grouping == "C-rate bin (5)": lab, off = "rate bin, policy", (4, -8)
                if ds == "matr" and r.grouping.startswith("input"): off = (4, -8)
                if ds == "hust" and r.grouping.startswith("ident"): off = (3, 6)
                ax.annotate(lab, (r.dist_ratio, r.ratio_P3_P2), xytext=off, textcoords="offset points", fontsize=5)
        ax.axhline(1, color="#888", lw=0.5, ls=":"); ax.axvline(1, color="#888", lw=0.5, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Distance of held-out rows relative to P2 (ratio)"); ax.set_ylabel("RMSE of held-out group / RMSE of P2")
        ax.set_title(f"{m} (Spearman {NUM['gt_rho_lgb'] if m == 'LightGBM' else NUM['gt_rho_ridge']})", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.2, lw=0.4, which="both")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g")); ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axs[0].legend(frameon=False, loc="upper left", fontsize=6, ncol=2)
    plabel(axs[0], "a", -0.16, 1.03); plabel(axs[1], "b", -0.16, 1.03)
    save(fig, "fig_typology")
