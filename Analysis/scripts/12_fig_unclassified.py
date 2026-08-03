"""Mine the play the classifier declines to name (confidence < 0.90).

F30 -- anatomy of the abstention set: how big, who produces it, and what shape
       the network's uncertainty takes (a two-way tie, not a shrug)
F31 -- reactive-strategy geometry: the abstention set lives in the interior of
       the square, farther from every canonical corner than any other bucket
F32 -- a wider vocabulary: how much of it a library of non-canonical rules
       names exactly, against a permutation null
F33 -- temporal and behavioural signatures, including whether unnameable play
       is worse play (it is not)
F34 -- strategy or noise: replicate stability, split-half coherence, and
       whether any discrete cluster survives resampling
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from pdlib.unclassified import (BUCKET_LABEL, BUCKETS, CANONICAL4, CORNERS,
                                STRATEGIES4, THRESHOLD, add_buckets,
                                corner_distance, library_table,
                                posterior_geometry, reactive_coordinates,
                                sequence_stats, split_half_reactive)
from pdlib.style import (CMAP_SEQ, C_COOP, C_DEFECT, DATADIR, INK, INK2, MODEL,
                         MODEL_ORDER, MUTED, STRATEGY, SURFACE, TABDIR,
                         footnote, panel_tag, savefig, use_paper_style)

use_paper_style()

# The abstention set is the subject, so it gets the one warm hue; the three
# buckets it is contrasted against are cool and deliberately recede.
BUCKET_COLOR = {"exact": "#2a78d6", "ambiguous": "#9ec5f4",
                "confident": "#1baf7a", "unclassified": "#eb6834"}
FOCUS = BUCKET_COLOR["unclassified"]
FAM_LEN = {"frontier": 10, "small": 30}
FAM_LABEL = {"frontier": "frontier (10 rd)", "small": "open-weight (30 rd)"}

# A conditional rate estimated from two observations can only be 0, 0.5 or 1,
# which stamps a spurious band across the middle of the reactive square.  Four
# is the smallest count that puts more grid points between the corners than on
# the midline; F31 states the requirement on the figure itself.
MIN_OBS = 4


def _short(m):
    return m.split("-")[0]


def _arrow(s):
    return s.replace("->", "→")


def _bucket_legend(fig, buckets=BUCKETS, y=0.935, ncol=4):
    fig.legend(handles=[Patch(facecolor=BUCKET_COLOR[b], edgecolor=SURFACE,
                              label=BUCKET_LABEL[b]) for b in buckets],
               loc="upper left", bbox_to_anchor=(0.015, y), ncol=ncol,
               fontsize=7, frameon=False, handlelength=1.1)


# ==========================================================================
# F30  anatomy of the abstention set
# ==========================================================================
def fig_anatomy(arche, resid_conf):
    fig = plt.figure(figsize=(11.6, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.58, wspace=0.44)

    # (a) the confidence distribution is bimodal; 0.90 falls in the valley ---
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0.25, 1.0, 46)
    for fam, col in (("frontier", C_COOP), ("small", C_DEFECT)):
        v = resid_conf.loc[resid_conf.family == fam, "confidence"]
        ax.hist(v, bins=bins, histtype="step", lw=1.9, color=col,
                label=FAM_LABEL[fam])
    ax.axvline(THRESHOLD, color=INK, lw=1.2, ls="--")
    ax.text(THRESHOLD - 0.015, ax.get_ylim()[1] * 0.45, "0.90", ha="right",
            fontsize=7, color=INK, fontweight="semibold")
    ax.set_xlabel("LSTM top posterior")
    ax.set_ylabel("trajectories")
    ax.set_yscale("log")
    ax.legend(fontsize=6.2, loc="upper left")
    ax.set_title("Confidence is bimodal")
    panel_tag(ax, "a", dx=-0.26)

    # (b) risk-coverage, from the synthetic test sets where truth is known ---
    ax = fig.add_subplot(gs[0, 1])
    rc_rows = []
    for tag, fam, col in (("h10", "frontier", C_COOP), ("h30", "small", C_DEFECT)):
        d = np.load(DATADIR / f"clf_test_{tag}.npz")
        pr, y = d["proba"], d["y"]
        conf, hit = pr.max(1), pr.argmax(1) == y
        ths = np.linspace(0.25, 0.999, 120)
        cov = np.array([(conf >= t).mean() for t in ths])
        acc = np.array([hit[conf >= t].mean() if (conf >= t).any() else np.nan
                        for t in ths])
        ax.plot(cov, acc, "-", color=col, lw=2.0, label=FAM_LABEL[fam])
        j = int(np.argmin(np.abs(ths - THRESHOLD)))
        ax.plot(cov[j], acc[j], "o", ms=7, color=col, mec=SURFACE, mew=1.4,
                zorder=5)
        rc_rows.append({"corpus": tag, "threshold": THRESHOLD,
                        "coverage": cov[j], "accuracy_kept": acc[j],
                        "accuracy_all": hit.mean()})
    pd.DataFrame(rc_rows).to_csv(TABDIR / "T25_risk_coverage.csv", index=False)
    ax.set_xlabel("coverage (share kept)")
    ax.set_ylabel("accuracy on what is kept")
    ax.legend(fontsize=6.2, loc="lower left")
    ax.set_title("What the floor buys")
    ax.text(0.97, 0.55, "dot = 0.90", transform=ax.transAxes, ha="right",
            fontsize=6.4, color=INK2)
    panel_tag(ax, "b", dx=-0.30)

    # (c) the full read-out, per model -------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    comp = (arche.groupby("model").bucket.value_counts(normalize=True)
            .unstack().reindex(index=MODEL_ORDER, columns=BUCKETS).fillna(0))
    comp.to_csv(TABDIR / "T26_bucket_composition.csv")
    bottom = np.zeros(len(comp))
    x = np.arange(len(comp))
    for b in BUCKETS:
        v = comp[b].to_numpy()
        ax.bar(x, v, bottom=bottom, color=BUCKET_COLOR[b], edgecolor=SURFACE,
               linewidth=1.1, width=0.66)
        bottom += v
    for xi, v in enumerate(comp["unclassified"]):
        ax.text(xi, 1.015, f"{v:.2f}", ha="center", fontsize=6.4, color=FOCUS,
                fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels([_short(m) for m in comp.index], rotation=24, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(False)
    ax.set_title("Abstention is a model trait")
    panel_tag(ax, "c", dx=-0.30)

    # (d) the tie is two-way, not four-way ----------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for b in ("confident", "unclassified"):
        v = arche.loc[arche.bucket == b, "top2_mass"]
        ax.hist(v, bins=np.linspace(0.5, 1.0, 40), histtype="stepfilled",
                alpha=0.55 if b == "unclassified" else 0.35,
                color=BUCKET_COLOR[b], density=True)
    u = arche[arche.bucket == "unclassified"]
    ax.text(0.52, ax.get_ylim()[1] * 0.86,
            f"{(u.top2_mass > 0.9).mean():.0%} of the abstention set\n"
            f"puts >0.90 on just two rules;\n"
            f"only {(u.entropy > 0.9).mean():.1%} is near-uniform",
            fontsize=6.6, color=INK2, va="top")
    ax.set_xlabel("posterior mass on the top two labels")
    ax.set_ylabel("density")
    ax.set_title("Caught between two rules")
    panel_tag(ax, "d", dx=-0.26)

    # (e) which two -----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    pr = u.pair.value_counts(normalize=True).sort_values()
    y = np.arange(len(pr))
    cols = [STRATEGY["TFT"] if "TFT" in p else MUTED for p in pr.index]
    ax.barh(y, pr.to_numpy(), color=cols, edgecolor=SURFACE, linewidth=1.0,
            height=0.66)
    for yi, v in zip(y, pr.to_numpy()):
        ax.text(v + 0.006, yi, f"{v:.2f}", va="center", fontsize=6.8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([p.replace("+", " vs ") for p in pr.index], fontsize=7)
    ax.set_xlim(0, float(pr.max()) * 1.26)
    ax.set_xlabel("share of the abstention set")
    tft_share = u.pair.str.contains("TFT").mean()
    ax.set_title(f"TFT is in {tft_share:.0%} of the ties")
    panel_tag(ax, "e", dx=-0.44)

    # (f) positive control: a real fifth strategy the network never saw ------
    ax = fig.add_subplot(gs[1, 2])
    rows = []
    names = np.array(STRATEGIES4)
    for tag in ("h10", "h30"):
        pr_ = np.load(DATADIR / f"clf_unseen_{tag}.npz")["proba"]
        o = np.argsort(-pr_, axis=1)
        pair = pd.Series(["+".join(sorted(names[[a, b]], key=STRATEGIES4.index))
                          for a, b in o[:, :2]])
        for k, v in pair.value_counts(normalize=True).items():
            rows.append({"source": "GTFT (held out)", "pair": k, "share": v})
    for k, v in u.pair.value_counts(normalize=True).items():
        rows.append({"source": "observed abstention set", "pair": k, "share": v})
    ctrl = (pd.DataFrame(rows).groupby(["source", "pair"]).share.mean()
            .unstack(fill_value=0))
    ctrl.to_csv(TABDIR / "T27_gtft_control.csv")
    order = [p for p in ["AllC+TFT", "TFT+AllD", "AllC+WSLS", "WSLS+AllD",
                         "TFT+WSLS", "AllC+AllD"] if p in ctrl.columns]
    xx = np.arange(len(order))
    for k, (src, col) in enumerate((("GTFT (held out)", "#4a3aa7"),
                                    ("observed abstention set", FOCUS))):
        ax.bar(xx + (k - 0.5) * 0.38, ctrl.loc[src, order], width=0.36,
               color=col, edgecolor=SURFACE, linewidth=1.0, label=src)
    ax.set_xticks(xx)
    ax.set_xticklabels([o.replace("+", "\nvs ") for o in order], fontsize=6.0)
    ax.set_ylabel("share")
    ax.legend(fontsize=6.0, loc="upper right")
    ax.set_title("A known fifth rule looks different")
    panel_tag(ax, "f", dx=-0.26)

    _bucket_legend(fig, y=0.945)
    fig.suptitle("The play the classifier declines to name", x=0.015, y=1.0,
                 ha="left", fontweight="bold", color=INK)
    savefig(fig, "F30_abstention_anatomy")


# ==========================================================================
# F31  reactive-strategy geometry
# ==========================================================================
def fig_geometry(arche, react):
    fig = plt.figure(figsize=(11.6, 7.0))
    gs = fig.add_gridspec(2, 3, hspace=0.56, wspace=0.42)

    m = arche.merge(react, on=["game_uid", "agent"], how="left")
    ok = m[(m.n_after_C >= MIN_OBS) & (m.n_after_D >= MIN_OBS)].copy()

    def _corners(ax, labels=True):
        for name, (px, qx) in CORNERS.items():
            ax.plot(px, qx, "*", ms=15, color=STRATEGY[name], mec=SURFACE,
                    mew=1.4, zorder=8, clip_on=False)
            if labels:
                ax.annotate(name, (px, qx), textcoords="offset points",
                            xytext=(-4 if px > 0.5 else 8, 10 if qx < 0.5 else -15),
                            ha="right" if px > 0.5 else "left", fontsize=7,
                            fontweight="bold", color=STRATEGY[name], zorder=8)
        ax.plot([0, 1], [0, 0], "-", color=MUTED, lw=1.0, alpha=0.6, zorder=2)
        ax.plot([1, 1], [0, 1], "-", color=MUTED, lw=1.0, alpha=0.6, zorder=2)
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.06, 1.06)
        ax.set_xlabel("p = P(C | opponent cooperated)")
        ax.set_ylabel("q = P(C | opponent defected)")
        ax.grid(True, axis="both")

    # (a,b) density of each bucket in the square -----------------------------
    for j, b in enumerate(("confident", "unclassified")):
        ax = fig.add_subplot(gs[0, j])
        s = ok[ok.bucket == b]
        hb = ax.hexbin(s.p_CgivenC, s.q_CgivenD, gridsize=19, cmap=CMAP_SEQ,
                       mincnt=1, linewidths=0, extent=(0, 1, 0, 1))
        _corners(ax)
        ax.set_title(f"{BUCKET_LABEL[b]}   (n = {len(s):,})")
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6)
        panel_tag(ax, "ab"[j], dx=-0.24)

    # (c) distance to the nearest corner --------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    dist = corner_distance(ok)
    dist.groupby("bucket").d_nearest_corner.describe().to_csv(
        TABDIR / "T28_corner_distance.csv")
    for b in BUCKETS:
        v = np.sort(dist.loc[dist.bucket == b, "d_nearest_corner"].dropna())
        if len(v) < 20:
            continue
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), "-",
                color=BUCKET_COLOR[b], lw=2.0)
    ax.set_xlabel("distance to nearest corner")
    ax.set_ylabel("cumulative share")
    ax.set_title("Farthest from the vocabulary")
    panel_tag(ax, "c", dx=-0.28)

    # (d) each tie sits in its own region -------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    u = ok[ok.bucket == "unclassified"]
    cent = u.groupby("pair").agg(n=("p_CgivenC", "size"),
                                 p=("p_CgivenC", "mean"),
                                 q=("q_CgivenD", "mean"),
                                 p_se=("p_CgivenC", "sem"),
                                 q_se=("q_CgivenD", "sem"))
    cent = cent[cent.n >= 15]
    cent.to_csv(TABDIR / "T29_pair_centroids.csv")
    _corners(ax)
    offsets = {"AllC+TFT": (11, -3), "TFT+AllD": (11, -10), "AllC+WSLS": (11, 5),
               "WSLS+AllD": (-11, 5), "TFT+WSLS": (11, -13), "AllC+AllD": (-11, -12)}
    for name, row in cent.iterrows():
        col = STRATEGY["TFT"] if "TFT" in name else "#4a3aa7"
        ax.errorbar(row.p, row.q, xerr=row.p_se, yerr=row.q_se, fmt="o",
                    ms=5 + 7 * row.n / cent.n.max(), color=col, mec=SURFACE,
                    mew=1.2, ecolor=col, elinewidth=1.1, capsize=2, zorder=7)
        dx, dy = offsets.get(name, (8, 8))
        ax.annotate(name.replace("+", "/"), (row.p, row.q),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="left" if dx > 0 else "right", fontsize=6.4, color=col,
                    fontweight="semibold", zorder=7)
    ax.set_title("Each tie has its own region")
    panel_tag(ax, "d", dx=-0.24)

    # (e) anti-reciprocity: cooperating more after being defected on ----------
    ax = fig.add_subplot(gs[1, 1])
    ok["anti"] = ok.q_CgivenD > ok.p_CgivenC
    grp = ok.groupby(["model", "bucket"]).anti.agg(["mean", "size"])
    # A share computed on a dozen trajectories is not a share; cells below the
    # floor are left blank rather than drawn as a confident-looking bar.
    MIN_CELL = 25
    x = np.arange(len(MODEL_ORDER))
    for k, b in enumerate(("confident", "unclassified")):
        vals, ns = [], []
        for mdl in MODEL_ORDER:
            if (mdl, b) in grp.index and grp.loc[(mdl, b), "size"] >= MIN_CELL:
                vals.append(grp.loc[(mdl, b), "mean"])
                ns.append(int(grp.loc[(mdl, b), "size"]))
            else:
                vals.append(np.nan)
                ns.append(0)
        ax.bar(x + (k - 0.5) * 0.38, vals, width=0.36, color=BUCKET_COLOR[b],
               edgecolor=SURFACE, linewidth=1.0)
        for xi, (v, n) in enumerate(zip(vals, ns)):
            if np.isfinite(v):
                ax.text(xi + (k - 0.5) * 0.38, v + 0.015, str(n), ha="center",
                        fontsize=5.4, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(i) for i in MODEL_ORDER], rotation=24, ha="right")
    ax.set_ylabel("share with q > p")
    ax.set_ylim(0, 1.05)
    ax.set_title("Anti-reciprocity: kinder after D")
    ax.text(0.99, 0.96, f"blank: n < {MIN_CELL}", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=MUTED)
    panel_tag(ax, "e", dx=-0.26)

    # (f) how much reciprocity is left ----------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    for b, ls in (("confident", "--"), ("unclassified", "-")):
        s = ok[ok.bucket == b]
        ax.hist(s.p_CgivenC - s.q_CgivenD, bins=np.linspace(-1, 1, 38),
                histtype="step", lw=2.0, ls=ls, color=BUCKET_COLOR[b],
                density=True)
    ax.axvline(0, color=INK, lw=1.0)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.18)
    ax.annotate("anti-reciprocal", (-0.62, top * 1.08), fontsize=6.2,
                color=MUTED, ha="center")
    ax.annotate("reciprocal", (0.62, top * 1.08), fontsize=6.2, color=MUTED,
                ha="center")
    ax.set_xlabel("reciprocity  p − q")
    ax.set_ylabel("density")
    ax.set_title("Reciprocity weaker, not absent")
    panel_tag(ax, "f", dx=-0.26)

    _bucket_legend(fig, y=0.945)
    fig.suptitle("Where unnameable play sits in the reactive-strategy square",
                 x=0.015, y=1.0, ha="left", fontweight="bold", color=INK)
    footnote(fig, f"Rates estimated per trajectory; only games with at least "
                  f"{MIN_OBS} rounds after each opponent action are shown "
                  f"(n = {len(ok):,} of {len(m):,}).", y=0.0)
    savefig(fig, "F31_reactive_geometry")
    return ok


# ==========================================================================
# F32  a wider vocabulary
# ==========================================================================
def fig_vocabulary(arche, lib):
    fig = plt.figure(figsize=(11.6, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.46)

    m = arche.merge(lib, on=["game_uid", "agent"], how="left")
    for c in ("dev_canonical", "dev_extended", "dev_extended_null"):
        m[c + "_rate"] = m[c] / m.n_rounds

    # (a) how far the nearest rule is, per round ------------------------------
    ax = fig.add_subplot(gs[0, 0])
    agg = m.groupby("bucket")[["dev_canonical_rate", "dev_extended_rate",
                               "dev_extended_null_rate"]].mean().reindex(BUCKETS)
    agg.to_csv(TABDIR / "T30_deviation_rates.csv")
    keep = ["confident", "unclassified"]
    sub = agg.loc[keep]
    x = np.arange(len(sub))
    for k, (c, col, lab) in enumerate((
            ("dev_canonical_rate", "#2a78d6", "canonical four"),
            ("dev_extended_rate", FOCUS, "wide library"),
            ("dev_extended_null_rate", MUTED, "wide library, shuffled"))):
        ax.bar(x + (k - 1) * 0.27, sub[c], width=0.25, color=col,
               edgecolor=SURFACE, linewidth=1.0, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABEL[b] for b in keep], fontsize=7)
    ax.set_ylabel("deviations per round")
    ax.legend(fontsize=6.2, loc="upper left")
    ax.set_title("A wider vocabulary closes the gap")
    ax.text(0.5, -0.30, "the two rule-matched buckets are 0 by construction",
            transform=ax.transAxes, ha="center", fontsize=6.0, color=MUTED)
    panel_tag(ax, "a", dx=-0.28)

    # (b) exact fits, against the permutation null ----------------------------
    ax = fig.add_subplot(gs[0, 1])
    m["exact_ext"] = m.dev_extended == 0
    m["exact_null"] = m.dev_extended_null == 0
    ex = (m[m.bucket.isin(keep)]
          .groupby(["family", "bucket"])[["exact_ext", "exact_null"]].mean())
    lab, obs, nul, cols = [], [], [], []
    for fam in ("frontier", "small"):
        for b in keep:
            lab.append(f"{fam}\n{BUCKET_LABEL[b]}")
            obs.append(ex.loc[(fam, b), "exact_ext"])
            nul.append(ex.loc[(fam, b), "exact_null"])
            cols.append(BUCKET_COLOR[b])
    x = np.arange(len(lab))
    ax.bar(x - 0.19, obs, width=0.36, color=cols, edgecolor=SURFACE,
           linewidth=1.0)
    ax.bar(x + 0.19, nul, width=0.36, color=MUTED, alpha=0.5, hatch="///",
           edgecolor=SURFACE, linewidth=1.0, label="shuffled null")
    for xi, o in enumerate(obs):
        ax.text(xi - 0.19, o + 0.008, f"{o:.2f}", ha="center", fontsize=6.4,
                color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=5.8)
    ax.set_ylabel("share fitted exactly")
    ax.legend(fontsize=6.2, loc="upper right")
    ax.set_title("Named more often\nthan confident play", fontsize=8.6)
    panel_tag(ax, "b", dx=-0.26)

    # (c) which rule is nearest -----------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    u = m[m.bucket == "unclassified"]
    top = u.best_family.value_counts(normalize=True).head(10).sort_values()
    y = np.arange(len(top))
    cols = [FOCUS if f not in CANONICAL4 else "#9ec5f4" for f in top.index]
    ax.barh(y, top.to_numpy(), color=cols, edgecolor=SURFACE, linewidth=1.0,
            height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels([_arrow(f) for f in top.index], fontsize=6.6,
                       family="monospace")
    ax.set_xlabel("share of the abstention set")
    ax.set_title("Nearest rule, wide library")
    ax.text(0.98, 0.03, "orange = non-canonical", transform=ax.transAxes,
            ha="right", fontsize=6.0, color=MUTED)
    panel_tag(ax, "c", dx=-0.42)

    # (d) among the exact fits, what actually names them ----------------------
    ax = fig.add_subplot(gs[1, 0])
    exact = u[u.dev_extended == 0]
    tab = (exact.groupby("family").best_family.value_counts(normalize=True)
           .unstack(fill_value=0))
    keep_r = tab.sum(0).sort_values(ascending=False).head(7).index
    xx = np.arange(len(keep_r))
    for k, (fam, col) in enumerate((("frontier", C_COOP), ("small", C_DEFECT))):
        if fam in tab.index:
            ax.bar(xx + (k - 0.5) * 0.38, tab.loc[fam, keep_r], width=0.36,
                   color=col, edgecolor=SURFACE, linewidth=1.0,
                   label=FAM_LABEL[fam])
    ax.set_xticks(xx)
    ax.set_xticklabels([_arrow(k) for k in keep_r], rotation=30, ha="right",
                       fontsize=6.0, family="monospace")
    ax.set_ylabel("share of exact fits")
    ax.legend(fontsize=6.2)
    ax.set_title(f"What names them  (n = {len(exact):,})")
    panel_tag(ax, "d", dx=-0.22)

    # (e) when the clock rules switch ----------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    clk = exact[exact.best_rule.str.contains("@")].copy()
    clk["k_frac"] = clk.switch_round / clk.n_rounds
    for fam, col in (("frontier", C_COOP), ("small", C_DEFECT)):
        v = clk.loc[clk.family == fam, "k_frac"]
        if len(v):
            ax.hist(v, bins=np.linspace(0, 1, 21), histtype="step", lw=2.0,
                    color=col, density=True, label=FAM_LABEL[fam])
    top_y = ax.get_ylim()[1]
    ax.set_ylim(0, top_y * 1.30)
    ax.axvspan(0.75, 1.0, color=MUTED, alpha=0.13, lw=0)
    ax.text(0.875, top_y * 0.55, "endgame", ha="center", va="center",
            fontsize=6.2, color=MUTED, rotation=90)
    ax.text(0.26, top_y * 1.24,
            f"only {(clk.k_frac > 0.75).mean():.0%} of switches are late:\n"
            "no backward induction", fontsize=6.4, color=INK2, va="top")
    ax.set_xlabel("switch point, fraction of the game")
    ax.set_ylabel("density")
    ax.legend(fontsize=6.2, loc="center right")
    ax.set_title("Switches are openings, not endgames")
    panel_tag(ax, "e", dx=-0.26)

    # (f) who plays outside the vocabulary ------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    sh = (u.assign(nc=lambda d: ~d.is_canonical).groupby("model").nc.mean()
          .reindex(MODEL_ORDER))
    y = np.arange(len(sh))[::-1]
    ax.barh(y, sh.to_numpy(), color=[MODEL[i] for i in sh.index], height=0.64,
            edgecolor=SURFACE, linewidth=1.0)
    for yi, v in zip(y, sh.to_numpy()):
        ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=6.8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(i) for i in sh.index])
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("share non-canonical")
    ax.set_title("Every model, most of the time")
    panel_tag(ax, "f", dx=-0.40)

    fig.suptitle("Does a wider vocabulary name the abstention set?", x=0.015,
                 y=1.0, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F32_wider_vocabulary")
    return m


# ==========================================================================
# F33  temporal and behavioural signatures
# ==========================================================================
def fig_signatures(arche, rounds, seqs):
    fig = plt.figure(figsize=(11.6, 7.0))
    gs = fig.add_gridspec(2, 3, hspace=0.56, wspace=0.42)

    r = rounds.merge(arche[["game_uid", "agent", "bucket"]],
                     on=["game_uid", "agent"])

    # (a) cooperation over normalised game time -------------------------------
    ax = fig.add_subplot(gs[0, 0])
    r["bin"] = pd.cut(r.round_frac, np.linspace(0, 1, 11))
    curve = r.groupby(["bucket", "bin"], observed=True).coop.mean().unstack()
    for b in BUCKETS:
        if b in curve.index:
            ax.plot(np.linspace(0.05, 0.95, curve.shape[1]), curve.loc[b],
                    "-o", ms=3.4, color=BUCKET_COLOR[b], mec=SURFACE, mew=0.8)
    ax.set_xlabel("position in the game")
    ax.set_ylabel("P(cooperate)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Slow erosion, no cliff")
    panel_tag(ax, "a", dx=-0.26)

    # (b) the endgame rounds --------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    r["from_end"] = r["round"] - r.groupby(["game_uid", "agent"])["round"].transform("max")
    tail = r[r.from_end >= -3]
    end = tail.groupby(["bucket", "from_end"]).coop.mean().unstack()
    for b in ("confident", "unclassified"):
        if b in end.index:
            ax.plot(end.columns, end.loc[b], "-o", ms=5.5, color=BUCKET_COLOR[b],
                    mec=SURFACE, mew=1.1)
    ax.set_xticks([-3, -2, -1, 0])
    ax.set_xticklabels(["−3", "−2", "−1", "last"])
    ax.set_xlabel("round, counted from the end")
    ax.set_ylabel("P(cooperate)")
    ax.set_title("Endgame drop: confident only")
    panel_tag(ax, "b", dx=-0.28)

    # (c) alternation against the iid expectation -----------------------------
    ax = fig.add_subplot(gs[0, 2])
    s = seqs.merge(arche[["game_uid", "agent", "bucket", "model"]],
                   on=["game_uid", "agent"])
    for b in ("confident", "unclassified"):
        v = s.loc[s.bucket == b, "alternation_excess"]
        ax.hist(v, bins=np.linspace(-0.55, 0.55, 46), histtype="step", lw=2.0,
                color=BUCKET_COLOR[b], density=True)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_xlabel("switch rate − iid expectation")
    ax.set_ylabel("density")
    ax.set_title("Not clock-driven alternation")
    panel_tag(ax, "c", dx=-0.28)

    # (d) run structure -------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    pos, cols, ticks = [], [], []
    for i, fam in enumerate(("frontier", "small")):
        sub = s[s.n_rounds == FAM_LEN[fam]]
        for j, b in enumerate(("confident", "unclassified")):
            pos.append(i * 1.4 + j * 0.42)
            cols.append(b)
        ticks.append(i * 1.4 + 0.21)
    data = []
    for fam in ("frontier", "small"):
        sub = s[s.n_rounds == FAM_LEN[fam]]
        for b in ("confident", "unclassified"):
            data.append(sub.loc[sub.bucket == b, "max_run"] / FAM_LEN[fam])
    bp = ax.boxplot(data, positions=pos, widths=0.36, patch_artist=True,
                    showfliers=False, medianprops=dict(color=INK, lw=1.3),
                    whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED))
    for patch, b in zip(bp["boxes"], cols):
        patch.set_facecolor(BUCKET_COLOR[b])
        patch.set_edgecolor(SURFACE)
    ax.set_xticks(ticks)
    ax.set_xticklabels([FAM_LABEL[f] for f in ("frontier", "small")], fontsize=6.8)
    ax.set_xlim(-0.4, 2.2)
    ax.set_ylabel("longest single-action run\n(fraction of the game)")
    ax.set_title("Run length flips with horizon")
    panel_tag(ax, "d", dx=-0.28)

    # (e,f) does unnameable play pay? -----------------------------------------
    for j, (col_name, lab, title) in enumerate((
            ("efficiency", "efficiency", "Unnameable play is not worse"),
            ("payoff_per_round", "payoff per round", "…and earns more"))):
        ax = fig.add_subplot(gs[1, 1 + j])
        piv = (arche.groupby(["family", "bucket"])[col_name].mean()
               .unstack().reindex(columns=BUCKETS))
        x = np.arange(len(piv))
        w = 0.2
        for k, b in enumerate(BUCKETS):
            ax.bar(x + (k - 1.5) * w, piv[b], width=w * 0.92,
                   color=BUCKET_COLOR[b], edgecolor=SURFACE, linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([FAM_LABEL[i] for i in piv.index], fontsize=6.8)
        ax.set_ylabel(lab)
        ax.set_title(title)
        panel_tag(ax, "ef"[j], dx=-0.28)

    _bucket_legend(fig, y=0.945)
    fig.suptitle("Temporal and behavioural signatures of the abstention set",
                 x=0.015, y=1.0, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F33_behavioural_signatures")


# ==========================================================================
# F34  strategy or noise?
# ==========================================================================
def fig_coherence(arche, react, halves):
    fig = plt.figure(figsize=(11.6, 3.8))
    gs = fig.add_gridspec(1, 4, wspace=0.48)

    # (a) do the ten replicates of a cell agree? ------------------------------
    ax = fig.add_subplot(gs[0, 0])
    a = arche.copy()
    a["cell"] = (a.model + "|" + a.language + "|" + a.scale_nominal.astype(str)
                 + "|" + a.personality + "|" + a.opp_personality + "|"
                 + a.agent.astype(str))
    cell = a.groupby("cell").agg(
        n=("archetype", "size"),
        modal=("archetype", lambda s: s.value_counts(normalize=True).iloc[0]),
        unc=("bucket", lambda s: (s == "unclassified").mean()))
    cell["bin"] = pd.cut(cell.unc, [-0.01, 0.0001, 0.25, 0.5, 1.01],
                         labels=["none", "≤25%", "25–50%", ">50%"])
    g = cell.groupby("bin", observed=True).agg(n=("modal", "size"),
                                               modal=("modal", "mean"),
                                               se=("modal", "sem"))
    g.to_csv(TABDIR / "T31_replicate_stability.csv")
    x = np.arange(len(g))
    ax.bar(x, g.modal, yerr=g.se, width=0.62,
           color=CMAP_SEQ(np.linspace(0.30, 0.92, len(g))), edgecolor=SURFACE,
           linewidth=1.1, error_kw=dict(ecolor=INK2, lw=1.0))
    ax.axhline(0.2, color=INK, ls=":", lw=1.1)
    ax.text(len(g) - 0.45, 0.225, "chance", fontsize=6.2, color=MUTED, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, fontsize=6.8)
    ax.set_xlabel("share of the cell that abstains")
    ax.set_ylabel("modal archetype share\nacross 10 replicates")
    ax.set_ylim(0, 1)
    ax.set_title("Abstention tracks instability")
    panel_tag(ax, "a", dx=-0.36)

    # (b) split-half coherence ------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    h = halves.merge(arche[["game_uid", "agent", "bucket"]],
                     on=["game_uid", "agent"])
    rows = []
    rng = np.random.default_rng(0)
    for b in ("exact", "confident", "unclassified"):
        s = h[h.bucket == b]
        if len(s) < 30:
            continue
        for coord, a1, a2 in (("p", "mean_first_C", "mean_second_C"),
                              ("q", "mean_first_D", "mean_second_D")):
            v1, v2 = s[a1].to_numpy(), s[a2].to_numpy()
            rows.append({"bucket": b, "coord": coord,
                         "r": np.corrcoef(v1, v2)[0, 1],
                         "r_null": np.corrcoef(v1, rng.permutation(v2))[0, 1],
                         "n": len(s)})
    coh = pd.DataFrame(rows)
    coh.to_csv(TABDIR / "T32_split_half.csv", index=False)
    piv = coh.pivot(index="bucket", columns="coord", values="r").reindex(
        [b for b in BUCKETS if b in coh.bucket.values])
    x = np.arange(len(piv))
    for k, c in enumerate(("p", "q")):
        ax.bar(x + (k - 0.5) * 0.38, piv[c], width=0.36,
               color=[C_COOP, C_DEFECT][k], edgecolor=SURFACE, linewidth=1.0,
               label=f"{c} = P(C | opp {'C' if c == 'p' else 'D'})")
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABEL[i] for i in piv.index], fontsize=6.2,
                       rotation=12, ha="right")
    ax.set_ylabel("first half vs second half\ncorrelation")
    ax.legend(fontsize=5.8, loc="upper right")
    ax.set_title("Within-game coherence")
    panel_tag(ax, "b", dx=-0.36)

    # (c,d) is there a discrete cluster in there? -----------------------------
    m = arche.merge(react, on=["game_uid", "agent"], how="left")
    u = m[(m.bucket == "unclassified") & (m.n_after_C >= MIN_OBS)
          & (m.n_after_D >= MIN_OBS)]
    Z = StandardScaler().fit_transform(
        u[["p_CgivenC", "q_CgivenD", "coop_rate"]].to_numpy())

    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(0)
    rows = []
    for k in range(2, 9):
        base = KMeans(k, n_init=10, random_state=0).fit(Z)
        aris = []
        for b in range(12):
            idx = rng.choice(len(Z), len(Z), replace=True)
            lab = KMeans(k, n_init=6, random_state=b).fit(Z[idx]).predict(Z)
            aris.append(adjusted_rand_score(base.labels_, lab))
        rows.append({"k": k, "ari_mean": np.mean(aris), "ari_sd": np.std(aris),
                     "silhouette": silhouette_score(Z, base.labels_)})
    st = pd.DataFrame(rows)
    st.to_csv(TABDIR / "T33_cluster_stability.csv", index=False)
    ax.errorbar(st.k, st.ari_mean, yerr=st.ari_sd, fmt="-o", color=FOCUS,
                ms=5, mec=SURFACE, mew=1.1, capsize=2.5, label="bootstrap ARI")
    ax.plot(st.k, st.silhouette, "-s", color=MUTED, ms=4, mec=SURFACE, mew=1.0,
            label="silhouette")
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("stability")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.0, loc="lower left")
    ax.set_title("No natural k")
    panel_tag(ax, "c", dx=-0.34)

    ax = fig.add_subplot(gs[0, 3])
    km = KMeans(4, n_init=10, random_state=0).fit(Z)
    u = u.assign(cluster=km.labels_)
    cen = u.groupby("cluster")[["p_CgivenC", "q_CgivenD", "coop_rate"]].mean()
    cen["n"] = u.cluster.value_counts().sort_index()
    cen.to_csv(TABDIR / "T34_cluster_centres.csv")
    ax.imshow(cen[["p_CgivenC", "q_CgivenD", "coop_rate"]].to_numpy(),
              cmap=CMAP_SEQ, aspect="auto", vmin=0, vmax=1)
    for i in range(cen.shape[0]):
        for j in range(3):
            v = cen.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                    color="white" if v > 0.55 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["p", "q", "coop rate"], fontsize=7)
    ax.set_yticks(range(len(cen)))
    ax.set_yticklabels([f"C{i}  n={int(v)}" for i, v in enumerate(cen.n)],
                       fontsize=6.6)
    ax.grid(False)
    ax.set_title("k = 4 centres")
    panel_tag(ax, "d", dx=-0.44)

    fig.suptitle("Is the abstention set a strategy or noise?", x=0.015, y=1.06,
                 ha="left", fontweight="bold", color=INK)
    savefig(fig, "F34_coherence")


# ==========================================================================
def main():
    rounds = pd.read_parquet(DATADIR / "rounds.parquet")
    arche = pd.read_parquet(DATADIR / "llm_archetypes.parquet")

    arche = posterior_geometry(add_buckets(arche))
    resid_conf = arche[arche.assignment == "approx"]
    print(f"corpus {len(arche):,}   residual {len(resid_conf):,}   "
          f"abstention {(arche.bucket == 'unclassified').sum():,} "
          f"({(arche.bucket == 'unclassified').mean():.1%})")

    react = reactive_coordinates(rounds)
    seqs = sequence_stats(rounds)
    halves = split_half_reactive(rounds)
    print("  building the wide-library deviation table (the slow step)")
    lib = library_table(rounds)

    fig_anatomy(arche, resid_conf)
    fig_geometry(arche, react)
    m = fig_vocabulary(arche, lib)
    fig_signatures(arche, rounds, seqs)
    fig_coherence(arche, react, halves)

    u = m[m.bucket == "unclassified"]
    print()
    print("nearest rule in the abstention set (top 8):")
    print(u.best_family.value_counts(normalize=True).head(8).round(3).to_string())
    print()
    print(f"exactly fitted by the wide library: {(u.dev_extended == 0).mean():.3f}  "
          f"(shuffled null {(u.dev_extended_null == 0).mean():.3f})")


if __name__ == "__main__":
    main()
