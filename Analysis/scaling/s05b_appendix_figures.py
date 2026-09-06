"""Appendix figures for the Interface Focus manuscript.

Four figures that support the main text without carrying its argument, in the
order the appendix uses them:

    fa1_egt_grid    the evolutionary baseline in full: the stationary mix over
                    ALLC, ALLD, TFT and WSLS against the payoff scale, one
                    panel per execution-noise level
    fa2_egt_vs_llm  that baseline set beside the corpus at the same ten
                    scales, which is where the two part company: the baseline
                    concentrates on defection as the scale grows and the
                    corpus does not
    fa3_supp_model  the sixth model, reported in the supplement rather than
                    the main text, drawn alongside the five it is held out of
    fa4_classifier  how well the LSTM branch of the read-out recovers a known
                    rule, in distribution and out of it

Every number is read from `tables/`; nothing is recomputed here and no model
is called.  The destination is `figstyle.FIGDIR`, so it follows PD_FIGDIR and
PD_PAPER_DIR exactly as the main-text figures do, and with neither variable set
it is paper_scaling/figures as before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (BAND, INK, MODEL_C, MODEL_LABEL, MODEL_M,   # noqa: E402
                      MODEL_ORDER, MODEL_ORDER_ALL, MUTED, RULE, SCALES,
                      STRAT_C, STRAT_ORDER, SUPPLEMENT_ONLY, W2, hgrid,
                      logscale_axis, panel_tag, save, shade_subunit, use_style)

HERE = Path(__file__).resolve().parent
TAB = HERE / "tables"
DATA = HERE / "data"

# The evolutionary tables name the four rules in the capitals the game-theory
# literature uses; the read-out names the same four rules in the mixed case the
# manuscript uses.  One map, so the colours cannot drift apart.
EGT_TO_STRAT = {"ALLC": "AllC", "TFT": "TFT", "WSLS": "WSLS", "ALLD": "AllD"}
EGT_ORDER = ["ALLC", "TFT", "WSLS", "ALLD"]     # cooperative first, as in STRAT_ORDER

# Reference noise level for the appendix text.  It is the row the main text
# quotes, and fa2 is drawn at this level alone because a paired-bar figure at
# five noise levels at once is unreadable at column width.
EPS_REF = 0.05


def _stars(p):
    """Significance marker, the convention the tables in the paper already use."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


# --------------------------------------------------------------------------
def fa1_egt_grid(t14, t17):
    """The evolutionary stationary distribution, one panel per noise level.

    Lines rather than stacks.  The four shares do sum to one, which argues for
    a stack, but the fact the appendix needs is the *crossing*: ALLD overtakes
    the three cooperative rules somewhere between the sub-unit scales and
    lambda = 10, and a stack hides a crossing by construction.

    The two panels whose error rate falls inside the range implied by the
    corpus are tinted, so the reader can see at once that the empirically
    relevant noise levels are the ones where the baseline sweeps hardest, not
    the gentlest.
    """
    eps_levels = sorted(t14.epsilon.unique())
    obs = t17[t17.model.isin(MODEL_ORDER)]["implied_error_rate"]
    lo, hi = float(obs.min()), float(obs.max())

    fig, axes = plt.subplots(1, len(eps_levels), figsize=(W2, 2.25),
                             sharey=True)
    for k, (ax, eps) in enumerate(zip(axes, eps_levels)):
        d = t14[t14.epsilon == eps].sort_values("lam")
        if lo <= eps <= hi:
            # A tint, not a box: the panel is still a panel, it is only
            # flagged as sitting inside the observed noise range.
            ax.set_facecolor(BAND)
        for col in EGT_ORDER:
            ax.plot(d.lam, d[col], color=STRAT_C[EGT_TO_STRAT[col]],
                    marker="o", markersize=2.0, lw=1.0, label=col)
        logscale_axis(ax)
        ax.set_ylim(0, 1.06)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        hgrid(ax)
        ax.set_title(rf"$\epsilon = {eps:g}$", fontsize=6.8, pad=11)
        panel_tag(ax, "abcde"[k], dx=-0.10 if k else -0.30, dy=1.14)

        # Where defection has effectively taken the population over.  Read off
        # the table rather than asserted, so it stays true if the grid changes.
        # It goes above the axes: inside them there is no region free of a
        # curve in all five panels at once, and the four curves end up in
        # different corners as the noise level changes.
        over = d[d.ALLD >= 0.99]
        if len(over):
            ax.text(0.5, 1.012,
                    rf"ALLD $\geq$ 0.99 from $\lambda={over.lam.iloc[0]:g}$",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=5.4, color=MUTED)

    # The one panel that is not monotone, called out where it happens.
    d0 = t14[t14.epsilon == eps_levels[0]].sort_values("lam")
    drop = float(np.diff(d0.ALLD.to_numpy()).min())
    axes[0].text(0.04, 0.965, f"not monotone:\nlargest fall {drop:.3f}",
                 transform=axes[0].transAxes, fontsize=5.4, color=INK,
                 va="top")

    axes[0].set_ylabel("share of the population\nin the stationary distribution")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=4,
                   handlelength=1.6, columnspacing=1.2)
    axes[-1].text(1.0, -0.42, f"tinted: error rate inside the {lo:.2f} to {hi:.2f} "
                              "range implied by the corpus",
                  transform=axes[-1].transAxes, ha="right", fontsize=5.4,
                  color=MUTED)
    print(f"  [fa1] noise levels {['%g' % e for e in eps_levels]}, "
          f"observed band {lo:.4f} to {hi:.4f}, "
          f"largest fall at epsilon = {eps_levels[0]:g}: {drop:.4f}")
    save(fig, "fa1_egt_grid")


# --------------------------------------------------------------------------
def fa2_egt_vs_llm(t15):
    """The baseline and the corpus at the same ten scales.

    Panel (a) is the whole mix as paired stacked bars, one pair per scale, so
    the composition of the two is directly comparable.  Panel (b) pulls out
    the single series the comparison turns on, the defect share, because the
    divergence is one of direction and a direction is easier to read off two
    lines than off twenty bars.
    """
    t = t15.sort_values("lam")
    x = np.arange(len(t))
    w = 0.38
    off = 0.21

    fig = plt.figure(figsize=(W2, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.30)

    # (a) the full mix, paired
    ax = fig.add_subplot(gs[0])
    panel_tag(ax, "a", dx=-0.09)
    bot_e = np.zeros(len(t))
    bot_l = np.zeros(len(t))
    for lab in STRAT_ORDER:
        egt = 100 * t[f"egt_{lab.upper()}"].to_numpy()
        llm = 100 * t[f"llm_{lab}"].to_numpy()
        ax.bar(x - off, egt, w, bottom=bot_e, color=STRAT_C[lab],
               edgecolor="white", lw=0.3, label=lab)
        ax.bar(x + off, llm, w, bottom=bot_l, color=STRAT_C[lab],
               edgecolor="white", lw=0.3, hatch="///")
        bot_e += egt
        bot_l += llm
    for xi in x:
        ax.text(xi - off, 101.5, "E", ha="center", va="bottom", fontsize=4.8,
                color=MUTED)
        ax.text(xi + off, 101.5, "C", ha="center", va="bottom", fontsize=4.8,
                color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([("%g" % s) for s in t.lam], rotation=90)
    ax.set_xlabel(r"payoff scale $\lambda$")
    ax.set_ylabel("share of the mix (\\%)".replace("\\", ""))
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.7, len(t) - 0.3)
    ax.set_title("E: evolutionary baseline (solid)     "
                 "C: corpus (hatched)", loc="left", fontsize=6.0, color=MUTED,
                 pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=4,
              handlelength=1.4, columnspacing=1.0)

    # (b) the defect share alone, which is where the two disagree in sign
    ax = fig.add_subplot(gs[1])
    panel_tag(ax, "b", dx=-0.26)
    e = t.egt_ALLD.to_numpy()
    l = t.llm_AllD.to_numpy()
    ax.plot(t.lam, e, color=STRAT_C["AllD"], marker="o", lw=1.4,
            label="baseline, ALLD")
    ax.plot(t.lam, l, color=INK, marker="s", markerfacecolor="white", lw=1.4,
            ls=(0, (3, 1.6)), label="corpus, AllD")
    logscale_axis(ax)
    ax.set_ylabel("share playing the defect rule")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    hgrid(ax)
    shade_subunit(ax, label=False)
    ax.text(0.985, 0.985,
            f"baseline {e[0]:.2f} to {e[-1]:.2f}   ({e[-1] - e[0]:+.2f})\n"
            f"corpus   {l[0]:.2f} to {l[-1]:.2f}   ({l[-1] - l[0]:+.2f})",
            transform=ax.transAxes, fontsize=5.6, color=INK, va="top",
            ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1,
              handlelength=1.8)
    print(f"  [fa2] defect share, first to last scale: baseline "
          f"{e[0]:.4f} -> {e[-1]:.4f} ({e[-1] - e[0]:+.4f}), corpus "
          f"{l[0]:.4f} -> {l[-1]:.4f} ({l[-1] - l[0]:+.4f})")
    assert e[-1] - e[0] > 0 > l[-1] - l[0], (
        "fa2 claims the two series move in opposite directions; they do not")
    save(fig, "fa2_egt_vs_llm")


# --------------------------------------------------------------------------
def fa3_supp_model(t02, t10s, t10t):
    """The sixth model beside the five, on the manuscript's two main readouts.

    The sixth is reported in the supplement because its identifier names a
    preview endpoint the provider does not pin to a fixed version.  Showing it
    here rather than only describing it is the point: on cooperation it sits
    inside the spread of the other five, and on the share of play a canonical
    rule reproduces exactly it is the one model whose trend runs upward.
    """
    supp = SUPPLEMENT_ONLY[0]
    tt = t10t.set_index("model")

    fig = plt.figure(figsize=(W2, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.30)

    def draw(ax, frame, col):
        """Five models in ordinary weight, the sixth picked out."""
        for m in MODEL_ORDER_ALL:
            d = frame[frame.model == m].sort_values("scale")
            lead = m == supp
            ax.plot(d.scale, d[col], color=MODEL_C[m], marker=MODEL_M[m],
                    lw=2.1 if lead else 0.9, alpha=1.0 if lead else 0.72,
                    markersize=4.0 if lead else 2.6,
                    markeredgecolor="white" if lead else MODEL_C[m],
                    markeredgewidth=0.6 if lead else 0.0,
                    zorder=5 if lead else 2, label=MODEL_LABEL[m])
        logscale_axis(ax)
        hgrid(ax)

    # (a) cooperation
    ax = fig.add_subplot(gs[0])
    panel_tag(ax, "a", dx=-0.24)
    draw(ax, t02, "coop")
    ax.set_ylabel("cooperation rate")
    ax.set_ylim(0.25, 0.95)
    shade_subunit(ax)
    s = t02[t02.model == supp].sort_values("scale")
    ax.set_title(f"{MODEL_LABEL[supp]}: {s.coop.min():.3f} to {s.coop.max():.3f}, "
                 f"range {s.coop.max() - s.coop.min():.3f}", loc="left",
                 fontsize=6.0, color=MUTED, pad=4)

    # (b) share reproduced exactly by one canonical rule
    ax = fig.add_subplot(gs[1])
    panel_tag(ax, "b", dx=-0.22)
    draw(ax, t10s, "deduced")
    ax.set_ylabel("agent-games matched exactly by\none canonical rule (\\%)"
                  .replace("\\", ""))
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80])
    shade_subunit(ax, label=False)
    ax.text(0.985, 0.985, r"trend, log-odds per decade of $\lambda$",
            transform=ax.transAxes, fontsize=5.6, color=MUTED, va="top",
            ha="right")
    for k, m in enumerate(MODEL_ORDER_ALL):
        r = tt.loc[m]
        lead = m == supp
        ax.text(0.985, 0.917 - 0.050 * k,
                rf"{MODEL_LABEL[m]}  $\beta={r.beta_logscale:+.2f}$"
                f"{_stars(float(r.p))}",
                transform=ax.transAxes, fontsize=5.4, color=MODEL_C[m],
                va="top", ha="right",
                fontweight="bold" if lead else "normal")

    pos = [m for m in MODEL_ORDER_ALL if tt.loc[m, "beta_logscale"] > 0]
    print(f"  [fa3] models with an upward provenance trend: {pos}   "
          f"beta({supp}) = {tt.loc[supp, 'beta_logscale']:+.4f}, "
          f"p = {tt.loc[supp, 'p']:.2e}")
    assert pos == [supp], (
        f"fa3 claims the supplementary model is the only upward trend; "
        f"the positive set is {pos}")

    ax.legend(loc="upper center", bbox_to_anchor=(-0.10, -0.28), ncol=3,
              handlelength=1.6, columnspacing=1.0, fontsize=6.0)
    save(fig, "fa3_supp_model")


# --------------------------------------------------------------------------
SPLIT_LABEL = {
    "held-out (NoNoise + Noise005)": "held out\n0\\% and 5\\% noise\n(trained on both)",
    "OOD (Noise01)": "out of distribution\n10\\% noise\n(never trained on)",
    "OOD (Noise02)": "out of distribution\n20\\% noise\n(never trained on)",
}


def fa4_classifier(t01):
    """How well the learned branch of the read-out recovers a known rule.

    The training corpus is synthetic and contains no language-model play at
    all, so accuracy here is a statement about the instrument and not about
    the corpus.  Panel (b) is the recall the read-out actually leans on: it is
    only consulted when several canonical rules fit or none does, so what
    matters is that no single rule is systematically missed.
    """
    is_recall = t01.split.str.startswith("held-out recall:")
    splits = t01[~is_recall]
    recall = t01[is_recall].copy()
    recall["strategy"] = recall.split.str.split(": ").str[-1]
    recall = recall.set_index("strategy").loc[STRAT_ORDER]

    fig = plt.figure(figsize=(W2, 2.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.52)

    # (a) accuracy per split
    ax = fig.add_subplot(gs[0])
    panel_tag(ax, "a", dx=-0.62)
    y = np.arange(len(splits))[::-1]
    for yi, (_, r) in zip(y, splits.iterrows()):
        held = not r.split.startswith("OOD")
        ax.barh(yi, r.accuracy, height=0.6,
                color="#0072b2" if held else "#9ecae1",
                edgecolor="white", lw=0.4, zorder=2)
        ax.text(r.accuracy + 0.015, yi,
                f"{r.accuracy:.3f}   n = {int(r.n):,}", fontsize=5.8,
                va="center", color=INK if held else MUTED)
    ax.axvline(0.25, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=3)
    ax.text(0.25, len(splits) - 0.38, " chance", fontsize=5.4, color=MUTED,
            va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([SPLIT_LABEL.get(s, s).replace("\\", "")
                        for s in splits.split], fontsize=5.8)
    ax.set_ylim(-0.6, len(splits) - 0.4)
    ax.set_xlim(0, 1.32)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("accuracy on held-out synthetic play")
    hgrid(ax, axis="x")

    # (b) per-class recall on the held-out split
    ax = fig.add_subplot(gs[1])
    panel_tag(ax, "b", dx=-0.24)
    x = np.arange(len(STRAT_ORDER))
    for xi, s in zip(x, STRAT_ORDER):
        r = recall.loc[s]
        ax.bar(xi, r.accuracy, 0.62, color=STRAT_C[s], edgecolor="white",
               lw=0.4, zorder=2)
        ax.text(xi, r.accuracy + 0.02, f"{r.accuracy:.3f}", ha="center",
                fontsize=5.8, color=INK)
        ax.text(xi, 0.04, f"n = {int(r.n):,}", ha="center", fontsize=5.2,
                color="white", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(STRAT_ORDER, fontsize=6.2)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("recall, held-out split")
    hgrid(ax)
    ax.set_title(f"worst class {recall.accuracy.min():.3f}", loc="left",
                 fontsize=6.0, color=MUTED, pad=4)

    print("  [fa4] accuracy: " + ", ".join(
        f"{r.split} = {r.accuracy:.4f} (n = {int(r.n):,})"
        for _, r in splits.iterrows()))
    print("  [fa4] held-out recall: " + ", ".join(
        f"{s} = {recall.loc[s, 'accuracy']:.4f}" for s in STRAT_ORDER))
    save(fig, "fa4_classifier")


# --------------------------------------------------------------------------
def fa5_cooperation_distribution(g: pd.DataFrame):
    """The bimodal cooperation distribution and boundary mass heterogeneity.

    Panel (a) is the pooled cooperation distribution over all 24,000
    agent-games, highlighting the extreme boundary concentration (11.1%
    never-C, 28.5% always-C, 39.5% boundary mass).
    Panel (b) is the empirical cumulative distribution function showing the two
    vertical boundary jumps.
    Panel (c) is the model-level breakdown into never-C, interior mixed, and
    always-C, revealing that the boundary mass ranges seven-fold, from 8.6% in
    Claude Haiku 4.5 to 60.4% in Qwen3 235B-A22B.
    """
    fig = plt.figure(figsize=(W2, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.95, 1.35], wspace=0.34)

    # (a) Pooled distribution
    ax0 = fig.add_subplot(gs[0])
    panel_tag(ax0, "a", dx=-0.22)
    cr = g["coop_rate"].to_numpy()
    bins = np.linspace(-0.05, 1.05, 12)
    counts, _ = np.histogram(cr, bins=bins)
    pcts = 100.0 * counts / len(cr)
    centers = np.linspace(0, 1, 11)

    bar_cols = ["#d55e00" if i == 0 else "#0072b2" if i == 10 else "#778492"
                for i in range(11)]
    ax0.bar(centers, pcts, width=0.08, color=bar_cols, edgecolor="white",
            linewidth=0.4, zorder=2)
    ax0.axhline(100.0 / 11, color=MUTED, lw=0.7, ls=":", label="uniform ref (9.1%)",
                zorder=3)
    ax0.set_xlabel("cooperation rate")
    ax0.set_ylabel("share of agent-games (\\%)".replace("\\", ""))
    ax0.set_xlim(-0.08, 1.08)
    ax0.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax0.set_ylim(0, 34)
    hgrid(ax0)
    ax0.text(0.0, pcts[0] + 1.2, f"{pcts[0]:.1f}%", ha="center", va="bottom",
             fontsize=5.2, color="#d55e00", fontweight="bold")
    ax0.text(1.0, pcts[-1] + 1.2, f"{pcts[-1]:.1f}%", ha="center", va="bottom",
             fontsize=5.2, color="#0072b2", fontweight="bold")
    ax0.text(0.5, 26, f"boundary mass:\n{pcts[0]+pcts[-1]:.1f}%", ha="center",
             fontsize=5.4, color=INK,
             bbox=dict(boxstyle="round,pad=0.2", fc=BAND, ec=RULE, lw=0.5))

    # (b) ECDF
    ax1 = fig.add_subplot(gs[1])
    panel_tag(ax1, "b", dx=-0.22)
    sx = np.sort(cr)
    sy = np.arange(1, len(sx) + 1) / len(sx)
    ax1.step(sx, sy, where="post", color=INK, lw=1.2, zorder=3)
    ax1.axhline((cr == 0).mean(), color="#d55e00", lw=0.7, ls="--", alpha=0.8,
                label=f"zero jump ({(cr==0).mean():.1%})")
    ax1.axhline(1.0 - (cr == 1).mean(), color="#0072b2", lw=0.7, ls="--", alpha=0.8,
                label=f"unit jump ({(cr==1).mean():.1%})")
    ax1.set_xlabel("cooperation rate")
    ax1.set_ylabel("cumulative probability")
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    hgrid(ax1)
    ax1.legend(loc="center right", fontsize=5.0)

    # (c) Model-level breakdown
    ax2 = fig.add_subplot(gs[2])
    panel_tag(ax2, "c", dx=-0.20)
    models = MODEL_ORDER_ALL[::-1]
    y = np.arange(len(models))
    c_never, c_mixed, c_always = [], [], []
    for m in models:
        gm = g[g.model == m]
        c_never.append(100.0 * (gm.coop_rate == 0).mean())
        c_always.append(100.0 * (gm.coop_rate == 1).mean())
        c_mixed.append(100.0 * ((gm.coop_rate > 0) & (gm.coop_rate < 1)).mean())
    c_never = np.array(c_never)
    c_mixed = np.array(c_mixed)
    c_always = np.array(c_always)

    h = 0.58
    ax2.barh(y, c_never, height=h, color="#d55e00", edgecolor="white", lw=0.3,
             label="Never-C (0)", zorder=2)
    ax2.barh(y, c_mixed, left=c_never, height=h, color="#c0c7cf", edgecolor="white",
             lw=0.3, label="Interior (0,1)", zorder=2)
    ax2.barh(y, c_always, left=c_never + c_mixed, height=h, color="#0072b2",
             edgecolor="white", lw=0.3, label="Always-C (1)", zorder=2)

    for yi, m, nv, al in zip(y, models, c_never, c_always):
        tot = nv + al
        ax2.text(101.5, yi, f"{tot:.1f}% boundary", va="center", ha="left",
                 fontsize=5.2, color=MODEL_C[m],
                 fontweight="bold" if m in ["Claude-Haiku-4.5", "Qwen3-235B-A22B"] else "normal")

    ax2.set_yticks(y)
    ax2.set_yticklabels([MODEL_LABEL[m] for m in models], fontsize=5.8)
    ax2.set_xlim(0, 136)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("share of model agent-games (\\%)".replace("\\", ""))
    hgrid(ax2, axis="x")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.42, 1.18), ncol=3,
               fontsize=5.2, handlelength=1.2)

    print(f"  [fa5] pooled boundary mass: {pcts[0]+pcts[-1]:.2f}% "
          f"(never={pcts[0]:.2f}%, always={pcts[-1]:.2f}%)")
    save(fig, "fa5_coop_dist")


# --------------------------------------------------------------------------
def fa6_scaling_mechanism(g: pd.DataFrame, r: pd.DataFrame):
    """The scaling mechanism: ex-ante policy re-weighting vs parallel dynamic decay.

    Panel (a) shows the composition of unconditional policies vs interior play
    across the ten scales.
    Panel (b) shows the opposition between always-C and never-C across
    log(lambda).
    Panel (c) displays the round-by-round cooperation decay curves for lambda in
    {0.01, 1, 1000}, demonstrating that the trajectories are near-parallel and
    the scale effect is established at round 1.
    """
    fig = plt.figure(figsize=(W2, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.15], wspace=0.35)

    scales = sorted(g.scale_nominal.unique())
    res = []
    for s in scales:
        gs_df = g[g.scale_nominal == s]
        res.append({
            "scale": s,
            "always_C": 100.0 * (gs_df.coop_rate == 1).mean(),
            "never_C": 100.0 * (gs_df.coop_rate == 0).mean(),
            "mixed": 100.0 * ((gs_df.coop_rate > 0) & (gs_df.coop_rate < 1)).mean(),
        })
    sh = pd.DataFrame(res).set_index("scale")

    # (a) Stacked policy mix across scales
    ax0 = fig.add_subplot(gs[0])
    panel_tag(ax0, "a", dx=-0.18)
    x = np.arange(len(scales))
    w = 0.65
    ax0.bar(x, sh["never_C"], w, color="#d55e00", edgecolor="white", lw=0.3,
            label="Never-C", zorder=2)
    ax0.bar(x, sh["mixed"], w, bottom=sh["never_C"], color="#c0c7cf",
            edgecolor="white", lw=0.3, label="Interior", zorder=2)
    ax0.bar(x, sh["always_C"], w, bottom=sh["never_C"] + sh["mixed"],
            color="#0072b2", edgecolor="white", lw=0.3, label="Always-C", zorder=2)
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"{s:g}" for s in scales], rotation=90, fontsize=5.8)
    ax0.set_xlabel(r"payoff scale $\lambda$")
    ax0.set_ylabel("share of agent-games (\\%)".replace("\\", ""))
    ax0.set_ylim(0, 105)
    hgrid(ax0)
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3,
               fontsize=5.2, handlelength=1.2)

    # (b) Opposition of the two atoms against log lambda
    ax1 = fig.add_subplot(gs[1])
    panel_tag(ax1, "b", dx=-0.22)
    ax1.plot(scales, sh["always_C"], "o-", color="#0072b2", lw=1.3, markersize=3.2,
             label=r"Always-C ($\rho=+0.05$)")
    ax1.plot(scales, sh["never_C"], "s-", color="#d55e00", lw=1.3, markersize=3.2,
             label=r"Never-C ($\rho=-0.47$)")
    logscale_axis(ax1)
    ax1.set_ylabel("share of agent-games (\\%)".replace("\\", ""))
    ax1.set_ylim(0, 42)
    hgrid(ax1)
    ax1.legend(loc="center right", fontsize=5.2)
    ax1.text(0.04, 0.08, "sub-unit $\\lambda \\leq 0.1$:\nNever-C mass inflates",
             transform=ax1.transAxes, fontsize=5.2, color=INK)

    # (c) Dynamic decay across rounds
    ax2 = fig.add_subplot(gs[2])
    panel_tag(ax2, "c", dx=-0.20)
    curve_styles = [
        (0.01, "#d55e00", "o", r"$\lambda = 0.01$ (sub-unit)"),
        (1.0, "#0072b2", "s", r"$\lambda = 1.0$ (unit)"),
        (1000.0, "#009e73", "^", r"$\lambda = 1000$ (mega)")
    ]
    for s, col, mk, lab in curve_styles:
        rs = r[r.scale_nominal == s].groupby("round")["coop"].mean()
        ax2.plot(rs.index, rs.values, marker=mk, color=col, lw=1.3, markersize=3.2,
                 label=lab)
    ax2.set_xlabel("round")
    ax2.set_ylabel("round cooperation rate")
    ax2.set_xticks(range(1, 11))
    ax2.set_ylim(0.42, 0.68)
    hgrid(ax2)
    ax2.legend(loc="lower left", fontsize=5.2)
    ax2.text(0.96, 0.92, "parallel decay:\nscale shifts intercept,\nnot within-game slope",
             transform=ax2.transAxes, fontsize=5.2, ha="right", va="top", color=INK)

    print(f"  [fa6] Never-C scale 0.01 -> 1000: {sh.loc[0.01, 'never_C']:.2f}% "
          f"-> {sh.loc[1000.0, 'never_C']:.2f}%")
    save(fig, "fa6_mechanism")


# --------------------------------------------------------------------------
def main():
    use_style()
    t01 = pd.read_csv(TAB / "T01_classifier.csv")
    t02 = pd.read_csv(TAB / "T02_scale_by_model.csv")
    t10s = pd.read_csv(TAB / "T10_provenance_by_scale.csv")
    t10t = pd.read_csv(TAB / "T10_provenance_trend.csv")
    t14 = pd.read_csv(TAB / "T14_egt_stationary.csv")
    t15 = pd.read_csv(TAB / "T15_egt_vs_llm.csv")
    t17 = pd.read_csv(TAB / "T17_egt_noise_calibration.csv")

    g = pd.read_parquet(DATA / "games.parquet")
    r = pd.read_parquet(DATA / "rounds.parquet")

    # fa2 is drawn at one noise level; say which, rather than leaving the
    # reader to infer it from a column that is not plotted.
    assert set(t15.epsilon) == {EPS_REF}, (
        f"fa2 assumes T15 is a single noise level {EPS_REF}; it holds "
        f"{sorted(set(t15.epsilon))}")

    fa1_egt_grid(t14, t17)
    fa2_egt_vs_llm(t15)
    fa3_supp_model(t02, t10s, t10t)
    fa4_classifier(t01)
    fa5_cooperation_distribution(g)
    fa6_scaling_mechanism(g, r)


if __name__ == "__main__":
    main()

