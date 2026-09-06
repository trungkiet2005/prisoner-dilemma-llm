"""Figures for the payoff-scaling manuscript.

Eight figures, each carrying one claim, in the argument order of the
manuscript:

    f1_scaling        the manipulation is inert in theory, and cooperation
                      moves anyway, differently in each model, so pooling
                      cancels it
    f2_inference      the movement survives every inferential test we can put
                      to it, and the model curves are uncorrelated in shape
    f3_language       the movement is language-dependent
    f4_heterogeneity  where the variance actually sits: the model-by-scale
                      interaction dwarfs the scale main effect
    f5_persona        the persona instruction only takes hold above the
                      sub-unit boundary
    f6_ruleforce      play becomes less rule-governed as the payoffs grow
    f7_strategy_mix   which of the four canonical strategies is played, one
                      panel each
    f8_firstmove      the effect is already present at the opening move

Filenames.  These stems were fig1_..fig6_ while paper_scaling/ was the only
consumer.  The second manuscript's figure directory already holds 29 legacy
PDFs whose names start fig01_, fig02_, fig10_, so a stem like fig1_scaling sat
one character away from fig10_egt_overview and sorted into the middle of them.
The stems are therefore f1_..f8_.  When the figures are written to their
historical home the old stems are emitted as well, so paper_scaling/main.tex
keeps building from an unedited source; see LEGACY_ALIAS below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (BAND, INK, IS_LEGACY_FIGDIR, LANG_C,  # noqa: E402
                      LANG_LABEL, LANG_M, LANG_ORDER,
                      MODEL_C, MODEL_LABEL, MODEL_M, MODEL_ORDER, MODEL_SHORT, MUTED,
                      PROV_C, PROV_LABEL, PROV_ORDER, RULE, SCALES, STRAT_C,
                      STRAT_ORDER, STRAT_TITLE, W1, W2, hgrid, logscale_axis,
                      panel_tag,
                      save, shade_subunit, use_style)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TAB = HERE / "tables"

# Old stem for every figure that only changed name, so that a manuscript
# directory written before the rename still finds its files.  f2_inference and
# f4_heterogeneity are new and have no alias.  f6_ruleforce is deliberately
# absent too: its panel (a) is not the panel the old fig4_strategy carried, and
# silently swapping the content under the old name would leave the existing
# caption describing a panel that is no longer there.  The old figure is
# redrawn unchanged by fig4_strategy_legacy() instead.
LEGACY_ALIAS = {
    "f1_scaling": ("fig1_scaling",),
    "f3_language": ("fig2_language",),
    "f5_persona": ("fig3_persona",),
    "f7_strategy_mix": ("fig5_strategy_mix",),
    "f8_firstmove": ("fig6_firstmove",),
}


def alias(name):
    """Legacy stems to emit beside `name`, empty away from the default dir."""
    return LEGACY_ALIAS.get(name, ()) if IS_LEGACY_FIGDIR else ()


# MODEL_SHORT carries a newline so it can sit under a tick; the annotation
# blocks want the same brevity on one line.
COMPACT = {m: MODEL_SHORT[m].replace("\n", " ") for m in MODEL_ORDER}


def stars(p):
    """Significance marker, the convention the tables in the paper already use."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def ptext(p):
    """A p value as prose, floored where the test cannot resolve further."""
    return "$p<0.001$" if p < 1e-3 else f"$p={p:.3f}$"


# --------------------------------------------------------------------------
def f1_scaling(g, t02, t03, t04):
    fig = plt.figure(figsize=(W2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.35, 1.0], wspace=0.34)

    # (a) the same game, three renderings
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_tag(ax, "a", dx=-0.02, dy=1.02)
    ax.set_title("one game, three renderings", loc="left", pad=10)
    # The agent is shown penalties and told to minimise them, so a
    # smaller number is a better outcome and OptionA is the dominant action.
    xs = [0.46, 0.76]
    for k, lam in enumerate([0.01, 1, 1000]):
        top = 0.93 - 0.26 * k
        cells = [[6 * lam, 0 * lam], [10 * lam, 2 * lam]]
        ax.text(0.0, top, rf"$\lambda={lam:g}$", fontsize=7, color=INK,
                va="center")
        for j, opp in enumerate(["A", "B"]):
            ax.text(xs[j], top, f"opp {opp}", fontsize=5.6, color=MUTED,
                    ha="center", va="center")
        for i, own in enumerate(["A", "B"]):
            yy = top - 0.075 - 0.062 * i
            ax.text(0.30, yy, f"you {own}", fontsize=5.6, color=MUTED,
                    ha="right", va="center")
            for j in range(2):
                ax.text(xs[j], yy, ("%g" % cells[i][j]), fontsize=6.8,
                        color=INK, ha="center", va="center")
        ax.add_patch(plt.Rectangle((0.33, top - 0.170), 0.60, 0.135,
                                   fill=False, ec=RULE, lw=0.6))
    # "penalties", not "years of imprisonment": the prompt casts the game as an
    # arrest but never names a unit, so the gloss was an interpretation.
    ax.text(0.0, 0.05, "penalties, to be minimized.\n"
                       "Identical preferences, best replies,\n"
                       "equilibria and replicator dynamics.",
            fontsize=6, color=MUTED, style="italic", va="bottom")

    # (b) cooperation against the scale, per model
    ax = fig.add_subplot(gs[1])
    panel_tag(ax, "b")
    for m in MODEL_ORDER:
        d = t02[t02.model == m].sort_values("scale")
        ax.fill_between(d.scale, d.lo, d.hi, color=MODEL_C[m], alpha=0.13, lw=0)
        ax.plot(d.scale, d.coop, color=MODEL_C[m], marker=MODEL_M[m],
                label=MODEL_LABEL[m])
    logscale_axis(ax)
    ax.set_ylabel("cooperation rate")
    ax.set_ylim(0.25, 0.95)
    hgrid(ax)
    shade_subunit(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
              handlelength=1.6, columnspacing=1.0)

    # (c) pooling cancels it
    #
    # The frame `g` carries all six models; the main text reports five and the
    # supplement the sixth.  Averaging the unfiltered frame put a six-model
    # black curve next to an annotation quoting the five-model pooled range
    # from T04, and the two disagreed: 0.125 drawn against 0.154 printed, and
    # 0.154 is the number the abstract and the contributions both quote.
    # Filter first so the curve and its caption are the same quantity.
    ax = fig.add_subplot(gs[2])
    panel_tag(ax, "c")
    per_model = (g[g.model.isin(MODEL_ORDER)]
                 .groupby(["model", "scale_nominal"])["coop_rate"]
                 .mean().unstack())
    pooled = per_model.mean(axis=0)
    for m in MODEL_ORDER:
        d = per_model.loc[m]
        ax.plot(d.index, d.values - d.mean(), color=MODEL_C[m], lw=0.8,
                alpha=0.75, marker=MODEL_M[m], markersize=2.2)
    ax.plot(pooled.index, pooled.values - pooled.mean(), color=INK, lw=1.8,
            marker="o", markersize=3.4, label="average over models", zorder=5)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    logscale_axis(ax)
    ax.set_ylabel("cooperation, centred within model")
    hgrid(ax)
    pr = float(t04.pooled_range.iloc[0])
    mw = float(t04.mean_within_model_range.iloc[0])
    drawn = float(pooled.max() - pooled.min())
    print(f"  [check] pooled curve drawn in f1(c): range = {drawn:.4f}   "
          f"T04_pooling.pooled_range = {pr:.4f}   "
          f"{'MATCH' if round(drawn, 3) == round(pr, 3) else 'MISMATCH'}")
    assert round(drawn, 3) == round(pr, 3), (
        f"panel (c) pooled range {drawn:.4f} does not match the annotated "
        f"T04 pooled_range {pr:.4f}; the two are computed over different "
        f"model sets")
    ax.set_title(f"within model {mw:.3f}   pooled {pr:.3f}", loc="left",
                 fontsize=6.5, color=MUTED, pad=3)
    ax.legend(loc="lower right")
    save(fig, "f1_scaling", alias("f1_scaling"))


# --------------------------------------------------------------------------
def f2_inference(t03, t04, t04c, t09):
    """Three ways of asking whether the movement in figure 1 is real.

    (a) is the effect size with its uncertainty and its permutation test,
    (b) is the same question asked of a clustered logistic regression, which
    respects that the twenty rounds of a game are one unit and not twenty, and
    (c) is the reason the first two cannot be pooled: the curves the five
    models trace have no common shape, so an average over models is not an
    estimate of a shared effect.
    """
    fig = plt.figure(figsize=(W2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.15], wspace=0.62)

    # (a) forest plot of the across-scale range, with bootstrap interval
    ax = fig.add_subplot(gs[0])
    panel_tag(ax, "a", dx=-0.52)
    t = t03.set_index("model").loc[MODEL_ORDER]
    y = np.arange(len(MODEL_ORDER))[::-1]
    for yi, m in zip(y, MODEL_ORDER):
        r = t.loc[m]
        # Caps rather than a bare segment: the intervals are narrow enough
        # that a plain line all but disappears under the point marker.
        ax.plot([r.lo, r.hi], [yi, yi], color=MODEL_C[m], lw=1.0, zorder=2)
        for e in (r.lo, r.hi):
            ax.plot([e, e], [yi - 0.19, yi + 0.19], color=MODEL_C[m], lw=1.0,
                    zorder=2)
        ax.scatter([r["range"]], [yi], s=18, color=MODEL_C[m], zorder=3,
                   marker=MODEL_M[m], edgecolor="white", linewidth=0.5)
        ax.text(r.hi + 0.022, yi, ptext(float(r.p_perm)), fontsize=5.6,
                color=MUTED, va="center")
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=6.2)
    ax.set_ylim(-0.7, len(MODEL_ORDER) - 0.3)
    ax.set_xlim(0, 0.95)
    ax.set_xlabel("cooperation range across the\nten scales (95\\% bootstrap)"
                  .replace("\\", ""))
    hgrid(ax, axis="x")
    ax.set_title("effect size", loc="left", fontsize=6.8, color=MUTED, pad=3)

    # (b) largest log-odds contrast against the reference scale
    ax = fig.add_subplot(gs[1])
    panel_tag(ax, "b", dx=-0.30)
    q = t09.set_index("model").loc[MODEL_ORDER]
    for yi, m in zip(y, MODEL_ORDER):
        r = q.loc[m]
        ax.barh(yi, r.max_abs_logodds_vs_lambda1, height=0.62,
                color=MODEL_C[m], edgecolor="white", lw=0.4)
        ax.text(r.max_abs_logodds_vs_lambda1 + 0.09, yi,
                rf"$\chi^2_{{{int(r.df)}}}={r.wald_chi2_scale:.0f}$"
                f"{stars(float(r.p_scale))}",
                fontsize=5.6, color=MUTED, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)   # rows are named once, in panel (a)
    ax.set_ylim(-0.7, len(MODEL_ORDER) - 0.3)
    ax.set_xlim(0, 5.9)
    ax.set_xlabel("largest absolute log-odds\n"
                  # "dyad", not "game": the manuscripts use the dyad as the
                  # unit of clustering throughout, and the two words naming the
                  # same object in the figure and in the prose reads as two
                  # different specifications.
                  r"against $\lambda=1$, clustered on the dyad")
    hgrid(ax, axis="x")
    ax.set_title("regression", loc="left", fontsize=6.8, color=MUTED, pad=3)

    # (c) the model curves have no shape in common
    ax = fig.add_subplot(gs[2])
    panel_tag(ax, "c", dx=-0.36)
    C = t04c.set_index("model").reindex(index=MODEL_ORDER,
                                        columns=MODEL_ORDER).to_numpy(float)
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(MODEL_ORDER)):
        for j in range(len(MODEL_ORDER)):
            v = C[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                    color="white" if abs(v) > 0.55 else INK)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_SHORT[m].replace("\n", " ") for m in MODEL_ORDER],
                       fontsize=5.2, rotation=90)
    ax.set_yticklabels([MODEL_SHORT[m].replace("\n", " ") for m in MODEL_ORDER],
                       fontsize=5.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, ticks=[-1, 0, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=5.6)
    cb.set_label("correlation of curve shape", fontsize=5.8)
    mean_r = float(t04.mean_pairwise_shape_corr.iloc[0])
    n_neg = int(t04.n_negative_pairs.iloc[0])
    n_pairs = int(t04.n_pairs.iloc[0])
    ax.set_title(f"mean $r$ = {mean_r:.2f}, {n_neg} of {n_pairs} "
                 "pairs negative", loc="left", fontsize=6.0, color=MUTED,
                 pad=4)
    save(fig, "f2_inference", alias("f2_inference"))


# --------------------------------------------------------------------------
def f3_language(g, t05):
    fig, axes = plt.subplots(1, 5, figsize=(W2, 2.0), sharey=True)
    for ax, m in zip(axes, MODEL_ORDER):
        d = g[g.model == m]
        for lang in LANG_ORDER:
            s = (d[d.language == lang].groupby("scale_nominal")["coop_rate"]
                 .mean())
            ax.plot(s.index, s.values, color=LANG_C[lang], lw=0.95,
                    marker=LANG_M[lang], markersize=2.2,
                    label=LANG_LABEL[lang])
        logscale_axis(ax)
        ax.set_title(MODEL_LABEL[m], fontsize=6.8)
        ax.set_ylim(0.0, 1.0)
        hgrid(ax)
        rng = t05[t05.model == m].range_over_scales
        ax.text(0.03, 0.03, f"range {rng.min():.2f}-{rng.max():.2f}",
                transform=ax.transAxes, fontsize=5.8, color=MUTED)
    axes[0].set_ylabel("cooperation rate")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=5,
                   handlelength=1.5, columnspacing=1.2)
    save(fig, "f3_language", alias("f3_language"))


# --------------------------------------------------------------------------
VAR_ORDER = [
    ("C(model)", "model"),
    ("C(model):C(language)", r"model $\times$ language"),
    ("C(language)", "language"),
    ("C(language):C(scale_nominal)", r"language $\times$ scale"),
    ("C(model):C(scale_nominal)", r"model $\times$ scale"),
    ("C(scale_nominal)", "scale"),
    ("Residual", "residual"),
]
# The two rows the argument turns on are placed next to each other on purpose:
# the interaction sits directly above the main effect, so the reader compares
# their lengths without having to hold a number in mind.
HILITE = {"C(model):C(scale_nominal)", "C(scale_nominal)"}


def f4_heterogeneity(t05, t06):
    """Where the movement actually lives.

    (a) the across-scale range is not one number per model: it depends on the
    prompt language, by a factor of several within the same model.
    (b) the same fact stated as a variance decomposition over the 250 cell
    means: the model-by-scale interaction carries about three times the share
    the scale main effect does, which is why a pooled scale effect is the
    wrong summary.
    """
    fig = plt.figure(figsize=(W2, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.62)

    # (a) range by model and language
    ax = fig.add_subplot(gs[0])
    panel_tag(ax, "a", dx=-0.40)
    M = (t05.pivot(index="model", columns="language", values="range_over_scales")
         .reindex(index=MODEL_ORDER, columns=LANG_ORDER).to_numpy(float))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=float(np.nanmax(M)))
    hi = 0.68 * float(np.nanmax(M))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.6, color="white" if M[i, j] > hi else INK)
    ax.set_xticks(range(len(LANG_ORDER)))
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([LANG_LABEL[l] for l in LANG_ORDER], fontsize=5.8,
                       rotation=30, ha="right")
    ax.set_yticklabels([MODEL_SHORT[m].replace("\n", " ") for m in MODEL_ORDER],
                       fontsize=5.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=5.6)
    cb.set_label("range", fontsize=5.8, labelpad=1)
    ax.set_title("range across the ten scales,\nby model and prompt language",
                 loc="left", fontsize=6.4, color=MUTED, pad=4)

    # (b) variance decomposition
    ax = fig.add_subplot(gs[1])
    panel_tag(ax, "b", dx=-0.34)
    v = t06.set_index("term")
    terms = [t for t, _ in VAR_ORDER if t in v.index]
    y = np.arange(len(terms))[::-1]
    for yi, term in zip(y, terms):
        pct = float(v.loc[term, "pct_variance"])
        strong = term in HILITE
        ax.barh(yi, pct, height=0.66,
                color="#0072b2" if strong else "#c9d6df",
                edgecolor="white", lw=0.4, zorder=2)
        pr = v.loc[term, "PR(>F)"]
        mark = "" if pd.isna(pr) else "  " + stars(float(pr))
        ax.text(pct + 0.9, yi, f"{pct:.1f}\\%{mark}".replace("\\", ""),
                fontsize=5.8, va="center",
                color=INK if strong else MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for t, lab in VAR_ORDER if t in v.index],
                       fontsize=6.2)
    for tick, term in zip(ax.get_yticklabels(), terms):
        if term in HILITE:
            tick.set_color(INK)
            tick.set_fontweight("bold")
    ax.set_xlabel("share of the variance across the 250 cell means (\\%)"
                  .replace("\\", ""))
    ax.set_xlim(0, 56)
    hgrid(ax, axis="x")

    inter = float(v.loc["C(model):C(scale_nominal)", "pct_variance"])
    main = float(v.loc["C(scale_nominal)", "pct_variance"])
    i_y = list(y)[terms.index("C(model):C(scale_nominal)")]
    m_y = list(y)[terms.index("C(scale_nominal)")]
    xb = max(inter, main) + 12.0
    ax.plot([xb, xb + 1.4, xb + 1.4, xb], [m_y, m_y, i_y, i_y],
            color=MUTED, lw=0.6, clip_on=False)
    ax.text(xb + 2.4, (i_y + m_y) / 2,
            f"interaction is\n{inter / main:.1f}$\\times$ the\nmain effect",
            fontsize=5.8, color=INK, va="center")
    ax.set_title("250 cell means: 5 models $\\times$ 5 languages "
                 "$\\times$ 10 scales", loc="left", fontsize=6.0,
                 color=MUTED, pad=4)
    save(fig, "f4_heterogeneity", alias("f4_heterogeneity"))


# --------------------------------------------------------------------------
def f5_persona(per, t07):
    fig = plt.figure(figsize=(W2, 2.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.26)

    ax = fig.add_subplot(gs[0])
    panel_tag(ax, "a")
    for m in MODEL_ORDER:
        d = per[per.model == m].sort_values("scale")
        ax.plot(d.scale, d.persona_effect, color=MODEL_C[m], marker=MODEL_M[m],
                label=MODEL_LABEL[m])
    ax.axhline(0, color=INK, lw=0.7)
    logscale_axis(ax)
    ax.set_ylabel("persona effect\n(cooperative $-$ selfish)")
    ax.set_ylim(-1.02, 0.85)
    hgrid(ax)
    shade_subunit(ax, label=False)
    ax.text(0.0316, 0.80, "all payoffs $\\leq 1$", ha="center", fontsize=5.8,
            color=MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3,
              handlelength=1.6, columnspacing=1.0)

    ax = fig.add_subplot(gs[1])
    panel_tag(ax, "b")
    y = np.arange(len(MODEL_ORDER))[::-1]
    for yi, m in zip(y, MODEL_ORDER):
        r = t07[t07.model == m].iloc[0]
        ax.plot([r.persona_effect_subunit, r.persona_effect_suprunit], [yi, yi],
                color=MODEL_C[m], lw=1.1, zorder=1)
        ax.scatter([r.persona_effect_subunit], [yi], s=22, facecolor="white",
                   edgecolor=MODEL_C[m], zorder=3, lw=1.0)
        ax.scatter([r.persona_effect_suprunit], [yi], s=22,
                   color=MODEL_C[m], zorder=3)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=6.2)
    ax.set_xlabel("persona effect")
    ax.set_xlim(-1.05, 0.85)
    hgrid(ax, axis="x")
    ax.scatter([], [], s=22, facecolor="white", edgecolor=MUTED,
               label=r"$\lambda \leq 0.1$")
    ax.scatter([], [], s=22, color=MUTED, label=r"$\lambda \geq 0.25$")
    ax.legend(loc="lower right", handletextpad=0.3)
    save(fig, "f5_persona", alias("f5_persona"))


# --------------------------------------------------------------------------
def f6_ruleforce(d, t10s, t10t):
    """How far play sits from the canonical rule base, two ways.

    Panel (a) was a single stack pooled over models.  Pooling is the thing
    this paper argues against, and the per-model trends here do not agree
    either in size or, once the sixth model is included, in sign, so the
    pooled stack was reporting an average of curves that have no common shape.
    It is now one line per model with that model's own trend beside it.
    """
    fig = plt.figure(figsize=(W2, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.25], wspace=0.30)

    # (a) share matched exactly by one canonical rule, per model
    ax = fig.add_subplot(gs[0])
    panel_tag(ax, "a")
    tt = t10t.set_index("model")
    for m in MODEL_ORDER:
        s = t10s[t10s.model == m].sort_values("scale")
        ax.plot(s.scale, s.deduced, color=MODEL_C[m], marker=MODEL_M[m],
                label=MODEL_LABEL[m])
    logscale_axis(ax)
    ax.set_ylabel("agent-games matched exactly by\none canonical rule (\\%)"
                  .replace("\\", ""))
    # Headroom for the trend block.  The curves top out near 78 per cent and
    # the block has to clear the Qwen curve, the highest across the whole
    # axis, so the block sits above the data rather than inside it.
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80])
    hgrid(ax)
    shade_subunit(ax, label=False)
    # The trend is a clustered logistic slope on log10(lambda), so the units
    # are log-odds per decade of the payoff scale.
    ax.text(0.985, 0.985, r"trend, log-odds per decade of $\lambda$",
            transform=ax.transAxes, fontsize=5.6, color=MUTED, va="top",
            ha="right")
    for k, m in enumerate(MODEL_ORDER):
        r = tt.loc[m]
        ax.text(0.985, 0.917 - 0.058 * k,
                rf"{COMPACT[m]}  $\beta={r.beta_logscale:+.2f}$"
                f"{stars(float(r.p))}",
                transform=ax.transAxes, fontsize=5.6, color=MODEL_C[m],
                va="top", ha="right")

    # (b) distance to the nearest canonical rule
    ax = fig.add_subplot(gs[1])
    panel_tag(ax, "b")
    for m in MODEL_ORDER:
        s = d[d.model == m].groupby("scale_nominal")["min_deviations"].mean()
        ax.plot(s.index, s.values, color=MODEL_C[m], marker=MODEL_M[m],
                label=MODEL_LABEL[m])
    logscale_axis(ax)
    ax.set_ylabel("rounds violating the\nnearest canonical rule")
    hgrid(ax)
    ax.set_ylim(0, 2.6)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              handlelength=1.6, columnspacing=0.9, fontsize=6.0)
    save(fig, "f6_ruleforce", alias("f6_ruleforce"))


def fig4_strategy_legacy(d, t13):
    """The pre-rename fig4_strategy, drawn only for paper_scaling.

    paper_scaling/main.tex carries a caption describing the pooled provenance
    stack, so that file keeps getting the figure its caption describes.  The
    second manuscript gets f6_ruleforce instead, whose panel (a) is per model.
    """
    fig = plt.figure(figsize=(W2, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)

    ax = fig.add_subplot(gs[0])
    panel_tag(ax, "a")
    x = np.arange(len(SCALES))
    bottom = np.zeros(len(SCALES))
    for p in PROV_ORDER:
        v = t13[p].to_numpy()
        ax.bar(x, v, bottom=bottom, color=PROV_C[p], width=0.78,
               label=PROV_LABEL[p], edgecolor="white", lw=0.4)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([("%g" % s) for s in SCALES], rotation=90)
    ax.set_xlabel(r"payoff scale $\lambda$")
    ax.set_ylabel("agent-games (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=1,
              handlelength=1.2)

    ax = fig.add_subplot(gs[1])
    panel_tag(ax, "b")
    for m in MODEL_ORDER:
        s = d[d.model == m].groupby("scale_nominal")["min_deviations"].mean()
        ax.plot(s.index, s.values, color=MODEL_C[m], marker=MODEL_M[m],
                label=MODEL_LABEL[m])
    logscale_axis(ax)
    ax.set_ylabel("rounds violating the\nnearest canonical rule")
    hgrid(ax)
    ax.set_ylim(0, 2.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              handlelength=1.6, columnspacing=0.9, fontsize=6.0)
    save(fig, "fig4_strategy")


# --------------------------------------------------------------------------
def f7_strategy_mix(d):
    """One panel per canonical strategy: share against the payoff scale.

    The four panels share a y-axis on purpose. The four shares sum to 100
    within every model and scale, so a common axis is what makes them
    comparable, and it keeps the reader from reading a 4-point move in the
    Tit-for-Tat panel as if it were the same size as a 20-point move in AllC.
    """
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.35), sharey=True)
    for ax, lab in zip(axes, STRAT_ORDER):
        for m in MODEL_ORDER:
            s = (d[d.model == m].groupby("scale_nominal")["label"]
                 .apply(lambda x, k=lab: 100 * (x == k).mean()))
            ax.plot(s.index, s.values, color=MODEL_C[m], marker=MODEL_M[m],
                    label=MODEL_LABEL[m])
        logscale_axis(ax)
        # 6.8 rather than 7.5 so the centred title clears the bold panel
        # letter now that both are set in INK.
        ax.set_title(STRAT_TITLE[lab], fontsize=6.8, color=INK,
                     fontweight="bold")
        ax.set_ylim(0, 90)   # AllC reaches 82.0; 80 clipped it
        hgrid(ax)
        shade_subunit(ax, label=False)
    axes[0].set_ylabel("agent-games with the label (\\%)".replace("\\", ""))
    panel_tag(axes[0], "a", dx=-0.28)
    for tag, ax in zip("bcd", axes[1:]):
        panel_tag(ax, tag, dx=-0.10)
    for ax in axes:
        ax.text(0.011, 76, "payoffs $\\leq 1$", ha="left", fontsize=5.6,
                color=MUTED)
    axes[1].legend(loc="upper center", bbox_to_anchor=(1.05, -0.26), ncol=5,
                   handlelength=1.6, columnspacing=1.2)
    save(fig, "f7_strategy_mix", alias("f7_strategy_mix"))


# --------------------------------------------------------------------------
def f8_firstmove(r, t08):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.1),
                             gridspec_kw={"width_ratios": [1.5, 1.0],
                                          "wspace": 0.26})
    ax = axes[0]
    panel_tag(ax, "a")
    for m in MODEL_ORDER:
        d = r[r.model == m]
        s = d[d["round"] == 1].groupby("scale_nominal")["coop"].mean()
        ax.plot(s.index, s.values, color=MODEL_C[m], marker=MODEL_M[m],
                label=MODEL_LABEL[m])
    logscale_axis(ax)
    ax.set_ylabel("cooperation on the opening move")
    ax.set_ylim(0.0, 1.0)   # Grok reaches 0.065; a 0.1 floor hid the point
    hgrid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3,
              handlelength=1.6, columnspacing=1.0)

    ax = axes[1]
    panel_tag(ax, "b")
    x = np.arange(len(MODEL_ORDER))
    w = 0.36
    t = t08.set_index("model").loc[MODEL_ORDER]
    ax.bar(x - w / 2, t.range_round1, w, color=[MODEL_C[m] for m in MODEL_ORDER],
           label="round 1 (no history)")
    ax.bar(x + w / 2, t.range_later, w, color="white",
           edgecolor=[MODEL_C[m] for m in MODEL_ORDER], lw=0.9, hatch="////",
           label="rounds 2-10")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], fontsize=5.8)
    ax.set_ylabel("range of cooperation\nacross the ten scales")
    hgrid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              handlelength=1.4)
    save(fig, "f8_firstmove", alias("f8_firstmove"))


def main():
    use_style()
    g = pd.read_parquet(DATA / "games.parquet")
    r = pd.read_parquet(DATA / "rounds.parquet")
    d = pd.read_parquet(DATA / "readout.parquet")
    t02 = pd.read_csv(TAB / "T02_scale_by_model.csv")
    t03 = pd.read_csv(TAB / "T03_effect_size.csv")
    t04 = pd.read_csv(TAB / "T04_pooling.csv")
    t04c = pd.read_csv(TAB / "T04_shape_correlations.csv")
    t05 = pd.read_csv(TAB / "T05_language.csv")
    t06 = pd.read_csv(TAB / "T06_variance.csv")
    t07 = pd.read_csv(TAB / "T07_persona_gating.csv")
    per = pd.read_csv(TAB / "T07_persona_by_scale.csv")
    t08 = pd.read_csv(TAB / "T08_first_move.csv")
    t09 = pd.read_csv(TAB / "T09_regression.csv")
    t10s = pd.read_csv(TAB / "T10_provenance_by_scale.csv")
    t10t = pd.read_csv(TAB / "T10_provenance_trend.csv")
    t13 = pd.read_csv(TAB / "T13_pooled_strategy.csv")

    f1_scaling(g, t02, t03, t04)
    f2_inference(t03, t04, t04c, t09)
    f3_language(g, t05)
    f4_heterogeneity(t05, t06)
    f5_persona(per, t07)
    f6_ruleforce(d, t10s, t10t)
    f7_strategy_mix(d)
    f8_firstmove(r, t08)
    if IS_LEGACY_FIGDIR:
        # paper_scaling/main.tex still includes fig4_strategy.pdf and its
        # caption still describes the pooled stack, so that exact figure is
        # redrawn here rather than aliased onto the new panel (a).
        fig4_strategy_legacy(d, t13)


if __name__ == "__main__":
    main()
