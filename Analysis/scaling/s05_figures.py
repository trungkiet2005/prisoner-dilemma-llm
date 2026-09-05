"""Figures for the payoff-scaling manuscript.

Six figures, each carrying one claim:

    fig1  the manipulation is inert in theory, and cooperation moves anyway,
          differently in each model, so pooling cancels it
    fig2  the movement is language-dependent
    fig3  the persona instruction only takes hold above the sub-unit boundary
    fig4  play becomes less rule-governed as the payoffs grow
    fig5  which of the four canonical strategies is played, one panel each
    fig6  the effect is already present at the opening move
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (BAND, INK, LANG_C, LANG_LABEL, LANG_ORDER,  # noqa: E402
                      MODEL_C, MODEL_LABEL, MODEL_M, MODEL_ORDER, MODEL_SHORT, MUTED,
                      PROV_C, PROV_LABEL, PROV_ORDER, RULE, SCALES, STRAT_C,
                      STRAT_ORDER, STRAT_TITLE, W1, W2, hgrid, logscale_axis,
                      panel_tag,
                      save, shade_subunit, use_style)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TAB = HERE / "tables"


# --------------------------------------------------------------------------
def fig1(g, t02, t03, t04):
    fig = plt.figure(figsize=(W2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.35, 1.0], wspace=0.34)

    # (a) the same game, three renderings
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_tag(ax, "a", dx=-0.02, dy=1.02)
    ax.set_title("one game, three renderings", loc="left", pad=10)
    # The agent is shown years of imprisonment and told to minimise them, so a
    # smaller number is a better outcome and OptionA is the dominant action.
    xs = [0.46, 0.76]
    for k, lam in enumerate([0.01, 1, 1000]):
        top = 0.93 - 0.26 * k
        cells = [[0 * lam, 2 * lam], [6 * lam, 10 * lam]]
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
    ax.text(0.0, 0.05, "years of imprisonment, to be minimised.\n"
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
    ax = fig.add_subplot(gs[2])
    panel_tag(ax, "c")
    per_model = g.groupby(["model", "scale_nominal"])["coop_rate"].mean().unstack()
    pooled = per_model.mean(axis=0)
    for m in MODEL_ORDER:
        d = per_model.loc[m]
        ax.plot(d.index, d.values - d.mean(), color=MODEL_C[m], lw=0.8,
                alpha=0.55)
    ax.plot(pooled.index, pooled.values - pooled.mean(), color=INK, lw=1.8,
            marker="o", markersize=3.4, label="average over models", zorder=5)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    logscale_axis(ax)
    ax.set_ylabel("cooperation, centred within model")
    hgrid(ax)
    pr = float(t04.pooled_range.iloc[0])
    mw = float(t04.mean_within_model_range.iloc[0])
    ax.set_title(f"within model {mw:.3f}   pooled {pr:.3f}", loc="left",
                 fontsize=6.5, color=MUTED, pad=3)
    ax.legend(loc="lower right")
    save(fig, "fig1_scaling")


# --------------------------------------------------------------------------
def fig2(g, t05):
    fig, axes = plt.subplots(1, 5, figsize=(W2, 2.0), sharey=True)
    for ax, m in zip(axes, MODEL_ORDER):
        d = g[g.model == m]
        for lang in LANG_ORDER:
            s = (d[d.language == lang].groupby("scale_nominal")["coop_rate"]
                 .mean())
            ax.plot(s.index, s.values, color=LANG_C[lang], lw=0.95,
                    marker="o", markersize=2.2, label=LANG_LABEL[lang])
        logscale_axis(ax)
        ax.set_title(MODEL_LABEL[m], fontsize=6.8)
        ax.set_ylim(0.0, 1.0)
        hgrid(ax)
        rng = t05[t05.model == m].range_over_scales
        ax.text(0.03, 0.03, f"range {rng.min():.2f}-{rng.max():.2f}",
                transform=ax.transAxes, fontsize=5.8, color=MUTED)
    axes[0].set_ylabel("cooperation rate")
    panel_tag(axes[0], "a", dx=-0.30)
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=5,
                   handlelength=1.5, columnspacing=1.2)
    save(fig, "fig2_language")


# --------------------------------------------------------------------------
def fig3(per, t07):
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
    save(fig, "fig3_persona")


# --------------------------------------------------------------------------
def fig4(d, t13, t11):
    fig = plt.figure(figsize=(W2, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)

    # (a) provenance of the read-out
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
    save(fig, "fig4_strategy")


# --------------------------------------------------------------------------
def fig5_mix(d):
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
        ax.set_title(STRAT_TITLE[lab], fontsize=7.5, color=STRAT_C[lab],
                     fontweight="bold")
        ax.set_ylim(0, 80)
        hgrid(ax)
        shade_subunit(ax, label=False)
    axes[0].set_ylabel("agent-games with the label (\\%)".replace("\\", ""))
    panel_tag(axes[0], "a", dx=-0.28)
    for tag, ax in zip("bcd", axes[1:]):
        panel_tag(ax, tag, dx=-0.10)
    axes[0].text(0.0316, 76, "payoffs $\\leq 1$", ha="center", fontsize=5.6,
                 color=MUTED)
    axes[1].legend(loc="upper center", bbox_to_anchor=(1.05, -0.26), ncol=5,
                   handlelength=1.6, columnspacing=1.2)
    save(fig, "fig5_strategy_mix")


# --------------------------------------------------------------------------
def fig6(r, t08):
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
    ax.set_ylim(0.1, 1.0)
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
    ax.set_xticklabels([MODEL_LABEL[m].replace(" ", "\n", 1)
                        for m in MODEL_ORDER], fontsize=5.6)
    ax.set_ylabel("range of cooperation\nacross the ten scales")
    hgrid(ax)
    ax.legend(loc="upper right", handlelength=1.4)
    save(fig, "fig6_firstmove")


def main():
    use_style()
    g = pd.read_parquet(DATA / "games.parquet")
    r = pd.read_parquet(DATA / "rounds.parquet")
    d = pd.read_parquet(DATA / "readout.parquet")
    t02 = pd.read_csv(TAB / "T02_scale_by_model.csv")
    t03 = pd.read_csv(TAB / "T03_effect_size.csv")
    t04 = pd.read_csv(TAB / "T04_pooling.csv")
    t05 = pd.read_csv(TAB / "T05_language.csv")
    t07 = pd.read_csv(TAB / "T07_persona_gating.csv")
    per = pd.read_csv(TAB / "T07_persona_by_scale.csv")
    t08 = pd.read_csv(TAB / "T08_first_move.csv")
    t11 = pd.read_csv(TAB / "T11_rule_distance.csv")
    t13 = pd.read_csv(TAB / "T13_pooled_strategy.csv")

    fig1(g, t02, t03, t04)
    fig2(g, t05)
    fig3(per, t07)
    fig4(d, t13, t11)
    fig5_mix(d)
    fig6(r, t08)


if __name__ == "__main__":
    main()
