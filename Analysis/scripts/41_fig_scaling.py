"""Main-text figures for the payoff-scaling manuscript.

    fig1_design     the game, what a rescaling does to it, and the design
    fig2_response   the response curve, and the invariance test it fails
    fig3_notation   magnitude or notation: which one the response tracks
    fig4_where      where in the game the effect lives, and what does not move
    fig5_gating     what the scale gates: the persona, the language, the labels

Every panel reads its numbers from ``tables/T_PS*.csv`` and computes none of
its own, so a figure cannot drift from the table behind it.  Run
``40_scaling_stats.py`` first.

Drawn in the house style of ``pdlib.egtstyle``: one canvas width, serif type,
rules behind the data, the panel letter inside its own title.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pdlib.egtstyle import (FILL, FIG_W, FS, INK, MUTED, PAGE, RULE, SPINE,
                            TABDIR, bare, figure, fitted_legend, grid,
                            panel_title, save, swatches, use_paper_style)

use_paper_style()

PAPERFIG = Path(__file__).resolve().parents[2] / "paper_scaling" / "figures"
PAPERFIG.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# encodings
# --------------------------------------------------------------------------
# Okabe-Ito, so the seven stay separable under the common dichromacies; the
# two arms are additionally separated by line style, which survives greyscale.
MODEL_C = {
    "Claude-3.5-Haiku": "#0072b2",
    "Gemini-3.5-Flash-Lite": "#009e73",
    "GPT-4o": "#d55e00",
    "Mistral-Large": "#cc79a7",
    "Gemma-3-12B": "#e69f00",
    "Llama-3.1-8B": "#56b4e9",
    "Qwen3-8B": "#7a5195",
}
MODEL_M = {
    "Claude-3.5-Haiku": "o", "Gemini-3.5-Flash-Lite": "s",
    "GPT-4o": "^", "Mistral-Large": "D",
    "Gemma-3-12B": "o", "Llama-3.1-8B": "s", "Qwen3-8B": "^",
}
MODEL_LABEL = {
    "Claude-3.5-Haiku": "Claude 3.5 Haiku",
    "Gemini-3.5-Flash-Lite": "Gemini 3.5 Flash-Lite",
    "GPT-4o": "GPT-4o", "Mistral-Large": "Mistral Large",
    "Gemma-3-12B": "Gemma 3 12B", "Llama-3.1-8B": "Llama 3.1 8B",
    "Qwen3-8B": "Qwen3 8B",
}
ARM_ORDER = ["open-weight", "frontier"]
ARM_LABEL = {"open-weight": "open-weight arm (30 rounds)",
             "frontier": "frontier arm (10 rounds)"}

# The three notation regimes are an ordered ladder of how the cell values are
# printed, so the ramp runs monotonically rather than categorically.
REGIME_ORDER = ["fractional", "unit", "large"]
REGIME_C = {"fractional": "#8c3800", "unit": "#0072b2", "large": "#7ba8c9"}
REGIME_LABEL = {"fractional": "fractional\n$\\lambda<1$",
                "unit": "unit\n$\\lambda\\in\\{1,10\\}$",
                "large": "large\n$\\lambda\\geq100$"}

LANG_ORDER = ["en", "fr", "vn", "cn", "ar"]
LANG_LABEL = {"en": "English", "fr": "French", "vn": "Vietnamese",
              "cn": "Chinese", "ar": "Arabic"}
LANG_C = {"en": "#0072b2", "fr": "#009e73", "vn": "#e69f00",
          "cn": "#cc79a7", "ar": "#d55e00"}

STRAT_ORDER = ["AllC", "TFT", "WSLS", "AllD", "Ambiguous"]
STRAT_C = {"AllC": "#0072b2", "TFT": "#009e73", "WSLS": "#cc79a7",
           "AllD": "#d55e00", "Ambiguous": "#c9c9c9"}

SPEC_SHORT = {
    "controls only": "no $\\lambda$ term",
    "linear in log10 lambda": "linear in $\\log_{10}\\lambda$",
    "quadratic in log10 lambda": "quadratic in $\\log_{10}\\lambda$",
    "cubic in log10 lambda": "cubic in $\\log_{10}\\lambda$",
    "fractional flag": "is the cell fractional?",
    "fractional + glyph count": "fractional + glyph count",
    "notation regime (3 levels)": "notation regime",
    "saturated in lambda": "saturated in $\\lambda$",
}
SPEC_ORDER = ["controls only", "linear in log10 lambda",
              "quadratic in log10 lambda", "cubic in log10 lambda",
              "fractional flag", "fractional + glyph count",
              "notation regime (3 levels)", "saturated in lambda"]
# Two families of description: one reads the magnitude, one reads the string.
SPEC_FAMILY = {"controls only": "none",
               "linear in log10 lambda": "magnitude",
               "quadratic in log10 lambda": "magnitude",
               "cubic in log10 lambda": "magnitude",
               "fractional flag": "notation",
               "fractional + glyph count": "notation",
               "notation regime (3 levels)": "notation",
               "saturated in lambda": "saturated"}
FAMILY_C = {"none": "#c9c9c9", "magnitude": "#d55e00",
            "notation": "#0072b2", "saturated": "#555555"}


def T(name: str) -> pd.DataFrame:
    return pd.read_csv(TABDIR / f"T_PS{name}.csv")


def logx(ax, lams) -> None:
    """Log-spaced scale axis with the actual lambdas as ticks.

    The decade minor ticks a log axis draws by default are noise here: the
    sweep visits six points and nothing lies between them.
    """
    ax.set_xscale("log")
    ax.set_xticks(list(lams))
    ax.set_xticklabels([("%g" % v) for v in lams])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("payoff scale $\\lambda$")


# ==========================================================================
# Figure 1 -- the game, the rescaling, and the design
# ==========================================================================
def fig1_design() -> None:
    notation = T("02_notation")
    corpus = T("01_corpus")

    fig = figure(2.75)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.0])
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) the stage game, and the list of what a rescaling leaves alone -------
    bare(ax0)
    panel_title(ax0, "a", "the stage game")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    pay = [["6, 6", "0, 10"], ["10, 0", "2, 2"]]
    x0, y0, w, h = 0.30, 0.66, 0.28, 0.15
    for i in range(2):
        for j in range(2):
            ax0.add_patch(Rectangle((x0 + j * w, y0 - i * h), w, h,
                                    facecolor=PAGE, edgecolor=SPINE, lw=0.7))
            ax0.text(x0 + (j + 0.5) * w, y0 - (i - 0.5) * h, pay[i][j],
                     ha="center", va="center", fontsize=FS["annot"], color=INK)
    for j, lab in enumerate(("A", "B")):
        ax0.text(x0 + (j + 0.5) * w, y0 + h + 0.02, f"opp. {lab}",
                 ha="center", va="bottom", fontsize=FS["tiny"], color=MUTED)
        ax0.text(x0 - 0.02, y0 - (j - 0.5) * h, f"you {lab}",
                 ha="right", va="center", fontsize=FS["tiny"], color=MUTED)
    ax0.text(0.0, y0 + h + 0.115, "penalties, to be minimised",
             ha="left", va="bottom", fontsize=FS["tick"], color=INK)
    ax0.text(0.0, y0 - 2 * h - 0.055,
             "multiply every cell by $\\lambda>0$:", ha="left",
             va="top", fontsize=FS["tick"], color=INK)
    ax0.text(0.0, y0 - 2 * h - 0.135, "nothing below changes",
             ha="left", va="top", fontsize=FS["tick"], color=REGIME_C["unit"])
    invariant = ["the ordering of the four outcomes",
                 "the dominant action and the Nash set",
                 "the greed and fear indices",
                 "best replies and the replicator flow"]
    for k, line in enumerate(invariant):
        ax0.text(0.02, 0.150 - k * 0.055, "•", ha="left", va="center",
                 fontsize=FS["tiny"], color=REGIME_C["unit"])
        ax0.text(0.08, 0.150 - k * 0.055, line, ha="left", va="center",
                 fontsize=FS["tiny"], color=INK)

    # (b) what the agent actually reads at each scale ------------------------
    bare(ax1)
    panel_title(ax1, "b", "what the prompt prints")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ow = notation[notation.arm == "open-weight"].sort_values("lambda")
    ax1.text(0.0, 0.94, "$\\lambda$", fontsize=FS["tiny"], color=MUTED)
    ax1.text(0.17, 0.94, "the four cells, as printed",
             fontsize=FS["tiny"], color=MUTED)
    ax1.text(0.88, 0.94, "regime", fontsize=FS["tiny"], color=MUTED, ha="center")
    for k, (_, r) in enumerate(ow.iterrows()):
        y = 0.80 - k * 0.125
        ax1.text(0.0, y, "%g" % r["lambda"], fontsize=FS["tiny"], color=INK,
                 va="center")
        ax1.text(0.17, y, r.printed_cells, fontsize=FS["tiny"], color=INK,
                 va="center", family="monospace")
        ax1.add_patch(Rectangle((0.76, y - 0.040), 0.24, 0.080,
                                facecolor=REGIME_C[r.regime], alpha=0.20,
                                edgecolor="none"))
        ax1.text(0.88, y, r.regime, fontsize=FS["tiny"], ha="center",
                 va="center", color=REGIME_C[r.regime])
    ax1.text(0.0, 0.025,
             "greed and fear are identical on all six rows",
             fontsize=FS["tiny"], color=MUTED)

    # (c) the design ---------------------------------------------------------
    bare(ax2)
    panel_title(ax2, "c", "the corpus")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    y = 0.90
    for arm in ARM_ORDER:
        d = corpus[corpus.arm == arm]
        ax2.text(0.0, y, ARM_LABEL[arm], fontsize=FS["tick"], color=INK)
        y -= 0.085
        for _, r in d.iterrows():
            ax2.plot([0.05], [y], marker=MODEL_M[r.model], ms=3.4,
                     color=MODEL_C[r.model], mec=PAGE, mew=0.4)
            ax2.text(0.11, y, MODEL_LABEL[r.model], fontsize=FS["tiny"],
                     va="center", color=INK)
            ax2.text(0.72, y, f"{r.scales} scales", fontsize=FS["tiny"],
                     va="center", color=MUTED)
            y -= 0.072
        y -= 0.035
    ax2.text(0.0, y, "5 languages $\\times$ 4 persona pairings $\\times$ 10 replicates",
             fontsize=FS["tiny"], color=MUTED)
    ax2.text(0.0, y - 0.075,
             "6,000 dyads, 12,000 agent-games, 264,000 decisions",
             fontsize=FS["tiny"], color=INK)
    ax2.text(0.0, y - 0.150, "80 agent-games in every model $\\times$ language",
             fontsize=FS["tiny"], color=MUTED)
    ax2.text(0.0, y - 0.215, "$\\times$ scale cell, with none missing",
             fontsize=FS["tiny"], color=MUTED)

    save(fig, PAPERFIG / "fig1_design")


# ==========================================================================
# Figure 2 -- the response curve
# ==========================================================================
def fig2_response() -> None:
    resp = T("03_response")
    sens = T("04_sensitivity")

    fig = figure(2.45)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.85, 1.0])
    axo, axf, axs = (fig.add_subplot(gs[0, i]) for i in range(3))

    for ax, arm in ((axo, "open-weight"), (axf, "frontier")):
        d = resp[(resp.arm == arm) & (resp.model != "pooled")]
        lams = sorted(d["lambda"].unique())
        for mdl, dm in d.groupby("model"):
            dm = dm.sort_values("lambda")
            ax.fill_between(dm["lambda"], dm.lo, dm.hi,
                            color=MODEL_C[mdl], alpha=0.14, lw=0)
            ax.plot(dm["lambda"], dm.coop, color=MODEL_C[mdl],
                    marker=MODEL_M[mdl], ms=3.4, mec=PAGE, mew=0.5, lw=1.3,
                    label=MODEL_LABEL[mdl])
        grid(ax, "y")
        logx(ax, lams)
        ax.set_ylim(0.22, 0.86)
        ax.axvspan(1 / np.sqrt(10), 10 * np.sqrt(10), color=REGIME_C["unit"],
                   alpha=0.05, lw=0, zorder=0)
        fitted_legend(ax, loc="lower right", ncol=1, fontsize=FS["tiny"],
                      handlelength=1.2)
    axo.set_ylabel("cooperation rate")
    panel_title(axo, "a", "open-weight arm, six scales")
    panel_title(axf, "b", "frontier arm")

    # (c) the invariance test: observed swing against its permutation null ---
    order = (sens.sort_values(["arm", "swing"], ascending=[True, True]))
    ypos = np.arange(len(order))
    for y, (_, r) in zip(ypos, order.iterrows()):
        axs.barh(y, r.gap_obs, height=0.62, color=MODEL_C[r.model],
                 edgecolor=PAGE, lw=0.4)
        axs.plot([r.gap_null95, r.gap_null95], [y - 0.36, y + 0.36],
                 color=INK, lw=1.0, solid_capstyle="butt")
    axs.set_yticks(ypos)
    axs.set_yticklabels(
        [f"{MODEL_LABEL[m]}" for m in order.model], fontsize=FS["tiny"])
    axs.set_xlabel("largest shift in cooperation")
    axs.set_xlim(0, 0.43)
    grid(axs, "x")
    panel_title(axs, "c", "the invariance test")
    axs.text(0.425, len(order) - 0.5, "| = null 95th pct.", ha="right",
             va="center", fontsize=FS["tiny"], color=INK)
    for y, (_, r) in zip(ypos, order.iterrows()):
        # 3 dp would render the one borderline value as 0.007 or 0.008
        # depending on the rounding rule, and the manuscript quotes it; 4 dp
        # below 0.01 removes the ambiguity rather than papering over it.
        star = ("n.s." if r.gap_p >= 0.05 else
                "$P<0.001$" if r.gap_p < 0.001 else
                f"$P={r.gap_p:.4f}$" if r.gap_p < 0.01 else
                f"$P={r.gap_p:.3f}$")
        axs.text(r.gap_obs + 0.008, y, star, va="center", ha="left",
                 fontsize=FS["tiny"], color=MUTED)

    save(fig, PAPERFIG / "fig2_response")


# ==========================================================================
# Figure 3 -- magnitude or notation
# ==========================================================================
def fig3_notation() -> None:
    reg = T("05b_regime_means")
    lad = T("05_magnitude_or_notation")
    resp = T("03_response")

    fig = figure(2.5)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.25, 0.95])
    axr, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) the three regimes --------------------------------------------------
    pooled = reg[reg.model == "pooled"]
    xs, labels, offset = [], [], {"open-weight": -0.17, "frontier": 0.17}
    for k, rg in enumerate(REGIME_ORDER):
        labels.append(REGIME_LABEL[rg])
        for arm in ARM_ORDER:
            row = pooled[(pooled.arm == arm) & (pooled.regime == rg)]
            if row.empty:
                continue
            r = row.iloc[0]
            x = k + offset[arm]
            axr.bar(x, r.coop, width=0.30, color=REGIME_C[rg],
                    alpha=1.0 if arm == "open-weight" else 0.45,
                    edgecolor=PAGE, lw=0.5)
            axr.plot([x, x], [r.lo, r.hi], color=INK, lw=0.8)
        xs.append(k)
    axr.set_xticks(xs)
    axr.set_xticklabels(labels, fontsize=FS["tiny"])
    axr.set_ylabel("cooperation rate")
    axr.set_ylim(0, 0.72)
    grid(axr, "y")
    panel_title(axr, "a", "by notation regime")
    axr.legend(handles=[
        Rectangle((0, 0), 1, 1, fc=MUTED, alpha=1.0, label="open-weight"),
        Rectangle((0, 0), 1, 1, fc=MUTED, alpha=0.45, label="frontier")],
        loc="upper left", fontsize=FS["tiny"], handlelength=1.0)

    # (b) the description ladder --------------------------------------------
    ow = (lad[(lad.arm == "open-weight") & (lad.scope == "pooled")]
          .set_index("model_spec").reindex(SPEC_ORDER))
    ypos = np.arange(len(SPEC_ORDER))[::-1]
    for y, spec in zip(ypos, SPEC_ORDER):
        r = ow.loc[spec]
        fam = SPEC_FAMILY[spec]
        axb.barh(y, r.delta_bic, height=0.62, color=FAMILY_C[fam],
                 edgecolor=PAGE, lw=0.4)
        axb.text(min(r.delta_bic + 8, 372), y,
                 f"{r.delta_bic:.0f}" if r.delta_bic > 0 else "best",
                 va="center", ha="left", fontsize=FS["tiny"], color=MUTED)
    axb.set_yticks(ypos)
    axb.set_yticklabels([SPEC_SHORT[s] for s in SPEC_ORDER], fontsize=FS["tiny"])
    axb.set_xlabel("$\\Delta$BIC from the best description")
    axb.set_xlim(0, 415)
    grid(axb, "x")
    panel_title(axb, "b", "what the response tracks")
    axb.legend(handles=swatches(["reads the magnitude", "reads the printed string"],
                                [FAMILY_C["magnitude"], FAMILY_C["notation"]]),
               loc="lower right", fontsize=FS["tiny"], handlelength=1.0)

    # (c) the ranking crossings ---------------------------------------------
    # Cooperation is plotted against an evenly spaced scale index rather than
    # against lambda itself, because what this panel is about is the ordering
    # of the three models and not the shape of any one curve.
    d = resp[(resp.arm == "open-weight") & (resp.model != "pooled")]
    lams = sorted(d["lambda"].unique())
    xi = np.arange(len(lams))
    for mdl, dm in d.groupby("model"):
        dm = dm.sort_values("lambda")
        axc.plot(xi, dm.coop.to_numpy(), color=MODEL_C[mdl],
                 marker=MODEL_M[mdl], ms=3.4, mec=PAGE, mew=0.5, lw=1.3,
                 label=MODEL_LABEL[mdl])
    axc.set_xticks(xi)
    axc.set_xticklabels([("%g" % v) for v in lams], fontsize=FS["tick"])
    axc.set_xlim(-0.4, len(lams) - 0.6)
    axc.set_ylim(0.26, 0.78)
    axc.set_xlabel("payoff scale $\\lambda$")
    axc.set_ylabel("cooperation rate")
    grid(axc, "y")
    panel_title(axc, "c", "the model ranking")
    fitted_legend(axc, loc="upper left", ncol=1, fontsize=FS["tiny"],
                  handlelength=1.2)

    save(fig, PAPERFIG / "fig3_notation")


# ==========================================================================
# Figure 4 -- where the effect lives, and what does not move
# ==========================================================================
def fig4_where() -> None:
    blocks = T("06_round_blocks")
    first = T("06b_opening_move")
    inv = T("09_invariants")

    fig = figure(2.45)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15])
    axb, axo, axi = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) the effect across the game ----------------------------------------
    for arm, ls in (("open-weight", "-"), ("frontier", "--")):
        d = blocks[blocks.arm == arm].sort_values("block_order")
        for rg, dr in d.groupby("regime"):
            n = dr.block_order.max() + 1
            x = dr.block_order / max(n - 1, 1)
            axb.plot(x, dr.coop, ls=ls, color=REGIME_C[rg], lw=1.3,
                     marker="o" if arm == "open-weight" else "s",
                     ms=3.0, mec=PAGE, mew=0.4)
    axb.set_xticks([0, 1 / 3, 2 / 3, 1.0])
    axb.set_xticklabels(["round 1", "early", "middle", "late"],
                        fontsize=FS["tick"])
    axb.set_ylabel("cooperation rate")
    axb.set_ylim(0.26, 0.72)
    grid(axb, "y")
    panel_title(axb, "a", "across the game")
    axb.legend(handles=[Line2D([], [], color=REGIME_C[r], lw=1.3,
                               label=r) for r in REGIME_ORDER]
               + [Line2D([], [], color=MUTED, lw=1.1, ls="--", label="frontier")],
               loc="upper left", fontsize=FS["tiny"], handlelength=1.4, ncol=2)

    # (b) the opening move on its own ---------------------------------------
    for arm, ls in (("open-weight", "-"), ("frontier", "--")):
        d = first[(first.arm == arm) & (first.model != "pooled")]
        for mdl, dm in d.groupby("model"):
            dm = dm.sort_values("lambda")
            axo.plot(dm["lambda"], dm.coop_r1, ls=ls, color=MODEL_C[mdl],
                     marker=MODEL_M[mdl], ms=3.2, mec=PAGE, mew=0.5, lw=1.2)
    grid(axo, "y")
    logx(axo, sorted(first["lambda"].unique()))
    axo.set_ylabel("cooperation on round 1")
    axo.set_ylim(0.0, 1.0)
    panel_title(axo, "b", "before any history")

    # (c) what moves and what does not --------------------------------------
    order = ["miscoordination (CD+DC)", "efficiency", "cooperation rate",
             "mutual cooperation (CC)", "opening cooperation",
             "mutual defection (DD)"]
    ypos = np.arange(len(order))[::-1]
    w = 0.36
    for y, meas in zip(ypos, order):
        for arm, off in (("open-weight", +w / 2), ("frontier", -w / 2)):
            r = inv[(inv.arm == arm) & (inv.measure == meas)].iloc[0]
            axi.barh(y + off, r.gap_obs, height=w * 0.92,
                     color=REGIME_C["unit"] if r.moves else "#c9c9c9",
                     alpha=1.0 if arm == "open-weight" else 0.5,
                     edgecolor=PAGE, lw=0.4)
            axi.plot([r.null95, r.null95], [y + off - w / 2, y + off + w / 2],
                     color=INK, lw=0.9, solid_capstyle="butt")
    axi.set_yticks(ypos)
    axi.set_yticklabels([m.replace(" (", "\n(") for m in order],
                        fontsize=FS["tiny"])
    axi.set_xlabel("largest shift across $\\lambda$")
    axi.set_xlim(0, 0.27)
    grid(axi, "x")
    panel_title(axi, "c", "what does not move")

    save(fig, PAPERFIG / "fig4_where")


# ==========================================================================
# Figure 5 -- what the scale gates
# ==========================================================================
def fig5_gating() -> None:
    per = T("07_persona_by_scale")
    cells = T("08_language_cells")
    curves = T("10e_strategy_curves")

    fig = figure(2.7)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.05])
    axp, axl, axm = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) the persona instruction, scale by scale, one line per model -------
    # Pooling is what must not be drawn here.  The three open-weight models
    # carry persona effects of opposite constant sign, so their average
    # crosses zero although none of them does; the per-model lines are the
    # only honest version of the panel.
    d = per[per.model != "pooled"]
    for arm, ls in (("open-weight", "-"), ("frontier", "--")):
        for mdl, dm in d[d.arm == arm].groupby("model"):
            dm = dm.sort_values("lambda")
            axp.plot(dm["lambda"], dm.effect, ls=ls, color=MODEL_C[mdl],
                     marker=MODEL_M[mdl], ms=3.2, mec=PAGE, mew=0.5, lw=1.2,
                     zorder=3)
    axp.axhline(0, color=INK, lw=0.9, zorder=2)
    grid(axp, "y")
    logx(axp, sorted(d["lambda"].unique()))
    axp.set_ylabel("effect of the cooperative persona")
    axp.set_ylim(-0.70, 0.50)
    panel_title(axp, "a", "the persona instruction")
    # colours are figure 2's, so only the arm needs a key here
    axp.legend(handles=[Line2D([], [], color=MUTED, lw=1.1, ls="--",
                               label="frontier")],
               loc="lower right", fontsize=FS["tiny"], handlelength=1.4)
    axp.text(0.03, 0.955, "instruction followed", transform=axp.transAxes,
             fontsize=FS["tiny"], color=REGIME_C["unit"], va="top")
    axp.text(0.03, 0.035, "instruction inverted", transform=axp.transAxes,
             fontsize=FS["tiny"], color=REGIME_C["fractional"], va="bottom")

    # (b) the language spread across the scale ------------------------------
    d = cells[cells.arm == "open-weight"]
    for lang in LANG_ORDER:
        dl = d[d.language == lang].sort_values("scale_nominal")
        axl.plot(dl.scale_nominal, dl.coop_rate, color=LANG_C[lang],
                 marker="o", ms=3.0, mec=PAGE, mew=0.4, lw=1.2,
                 label=LANG_LABEL[lang])
    grid(axl, "y")
    logx(axl, sorted(d.scale_nominal.unique()))
    axl.set_ylabel("cooperation rate")
    axl.set_ylim(0.15, 0.88)
    panel_title(axl, "b", "the five languages")
    fitted_legend(axl, loc="upper left", ncol=2, fontsize=FS["tiny"],
                  handlelength=1.1, columnspacing=0.7)

    # (c) one curve per strategy label --------------------------------------
    # A stacked bar shows the mix but hides the shape.  What matters is that
    # the two unconditional labels trace the cooperation curve of fig. 2 while
    # the two conditional ones stay nearly flat, and only separate lines say
    # so.  Colour is the label, line style is the arm, as in fig. 4a.
    for arm, ls, mk in (("open-weight", "-", "o"), ("frontier", "--", "s")):
        d = curves[curves.arm == arm]
        for strat in STRAT_ORDER:
            ds = d[d.archetype == strat].sort_values("lambda")
            axm.plot(ds["lambda"], ds.share, ls=ls, color=STRAT_C[strat],
                     marker=mk, ms=2.8, mec=PAGE, mew=0.4, lw=1.2,
                     zorder=4 if strat in ("AllC", "AllD") else 3,
                     alpha=1.0 if arm == "open-weight" else 0.75)
    grid(axm, "y")
    logx(axm, sorted(curves["lambda"].unique()))
    axm.set_ylabel("share of agent-games")
    axm.set_ylim(0, 0.66)
    panel_title(axm, "c", "the strategy label")
    axm.legend(handles=[Line2D([], [], color=STRAT_C[s], lw=1.4, label=s)
                        for s in STRAT_ORDER]
               + [Line2D([], [], color=MUTED, lw=1.1, ls="--", label="frontier")],
               loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3,
               fontsize=FS["tiny"], handlelength=1.3, columnspacing=0.8)

    save(fig, PAPERFIG / "fig5_gating")


def main() -> None:
    fig1_design()
    fig2_response()
    fig3_notation()
    fig4_where()
    fig5_gating()
    print(f"figures written to {PAPERFIG}")


if __name__ == "__main__":
    main()
