"""Figure style for the payoff-scaling manuscript.

Self-contained rather than imported from `pdlib.natstyle`, which hard-codes the
three-scale grid and the model roster of the earlier study; carrying those
defaults into a ten-scale, five-model figure was a source of silent mislabelling.

The categorical palette is Okabe-Ito, chosen so the five model series stay
separable under deuteranopia and protanopia, and every series also carries a
distinct marker so the figures survive greyscale printing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
# Figures are written straight into the manuscript directory so that
# paper_scaling/ is self-contained and there is still no manual copy step.
FIGDIR = HERE.parents[1] / "paper_scaling" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
W1 = 89 * MM          # single column
W2 = 183 * MM         # double column

INK = "#111111"
MUTED = "#8a8a8a"
RULE = "#dcdcdc"
SPINE = "#4d4d4d"
BAND = "#eef1f4"      # sub-unit regime shading

# Okabe-Ito, assigned in a fixed order so colours never drift between figures
MODEL_C = {
    "Claude-Haiku-4.5": "#d55e00",
    "GPT-5.4-Nano": "#0072b2",
    "Gemini-3.1-Flash-Lite-Preview": "#009e73",
    "Gemini-3.5-Flash-Lite": "#cc79a7",
    "Qwen3-235B-A22B": "#5d3a9b",
}
MODEL_M = {
    "Claude-Haiku-4.5": "o",
    "GPT-5.4-Nano": "s",
    "Gemini-3.1-Flash-Lite-Preview": "^",
    "Gemini-3.5-Flash-Lite": "D",
    "Qwen3-235B-A22B": "v",
}
MODEL_LABEL = {
    "Claude-Haiku-4.5": "Claude Haiku 4.5",
    "GPT-5.4-Nano": "GPT-5.4 Nano",
    "Gemini-3.1-Flash-Lite-Preview": "Gemini 3.1 Flash-Lite",
    "Gemini-3.5-Flash-Lite": "Gemini 3.5 Flash-Lite",
    "Qwen3-235B-A22B": "Qwen3 235B-A22B",
}
MODEL_SHORT = {
    "Claude-Haiku-4.5": "Claude\nHaiku 4.5",
    "GPT-5.4-Nano": "GPT-5.4\nNano",
    "Gemini-3.1-Flash-Lite-Preview": "Gemini\n3.1 FL",
    "Gemini-3.5-Flash-Lite": "Gemini\n3.5 FL",
    "Qwen3-235B-A22B": "Qwen3\n235B",
}
MODEL_ORDER = ["Claude-Haiku-4.5", "GPT-5.4-Nano",
               "Gemini-3.1-Flash-Lite-Preview", "Gemini-3.5-Flash-Lite",
               "Qwen3-235B-A22B"]

LANG_ORDER = ["en", "fr", "vn", "cn", "ar"]
LANG_LABEL = {"en": "English", "fr": "French", "vn": "Vietnamese",
              "cn": "Chinese", "ar": "Arabic"}
LANG_C = {"en": "#111111", "fr": "#0072b2", "vn": "#009e73",
          "cn": "#e69f00", "ar": "#cc79a7"}

STRAT_ORDER = ["AllC", "TFT", "WSLS", "AllD"]
STRAT_C = {"AllC": "#0072b2", "TFT": "#2a7fa8",
           "WSLS": "#a86a00", "AllD": "#d55e00"}
STRAT_TITLE = {
    "AllC": "AllC  (always cooperate)",
    "TFT": "TFT  (tit-for-tat)",
    "WSLS": "WSLS  (win-stay-lose-shift)",
    "AllD": "AllD  (always defect)",
}

PROV_ORDER = ["deduced", "ambiguous", "unmatched"]
PROV_C = {"deduced": "#0072b2", "ambiguous": "#9ecae1", "unmatched": "#e0e0e0"}
PROV_LABEL = {"deduced": "deduced (one rule fits)",
              "ambiguous": "ambiguous (several fit)",
              "unmatched": "no canonical rule fits"}

SCALES = [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 100, 1000]
SUBUNIT_MAX = 0.1     # every payoff printed is at most 1 at and below this


def use_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Segoe UI", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": SPINE, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": SPINE, "ytick.color": SPINE,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.grid": False,
        "lines.linewidth": 1.1, "lines.markersize": 3.0,
    })


def hgrid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=RULE, linewidth=0.5)


LABELLED = [0.01, 0.1, 1, 10, 1000]


def logscale_axis(ax, label=r"payoff scale $\lambda$", sparse=True):
    """A log x-axis with a tick at each of the ten scales actually run.

    Only the decades carry text: the grid is geometric but unevenly spaced
    (0.25, 0.5, 2, 5 sit close together on a log axis) and labelling all ten
    collides at column width.
    """
    ax.set_xscale("log")
    ax.set_xticks(SCALES)
    keep = LABELLED if sparse else SCALES
    ax.set_xticklabels([("%g" % s) if s in keep else "" for s in SCALES])
    ax.set_xlabel(label)
    ax.tick_params(axis="x", which="minor", length=0)


def shade_subunit(ax, label=True):
    """Mark the regime where every payoff printed is at most 1.

    The boundary is a fact about the rendering, not a free parameter: with base
    payoffs 0, 2, 6, 10 the largest number shown is 10*lambda, so it first
    exceeds 1 between lambda = 0.1 and lambda = 0.25.
    """
    lo, hi = ax.get_xlim()
    ax.axvspan(lo, 0.158, color=BAND, zorder=0, lw=0)
    if label:
        ax.text(0.0316, ax.get_ylim()[1], "all payoffs $\\leq 1$",
                ha="center", va="top", fontsize=5.8, color=MUTED)
    ax.set_xlim(lo, hi)


def panel_tag(ax, tag, dx=-0.13, dy=1.06):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {FIGDIR / name}.pdf")
