"""Figure style for the payoff-scaling manuscript.

Self-contained rather than imported from `pdlib.natstyle`, which hard-codes the
three-scale grid and the model roster of the earlier study; carrying those
defaults into a ten-scale, five-model figure was a source of silent mislabelling.

The categorical palette is Okabe-Ito, chosen so the five model series stay
separable under deuteranopia and protanopia, and every series also carries a
distinct marker so the figures survive greyscale printing.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
# Figures are written straight into the manuscript directory so that the
# manuscript directory is self-contained and there is still no manual copy
# step.  The destination is configurable so that the same pipeline can feed a
# second manuscript, but the default is unchanged: with neither variable set
# this resolves to paper_scaling/figures exactly as before.
DEFAULT_PAPER_DIR = HERE.parents[1] / "paper_scaling"
PAPER_DIR = Path(os.environ.get("PD_PAPER_DIR", DEFAULT_PAPER_DIR))
FIGDIR = Path(os.environ.get("PD_FIGDIR", PAPER_DIR / "figures"))

# Deliberately NOT created here.  This module is imported by the table scripts
# too, and creating the directory at import time left an empty `figures/` beside
# every manuscript that only ever received tables.  `save()` creates it.

# True only when the figures are going to their historical home.  The figure
# files were renamed f1_..f8_ for the second manuscript, whose figure directory
# already holds 29 legacy PDFs called fig01_, fig02_ and so on; writing the old
# fig1_..fig6_ names alongside the new ones keeps paper_scaling building from
# an unedited main.tex.  See LEGACY_ALIAS in s05_figures.py.
IS_DEFAULT_FIGDIR = FIGDIR.resolve() == (DEFAULT_PAPER_DIR / "figures").resolve()

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
    "Grok-4.20-Non-Reasoning": "#e69f00",
}
MODEL_M = {
    "Claude-Haiku-4.5": "o",
    "GPT-5.4-Nano": "s",
    "Gemini-3.1-Flash-Lite-Preview": "^",
    "Gemini-3.5-Flash-Lite": "D",
    "Qwen3-235B-A22B": "v",
    "Grok-4.20-Non-Reasoning": "P",
}
MODEL_LABEL = {
    "Claude-Haiku-4.5": "Claude Haiku 4.5",
    "GPT-5.4-Nano": "GPT-5.4 Nano",
    "Gemini-3.1-Flash-Lite-Preview": "Gemini 3.1 Flash-Lite",
    "Gemini-3.5-Flash-Lite": "Gemini 3.5 Flash-Lite",
    "Qwen3-235B-A22B": "Qwen3 235B-A22B",
    "Grok-4.20-Non-Reasoning": "Grok 4.20",
}
MODEL_SHORT = {
    "Claude-Haiku-4.5": "Claude\nHaiku 4.5",
    "GPT-5.4-Nano": "GPT-5.4\nNano",
    "Gemini-3.1-Flash-Lite-Preview": "Gemini\n3.1 FL",
    "Gemini-3.5-Flash-Lite": "Gemini\n3.5 FL",
    "Qwen3-235B-A22B": "Qwen3\n235B",
    "Grok-4.20-Non-Reasoning": "Grok\n4.20",
}
# The corpus holds six models.  The main text reports five of them and the
# electronic supplementary material reports the sixth, Gemini 3.1 Flash-Lite,
# in full.  The reason is that its identifier names a *preview* endpoint, which
# the provider does not pin to a fixed version, so it is not a stable object of
# study in the way the other five are.  Nothing is hidden by the split: every
# figure and table the main text carries for the five is reproduced for the
# sixth in the supplement, and the pooled quantities are given both ways.
MODEL_ORDER = ["Claude-Haiku-4.5", "GPT-5.4-Nano", "Gemini-3.5-Flash-Lite",
               "Qwen3-235B-A22B", "Grok-4.20-Non-Reasoning"]

SUPPLEMENT_ONLY = ["Gemini-3.1-Flash-Lite-Preview"]

# Every model in the corpus, main text first, for supplementary tables that
# report all six together.
MODEL_ORDER_ALL = MODEL_ORDER + SUPPLEMENT_ONLY

LANG_ORDER = ["en", "fr", "vn", "cn", "ar"]
LANG_LABEL = {"en": "English", "fr": "French", "vn": "Vietnamese",
              "cn": "Chinese", "ar": "Arabic"}
LANG_C = {"en": "#111111", "fr": "#0072b2", "vn": "#009e73",
          "cn": "#e69f00", "ar": "#cc79a7"}
# One marker per language as well as one colour: four of the five colours land
# within 50 luminance levels of one another, and the journal prints in black
# and white unless colour figures are paid for.
LANG_M = {"en": "o", "fr": "s", "vn": "^", "cn": "D", "ar": "v"}

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
        # Royal Society production requires embedded, editable fonts: type 42
        # embeds the TrueType outlines in the PDF and the EPS instead of
        # writing type 3 bitmap-ish subsets, and "none" leaves SVG text as
        # text rather than converting it to paths.
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def hgrid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=RULE, linewidth=0.5)


LABELLED = [0.01, 0.1, 1, 10, 1000]


def logscale_axis(ax, label=r"payoff scale $\lambda$", sparse=True):
    """A log x-axis with a tick at each of the ten scales actually run.

    Only the decades carry text: the grid is geometric but unevenly spaced
    (0.25, 0.5, 2, 5 sit close together on a log axis) and labelling all ten
    collides at column width.  Checked again: adding 0.25 overprints the 0.1 label
    and adding 100 overprints the 1000 label in the five-panel figure 2, so this
    list is the widest one that stays legible.  The scales the prose names by
    value are located for the reader by the shaded sub-unit band instead.
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


def save(fig, name, aliases=()):
    """Write one figure as PDF and PNG, plus any legacy filenames.

    `aliases` exists so a renamed figure can keep its previous filename in the
    manuscript directory that still refers to it, which is what stops the
    rename from breaking a build.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for stem in (name, *aliases):
        for ext in ("pdf", "png"):
            fig.savefig(FIGDIR / f"{stem}.{ext}")
    plt.close(fig)
    extra = ("  (also as %s.pdf)" % ", ".join(aliases)) if aliases else ""
    print(f"  wrote {FIGDIR / name}.pdf{extra}")
