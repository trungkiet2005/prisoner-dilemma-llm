"""Journal-grade figure system for the frontier-LLM prisoner's-dilemma suite.

Design rules encoded here (Nature / Science / PNAS house style)
--------------------------------------------------------------
* **Canvas.**  Figures are sized in millimetres, not "whatever looked right":
  89 mm single column, 120 mm intermediate, 183 mm double column.  Nothing is
  ever rescaled by the typesetter, so the 7 pt type stays 7 pt on the page.
* **Panel budget.**  At most three panels per figure, one or two by default.
  A figure is an argument, not a dashboard; if a claim needs a fourth panel it
  needs its own figure.
* **Type.**  Arial (Nature's specified face) at 7 pt for labels, 6 pt for tick
  text, 8 pt bold lowercase panel letters.  Nothing below 5 pt, ever.
* **Ink.**  White page, hairline 0.5 pt spines, no top/right spine, no frame
  around legends, no drop shadows, no gradients behind data.
* **Colour.**  The four frontier models use the Okabe-Ito colour-blind-safe
  set, which stays separable under deuteranopia, protanopia and in greyscale
  (the four have L* = 49, 44, 57, 71).  Colour is always doubled by a marker
  shape so the encoding survives a monochrome print.
* **Overlap.**  `finalize()` solves the constrained layout, *freezes* it, and
  only then stamps the panel letters in frozen figure coordinates.  That is
  what makes the letters immune to the reflow that `bbox_inches='tight'`
  would otherwise trigger at save time.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .style import (ANALYSIS, DATADIR, DATASET, MODELDIR,  # noqa: F401
                    TABDIR)  # re-exported so scripts need only one import

FIGDIR = ANALYSIS / "figures" / "frontier"
FIGDIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# canvas widths (mm -> inch)
# --------------------------------------------------------------------------
MM = 1 / 25.4
W1 = 89 * MM      # 3.50 in -- single column
W15 = 120 * MM    # 4.72 in -- intermediate
W2 = 183 * MM     # 7.20 in -- double column

# --------------------------------------------------------------------------
# ink
# --------------------------------------------------------------------------
PAGE = "#ffffff"
INK = "#111111"        # primary text
INK2 = "#3d3d3d"       # axis labels
MUTED = "#8a8a8a"      # ticks, annotations
RULE = "#d9d9d9"       # gridlines
SPINE = "#4d4d4d"      # axis lines
FILL = "#f2f2f2"       # neutral fill / reference band

# --------------------------------------------------------------------------
# the four frontier models (Okabe-Ito)
# --------------------------------------------------------------------------
FRONTIER = ["Claude-3.5-Haiku", "Gemini-3.5-Flash-Lite", "GPT-4o", "Mistral-Large"]

MODEL_C = {
    "Claude-3.5-Haiku": "#d55e00",       # vermillion
    "Gemini-3.5-Flash-Lite": "#0072b2",  # blue
    "GPT-4o": "#009e73",                 # bluish green
    "Mistral-Large": "#e69f00",          # orange
}
MODEL_M = {                              # redundant shape encoding
    "Claude-3.5-Haiku": "o",
    "Gemini-3.5-Flash-Lite": "s",
    "GPT-4o": "^",
    "Mistral-Large": "D",
}
MODEL_LS = {
    "Claude-3.5-Haiku": "-",
    "Gemini-3.5-Flash-Lite": (0, (4, 1.4)),
    "GPT-4o": (0, (1.2, 1.2)),
    "Mistral-Large": (0, (5, 1.3, 1, 1.3)),
}
MODEL_LABEL = {                          # what appears in legends and axes
    "Claude-3.5-Haiku": "Claude 3.5 Haiku",
    "Gemini-3.5-Flash-Lite": "Gemini 3.5 Flash-Lite",
    "GPT-4o": "GPT-4o",
    "Mistral-Large": "Mistral Large",
}
MODEL_SLUG = {
    "Claude-3.5-Haiku": "claude35haiku",
    "Gemini-3.5-Flash-Lite": "gemini35flashlite",
    "GPT-4o": "gpt4o",
    "Mistral-Large": "mistrallarge",
}
# horizon condition each arm was actually run under -- see 20_frontier_build.py
HORIZON = {
    "Claude-3.5-Haiku": "unknown",
    "Gemini-3.5-Flash-Lite": "known",
    "GPT-4o": "unknown",
    "Mistral-Large": "unknown",
}

# --------------------------------------------------------------------------
# semantic colours
# --------------------------------------------------------------------------
C_COOP = "#0072b2"
C_DEFECT = "#d55e00"
ACTION_C = {"C": C_COOP, "D": C_DEFECT}

OUTCOME_ORDER = ["CC", "CD", "DC", "DD"]
OUTCOME_C = {
    "CC": "#0072b2",   # mutual cooperation   R
    "CD": "#8ecae6",   # exploited (sucker)   S
    "DC": "#f0c05a",   # exploiting           T
    "DD": "#d55e00",   # mutual defection     P
}
OUTCOME_LABEL = {
    "CC": "CC  mutual cooperation",
    "CD": "CD  exploited",
    "DC": "DC  exploiting",
    "DD": "DD  mutual defection",
}

# Both agents of every dyad are rows in the corpus, so the CD and DC shares
# are equal by construction.  For composition stacks they are merged into one
# anti-coordination band; the four-way split is kept only where the asymmetry
# between the two roles is the point.
STACK_ORDER = ["CC", "ANTI", "DD"]
STACK_C = {"CC": "#0072b2", "ANTI": "#f0c05a", "DD": "#d55e00"}
STACK_LABEL = {
    "CC": "CC  mutual cooperation",
    "ANTI": "CD / DC  anti-coordination",
    "DD": "DD  mutual defection",
}
STACK_COLS = {"CC": ["cc_rate"], "ANTI": ["cd_rate", "dc_rate"],
              "DD": ["dd_rate"]}

PERSONALITY_C = {"cooperative": "#0072b2", "selfish": "#d55e00"}
PERSONALITY_ORDER = ["cooperative", "selfish"]

DYAD_ORDER = ["CvC", "CvS", "SvC", "SvS"]
DYAD_LABEL = {
    "CvC": "coop. vs coop.",
    "CvS": "coop. vs selfish",
    "SvC": "selfish vs coop.",
    "SvS": "selfish vs selfish",
}

LANG_ORDER = ["en", "fr", "vn", "cn", "ar"]
LANG_LABEL = {"en": "English", "fr": "French", "vn": "Vietnamese",
              "cn": "Chinese", "ar": "Arabic"}
LANG_SHORT = {"en": "EN", "fr": "FR", "vn": "VI", "cn": "ZH", "ar": "AR"}

SCALE_ORDER = [0.1, 1.0, 10.0]

# sequential + diverging ramps (perceptually ordered, print-safe)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "fr_seq", ["#f7fbff", "#dceaf7", "#b8d4ec", "#8dbadf", "#5c9bce",
               "#2f7cb8", "#0072b2", "#005b8f", "#00456c"])
CMAP_DIV = LinearSegmentedColormap.from_list(
    "fr_div", ["#8c3800", "#d55e00", "#f0ad72", "#f7f7f7",
               "#8ec5e0", "#0072b2", "#004a72"])

# canonical memory-one strategies, in the order P(C | R), P(C | S), P(C | T), P(C | P)
MEMORY1 = {
    "AllC": (1, 1, 1, 1),
    "AllD": (0, 0, 0, 0),
    "TFT": (1, 0, 1, 0),
    "WSLS": (1, 0, 0, 1),
    "GRIM": (1, 0, 0, 0),
}

_FONTS = ["Arial", "Helvetica", "Segoe UI", "DejaVu Sans", "sans-serif"]


# --------------------------------------------------------------------------
# rcParams
# --------------------------------------------------------------------------
def use_journal_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": PAGE,
        "savefig.facecolor": PAGE,
        "axes.facecolor": PAGE,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "font.family": "sans-serif",
        "font.sans-serif": _FONTS,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6.2,
        "figure.titlesize": 8,

        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.5,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.titlepad": 3.5,
        "axes.labelpad": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": False,
        "grid.color": RULE,
        "grid.linewidth": 0.4,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,

        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 1.8,
        "ytick.major.pad": 1.8,

        "lines.linewidth": 1.0,
        "lines.markersize": 3.2,
        "lines.markeredgewidth": 0.6,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.4,

        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 1.0,
        "legend.borderpad": 0.0,
        "legend.borderaxespad": 0.2,
        "legend.labelcolor": INK2,

        "pdf.fonttype": 42,      # embed as TrueType -> editable in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "pdf.compression": 9,
    })


# --------------------------------------------------------------------------
# figure construction
# --------------------------------------------------------------------------
def figure(width: float = W2, height: float = 2.4, **kw):
    """A constrained-layout figure at an exact journal column width."""
    fig = plt.figure(figsize=(width, height), layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.05, hspace=0.05)
    return fig


def hgrid(ax, axis: str = "y") -> None:
    """Hairline reference rules behind the data -- the only grid we allow."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=RULE, lw=0.4, zorder=0)
    ax.grid(False, axis="x" if axis == "y" else "y")


def finalize(fig, axes=None, letters=None, *, dx=-0.004, dy=0.010) -> None:
    """Freeze the solved layout, then stamp panel letters that cannot collide.

    Two things make the letters overlap-proof.  First, constrained layout
    re-solves at draw time and `bbox_inches='tight'` triggers one more draw at
    save time, so the layout is solved once and then *frozen* before anything
    is positioned from an axes bounding box.  Second, the anchor is the axes'
    **tight** bounding box -- the one that already contains the tick labels,
    the axis label, the title and any in-axes legend -- so the letter is placed
    above and to the left of whatever that panel actually drew, not above a
    nominal rectangle it may have overflowed.
    """
    fig.canvas.draw()
    fig.set_layout_engine("none")
    if axes is None:
        return
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    letters = letters or [chr(ord("a") + i) for i in range(len(axes))]
    for ax, lab in zip(axes, letters):
        tb = ax.get_tightbbox(rend)
        (x0, _), (_, y1) = inv.transform(((tb.x0, tb.y0), (tb.x1, tb.y1)))
        fig.text(max(0.001, x0 + dx), min(0.999, y1 + dy), lab,
                 fontsize=8, fontweight="bold", color=INK,
                 ha="left", va="bottom")


def save(fig, name: str, *, dpi: int = 600) -> None:
    """Write <name>.pdf (vector, for the typesetter) and <name>.png (600 dpi)."""
    for ext, kw in (("pdf", {"metadata": {"CreationDate": None}}),
                    ("png", {"dpi": dpi})):
        fig.savefig(FIGDIR / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"  [fig] frontier/{name}.pdf + .png")


def caption(fig, text: str, *, size: float = 5.8, pad: float = 0.010,
            below=None) -> None:
    """Qualifier text under the figure.

    Call *after* `finalize`, so it does not perturb the frozen layout.  Pass
    `below=<artist>` (typically the shared legend) to anchor the text under
    that artist instead of under the figure box -- a figure-fraction y of 0 is
    the figure edge, not the bottom of a legend the layout engine placed
    inside it, so without this the two collide.
    """
    y = 0.0
    if below is not None:
        bb = below.get_window_extent(fig.canvas.get_renderer())
        y = float(fig.transFigure.inverted().transform((0, bb.y0))[1])
    # Wrap by hand rather than with Text(wrap=True): matplotlib re-wraps at
    # draw time against the current canvas, and `savefig(bbox_inches='tight')`
    # draws once more at a different size, which moves the block off its
    # anchor.  A fixed line list is stable across both draws.
    width = int(fig.get_figwidth() * 72 / (size * 0.50))
    body = "\n".join(textwrap.wrap(text, width=width)) or text
    fig.text(0.0, y - pad, body, ha="left", va="top", fontsize=size,
             color=MUTED, linespacing=1.45)


# --------------------------------------------------------------------------
# reusable marks
# --------------------------------------------------------------------------
def model_line(ax, x, y, model, *, lo=None, hi=None, band_alpha=0.14, **kw):
    """One model's trace: colour + shape + dash, plus an optional CI ribbon."""
    c = MODEL_C[model]
    if lo is not None:
        ax.fill_between(x, lo, hi, color=c, alpha=band_alpha, lw=0, zorder=2)
    ax.plot(x, y, color=c, ls=MODEL_LS[model], marker=MODEL_M[model],
            mfc=c, mec=PAGE, mew=0.55, label=MODEL_LABEL[model], zorder=3,
            clip_on=False, **kw)


def model_handles(*, lines: bool = True):
    if lines:
        return [plt.Line2D([], [], color=MODEL_C[m], ls=MODEL_LS[m],
                           marker=MODEL_M[m], mfc=MODEL_C[m], mec=PAGE,
                           mew=0.55, markersize=3.2, lw=1.0,
                           label=MODEL_LABEL[m]) for m in FRONTIER]
    return [plt.Line2D([], [], ls="none", marker=MODEL_M[m], mfc=MODEL_C[m],
                       mec=PAGE, mew=0.55, markersize=4.0,
                       label=MODEL_LABEL[m]) for m in FRONTIER]


def model_legend(ax_or_fig, *, ncol: int = 4, loc="upper center",
                 bbox=(0.5, 1.02), lines: bool = True, **kw):
    return ax_or_fig.legend(handles=model_handles(lines=lines), ncol=ncol,
                            loc=loc, bbox_to_anchor=bbox, **kw)


def shared_model_legend(fig, axes, *, ncol: int = 4, lines: bool = True,
                        gap: float = 0.030, **kw):
    """One legend for the whole figure, pinned below the lowest panel.

    Call this *after* `finalize`.  `loc='outside lower center'` looks like the
    obvious tool, but it is resolved by the constrained-layout engine, and
    `finalize` has deliberately switched that engine off -- the legend would
    then be re-anchored at save time and land back on top of the caption.  An
    explicit figure-coordinate anchor, measured from the frozen axes, survives
    the extra draw that `savefig(bbox_inches='tight')` performs.
    """
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y = min(inv.transform((0, ax.get_tightbbox(rend).y0))[1] for ax in axes)
    return fig.legend(handles=model_handles(lines=lines), ncol=ncol,
                      loc="upper center", bbox_to_anchor=(0.5, y - gap),
                      bbox_transform=fig.transFigure, **kw)


def bars(ax, x, h, color, *, width=0.8, **kw):
    """Flat fills separated by a hairline of page white -- no outlines."""
    return ax.bar(x, h, width=width, color=color, edgecolor=PAGE,
                  linewidth=0.5, zorder=3, **kw)


def errorbars(ax, x, lo, hi, *, color=INK, lw=0.7, horizontal=False):
    if horizontal:
        ax.hlines(x, lo, hi, color=color, lw=lw, zorder=4)
    else:
        ax.vlines(x, lo, hi, color=color, lw=lw, zorder=4)


def annotate_heatmap(ax, mat, *, fmt="{:.2f}", size=5.6, thresh=None,
                     cmap=None, vmin=None, vmax=None):
    """Value labels that flip to white only where the fill is genuinely dark."""
    mat = np.asarray(mat, dtype=float)
    if thresh is None and cmap is not None:
        thresh = (vmin + vmax) / 2 if vmin is not None else np.nanmean(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            dark = thresh is not None and v >= thresh
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=size, color=PAGE if dark else INK)


def colorbar(fig, im, ax, *, label="", **kw):
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015, **kw)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.6, width=0.4, color=SPINE, labelsize=5.6,
                      labelcolor=INK2, pad=1.4)
    if label:
        cb.set_label(label, fontsize=6, color=INK2, labelpad=2.5)
    return cb


def scale_axis(ax, scales=SCALE_ORDER, *, label="payoff scale, $\\lambda$"):
    """log10 x-axis over the three realised payoff multipliers."""
    ax.set_xticks(np.log10(scales))
    ax.set_xticklabels([f"$\\times${s:g}" for s in scales])
    ax.set_xlim(np.log10(scales[0]) - 0.35, np.log10(scales[-1]) + 0.35)
    ax.set_xlabel(label)


def refline(ax, y, text=None, *, axis="h", color=MUTED, lw=0.5,
            ls=(0, (2.5, 2)), tx=0.995, ha="right", va="bottom", size=5.6):
    line = ax.axhline if axis == "h" else ax.axvline
    line(y, color=color, lw=lw, ls=ls, zorder=1)
    if text:
        if axis == "h":
            ax.text(tx, y, f" {text}", transform=ax.get_yaxis_transform(),
                    ha=ha, va=va, fontsize=size, color=color)
        else:
            ax.text(y, tx, f" {text}", transform=ax.get_xaxis_transform(),
                    ha="left", va="top", fontsize=size, color=color)
    return line


def tidy_ylim(ax, lo, hi, *, pad=0.02):
    ax.set_ylim(lo - pad * (hi - lo), hi + pad * (hi - lo))
