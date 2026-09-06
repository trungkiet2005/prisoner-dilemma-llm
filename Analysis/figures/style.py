"""Publication style for the frontier-agent payoff-scaling manuscript.

Geometry.  Interface Focus sets a single column 483.697 pt wide, which is
6.693 in or 170.1 mm, measured by asking the class rather than by reading a
guide.  The previous figure set was drawn 183 mm wide, so LaTeX scaled every
figure to 93% and every 7 pt label printed at 6.5 pt.  FULL is the true text
width, so a figure drawn here renders 1:1 and its type is the size it says.

Colour.  Okabe-Ito for model identity, which is the palette evolutionary game
theory has settled on and which stays separable under deuteranopia, protanopia
and tritanopia.  Identity is never carried by hue alone: every model also owns
a marker and a drawn glyph, so the figures survive greyscale print.  Sequential
fields use viridis, diverging fields a blue-white-red centred on zero, and the
four canonical strategies own four inks used nowhere else in the paper.

Relief rule, inherited from the group's other manuscripts: hue is decoration,
never the only channel.  Every multi-series panel carries direct labels, a
distinct marker, or a facet title in the series colour.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER_DIR = Path(os.environ.get("PD_PAPER_DIR", REPO / "papers" / "interface-focus"))
FIGDIR = Path(os.environ.get("PD_FIGDIR", PAPER_DIR / "figures"))
DATA = REPO / "Analysis" / "scaling" / "data"
TABLES = REPO / "Analysis" / "scaling" / "tables"

# --- geometry ---------------------------------------------------------------
PT = 1 / 72.27
FULL = 483.69684 * PT          # 6.693 in, the class's own linewidth
HALF = (FULL - 0.16) / 2       # two panels side by side with a 0.16 in gutter
TWOTHIRD = FULL * 2 / 3
DPI = 400
PAD = 0.015

# --- ink --------------------------------------------------------------------
INK = "#101418"
INK_2 = "#454f59"
MUTED = "#8b949d"
HAIRLINE = "#c8ced4"
GRID = "#dfe4e8"
SURFACE = "#ffffff"
BAND = "#eef1f4"               # the sub-unit payoff regime shading

# --- model identity (Okabe-Ito) ---------------------------------------------
# Assigned once, in a fixed order, so a model keeps its hue and its glyph in
# every figure of the paper and of the supplement.
MODEL_C = {
    "Claude-Haiku-4.5":              "#d55e00",   # vermillion
    "GPT-5.4-Nano":                  "#0072b2",   # blue
    "Gemini-3.5-Flash-Lite":         "#cc79a7",   # reddish purple
    "Qwen3-235B-A22B":               "#5d3a9b",   # violet
    "Grok-4.20-Non-Reasoning":       "#e69f00",   # orange
    "Gemini-3.1-Flash-Lite-Preview": "#009e73",   # bluish green (supplement)
}
MODEL_M = {
    "Claude-Haiku-4.5":              "o",
    "GPT-5.4-Nano":                  "s",
    "Gemini-3.5-Flash-Lite":         "D",
    "Qwen3-235B-A22B":               "v",
    "Grok-4.20-Non-Reasoning":       "P",
    "Gemini-3.1-Flash-Lite-Preview": "^",
}
MODEL_LABEL = {
    "Claude-Haiku-4.5":              "Claude Haiku 4.5",
    "GPT-5.4-Nano":                  "GPT-5.4 Nano",
    "Gemini-3.5-Flash-Lite":         "Gemini 3.5 Flash-Lite",
    "Qwen3-235B-A22B":               "Qwen3 235B-A22B",
    "Grok-4.20-Non-Reasoning":       "Grok 4.20",
    "Gemini-3.1-Flash-Lite-Preview": "Gemini 3.1 Flash-Lite",
}
MODEL_SHORT = {
    "Claude-Haiku-4.5":              "Claude",
    "GPT-5.4-Nano":                  "GPT",
    "Gemini-3.5-Flash-Lite":         "Gemini 3.5",
    "Qwen3-235B-A22B":               "Qwen3",
    "Grok-4.20-Non-Reasoning":       "Grok",
    "Gemini-3.1-Flash-Lite-Preview": "Gemini 3.1",
}
# The five the main text reports, in the order they are always listed.
MODEL_ORDER = ["Claude-Haiku-4.5", "GPT-5.4-Nano", "Gemini-3.5-Flash-Lite",
               "Qwen3-235B-A22B", "Grok-4.20-Non-Reasoning"]
MODEL_ORDER_ALL = MODEL_ORDER + ["Gemini-3.1-Flash-Lite-Preview"]

SCALES = [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0]

# The sub-unit regime is the two scales at which every payoff the prompt prints
# is at most 1, since the largest entry of the matrix is 10.  It is NOT
# "lambda < 1": at lambda = 0.25 the prompt already prints 2.5.  This is
# s03_stats.SUBUNIT and figstyle.SUBUNIT_MAX, and the statistics the manuscript
# reports use that split, so the shaded band has to agree with them.
SUBUNIT = [0.01, 0.1]
SUBUNIT_MAX = 0.1
SUBUNIT_EDGE = 0.158        # geometric midpoint of 0.1 and 0.25, for the band

LANG_LABEL = {"en": "English", "fr": "French", "ar": "Arabic",
              "cn": "Chinese", "vn": "Vietnamese"}
LANG_ORDER = ["en", "fr", "cn", "ar", "vn"]

PERSONA_LABEL = {"cooperative": "told cooperative", "selfish": "told selfish"}

# --- the four canonical rules ----------------------------------------------
# One ink each, used for nothing else in the paper, so a reader who has learnt
# the strategy colours in one figure reads every later figure for free.
STRAT_C = {"ALLC": "#1b7837", "TFT": "#3a7ebf", "WSLS": "#e08214", "ALLD": "#b2182b"}
STRAT_LABEL = {"ALLC": "AllC", "TFT": "TFT", "WSLS": "WSLS", "ALLD": "AllD"}
STRAT_ORDER = ["ALLC", "TFT", "WSLS", "ALLD"]

# Memory-one representation, (p0, pCC, pCD, pDC, pDD): the probability of
# cooperating on the first round and then after each of the four predecessors,
# where the first letter is the agent's own last action and the second is the
# opponent's.
STRAT_VEC = {
    "ALLC": (1.0, 1.0, 1.0, 1.0, 1.0),
    "ALLD": (0.0, 0.0, 0.0, 0.0, 0.0),
    "TFT":  (1.0, 1.0, 0.0, 1.0, 0.0),
    "WSLS": (1.0, 1.0, 0.0, 0.0, 1.0),
}
MEM1 = ["p0", "pCC", "pCD", "pDC", "pDD"]
MEM1_LABEL = {
    "p0":  "first move",
    "pCC": "after C,C",
    "pCD": "after C,D",
    "pDC": "after D,C",
    "pDD": "after D,D",
}

# --- rcParams ---------------------------------------------------------------
RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # Without this, matplotlib sets body text in the sans family above but
    # every $\lambda$ in DejaVu, so a figure ships two typefaces and the Greek
    # letters do not match the prose beside them.  "custom" routes mathtext
    # back through font.sans-serif.
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.default": "it",
    "font.size": 7.4, "axes.titlesize": 7.8, "axes.labelsize": 7.4,
    "xtick.labelsize": 6.9, "ytick.labelsize": 6.9, "legend.fontsize": 7.0,
    "axes.edgecolor": HAIRLINE, "axes.linewidth": 0.6, "axes.labelcolor": INK_2,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.size": 1.3, "ytick.minor.size": 1.3,
    "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "legend.frameon": False, "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.32, "legend.columnspacing": 1.4,
    "lines.solid_capstyle": "butt", "lines.linewidth": 1.3,
    "lines.markersize": 3.4,
    "patch.linewidth": 0.6,
    # Type 3 fonts are matplotlib's default and are rejected by the publisher.
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
}
mpl.rcParams.update(RC)

# Annotation sizes actually used, kept here so they stay consistent.
FS_PANEL = 8.4      # panel letter
FS_CLAIM = 7.5      # the claim sentence beside the panel letter
FS_NOTE = 6.3       # in-plot notes, direct labels, counts
FS_TICK = 6.9


def strip(ax, *, left=True, bottom=True, grid_axis="y"):
    """Recessive frame: keep only the spines that carry a scale."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    if not bottom:
        ax.spines["bottom"].set_visible(False)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)


def panel(ax, letter, claim=None, *, pad=6, x=0.0, gap=10.5):
    """Panel letter in bold, then the claim the panel makes, above the axes.

    The claim is the figure's argument and belongs where the eye lands first,
    not at the end of a caption three inches below.

    The gap between the letter and the claim is in points, not in axes
    fractions.  An axes fraction is a different distance on a narrow panel than
    on a wide one, which put the claim on top of the letter every time a figure
    had a narrow panel in it.
    """
    ax.set_title(letter, loc="left", pad=pad, x=x,
                 fontsize=FS_PANEL, color=INK, fontweight="bold")
    if claim:
        ax.annotate(claim, xy=(x, 1.0), xycoords="axes fraction",
                    xytext=(gap, pad - 4), textcoords="offset points",
                    ha="left", va="bottom", fontsize=FS_CLAIM, color=INK_2,
                    annotation_clip=False)


def scale_axis(ax, *, band=True, label=r"payoff scale $\lambda$", ticks=True):
    """The log lambda axis shared by most panels, with the sub-unit band."""
    ax.set_xscale("log")
    if ticks:
        ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000])
        ax.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1000"])
    ax.set_xlim(0.006, 1700)
    if band:
        ax.axvspan(0.006, SUBUNIT_EDGE, color=BAND, lw=0, zorder=0)
    if label:
        ax.set_xlabel(label)


def zero_rule(ax, value=0.0, *, vertical=False, lw=0.9, color=None):
    fn = ax.axvline if vertical else ax.axhline
    fn(value, color=color or MUTED, lw=lw, zorder=1)


def dot(ax, x, y, *, color, marker="o", size=18, filled=True, zorder=4,
        ring=True, lw=None):
    """A marker with a surface-coloured ring, so coincident points stay two."""
    if filled:
        face, edge, elw = color, (SURFACE if ring else color), (0.7 if ring else 0.0)
    else:
        face, edge, elw = SURFACE, color, 1.0
    ax.scatter([x], [y], s=size, marker=marker, zorder=zorder,
               facecolors=face, edgecolors=edge,
               linewidths=elw if lw is None else lw)


def heat_tiles(ax, arr, row_labels, col_labels, *, cmap="viridis", vmin=None,
               vmax=None, fmt="{:.2f}", fs=None, gutter=2.4, textcolor_flip=0.62):
    """imshow with a white gutter grid, so a heatmap reads as tiles.

    Returns the image, so the caller can hang a colourbar on it.
    """
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if fmt:
        import numpy as np
        lo = vmin if vmin is not None else float(np.nanmin(arr))
        hi = vmax if vmax is not None else float(np.nanmax(arr))
        rng = (hi - lo) or 1.0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if v != v:
                    continue
                t = (v - lo) / rng
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=fs or FS_NOTE,
                        color=SURFACE if t > textcolor_flip else INK)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks([x - 0.5 for x in range(len(col_labels) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(row_labels) + 1)], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=gutter)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def caption(fig, text, *, y=-0.02, width_frac=0.98):
    """A bottom caption wrapped to the figure's own width.

    A single long line of `fig.text` does not wrap.  With `bbox_inches="tight"`
    it then pushes the saved bounding box out past the figure, and the file on
    disk is wider than the column even though `fig.get_size_inches()` still
    says otherwise, so LaTeX scales the whole figure down and every label with
    it.  Wrapping here is what keeps the saved width honest.
    """
    import textwrap
    # 0.50 em per character is a good enough average for this sans face at
    # these sizes; the assertion in `save` catches the cases where it is not.
    chars = int(fig.get_size_inches()[0] * width_frac * 72.0 / (FS_NOTE * 0.50))
    wrapped = "\n".join(textwrap.wrap(" ".join(text.split()), chars))
    return fig.text(0.5, y, wrapped, ha="center", va="top",
                    fontsize=FS_NOTE, color=MUTED, linespacing=1.4)


def _tight_size(fig):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    return bb.width + 2 * PAD, bb.height + 2 * PAD


def save(fig, name, *, figdir=None, tight=True, strict=True, fit=True):
    """Save at exactly the column width, so the type is the size it says.

    `bbox_inches="tight"` trims to the ink, so the file on disk is almost never
    the size `figsize` declared: a caption or an outside legend makes it wider,
    and ordinary margins make it narrower.  LaTeX then scales whatever it gets
    to `\\linewidth`, which silently rescales every label in the figure, by 128%
    in one case here and by 74% in another.  Two figures at nominally the same
    font size then print at different sizes.

    So measure the tight box, grow or shrink the canvas by the difference, and
    measure again.  Font sizes are in points and do not move when the canvas
    does, so this changes only how much room the axes get.  Two passes are
    enough to land inside a thousandth of an inch.
    """
    d = Path(figdir) if figdir else FIGDIR
    d.mkdir(parents=True, exist_ok=True)
    declared = tuple(fig.get_size_inches())

    if tight and fit:
        # A wrapped caption re-wraps as the canvas grows, so the fit can
        # oscillate by a line's worth of width, so it cannot be driven to zero.
        # Stop within a sixteenth of an inch, which is under a 1% rescale and is
        # invisible, and keep the best pass rather than the last one.
        best = None
        for _ in range(6):
            real_w, _real_h = _tight_size(fig)
            err = abs(real_w - FULL)
            if best is None or err < best[0]:
                best = (err, tuple(fig.get_size_inches()))
            if err < 6e-2:
                break
            w, h = fig.get_size_inches()
            fig.set_size_inches(w + (FULL - real_w), h)
        if abs(_tight_size(fig)[0] - FULL) > best[0]:
            fig.set_size_inches(*best[1])

    kw = dict(dpi=DPI, facecolor=SURFACE)
    if tight:
        kw.update(bbox_inches="tight", pad_inches=PAD)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{name}.{ext}", **kw)

    real_w, real_h = _tight_size(fig)
    plt.close(fig)

    flag = ""
    if abs(real_w - FULL) > 8e-2:
        flag = (f"  !! saved {real_w:.3f} in against a {FULL:.3f} in column, "
                f"so LaTeX will rescale to {100 * FULL / real_w:.0f}%")
        if strict:
            raise SystemExit(
                f"{name}:{flag}\n"
                "     the fit pass could not reach the column width.  Usually "
                "an unwrapped caption: use S.caption.  Pass fit=False only if "
                "the figure is deliberately narrower than the column.")
    print(f"  wrote {name}.pdf/.png  (declared {declared[0]:.2f} x "
          f"{declared[1]:.2f}, saved {real_w:.2f} x {real_h:.2f} in){flag}")
