r"""House figure style shared with the `research_egt` manuscripts.

Why a second style module
-------------------------
`natstyle` encodes a Nature-style sheet: Arial, hairline spines, no grid, panel
letters stamped outside the axes after the layout is frozen, and a canvas
cropped to the ink by ``bbox_inches='tight'``.  It is a fine sheet, but on a
three-panel figure whose panels each carry a legend and a note it produces the
look this module exists to replace - panels that float in whitespace, letters
detached from the titles they belong to, legends parked above the axes on
hand-tuned anchors, and a different amount of air around every panel because
each one is cropped to its own ink.

This module is the sheet the `research_egt` papers use (see
``research_egt/*/src/*/plotting.py``), transplanted unchanged in its essentials:

* **One canvas width, never cropped.**  Every figure is saved at exactly
  :data:`FIG_W` inches with ``savefig.bbox=None``, so the whole set is reduced
  by one factor on the page and 7 pt of figure type is 7 pt everywhere.  The
  manuscript's ``\textwidth`` is 6.27 in (a4, 1 in margins), so the reduction
  is 0.909 and the sizes in :data:`FS` print at 0.909 of their value.
* **Serif type.**  DejaVu Serif with STIX mathtext, which sits beside the
  body text rather than fighting it.
* **A grid, and ink you can see.**  A #D9D9D9 rule behind the data on both
  axes and black - not grey - axis labels.  The washed-out greys of the
  Nature sheet only work when a panel carries almost nothing.
* **The panel letter lives in the title.**  ``A   what the panel shows``, left
  aligned.  A letter that is part of the title cannot drift away from it, and
  no post-hoc stamping pass is needed.
* **Legends sit inside their axes.**  Frameless, at whichever corner the data
  leaves free.  Nothing is anchored above the axes, so no title needs a 44 pt
  pad to clear it.

Qualifiers belong in the LaTeX caption, not on the canvas; see the captions in
``paper/main.tex``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Data-domain constants (model names, colours, orderings) are not a matter of
# style and are not duplicated here.
from .natstyle import (FRONTIER, HORIZON, LANG_ORDER, LANG_SHORT,  # noqa: F401
                       MEMORY1, MODEL_C, MODEL_LABEL, MODEL_M, SCALE_ORDER,
                       TABDIR)

# --------------------------------------------------------------------------
# canvas
# --------------------------------------------------------------------------
#: Width in inches of every saved figure.  Included at ``\textwidth`` = 6.27 in,
#: so the printed size of any type is 0.909 of the value set here.
FIG_W = 6.9

#: The only type sizes used anywhere, in points *before* the 0.909 page
#: reduction.  ``tiny`` prints at 5.9 pt, which is the floor.
FS = {
    "title": 9.0,    # panel titles, including the bold letter
    "label": 8.5,    # axis labels
    "tick": 7.5,     # tick labels
    "legend": 7.5,   # legend entries
    "annot": 7.0,    # in-axes annotations: bar values, callouts
    "tiny": 6.5,     # dense in-axes annotations: values inside stacked bands
}

# --------------------------------------------------------------------------
# ink
# --------------------------------------------------------------------------
PAGE = "#ffffff"
INK = "#000000"       # text and axis furniture
MUTED = "#555555"     # the few annotations that must recede
RULE = "#d9d9d9"      # gridlines
SPINE = "#333333"     # axis lines
FILL = "#ededed"      # neutral fill: "no answer", reference bands


def use_paper_style() -> None:
    """Apply the manuscript figure style."""
    mpl.rcParams.update({
        "figure.facecolor": PAGE,
        "savefig.facecolor": PAGE,
        "axes.facecolor": PAGE,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        # A fixed saved size is what keeps the on-page scale uniform, so the
        # bounding box must not be cropped to the ink.
        "savefig.bbox": None,

        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        # Matplotlib's default PDF font type is Type 3: bitmap-like glyphs with
        # no usable Unicode map, which fail publisher preflight.  Embed
        # TrueType outlines so labels stay searchable and selectable.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "pdf.compression": 9,

        "font.size": FS["label"],
        "axes.titlesize": FS["title"],
        "axes.labelsize": FS["label"],
        "xtick.labelsize": FS["tick"],
        "ytick.labelsize": FS["tick"],
        "legend.fontsize": FS["legend"],

        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 5.0,
        "axes.labelpad": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "grid.color": RULE,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        "axes.axisbelow": True,

        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,

        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.6,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.5,

        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.1,
        "legend.borderpad": 0.0,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.35,
    })


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def figure(height: float, **kw):
    """A constrained-layout figure at the one canvas width.

    Constrained layout fits the decorations *inside* a fixed canvas, which is
    what lets every figure keep the same saved width and therefore the same
    reduction on the page.
    """
    fig = plt.figure(figsize=(FIG_W, height), layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)
    return fig


def panel_title(ax, letter: str, text: str = "", *, pad: float = 5.0,
                size: float | None = None) -> None:
    """Left-aligned title carrying its own bold panel letter.

    Keeping the letter inside the title is what makes it impossible for the two
    to drift apart, and removes the separate stamping pass that a figure-
    coordinate label needs in order to survive a re-solved layout.
    """
    label = rf"$\bf{{{letter}}}$" + (f"   {text}" if text else "")
    ax.set_title(label, loc="left", fontsize=size or FS["title"], pad=pad)


def grid(ax, axis: str = "both") -> None:
    """Rules behind the data on `axis`, and none on the other one.

    Categorical axes get no grid: a rule through the middle of every bar is
    noise, not a reference.
    """
    ax.set_axisbelow(True)
    ax.grid(False)
    if axis in ("x", "both"):
        ax.grid(True, axis="x", color=RULE, lw=0.5, alpha=0.8, zorder=0)
    if axis in ("y", "both"):
        ax.grid(True, axis="y", color=RULE, lw=0.5, alpha=0.8, zorder=0)


def bare(ax) -> None:
    """A blank canvas for a schematic panel: no spines, no grid, no ticks."""
    ax.set_axis_off()
    ax.grid(False)


def fitted_legend(ax, **kw):
    """Draw a legend inside `ax` and warn if it overhangs the axes box.

    A legend wider than its panel spills onto the neighbouring one, which is
    easy to miss in a three-panel figure and invisible in the vector output
    until it is on the page.
    """
    kw.setdefault("fontsize", FS["legend"])
    legend = ax.legend(**kw)
    fig = ax.get_figure()
    fig.canvas.draw()
    box = legend.get_window_extent()
    frame = ax.get_window_extent()
    overhang = max(frame.x0 - box.x0, box.x1 - frame.x1)
    if overhang > 1.0:
        warnings.warn(
            f"legend overhangs its axes by {overhang:.0f} px; "
            "reduce ncol, handlelength or columnspacing",
            stacklevel=2,
        )
    return legend


def swatches(labels, colours):
    """Legend handles for filled categories, drawn the way the bars are."""
    return [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=PAGE, lw=0.5,
                          label=lab)
            for lab, c in zip(labels, colours)]


def save(fig, path: Path | str, *, also_png: bool = True) -> None:
    """Write ``<path>.pdf`` for the typesetter and ``<path>.png`` to look at."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})
    if also_png:
        fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  [fig] {path.name}.pdf + .png")
