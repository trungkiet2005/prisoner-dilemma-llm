"""Publication style + validated colour system for the FAIRGAME-PD analysis.

The categorical palette is the dataviz reference instance; every group used
below was checked with the six-check validator on the light chart surface
(#fcfcfb) and passes the lightness band, chroma floor, CVD separation and
normal-vision floors.  Slots below 3:1 contrast (aqua, yellow, magenta) are
always shipped with a legend or a direct label, which is the documented relief.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "Analysis"
DATASET = ROOT / "Dataset"
FIGDIR = ANALYSIS / "figures"
TABDIR = ANALYSIS / "tables"
DATADIR = ANALYSIS / "data"
MODELDIR = ANALYSIS / "models"
for _d in (FIGDIR, TABDIR, DATADIR, MODELDIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# ink / chrome
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --------------------------------------------------------------------------
# semantic palettes (all validator-checked)
# --------------------------------------------------------------------------
# actions -- PASS on all-pairs (CVD dE 21.6, normal dE 32.3)
C_COOP = "#2a78d6"      # blue   : Cooperate  (OptionA)
C_DEFECT = "#e34948"    # red    : Defect     (OptionB)
ACTION = {"C": C_COOP, "D": C_DEFECT}

# joint outcomes, stack order CC -> CD -> DC -> DD -- PASS adjacent
OUTCOME = {
    "CC": "#2a78d6",    # blue   : mutual cooperation  (R)
    "CD": "#1baf7a",    # aqua   : exploited / sucker  (S)
    "DC": "#eda100",    # yellow : exploiting          (T)
    "DD": "#e34948",    # red    : mutual defection    (P)
}
OUTCOME_ORDER = ["CC", "CD", "DC", "DD"]
OUTCOME_LABEL = {
    "CC": "CC  mutual cooperation (R)",
    "CD": "CD  exploited (S)",
    "DC": "DC  exploiting (T)",
    "DD": "DD  mutual defection (P)",
}

# canonical strategies, order AllC -> TFT -> WSLS -> AllD -- PASS adjacent
STRATEGY = {
    "AllC": "#2a78d6",
    "TFT": "#1baf7a",
    "WSLS": "#4a3aa7",
    "AllD": "#e34948",
}
STRATEGY_ORDER = ["AllC", "TFT", "WSLS", "AllD"]

# model families.  Frontier trio validated all-pairs; small trio validated
# all-pairs; the six together validated on the adjacent pairlist.
MODEL = {
    "Claude-3.5-Haiku": "#2a78d6",
    "GPT-4o": "#eb6834",
    "Mistral-Large": "#1baf7a",
    "Gemma-3-12B": "#4a3aa7",
    "Llama-3.1-8B": "#eda100",
    "Qwen3-8B": "#e87ba4",
    # Gemini joined the frontier arm later; the legacy figures below keep the
    # original trio, the frontier-only suite (pdlib.natstyle) plots all four.
    "Gemini-3.5-Flash-Lite": "#4a3aa7",
}
FRONTIER_MODELS = ["Claude-3.5-Haiku", "GPT-4o", "Mistral-Large"]
SMALL_MODELS = ["Gemma-3-12B", "Llama-3.1-8B", "Qwen3-8B"]
MODEL_ORDER = FRONTIER_MODELS + SMALL_MODELS

FAMILY = {"frontier": "#2a78d6", "small": "#eb6834"}

# personality (ordinal: cooperative -> selfish)
PERSONALITY = {"cooperative": "#2a78d6", "selfish": "#e34948"}
PERSONALITY_ORDER = ["cooperative", "selfish"]

LANG_ORDER = ["en", "fr", "vn", "cn", "ar"]
LANG_LABEL = {
    "en": "English",
    "fr": "French",
    "vn": "Vietnamese",
    "cn": "Chinese",
    "ar": "Arabic",
}

# sequential blue ramp (100 -> 700), for magnitude heatmaps
SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CMAP_SEQ = LinearSegmentedColormap.from_list("pd_seq_blue", SEQ_BLUE)
# diverging blue <-> red through a neutral gray midpoint (never a hue at 0)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "pd_div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#8f1f1e"]
)

# --------------------------------------------------------------------------
# rcParams
# --------------------------------------------------------------------------
_FONT_STACK = ["DejaVu Sans", "Segoe UI", "Arial", "sans-serif"]


def use_paper_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.dpi": 110,

        "font.family": "sans-serif",
        "font.sans-serif": _FONT_STACK,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.titlesize": 11,

        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,

        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,

        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "lines.solid_capstyle": "round",

        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.labelcolor": INK2,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def savefig(fig, name: str, *, dpi: int = 400, subdir: str | None = None) -> None:
    """Write <name>.png and <name>.pdf into Analysis/figures[/subdir]."""
    out = FIGDIR if subdir is None else FIGDIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    # Suppressing CreationDate makes the PDF byte-reproducible, so rerunning
    # the pipeline does not produce a diff in every figure.
    for ext, kw in (("png", {"dpi": dpi}),
                    ("pdf", {"metadata": {"CreationDate": None}})):
        fig.savefig(out / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"  [fig] {out.name}/{name}.png  +  .pdf")


def panel_tag(ax, tag: str, *, dx: float = -0.16, dy: float = 1.06) -> None:
    """Bold (a)/(b)/... panel letter in the top-left corner of an axes."""
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=INK, ha="left", va="bottom")


def bar_style(container, color, *, alpha: float = 1.0):
    """2px surface gap between adjacent fills -- the mark spec."""
    for p in container:
        p.set_facecolor(color)
        p.set_edgecolor(SURFACE)
        p.set_linewidth(1.0)
        p.set_alpha(alpha)
    return container


def despine_all(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(False)


def footnote(fig, text: str, y: float = -0.01) -> None:
    fig.text(0.5, y, text, ha="center", va="top", fontsize=7, color=MUTED)
