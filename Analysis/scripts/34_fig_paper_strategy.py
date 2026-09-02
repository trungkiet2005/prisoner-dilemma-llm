"""Main-text figures for the strategy-first manuscript.

One figure per step of the argument, at most three panels each:

    fig1_setup       the game, the design, and the read-out that names play
    fig2_readout     what the read-out is worth, on synthetic play with a
                     known generating rule
    fig3_labels      what a strategy label is actually made of
    fig4_conditions  the mix moves with manipulations that cannot matter
    fig5_abstention  the anatomy of the play the read-out declines to name
    fig6_hidden      what that play is, and whether it is a strategy at all

Every number is read from `tables/T_S*.csv` and `tables/T_FR*.csv`, so a
main-text panel cannot drift from the supplementary table behind it.  Run
`Analysis/run_frontier.py` and then `scripts/33_strategy_stats.py` first.

Drawn in the `research_egt` house style (`pdlib.egtstyle`): one canvas width,
serif type, a grid behind the data, the panel letter inside its own title, and
legends inside the axes they belong to.  Qualifiers live in the LaTeX caption
and not on the canvas, which is why no panel carries a footnote.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

from pdlib.egtstyle import (FILL, FRONTIER, FS, HORIZON, INK, LANG_ORDER,
                            LANG_SHORT, MODEL_C, MODEL_LABEL, MODEL_M, MUTED,
                            PAGE, SCALE_ORDER, SPINE, TABDIR, bare, figure,
                            fitted_legend, grid, panel_title, save, swatches,
                            use_paper_style)
from pdlib.ingest import payoff_matrix

use_paper_style()

PAPERFIG = Path(__file__).resolve().parents[2] / "paper" / "figures"

# --------------------------------------------------------------------------
# encodings
# --------------------------------------------------------------------------
# Strategy colours are Okabe-Ito, so the four stay separable under the common
# dichromacies and in greyscale; "Ambiguous" is deliberately achromatic because
# it is not a fifth strategy but the absence of a decision between several.
STRAT_ORDER = ["AllC", "TFT", "WSLS", "AllD", "Ambiguous"]
STRAT_C = {"AllC": "#0072b2", "TFT": "#009e73", "WSLS": "#cc79a7",
           "AllD": "#d55e00", "Ambiguous": "#c9c9c9"}

# The four read-out outcomes form a ladder of decreasing warrant, so the ramp
# runs monotonically from dark (a deduction) to pale (no answer at all).
BUCKETS = ["exact", "ambiguous", "confident", "unclassified"]
BUCKET_C = {"exact": "#00456c", "ambiguous": "#4d92c0",
            "confident": "#b8d4ec", "unclassified": FILL}
# Legend glosses.  The full statement of each rung is in the caption; what a
# legend has room for is the term the manuscript uses for it plus a hint.
BUCKET_LEGEND = {
    "exact": "deduced: one rule fits",
    "ambiguous": "ambiguous: several do",
    "confident": "nearest rule (classifier)",
    "unclassified": "abstained (no label)",
}
BUCKET_SHORT = {"exact": "deduced", "ambiguous": "rule set",
                "confident": "nearest rule", "unclassified": "abstained"}

FACTOR_LABEL = {"scale_nominal": "payoff scale $\\lambda$",
                "language": "prompt language",
                "personality": "own persona",
                "dyad": "persona pairing"}

# First word of each model name.  Tick labels are read at a glance and against
# a 1.5 in axes; the full names live in the legends and in the caption.
SHORT = {m: MODEL_LABEL[m].split()[0] for m in FRONTIER}


def T(name: str) -> pd.DataFrame:
    return pd.read_csv(TABDIR / name)


def out(fig, name: str) -> None:
    save(fig, PAPERFIG / name)


def dec(v: float, places: int = 2) -> str:
    """A share as `.30`, the way a probability is set in a table."""
    return f"{v:.{places}f}".lstrip("0") or "0"


def stacked(ax, x, shares, order, colours, *, width=0.8, base=None):
    """One stacked bar per x position, in a fixed category order."""
    bottom = np.zeros(len(x)) if base is None else np.asarray(base, float)
    for cat in order:
        h = np.asarray([s.get(cat, 0.0) for s in shares], dtype=float)
        ax.bar(x, h, bottom=bottom, width=width, color=colours[cat],
               edgecolor=PAGE, linewidth=0.5, zorder=3)
        bottom = bottom + h
    return bottom


# ==========================================================================
# Figure 1 -- the game, the design, and the read-out
# ==========================================================================
def fig_setup():
    m = payoff_matrix("frontier")
    census = T("T_S05_census.csv").set_index("model")

    fig = figure(2.75)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.94, 1.18, 1.06])

    # (a) the stage game -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    bare(ax)
    ax.set_xlim(-1.60, 1.62)
    ax.set_ylim(-1.05, 2.30)

    # The agent is shown penalties and told to minimise them, so cooperation is
    # the action with the lower symmetric number.  Rows and columns carry the
    # option names the agent actually sees, because which of them cooperates is
    # never stated in the prompt and has to be read off the table.
    cell = {("C", "C"): (m["R"], m["R"]), ("C", "D"): (m["S"], m["T"]),
            ("D", "C"): (m["T"], m["S"]), ("D", "D"): (m["P"], m["P"])}
    tint = {("C", "C"): "#dbe9f5", ("C", "D"): "#f2f2f2",
            ("D", "C"): "#f2f2f2", ("D", "D"): "#f7e0d0"}
    for i, own in enumerate(("C", "D")):
        for j, opp in enumerate(("C", "D")):
            x, y = j, 1 - i
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1.0, 1.0,
                                   facecolor=tint[(own, opp)], edgecolor=PAGE,
                                   linewidth=1.4, zorder=1))
            a, b = cell[(own, opp)]
            ax.text(x, y + 0.11, f"{a:g},  {b:g}", ha="center", va="center",
                    fontsize=FS["title"], color=INK, zorder=3)
            ax.text(x, y - 0.24, own + opp, ha="center", va="center",
                    fontsize=FS["tiny"], color=MUTED, zorder=3)
    for j, lab in enumerate(("Option B\ncooperate", "Option A\ndefect")):
        ax.text(j, 1.72, lab, ha="center", va="center", fontsize=FS["tiny"],
                color=INK, linespacing=1.35)
    ax.text(0.5, 2.14, "opponent", ha="center", va="center",
            fontsize=FS["annot"], color=MUTED, style="italic")
    for i, lab in enumerate(("Option B\ncooperate", "Option A\ndefect")):
        ax.text(-0.62, 1 - i, lab, ha="right", va="center",
                fontsize=FS["tiny"], color=INK, linespacing=1.35)
    ax.text(-1.52, 0.5, "focal player", ha="center", va="center",
            fontsize=FS["annot"], color=MUTED, style="italic", rotation=90)
    ax.text(-1.58, -0.78,
            f"penalties, to be minimised: $T$<$R$<$P$<$S$\n"
            f"greed $=${m['greed']:.2f}, fear $=${m['fear']:.2f};  "
            f"shown at $\\lambda=1$",
            ha="left", va="center", fontsize=FS["tiny"], color=MUTED,
            linespacing=1.45)
    panel_title(ax, "A", "the stage game")

    # (b) the crossed design --------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    bare(ax)
    rows = [
        ("model", [SHORT[k] for k in FRONTIER],
         [MODEL_C[k] for k in FRONTIER], [PAGE] * 4, "4", 0.24),
        ("horizon", ["hidden" if HORIZON[k] == "unknown" else "known"
                     for k in FRONTIER],
         [FILL if HORIZON[k] == "unknown" else "#4d4d4d" for k in FRONTIER],
         [INK if HORIZON[k] == "unknown" else PAGE for k in FRONTIER],
         "", 0.15),
        ("payoff scale $\\lambda$", [f"$\\times${s:g}" for s in SCALE_ORDER],
         ["#9fbfd6", "#6b9dbe", "#3c7ea6"], [INK, INK, PAGE], "3", 0.24),
        ("language", [LANG_SHORT[l] for l in LANG_ORDER],
         ["#cfd9c8", "#b8c8ad", "#a1b792", "#8aa678", "#73955d"],
         [INK, INK, INK, INK, PAGE], "5", 0.24),
        ("persona pairing", ["C-C", "C-S", "S-C", "S-S"],
         ["#dcc9b0", "#cdb193", "#be9976", "#af8159"],
         [INK, INK, INK, PAGE], "4", 0.24),
        ("replicate", [""] * 10, ["#e8e8e8", "#dedede"] * 5, [INK] * 10,
         "10", 0.24),
    ]
    ypos = np.cumsum([0.0] + [0.66 if h > 0.2 else 0.38 for *_, h in rows])
    ax.set_xlim(-0.58, 1.10)
    ax.set_ylim(ypos[-1] + 0.30, -0.60)
    for (name, labels, cols, tcols, k, hh), y in zip(rows, ypos):
        w = 1 / len(labels)
        # a cell is 1/len(labels) of the band, so the type has to answer to the
        # longest label in the row rather than to one size for the whole panel
        size = FS["tiny"] - (0.7 if max(len(l) for l in labels) > 5 else 0.0)
        for i, (lab, c, tc) in enumerate(zip(labels, cols, tcols)):
            ax.add_patch(Rectangle((i * w, y - hh), w, 2 * hh, facecolor=c,
                                   edgecolor=PAGE, linewidth=0.9, zorder=2))
            if lab:
                ax.text(i * w + w / 2, y, lab, ha="center", va="center",
                        fontsize=size, color=tc, zorder=3)
        ax.text(-0.025, y, name, ha="right", va="center",
                fontsize=FS["tiny"], color=INK)
        if k:
            ax.text(1.025, y, rf"$\times${k}", ha="left", va="center",
                    fontsize=FS["tiny"], color=MUTED)
    ax.text(0.5, ypos[-1] + 0.14, "10 rounds each  =  48,000 decisions",
            ha="center", va="center", fontsize=FS["tiny"], color=MUTED)
    ax.text(0.5, ypos[-1] + 0.30,
            "2,400 dyads $\\times$ 2 seats  =  4,800 agent-games",
            ha="center", va="center", fontsize=FS["annot"], color=INK)
    panel_title(ax, "B", "the crossed design")

    # (c) the read-out --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    bare(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.0)

    total = census.loc["all models"]
    steps = [
        ("exactly one rule\nreproduces the game", "exact", "a deduction"),
        ("several rules do", "ambiguous", "a set, not a label"),
        ("no rule fits;\nposterior $\\geq$ 0.90", "confident", "nearest rule"),
        ("no rule fits;\nposterior < 0.90", "unclassified", "no answer"),
    ]
    ytop, h, gap, x0, w = 0.98, 0.196, 0.062, 0.00, 0.50
    for i, (q, key, verdict) in enumerate(steps):
        y = ytop - i * (h + gap)
        ax.add_patch(Rectangle((x0, y - h), w, h, facecolor=BUCKET_C[key],
                               edgecolor=PAGE, linewidth=0.9, zorder=2))
        dark = key in ("exact", "ambiguous")
        ax.text(x0 + w / 2, y - h / 2, q, ha="center", va="center",
                fontsize=FS["tiny"], color=PAGE if dark else INK, zorder=3,
                linespacing=1.35)
        ax.text(x0 + w + 0.055, y - h / 2 + 0.030, f"{total[key]:.1%}",
                ha="left", va="center", fontsize=FS["title"], color=INK,
                zorder=3)
        ax.text(x0 + w + 0.055, y - h / 2 - 0.042, verdict, ha="left",
                va="center", fontsize=FS["tiny"], color=MUTED, zorder=3,
                style="italic")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w / 2, y - h),
                                         (x0 + w / 2, y - h - gap),
                                         arrowstyle="-|>", mutation_scale=6,
                                         color=MUTED, lw=0.7, zorder=1))
    panel_title(ax, "C", "the read-out")

    out(fig, "fig1_setup")


# ==========================================================================
# Figure 2 -- what the read-out is worth
# ==========================================================================
def fig_readout():
    ident = T("T_FR29_identifiability.csv")
    rc = T("T_S01_risk_coverage.csv")
    gtft = T("T_FR48_unseen_gtft.csv")

    fig = figure(2.35)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.10, 0.92])

    # (a) how many rounds it takes to identify a rule -------------------------
    ax = fig.add_subplot(gs[0, 0])
    grid(ax)
    ax.plot(ident.rounds, ident.bayes_ceiling, color=MUTED, lw=1.1,
            ls=(0, (3, 1.8)), zorder=3, label="Bayes ceiling")
    ax.plot(ident.rounds, ident.lstm, color="#0072b2", lw=1.4, marker="o",
            ms=3.4, mfc="#0072b2", mec=PAGE, mew=0.6, zorder=4,
            label="classifier")
    ax.set_xticks(range(1, 11, 3))
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(0.42, 1.03)
    ax.set_xlabel("rounds observed")
    ax.set_ylabel("accuracy on synthetic play")
    ax.annotate(f"{ident.lstm.iloc[-1]:.3f}", xy=(10, ident.lstm.iloc[-1]),
                xytext=(-2, -11), textcoords="offset points", ha="right",
                va="top", fontsize=FS["annot"], color="#0072b2")
    fitted_legend(ax, loc="lower right")
    panel_title(ax, "A", "identifiability")

    # (b) risk-coverage -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    grid(ax)
    styles = {"test (0-5% noise)": ("#0072b2", "-"),
              "unseen 10% noise": ("#d55e00", (0, (3.5, 1.6)))}
    for corpus, (c, ls) in styles.items():
        d = rc[rc.corpus == corpus].sort_values("coverage")
        ax.plot(d.coverage, d.error_kept, color=c, ls=ls, lw=1.4, zorder=3,
                label=corpus)
        at = d[np.isclose(d.threshold, 0.90)]
        ax.plot(at.coverage, at.error_kept, ls="none", marker="o", ms=4.6,
                mfc=PAGE, mec=c, mew=1.2, zorder=5)
        ax.plot(d[np.isclose(d.threshold, 0.0)].coverage,
                d[np.isclose(d.threshold, 0.0)].error_kept, ls="none",
                marker="s", ms=3.6, mfc=c, mec=PAGE, mew=0.6, zorder=5)
    ax.annotate("floor = 0.90", xy=(0.909, 0.0261), xytext=(0.775, 0.0455),
                fontsize=FS["annot"], color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlabel("coverage (share of games given a label)")
    ax.set_ylabel("error among labelled games")
    ax.set_xlim(0.72, 1.015)
    ax.set_ylim(0, 0.072)
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=1.4, label=k)
               for k, (c, ls) in styles.items()]
    handles += [plt.Line2D([], [], ls="none", marker="s", ms=3.6, mfc=MUTED,
                           mec=PAGE, label="label every game")]
    fitted_legend(ax, handles=handles, loc="upper left")
    panel_title(ax, "B", "the cost of abstaining")

    # (c) an unseen rule ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    grid(ax, axis="x")
    g = gtft.set_index("strategy").reindex(["AllC", "TFT", "WSLS", "AllD"])
    y = np.arange(len(g))[::-1].astype(float)
    ax.barh(y, g.share, height=0.66, color=[STRAT_C[s] for s in g.index],
            edgecolor=PAGE, linewidth=0.5, zorder=3)
    for yi, v in zip(y, g.share):
        ax.text(v + 0.025, yi, dec(v), ha="left", va="center",
                fontsize=FS["annot"], color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(g.index)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.10)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_ylim(-0.6, len(g) - 0.4)
    ax.set_xlabel("share of games labelled")
    panel_title(ax, "C", "an unseen rule")

    out(fig, "fig2_readout")


# ==========================================================================
# Figure 3 -- what a strategy label is made of
# ==========================================================================
def fig_labels():
    census = T("T_S05_census.csv").set_index("model")
    prov = T("T_S03_label_provenance.csv").set_index("label")
    bymodel = T("T_S04_label_provenance_by_model.csv")

    fig = figure(2.55)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.02, 1.02])

    # (a) how each model's games were reached ---------------------------------
    ax = fig.add_subplot(gs[0, 0])
    grid(ax, axis="x")
    order = FRONTIER + ["all models"]
    y = np.arange(len(order))[::-1].astype(float)
    left = np.zeros(len(order))
    for b in BUCKETS:
        v = census.loc[order, b].to_numpy()
        ax.barh(y, v, left=left, height=0.70, color=BUCKET_C[b],
                edgecolor=PAGE, linewidth=0.6, zorder=3)
        for yi, li, vi in zip(y, left, v):
            if vi > 0.06:
                ax.text(li + vi / 2, yi, f"{vi * 100:.0f}", ha="center",
                        va="center", fontsize=FS["tiny"],
                        color=PAGE if b in ("exact", "ambiguous") else INK)
        left = left + v
    # the pooled row is a summary of the four above it, not a fifth model
    ax.axhline(0.5, color=SPINE, lw=0.6, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT.get(k, k) for k in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("share of agent-games")
    panel_title(ax, "A", "how a label was reached")

    # One key for the whole figure, in a strip the layout engine reserves for
    # it.  Panel A's bars run the full width of their axes, so there is no
    # corner inside it for a four-entry legend to sit in.
    fig.legend(handles=swatches([BUCKET_LEGEND[b] for b in BUCKETS],
                                [BUCKET_C[b] for b in BUCKETS]),
               loc="outside upper left", ncol=4, fontsize=FS["legend"],
               handlelength=1.1, columnspacing=1.2, borderaxespad=0.1)

    # (b) provenance of each strategy name ------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    grid(ax, axis="x")
    labs = ["AllC", "AllD", "TFT", "WSLS"]
    p = prov.loc[labs]
    y = np.arange(len(labs))[::-1].astype(float)
    ex = p.n_exact.to_numpy() / 4800
    ls = p.n_lstm.to_numpy() / 4800
    ax.barh(y, ex, height=0.62, color=[STRAT_C[s] for s in labs],
            edgecolor=PAGE, linewidth=0.5, zorder=3)
    ax.barh(y, ls, left=ex, height=0.62, color=FILL, edgecolor=PAGE,
            linewidth=0.5, zorder=3)
    for yi, e, l in zip(y, ex, ls):
        ax.text(e + l + 0.014, yi, f"{e / (e + l):.0%} deduced", ha="left",
                va="center", fontsize=FS["annot"], color=INK)
    # The key goes in the headroom above the longest bar.  Naming the two
    # halves where they lie would collide: the deduced half of every bar is
    # narrower than the words that describe it.
    fitted_legend(ax, handles=swatches(["a rule reproduces the game",
                                        "nearest neighbour only"],
                                       [MUTED, FILL]),
                  loc="upper left", handlelength=1.1, labelspacing=0.28)
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.52)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticklabels(["0", "10", "20", "30", "40%"])
    ax.set_ylim(-0.6, len(labs) + 0.95)
    ax.set_xlabel("share of agent-games with the label")
    panel_title(ax, "B", "what each name rests on")

    # (c) how far the named play is from the name -----------------------------
    ax = fig.add_subplot(gs[0, 2])
    grid(ax, axis="y")
    x = np.arange(len(labs))
    w = 0.19
    for k, mdl in enumerate(FRONTIER):
        d = bymodel[bymodel.model == mdl].set_index("label").reindex(labs)
        ax.plot(x + (k - 1.5) * w, d.prov_exact, ls="none",
                marker=MODEL_M[mdl], ms=4.2, mfc=MODEL_C[mdl], mec=PAGE,
                mew=0.6, zorder=4, label=SHORT[mdl])
    for xi, lab in zip(x, labs):
        v = prov.loc[lab, "prov_exact"]
        ax.hlines(v, xi - 0.42, xi + 0.42, color=INK, lw=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(labs) - 0.45)
    ax.set_ylim(-0.04, 0.88)
    ax.set_ylabel("share of the label that was deduced")
    fitted_legend(ax, loc="upper right", ncol=2, handletextpad=0.3,
                  columnspacing=0.9, borderaxespad=0.3)
    panel_title(ax, "C", "per model")

    out(fig, "fig3_labels")


# ==========================================================================
# Figure 4 -- the mix moves with the framing
# ==========================================================================
def fig_conditions():
    mix = T("T_S07_mix_by_condition.csv")
    tests = T("T_S08_mix_shift_tests.csv")

    fig = figure(2.70)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.10, 0.84, 1.12])

    def mix_panel(ax, factor, levels, tick, *, group_size=FS["tick"]):
        grid(ax, axis="y")
        x, ticks, labels = [], [], []
        pos = 0.0
        for mdl in FRONTIER:
            d = mix[(mix.model == mdl) & (mix.factor == factor)]
            shares = []
            for lev in levels:
                s = d[d.level.astype(str) == str(lev)]
                shares.append(dict(zip(s.archetype, s.share)))
                x.append(pos)
                labels.append(tick[lev])
                pos += 1.0
            stacked(ax, x[-len(levels):], shares, STRAT_ORDER, STRAT_C,
                    width=0.86)
            ticks.append(np.mean(x[-len(levels):]))
            pos += 0.8
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FS["tiny"], rotation=90)
        ax.tick_params(axis="x", length=0, pad=1.5)
        for xc, mdl in zip(ticks, FRONTIER):
            ax.text(xc, -0.20, SHORT[mdl], ha="center", va="center",
                    fontsize=group_size, color=INK,
                    transform=ax.get_xaxis_transform())
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.7, pos - 1.4)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25", "50", "75", "100%"])

    # (a) payoff scale: a manipulation that cannot change the game ------------
    ax = fig.add_subplot(gs[0, 0])
    mix_panel(ax, "scale_nominal", [0.1, 1.0, 10.0],
              {0.1: "$\\times$0.1", 1.0: "$\\times$1", 10.0: "$\\times$10"})
    ax.set_ylabel("share of agent-games")
    panel_title(ax, "A", "mix by payoff scale")

    # One key for the two composition panels, in a reserved strip.
    fig.legend(handles=swatches(STRAT_ORDER, [STRAT_C[s] for s in STRAT_ORDER]),
               loc="outside upper left", ncol=5, fontsize=FS["legend"],
               handlelength=1.1, columnspacing=1.4, borderaxespad=0.1)

    # (b) own persona ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    mix_panel(ax, "personality", ["cooperative", "selfish"],
              {"cooperative": "coop.", "selfish": "selfish"},
              group_size=FS["tiny"])
    ax.set_yticklabels([])
    panel_title(ax, "B", "by assigned persona")

    # (c) does the mix move at all? ------------------------------------------
    # One row per factor, four markers per row.  The tick beside each marker is
    # that model's own permutation null, so "beyond the null" is read within a
    # row rather than against a single pooled threshold.
    ax = fig.add_subplot(gs[0, 2])
    grid(ax, axis="x")
    facs = ["scale_nominal", "language", "personality", "dyad"]
    off = 0.21
    for i, fac in enumerate(facs):
        for k, mdl in enumerate(FRONTIER):
            r = tests[(tests.model == mdl) & (tests.factor == fac)].iloc[0]
            yi = i + (k - 1.5) * off
            c = MODEL_C[mdl]
            ax.plot([r.null_95, r.max_tv], [yi, yi], color=c, lw=0.9,
                    alpha=0.55, zorder=3)
            ax.plot(r.max_tv, yi, marker=MODEL_M[mdl], ms=4.4, mfc=c,
                    mec=PAGE, mew=0.6, ls="none", zorder=5)
            ax.plot(r.null_95, yi, marker="|", ms=5.0, mec=INK, mew=1.0,
                    ls="none", zorder=4)
    ax.set_yticks(range(len(facs)))
    ax.set_yticklabels([FACTOR_LABEL[f] for f in facs])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(facs) - 0.45, -0.60)
    ax.set_xlim(0, 1.00)
    ax.set_xlabel("largest shift in the strategy mix\n"
                  "(total variation between levels)")
    handles = [plt.Line2D([], [], ls="none", marker=MODEL_M[m], ms=4.4,
                          mfc=MODEL_C[m], mec=PAGE, mew=0.6, label=SHORT[m])
               for m in FRONTIER]
    handles += [plt.Line2D([], [], marker="|", ms=5.0, mec=INK, mew=1.0,
                           ls="none", label="permutation null")]
    fitted_legend(ax, handles=handles, loc="upper right", handletextpad=0.3,
                  labelspacing=0.30, borderaxespad=0.4)
    panel_title(ax, "C", "every factor moves it")

    out(fig, "fig4_conditions")


# ==========================================================================
# Figure 5 -- the anatomy of what the read-out will not name
# ==========================================================================
def fig_abstention():
    pairs = T("T_FR41_abstention_pairs.csv")
    corner = T("T_S17_corner_distance.csv").set_index("bucket")
    motif = T("T_S19_motifs.csv")

    fig = figure(2.35)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 0.86, 1.08])

    # (a) caught between which two rules? -------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    grid(ax, axis="x")
    p = pairs.sort_values("share")
    y = np.arange(len(p)).astype(float)
    for yi, r in zip(y, p.itertuples()):
        # each bar is halved between the two rules the posterior is split over,
        # so the pair is legible without reading the tick label
        a, b = r.pair.split("+")
        ax.barh(yi, r.share / 2, height=0.66, color=STRAT_C[a],
                edgecolor=PAGE, linewidth=0.5, zorder=3)
        ax.barh(yi, r.share / 2, left=r.share / 2, height=0.66,
                color=STRAT_C[b], edgecolor=PAGE, linewidth=0.5, zorder=3)
        ax.text(r.share + 0.010, yi, dec(r.share), ha="left", va="center",
                fontsize=FS["annot"], color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r.replace("+", " vs ") for r in p.pair])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.40)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_ylim(-0.6, len(p) - 0.4)
    ax.set_xlabel("share of abstained games")
    panel_title(ax, "A", "caught between two rules")

    # (b) distance from the vocabulary ---------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    grid(ax, axis="y")
    order = ["exact", "confident", "unclassified"]
    x = np.arange(len(order)).astype(float)
    for xi, b in zip(x, order):
        r = corner.loc[b]
        ax.bar(xi, r["mean"], width=0.62, color=BUCKET_C[b], edgecolor=SPINE,
               linewidth=0.5, zorder=3)
        ax.vlines(xi, r.q25, r.q75, color=INK, lw=1.0, zorder=5)
        ax.plot(xi, r["median"], marker="_", ms=8, mec=INK, mew=1.2, ls="none",
                zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_SHORT[b] for b in order], rotation=30,
                       ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.62, len(order) - 0.38)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("distance to the nearest\ncanonical corner")
    panel_title(ax, "B", "it fills the interior")

    # (c) which action motifs recur beyond chance ----------------------------
    ax = fig.add_subplot(gs[0, 2])
    grid(ax, axis="x")
    want = ["CCCC", "DDDD", "DDCC", "CCDD", "CDCD", "DCDC"]
    d = (motif[(motif.bucket == "unclassified") & motif.motif.isin(want)]
         .set_index("motif").reindex(want))
    y = np.arange(len(want))[::-1].astype(float)
    for yi, name, r in zip(y, want, d.itertuples()):
        alt = name in ("CDCD", "DCDC")
        c = "#c9c9c9" if alt else ("#0072b2" if name.startswith("C")
                                   else "#d55e00")
        ax.barh(yi, r.lift - 1, left=1, height=0.64, color=c, edgecolor=PAGE,
                linewidth=0.5, zorder=3)
        ax.text(max(r.lift, 1.0) + 0.016, yi, f"{r.lift:.2f}", ha="left",
                va="center", fontsize=FS["annot"], color=INK)
    ax.axvline(1.0, color=INK, lw=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(want, family="monospace")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.86, 1.78)
    ax.set_ylim(-0.6, len(want) - 0.4)
    ax.set_xlabel("frequency relative to\na within-game shuffle")
    panel_title(ax, "C", "four-round motifs")

    out(fig, "fig5_abstention")


# ==========================================================================
# Figure 6 -- what the abstained play is, and whether it is a strategy
# ==========================================================================
def fig_hidden():
    bybucket = T("T_S09_library_by_bucket.csv").set_index("bucket")
    rules = T("T_S11_library_rules.csv")
    coh = T("T_S18_within_game_coherence.csv").set_index("bucket")

    fig = figure(2.50)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.90, 1.24, 0.90])

    # (a) does a wider vocabulary name it? -----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    grid(ax, axis="y")
    order = ["confident", "unclassified"]
    x = np.arange(len(order)).astype(float)
    w = 0.32
    for xi, b in zip(x, order):
        r = bybucket.loc[b]
        ax.bar(xi - w / 2, r.extended_exact, width=w * 0.94, color="#0072b2",
               edgecolor=PAGE, linewidth=0.5, zorder=3)
        ax.bar(xi + w / 2, r.extended_null, width=w * 0.94, color="#c9c9c9",
               edgecolor=PAGE, linewidth=0.5, zorder=3)
        ax.text(xi, max(r.extended_exact, r.extended_null) + 0.030,
                f"+{dec(r.excess)}", ha="center", va="bottom",
                fontsize=FS["annot"], color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_SHORT[b] for b in order], rotation=30,
                       ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.62, len(order) - 0.38)
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("share reproduced exactly by\nthe extended vocabulary")
    fitted_legend(ax, handles=swatches(["observed", "shuffled null"],
                                       ["#0072b2", "#c9c9c9"]),
                  loc="upper left", handlelength=1.1)
    panel_title(ax, "A", "a wider vocabulary")

    # (b) which rules land ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    grid(ax, axis="x")
    d = rules.head(8).iloc[::-1]
    y = np.arange(len(d)).astype(float)
    for yi, r in zip(y, d.itertuples()):
        two = "->" in r.best_family
        ax.barh(yi, r.share, height=0.66,
                color="#0072b2" if two else "#8ecae6", edgecolor=PAGE,
                linewidth=0.5, zorder=3)
        ax.text(r.share + 0.010, yi, dec(r.share), ha="left", va="center",
                fontsize=FS["annot"], color=INK)
    pretty = {"SuspiciousTFT": "Suspicious TFT",
              "TwoTitsForTat": "Two tits for tat",
              "ContriteTFT": "Contrite TFT", "SoftMajority": "Soft majority",
              "HardMajority": "Hard majority",
              "AntiAlternator": "Anti-alternator"}
    ax.set_yticks(y)
    ax.set_yticklabels([pretty.get(r, r).replace("->", " $\\rightarrow$ ")
                        for r in d.best_family])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.50)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel("share of abstained games\na rule reproduces exactly")
    fitted_legend(ax, handles=swatches(["two regimes, one switch",
                                        "a single non-canonical rule"],
                                       ["#0072b2", "#8ecae6"]),
                  loc="lower right", handlelength=1.1)
    panel_title(ax, "B", "what names it")

    # (c) is it a strategy? ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    grid(ax, axis="y")
    order = ["exact", "confident", "unclassified"]
    x = np.arange(len(order)).astype(float)
    for xi, b in zip(x, order):
        ax.bar(xi, coh.loc[b, "r_halves"], width=0.62, color=BUCKET_C[b],
               edgecolor=SPINE, linewidth=0.5, zorder=3)
        ax.text(xi, coh.loc[b, "r_halves"] + 0.028,
                dec(coh.loc[b, "r_halves"]), ha="center", va="bottom",
                fontsize=FS["annot"], color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_SHORT[b] for b in order], rotation=30,
                       ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.62, len(order) - 0.38)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("correlation between the two\nhalves of the same game")
    panel_title(ax, "C", "strategy, or drift?")

    out(fig, "fig6_hidden")


def main():
    PAPERFIG.mkdir(parents=True, exist_ok=True)
    fig_setup()
    fig_readout()
    fig_labels()
    fig_conditions()
    fig_abstention()
    fig_hidden()


if __name__ == "__main__":
    main()
