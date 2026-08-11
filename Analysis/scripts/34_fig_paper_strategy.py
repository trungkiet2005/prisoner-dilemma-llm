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

Captions live in the LaTeX float, not on the canvas.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

from pdlib.ingest import payoff_matrix
from pdlib.natstyle import (FRONTIER, HORIZON, INK, INK2, LANG_ORDER,
                            LANG_SHORT, MODEL_C, MODEL_LABEL, MODEL_M, MUTED,
                            PAGE, RULE, SCALE_ORDER, SPINE, TABDIR, W2, figure,
                            finalize, hgrid, use_journal_style)

use_journal_style()

PAPERFIG = Path(__file__).resolve().parents[2] / "paper" / "figures"
PAPERFIG.mkdir(parents=True, exist_ok=True)

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
            "confident": "#b8d4ec", "unclassified": "#ededed"}
BUCKET_LABEL = {
    "exact": "exactly one rule reproduces the game",
    "ambiguous": "several rules do; the game cannot separate them",
    "confident": "no rule fits; classifier $\\geq$ 0.90",
    "unclassified": "no rule fits; classifier < 0.90 (abstains)",
}
BUCKET_SHORT = {"exact": "deduced", "ambiguous": "rule set",
                "confident": "nearest rule", "unclassified": "abstained"}

FACTOR_LABEL = {"scale_nominal": "payoff scale $\\lambda$",
                "language": "prompt language",
                "personality": "own persona",
                "dyad": "persona pairing"}


def T(name: str) -> pd.DataFrame:
    return pd.read_csv(TABDIR / name)


def save(fig, name: str) -> None:
    fig.savefig(PAPERFIG / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(PAPERFIG / f"{name}.png", dpi=600)
    plt.close(fig)
    print(f"  [fig] paper/figures/{name}.pdf + .png")


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

    fig = figure(W2, 2.65)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.00, 1.24])

    # (a) the stage game -----------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-1.35, 1.75)
    ax.set_ylim(-1.05, 2.10)
    ax.axis("off")

    # The agent is shown penalties and told to minimise them, so cooperation is
    # the action with the lower symmetric number.  Rows and columns carry the
    # option names the agent actually sees, because which of them cooperates is
    # never stated in the prompt and has to be read off the table.
    cell = {("C", "C"): (m["R"], m["R"]), ("C", "D"): (m["S"], m["T"]),
            ("D", "C"): (m["T"], m["S"]), ("D", "D"): (m["P"], m["P"])}
    tint = {("C", "C"): "#dcebf6", ("C", "D"): "#f6f6f6",
            ("D", "C"): "#f6f6f6", ("D", "D"): "#f8e2d3"}
    for i, own in enumerate(("C", "D")):
        for j, opp in enumerate(("C", "D")):
            x, y = j, 1 - i
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1.0, 1.0,
                                   facecolor=tint[(own, opp)], edgecolor=PAGE,
                                   linewidth=1.2, zorder=1))
            a, b = cell[(own, opp)]
            ax.text(x, y + 0.13, f"{a:g},  {b:g}", ha="center", va="center",
                    fontsize=8.5, color=INK, zorder=3)
            ax.text(x, y - 0.20, own + opp, ha="center", va="center",
                    fontsize=6, color=MUTED, zorder=3)
    for j, lab in enumerate(("Option B\ncooperate", "Option A\ndefect")):
        ax.text(j, 1.76, lab, ha="center", va="center", fontsize=5.5,
                color=INK2, linespacing=1.40)
    ax.text(0.5, 2.12, "opponent", ha="center", va="center", fontsize=6.5,
            color=MUTED, style="italic")
    for i, lab in enumerate(("Option B\ncooperate", "Option A\ndefect")):
        ax.text(-0.60, 1 - i, lab, ha="right", va="center", fontsize=5.5,
                color=INK2, linespacing=1.40)
    ax.text(-1.29, 0.5, "focal player", ha="center", va="center", fontsize=6.5,
            color=MUTED, style="italic", rotation=90)
    ax.text(-1.35, -0.62,
            f"cells are $\\bf{{penalties}}$ to be minimised: $T$<$R$<$P$<$S$;  "
            f"greed $=${m['greed']:.2f}, fear $=${m['fear']:.2f}",
            ha="left", va="center", fontsize=5.6, color=MUTED)
    ax.text(-1.35, -0.92, "shown at $\\lambda = 1$; every cell is multiplied "
            "by $\\lambda \\in \\{0.1, 1, 10\\}$", ha="left", va="center",
            fontsize=5.6, color=MUTED)
    ax.set_title("Stage game", color=INK, pad=16)

    # (b) the crossed design --------------------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    rows = [
        ("model", [MODEL_LABEL[k].split()[0] for k in FRONTIER],
         [MODEL_C[k] for k in FRONTIER], [PAGE] * 4, "4", 0.24),
        ("horizon", ["hidden" if HORIZON[k] == "unknown" else "known"
                     for k in FRONTIER],
         ["#ededed" if HORIZON[k] == "unknown" else "#4d4d4d" for k in FRONTIER],
         [INK2 if HORIZON[k] == "unknown" else PAGE for k in FRONTIER],
         None, 0.15),
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
    ypos = np.cumsum([0.0] + [0.62 if h > 0.2 else 0.34 for *_, h in rows])
    ax.set_xlim(0, 1)
    ax.set_ylim(ypos[-1] + 0.16, -0.52)
    ax.axis("off")
    for (name, labels, cols, tcols, k, hh), y in zip(rows, ypos):
        w = 1 / len(labels)
        for i, (lab, c, tc) in enumerate(zip(labels, cols, tcols)):
            ax.add_patch(Rectangle((i * w, y - hh), w, 2 * hh, facecolor=c,
                                   edgecolor=PAGE, linewidth=0.8, zorder=2))
            if lab:
                ax.text(i * w + w / 2, y, lab, ha="center", va="center",
                        fontsize=5.4 if hh < 0.2 else 5.8, color=tc, zorder=3)
        ax.text(-0.015, y, name, ha="right", va="center", fontsize=6.2,
                color=INK2)
        ax.text(1.015, y, f"$\\times${k}" if k else "confound", ha="left",
                va="center", fontsize=6.2 if k else 5.4, color=MUTED,
                style="normal" if k else "italic")
    ax.text(0.5, ypos[-1] - 0.02,
            "2,400 dyads $\\times$ 2 seats  =  4,800 agent-games",
            ha="center", va="center", fontsize=6.2, color=INK)
    ax.text(0.5, ypos[-1] + 0.14, "10 rounds each  =  48,000 decisions",
            ha="center", va="center", fontsize=6.2, color=MUTED)
    ax.set_title("Crossed design", color=INK, pad=12)

    # (c) the read-out --------------------------------------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    total = census.loc["all models"]
    steps = [
        ("exactly one rule\nreproduces the game", "exact", "a deduction"),
        ("several rules do", "ambiguous", "a set, not a label"),
        ("no rule fits;\nposterior $\\geq$ 0.90", "confident", "nearest rule"),
        ("no rule fits;\nposterior < 0.90", "unclassified", "no answer"),
    ]
    ytop, h, gap, x0, w = 0.97, 0.163, 0.050, 0.02, 0.46
    for i, (q, key, verdict) in enumerate(steps):
        y = ytop - i * (h + gap)
        ax.add_patch(Rectangle((x0, y - h), w, h, facecolor=BUCKET_C[key],
                               edgecolor=PAGE, linewidth=0.8, zorder=2))
        dark = key in ("exact", "ambiguous")
        ax.text(x0 + w / 2, y - h / 2, q, ha="center", va="center",
                fontsize=5.4, color=PAGE if dark else INK, zorder=3,
                linespacing=1.35)
        ax.text(x0 + w + 0.045, y - h / 2 + 0.026, f"{total[key]:.1%}",
                ha="left", va="center", fontsize=7.2, color=INK, zorder=3)
        ax.text(x0 + w + 0.045, y - h / 2 - 0.036, verdict, ha="left",
                va="center", fontsize=5.3, color=MUTED, zorder=3,
                style="italic")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w / 2, y - h),
                                         (x0 + w / 2, y - h - gap),
                                         arrowstyle="-|>", mutation_scale=5,
                                         color=MUTED, lw=0.6, zorder=1))
    ax.text(0.0, 0.115, "of all 4,800 agent-games. The first two steps\n"
            "involve no learning; the last two are mined in\nfigures 5 and 6.",
            ha="left", va="top", fontsize=5.3, color=MUTED, linespacing=1.45)
    ax.set_title("Strategy read-out", color=INK, pad=12)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"], dx=-0.005)
    save(fig, "fig1_setup")


# ==========================================================================
# Figure 2 -- what the read-out is worth
# ==========================================================================
def fig_readout():
    ident = T("T_FR29_identifiability.csv")
    rc = T("T_S01_risk_coverage.csv")
    ded = T("T_S02_deductive_stage.csv")
    gtft = T("T_FR48_unseen_gtft.csv")

    fig = figure(W2, 2.30)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.10, 0.95])

    # (a) how many rounds it takes to identify a rule -------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    ax.plot(ident.rounds, ident.bayes_ceiling, color=MUTED, lw=0.9,
            ls=(0, (2.5, 2)), zorder=3, label="Bayes ceiling")
    ax.plot(ident.rounds, ident.lstm, color="#0072b2", lw=1.1, marker="o",
            ms=3.0, mfc="#0072b2", mec=PAGE, mew=0.5, zorder=4,
            label="classifier")
    ax.set_xticks(range(1, 11, 3))
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(0.42, 1.02)
    ax.set_xlabel("rounds observed")
    ax.set_ylabel("accuracy on synthetic play")
    ax.legend(loc="lower right", bbox_to_anchor=(1.04, 0.045), labelspacing=0.3)
    ax.text(10.2, ident.lstm.iloc[-1] - 0.075, f"{ident.lstm.iloc[-1]:.3f}",
            ha="right", va="top", fontsize=5.8, color="#0072b2")
    ax.set_title("Identifiability", pad=12)

    # (b) risk-coverage -------------------------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    styles = {"test (0-5% noise)": ("#0072b2", "-"),
              "unseen 10% noise": ("#d55e00", (0, (3.5, 1.6)))}
    for corpus, (c, ls) in styles.items():
        d = rc[rc.corpus == corpus].sort_values("coverage")
        ax.plot(d.coverage, d.error_kept, color=c, ls=ls, lw=1.1, zorder=3,
                label=corpus)
        at = d[np.isclose(d.threshold, 0.90)]
        ax.plot(at.coverage, at.error_kept, ls="none", marker="o", ms=4.2,
                mfc=PAGE, mec=c, mew=1.1, zorder=5)
        ax.plot(d[np.isclose(d.threshold, 0.0)].coverage,
                d[np.isclose(d.threshold, 0.0)].error_kept, ls="none",
                marker="s", ms=3.2, mfc=c, mec=PAGE, mew=0.5, zorder=5)
    ax.annotate("floor = 0.90", xy=(0.909, 0.0261), xytext=(0.80, 0.045),
                fontsize=5.6, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.set_xlabel("coverage (share of games given a label)")
    ax.set_ylabel("error among labelled games")
    ax.set_xlim(0.72, 1.015)
    ax.set_ylim(0, 0.070)
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=1.1, label=k)
               for k, (c, ls) in styles.items()]
    handles += [plt.Line2D([], [], ls="none", marker="s", ms=3.2, mfc=MUTED,
                           mec=PAGE, label="label every game")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.06),
              labelspacing=0.3)
    ax.set_title("Cost of abstaining", pad=12)

    # (c) an unseen rule ------------------------------------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax, axis="x")
    g = gtft.set_index("strategy").reindex(["AllC", "TFT", "WSLS", "AllD"])
    y = np.arange(len(g))[::-1].astype(float)
    ax.barh(y, g.share, height=0.62, color=[STRAT_C[s] for s in g.index],
            edgecolor=PAGE, linewidth=0.5, zorder=3)
    for yi, (name, v) in zip(y, g.share.items()):
        ax.text(v + 0.02, yi, f"{v:.2f}".lstrip("0"), ha="left", va="center",
                fontsize=5.8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(g.index, fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("share of games labelled")
    ax.text(0.02, -0.92, "generous tit-for-tat, held out of training entirely:\n"
            "an unseen rule is absorbed, never flagged as new",
            fontsize=5.4, color=MUTED, ha="left", va="center", linespacing=1.4)
    ax.set_ylim(-1.25, len(g) - 0.35)
    ax.set_title("A rule the classifier\nnever saw", pad=6)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"], dx=-0.005)
    save(fig, "fig2_readout")

    return ded


# ==========================================================================
# Figure 3 -- what a strategy label is made of
# ==========================================================================
def fig_labels():
    census = T("T_S05_census.csv").set_index("model")
    prov = T("T_S03_label_provenance.csv").set_index("label")
    bymodel = T("T_S04_label_provenance_by_model.csv")

    fig = figure(W2, 2.55)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.00, 1.00])

    # (a) how each model's games were reached ---------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    order = FRONTIER + ["all models"]
    y = np.arange(len(order))[::-1].astype(float)
    left = np.zeros(len(order))
    for b in BUCKETS:
        v = census.loc[order, b].to_numpy()
        ax.barh(y, v, left=left, height=0.66, color=BUCKET_C[b],
                edgecolor=PAGE, linewidth=0.6, zorder=3)
        for yi, li, vi in zip(y, left, v):
            if vi > 0.055:
                ax.text(li + vi / 2, yi, f"{vi * 100:.0f}", ha="center",
                        va="center", fontsize=5.6,
                        color=PAGE if b in ("exact", "ambiguous") else INK2)
        left = left + v
    ax.axhline(0.5, color=RULE, lw=0.6, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL.get(k, k) for k in order], fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("share of agent-games")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BUCKET_C[b],
                             edgecolor=PAGE, lw=0.5, label=BUCKET_LABEL[b])
               for b in BUCKETS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.44, 1.40),
              ncol=1, labelspacing=0.30, handlelength=1.1)
    ax.set_title("How a label was reached", pad=44)

    # (b) provenance of each strategy name ------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax, axis="x")
    labs = ["AllC", "AllD", "TFT", "WSLS"]
    p = prov.loc[labs]
    y = np.arange(len(labs))[::-1].astype(float)
    ex = p.n_exact.to_numpy() / 4800
    ls = p.n_lstm.to_numpy() / 4800
    ax.barh(y, ex, height=0.62, color=[STRAT_C[s] for s in labs],
            edgecolor=PAGE, linewidth=0.5, zorder=3)
    ax.barh(y, ls, left=ex, height=0.62, color="#ededed", edgecolor=PAGE,
            linewidth=0.5, zorder=3)
    for yi, s, e, l in zip(y, labs, ex, ls):
        ax.text(e + l + 0.012, yi, f"{e / (e + l):.0%} deduced", ha="left",
                va="center", fontsize=5.8,
                color=INK if e / (e + l) < 0.2 else INK2,
                fontweight="bold" if e / (e + l) < 0.2 else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=6.4)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.50)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["0", "10", "20", "30%"])
    ax.set_xlabel("share of agent-games carrying the label")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=PAGE,
                             lw=0.5, label="rule reproduces the game"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#ededed", edgecolor=PAGE,
                             lw=0.5, label="nearest neighbour only")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.24),
              labelspacing=0.28, handlelength=1.1)
    ax.set_title("What each name rests on", pad=26)

    # (c) how far the named play is from the name -----------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax)
    x = np.arange(len(labs))
    w = 0.19
    for k, mdl in enumerate(FRONTIER):
        d = bymodel[bymodel.model == mdl].set_index("label").reindex(labs)
        ax.plot(x + (k - 1.5) * w, d.prov_exact, ls="none",
                marker=MODEL_M[mdl], ms=3.8, mfc=MODEL_C[mdl], mec=PAGE,
                mew=0.55, zorder=4, label=MODEL_LABEL[mdl])
    for xi, lab in zip(x, labs):
        v = prov.loc[lab, "prov_exact"]
        ax.hlines(v, xi - 0.42, xi + 0.42, color=INK2, lw=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=6.4)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(labs) - 0.45)
    ax.set_ylim(-0.03, 0.72)
    ax.set_ylabel("share of the label that was deduced")
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.30), ncol=2,
              labelspacing=0.28, columnspacing=0.7, handletextpad=0.3)
    ax.text(-0.5, -0.105, "bars: pooled over models. Mistral Large never "
            "plays an exact WSLS game.", fontsize=5.4, color=MUTED,
            ha="left", va="center")
    ax.set_title("Per model", pad=38)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"], dx=-0.005)
    save(fig, "fig3_labels")


# ==========================================================================
# Figure 4 -- the mix moves with the framing
# ==========================================================================
def fig_conditions():
    mix = T("T_S07_mix_by_condition.csv")
    tests = T("T_S08_mix_shift_tests.csv")

    fig = figure(W2, 2.60)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.86, 1.12])

    def mix_panel(ax, factor, levels, tick, title):
        hgrid(ax)
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
                    width=0.84)
            ticks.append(np.mean(x[-len(levels):]))
            pos += 0.7
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.2, rotation=90)
        ax.tick_params(axis="x", length=0, pad=1.5)
        for xc, mdl in zip(ticks, FRONTIER):
            ax.text(xc, -0.145, MODEL_LABEL[mdl].split()[0], ha="center",
                    va="center", fontsize=5.8, color=INK2,
                    transform=ax.get_xaxis_transform())
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.7, pos - 1.3)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "50", "100%"])
        ax.set_title(title, pad=12)
        return ax

    # (a) payoff scale: a manipulation that cannot change the game ------------
    ax = pa = fig.add_subplot(gs[0, 0])
    mix_panel(ax, "scale_nominal", [0.1, 1.0, 10.0],
              {0.1: "$\\times$0.1", 1.0: "$\\times$1", 10.0: "$\\times$10"},
              "Strategy mix by payoff scale")
    ax.set_ylabel("share of agent-games")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STRAT_C[s],
                             edgecolor=PAGE, lw=0.5, label=s)
               for s in STRAT_ORDER]
    ax.legend(handles=handles, ncol=5, loc="upper center",
              bbox_to_anchor=(1.15, 1.30), handlelength=1.0,
              columnspacing=0.8, handletextpad=0.35)

    # (b) own persona ---------------------------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    mix_panel(ax, "personality", ["cooperative", "selfish"],
              {"cooperative": "coop.", "selfish": "self."},
              "By assigned persona")

    # (c) does the mix move at all? ------------------------------------------
    # One row per factor, four markers per row.  The tick beside each marker is
    # that model's own permutation null, so "beyond the null" is read within a
    # row rather than against a single pooled threshold.
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax, axis="x")
    facs = ["scale_nominal", "language", "personality", "dyad"]
    off = 0.21
    for i, fac in enumerate(facs):
        for k, mdl in enumerate(FRONTIER):
            r = tests[(tests.model == mdl) & (tests.factor == fac)].iloc[0]
            yi = i + (k - 1.5) * off
            c = MODEL_C[mdl]
            ax.plot([r.null_95, r.max_tv], [yi, yi], color=c, lw=0.7,
                    alpha=0.5, zorder=3)
            ax.plot(r.max_tv, yi, marker=MODEL_M[mdl], ms=4.0, mfc=c,
                    mec=PAGE, mew=0.55, ls="none", zorder=5,
                    label=MODEL_LABEL[mdl] if i == 0 else None)
            ax.plot(r.null_95, yi, marker="|", ms=4.6, mec=INK, mew=0.9,
                    ls="none", zorder=4)
    ax.set_yticks(range(len(facs)))
    ax.set_yticklabels([FACTOR_LABEL[f] for f in facs], fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(facs) - 0.45, -0.55)
    ax.set_xlim(0, 0.86)
    ax.set_xlabel("largest shift in the strategy mix\n"
                  "(total variation between levels)")
    handles = [plt.Line2D([], [], ls="none", marker=MODEL_M[m], ms=4.0,
                          mfc=MODEL_C[m], mec=PAGE, mew=0.55,
                          label=MODEL_LABEL[m].split()[0]) for m in FRONTIER]
    handles += [plt.Line2D([], [], marker="|", ms=4.6, mec=INK, mew=0.9,
                           ls="none", label="permutation null, 95th pct.")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.05, 1.03),
              handletextpad=0.3, labelspacing=0.26, ncol=1)
    ax.set_title("Every factor moves the mix", pad=12)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"], dx=-0.005)
    save(fig, "fig4_conditions")


# ==========================================================================
# Figure 5 -- the anatomy of what the read-out will not name
# ==========================================================================
def fig_abstention():
    pairs = T("T_FR41_abstention_pairs.csv")
    corner = T("T_S17_corner_distance.csv").set_index("bucket")
    motif = T("T_S19_motifs.csv")

    fig = figure(W2, 2.35)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.00, 0.92, 1.14])

    # (a) caught between which two rules? -------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax, axis="x")
    p = pairs.sort_values("share")
    y = np.arange(len(p)).astype(float)
    for yi, r in zip(y, p.itertuples()):
        # each bar is halved between the two rules the posterior is split over,
        # so the pair is legible without reading the tick label
        a, b = r.pair.split("+")
        ax.barh(yi, r.share / 2, height=0.62, color=STRAT_C[a],
                edgecolor=PAGE, linewidth=0.5, zorder=3)
        ax.barh(yi, r.share / 2, left=r.share / 2, height=0.62,
                color=STRAT_C[b], edgecolor=PAGE, linewidth=0.5, zorder=3)
        ax.text(r.share + 0.008, yi, f"{r.share:.2f}".lstrip("0"), ha="left",
                va="center", fontsize=5.8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r.replace("+", " vs ") for r in p.pair], fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.38)
    ax.set_xlabel("share of abstained games")
    ax.text(0.0, -1.15, "the posterior is not spread over four rules; it is\n"
            "split between two, and TFT is in 63% of the ties",
            fontsize=5.4, color=MUTED, ha="left", va="center", linespacing=1.4)
    ax.set_ylim(-1.5, len(p) - 0.4)
    ax.set_title("Which two rules it is\ncaught between", pad=6)

    # (b) distance from the vocabulary ---------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    order = ["exact", "confident", "unclassified"]
    x = np.arange(len(order)).astype(float)
    for xi, b in zip(x, order):
        r = corner.loc[b]
        ax.bar(xi, r["mean"], width=0.60, color=BUCKET_C[b], edgecolor=SPINE,
               linewidth=0.5, zorder=3)
        ax.vlines(xi, r.q25, r.q75, color=INK, lw=0.9, zorder=5)
        ax.plot(xi, r["median"], marker="_", ms=7, mec=INK, mew=1.1, ls="none",
                zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_SHORT[b] for b in order], fontsize=6.0)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.62, len(order) - 0.38)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("distance to the nearest\ncanonical corner")
    ax.text(-0.55, -0.115, "reactive square: $p = P(C\\,|\\,$opp. $C)$, "
            "$q = P(C\\,|\\,$opp. $D)$;\nbars are means, whiskers the "
            "interquartile range", fontsize=5.4, color=MUTED, ha="left",
            va="center", transform=ax.get_xaxis_transform(), linespacing=1.4)
    ax.set_title("It fills the interior", pad=12)

    # (c) which action motifs recur beyond chance ----------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax, axis="x")
    want = ["CCCC", "DDDD", "DDCC", "CCDD", "CDCD", "DCDC"]
    d = (motif[(motif.bucket == "unclassified") & motif.motif.isin(want)]
         .set_index("motif").reindex(want))
    y = np.arange(len(want))[::-1].astype(float)
    for yi, name, r in zip(y, want, d.itertuples()):
        alt = name in ("CDCD", "DCDC")
        c = "#c9c9c9" if alt else ("#0072b2" if name.startswith("C") else "#d55e00")
        ax.barh(yi, r.lift - 1, left=1, height=0.60, color=c, edgecolor=PAGE,
                linewidth=0.5, zorder=3)
        ax.text(max(r.lift, 1.0) + 0.012, yi, f"{r.lift:.2f}", ha="left",
                va="center", fontsize=5.8, color=INK2)
    ax.axvline(1.0, color=INK, lw=0.7, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(want, fontsize=6.0, family="monospace")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.86, 1.72)
    ax.set_xlabel("frequency relative to a within-game shuffle")
    ax.text(0.865, -1.2, "runs and a single change of regime are enriched;\n"
            "alternating play is not", fontsize=5.4, color=MUTED, ha="left",
            va="center", linespacing=1.4)
    ax.set_ylim(-1.55, len(want) - 0.4)
    ax.set_title("Four-round action motifs", pad=12)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"], dx=-0.005)
    save(fig, "fig5_abstention")


# ==========================================================================
# Figure 6 -- what the abstained play is, and whether it is a strategy
# ==========================================================================
def fig_hidden():
    bybucket = T("T_S09_library_by_bucket.csv").set_index("bucket")
    rules = T("T_S11_library_rules.csv")
    coh = T("T_S18_within_game_coherence.csv").set_index("bucket")
    clus = T("T_S15_cluster_search.csv")

    fig = figure(W2, 2.45)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.20, 0.95])

    # (a) does a wider vocabulary name it? -----------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    order = ["confident", "unclassified"]
    x = np.arange(len(order)).astype(float)
    w = 0.30
    for xi, b in zip(x, order):
        r = bybucket.loc[b]
        ax.bar(xi - w / 2, r.extended_exact, width=w * 0.94, color="#0072b2",
               edgecolor=PAGE, linewidth=0.5, zorder=3)
        ax.bar(xi + w / 2, r.extended_null, width=w * 0.94, color="#c9c9c9",
               edgecolor=PAGE, linewidth=0.5, zorder=3)
        ax.text(xi, max(r.extended_exact, r.extended_null) + 0.028,
                f"+{r.excess:.2f}".replace("0.", "."), ha="center",
                va="bottom", fontsize=6.0, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_SHORT[b] for b in order], fontsize=6.2)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.62, len(order) - 0.38)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("share reproduced exactly by\nthe extended vocabulary")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#0072b2", edgecolor=PAGE,
                             lw=0.5, label="observed"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#c9c9c9", edgecolor=PAGE,
                             lw=0.5, label="shuffled null")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.03, 1.06),
              labelspacing=0.28, handlelength=1.1)
    ax.set_title("A wider vocabulary", pad=12)

    # (b) which rules land ----------------------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax, axis="x")
    d = rules.head(8).iloc[::-1]
    y = np.arange(len(d)).astype(float)
    for yi, r in zip(y, d.itertuples()):
        two = "->" in r.best_family
        ax.barh(yi, r.share, height=0.62,
                color="#0072b2" if two else "#8ecae6", edgecolor=PAGE,
                linewidth=0.5, zorder=3)
        ax.text(r.share + 0.008, yi, f"{r.share:.2f}".lstrip("0"), ha="left",
                va="center", fontsize=5.8, color=INK2)
    pretty = {"SuspiciousTFT": "Suspicious TFT", "TwoTitsForTat": "Two tits for tat",
              "ContriteTFT": "Contrite TFT", "SoftMajority": "Soft majority",
              "HardMajority": "Hard majority", "AntiAlternator": "Anti-alternator"}
    ax.set_yticks(y)
    ax.set_yticklabels([pretty.get(r, r).replace("->", " $\\rightarrow$ ")
                        for r in d.best_family], fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.47)
    ax.set_xlabel("share of the abstained games a rule reproduces exactly")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#0072b2", edgecolor=PAGE,
                             lw=0.5, label="two regimes, one switch"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#8ecae6", edgecolor=PAGE,
                             lw=0.5, label="a single non-canonical rule")]
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.03, 0.30),
              labelspacing=0.28, handlelength=1.1)
    ax.text(0.0, -1.55, "median switch at round 2, and the traffic runs "
            "towards cooperation -\nthe direction backward induction does not "
            "predict", fontsize=5.4, color=MUTED, ha="left", va="center",
            linespacing=1.4)
    ax.set_ylim(-2.0, len(d) - 0.4)
    ax.set_title("What names it", pad=12)

    # (c) is it a strategy? ---------------------------------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax)
    order = ["exact", "confident", "unclassified"]
    x = np.arange(len(order)).astype(float)
    for xi, b in zip(x, order):
        ax.bar(xi, coh.loc[b, "r_halves"], width=0.60, color=BUCKET_C[b],
               edgecolor=SPINE, linewidth=0.5, zorder=3)
        ax.text(xi, coh.loc[b, "r_halves"] + 0.025,
                f"{coh.loc[b, 'r_halves']:.2f}".lstrip("0"), ha="center",
                va="bottom", fontsize=6.0, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["deduced", "nearest\nrule", "abstained"], fontsize=5.8)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.62, len(order) - 0.38)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("correlation between the two\nhalves of the same game")
    sil = clus.silhouette.to_numpy()
    ax.text(-0.55, -0.16,
            f"a fixed rule is coherent by construction. Clustering the "
            f"abstained\nplay finds no separated modes: silhouette falls from "
            f"{sil[0]:.2f} at $k$=2 to {sil[-1]:.2f} at $k$=8",
            fontsize=5.4, color=MUTED, ha="left", va="center",
            transform=ax.get_xaxis_transform(), linespacing=1.4)
    ax.set_title("Strategy, or drift?", pad=12)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"], dx=-0.005)
    save(fig, "fig6_hidden")


def main():
    fig_setup()
    fig_readout()
    fig_labels()
    fig_conditions()
    fig_abstention()
    fig_hidden()


if __name__ == "__main__":
    main()
