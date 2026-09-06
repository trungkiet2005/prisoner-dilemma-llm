"""What was collected, what was computed, and where the two meet.

The study has two lanes that never touch until the last step, and the figure is
laid out that way on purpose.  The upper lane is empirical: one game, rendered
at ten payoff scales, put to six frontier models in five languages under four
persona pairings, giving twelve thousand dyads.  The lower lane is analytical:
the same game handed to a finite-population evolutionary model, solved rather
than simulated.  Neither lane can see the other.  They are compared only in the
box on the right, and that comparison is the paper.

The numbers in the boxes are read from the corpus at import time rather than
typed, so a figure that disagrees with the data cannot be produced.

Panel b is the object the whole study turns on.  Every payoff in the matrix is
multiplied by lambda.  Under the theory of the game that is inert: preferences,
best replies, equilibria and the replicator field are all unchanged.  It is not
inert for a finite population, where lambda enters the Fermi update rule only
through its product with the selection intensity, and it is not inert for the
models.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S      # noqa: E402
import data as D       # noqa: E402

W, H = 100.0, 40.0     # schematic coordinates, chosen so 1 unit is legible


def box(ax, x, y, w, h, title, body, *, ec, fc=None, title_c=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc or S.SURFACE, ec=ec, lw=0.9, zorder=3))
    ax.text(x + w / 2, y + h - 1.9, title, ha="center", va="top",
            fontsize=S.FS_CLAIM, color=title_c or ec, zorder=4,
            fontweight="bold")
    ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top",
            fontsize=S.FS_NOTE, color=S.INK_2, zorder=4, linespacing=1.35)


def arrow(ax, p, q, *, color=None, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7,
                                 color=color or S.MUTED, lw=0.8,
                                 shrinkA=1.5, shrinkB=1.5, zorder=2))


def main():
    g = D.games()
    r = D.rounds()
    n_models = g.model.nunique()
    n_dyads = g.game_uid.nunique()
    n_games = len(g)
    n_dec = len(r)
    n_lang = g.language.nunique()
    n_scale = g.scale_nominal.nunique()
    coop = g.coop_rate.mean()
    print(f"  {n_models} models, {n_lang} languages, {n_scale} scales, "
          f"{n_dyads:,} dyads, {n_games:,} agent-games, {n_dec:,} decisions, "
          f"overall cooperation {coop:.4f}")

    fig = plt.figure(figsize=(S.FULL, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.15, 1.0], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    EMP = S.MODEL_C["GPT-5.4-Nano"]        # the empirical lane
    THY = S.STRAT_C["ALLD"]                # the analytical lane

    # --- the empirical lane, upper -----------------------------------------
    box(ax, 1, 23.5, 21, 14.5, "the corpus",
        f"{n_models} frontier models\n{n_lang} languages\n"
        "4 persona pairings\n"
        f"{n_scale} payoff scales", ec=EMP)
    box(ax, 25, 23.5, 22, 14.5, "what was played",
        f"{n_dyads:,} dyads\n{n_games:,} agent-games\n"
        f"{n_dec:,} decisions\n10 rounds each", ec=EMP)
    box(ax, 50, 23.5, 23, 14.5, "read out",
        "memory-one vector\nper cell, plus a\nhybrid rule-base and\n"
        "LSTM strategy label", ec=EMP)

    # --- the analytical lane, lower ----------------------------------------
    box(ax, 1, 2.0, 21, 14.5, "the same game",
        "4 canonical rules\nAllC, TFT,\nWSLS, AllD\n"
        r"error $\epsilon$ per round", ec=THY)
    box(ax, 25, 2.0, 22, 14.5, "solved, not run",
        "expected payoffs\nover 10 rounds,\n"
        "in closed form", ec=THY)
    box(ax, 50, 2.0, 23, 14.5, "finite population",
        r"$Z=100$, Fermi rule," "\n"
        r"$\beta=0.1$, egttools;" "\n"
        "200-digit stationary\ndistribution", ec=THY)

    # --- the meeting point --------------------------------------------------
    box(ax, 76, 12.0, 22, 16.0, "the comparison",
        "the scale is inert\nin the theory of the\ngame, decisive in\n"
        "the population, and\nthe corpus matches\nneither", ec=S.INK,
        fc="#f7f9fb", title_c=S.INK)

    for y in (30.7, 9.2):
        arrow(ax, (22, y), (25, y))
        arrow(ax, (47, y), (50, y))
    arrow(ax, (73, 30.7), (76, 24.0), color=EMP)
    arrow(ax, (73, 9.2), (76, 16.0), color=THY)

    ax.text(11.5, 38.6, "collected", fontsize=S.FS_NOTE, color=EMP,
            ha="center", va="bottom", fontweight="bold")
    ax.text(11.5, 17.1, "computed", fontsize=S.FS_NOTE, color=THY,
            ha="center", va="bottom", fontweight="bold")
    S.panel(ax, "a", "two lanes, compared once")

    # --- the payoff matrix, panel b ----------------------------------------
    axm.set_xlim(0, 10)
    axm.set_ylim(0, 10)
    axm.set_axis_off()
    cells = [("6", "0"), ("10", "2")]
    for i in range(2):
        for j in range(2):
            axm.add_patch(plt.Rectangle((3.0 + 2.6 * j, 5.4 - 2.2 * i), 2.6, 2.2,
                                        fc=S.SURFACE, ec=S.HAIRLINE, lw=0.7))
            axm.text(4.3 + 2.6 * j, 6.5 - 2.2 * i,
                     f"λ·{cells[i][j]}", ha="center", va="center",
                     fontsize=S.FS_NOTE, color=S.INK)
    for j, lab in enumerate(["opp A", "opp B"]):
        axm.text(4.3 + 2.6 * j, 7.9, lab, ha="center", va="bottom",
                 fontsize=S.FS_NOTE, color=S.MUTED)
    for i, lab in enumerate(["you A", "you B"]):
        axm.text(2.8, 6.5 - 2.2 * i, lab, ha="right", va="center",
                 fontsize=S.FS_NOTE, color=S.MUTED)
    axm.text(5.0, 2.6,
             "penalties, to be\nminimised.  Every\nentry scaled by "
             r"$\lambda$," "\nwhich changes no\npreference and no\nbest reply.",
             ha="center", va="top", fontsize=S.FS_NOTE, color=S.INK_2,
             linespacing=1.4)
    S.panel(axm, "b", "the knob")

    S.save(fig, "f_overview")


if __name__ == "__main__":
    main()
