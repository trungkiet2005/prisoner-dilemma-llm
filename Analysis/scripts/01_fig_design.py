"""F01 -- experimental design: payoff geometry, scale ladder, run inventory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from pdlib.ingest import payoff_matrix
from pdlib.style import (CMAP_SEQ, C_COOP, C_DEFECT, DATADIR, FRONTIER_MODELS,
                         GRID, INK, INK2, MODEL, MODEL_ORDER, MUTED, OUTCOME,
                         SMALL_MODELS, SURFACE, panel_tag, savefig,
                         use_paper_style)

use_paper_style()


def draw_matrix(ax, fam, title):
    m = payoff_matrix(fam)
    cells = {(0, 0): ("CC", m["R"], m["R"]), (0, 1): ("CD", m["S"], m["T"]),
             (1, 0): ("DC", m["T"], m["S"]), (1, 1): ("DD", m["P"], m["P"])}
    for (i, j), (code, u, v) in cells.items():
        ax.add_patch(Rectangle((j, 1 - i), 1, 1, facecolor=OUTCOME[code],
                               edgecolor=SURFACE, linewidth=2.0, alpha=0.85))
        ax.text(j + 0.5, 1 - i + 0.60, f"{u:g} , {v:g}", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")
        ax.text(j + 0.5, 1 - i + 0.28, code, ha="center", va="center",
                fontsize=8, color="white", alpha=0.9)
    ax.set_xlim(-0.02, 2.02)
    ax.set_ylim(-0.02, 2.02)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["opponent C\n(OptionA)", "opponent D\n(OptionB)"])
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["focal C\n(OptionA)", "focal D\n(OptionB)"])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)
    ax.text(0.5, -0.20, f"greed (T−R)/(T−S) = {m['greed']:.2f}    ·    "
                        f"fear (P−S)/(T−S) = {m['fear']:.2f}",
            ha="center", va="top", fontsize=7.5, color=MUTED,
            transform=ax.transAxes)


def main():
    games = pd.read_parquet(DATADIR / "games.parquet")

    fig = plt.figure(figsize=(11.2, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.42)

    # (a)(b) payoff matrices ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    draw_matrix(ax, "frontier", "Frontier runs  (base units)")
    panel_tag(ax, "a", dx=-0.30)

    ax = fig.add_subplot(gs[0, 1])
    draw_matrix(ax, "small", "Open-weight runs  (base units)")
    panel_tag(ax, "b", dx=-0.30)

    # (c) greed-fear plane ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    # With T and S normalised to 1 and 0, the PD constraints T>R>P>S and
    # 2R>T+S map to 0 < greed < 1/2 and 0 < fear < 1 - greed.
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f0efec", edgecolor="none"))
    xs = np.linspace(0, 0.5, 200)
    ax.fill_between(xs, 0, 1 - xs, color=CMAP_SEQ(0.12), alpha=0.75, lw=0,
                    label="proper PD region\nT>R>P>S and 2R>T+S")
    for fam, mk, col in (("frontier", "o", C_COOP), ("small", "s", C_DEFECT)):
        m = payoff_matrix(fam)
        ax.plot(m["greed"], m["fear"], mk, ms=11, color=col, mec=SURFACE, mew=1.6,
                zorder=5)
        ax.annotate(f"{fam}\nR={m['R']:g}", (m["greed"], m["fear"]),
                    textcoords="offset points", xytext=(10, 8), fontsize=7.5,
                    color=INK2)
    ax.set_xlabel("greed  (T−R)/(T−S)")
    ax.set_ylabel("fear  (P−S)/(T−S)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Where the two matrices sit")
    ax.legend(loc="upper right", fontsize=6.8)
    ax.grid(True, axis="both")
    panel_tag(ax, "c", dx=-0.26)

    # (d) design grid: model x payoff scale ----------------------------------
    ax = fig.add_subplot(gs[1, :2])
    scales = sorted(games.scale_nominal.unique())
    xpos = {s: i for i, s in enumerate(scales)}
    for yi, mdl in enumerate(MODEL_ORDER):
        sub = games[games.model == mdl]
        ax.plot([0, len(scales) - 1], [yi, yi], color=GRID, lw=1.0, zorder=0)
        for s, g in sub.groupby("scale_nominal"):
            ax.scatter(xpos[s], yi, s=np.sqrt(g.game_uid.nunique()) * 22,
                       color=MODEL[mdl], edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([f"{m}  ({'10' if m in FRONTIER_MODELS else '30'} rounds)"
                        for m in MODEL_ORDER])
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f"×{s:g}" for s in scales])
    ax.set_xlabel("payoff scale multiplier")
    ax.set_ylim(-0.7, len(MODEL_ORDER) - 0.15)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_title("Design grid  ·  200 games (400 agent-games) per filled cell")
    panel_tag(ax, "d", dx=-0.24)

    # (e) condition inventory -----------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    # count each *game* once, from agent 1's point of view, so the ordered
    # dyad labels stay meaningful
    a1 = games[games.agent == 1]
    inv = a1.groupby(["family", "dyad"]).game_uid.nunique().unstack(0)
    inv = inv.loc[["CvC", "CvS", "SvC", "SvS"]]
    y = np.arange(len(inv))
    h = 0.38
    for k, (fam, col) in enumerate((("frontier", C_COOP), ("small", C_DEFECT))):
        b = ax.barh(y + (k - 0.5) * h, inv[fam], height=h, color=col,
                    edgecolor=SURFACE, linewidth=1.0, label=fam)
        ax.bar_label(b, fmt="%.0f", padding=3, fontsize=6.8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(["coop vs coop", "coop vs selfish", "selfish vs coop",
                        "selfish vs selfish"])
    ax.invert_yaxis()
    ax.set_xlabel("games")
    ax.set_title("Personality dyads")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.14), ncol=2)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    panel_tag(ax, "e", dx=-0.42)

    fig.suptitle("Experimental design of the FAIRGAME prisoner's-dilemma runs",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F01_design_overview")


if __name__ == "__main__":
    main()
