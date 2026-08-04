"""FR9-FR12: one three-panel model card per frontier LLM.

These are the per-model figures.  They deliberately do not repeat the
cross-model comparisons in FR2-FR8; each card answers the three questions a
reader has about a single model once they know how it ranks:

  a  where in the language x payoff-scale grid does its cooperation sit?
  b  how does the dyad's joint state evolve over the ten rounds?
  c  does the assigned persona change the trajectory, or only its level?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdlib.natstyle import (CMAP_SEQ, DATADIR, DYAD_ORDER, FRONTIER, HORIZON,
                            INK, INK2, LANG_ORDER, LANG_SHORT, MODEL_C,
                            MODEL_LABEL, MODEL_SLUG, MUTED, PAGE, RULE,
                            SCALE_ORDER, SPINE, STACK_C, STACK_LABEL,
                            STACK_ORDER, TABDIR, W2, annotate_heatmap, caption,
                            colorbar, figure, finalize, hgrid, save,
                            use_journal_style)

use_journal_style()

DYAD_SHORT = {"CvC": "coop. vs coop.", "CvS": "coop. vs selfish",
              "SvC": "selfish vs coop.", "SvS": "selfish vs selfish"}
# a two-hue ramp within the model's own colour would collide with the model
# palette; personas get a fixed ordinal ramp instead
DYAD_C = {"CvC": "#0072b2", "CvS": "#6aa8cf", "SvC": "#e0a05a", "SvS": "#d55e00"}

# shared colour limits, so the four cards can be read against each other
VMIN, VMAX = 0.05, 0.80


def card(games, rounds, mdl):
    g = games[games.model == mdl]
    r = rounds[rounds.model == mdl]

    fig = figure(W2, 2.45)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.05])

    # (a) language x payoff-scale grid ---------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    piv = (g.pivot_table(index="language", columns="scale_nominal",
                         values="coop_rate", aggfunc="mean")
           .reindex(index=LANG_ORDER, columns=SCALE_ORDER))
    im = ax.imshow(piv.to_numpy(), cmap=CMAP_SEQ, vmin=VMIN, vmax=VMAX,
                   aspect="auto")
    annotate_heatmap(ax, piv.to_numpy(), thresh=0.50, size=6.2)
    ax.set_xticks(range(len(SCALE_ORDER)))
    ax.set_xticklabels([f"$\\times${s:g}" for s in SCALE_ORDER])
    ax.set_yticks(range(len(LANG_ORDER)))
    ax.set_yticklabels([LANG_SHORT[l] for l in LANG_ORDER])
    ax.tick_params(length=0)
    ax.set_xlabel("payoff scale, $\\lambda$")
    for s in ax.spines.values():
        s.set_visible(False)
    colorbar(fig, im, ax, label="cooperation rate")
    ax.set_title("Cooperation surface", pad=6)

    # (b) joint state over the ten rounds ------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    share = {code: [] for code in STACK_ORDER}
    rounds_ix = sorted(r["round"].unique())
    for t in rounds_ix:
        d = r[r["round"] == t]
        share["CC"].append((d.outcome == "CC").mean())
        share["ANTI"].append(d.outcome.isin(["CD", "DC"]).mean())
        share["DD"].append((d.outcome == "DD").mean())
    ax.stackplot(rounds_ix, [share[c] for c in STACK_ORDER],
                 colors=[STACK_C[c] for c in STACK_ORDER],
                 labels=["CC", "CD / DC", "DD"],
                 edgecolor=PAGE, linewidth=0.4)
    ax.set_xticks(rounds_ix)
    ax.set_xlim(rounds_ix[0], rounds_ix[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("round")
    ax.set_ylabel("share of dyads")
    # above the axes, never on the fills -- a legend inside a stackplot has to
    # fight the very colours it is describing
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=0.9, columnspacing=1.0, fontsize=6.0)
    ax.set_title("Joint state by round", pad=16)

    # (c) trajectory by persona pairing --------------------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax)
    for d in DYAD_ORDER:
        s = (r[r.dyad == d].groupby("round").coop.mean())
        ax.plot(s.index, s.to_numpy(), color=DYAD_C[d], lw=1.0,
                marker="o", ms=2.6, mec=PAGE, mew=0.4, label=DYAD_SHORT[d],
                clip_on=False)
    ax.axhline(0.5, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    ax.set_xticks(rounds_ix)
    ax.set_xlim(0.7, rounds_ix[-1] + 0.3)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("round")
    ax.set_ylabel("cooperation rate")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.0, columnspacing=0.8, fontsize=5.8)
    ax.set_title("Persona pairing", pad=26)

    fig.suptitle(MODEL_LABEL[mdl], x=0.0, ha="left", fontsize=8.5,
                 fontweight="bold", color=MODEL_C[mdl])
    finalize(fig, [pa, pb, pc], ["a", "b", "c"])
    caption(fig, f"{g.game_uid.nunique():,} dyads, "
                 f"{g.game_uid.nunique() * 20:,} decisions. Horizon was "
                 f"{HORIZON[mdl]} to the agents. Colour limits in a are shared "
                 f"across the four model cards, so the panels are directly "
                 f"comparable; the CD and DC bands in b are merged because both "
                 f"agents of every dyad are in the corpus, which makes the two "
                 f"shares equal by construction.")

    idx = FRONTIER.index(mdl) + 9
    save(fig, f"FR{idx}_profile_{MODEL_SLUG[mdl]}")

    piv.to_csv(TABDIR / f"T_FR18_{MODEL_SLUG[mdl]}_surface.csv")
    return piv


def main():
    games = pd.read_parquet(DATADIR / "frontier_games.parquet")
    rounds = pd.read_parquet(DATADIR / "frontier_rounds.parquet")
    for mdl in FRONTIER:
        piv = card(games, rounds, mdl)
        print(f"\n{MODEL_LABEL[mdl]}  (cooperation rate, language x lambda)")
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
