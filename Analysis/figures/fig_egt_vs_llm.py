"""The evolutionary baseline against the corpus: the paper's central contrast.

Multiplying every payoff by lambda is inert under the theory of the game, but
it is not inert under the standard *evolutionary* account of the game, and the
two predictions point in opposite directions.  In a finite population under the
Fermi pairwise comparison rule, lambda and the selection intensity beta enter
the update rule only through their product, so raising lambda is exactly
raising selection pressure: small lambda is near-neutral drift, large lambda is
near a best response.  The small-mutation-limit stationary distribution
therefore marches from a flat mix over the four rules to a point mass on ALLD.

The corpus does the opposite.  This figure puts the two side by side.

Panels
  a  The baseline's stationary distribution across lambda, as a stacked band.
     One reading: the green AllC region is squeezed out and the red AllD region
     takes the whole simplex.
  b  The corpus's strategy mix on the same axis, drawn identically, so the
     comparison is a matter of looking rather than of arithmetic.
  c  The single quantity that summarises both, the AllD share, with the two
     curves on one axis and the gap between them shaded.

Everything is read from the tables; nothing is recomputed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S      # noqa: E402
import data as D       # noqa: E402

EPS = 0.05             # the execution error rate the manuscript reports


def main():
    t = D.table("T15_egt_vs_llm.csv")
    t = t[np.isclose(t.epsilon, EPS)].sort_values("lam")
    lam = t.lam.to_numpy()

    egt = {k: t["egt_" + k].to_numpy() for k in S.STRAT_ORDER}
    llm = {"ALLC": t.llm_AllC.to_numpy(), "TFT": t.llm_TFT.to_numpy(),
           "WSLS": t.llm_WSLS.to_numpy(), "ALLD": t.llm_AllD.to_numpy()}

    fig = plt.figure(figsize=(S.FULL, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.12], wspace=0.55)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # --- a, b: the two strategy mixes, drawn the same way ------------------
    for ax, mix, letter, claim in (
        (axA, egt, "a", "evolutionary baseline"),
        (axB, llm, "b", "frontier models"),
    ):
        base = np.zeros_like(lam)
        for k in S.STRAT_ORDER:
            ax.fill_between(lam, base, base + mix[k], color=S.STRAT_C[k],
                            lw=0, alpha=0.92, zorder=2)
            base = base + mix[k]
        S.scale_axis(ax, band=False)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25", "50", "75", "100%"])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        S.panel(ax, letter, claim)

    axA.set_ylabel("share of the population")
    axB.tick_params(labelleft=False, left=False)

    # Name the regions in place, so neither panel needs a legend.
    for k, y in (("ALLC", 0.11), ("TFT", 0.35), ("WSLS", 0.60), ("ALLD", 0.88)):
        axA.text(0.0115, y, S.STRAT_LABEL[k], fontsize=S.FS_NOTE,
                 color=S.SURFACE, ha="left", va="center", zorder=4,
                 fontweight="bold")
    axA.text(300, 0.45, "AllD", fontsize=S.FS_NOTE, color=S.SURFACE,
             ha="center", va="center", zorder=4, fontweight="bold")
    for k, y in (("ALLC", 0.22), ("TFT", 0.54), ("WSLS", 0.66), ("ALLD", 0.86)):
        axB.text(300, y, S.STRAT_LABEL[k], fontsize=S.FS_NOTE,
                 color=S.SURFACE, ha="center", va="center", zorder=4,
                 fontweight="bold")

    # --- c: the AllD share, both accounts on one axis ----------------------
    axC.fill_between(lam, llm["ALLD"], egt["ALLD"], color=S.STRAT_C["ALLD"],
                     alpha=0.10, lw=0, zorder=1)
    axC.plot(lam, egt["ALLD"], color=S.INK, lw=1.6, zorder=3,
             marker="o", ms=3.0, markerfacecolor=S.SURFACE, markeredgewidth=0.9)
    axC.plot(lam, llm["ALLD"], color=S.STRAT_C["ALLD"], lw=1.6, zorder=3,
             marker="o", ms=3.0, markerfacecolor=S.SURFACE, markeredgewidth=0.9)
    S.scale_axis(axC, band=False)
    S.strip(axC)
    axC.set_ylim(0, 1.02)
    axC.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axC.set_yticklabels(["0", "25", "50", "75", "100%"])
    axC.set_ylabel("share labelled AllD")
    S.panel(axC, "c", "the gap is the result")

    axC.annotate("evolutionary\nbaseline", xy=(1000, egt["ALLD"][-1]),
                 xytext=(-4, -12), textcoords="offset points",
                 fontsize=S.FS_NOTE, color=S.INK, ha="right", va="top",
                 linespacing=1.15)
    axC.annotate("frontier models", xy=(1000, llm["ALLD"][-1]),
                 xytext=(-4, 10), textcoords="offset points",
                 fontsize=S.FS_NOTE, color=S.STRAT_C["ALLD"], ha="right",
                 va="bottom")

    fig.text(0.5, -0.06,
             r"finite population $Z=100$, Fermi pairwise comparison, "
             r"$\beta=0.1$, execution error $\epsilon=0.05$, ten rounds; "
             r"corpus pooled over five models, five languages and four persona pairings",
             ha="center", va="top", fontsize=S.FS_NOTE, color=S.MUTED)

    S.save(fig, "f_egt_vs_llm")


if __name__ == "__main__":
    main()
