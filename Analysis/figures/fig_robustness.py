"""Is the baseline's prediction a fact about the model or about three numbers?

The comparison in `fig_egt_vs_llm.py` is made at one setting of the
evolutionary model: population Z = 100, selection intensity beta = 0.1,
execution error epsilon = 0.05.  A reviewer is entitled to ask whether the
disagreement with the corpus survives a different setting, and the honest
answer has to be computed rather than asserted.

`s10_egt_robustness.py` solves the same model on 80 settings, four population
sizes by five selection intensities by four execution errors, at the ten payoff
scales the corpus actually ran, each cell in 200 digit arithmetic.  This figure
reports what the grid found.

Panels
  a  The AllD share against the payoff scale, one thin line per setting, with
     the setting the manuscript reports drawn heavy on top.  Eighty lines would
     normally be forbidden here, but they are not five competing series to be
     told apart: they are one family whose envelope is the message, so they are
     drawn as a hairline sheaf and only the reference curve is identified.
  b  The rise in the AllD share from the smallest scale to the largest, one row
     per execution error, spread over the population sizes and selection
     intensities.  This is where a setting that behaves differently would show.
  c  The corpus on the same axis as the sheaf, which is the point of the whole
     figure: the empirical curve does not lie inside the family at all.

The caveat the grid itself reports is carried in the caption rather than
hidden: the rise is monotone at every positive execution error, and the
exceptions are at epsilon = 0, where the AllD share dips before it rises.
`s09_egt.py` explains why, and the manuscript already carries the same
qualification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S      # noqa: E402
import data as D       # noqa: E402

REF = dict(Z=100, beta=0.1, epsilon=0.05)


def main():
    grid = D.table("T18_egt_grid.csv")
    summ = D.table("T19_egt_grid_summary.csv")
    t15 = D.table("T15_egt_vs_llm.csv")
    t15 = t15[np.isclose(t15.epsilon, REF["epsilon"])].sort_values("lam")

    n_set = len(summ)
    rises = summ.rise
    print(f"  {n_set} settings, {len(grid)} cells")
    print(f"  AllD rises with lambda in {int((rises > 0).sum())}/{n_set}")
    print(f"  median rise {rises.median():.3f}, "
          f"range {rises.min():.3f} to {rises.max():.3f}")

    fig = plt.figure(figsize=(S.FULL, 2.42))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.16, 1.0], wspace=0.44)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # --- a: the sheaf -------------------------------------------------------
    for _, g in grid.groupby(["Z", "beta", "epsilon"]):
        g = g.sort_values("lam")
        axA.plot(g.lam, g.ALLD, color=S.HAIRLINE, lw=0.4, alpha=0.75, zorder=2)
    ref = grid[(grid.Z == REF["Z"]) & np.isclose(grid.beta, REF["beta"])
               & np.isclose(grid.epsilon, REF["epsilon"])].sort_values("lam")
    axA.plot(ref.lam, ref.ALLD, color=S.INK, lw=1.7, zorder=4)
    S.scale_axis(axA, band=False)
    S.strip(axA)
    axA.set_ylim(0, 1.03)
    axA.set_ylabel("share labelled AllD")
    axA.annotate("the setting\nthe paper reports", xy=(10, 0.969),
                 xytext=(-4, -16), textcoords="offset points",
                 fontsize=S.FS_NOTE, color=S.INK, ha="right", va="top",
                 linespacing=1.15)
    S.panel(axA, "a", f"{n_set} settings, one direction")

    # --- b: the rise, by execution error ------------------------------------
    eps_vals = sorted(summ.epsilon.unique())
    ys = np.arange(len(eps_vals))[::-1]
    for y, e in zip(ys, eps_vals):
        sub = summ[np.isclose(summ.epsilon, e)]
        jit = np.linspace(-0.16, 0.16, len(sub))
        axB.scatter(sub.rise, y + jit, s=11, color=S.STRAT_C["ALLD"],
                    alpha=0.75, linewidths=0, zorder=3)
        axB.plot([sub.rise.min(), sub.rise.max()], [y, y], color=S.HAIRLINE,
                 lw=0.7, zorder=2)
        axB.text(1.03, y, f"n={len(sub)}", fontsize=S.FS_NOTE,
                 color=S.MUTED, ha="left", va="center")
    S.zero_rule(axB, 0, vertical=True)
    axB.set_yticks(ys)
    axB.set_yticklabels([rf"$\epsilon={e:g}$" for e in eps_vals])
    axB.set_xlim(-0.05, 1.02)
    axB.set_xlabel("rise in the AllD share")
    S.strip(axB, grid_axis="x")
    axB.tick_params(axis="y", length=0)
    S.panel(axB, "b", "it rises at every setting")

    # --- c: the corpus against the family -----------------------------------
    for _, g in grid.groupby(["Z", "beta", "epsilon"]):
        g = g.sort_values("lam")
        axC.plot(g.lam, g.ALLD, color=S.HAIRLINE, lw=0.4, alpha=0.75, zorder=2)
    axC.plot(t15.lam, t15.llm_AllD, color=S.STRAT_C["ALLD"], lw=1.8, zorder=5,
             marker="o", ms=3.0, markerfacecolor=S.SURFACE,
             markeredgewidth=0.9)
    S.scale_axis(axC, band=False)
    S.strip(axC)
    axC.set_ylim(0, 1.03)
    axC.set_ylabel("share labelled AllD")
    axC.annotate("frontier models", xy=(1000, float(t15.llm_AllD.iloc[-1])),
                 xytext=(-4, 9), textcoords="offset points",
                 fontsize=S.FS_NOTE, color=S.STRAT_C["ALLD"], ha="right",
                 va="bottom")
    S.panel(axC, "c", "the corpus is outside it")

    handles = [Line2D([], [], color=S.HAIRLINE, lw=0.9,
                      label="one evolutionary setting")]
    fig.legend(handles=handles, loc="lower center", ncol=1,
               bbox_to_anchor=(0.5, -0.12), fontsize=S.FS_NOTE)
    fig.text(0.5, -0.175,
             rf"{n_set} settings: $Z\in\{{50,100,200,500\}}$, "
             rf"$\beta\in\{{0.01,0.05,0.1,0.5,1\}}$, "
             rf"$\epsilon\in\{{0,0.05,0.1,0.2\}}$, ten payoff scales, "
             "each solved at 200 digits.  The rise is monotone at every "
             r"positive $\epsilon$; the exceptions are at $\epsilon=0$, "
             "where the share dips before it rises",
             ha="center", va="top", fontsize=S.FS_NOTE, color=S.MUTED)

    S.save(fig, "f_robustness")


if __name__ == "__main__":
    main()
