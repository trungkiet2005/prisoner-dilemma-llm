"""What the models condition on, in the space the evolutionary account uses.

The predecessor of a round is a pair (own last action, opponent's last action),
a two-by-two design, so the four conditional cooperation probabilities split
without remainder into a level and three named contrasts: reciprocity, the
opponent main effect that tit-for-tat is made of; persistence, the own main
effect; and matching, the interaction that win-stay lose-shift is made of.

That decomposition is the point of the figure.  Reciprocity is the mechanism
the evolutionary account relies on: conditioning on the other player is what
makes cooperation stable against invasion by defectors.  These agents barely
use it.  In every model the own main effect is the larger of the two, by a
factor between 1.2 and 3.7, and in one model reciprocity is negative outright.
They are conditioning on themselves rather than on their partner, which is why
the finite-population baseline in the previous figure does not describe them.

Panels
  a  Cells in the reciprocity by persistence plane, with the textbook rules at
     their exact coordinates.  Tit-for-tat is far to the right; the corpus is
     stacked up the vertical axis instead.
  b  The three contrasts per model, against the value each rule would give.
  c  An unsupervised check.  Principal components of the raw five-vector, which
     is given no knowledge of the contrasts, recover the same geometry.

The cell is one model, payoff scale and language; a context seen fewer than
thirty times returns no estimate, and the cell is dropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S      # noqa: E402
import data as D       # noqa: E402

CELL = ["model", "scale_nominal", "language"]


def main():
    r = D.rounds()
    r = r[r.model.isin(S.MODEL_ORDER)]
    fp = D.fingerprints(r, CELL).dropna(subset=S.MEM1)
    c = D.contrasts(fp)
    c[CELL] = fp[CELL].values
    print(f"  {len(c)} cells "
          f"({', '.join(f'{k.split(chr(45))[0]} {v}' for k, v in c.model.value_counts().items())})")

    canon = D.canonical_frame().set_index("rule")
    cc = D.contrasts(canon)

    fig = plt.figure(figsize=(S.FULL, 5.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], hspace=0.45,
                          wspace=0.28, left=0.085, right=0.975,
                          top=0.94, bottom=0.14)
    axA = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, :])
    axA.set_box_aspect(1)
    axC.set_box_aspect(1)

    # --- a: the plane the two main effects span ----------------------------
    for m in S.MODEL_ORDER:
        sel = (c.model == m).to_numpy()
        axA.scatter(c.reciprocity[sel], c.persistence[sel], s=11, alpha=0.6,
                    color=S.MODEL_C[m], marker=S.MODEL_M[m], linewidths=0,
                    zorder=3)
    S.zero_rule(axA, 0, vertical=True, lw=0.7)
    S.zero_rule(axA, 0, lw=0.7)
    # AllC, AllD and WSLS all have both main effects zero: on this plane the
    # three rules that never answer the opponent are one point, and saying so
    # is more honest than three markers stacked on each other.
    axA.scatter([1.0], [0.0], s=46, marker="s", facecolors=S.SURFACE,
                edgecolors=S.STRAT_C["TFT"], linewidths=1.4, zorder=6)
    axA.annotate("TFT", (1.0, 0.0), xytext=(0, 9), textcoords="offset points",
                 fontsize=S.FS_NOTE, color=S.STRAT_C["TFT"], ha="center",
                 va="bottom", fontweight="bold", zorder=7)
    axA.scatter([0.0], [0.0], s=46, marker="s", facecolors=S.SURFACE,
                edgecolors=S.INK_2, linewidths=1.4, zorder=6)
    axA.annotate("AllC, AllD,\nWSLS", (0.0, 0.0), xytext=(-10, 6),
                 textcoords="offset points", fontsize=S.FS_NOTE,
                 color=S.INK_2, ha="right", va="bottom", zorder=7,
                 linespacing=1.15)
    S.strip(axA, grid_axis="both")
    axA.set_xlabel("reciprocity      answer the opponent")
    axA.set_ylabel("persistence      answer yourself")
    axA.set_xlim(-0.62, 1.12)
    axA.set_ylim(-0.12, 1.08)
    S.panel(axA, "a", "off the reciprocity axis")

    # --- b: the three contrasts, model by model ----------------------------
    keys = ["reciprocity", "persistence", "matching"]
    xs = np.arange(len(keys))
    w = 0.15
    for i, m in enumerate(S.MODEL_ORDER):
        sel = (c.model == m).to_numpy()
        vals = [c[k][sel].mean() for k in keys]
        los, his = zip(*(D.bootstrap_ci(c[k][sel]) for k in keys))
        off = (i - (len(S.MODEL_ORDER) - 1) / 2) * w
        axB.bar(xs + off, vals, width=w * 0.86, color=S.MODEL_C[m], lw=0,
                zorder=3)
        axB.vlines(xs + off, los, his, color=S.INK, lw=0.7, zorder=4)
    S.zero_rule(axB, 0, lw=0.7)
    # where each textbook rule would land
    for j, k in enumerate(keys):
        for rule in ("TFT", "WSLS"):
            v = cc[k][rule]
            if abs(v) > 1e-9:
                axB.hlines(v, j - 0.34, j + 0.34, color=S.STRAT_C[rule],
                           lw=1.1, linestyles=(0, (3, 2)), zorder=5)
                axB.text(j + 0.36, v, S.STRAT_LABEL[rule], fontsize=S.FS_NOTE,
                         color=S.STRAT_C[rule], ha="left", va="center")
    axB.set_xticks(xs)
    axB.set_xticklabels(["reciprocity", "persistence", "matching"])
    axB.set_ylim(-0.22, 1.14)
    axB.set_ylabel("contrast")
    S.strip(axB)
    axB.tick_params(axis="x", length=0)
    S.panel(axB, "b", "the own effect is larger, in every model")

    # --- c: unsupervised check ---------------------------------------------
    X = StandardScaler().fit_transform(fp[S.MEM1].to_numpy())
    pca = PCA(n_components=2, random_state=0).fit(X)
    Z = pca.transform(X)
    var = pca.explained_variance_ratio_
    for m in S.MODEL_ORDER:
        sel = (c.model == m).to_numpy()
        axC.scatter(Z[sel, 0], Z[sel, 1], s=11, alpha=0.6, color=S.MODEL_C[m],
                    marker=S.MODEL_M[m], linewidths=0, zorder=3)
    S.strip(axC, grid_axis="both")
    axC.set_xlabel(f"PC1  ({var[0]*100:.0f}%)")
    axC.set_ylabel(f"PC2  ({var[1]*100:.0f}%)")
    S.panel(axC, "c", "unsupervised, same")

    handles = [Line2D([], [], color=S.MODEL_C[m], marker=S.MODEL_M[m], lw=0,
                      ms=4.2, label=S.MODEL_LABEL[m]) for m in S.MODEL_ORDER]
    handles.append(Line2D([], [], color=S.INK_2, marker="s", lw=0, ms=5.0,
                          markerfacecolor=S.SURFACE, markeredgewidth=1.2,
                          label="textbook rule"))
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.04), fontsize=S.FS_NOTE)
    S.caption(fig,
              f"{len(c)} cells, one per model, payoff scale and language; "
              "bars are means over cells with a 95% bootstrap interval",
              y=0.005)

    S.save(fig, "f_strategy_space")


if __name__ == "__main__":
    main()
