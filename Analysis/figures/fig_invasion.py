"""The invasion structure the payoff scale destroys, drawn three times.

The canonical figure of finite-population evolutionary game theory: the four
monomorphic states as nodes, the fixation probability of a single mutant as
directed edges, and the small-mutation-limit stationary distribution as node
area.  An edge is drawn only where a mutant fixes more often than neutral
drift would carry it, 1/Z; the edges that survive are the invasions selection
actually favours.

Read left to right at three payoff scales.  At lambda = 0.01 every fixation
probability is within a whisker of drift, so almost nothing is selected and the
four rules share the population.  At lambda = 1 the structure the prisoner's
dilemma is famous for appears: AllC is invaded, tit-for-tat holds, AllD
accumulates.  By lambda = 10 selection is strong enough that AllD holds 96.9%
of the stationary distribution.

Why the panels stop at lambda = 10.  egttools computes fixation probabilities
in float64, and at epsilon = 0.05 they underflow to exactly zero once lambda
reaches 100.  The embedded chain then has two absorbing states, the eigenvalue
one eigenspace stops being one dimensional, and the eigen solver returns
whichever basis vector it lands on: asked for lambda = 1000 it returns
WSLS = 1.0 where the answer is AllD = 1.0.  `s09_egt.py` documents this and
solves those cells in 200 digit arithmetic instead.  Rather than trust the
float64 path silently, `_check` re-reads the mpmath answer from
T14_egt_stationary.csv and refuses to draw a panel that disagrees with it.

The payoff matrix and the numerical route are `s09_egt.py`'s, imported rather
than rewritten, so this figure and the tables cannot drift apart.  egttools
supplies the fixation probabilities and the drawing; the layout, palette and
type are restyled here, because the library's defaults are sized for a screen.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from egttools.analytical import PairwiseComparison
from egttools.games import Matrix2PlayerGameHolder
from egttools.plotting import draw_invasion_diagram

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S      # noqa: E402

SCALING = Path(__file__).resolve().parents[1] / "scaling"
LAMBDAS = ["0.01", "1", "10"]
EPS = "0.05"
Z = 100
BETA = 0.1
ROUNDS = 10
STRATS = ["ALLC", "ALLD", "TFT", "WSLS"]


def _load_s09():
    """Import s09_egt's payoff builder without running its main()."""
    sys.path.insert(0, str(SCALING))
    ns = runpy.run_path(str(SCALING / "s09_egt.py"), run_name="_s09_import")
    return ns["expected_payoff_matrix"]


def _check(lam, sd):
    """Refuse a panel whose float64 stationary distribution is not the answer.

    T14 is computed in 200 digit arithmetic by the Markov chain tree theorem
    and is the authority.  It also records how many absorbing states the
    float64 chain had; more than one means the eigen solve was degenerate and
    its answer is whichever basis vector it happened to land on.
    """
    import pandas as pd
    t = pd.read_csv(S.TABLES / "T14_egt_stationary.csv")
    row = t[(t.lam == float(lam)) & (t.epsilon == float(EPS))]
    if row.empty:
        raise SystemExit(f"T14 has no cell for lambda={lam}, epsilon={EPS}")
    row = row.iloc[0]
    if row.n_absorbing_float64 > 1:
        raise SystemExit(
            f"lambda={lam}: the float64 chain has {int(row.n_absorbing_float64)} "
            "absorbing states, so egttools' stationary distribution is not "
            "identified; use a scale that s09_egt.py solves in float64, or "
            "draw the node areas from T14 instead")
    ref = np.array([row[s] for s in STRATS], dtype=float)
    if np.max(np.abs(ref - sd)) > 1e-6:
        raise SystemExit(
            f"lambda={lam}: egttools gives {np.round(sd, 4)} but T14 gives "
            f"{np.round(ref, 4)}; the figure and the table disagree")
    return ref


def main():
    expected_payoff_matrix = _load_s09()

    fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 2.42))
    fig.subplots_adjust(wspace=0.10)
    claims = ["nothing is selected", "the dilemma appears", "AllD takes 96.9%"]

    for ax, lam, claim, letter in zip(axes, LAMBDAS, claims, "abc"):
        mat = np.array(expected_payoff_matrix(EPS, lam, ROUNDS, float),
                       dtype=float)
        game = Matrix2PlayerGameHolder(len(STRATS), mat)
        model = PairwiseComparison(Z, game)
        tm, rho = model.calculate_transition_and_fixation_matrix_sml(BETA)
        w, v = np.linalg.eig(tm.T)
        sd = np.real(v[:, np.argmin(np.abs(w - 1))])
        sd = sd / sd.sum()
        sd = _check(lam, sd)

        # networkx takes a per-node size, so the node area can carry the
        # stationary probability.  Area, not radius: the eye compares areas.
        sizes = [260 + 2400 * float(p) for p in sd]
        draw_invasion_diagram(
            [S.STRAT_LABEL[s] for s in STRATS], 1 / Z, rho, sd,
            node_size=sizes, font_size_node_labels=7,
            font_size_edge_labels=6, font_size_sd_labels=6,
            edge_width=1.4, node_linewidth=0.8, node_edgecolors=S.SURFACE,
            max_displayed_label_letters=4,
            colors=[S.STRAT_C[s] for s in STRATS], ax=ax)
        ax.set_axis_off()
        ax.set_title("")
        ax.text(0.0, 1.0, letter, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=S.FS_PANEL, color=S.INK,
                fontweight="bold")
        ax.text(0.045, 1.0, rf"$\lambda={lam}$   {claim}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=S.FS_CLAIM, color=S.INK_2)

        print(f"  lambda={lam:>6s}  stationary "
              + "  ".join(f"{s} {p:.3f}" for s, p in zip(STRATS, sd)))

    fig.text(0.5, 0.005,
             r"finite population $Z=100$, Fermi rule, $\beta=0.1$, "
             r"execution error $\epsilon=0.05$; node area is the stationary "
             r"probability, edges are fixation probabilities above drift $1/Z$",
             ha="center", va="top", fontsize=S.FS_NOTE, color=S.MUTED)

    S.save(fig, "f_invasion")


if __name__ == "__main__":
    main()
