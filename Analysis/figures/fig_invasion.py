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
import networkx as nx
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


def _edge_labels(ax, G, pos, sizes):
    """The fixation labels, drawn here so the two diagonals do not collide.

    egttools puts every label at the midpoint of its edge.  On the circular
    layout the TFT-AllC and WSLS-AllD edges cross at the centre, so their two
    labels land on the same point and one hides the other.  The vertical one
    is moved down its edge, to where only it runs.
    """
    for u, v, d in G.edges(data=True):
        if d["weight"] <= 1 + 1e-4:
            continue
        (x1, y1), (x2, y2) = pos[u], pos[v]
        lp = 0.5
        if abs(x1 - x2) < 1e-6:
            # label_pos runs from the source u (0) to the target v (1), so the
            # label sits at y1 + lp * (y2 - y1); put it at y = -0.42
            lp = (-0.42 - y1) / (y2 - y1)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={(u, v): rf"{d['weight']:.2f}$\rho_N$"},
            label_pos=lp, font_size=6, node_size=sizes, ax=ax)


def _frame(ax, fig, pos, radius_pt, label_pt):
    """Axes limits wide enough for the largest node and its label.

    Node size is an area in points, not in data units, so the autoscaled
    limits only enclose the node centres and a large node is cut at the axes
    edge.  A node of radius r points at distance 1 from the centre fits when
    the half-span h satisfies 1 + r * 2h / L <= h, with L the axes length in
    points, which gives h = 1 / (1 - 2r / L).
    """
    box = ax.get_position()
    w_pt = box.width * fig.get_figwidth() * 72
    h_pt = box.height * fig.get_figheight() * 72
    hx = hy = 1.0
    for name, (x, y) in pos.items():
        r = radius_pt[name] + 2
        if abs(x) > 0.5:
            hx = max(hx, 1 / (1 - 2 * r / w_pt))
        if abs(y) > 0.5:
            hy = max(hy, 1 / (1 - 2 * (r + label_pt) / h_pt))
    ax.set_xlim(-hx, hx)
    ax.set_ylim(-hy, hy)


def main():
    expected_payoff_matrix = _load_s09()

    fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 3.0))
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
        labels = [S.STRAT_LABEL[s] for s in STRATS]
        G = draw_invasion_diagram(
            labels, 1 / Z, rho, sd,
            node_size=sizes, font_size_node_labels=7,
            display_edge_labels=False, display_sd_labels=False,
            edge_width=1.4, node_linewidth=0.8, node_edgecolors=S.SURFACE,
            max_displayed_label_letters=4,
            colors=[S.STRAT_C[s] for s in STRATS], ax=ax)
        pos = nx.circular_layout(G)       # the layout egttools drew with
        _edge_labels(ax, G, pos, sizes)

        # The stationary probability goes just outside its node, above the top
        # one and below the rest, offset by the node's radius in points so a
        # large node cannot swallow its own label.
        radius = {n: np.sqrt(s) / 2 for n, s in zip(labels, sizes)}
        for n, p in zip(labels, sd):
            x, y = pos[n]
            up = y > 0.5
            ax.annotate(f"{p:.2f}", xy=(x, y), xytext=(0, (1 if up else -1)
                                                         * (radius[n] + 1.5)),
                        textcoords="offset points", ha="center",
                        va="bottom" if up else "top", fontsize=S.FS_NOTE,
                        color=S.INK)
        _frame(ax, fig, pos, radius, label_pt=S.FS_NOTE + 1.5)
        ax.set_axis_off()
        S.panel(ax, letter, rf"$\lambda={lam}$   {claim}")

        print(f"  lambda={lam:>6s}  stationary "
              + "  ".join(f"{s} {p:.3f}" for s, p in zip(STRATS, sd)))

    S.save(fig, "f_invasion")


if __name__ == "__main__":
    main()
