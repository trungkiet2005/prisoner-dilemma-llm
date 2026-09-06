"""The replicator flow, on the four faces of the tetrahedron.

Four strategies span a three-simplex, which is a solid and cannot be printed
usefully.  Its boundary is four triangles, one for each strategy left out, and
each triangle is a two-simplex the replicator flow can be drawn on exactly.
Together the four faces carry the whole boundary of the state space, and the
interior flow is what the arrows on the faces point into.

Each face is the standard evolutionary game theory picture: barycentric
coordinates, the replicator vector field as streamlines, and the rest points
filled where they are stable and open where they are not.  Reading it needs no
statistics, which is why it is here.

One row, not one row per payoff scale, and the reason is the result.  Scaling
every payoff by a positive constant multiplies the replicator field by that
constant and changes nothing else: the orbits are the same curves, traversed
faster, and every rest point is where it was.  `verify()` checks this rather
than asserting it, and finds the payoff matrix at lambda = 1000 to be exactly
1e5 times the matrix at lambda = 0.01, with the normalised field agreeing to
4e-14.  So there is only one picture to draw, and drawing it twice would
suggest a comparison the mathematics forbids.

A caution about drawing it twice anyway.  egttools finds rest points with
absolute tolerances, `atol=1e-7` and `atol_stability=1e-4`, so at lambda = 1000
the field is 1e5 larger than those thresholds were chosen for and the root
search reports rest points that are not there.  The flow is unchanged; only the
root finder moves.  This figure therefore draws the faces at lambda = 1, where
the tolerances mean what they were meant to mean.

The whole content of the payoff scale is thus in the *stochastic* picture, not
this one.  See `fig_invasion.py`, where lambda multiplies the selection
intensity of a finite population and moves the stationary distribution from a
flat mix to a point mass on AllD.
"""
from __future__ import annotations

import itertools
import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from egttools.plotting import plot_replicator_dynamics_in_simplex

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S      # noqa: E402

SCALING = Path(__file__).resolve().parents[1] / "scaling"
STRATS = ["ALLC", "ALLD", "TFT", "WSLS"]
EPS = "0.05"
ROUNDS = 10
LAM = "1"


def _epm():
    sys.path.insert(0, str(SCALING))
    ns = runpy.run_path(str(SCALING / "s09_egt.py"), run_name="_s09_import")
    return ns["expected_payoff_matrix"]


def _inside(simplex, roots_xy, stability, tol=1e-6):
    """Drop rest points the root search placed outside the triangle.

    The corners are read off the Simplex2D object rather than assumed, and a
    point is kept when its barycentric coordinates in that triangle are all
    non-negative.  Points outside are solutions of the field equation that are
    not states of the system.
    """
    c = np.asarray(simplex.corners, dtype=float)
    T = np.array([c[0] - c[2], c[1] - c[2]]).T
    keep_xy, keep_st = [], []
    for xy, st in zip(roots_xy, stability):
        p = np.asarray(xy, dtype=float).ravel()[:2]
        try:
            ab = np.linalg.solve(T, p - c[2])
        except np.linalg.LinAlgError:
            continue
        bary = np.array([ab[0], ab[1], 1.0 - ab.sum()])
        if np.all(bary >= -tol):
            keep_xy.append(xy)
            keep_st.append(st)
    return keep_xy, keep_st


def verify(epm):
    """The claim the single row rests on, checked rather than asserted."""
    A = np.array(epm(EPS, "0.01", ROUNDS, float), dtype=float)
    B = np.array(epm(EPS, "1000", ROUNDS, float), dtype=float)
    ratio = B / A
    spread = float(np.nanmax(np.abs(ratio - ratio.flat[0])))

    def field(M, x):
        f = M @ x
        return x * (f - x @ f)

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        x = rng.dirichlet(np.ones(len(STRATS)))
        fa, fb = field(A, x), field(B, x)
        na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
        if na > 1e-12 and nb > 1e-12:
            worst = max(worst, float(np.linalg.norm(fa / na - fb / nb)))
    print(f"  payoff matrix ratio lambda 1000 / 0.01 = {ratio.flat[0]:.6g}, "
          f"constant to {spread:.2g}")
    print(f"  normalised replicator field agrees to {worst:.2g} "
          f"over 2000 random states")
    if spread > 1e-6 or worst > 1e-9:
        raise SystemExit("the replicator field is not scale invariant here; "
                         "the single-row figure is not justified")
    return worst


def main():
    epm = _epm()
    worst = verify(epm)
    mat = np.array(epm(EPS, LAM, ROUNDS, float), dtype=float)

    faces = list(itertools.combinations(range(len(STRATS)), 3))
    fig, axes = plt.subplots(1, len(faces), figsize=(S.FULL, 1.72))

    for ax, tri in zip(axes, faces):
        sub = mat[np.ix_(tri, tri)]
        names = [STRATS[k] for k in tri]
        dropped = [k for k in range(len(STRATS)) if k not in tri][0]

        simplex, _, roots_xy, _, stability = \
            plot_replicator_dynamics_in_simplex(payoff_matrix=sub, ax=ax)
        # The root search returns solutions of the field equation wherever it
        # finds them, including outside the simplex, where they are not states
        # of the system and must not be drawn.  Keep the ones inside.
        roots_xy, stability = _inside(simplex, roots_xy, stability)

        ax.clear()          # redraw from scratch in the paper's own ink
        simplex.add_axis(ax=ax)
        simplex.draw_triangle(color=S.HAIRLINE, linewidth=0.8)
        simplex.draw_gradients(zorder=2, linewidth=0.5, density=0.85,
                               color=S.MUTED, arrowsize=0.6)
        simplex.draw_stationary_points(roots_xy, stability, zorder=6,
                                       linewidth=0.9)

        for k, (name, corner) in enumerate(zip(names, simplex.corners)):
            top = k == 2
            ax.text(corner[0], corner[1] + (0.05 if top else -0.05),
                    S.STRAT_LABEL[name], ha="center",
                    va="bottom" if top else "top", fontsize=S.FS_NOTE,
                    color=S.STRAT_C[name], fontweight="bold", zorder=8)
        ax.set_title(f"without {S.STRAT_LABEL[STRATS[dropped]]}",
                     fontsize=S.FS_NOTE, color=S.INK_2, pad=1)
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(-0.09, 1.09)
        ax.set_ylim(-0.15, 0.98)

    fig.text(0.0, 1.045, "a", ha="left", va="bottom", fontsize=S.FS_PANEL,
             color=S.INK, fontweight="bold")
    fig.text(0.028, 1.045,
             "the deterministic dynamics do not see the payoff scale at all",
             ha="left", va="bottom", fontsize=S.FS_CLAIM, color=S.INK_2)
    fig.text(0.5, -0.02,
             "replicator flow on each face of the three-simplex at "
             rf"$\lambda=1$, $\epsilon=0.05$, ten rounds; filled circles are "
             "stable rest points, open circles unstable.  Rescaling the payoffs "
             "multiplies the field by a constant and moves nothing: over 2000 "
             f"random states the normalised field agrees to {worst:.0e}",
             ha="center", va="top", fontsize=S.FS_NOTE, color=S.MUTED,
             wrap=True)

    fig.subplots_adjust(wspace=0.04)
    S.save(fig, "f_simplex")


if __name__ == "__main__":
    main()
