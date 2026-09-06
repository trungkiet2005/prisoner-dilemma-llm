"""Is the gap between the baseline and the corpus an artefact of three numbers?

`s09_egt.py` fixes the population at Z = 100, the selection intensity at
beta = 0.1 and the execution error at epsilon = 0.05, and on that setting the
stationary distribution marches from a flat mix over the four rules to a point
mass on AllD as the payoff scale rises.  The corpus does not.  A reviewer is
entitled to ask whether the disagreement is a fact about language models or a
fact about those three numbers.

This script answers it by solving the same model on a grid: four population
sizes, five selection intensities, four execution error rates and the ten
payoff scales the corpus actually ran, which is 800 cells.  Every cell is
solved in 200 digit arithmetic by the Markov chain tree theorem, the same route
`s09_egt.py` uses, and for the same reason: at large beta times lambda the
float64 fixation probabilities underflow to zero, the embedded chain acquires a
second absorbing state and a float64 eigen solver returns whichever basis
vector it lands on.

What the grid is asked.  For each (Z, beta, epsilon) setting, does the AllD
share rise with the payoff scale, and by how much?  If it rises everywhere then
the direction of the baseline's prediction is a property of the model and not
of the parameters, and the corpus contradicts the model rather than one corner
of it.

Outputs
  tables/T18_egt_grid.csv        one row per (Z, beta, epsilon, lambda)
  tables/T19_egt_grid_summary.csv  one row per (Z, beta, epsilon) setting

Both are written under `tables/`, redirected only by PD_SCALING_TABLES, so
nothing the manuscript already builds moves.
"""
from __future__ import annotations

import os
import runpy
import sys
import time
from itertools import product
from pathlib import Path

import pandas as pd
from mpmath import mp

HERE = Path(__file__).resolve().parent
TAB = Path(os.environ.get("PD_SCALING_TABLES", HERE / "tables"))
TAB.mkdir(parents=True, exist_ok=True)

DPS = 200
mp.dps = DPS

# The corpus's own ten scales, so the grid is asked about the design that was
# actually run rather than about a convenient geometric ladder.
LAMBDAS = ["0.01", "0.1", "0.25", "0.5", "1", "2", "5", "10", "100", "1000"]
POPULATIONS = [50, 100, 200, 500]
BETAS = ["0.01", "0.05", "0.1", "0.5", "1.0"]
EPSILONS = ["0.0", "0.05", "0.1", "0.2"]
ROUNDS = 10
STRATS = ["ALLC", "ALLD", "TFT", "WSLS"]


def _s09():
    ns = runpy.run_path(str(HERE / "s09_egt.py"), run_name="_s09_import")
    return ns["expected_payoff_matrix"], ns["fixation_mp"], ns["stationary_mp"]


def main():
    expected_payoff_matrix, fixation_mp, stationary_mp = _s09()
    n = len(STRATS)
    rows = []
    t0 = time.time()
    settings = list(product(POPULATIONS, BETAS, EPSILONS))
    print(f"{len(settings)} settings x {len(LAMBDAS)} scales = "
          f"{len(settings) * len(LAMBDAS)} cells at {DPS} digits")

    for s_i, (Z, beta_s, eps_s) in enumerate(settings, 1):
        beta = mp.mpf(beta_s)
        for lam_s in LAMBDAS:
            mat = expected_payoff_matrix(eps_s, lam_s, ROUNDS, mp.mpf)
            rho = [[mp.mpf(0)] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        rho[i][j] = fixation_mp(mat, i, j, Z, beta)
            pi = stationary_mp(rho, n)
            rows.append(dict(Z=Z, beta=float(beta_s), epsilon=float(eps_s),
                             lam=float(lam_s),
                             **{s: float(p) for s, p in zip(STRATS, pi)}))
        if s_i % 10 == 0 or s_i == len(settings):
            print(f"  {s_i}/{len(settings)} settings, {time.time()-t0:.0f} s")

    t18 = pd.DataFrame(rows)
    t18.to_csv(TAB / "T18_egt_grid.csv", index=False)

    # --- one row per setting ------------------------------------------------
    out = []
    for (Z, beta, eps), g in t18.groupby(["Z", "beta", "epsilon"], sort=True):
        g = g.sort_values("lam")
        a = g.ALLD.iloc[0]
        b = g.ALLD.iloc[-1]
        diffs = g.ALLD.diff().dropna()
        out.append(dict(
            Z=Z, beta=beta, epsilon=eps,
            alld_at_min_lambda=a, alld_at_max_lambda=b, rise=b - a,
            monotone_non_decreasing=bool((diffs >= -1e-12).all()),
            largest_decrease=float(diffs.min()),
            alld_ge_90pct_by=float(g.loc[g.ALLD >= 0.90, "lam"].min())
            if (g.ALLD >= 0.90).any() else float("nan"),
        ))
    t19 = pd.DataFrame(out)
    t19.to_csv(TAB / "T19_egt_grid_summary.csv", index=False)

    # --- what the grid says, in the terms the manuscript needs --------------
    n_set = len(t19)
    print(f"\n=== {n_set} settings ===")
    print(f"AllD share rises with lambda in "
          f"{int((t19.rise > 0).sum())} / {n_set} settings")
    print(f"rise is monotone non-decreasing in "
          f"{int(t19.monotone_non_decreasing.sum())} / {n_set}")
    print(f"AllD reaches 90% by some scale in "
          f"{int(t19.alld_ge_90pct_by.notna().sum())} / {n_set}")
    print(f"median rise {t19.rise.median():.3f}, "
          f"range {t19.rise.min():.3f} to {t19.rise.max():.3f}")

    nonmono = t19[~t19.monotone_non_decreasing]
    if len(nonmono):
        print(f"\nthe {len(nonmono)} settings that are not monotone:")
        print(nonmono[["Z", "beta", "epsilon", "rise", "largest_decrease"]]
              .to_string(index=False))
        print("epsilon values among them:",
              sorted(nonmono.epsilon.unique().tolist()))

    print(f"\nwrote T18_egt_grid.csv ({len(t18)} rows) and "
          f"T19_egt_grid_summary.csv ({len(t19)} rows) in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
