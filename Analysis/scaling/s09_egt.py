"""Evolutionary game theory baseline for the payoff-scaled prisoner's dilemma.

The manuscript needs a null model that answers one question: if the scale of the
payoffs mattered to a *standard* evolutionary account of this game, how would the
mix of strategies move across the ten scales the corpus actually ran?  Classical
evolutionary game theory has no scale invariance to appeal to.  Multiplying every
payoff by lambda multiplies every fitness difference by lambda, and the Fermi
pairwise comparison rule reads fitness differences through beta times delta, so
lambda and the selection intensity beta are the same knob.  Small lambda is
therefore near neutral drift and large lambda is near a best response dynamic.
The stationary distribution of the small mutation limit chain moves from a flat
mix over the four strategies to a point mass on ALLD.  That prediction is the
baseline the language model corpus is compared against.

Model, stated exactly as the manuscript states it.

  Strategies.  Four canonical memory one rules, ALLC, ALLD, TFT and WSLS, each
  carrying a per round execution error rate epsilon.  Where the rule prescribes
  cooperation the realised cooperation probability is hi = 1 - epsilon, where it
  prescribes defection it is lo = epsilon.  ALLC cooperates always, ALLD defects
  always, TFT cooperates if and only if the opponent cooperated in the previous
  round, WSLS cooperates if and only if its own and the opponent's previous
  actions agreed.  In the first round ALLC, TFT and WSLS open with p(C) = hi and
  ALLD opens with p(C) = lo.

  Payoffs.  The FAIRGAME prompt states penalties to be minimised, with
  T_pen = 0, R_pen = 2, P_pen = 6 and S_pen = 10.  Negating gives utilities
  T = 0, R = -2, P = -6 and S = -10, which satisfy T > R > P > S, and the whole
  matrix is then multiplied by the payoff scale lambda.

  Horizon.  Ten rounds, matching the corpus.  The expected per round payoff is
  computed analytically by propagating the joint distribution over (own action,
  opponent action) forward for ten rounds and averaging, not by simulation.
  egttools' NormalFormGame with strategy instances is deliberately not used: it
  plays a single instance against itself on the diagonal, so the WSLS self play
  entry is corrupted by shared internal state.  The 4 by 4 expected payoff matrix
  is built here and handed to a Matrix2PlayerGameHolder.

  Evolution.  Finite population Z = 100, pairwise comparison with the Fermi rule,
  selection intensity beta = 0.1, small mutation limit.  The fixation probability
  of a single j mutant in an i resident population is the standard product
  formula, the embedded Markov chain runs over the four monomorphic states, and
  its unique stationary distribution is the reported quantity.

Numerics.  At epsilon = 0.05 with lambda = 100 and lambda = 1000 the float64
fixation probabilities underflow to exactly 0, the embedded chain acquires more
than one absorbing state, the eigenvalue 1 eigenspace stops being one
dimensional, and a float64 eigen solver returns whichever basis vector it
happens to land on.  Observed: it returns WSLS = 1.0 where the correct answer is
ALLD = 1.0.  So mpmath at 200 decimal digits is the primary solver here and the
float64 egttools path is run only as a check.  The stationary distribution in
the mpmath path is obtained from the Markov chain tree theorem rather than from
a linear solve, because the tree theorem is a sum of products of positive
numbers and so has no cancellation at all, which matters when the transition
probabilities in one chain span hundreds of orders of magnitude.  Every cell
records the largest absolute discrepancy between the two solvers and the number
of absorbing states the float64 chain had.

One qualification the manuscript has to carry.  The ALLD share rises
monotonically in lambda at every positive execution error rate, but not at
epsilon = 0, where it dips between lambda = 0.1 and lambda = 2 before rising.
That is a property of the model and not of the arithmetic; the note above the
monotonicity self check at the foot of this file gives the reason.

Outputs
  tables/T14_egt_stationary.csv   the full 10 lambda by 5 epsilon grid
  tables/T15_egt_vs_llm.csv       epsilon = 0.05 beside the pooled empirical mix

Both destinations default to Analysis/scaling/tables and are redirected only by
setting PD_SCALING_TABLES, so nothing the manuscript already builds moves.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from mpmath import mp

import egttools
from egttools.analytical import PairwiseComparison
from egttools.behaviors.NormalForm.TwoActions import Cooperator, Defector
from egttools.behaviors.NormalForm.TwoActions import TFT as EgtTFT
from egttools.games import Matrix2PlayerGameHolder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MODEL_LABEL, MODEL_ORDER, MODEL_ORDER_ALL, SCALES  # noqa: E402

# RuntimeWarnings are the signal this script exists to report, above all the
# degenerate chain warning egttools raises when a transition probability reaches
# 1.  Show every one of them, every time, and never swallow them.
warnings.simplefilter("always", RuntimeWarning)
np.seterr(over="warn", divide="warn", invalid="warn")

HERE = Path(__file__).resolve().parent
TAB = Path(os.environ.get("PD_SCALING_TABLES", HERE / "tables"))
TAB.mkdir(parents=True, exist_ok=True)
# The manuscript directory is configurable in the same way as figstyle.PAPER_DIR
# and s07_supplementary.PAPER_DIR, so one pipeline can feed either manuscript.
PAPER_DIR = Path(os.environ.get("PD_PAPER_DIR", HERE.parents[1] / "paper_scaling"))

DPS = 200
mp.dps = DPS

Z = 100
BETA = 0.1
R_ROUNDS = 10
EPSILONS = ["0.0", "0.05", "0.1", "0.2", "0.3"]
STRATS = ["ALLC", "ALLD", "TFT", "WSLS"]

# Penalties exactly as the FAIRGAME prompt states them, to be minimised, then
# negated into utilities to be maximised.  The negation is the whole content of
# the mapping, so it is done here once rather than by hand.
PEN = {"T": 0, "R": 2, "P": 6, "S": 10}
UTIL = {k: str(-v) for k, v in PEN.items()}
assert (-PEN["T"] > -PEN["R"] > -PEN["P"] > -PEN["S"]), \
    "the negated penalties do not satisfy T > R > P > S"


# ---------------------------------------------------------------- encoding ---
def _verify_action_encoding():
    """Read the cooperate and defect action codes off egttools, do not assume.

    Returns (COOPERATE, DEFECT).  The check runs at import time so a library
    upgrade that renumbered the actions stops the script instead of silently
    transposing the payoff matrix.
    """
    allc = [Cooperator().get_action(t, a) for t in (0, 1, 5) for a in (0, 1)]
    alld = [Defector().get_action(t, a) for t in (0, 1, 5) for a in (0, 1)]
    if len(set(allc)) != 1 or len(set(alld)) != 1 or allc[0] == alld[0]:
        raise AssertionError("egttools AllC/AllD are not constant and distinct: "
                             f"{allc} vs {alld}")
    coop, defect = allc[0], alld[0]
    if {coop, defect} != {0, 1}:
        raise AssertionError(f"unexpected action alphabet {coop}, {defect}")
    # TFT must echo the opponent's previous action under the same codes.
    if EgtTFT().get_action(1, coop) != coop or EgtTFT().get_action(1, defect) != defect:
        raise AssertionError("egttools TFT does not echo the opponent under the "
                             "codes read off AllC and AllD")
    if EgtTFT().get_action(0, defect) != coop:
        raise AssertionError("egttools TFT does not open with cooperation")
    return coop, defect


C, D = _verify_action_encoding()


# ------------------------------------------------------------- the strategy ---
def first_coop_prob(name, hi, lo):
    """Probability of cooperating in round one."""
    return lo if name == "ALLD" else hi


def coop_prob(name, own, opp, hi, lo):
    """Probability of cooperating given own and opponent previous actions."""
    if name == "ALLC":
        return hi
    if name == "ALLD":
        return lo
    if name == "TFT":
        return hi if opp == C else lo
    if name == "WSLS":
        return hi if own == opp else lo
    raise ValueError(name)


def expected_payoff_matrix(eps_s, lam_s, rounds, conv):
    """The 4 by 4 matrix of expected per round payoffs, computed analytically.

    conv turns a decimal string into the working number type, so the same code
    produces the float64 matrix and the 200 digit mpmath matrix.  The joint
    distribution over (own action, opponent action) is propagated forward for
    `rounds` rounds; conditional on that joint state the two players' next
    actions are independent, which is what makes the propagation exact.
    """
    eps = conv(eps_s)
    lam = conv(lam_s)
    hi = 1 - eps
    lo = eps
    zero = conv("0")
    one = conv("1")
    util = {(C, C): conv(UTIL["R"]) * lam, (C, D): conv(UTIL["S"]) * lam,
            (D, C): conv(UTIL["T"]) * lam, (D, D): conv(UTIL["P"]) * lam}
    states = [(C, C), (C, D), (D, C), (D, D)]
    n = len(STRATS)
    mat = [[zero] * n for _ in range(n)]
    for i, si in enumerate(STRATS):
        for j, sj in enumerate(STRATS):
            px = first_coop_prob(si, hi, lo)
            py = first_coop_prob(sj, hi, lo)
            dist = {(C, C): px * py, (C, D): px * (one - py),
                    (D, C): (one - px) * py, (D, D): (one - px) * (one - py)}
            total = zero
            for t in range(rounds):
                for s in states:
                    total = total + dist[s] * util[s]
                if t == rounds - 1:
                    break
                nxt = {s: zero for s in states}
                for (a, b) in states:
                    pr = dist[(a, b)]
                    qx = coop_prob(si, a, b, hi, lo)   # focal: own a, opponent b
                    qy = coop_prob(sj, b, a, hi, lo)   # opponent: own b, focal a
                    nxt[(C, C)] = nxt[(C, C)] + pr * qx * qy
                    nxt[(C, D)] = nxt[(C, D)] + pr * qx * (one - qy)
                    nxt[(D, C)] = nxt[(D, C)] + pr * (one - qx) * qy
                    nxt[(D, D)] = nxt[(D, D)] + pr * (one - qx) * (one - qy)
                dist = nxt
            mat[i][j] = total / rounds
    return mat


# ----------------------------------------------------------- the dynamics ----
def fixation_mp(mat, i, j, pop, beta):
    """Fixation probability of a single j mutant in an i resident population.

    Pairwise comparison with the Fermi rule leaves the ratio of the down and up
    transition probabilities equal to exp(-beta * (f_j(k) - f_i(k))), so the
    usual product formula collapses to a single running sum of fitness
    differences.  That is what keeps this stable at 200 digits when beta times
    lambda is large enough to send the exponent past 10^5.
    """
    zm = mp.mpf(pop)
    zm1 = mp.mpf(pop - 1)
    cum = mp.mpf(0)
    acc = mp.mpf(0)
    for k in range(1, pop):
        kk = mp.mpf(k)
        fj = ((kk - 1) * mat[j][j] + (zm - kk) * mat[j][i]) / zm1
        fi = (kk * mat[i][j] + (zm - kk - 1) * mat[i][i]) / zm1
        cum += fj - fi
        acc += mp.exp(-beta * cum)
    return 1 / (1 + acc)


def _verify_matrix_orientation():
    """Confirm payoff_matrix[i][j] is the payoff of i against j, do not assume.

    Strategy 1 strictly dominates strategy 0 in the probe matrix, so a single
    mutant of 1 must fixate in a resident 0 population more often than a neutral
    mutant would, and less often the other way round.  The transposed convention
    reverses both inequalities, so this discriminates.  The same probe also
    checks the fixation formula reimplemented above against the library's own,
    which pins down the Fermi sign convention and the Z - 1 denominators.
    """
    probe = np.array([[0.0, 0.0], [1.0, 1.0]])
    # The game object must be held in a named local: PairwiseComparison keeps a
    # bare reference to it across the C++ boundary, so handing it a temporary
    # lets Python free the game and the next call reads freed memory.  That is an
    # access violation, not an exception, and it kills the interpreter outright.
    game = Matrix2PlayerGameHolder(2, probe)
    model = PairwiseComparison(Z, game)
    _, fx = model.calculate_transition_and_fixation_matrix_sml(BETA)
    if not (fx[0, 1] > 1.0 / Z > fx[1, 0]):
        raise AssertionError(
            "egttools payoff matrix orientation is not [i][j] = payoff of i "
            f"against j: fixation of 1 in 0 = {fx[0, 1]}, of 0 in 1 = {fx[1, 0]}")
    mine = [[mp.mpf(probe[i][j]) for j in range(2)] for i in range(2)]
    for i, j in ((0, 1), (1, 0)):
        got = float(fixation_mp(mine, i, j, Z, mp.mpf(str(BETA))))
        if abs(got - fx[i, j]) > 1e-10:
            raise AssertionError(
                "reimplemented fixation probability disagrees with egttools at "
                f"({i},{j}): {got} vs {fx[i, j]}")


def _arborescences(n, root):
    """Every spanning tree on n nodes with all edges directed toward root."""
    others = [v for v in range(n) if v != root]
    choices = [[u for u in range(n) if u != v] for v in others]
    for succ in product(*choices):
        parent = dict(zip(others, succ))
        ok = True
        for v in others:
            seen, cur = set(), v
            while cur != root:
                if cur in seen:
                    ok = False
                    break
                seen.add(cur)
                cur = parent[cur]
            if not ok:
                break
        if ok:
            yield parent


def stationary_mp(rho, n):
    """Stationary distribution of the small mutation limit chain, exactly.

    The Markov chain tree theorem gives pi_i proportional to the sum, over
    spanning trees directed toward i, of the product of the transition
    probabilities along the tree's edges.  Every term is positive, so there is no
    cancellation and no linear solve, and the answer stays correct when the
    fixation probabilities span hundreds of orders of magnitude.  The uniform
    1 / (n - 1) mutation factor appears n - 1 times in every tree and so cancels
    in the normalisation.
    """
    weights = []
    for root in range(n):
        tot = mp.mpf(0)
        for parent in _arborescences(n, root):
            pr = mp.mpf(1)
            for v, u in parent.items():
                pr *= rho[v][u]
            tot += pr
        weights.append(tot)
    total = sum(weights)
    if total == 0:
        raise AssertionError("degenerate chain: every spanning tree has weight 0")
    return [w / total for w in weights]


def sml_transition_mp(rho, n):
    """Row stochastic embedded chain, T[i][j] = rho[i][j] / (n - 1)."""
    tm = [[mp.mpf(0)] * n for _ in range(n)]
    for i in range(n):
        off = mp.mpf(0)
        for j in range(n):
            if i != j:
                tm[i][j] = rho[i][j] / (n - 1)
                off += tm[i][j]
        tm[i][i] = 1 - off
    return tm


def residual_mp(pi, tm, n):
    """max_j | (pi T)_j - pi_j |, the check that the tree theorem solved it."""
    worst = mp.mpf(0)
    for j in range(n):
        acc = mp.mpf(0)
        for i in range(n):
            acc += pi[i] * tm[i][j]
        worst = max(worst, abs(acc - pi[j]))
    return worst


# ------------------------------------------------------------------ driver ---
def run_cell(lam_s, eps_s):
    n = len(STRATS)
    beta = mp.mpf(str(BETA))

    mat_mp = expected_payoff_matrix(eps_s, lam_s, R_ROUNDS, mp.mpf)
    rho = [[mp.mpf(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                rho[i][j] = fixation_mp(mat_mp, i, j, Z, beta)
    pi_mp = stationary_mp(rho, n)
    resid = residual_mp(pi_mp, sml_transition_mp(rho, n), n)

    mat_f = np.array(expected_payoff_matrix(eps_s, lam_s, R_ROUNDS, float),
                     dtype=np.float64)
    game = Matrix2PlayerGameHolder(n, mat_f)   # named local, see the note above
    model = PairwiseComparison(Z, game)
    tm_f, fx_f = model.calculate_transition_and_fixation_matrix_sml(BETA)
    # egttools' helper takes the right eigenvector, so it needs the column
    # stochastic form of a row stochastic transition matrix.
    pi_f = egttools.utils.calculate_stationary_distribution(tm_f.transpose())
    # An absorbing state is a row whose diagonal is exactly 1 in float64, which
    # is what happens when every fixation probability leaving it underflows.
    n_abs = int(np.sum(np.diag(tm_f) == 1.0))

    pi_mp_f = np.array([float(x) for x in pi_mp], dtype=np.float64)
    pi_f = np.asarray(pi_f, dtype=np.float64)
    diff = float(np.max(np.abs(pi_f - pi_mp_f)))
    rho_f = np.array([[float(rho[i][j]) if i != j else 0.0 for j in range(n)]
                      for i in range(n)])
    fx_diff = float(np.max(np.abs(fx_f - rho_f)))
    return pi_mp_f, pi_f, diff, n_abs, float(resid), fx_diff


def _fmt_share(v):
    """Print a stationary share so that a vanishing one still reads as small.

    The grid spans 245 orders of magnitude, so a fixed number of decimals would
    print most of the top of it as an unbroken column of 0.000 and lose the
    fact that the shares vanish at very different rates.
    """
    v = float(v)
    if v == 0.0:
        return "0"
    if v < 5e-4:
        m, e = f"{v:.0e}".split("e")
        return f"${m}\\!\\times\\!10^{{{int(e)}}}$"
    return f"{v:.3f}"


def write_latex_tables(t14, t15, out_path):
    """Emit egt_tables_auto.tex, which appendix.tex \\input's.

    Three tables: the reference grid at the execution-noise level the
    comparison uses, the defection share across every noise level, and the
    baseline beside the corpus.  The precision audit travels in the first
    table rather than in prose, because a reader checking the claim that
    double precision fails at the top of the grid should be able to see which
    cells failed and by how much.
    """
    eps_ref = 0.05
    ref = t14[np.isclose(t14["epsilon"], eps_ref)].sort_values("lam")

    rows = []
    for _, r in ref.iterrows():
        # Printed as an integer rather than blanked below 2: a reader checking
        # the numerics wants to see the count go 0, 0, ..., 2, and a run of
        # hyphens would also trip the project's em dash check.
        audit = f"{int(r['n_absorbing_float64'])}"
        rows.append(
            f"{r['lam']:g} & " + " & ".join(_fmt_share(r[k]) for k in STRATS)
            + f" & {r['max_float64_mpmath_diff']:.1e} & {audit} \\\\")
    t_a = f"""\\begin{{table}}[htbp]
\\centering
\\footnotesize
\\caption{{\\textbf{{Stationary distribution of the evolutionary baseline at
$\\varepsilon = {eps_ref}$.}} Long-run frequency of each canonical rule under the
pairwise-comparison process in the small-mutation limit, $Z = 100$,
$\\beta = 0.1$, $r = 10$ rounds, computed in 200-digit arithmetic. The last two
columns are the precision audit: the largest absolute disagreement between the
200-digit solver and a double-precision one, and the number of absorbing states
the double-precision chain acquires where that number exceeds one. Double
precision fails exactly where the audit says it does, at the top of the grid,
and returns a wrong answer there without any warning.}}
\\label{{tab:C-egt-eps05}}
\\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\\toprule
& \\multicolumn{{4}}{{c}}{{stationary frequency}} & \\multicolumn{{2}}{{c}}{{audit}} \\\\
\\cmidrule(lr){{2-5}} \\cmidrule(lr){{6-7}}
$\\lam$ & AllC & AllD & TFT & WSLS & $\\max|\\Delta|$ & absorbing \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    eps_list = sorted(t14["epsilon"].unique())
    piv = t14.pivot(index="lam", columns="epsilon", values="ALLD")
    rows = [f"{lam:g} & "
            + " & ".join(f"{piv.loc[lam, e]:.3f}" for e in eps_list) + " \\\\"
            for lam in sorted(piv.index)]
    t_b = f"""\\begin{{table}}[htbp]
\\centering
\\footnotesize
\\caption{{\\textbf{{Stationary frequency of Always Defect, across the payoff
grid and every execution-noise level.}} The same process as
table~\\ref{{tab:C-egt-eps05}}. Raising the payoff scale hands the population to
Always Defect at every noise level, and does so sooner the noisier execution is.
The one exception to monotonicity is the error-free corner $\\varepsilon = 0$,
where the share dips slightly before rising: with no execution error
Tit-for-Tat loses only its opening move to Always Defect, so raising the scale
first sharpens selection in Tit-for-Tat's favour. Any positive error rate
removes the exception, because it lets Always Defect exploit the retaliation lag
repeatedly rather than once.}}
\\label{{tab:C-egt-alld}}
\\begin{{tabular}}{{@{{}}l{'c' * len(eps_list)}@{{}}}}
\\toprule
& \\multicolumn{{{len(eps_list)}}}{{c}}{{execution error $\\varepsilon$}} \\\\
\\cmidrule(lr){{2-{1 + len(eps_list)}}}
$\\lam$ & {' & '.join(f'{e:g}' for e in eps_list)} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    rows = []
    for _, r in t15.sort_values("lam").iterrows():
        rows.append(
            f"{r['lam']:g} & "
            + " & ".join(_fmt_share(r[f"egt_{k}"]) for k in STRATS) + " & "
            + " & ".join(f"{r[c]:.3f}" for c in
                         ["llm_AllC", "llm_AllD", "llm_TFT", "llm_WSLS"])
            + " \\\\")
    t_c = f"""\\begin{{table}}[htbp]
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\caption{{\\textbf{{The evolutionary baseline beside the corpus, at matched
payoff scales.}} Left, the stationary frequency of each canonical rule at
$\\varepsilon = 0.05$; right, the share of agent-games carrying each label,
pooled over the five models the main text reports. The two move in opposite
directions: the baseline's Always Defect share rises from 0.267 to 1.000 across
the grid while the corpus's falls from 0.436 to 0.283.}}
\\label{{tab:C-egt-vs-llm}}
\\begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
\\toprule
& \\multicolumn{{4}}{{c}}{{evolutionary baseline}} & \\multicolumn{{4}}{{c}}{{corpus}} \\\\
\\cmidrule(lr){{2-5}} \\cmidrule(lr){{6-9}}
$\\lam$ & AllC & AllD & TFT & WSLS & AllC & AllD & TFT & WSLS \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    text = ("% Generated by Analysis/scaling/s09_egt.py - do not edit.\n"
            + t_a + "\n" + t_b + "\n" + t_c)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")


def main():
    t0 = time.time()
    print(f"egttools {egttools.VERSION}, mpmath dps = {mp.dps}, "
          f"action codes: cooperate = {C}, defect = {D}")
    _verify_matrix_orientation()
    print("import time checks passed: action encoding, matrix orientation, "
          "fixation formula")
    print(f"Z = {Z}, beta = {BETA}, rounds = {R_ROUNDS}, strategies = {STRATS}")
    print("payoff utilities before scaling: T = 0, R = -2, P = -6, S = -10\n")

    rows = []
    f64 = {}                      # (lam, eps) -> what the float64 path returned
    worst_resid = 0.0
    worst_fx = 0.0
    for eps_s in EPSILONS:
        for lam in SCALES:
            lam_s = repr(lam)
            pi_mp_f, pi_f, diff, n_abs, resid, fx_diff = run_cell(lam_s, eps_s)
            worst_resid = max(worst_resid, resid)
            worst_fx = max(worst_fx, fx_diff)
            f64[(lam, float(eps_s))] = pi_f
            rows.append({
                "lam": lam, "epsilon": float(eps_s), "Z": Z, "beta": BETA,
                "r": R_ROUNDS,
                "ALLC": pi_mp_f[0], "ALLD": pi_mp_f[1],
                "TFT": pi_mp_f[2], "WSLS": pi_mp_f[3],
                "max_float64_mpmath_diff": diff,
                "n_absorbing_float64": n_abs,
                "solver": f"mpmath_dps{DPS}_tree_theorem",
            })
    t14 = pd.DataFrame(rows)
    t14.to_csv(TAB / "T14_egt_stationary.csv", index=False)
    print(f"\nwrote {TAB / 'T14_egt_stationary.csv'}  ({len(t14)} rows)")
    print(f"worst mpmath stationarity residual over the grid: {worst_resid:.3e}")
    print(f"worst fixation probability gap, float64 vs mpmath: {worst_fx:.3e}")

    # ------------------------------------------------------------- T15 ------
    t13 = pd.read_csv(TAB / "T13_pooled_strategy.csv")
    ref = t14[np.isclose(t14["epsilon"], 0.05)].set_index("lam")
    out = []
    for lam in SCALES:
        e = ref.loc[lam]
        m = t13.loc[np.isclose(t13["scale_nominal"], lam)].iloc[0]
        out.append({
            "lam": lam, "epsilon": 0.05, "Z": Z, "beta": BETA, "r": R_ROUNDS,
            "egt_ALLC": e["ALLC"], "egt_ALLD": e["ALLD"],
            "egt_TFT": e["TFT"], "egt_WSLS": e["WSLS"],
            "llm_AllC": m["AllC"] / 100.0, "llm_AllD": m["AllD"] / 100.0,
            "llm_TFT": m["TFT"] / 100.0, "llm_WSLS": m["WSLS"] / 100.0,
            "n_models_pooled": len(MODEL_ORDER),
        })
    t15 = pd.DataFrame(out)
    t15.to_csv(TAB / "T15_egt_vs_llm.csv", index=False)
    print(f"wrote {TAB / 'T15_egt_vs_llm.csv'}  ({len(t15)} rows)")
    print("both egt_* and llm_* columns are shares in [0, 1]; the llm_* columns "
          "are the T13 percentages over the five main text models divided by 100")

    write_latex_tables(t14, t15, PAPER_DIR / "egt_tables_auto.tex")

    # ------------------------------------------------ reporting to stdout ----
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n=== T14, epsilon = 0.05, all ten payoff scales ===")
    print(ref.reset_index()[["lam", "ALLC", "ALLD", "TFT", "WSLS",
                             "max_float64_mpmath_diff",
                             "n_absorbing_float64"]].to_string(
        index=False, float_format=lambda v: f"{v:.6f}"))

    print("\n=== T14, full grid, ALLD share by epsilon (columns) ===")
    piv = t14.pivot(index="lam", columns="epsilon", values="ALLD")
    print(piv.to_string(float_format=lambda v: f"{v:.6f}"))

    print("\n=== T15, EGT at epsilon = 0.05 beside the pooled empirical mix ===")
    print(t15[["lam", "egt_ALLC", "egt_ALLD", "egt_TFT", "egt_WSLS",
               "llm_AllC", "llm_AllD", "llm_TFT", "llm_WSLS"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== float64 absorbing state census over the 50 cells ===")
    print(t14["n_absorbing_float64"].value_counts().sort_index().to_string())
    print(t14[t14["n_absorbing_float64"] >= 1][
        ["lam", "epsilon", "n_absorbing_float64", "max_float64_mpmath_diff"]
    ].to_string(index=False))

    bad = t14[t14["n_absorbing_float64"] > 1]
    print(f"\nfloat64 chains with more than one absorbing state: {len(bad)}")
    for _, row in bad.iterrows():
        got = f64[(row["lam"], row["epsilon"])]
        print(f"  lambda = {row['lam']:g}, epsilon = {row['epsilon']:g}: "
              f"{int(row['n_absorbing_float64'])} absorbing states; float64 gave "
              + ", ".join(f"{k} = {v:.6f}" for k, v in zip(STRATS, got))
              + "; mpmath gave "
              + ", ".join(f"{k} = {row[k]:.6f}" for k in STRATS))
    print("largest float64 versus mpmath discrepancy anywhere in the grid: "
          f"{t14['max_float64_mpmath_diff'].max():.6f}")
    top = t14.loc[t14["max_float64_mpmath_diff"].idxmax()]
    print(f"  attained at lambda = {top['lam']:g}, epsilon = {top['epsilon']:g}")
    ergodic = t14[t14["n_absorbing_float64"] == 0]["max_float64_mpmath_diff"]
    print("largest discrepancy over the cells whose float64 chain kept no "
          f"absorbing state at all: {ergodic.max():.3e}")

    print("\n=== distance of ALLD from 1 at epsilon = 0.05, as 1 - ALLD ===")
    for lam in SCALES:
        print(f"  lambda = {lam:<8g} 1 - ALLD = {1.0 - float(ref.loc[lam, 'ALLD']):.6e}")

    # ------------------------------------------- execution noise calibration --
    print("\n=== implied epsilon from the empirical rule distance (T11) ===")
    t11 = pd.read_csv(TAB / "T11_rule_distance.csv").set_index("model")
    cal = []
    for m in MODEL_ORDER_ALL:
        cal.append({"model": MODEL_LABEL[m],
                    "mean_dist": t11.loc[m, "mean_dist"],
                    "implied_epsilon": t11.loc[m, "mean_dist"] / R_ROUNDS,
                    "in_main_text": m in MODEL_ORDER})
    cal = pd.DataFrame(cal)
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    main5 = cal[cal["in_main_text"]]["implied_epsilon"]
    print(f"five main text models: mean {main5.mean():.4f}, "
          f"range {main5.min():.4f} to {main5.max():.4f}")

    # --------------------------------------------------------- self checks ---
    print("\n=== self checks ===")
    # Monotonicity holds at every positive execution error rate.  It does NOT
    # hold at epsilon = 0, and that is a property of the model rather than of the
    # arithmetic, so the manuscript states the qualified claim.  With no
    # execution error ALLC, TFT and WSLS all earn exactly -2 lambda against one
    # another and against themselves, so the three cooperative rules are
    # perfectly neutral among themselves and the ordering is decided entirely by
    # their exchanges with ALLD.  Over ten rounds TFT loses only its opening move
    # to ALLD, taking -6.4 lambda against ALLD's -5.4 lambda, whereas WSLS takes
    # -8 lambda against -3 lambda and ALLC takes -10 lambda against 0.  Raising
    # lambda therefore first sharpens selection in TFT's favour and the ALLD
    # share dips, before ALLD's one round advantage over TFT finally decides it.
    # Any execution error breaks the tie, because it lets ALLD exploit the
    # retaliation lag repeatedly rather than once, and the dip disappears.
    nonmono = []
    for eps in sorted(t14["epsilon"].unique()):
        sub = t14[t14["epsilon"] == eps].sort_values("lam")
        d = np.diff(sub["ALLD"].to_numpy())
        if d.min() <= -1e-12:
            nonmono.append((eps, float(d.min())))
            continue
        print(f"  ALLD monotone non-decreasing in lambda at epsilon = {eps} "
              f"(smallest step {d.min():+.3e})")
    assert [e for e, _ in nonmono] == [0.0], (
        "the only epsilon at which ALLD should be non-monotone is the error "
        f"free corner 0.0, got {nonmono}")
    print(f"  the single exception is epsilon = 0, where the largest decrease "
          f"is {nonmono[0][1]:+.3e} (see the note in the source)")
    zero = t14[t14["epsilon"] == 0.0].sort_values("lam")
    assert zero["ALLD"].iloc[-1] >= zero["ALLD"].iloc[0], \
        "even at epsilon = 0 the endpoints should not invert"
    assert (zero[zero["lam"] >= 2]["ALLD"].diff().dropna() > -1e-12).all(), \
        "at epsilon = 0 the ALLD share should still rise monotonically once " \
        "lambda is at least 2"
    print(f"  at epsilon = 0 the ALLD share still rises monotonically for "
          f"lambda >= 2 and ends at {zero['ALLD'].iloc[-1]:.6f}")

    a100 = float(t14[(np.isclose(t14["epsilon"], 0.05)) &
                     (t14["lam"] == 100)]["ALLD"].iloc[0])
    assert a100 >= 0.99, f"ALLD at lambda = 100, epsilon = 0.05 is {a100}"
    print(f"  ALLD >= 0.99 by lambda = 100 at epsilon = 0.05: {a100:.6f}")

    flat = t14[t14["lam"] == 0.01][STRATS].to_numpy()
    dev = float(np.max(np.abs(flat - 0.25)))
    assert dev <= 0.05, f"at lambda = 0.01 some share is {dev:.4f} away from 0.25"
    print(f"  at lambda = 0.01 every share is within {dev:.4f} of 0.25, at "
          "every epsilon")

    for col in STRATS:
        assert (t14[col] >= -1e-15).all(), f"negative share in {col}"
    s = t14[STRATS].sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-12), f"shares do not sum to 1: {s.min()}"
    print("  all shares non-negative and summing to 1")

    print(f"\ndone in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
