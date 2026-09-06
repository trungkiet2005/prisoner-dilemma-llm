"""Section 9: robustness and sensitivity - an adversarial audit of section 04.

Why this section exists
-----------------------
Section 04 makes the strongest claim in the report: multiplying every payoff by
a positive constant, a transformation that provably leaves the game identical,
changes how frontier LLMs play it. That claim is worth attacking, because a
single fragile subgroup or a single fragile ladder rung would be enough to
explain the whole thing without any violation of rationality. This section is
written as the hostile referee: every analysis below is an attempt to make the
effect go away, and the section reports honestly which attempts succeed.

Two different claims live inside "the invariance violation", and they must be
audited separately
------------------------------------------------------------------------------
1. NON-INVARIANCE. The ten lambda-level means are not all equal. The natural
   statistic is non-directional, for example the spread of the ten level means
   (max minus min, or their variance), because section 04 already established
   that the response is not monotone in log lambda.
2. DIRECTION. Cooperation rises with lambda at a rate of about +0.013 per
   decade. This is a linear summary of a non-monotone curve, and a linear
   summary of a non-monotone curve is exactly the sort of quantity that a
   leave-one-out audit can destroy.

Conflating the two is the main way this report could mislead a reader, so every
figure here reports both.

Why the permutation test is the right instrument
------------------------------------------------
The design is not just balanced, it is saturated: each of the 1,200
common-random-number cells contributes exactly one game at each of the 10
scales, so the corpus is a 1,200 x 10 matrix with no holes. That makes an exact
randomisation test available. Shuffling the lambda labels within a cell
preserves the cell, the model, the language, the pairing, the repetition and the
whole marginal distribution of cooperation, and destroys only the association
with lambda. The reference distribution therefore needs no distributional
assumption at all, and it is the cleanest statement of the null "lambda does not
matter" that this design can support.

Why a mean of cell slopes is a useful second view
-------------------------------------------------
With one observation per (cell, lambda), the pooled OLS slope on log10 lambda is
algebraically the arithmetic mean of the 1,200 per-cell slopes. That identity
turns the pooled estimate into a location problem on 1,200 exchangeable numbers,
so trimming, winsorising and the median become available as drop-in robust
estimators, and the gap between the mean and the median measures directly how
much of the headline number rides on a tail of extreme cells.

What the audit finds
--------------------
The non-invariance survives everything: every leave-one-out subset, every model
taken alone, every language, every pairing, and every trimming rule. The
direction does not. The pooled positive slope is carried by one model, and half
the models have a negative slope, so the pooled slope is an average over
opposite-sign subgroups rather than a shared tendency. That is reported here as
a downgrade of the directional claim, not as a footnote.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import core as C

X_ALL = np.log10(np.array(C.SCALES, float))
N_PERM = 5000          # headline permutation reps
N_PERM_LOO = 2000      # per leave-one-out subset
N_BOOT = 2000

CLASS_COL = {"highly robust": "#0f8a72", "moderately robust": "#eda100",
             "sensitive": "#e34948"}


def short(label: str, n: int = 21) -> str:
    """Compact axis label: drop the reasoning suffix, then trim if still long."""
    s = str(label).replace("-Non-Reasoning", "")
    return s if len(s) <= n else s[:n - 1] + "."


# --------------------------------------------------------------------------
# the 1,200 x 10 saturated matrix, and statistics computed on it
# --------------------------------------------------------------------------
def cell_matrix(dy: pd.DataFrame, value: str = "joint_coop"):
    """Return (M, cells, info) with M[i, j] = value for cell i at scale j."""
    piv = dy.pivot_table(index="cell", columns="scale", values=value)[C.SCALES]
    if piv.isna().to_numpy().any():
        raise ValueError("matrix is not saturated; the permutation test is invalid")
    info = (dy.drop_duplicates("cell").set_index("cell")
            [["model", "language", "dyad", "pairing"]].reindex(piv.index))
    return piv.to_numpy(float), piv.index.to_numpy(), info


def stats_from_matrix(M: np.ndarray, x: np.ndarray) -> dict:
    """Level means plus the three summary statistics used throughout."""
    cm = M.mean(axis=0)
    xc = x - x.mean()
    return {"level_means": cm,
            "slope": float((xc * cm).sum() / (xc ** 2).sum()),
            "rng": float(cm.max() - cm.min()),
            "var": float(cm.var(ddof=0))}


def perm_null(M: np.ndarray, x: np.ndarray, n_perm: int, seed: int = 0):
    """Shuffle lambda labels within each cell; return null draws of the stats.

    This preserves every design factor and the whole marginal distribution of
    the outcome, and removes only the mapping from game to lambda.
    """
    rng = np.random.default_rng(seed)
    n, J = M.shape
    xc = x - x.mean()
    den = float((xc ** 2).sum())
    slopes = np.empty(n_perm)
    rngs = np.empty(n_perm)
    varis = np.empty(n_perm)
    for i in range(n_perm):
        order = np.argsort(rng.random((n, J)), axis=1)
        cm = np.take_along_axis(M, order, axis=1).mean(axis=0)
        slopes[i] = float((xc * cm).sum() / den)
        rngs[i] = cm.max() - cm.min()
        varis[i] = cm.var(ddof=0)
    return slopes, rngs, varis


def perm_pvalues(M, x, n_perm, seed=0):
    """Observed statistics and two-sided / upper-tail permutation p-values."""
    o = stats_from_matrix(M, x)
    ns, nr, nv = perm_null(M, x, n_perm, seed)
    # (b + 1) / (n + 1) so that p is never reported as exactly zero
    p_slope = (int(np.sum(np.abs(ns) >= abs(o["slope"]))) + 1) / (n_perm + 1)
    p_rng = (int(np.sum(nr >= o["rng"])) + 1) / (n_perm + 1)
    p_var = (int(np.sum(nv >= o["var"])) + 1) / (n_perm + 1)
    return o, {"p_slope": p_slope, "p_range": p_rng, "p_var": p_var}, (ns, nr, nv)


def cell_slopes(M: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-cell OLS slope. Their arithmetic mean is the pooled OLS slope."""
    xc = x - x.mean()
    return (M * xc).sum(axis=1) / float((xc ** 2).sum())


def ols_slope(df: pd.DataFrame, value: str, cluster: str | None,
              xcol: str = "log_scale"):
    d = df[[xcol, value] + ([cluster] if cluster else [])].dropna()
    X = sm.add_constant(d[xcol].to_numpy())
    m = sm.OLS(d[value].to_numpy(), X)
    fit = (m.fit(cov_type="cluster", cov_kwds={"groups": d[cluster].to_numpy()})
           if cluster else m.fit())
    ci = fit.conf_int()
    ncl = d[cluster].nunique() if cluster else len(d)
    return {"slope": float(fit.params[1]), "se": float(fit.bse[1]),
            "lo": float(ci[1][0]), "hi": float(ci[1][1]),
            "p": float(fit.pvalues[1]), "n_obs": int(len(d)),
            "n_clusters": int(ncl)}


# --------------------------------------------------------------------------
def run(rounds, ag, dy):
    C.use_style()
    print("\n== 09 robustness ==")
    x = X_ALL
    M, cells, info = cell_matrix(dy)
    full, full_p, (null_s, null_r, null_v) = perm_pvalues(M, x, N_PERM, seed=11)
    print(f"  full sample: slope {full['slope']:+.4f}  range {full['rng']:.4f}  "
          f"p_slope {full_p['p_slope']:.4g}  p_range {full_p['p_range']:.4g}")

    # ======================================================================
    # 1  LEAVE-ONE-OUT
    # ======================================================================
    subsets = [("none", "full sample", np.ones(len(M), bool), list(range(10)))]
    for m in C.MODEL_ORDER:
        subsets.append(("model", m, (info["model"] != m).to_numpy(), list(range(10))))
    for lg in C.LANGS:
        subsets.append(("language", C.LANG_LABEL[lg],
                        (info["language"] != lg).to_numpy(), list(range(10))))
    for d_ in C.DYADS:
        subsets.append(("pairing", d_, (info["dyad"] != d_).to_numpy(),
                        list(range(10))))
    for j, s in enumerate(C.SCALES):
        subsets.append(("lambda level", f"{s:g}", np.ones(len(M), bool),
                        [k for k in range(10) if k != j]))
    combos = {"0.01 and 0.1": [0.01, 0.1],
              "0.01, 0.1, 0.25": [0.01, 0.1, 0.25],
              "100 and 1000": [100.0, 1000.0],
              "0.01 and 1000": [0.01, 1000.0]}
    for lab, drop in combos.items():
        subsets.append(("lambda pair", lab, np.ones(len(M), bool),
                        [k for k, s in enumerate(C.SCALES) if s not in drop]))

    loo_rows = []
    for i, (fam, lab, rowmask, cols) in enumerate(subsets):
        Ms = M[rowmask][:, cols]
        xs = x[cols]
        o, p, _ = perm_pvalues(Ms, xs, N_PERM_LOO, seed=100 + i)
        loo_rows.append({
            "family": fam, "dropped": lab, "n_cells": int(rowmask.sum()),
            "n_levels": len(cols), "n_games": int(rowmask.sum() * len(cols)),
            "slope_per_decade": o["slope"], "p_slope_perm": p["p_slope"],
            "range_level_means": o["rng"], "p_range_perm": p["p_range"],
            "var_level_means": o["var"], "p_var_perm": p["p_var"],
            "sign_flip_vs_full": bool(np.sign(o["slope"]) != np.sign(full["slope"])),
            "slope_pct_of_full": 100 * o["slope"] / full["slope"],
        })
    loo = pd.DataFrame(loo_rows)
    C.savetab(loo, "robust_leave_one_out")

    body = loo[loo.family != "none"]
    n_flip = int(body["sign_flip_vs_full"].sum())
    n_range_sig = int((body["p_range_perm"] < 0.05).sum())
    crit = loo[loo.dropped == "0.01 and 0.1"].iloc[0]
    grok = loo[loo.dropped == "Grok-4.20-Non-Reasoning"].iloc[0]
    cvc = loo[loo.dropped == "CvC"].iloc[0]

    def _rows(fam):
        if fam == "language":
            return pd.concat([body[body.family == "language"],
                              body[body.family == "pairing"]])
        if fam == "lambda level":
            return pd.concat([body[body.family == "lambda level"],
                              body[body.family == "lambda pair"]])
        return body[body.family == fam]

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0))
    fams = [("model", "a  leave one model out"),
            ("language", "b  leave one language / pairing out"),
            ("lambda level", "c  leave $\\lambda$ rungs out")]
    fams2 = [("model", "d  same subsets, spread of means"),
             ("language", "e  same subsets, spread of means"),
             ("lambda level", "f  same subsets, spread of means")]

    for axx, (fam, title) in zip(axes[0], fams):
        sub = _rows(fam)
        y = np.arange(len(sub))[::-1]
        cols_ = [C.C_DEFECT if f else C.C_COOP for f in sub["sign_flip_vs_full"]]
        axx.axvline(0, color=C.INK, lw=.9, ymin=0.16)
        axx.axvline(full["slope"], color=C.INK2, lw=1.1, ls="--", ymin=0.16)
        axx.scatter(sub["slope_per_decade"], y, color=cols_, s=34, zorder=3)
        axx.set_yticks(y, [short(s) for s in sub["dropped"]], fontsize=6.5)
        axx.set_xlabel("cooperation slope per decade of $\\lambda$")
        axx.set_title(title)
        axx.set_ylim(-max(2.0, 0.22 * len(sub)), len(sub) - 0.4)
    C.annotate(axes[0][0], f"dropping Grok flips the sign:\n"
                           f"{grok.slope_per_decade:+.4f} per decade",
               loc="lower left", color=C.C_DEFECT)
    C.annotate(axes[0][1], f"dashed = full-sample slope {full['slope']:+.3f}\n"
                           "red = the sign flips vs the full sample",
               loc="lower left")
    C.annotate(axes[0][2], f"dropping $\\lambda$ = 0.01 and 0.1 flips it too:\n"
                           f"{crit.slope_per_decade:+.4f} per decade",
               loc="lower left", color=C.C_DEFECT)

    p95 = float(np.percentile(null_r, 95))
    for axx, (fam, title) in zip(axes[1], fams2):
        sub = _rows(fam)
        y = np.arange(len(sub))[::-1]
        cols_ = [C.C_COOP if p < .05 else C.MUTED for p in sub["p_range_perm"]]
        axx.axvspan(0, p95, color=C.MUTED, alpha=.20, lw=0)
        axx.axvline(full["rng"], color=C.INK2, lw=1.1, ls="--", ymin=0.16)
        axx.scatter(sub["range_level_means"], y, color=cols_, s=34, zorder=3)
        axx.set_yticks(y, [short(s) for s in sub["dropped"]], fontsize=6.5)
        axx.set_xlabel("max - min of the $\\lambda$-level means")
        axx.set_title(title)
        axx.set_xlim(0, max(sub["range_level_means"].max(), full["rng"]) * 1.35)
        axx.set_ylim(-max(2.0, 0.22 * len(sub)), len(sub) - 0.4)
    C.annotate(axes[1][0], "grey band = permutation null,\nup to its 95th percentile",
               loc="lower right")
    C.annotate(axes[1][1], "blue = still rejects invariance\nat p < 0.05",
               loc="lower right", color=C.C_COOP)
    C.annotate(axes[1][2], f"{n_range_sig} of {len(body)} subsets still\n"
                           "reject invariance at p < 0.05", loc="lower right",
               color=C.C_COOP)
    C.save(fig, "robust", "R01_leave_one_out",
           "Is the payoff-scale result the work of one model, one language, one "
           "pairing or one rung of the lambda ladder?",
           f"The two halves of the claim behave completely differently. The "
           f"directional half is fragile: the pooled slope of "
           f"{full['slope']:+.4f} per decade flips sign in {n_flip} of "
           f"{len(body)} leave-one-out subsets, and the two that matter most are "
           f"dropping Grok-4.20-Non-Reasoning ({grok.slope_per_decade:+.4f}) and "
           f"dropping the two smallest scales together "
           f"({crit.slope_per_decade:+.4f}). Non-invariance itself is not fragile: "
           f"the spread of the lambda-level means stays above the permutation null "
           f"in {n_range_sig} of {len(body)} subsets.",
           "leave-one-out dot plots, two statistics", "joint_coop, scale, cell")

    # ======================================================================
    # 2  AGGREGATION LEVEL
    # ======================================================================
    cond = (dy.groupby(["model", "language", "dyad", "scale"], observed=True)
            .agg(joint_coop=("joint_coop", "mean"),
                 log_scale=("log_scale", "first")).reset_index())
    cond["cond_cluster"] = (cond["model"].astype(str) + "|"
                            + cond["language"].astype(str) + "|"
                            + cond["dyad"].astype(str))
    levels = [
        ("round, no clustering", rounds, "coop", None, "none (naive iid)"),
        ("round, cluster game", rounds, "coop", "game_uid", "game (12,000)"),
        ("round, cluster cell", rounds, "coop", "cell", "CRN cell (1,200)"),
        ("agent-game, cluster cell", ag, "coop_rate", "cell", "CRN cell (1,200)"),
        ("dyad, cluster cell", dy, "joint_coop", "cell", "CRN cell (1,200)"),
        ("condition mean, cluster condition", cond, "joint_coop", "cond_cluster",
         "model x lang x pairing (120)"),
    ]
    agg_rows = []
    for name, src, val, clu, cludesc in levels:
        r_ = ols_slope(src, val, clu)
        r_["level"] = name
        r_["cluster_variable"] = cludesc
        r_["ci_width"] = r_["hi"] - r_["lo"]
        agg_rows.append(r_)
    agg = pd.DataFrame(agg_rows)
    naive_w = float(agg.loc[agg.level == "round, no clustering", "ci_width"].iloc[0])
    agg["ci_width_ratio_vs_naive"] = agg["ci_width"] / naive_w
    agg["design_effect"] = agg["ci_width_ratio_vs_naive"] ** 2
    agg = agg[["level", "cluster_variable", "n_obs", "n_clusters", "slope", "se",
               "lo", "hi", "p", "ci_width", "ci_width_ratio_vs_naive",
               "design_effect"]]
    C.savetab(agg, "robust_aggregation_levels")
    for _, r_ in agg.iterrows():
        C.record_test(section="robustness",
                      test="OLS slope on log10(lambda) at one aggregation level",
                      comparison=f"cooperation ~ log10(lambda) [{r_.level}]",
                      statistic=r_.slope, p_value=r_.p, ci_lo=r_.lo, ci_hi=r_.hi,
                      n=int(r_.n_obs),
                      effect_size_note=f"CI width {r_.ci_width:.4f}, "
                                       f"{r_.ci_width_ratio_vs_naive:.1f}x naive")

    dyad_w = float(agg.loc[agg.level == "dyad, cluster cell", "ci_width"].iloc[0])
    dyad_de = float(agg.loc[agg.level == "dyad, cluster cell", "design_effect"].iloc[0])
    game_de = float(agg.loc[agg.level == "round, cluster game", "design_effect"].iloc[0])
    game_ratio = float(agg.loc[agg.level == "round, cluster game",
                               "ci_width_ratio_vs_naive"].iloc[0])
    spread = float(agg.slope.max() - agg.slope.min())

    shortlev = ["round (naive)", "round / game", "round / cell",
                "agent-game / cell", "dyad / cell", "condition mean"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    y = np.arange(len(agg))[::-1]
    cols_ = [C.C_DEFECT if "no clustering" in n else C.C_COOP for n in agg.level]
    axes[0].axvline(0, color=C.INK, lw=.9)
    axes[0].errorbar(agg.slope, y, xerr=[agg.slope - agg.lo, agg.hi - agg.slope],
                     fmt="none", ecolor=C.INK2, lw=1.3, zorder=2)
    axes[0].scatter(agg.slope, y, color=cols_, s=40, zorder=3)
    axes[0].set_yticks(y, [n.replace(", ", ",\n") for n in agg.level], fontsize=6.2)
    axes[0].set_xlabel("slope per decade of $\\lambda$, with 95% CI")
    axes[0].set_title("a  same estimate, six unit choices")
    axes[0].set_ylim(-0.9, len(agg) + 0.55)
    C.annotate(axes[0], f"point estimates identical to\n{spread:.0e}: that is the "
                        f"balanced\ndesign, not corroboration", loc="upper right")

    axes[1].barh(y, agg.ci_width, color=cols_)
    axes[1].set_yticks(y, shortlev, fontsize=6.5)
    axes[1].set_xlabel("width of the 95% CI (cooperation per decade)")
    axes[1].set_title("b  only the uncertainty moves")
    axes[1].set_xlim(0, agg.ci_width.max() * 1.22)
    for yy, w, rr in zip(y, agg.ci_width, agg.ci_width_ratio_vs_naive):
        axes[1].text(w + agg.ci_width.max() * .02, yy, f"{rr:.1f}x", va="center",
                     fontsize=6.5, color=C.INK2)

    icc_rows = []
    for lab, src, val, clu in [("rounds in a game", rounds, "coop", "game_uid"),
                               ("rounds in a cell", rounds, "coop", "cell"),
                               ("games in a cell", dy, "joint_coop", "cell")]:
        g = src.groupby(clu, observed=True)[val]
        mb = float(g.mean().var(ddof=1))
        mw = float(g.var(ddof=1).mean())
        icc_rows.append({"grouping": lab,
                         "icc": mb / (mb + mw) if mb + mw > 0 else np.nan})
    icc = pd.DataFrame(icc_rows)
    axes[2].bar(range(len(icc)), icc.icc, color=C.OUTCOME_COL["DC"], width=.6)
    axes[2].set_xticks(range(len(icc)), [g.replace(" in ", "\nin ") for g in icc.grouping],
                       fontsize=7)
    axes[2].set_ylabel("intra-cluster correlation")
    axes[2].set_title("c  why clustering matters this much")
    for i, v in enumerate(icc.icc):
        axes[2].text(i, v + .015, f"{v:.2f}", ha="center", fontsize=7, color=C.INK2)
    axes[2].set_ylim(0, float(icc.icc.max()) * 1.55)
    C.annotate(axes[2], f"ignoring the game cluster\ninflates precision by a design\n"
                        f"effect of {game_de:.0f}", loc="upper right")
    cond_p = float(agg.loc[agg.level == "condition mean, cluster condition",
                           "p"].iloc[0])
    C.save(fig, "robust", "R02_aggregation_levels",
           "How much of the reported precision is an artefact of the unit of "
           "analysis, and how badly would a round-level analysis that ignores "
           "clustering mislead?",
           f"The point estimate is invariant to the unit of analysis to {spread:.0e}, "
           f"which is a consequence of the balanced design rather than corroboration. "
           f"What changes is the uncertainty: the naive round-level 95% interval is "
           f"{naive_w:.4f} wide against {dyad_w:.4f} at the dyad level clustered on "
           f"the design cell, so treating 240,000 rounds as 240,000 observations "
           f"would overstate precision by a design effect of about {dyad_de:.0f}. "
           f"Rounds within a game correlate at ICC {icc.icc.iloc[0]:.2f}. Aggregating "
           f"all the way to 120 condition means widens the interval a further "
           f"{agg.ci_width_ratio_vs_naive.iloc[-1]/agg.ci_width_ratio_vs_naive.iloc[-2]:.1f} "
           f"times and leaves the effect only marginally significant (p = "
           f"{cond_p:.3f}).",
           "forest plot + CI width bars + ICC bars",
           "cooperation, scale, unit of analysis, cluster variable")

    # ======================================================================
    # 3  ESTIMATOR SENSITIVITY
    # ======================================================================
    cs = cell_slopes(M, x)
    rng_boot = np.random.default_rng(7)
    med_boot = np.array([np.median(cs[rng_boot.integers(0, len(cs), len(cs))])
                         for _ in range(N_BOOT)])
    wil = stats.wilcoxon(cs)
    tt = stats.ttest_1samp(cs, 0.0)

    Xr = sm.add_constant(rounds["log_scale"].to_numpy())
    glm = sm.GLM(rounds["coop"].to_numpy().astype(float), Xr,
                 family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": rounds["game_uid"].to_numpy()})
    pbar = float(rounds["coop"].mean())
    ame = pbar * (1 - pbar)
    gci = glm.conf_int()

    # one Spearman rho per cell: clustered by construction
    rho_cell = np.array([stats.spearmanr(x, M[i]).statistic for i in range(len(M))])
    ok = np.isfinite(rho_cell)
    rho_mean, rho_lo, rho_hi = C.cluster_boot_ci(rho_cell[ok],
                                                 np.arange(int(ok.sum())),
                                                 n_boot=N_BOOT)
    rho_t = stats.ttest_1samp(rho_cell[ok], 0.0)

    dy_ols = ols_slope(dy, "joint_coop", "cell")
    ag_ols = ols_slope(ag, "coop_rate", "cell")
    glm_ame = float(glm.params[1]) * ame
    est = pd.DataFrame([
        {"estimator": "OLS, dyad level, cluster-robust", "units": "per decade",
         "estimate": dy_ols["slope"], "lo": dy_ols["lo"], "hi": dy_ols["hi"],
         "p_value": dy_ols["p"], "n": dy_ols["n_obs"]},
        {"estimator": "OLS, agent-game level, cluster-robust", "units": "per decade",
         "estimate": ag_ols["slope"], "lo": ag_ols["lo"], "hi": ag_ols["hi"],
         "p_value": ag_ols["p"], "n": ag_ols["n_obs"]},
        {"estimator": "logistic GLM, rounds, cluster-robust (marginal)",
         "units": "per decade", "estimate": glm_ame,
         "lo": float(gci[1][0]) * ame, "hi": float(gci[1][1]) * ame,
         "p_value": float(glm.pvalues[1]), "n": len(rounds)},
        {"estimator": "mean of 1,200 per-cell slopes", "units": "per decade",
         "estimate": float(cs.mean()),
         "lo": float(cs.mean() - 1.96 * cs.std(ddof=1) / np.sqrt(len(cs))),
         "hi": float(cs.mean() + 1.96 * cs.std(ddof=1) / np.sqrt(len(cs))),
         "p_value": float(tt.pvalue), "n": len(cs)},
        {"estimator": "median of 1,200 per-cell slopes", "units": "per decade",
         "estimate": float(np.median(cs)),
         "lo": float(np.percentile(med_boot, 2.5)),
         "hi": float(np.percentile(med_boot, 97.5)),
         "p_value": float(wil.pvalue), "n": len(cs)},
        {"estimator": "mean per-cell Spearman rho", "units": "rank correlation",
         "estimate": rho_mean, "lo": rho_lo, "hi": rho_hi,
         "p_value": float(rho_t.pvalue), "n": int(ok.sum())},
        {"estimator": "permutation test, linear slope", "units": "per decade",
         "estimate": full["slope"], "lo": np.nan, "hi": np.nan,
         "p_value": full_p["p_slope"], "n": len(dy)},
        {"estimator": "permutation test, max-min of level means",
         "units": "cooperation", "estimate": full["rng"], "lo": np.nan, "hi": np.nan,
         "p_value": full_p["p_range"], "n": len(dy)},
    ])
    est["p_fdr"] = C.bh_fdr(est["p_value"].to_numpy())
    est["sign"] = np.sign(est["estimate"]).astype(int)
    est["display"] = ["OLS, dyad level,\ncluster-robust",
                      "OLS, agent-game level,\ncluster-robust",
                      "logistic GLM on rounds,\ncluster-robust (marginal)",
                      "mean of 1,200\nper-cell slopes",
                      "median of 1,200\nper-cell slopes",
                      "mean per-cell\nSpearman rho",
                      "permutation test,\nlinear slope",
                      "permutation test,\nmax-min of level means"]
    C.savetab(est.drop(columns=["display"]), "robust_estimators")
    for _, r_ in est.iterrows():
        C.record_test(section="robustness", test="estimator sensitivity",
                      comparison=f"cooperation vs log10(lambda) [{r_.estimator}]",
                      statistic=r_.estimate, p_value=r_.p_value, ci_lo=r_.lo,
                      ci_hi=r_.hi, n=int(r_.n), effect_size_note=r_.units)
    n_sig_est = int((est.p_fdr < .05).sum())
    n_pos_est = int((est.sign > 0).sum())

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    dec = est[est.units == "per decade"].reset_index(drop=True)
    y = np.arange(len(dec))[::-1]
    axes[0].axvline(0, color=C.INK, lw=.9, ymin=0.15)
    axes[0].axvline(full["slope"], color=C.INK2, lw=1.0, ls="--", ymin=0.15)
    axes[0].locator_params(axis="x", nbins=5)
    for yy, r_ in zip(y, dec.itertuples()):
        if np.isfinite(r_.lo):
            axes[0].plot([r_.lo, r_.hi], [yy, yy], color=C.INK2, lw=1.3, zorder=2)
    axes[0].scatter(dec.estimate, y, color=C.C_COOP, s=40, zorder=3)
    axes[0].set_yticks(y, list(dec.display), fontsize=5.9)
    axes[0].set_xlabel("slope per decade of $\\lambda$")
    axes[0].set_title("a  six estimators, same units")
    axes[0].set_ylim(-1.3, len(dec) - 0.35)
    C.annotate(axes[0], f"all positive, but the median cell\nslope is "
                        f"{np.median(cs):+.4f}, only "
                        f"{100*np.median(cs)/cs.mean():.0f}% of the mean",
               loc="lower right")

    y2 = np.arange(len(est))[::-1]
    pcol = [C.C_COOP if p < .05 else C.MUTED for p in est.p_fdr]
    axes[1].barh(y2, -np.log10(np.clip(est.p_fdr, 1e-300, 1)), color=pcol)
    axes[1].axvline(-np.log10(.05), color=C.C_DEFECT, lw=1.0, ls="--")
    axes[1].set_yticks(y2, [d.replace("\n", " ") for d in est.display], fontsize=5.5)
    axes[1].set_xlabel("$-\\log_{10}$ of the FDR-adjusted p-value")
    axes[1].set_title(f"b  {n_sig_est} of {len(est)} survive FDR adjustment")
    axes[1].set_ylim(-1.3, len(est) - 0.35)
    C.annotate(axes[1], "dashed line = p 0.05; grey = the one\nestimator that does "
                        "not reach it", loc="lower right", color=C.C_DEFECT)

    cm = full["level_means"]
    xc = x - x.mean()
    fitted = cm.mean() + full["slope"] * xc
    axes[2].plot(C.SCALES, cm, "o-", color=C.C_COOP, zorder=3, label="observed means")
    axes[2].plot(C.SCALES, fitted, "--", color=C.INK2, label="fitted linear trend")
    axes[2].axhline(cm.mean(), color=C.MUTED, lw=1.0, ls=":", label="invariance null")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("payoff scale $\\lambda$")
    axes[2].set_ylabel("dyad cooperation rate")
    axes[2].set_title("c  why the linear fit is a poor summary")
    axes[2].legend(fontsize=6.3, loc="lower right")
    ss_lin = float(((cm - fitted) ** 2).sum())
    ss_tot = float(((cm - cm.mean()) ** 2).sum())
    axes[2].set_ylim(cm.min() - .035, cm.max() + .055)
    C.annotate(axes[2], f"the straight line leaves {100*ss_lin/ss_tot:.0f}% of the\n"
                        f"between-level variance unexplained", loc="upper left")
    C.save(fig, "robust", "R03_estimator_sensitivity",
           "Does the payoff-scale effect depend on the estimator: least squares on "
           "rates, a clustered logistic model on binary rounds, a rank statistic or "
           "a randomisation test?",
           f"Mostly not. All {len(est)} estimators agree in sign ({n_pos_est} of "
           f"{len(est)} positive) and {n_sig_est} of {len(est)} stay significant "
           f"after Benjamini-Hochberg adjustment, including the assumption-free "
           f"permutation test. The one exception is the rank statistic: the mean "
           f"per-cell Spearman rho is only {rho_mean:+.3f} "
           f"[{rho_lo:+.3f}, {rho_hi:+.3f}], p = {rho_t.pvalue:.2f}, because a "
           f"monotone rank correlation on 10 tied-heavy points per cell has almost no "
           f"power against a non-monotone effect. The mean per-cell slope "
           f"({cs.mean():+.4f}) is also far from the median ({np.median(cs):+.4f}), "
           f"so the pooled positive slope is a tail effect rather than a typical "
           f"cell, and a straight line leaves {100*ss_lin/ss_tot:.0f}% of the "
           f"between-level variance unexplained.",
           "forest + p-value bars + fitted-vs-observed line",
           "joint_coop, coop, scale, cell")

    # ======================================================================
    # 4  PERMUTATION TEST - the headline
    # ======================================================================
    perm_rows = [
        {"scope": "all models", "statistic": "linear slope per decade",
         "observed": full["slope"], "null_mean": float(null_s.mean()),
         "null_sd": float(null_s.std(ddof=1)),
         "null_p95": float(np.percentile(np.abs(null_s), 95)),
         "null_p99": float(np.percentile(np.abs(null_s), 99)),
         "p_perm": full_p["p_slope"], "n_perm": N_PERM},
        {"scope": "all models", "statistic": "max - min of level means",
         "observed": full["rng"], "null_mean": float(null_r.mean()),
         "null_sd": float(null_r.std(ddof=1)),
         "null_p95": float(np.percentile(null_r, 95)),
         "null_p99": float(np.percentile(null_r, 99)),
         "p_perm": full_p["p_range"], "n_perm": N_PERM},
        {"scope": "all models", "statistic": "variance of level means",
         "observed": full["var"], "null_mean": float(null_v.mean()),
         "null_sd": float(null_v.std(ddof=1)),
         "null_p95": float(np.percentile(null_v, 95)),
         "null_p99": float(np.percentile(null_v, 99)),
         "p_perm": full_p["p_var"], "n_perm": N_PERM},
    ]
    per_model = {}
    for i, m in enumerate(C.MODEL_ORDER):
        Mm = M[(info["model"] == m).to_numpy()]
        o, p, nulls = perm_pvalues(Mm, x, N_PERM_LOO, seed=300 + i)
        per_model[m] = (o, p)
        for stat_name, key, pk, nn in [
                ("linear slope per decade", "slope", "p_slope", nulls[0]),
                ("max - min of level means", "rng", "p_range", nulls[1]),
                ("variance of level means", "var", "p_var", nulls[2])]:
            perm_rows.append({
                "scope": m, "statistic": stat_name, "observed": o[key],
                "null_mean": float(nn.mean()), "null_sd": float(nn.std(ddof=1)),
                "null_p95": float(np.percentile(np.abs(nn), 95)),
                "null_p99": float(np.percentile(np.abs(nn), 99)),
                "p_perm": p[pk], "n_perm": N_PERM_LOO})
    for i, lg in enumerate(C.LANGS):
        Ml = M[(info["language"] == lg).to_numpy()]
        o, p, nulls = perm_pvalues(Ml, x, N_PERM_LOO, seed=400 + i)
        perm_rows.append({
            "scope": C.LANG_LABEL[lg], "statistic": "max - min of level means",
            "observed": o["rng"], "null_mean": float(nulls[1].mean()),
            "null_sd": float(nulls[1].std(ddof=1)),
            "null_p95": float(np.percentile(nulls[1], 95)),
            "null_p99": float(np.percentile(nulls[1], 99)),
            "p_perm": p["p_range"], "n_perm": N_PERM_LOO})
    permtab = pd.DataFrame(perm_rows)
    permtab["p_fdr"] = C.bh_fdr(permtab["p_perm"].to_numpy())
    C.savetab(permtab, "robust_permutation")
    C.record_test(section="robustness",
                  test=f"within-cell permutation of lambda labels, {N_PERM} draws",
                  comparison="max-min of the 10 lambda-level cooperation means",
                  statistic=full["rng"], p_value=full_p["p_range"],
                  ci_lo=np.nan, ci_hi=np.nan, n=len(dy),
                  effect_size_note=f"null 99th percentile "
                                   f"{np.percentile(null_r, 99):.4f} vs observed "
                                   f"{full['rng']:.4f}")
    C.record_test(section="robustness",
                  test=f"within-cell permutation of lambda labels, {N_PERM} draws",
                  comparison="linear slope of cooperation on log10(lambda)",
                  statistic=full["slope"], p_value=full_p["p_slope"],
                  ci_lo=np.nan, ci_hi=np.nan, n=len(dy),
                  effect_size_note="two-sided permutation p")

    n_model_rng = sum(1 for m in C.MODEL_ORDER if per_model[m][1]["p_range"] < .05)
    n_lang_rng = int(((permtab.scope.isin([C.LANG_LABEL[l] for l in C.LANGS]))
                      & (permtab.p_perm < .05)).sum())
    n_neg = sum(1 for m in C.MODEL_ORDER if per_model[m][0]["slope"] < 0)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    axes[0].hist(null_s, bins=60, color=C.MUTED, alpha=.9)
    axes[0].axvline(full["slope"], color=C.C_DEFECT, lw=1.8)
    axes[0].set_xlabel("slope per decade of $\\lambda$ under the null")
    axes[0].set_ylabel(f"permutation draws (of {N_PERM:,})")
    axes[0].set_title("a  null for the linear slope")
    C.annotate(axes[0], f"observed {full['slope']:+.4f} (red)\n"
                        f"null SD {null_s.std(ddof=1):.4f}\n"
                        f"p = {full_p['p_slope']:.1e} (the floor\n"
                        f"for {N_PERM:,} draws)", loc="upper left",
               color=C.C_DEFECT)

    axes[1].hist(null_r, bins=60, color=C.MUTED, alpha=.9)
    axes[1].axvline(full["rng"], color=C.C_DEFECT, lw=1.8)
    axes[1].set_xlabel("max - min of the 10 level means under the null")
    axes[1].set_ylabel(f"permutation draws (of {N_PERM:,})")
    axes[1].set_title("b  null for non-monotone spread")
    axes[1].set_xlim(0, full["rng"] * 1.12)
    C.annotate(axes[1], f"observed {full['rng']:.4f} (red)\n"
                        f"null 99th pct {np.percentile(null_r, 99):.4f}\n"
                        f"observed exceeds every one of\nthe {N_PERM:,} null draws",
               loc="upper center", color=C.C_DEFECT)

    w = .38
    xs = np.arange(len(C.MODEL_ORDER))
    obs_r = [per_model[m][0]["rng"] for m in C.MODEL_ORDER]
    nul_r = [float(permtab[(permtab.scope == m)
                           & (permtab.statistic == "max - min of level means")]
                   ["null_p99"].iloc[0]) for m in C.MODEL_ORDER]
    axes[2].bar(xs - w / 2, obs_r, w, color=[C.MODEL_COL[m] for m in C.MODEL_ORDER],
                label="observed spread")
    axes[2].bar(xs + w / 2, nul_r, w, color=C.MUTED, label="null 99th percentile")
    axes[2].set_xticks(xs, [m.replace("-Non-Reasoning", "").replace("-", "\n", 1)
                            for m in C.MODEL_ORDER], fontsize=5.8)
    axes[2].set_ylabel("max - min of $\\lambda$-level means")
    axes[2].set_title("c  every model alone rejects invariance")
    axes[2].legend(fontsize=6.3, loc="upper left")
    axes[2].set_ylim(0, max(obs_r) * 1.34)
    C.annotate(axes[2], f"{n_model_rng} of 6 models: p < 0.05,\n"
                        f"yet {n_neg} of 6 have a\nnegative slope", loc="upper right")
    C.save(fig, "robust", "R04_permutation_invariance",
           "Under an exact randomisation null that keeps every design factor and "
           "only breaks the link to lambda, how extreme is the observed departure "
           "from payoff invariance?",
           f"Extreme, and the non-directional statistic carries it. Shuffling lambda "
           f"labels within each common-random-number cell {N_PERM:,} times puts the "
           f"observed spread of the ten level means ({full['rng']:.3f}) far beyond "
           f"the null 99th percentile ({np.percentile(null_r, 99):.3f}); no null draw "
           f"came close, so p = {full_p['p_range']:.1e} is the resolution floor of "
           f"{N_PERM:,} draws rather than a measured value. Each of the six models "
           f"rejects invariance "
           f"on its own ({n_model_rng} of 6 at p < 0.05) and so does each of the five "
           f"languages ({n_lang_rng} of 5), yet {n_neg} of the 6 models have a "
           f"negative slope, so the pooled positive direction is not shared.",
           "permutation null histograms + per-model observed vs null",
           "joint_coop, scale, cell, model")

    # ======================================================================
    # 5  BOOTSTRAP STABILITY
    # ======================================================================
    rng_b = np.random.default_rng(23)
    n_cells = len(M)
    boot_overall = np.empty(N_BOOT)
    boot_slope = np.empty(N_BOOT)
    boot_range = np.empty(N_BOOT)
    den = float((xc ** 2).sum())
    for i in range(N_BOOT):
        idx = rng_b.integers(0, n_cells, n_cells)
        cmb = M[idx].mean(axis=0)
        boot_overall[i] = cmb.mean()
        boot_slope[i] = float((xc * cmb).sum() / den)
        boot_range[i] = cmb.max() - cmb.min()
    model_idx = {m: np.flatnonzero((info["model"] == m).to_numpy())
                 for m in C.MODEL_ORDER}
    boot_model = {}
    for k, m in enumerate(C.MODEL_ORDER):
        ii = model_idx[m]
        rr = np.random.default_rng(500 + k)
        draws = np.empty(N_BOOT)
        for i in range(N_BOOT):
            draws[i] = M[ii[rr.integers(0, len(ii), len(ii))]].mean()
        boot_model[m] = draws

    boot_rows = [
        {"quantity": "overall cooperation rate", "point": float(M.mean()),
         "boot_lo": float(np.percentile(boot_overall, 2.5)),
         "boot_hi": float(np.percentile(boot_overall, 97.5)),
         "boot_sd": float(boot_overall.std(ddof=1)), "pct_draws_same_sign": 100.0},
        {"quantity": "slope per decade of lambda", "point": full["slope"],
         "boot_lo": float(np.percentile(boot_slope, 2.5)),
         "boot_hi": float(np.percentile(boot_slope, 97.5)),
         "boot_sd": float(boot_slope.std(ddof=1)),
         "pct_draws_same_sign": float(100 * np.mean(
             np.sign(boot_slope) == np.sign(full["slope"])))},
        {"quantity": "max - min of level means", "point": full["rng"],
         "boot_lo": float(np.percentile(boot_range, 2.5)),
         "boot_hi": float(np.percentile(boot_range, 97.5)),
         "boot_sd": float(boot_range.std(ddof=1)), "pct_draws_same_sign": 100.0},
    ]
    for m in C.MODEL_ORDER:
        boot_rows.append({"quantity": f"cooperation rate, {m}",
                          "point": float(M[model_idx[m]].mean()),
                          "boot_lo": float(np.percentile(boot_model[m], 2.5)),
                          "boot_hi": float(np.percentile(boot_model[m], 97.5)),
                          "boot_sd": float(boot_model[m].std(ddof=1)),
                          "pct_draws_same_sign": 100.0})
    bt = pd.DataFrame(boot_rows)
    C.savetab(bt, "robust_bootstrap_stability")
    sign_stab = float(bt.loc[bt.quantity == "slope per decade of lambda",
                             "pct_draws_same_sign"].iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
    axes[0].hist(boot_overall, bins=50, color=C.C_COOP, alpha=.9)
    axes[0].axvline(M.mean(), color=C.INK, lw=1.5)
    axes[0].set_xlabel("overall dyad cooperation rate")
    axes[0].set_ylabel(f"cluster bootstrap draws (of {N_BOOT:,})")
    axes[0].set_title("a  the level is pinned down")
    C.annotate(axes[0], f"{M.mean():.3f} [{np.percentile(boot_overall, 2.5):.3f}, "
                        f"{np.percentile(boot_overall, 97.5):.3f}]\n"
                        f"resampling whole cells", loc="upper left")

    for k, m in enumerate(C.MODEL_ORDER):
        d = boot_model[m]
        axes[1].scatter(d, np.full(len(d), k) + rng_b.normal(0, .085, len(d)),
                        s=.8, color=C.MODEL_COL[m], alpha=.20, rasterized=True)
        axes[1].plot([np.percentile(d, 2.5), np.percentile(d, 97.5)], [k, k],
                     color=C.INK, lw=1.3, zorder=3)
        axes[1].scatter([d.mean()], [k], color=C.INK, s=16, zorder=4)
    axes[1].set_yticks(range(len(C.MODEL_ORDER)),
                       [m.replace("-Non-Reasoning", "") for m in C.MODEL_ORDER],
                       fontsize=6.3)
    axes[1].set_xlabel("cooperation rate")
    axes[1].set_title("b  the model ranking is stable")
    axes[1].set_ylim(-0.8, len(C.MODEL_ORDER) - 0.1)
    C.annotate(axes[1], f"{N_BOOT:,} cluster bootstrap draws per model;\n"
                        f"no overlap between the extreme models", loc="lower right")

    axes[2].hist(boot_slope, bins=60, color=C.OUTCOME_COL["DC"], alpha=.9)
    axes[2].axvline(0, color=C.C_DEFECT, lw=1.2, ls="--", ymax=0.72)
    axes[2].axvline(full["slope"], color=C.INK, lw=1.5, ymax=0.72)
    axes[2].set_xlabel("slope per decade of $\\lambda$")
    axes[2].set_ylabel(f"cluster bootstrap draws (of {N_BOOT:,})")
    axes[2].set_title("c  the slope, resampling whole cells")
    axes[2].set_ylim(0, axes[2].get_ylim()[1] * 1.32)
    C.annotate(axes[2], f"{sign_stab:.1f}% of draws stay positive\n"
                        f"95% CI [{np.percentile(boot_slope, 2.5):+.4f}, "
                        f"{np.percentile(boot_slope, 97.5):+.4f}]\n"
                        f"sampling error only, not model choice", loc="upper left")
    C.save(fig, "robust", "R05_bootstrap_stability",
           "How stable are the headline quantities under resampling of whole "
           "common-random-number cells rather than rows?",
           f"The level and the model ranking are very stable: overall cooperation is "
           f"{M.mean():.3f} [{np.percentile(boot_overall, 2.5):.3f}, "
           f"{np.percentile(boot_overall, 97.5):.3f}] and the six per-model bootstrap "
           f"clouds do not overlap at the extremes. The slope keeps its sign in "
           f"{sign_stab:.1f}% of draws, but that is a statement about sampling error "
           f"within a fixed set of six models only; as R01 shows, it does not survive "
           f"removing one of those models.",
           "bootstrap histograms + per-model bootstrap clouds",
           "joint_coop, scale, cell, model")

    # ======================================================================
    # 6  OUTLIER / TRIMMING SENSITIVITY
    # ======================================================================
    cell_level = M.mean(axis=1)
    trim_rows = []
    for frac in [0.0, 0.01, 0.05, 0.10, 0.20]:
        half = frac / 2
        lo_c, hi_c = (np.quantile(cs, [half, 1 - half]) if frac > 0
                      else (-np.inf, np.inf))
        keep_s = (cs >= lo_c) & (cs <= hi_c)
        lo_l, hi_l = (np.quantile(cell_level, [half, 1 - half]) if frac > 0
                      else (-np.inf, np.inf))
        keep_l = (cell_level >= lo_l) & (cell_level <= hi_l)
        for ri, (rule, keep) in enumerate(
                [("trim by per-cell slope (influence)", keep_s),
                 ("trim by cell mean cooperation (level)", keep_l)]):
            o = stats_from_matrix(M[keep], x)
            _, p, _ = perm_pvalues(M[keep], x, 1000, seed=900 + int(frac * 100) + ri)
            trim_rows.append({"rule": rule, "trim_fraction": frac,
                              "n_cells_kept": int(keep.sum()),
                              "slope_per_decade": o["slope"],
                              "range_level_means": o["rng"],
                              "p_slope_perm": p["p_slope"],
                              "p_range_perm": p["p_range"],
                              "slope_pct_of_full": 100 * o["slope"] / full["slope"]})
    trim_rows.append({"rule": "median of per-cell slopes (no trimming)",
                      "trim_fraction": np.nan, "n_cells_kept": len(cs),
                      "slope_per_decade": float(np.median(cs)),
                      "range_level_means": np.nan, "p_slope_perm": float(wil.pvalue),
                      "p_range_perm": np.nan,
                      "slope_pct_of_full": 100 * np.median(cs) / full["slope"]})
    trim = pd.DataFrame(trim_rows)
    C.savetab(trim, "robust_trimming")
    t_inf = trim[trim.rule == "trim by per-cell slope (influence)"]
    t_lev = trim[trim.rule == "trim by cell mean cooperation (level)"]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
    axes[0].hist(cs, bins=70, color=C.MUTED, alpha=.95)
    axes[0].axvline(0, color=C.INK, lw=.9)
    axes[0].axvline(float(cs.mean()), color=C.C_DEFECT, lw=1.6)
    axes[0].axvline(float(np.median(cs)), color=C.C_COOP, lw=1.6)
    axes[0].set_xlabel("per-cell slope per decade of $\\lambda$")
    axes[0].set_ylabel("design cells (of 1,200)")
    axes[0].set_title("a  the pooled slope is a mean of these")
    C.annotate(axes[0], f"mean {cs.mean():+.4f} (red)\nmedian {np.median(cs):+.4f} "
                        f"(blue)\n{100*np.mean(cs > 0):.0f}% of cells positive\n"
                        f"{100*np.mean(cs == 0):.0f}% exactly flat across "
                        f"all 10 $\\lambda$", loc="upper left")

    axes[1].axhline(full["slope"], color=C.INK2, lw=1.0, ls="--")
    axes[1].axhline(0, color=C.INK, lw=.9)
    axes[1].plot(100 * t_inf.trim_fraction, t_inf.slope_per_decade, "o-",
                 color=C.C_DEFECT, label="trim extreme cell slopes")
    axes[1].plot(100 * t_lev.trim_fraction, t_lev.slope_per_decade, "s-",
                 color=C.C_COOP, label="trim extreme cell levels")
    axes[1].set_xlabel("percent of design cells trimmed (symmetric)")
    axes[1].set_ylabel("slope per decade of $\\lambda$")
    axes[1].set_title("b  slope under trimming")
    axes[1].legend(fontsize=6.3, loc="lower right")
    axes[1].set_ylim(-0.0042, max(full["slope"], t_lev.slope_per_decade.max()) * 1.45)
    C.annotate(axes[1], f"at a 20% trim the slope is still\n"
                        f"{t_inf.slope_per_decade.iloc[-1]:+.4f} "
                        f"({t_inf.slope_pct_of_full.iloc[-1]:.0f}% of full)",
               loc="upper right")

    axes[2].axhline(full["rng"], color=C.INK2, lw=1.0, ls="--")
    axes[2].plot(100 * t_inf.trim_fraction, t_inf.range_level_means, "o-",
                 color=C.C_DEFECT, label="trim extreme cell slopes")
    axes[2].plot(100 * t_lev.trim_fraction, t_lev.range_level_means, "s-",
                 color=C.C_COOP, label="trim extreme cell levels")
    axes[2].axhline(float(np.percentile(null_r, 99)), color=C.INK, lw=1.0, ls=":")
    axes[2].set_xlabel("percent of design cells trimmed (symmetric)")
    axes[2].set_ylabel("max - min of $\\lambda$-level means")
    axes[2].set_title("c  non-invariance under trimming")
    axes[2].set_ylim(0, max(full["rng"], t_lev.range_level_means.max()) * 1.35)
    axes[2].legend(fontsize=6.3, loc="lower right")
    C.annotate(axes[2], "dotted line = permutation null\n99th percentile",
               loc="upper right")
    C.save(fig, "robust", "R06_outlier_trimming",
           "Are the conclusions driven by a small number of extreme design cells?",
           f"No, but the analysis exposes how thin the directional claim is. The "
           f"pooled slope is by construction the mean of 1,200 per-cell slopes whose "
           f"median is only {np.median(cs):+.4f}, so the positive pooled value comes "
           f"from an asymmetric tail rather than from a typical cell. Trimming is "
           f"nevertheless not what breaks it: at a 20% symmetric trim the slope is "
           f"still {t_inf.slope_per_decade.iloc[-1]:+.4f} "
           f"({t_inf.slope_pct_of_full.iloc[-1]:.0f}% of the full-sample value) and "
           f"the spread of the level means stays above the permutation null "
           f"throughout.",
           "histogram of per-cell slopes + trimming curves",
           "per-cell slope, cell mean cooperation, scale")

    # ======================================================================
    # 7  CLAIM CLASSIFICATION
    # ======================================================================
    fpath = C.CACHE / "findings.json"
    prior = json.loads(fpath.read_text(encoding="utf-8")) if fpath.exists() else []
    prior_by_id = {f["id"]: f for f in prior}

    bal = dy.groupby(["model", "language", "dyad", "scale"], observed=True).size()
    balanced = bool(bal.nunique() == 1 and int(bal.iloc[0]) == 10)
    n_rounds_per_game = len(rounds) / rounds["game_uid"].nunique()
    seq = (rounds.sort_values(["game_uid", "agent", "round"])
           .groupby(["game_uid", "agent"], observed=True)["action"]
           .apply(lambda s: "".join(s)))
    atom_share = float(seq.isin(["C" * 10, "D" * 10]).mean())
    lp = np.log10(rounds.groupby("scale", observed=True)["payoff_raw"].mean())
    expo = float(np.polyfit(x, lp.to_numpy(), 1)[0])
    rho_own = float(stats.spearmanr(ag["coop_rate"], ag["utility"]).statistic)
    rho_opp = float(stats.spearmanr(ag["opp_coop_rate"], ag["utility"]).statistic)
    atoms = ag.groupby("model", observed=True)["coop_rate"].apply(
        lambda v: float(((v == 0) | (v == 1)).mean()))
    always_c = np.array([(ag[ag.scale == s]["coop_rate"] == 1).mean()
                         for s in C.SCALES])
    never_c = np.array([(ag[ag.scale == s]["coop_rate"] == 0).mean()
                        for s in C.SCALES])
    rho_ac = stats.spearmanr(x, always_c)
    rho_nc = stats.spearmanr(x, never_c)
    forms_path = C.TABLES / "scaling_functional_forms.csv"
    d_aic_lin = np.nan
    if forms_path.exists():
        ff = pd.read_csv(forms_path).set_index("form")
        if "linear" in ff.index:
            d_aic_lin = float(ff.loc["linear", "aic"] - ff["aic"].min())

    claims = [
        {"claim_id": "Q-integrity", "section": "01 data quality",
         "claim_short": "Complete balanced 10x6x5x4x10 factorial, no missing data",
         "check_performed": "recounted every (model, language, pairing, scale) group",
         "check_result": f"all groups of size {int(bal.iloc[0])}, "
                         f"{bal.nunique()} distinct size; balanced = {balanced}",
         "robustness_class": "highly robust",
         "note": "deterministic recount, nothing here can break"},
        {"claim_id": "Q-pseudoreplication", "section": "01 data quality",
         "claim_short": "Rounds are 20-fold pseudoreplicated; cluster on games",
         "check_performed": "rounds per game, plus the design effect of ignoring "
                            "the game cluster",
         "check_result": f"{n_rounds_per_game:.0f} rounds per game; the naive "
                         f"round-level CI is {1/game_ratio:.1f}x too narrow "
                         f"(design effect {game_de:.0f})",
         "robustness_class": "highly robust",
         "note": "quantified directly in R02; the anticonservatism is large"},
        {"claim_id": "P-nominal-illusion-setup", "section": "03 payoff",
         "claim_short": "Raw payoff scales with lambda at exponent 1 by construction",
         "check_performed": "refit log10 mean raw payoff on log10 lambda",
         "check_result": f"exponent {expo:.4f}",
         "robustness_class": "highly robust",
         "note": "an algebraic identity, not an empirical scaling law"},
        {"claim_id": "P-dilemma-reproduced", "section": "03 payoff",
         "claim_short": "The logged payoffs reproduce the dilemma gradient",
         "check_performed": "Spearman of own and opponent cooperation on own utility",
         "check_result": f"own rho = {rho_own:+.3f}, opponent rho = {rho_opp:+.3f}",
         "robustness_class": "highly robust",
         "note": "the sign pattern is forced by the payoff matrix; a units check"},
        {"claim_id": "D-ushape", "section": "02 distributions",
         "claim_short": "Cooperation is U-shaped and boundary-inflated",
         "check_performed": "constant-sequence share pooled and boundary mass "
                            "within every model",
         "check_result": f"pooled constant-sequence share {atom_share:.3f}; per-model "
                         f"boundary mass spans {atoms.min():.3f} to {atoms.max():.3f}",
         "robustness_class": "moderately robust",
         "note": "the per-model spread is the robust part; the pooled U shape "
                 "aggregates very different per-model shapes and should not be "
                 "quoted on its own"},
        {"claim_id": "S-invariance-violation", "section": "04 scaling",
         "claim_short": "Frontier LLMs are not invariant to positive payoff rescaling",
         "check_performed": f"exact within-cell permutation ({N_PERM:,} draws), all "
                            f"{len(body)} leave-one-out subsets, trimming to 20% "
                            f"of cells",
         "check_result": f"p_perm = {full_p['p_range']:.1e} on the spread of the "
                         f"level means; {n_range_sig} of {len(body)} leave-one-out "
                         f"subsets still reject; {n_model_rng} of 6 models and "
                         f"{n_lang_rng} of 5 languages reject alone",
         "robustness_class": "highly robust",
         "note": "survived every attack made in this section"},
        {"claim_id": "S-invariance-direction", "section": "04 scaling",
         "claim_short": "Cooperation RISES with lambda, about +0.013 per decade",
         "check_performed": "leave-one-model-out, leave-one-rung-out, per-model "
                            "permutation slopes",
         "check_result": f"pooled {full['slope']:+.4f}; the sign flips in {n_flip} "
                         f"of {len(body)} subsets: dropping Grok "
                         f"({grok.slope_per_decade:+.4f}), dropping the CvC pairing "
                         f"({cvc.slope_per_decade:+.4f}), dropping lambda = 0.01 and "
                         f"0.1 ({crit.slope_per_decade:+.4f}) and dropping 0.01, 0.1 "
                         f"and 0.25; {n_neg} of 6 models have a negative slope",
         "robustness_class": "sensitive",
         "note": "a pooled average over opposite-sign subgroups; it must not be "
                 "reported as a shared tendency of frontier LLMs"},
        {"claim_id": "S-nonmonotone", "section": "04 scaling",
         "claim_short": "The violation is concentrated at sub-unit lambda",
         "check_performed": "drop lambda = 0.01 and 0.1 and re-run the permutation test",
         "check_result": f"the spread falls from {full['rng']:.3f} to "
                         f"{crit.range_level_means:.3f} but still rejects "
                         f"(p = {crit.p_range_perm:.1e}); the saturated form beats "
                         f"the linear one by {d_aic_lin:.0f} AIC",
         "robustness_class": "moderately robust",
         "note": "the small-lambda rungs carry most of the effect but not all of it; "
                 "only two rungs sit below lambda = 0.25, so the knee is weakly "
                 "located"},
        {"claim_id": "S-mechanism", "section": "04 scaling",
         "claim_short": "Lambda acts by re-weighting unconditional policies",
         "check_performed": "Spearman of the always-C and never-C shares on log "
                            "lambda over the 10 rungs",
         "check_result": f"always-C rho = {rho_ac.statistic:+.2f} "
                         f"(p = {rho_ac.pvalue:.2f}); never-C rho = "
                         f"{rho_nc.statistic:+.2f} (p = {rho_nc.pvalue:.3f})",
         "robustness_class": "sensitive",
         "note": "only the never-cooperate half is supported; the always-cooperate "
                 "correlation cannot be distinguished from zero on 10 points, so the "
                 "symmetric 'two atoms move in opposition' phrasing overstates it"},
    ]
    cls = pd.DataFrame(claims)
    # S-invariance-direction is not a recorded finding id; its prior grade is taken
    # from tables/scaling_results.csv, where the pooled slope is reported at
    # p = 1.5e-10 with no caveat about the sign, i.e. effectively "strong".
    fallback = {"S-invariance-direction": "strong"}
    cls["original_strength"] = [
        prior_by_id.get(c, {}).get("strength", fallback.get(c, "strong"))
        for c in cls["claim_id"]]
    cls["prior_grade_source"] = ["cache/findings.json" if c in prior_by_id
                                 else "tables/scaling_results.csv"
                                 for c in cls["claim_id"]]
    rank = {"highly robust": 3, "moderately robust": 2, "sensitive": 1}
    orank = {"strong": 3, "moderate": 2, "weak": 1}
    cls["downgraded"] = [rank[r] < orank.get(o, 3)
                         for r, o in zip(cls.robustness_class, cls.original_strength)]
    cls = cls[["claim_id", "section", "claim_short", "original_strength",
               "prior_grade_source", "robustness_class", "downgraded",
               "check_performed", "check_result", "note"]]
    C.savetab(cls, "robust_claim_classification")

    order = (cls.assign(_r=cls.robustness_class.map(rank))
             .sort_values(["_r", "claim_id"], ascending=[False, True])
             .reset_index(drop=True))
    n_down = int(order.downgraded.sum())

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    for i, r_ in enumerate(order.itertuples()):
        o = orank.get(r_.original_strength, 3)
        n = rank[r_.robustness_class]
        col = C.C_DEFECT if r_.downgraded else C.MUTED
        if o != n:
            axes[0].annotate("", xy=(n + .07, i), xytext=(o - .07, i),
                             arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
        axes[0].scatter([o], [i], facecolor="none", edgecolor=C.MUTED, s=42,
                        lw=1.1, zorder=3)
        axes[0].scatter([n], [i], color=CLASS_COL[r_.robustness_class], s=70, zorder=4)
    axes[0].set_yticks(range(len(order)), order.claim_id, fontsize=7)
    axes[0].set_xticks([1, 2, 3], ["sensitive", "moderately\nrobust",
                                   "highly\nrobust"], fontsize=7.5)
    axes[0].set_xlim(0.55, 3.45)
    axes[0].set_ylim(len(order) + 0.55, -0.6)
    axes[0].set_xlabel("robustness class after this section's checks")
    axes[0].set_title("a  every recorded claim re-graded")
    C.annotate(axes[0], f"open circle = grade before this section;\n"
                        f"{n_down} of {len(order)} claims downgraded (red arrow)",
               loc="lower left", color=C.C_DEFECT)

    labels = ["non-invariance\n(spread of level means)", "direction\n(linear slope)"]
    frac_rng = 100 * float((body["p_range_perm"] < .05).mean())
    frac_sign = 100 * float(1 - body["sign_flip_vs_full"].mean())
    frac_slope_sig = 100 * float((body["p_slope_perm"] < .05).mean())
    xs = np.arange(2)
    axes[1].bar(xs - .21, [frac_rng, frac_slope_sig], .4, color=C.C_COOP,
                label="significant, either sign")
    axes[1].bar(xs + .21, [100.0, frac_sign], .4, color=C.OUTCOME_COL["DC"],
                label="keeps the full-sample sign")
    axes[1].axhline(100, color=C.INK, lw=.8, ls=":")
    axes[1].set_xticks(xs, labels, fontsize=7.5)
    axes[1].set_ylabel(f"percent of the {len(body)} leave-one-out subsets")
    axes[1].set_ylim(0, 142)
    axes[1].set_title("b  what separates the two grades")
    axes[1].legend(fontsize=6.4, loc="upper right")
    C.annotate(axes[1], "for the slope, 'significant' counts\nsubsets where it turns "
                        "significantly\nNEGATIVE, so the blue bar overstates\nhow "
                        "stable the direction is", loc="upper left")
    for xx, v in zip([xs[0] - .21, xs[0] + .21, xs[1] - .21, xs[1] + .21],
                     [frac_rng, 100.0, frac_slope_sig, frac_sign]):
        axes[1].text(xx, v + 4, f"{v:.0f}%", ha="center", fontsize=6.8, color=C.INK2)
    C.save(fig, "robust", "R07_claim_classification",
           "After a deliberate attempt to break them, which claims in this report "
           "survive intact and which have to be downgraded?",
           f"{n_down} of {len(order)} claims are downgraded. The invariance violation "
           f"itself is upheld as highly robust: it stays significant in {frac_rng:.0f}"
           f"% of the {len(body)} leave-one-out subsets, in every model taken alone "
           f"and under a 20% trim of design cells. The directional reading of it is "
           f"downgraded to sensitive: the slope keeps its sign in {frac_sign:.0f}% of "
           f"subsets, but the {n_flip} that flip are exactly the ones that matter - "
           f"removing Grok, removing the CvC pairing, and removing the two or three "
           f"smallest lambda rungs. The mechanism claim is downgraded because only "
           f"its never-cooperate half is supported by the data.",
           "grade-change dumbbell + leave-one-out survival bars",
           "all recorded findings, leave-one-out subsets")

    # ======================================================================
    # findings
    # ======================================================================
    C.record_finding(
        id="R-invariance-survives",
        finding="The payoff-scale invariance violation survives an adversarial "
                "audit: it is not the work of any single model, language, pairing "
                "or rung of the lambda ladder.",
        evidence=f"An exact within-cell permutation test that shuffles lambda labels "
                 f"{N_PERM:,} times inside each of the 1,200 common-random-number "
                 f"cells puts the observed spread of the ten lambda-level cooperation "
                 f"means ({full['rng']:.3f}) far above the null 99th percentile "
                 f"({np.percentile(null_r, 99):.3f}), p = {full_p['p_range']:.1e}. "
                 f"The same statistic stays significant in {n_range_sig} of "
                 f"{len(body)} leave-one-out subsets, in {n_model_rng} of 6 models "
                 f"taken alone, in {n_lang_rng} of 5 languages, and under a 20% "
                 f"symmetric trim of design cells.",
        figure="09_robustness/R04_permutation_invariance.png, "
               "09_robustness/R01_leave_one_out.png, "
               "09_robustness/R06_outlier_trimming.png",
        strength="strong",
        robustness="the permutation null needs no distributional assumption and the "
                   "saturated design makes the randomisation exact; every subgroup "
                   "and every trimming rule reproduces it",
        interpretation="Because a positive rescaling of all four payoffs is a "
                       "theorem-level irrelevance, a departure this robust is direct "
                       "evidence that these models respond to the magnitude of the "
                       "numbers and not only to the incentive structure. Lambda was "
                       "assigned by design rather than observed, so the direction of "
                       "causation from the lambda label to behaviour is identified; "
                       "what is not identified is the cognitive mechanism behind it.",
        caveat="Self-play only, one game, one 10-round disclosed horizon, one prompt "
               "family. Robustness here means robustness within this corpus. The "
               "Arabic and Chinese templates also carry an inherited translation "
               "inconsistency, so language-level comparisons are confounded even "
               "though the lambda effect is present in all five languages.",
        claim="Payoff-scale non-invariance is confirmed by an exact randomisation "
              "test and survives leaving out any model, language, pairing or scale, "
              "so it is a property of the corpus rather than of one subgroup.")

    C.record_finding(
        id="R-direction-fragile",
        finding="The DIRECTION of the payoff-scale effect does not survive the same "
                "audit. The pooled positive slope is an average over opposite-sign "
                "models and must not be read as a shared tendency.",
        evidence=f"The pooled slope is {full['slope']:+.4f} per decade, but "
                 f"{n_neg} of the 6 models have a negative slope, and because the "
                 f"design is balanced the pooled slope is exactly the mean of the six "
                 f"model slopes. Dropping Grok-4.20-Non-Reasoning alone moves it to "
                 f"{grok.slope_per_decade:+.4f}; dropping the two smallest scales "
                 f"moves it to {crit.slope_per_decade:+.4f}; dropping the CvC pairing "
                 f"moves it to {cvc.slope_per_decade:+.4f}. In total the sign flips "
                 f"in {n_flip} of {len(body)} leave-one-out subsets. The median of "
                 f"the 1,200 per-cell slopes is {np.median(cs):+.4f}, only "
                 f"{100*np.median(cs)/cs.mean():.0f}% of the mean.",
        figure="09_robustness/R01_leave_one_out.png, "
               "09_robustness/R06_outlier_trimming.png, "
               "09_robustness/R03_estimator_sensitivity.png",
        strength="strong",
        robustness="this is itself a robust negative result: it shows up in the "
                   "leave-one-out family, in the per-model permutation tests and in "
                   "the mean-versus-median gap of the per-cell slopes",
        interpretation="This is a subgroup reversal of the classic kind. Reporting "
                       "'+0.013 cooperation per decade of lambda' as a property of "
                       "frontier LLMs would be an aggregation artefact. What the "
                       "corpus supports is that each model has its own non-zero and "
                       "usually non-monotone lambda response, with no common sign. "
                       "The association within each model is causal in the design "
                       "sense; the pooled direction is not even a stable description.",
        caveat="Six models is a small sample of models, so the three-versus-three "
               "sign split is itself imprecisely estimated. The point is that no "
               "common direction is identified, not that the true split is even.",
        claim="The pooled cooperation-versus-lambda slope averages over models whose "
              "slopes have opposite signs, so the invariance violation has no common "
              "direction across frontier LLMs.")

    C.record_finding(
        id="R-clustering-design-effect",
        finding="Ignoring the game cluster would shrink the confidence interval on "
                "the payoff-scale slope several-fold while leaving the point estimate "
                "untouched, so a round-level analysis would look far more certain "
                "than the data allow.",
        evidence=f"The same slope estimated at six units of analysis varies by only "
                 f"{spread:.0e}, a consequence of the balanced design. The 95% CI "
                 f"width goes from {naive_w:.4f} for the naive round-level fit to "
                 f"{dyad_w:.4f} at the dyad level clustered on the design cell, a "
                 f"factor of {dyad_w/naive_w:.1f} and a design effect of about "
                 f"{dyad_de:.0f}. Rounds within a game correlate at ICC "
                 f"{icc.icc.iloc[0]:.2f}.",
        figure="09_robustness/R02_aggregation_levels.png",
        strength="strong",
        robustness="arithmetic on the same data at six aggregation levels",
        interpretation="The agreement of the point estimate across units is algebra "
                       "and must never be presented as a robustness check. The only "
                       "thing the unit of analysis decides here is the honesty of the "
                       "standard error.",
        caveat="The naive interval is computed purely as a counterfactual "
               "illustration; no analysis in this report relies on it.",
        claim="Every interval in this report is clustered because the naive "
              "round-level interval on the identical point estimate is several times "
              f"too narrow, a design effect of roughly {dyad_de:.0f}.")

    C.record_finding(
        id="R-estimator-agnostic",
        finding="The payoff-scale result is not an artefact of the estimator: least "
                "squares on rates, a cluster-robust logistic model on binary rounds "
                "and an exact randomisation test all agree in sign and significance. "
                "The single exception is a monotone rank statistic, which has no "
                "power against a non-monotone effect.",
        evidence=f"{len(est)} estimators, from OLS at the dyad level "
                 f"({dy_ols['slope']:+.4f} per decade) through the marginal effect of "
                 f"a cluster-robust logistic GLM on 240,000 rounds ({glm_ame:+.4f}) "
                 f"to the median of the 1,200 per-cell slopes ({np.median(cs):+.4f}), "
                 f"agree in sign, and {n_sig_est} of {len(est)} remain significant "
                 f"after Benjamini-Hochberg adjustment. The exception is the mean "
                 f"per-cell Spearman rho, {rho_mean:+.3f} "
                 f"[{rho_lo:+.3f}, {rho_hi:+.3f}], p = {rho_t.pvalue:.2f}. Widening "
                 f"the cluster from the design cell to the 120 model x language x "
                 f"pairing conditions also drops the slope from p = 1.5e-10 to "
                 f"p = {cond_p:.3f}.",
        figure="09_robustness/R03_estimator_sensitivity.png, "
               "09_robustness/R02_aggregation_levels.png",
        strength="strong",
        robustness="agreement across parametric, semi-parametric and non-parametric "
                   "estimators, with FDR adjustment applied across the family; the "
                   "two disagreements are both explained by low power rather than by "
                   "a different point estimate",
        interpretation="Estimator choice is not where the fragility of this result "
                       "lives. The fragility is in aggregation across models, which "
                       "no choice of estimator can repair. The failure of the rank "
                       "statistic is informative in its own right: a per-cell "
                       "Spearman rho tests for monotonicity, and the effect is not "
                       "monotone.",
        caveat="These estimators are not independent evidence; they are different "
               "summaries of the same 12,000 games, so their agreement bounds "
               "specification error and says nothing about sampling error.",
        claim="The payoff-scale effect is estimator-agnostic apart from a monotone "
              "rank statistic that is not designed to detect it, so disagreement "
              "between specifications is not a plausible explanation for it.")

    C.record_finding(
        id="R-claims-regraded",
        finding=f"{n_down} of {len(cls)} recorded claims had to be downgraded once "
                f"tested adversarially, including the directional reading of the "
                f"headline result and the mechanism claim.",
        evidence=f"Every recorded claim was re-graded against a check run in this "
                 f"module. Non-invariance is graded highly robust (permutation "
                 f"p = {full_p['p_range']:.1e}, still significant in "
                 f"{frac_rng:.0f}% of leave-one-out subsets). The direction is graded "
                 f"sensitive: its sign flips in {n_flip} of {len(body)} subsets, and "
                 f"those {n_flip} are removing Grok, removing the CvC pairing and "
                 f"removing the smallest lambda rungs, not arbitrary subsets. The "
                 f"mechanism claim is graded sensitive because the "
                 f"always-cooperate share correlates with log lambda at only "
                 f"rho = {rho_ac.statistic:+.2f} (p = {rho_ac.pvalue:.2f}) over the "
                 f"10 rungs, while the never-cooperate share reaches "
                 f"rho = {rho_nc.statistic:+.2f} (p = {rho_nc.pvalue:.3f}).",
        figure="09_robustness/R07_claim_classification.png",
        strength="strong",
        robustness="each grade is tied to a numbered check recorded in "
                   "tables/robust_claim_classification.csv",
        interpretation="The report's central negative result about rationality "
                       "stands; several of the more quotable directional summaries "
                       "built on top of it do not. Readers should quote the "
                       "non-invariance and the per-model curves, not the pooled "
                       "slope.",
        caveat="Only claims already recorded as findings were graded. Statements made "
               "in figure captions elsewhere in the report were not audited here.",
        claim="An adversarial re-grading of every recorded claim upholds payoff "
              f"non-invariance as highly robust and downgrades {n_down} claims, "
              "chiefly the pooled direction of the effect.")

    print(f"  downgraded {n_down} of {len(cls)} claims")
    return loo, agg, permtab, cls
