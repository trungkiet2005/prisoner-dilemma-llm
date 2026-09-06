"""Section 6: fairness and inequality.

Two different things are called "fairness" in this literature and they are not
the same measurement, do not have the same units, and do not even have the same
unit of observation. This section keeps them apart by name throughout.

(A) OUTCOME fairness, inside a dyad
    How evenly the two players of one game split the proceeds: utility_gap,
    signed_gap, gini, fairness = 1 - gini. Unit of observation: the game.

    The first thing to establish here is that this is not a free-standing
    construct. Both agents face the same symmetric matrix, so in any round where
    they choose the same action they collect the same payoff by construction. A
    utility gap can therefore only be produced by rounds in which the two acted
    differently, and the algebra is exact:

        signed_gap  = p_DC - p_CD              (focal-1 exploiting minus exploited)
        utility_gap = |p_DC - p_CD|
                    = p_exploit - 2 min(p_CD, p_DC)

    So "inequality" in this corpus is a re-description of net exploitation, not
    an independent dimension of behaviour. Section 6 shows that identity rather
    than assuming it, because the honest reading of every later inequality
    number depends on knowing that the gap is mechanically bounded by how often
    the pair miscoordinated.

(B) TREATMENT fairness, across groups
    The FAIRGAME disparity notion: does the same model treat the five language
    versions of an identical game alike, treat the two persona labels alike, and
    treat the two seat positions alike? Unit of observation: the group mean. The
    statistic is max-minus-min of the group means, and its null distribution is
    obtained by permuting the group label at the game level, which is the level
    at which the label was actually randomised. 5,000 to 10,000 permutations,
    Benjamini-Hochberg across the six models.

Why the permutation and not a t-test
------------------------------------
The max-minus-min of k group means has no standard closed-form null, it is
biased upward by construction (it is an extreme of k noisy quantities), and that
upward bias grows with k and shrinks with n. A permutation null carries the same
bias, so the comparison is like for like. Reporting the raw gap without it would
make every grouping look unfair.

The seat test is the design validation of the whole corpus
----------------------------------------------------------
In a CvC or an SvS game the two agents carry identical personas and structurally
identical prompts, so under seat exchangeability the expected signed gap is
exactly zero. That makes it a sharp, assumption-free specification check, and it
is tested with a sign-flip permutation, which is the exact test for that null.
It does not pass everywhere, and that is reported as a finding rather than
buried.

Caveats respected throughout
----------------------------
The Arabic and Chinese prompts carry an inherited FAIRGAME translation quirk:
their closing goal sentence says "maximise your rewards" while the payoff
sentences above them say "penalty". Any ar/cn disparity is therefore partly
prompt inconsistency and not evidence about language or culture. Everything here
is self-play, so none of it speaks to how a model treats a different model.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import core as C


# --------------------------------------------------------------------------
# permutation machinery for the group-disparity (FAIRGAME) statistic
# --------------------------------------------------------------------------
def perm_group_gap(y, labels, order, strata=None, n_perm=5000, seed=0):
    """max-minus-min of group means, with a label-permutation null.

    The label is shuffled *within strata* when strata are supplied, which keeps
    the composition of the nuisance factor fixed and removes its variance from
    the null. Group sizes are preserved exactly by construction.

    Returns a dict with the observed gap, the group means, the permutation p,
    the null mean gap (the bias floor), and eta^2 for the grouping factor.
    """
    y = np.asarray(y, float)
    lab = pd.Categorical(labels, categories=list(order))
    codes = lab.codes.astype(np.intp)
    ok = np.isfinite(y) & (codes >= 0)
    y, codes = y[ok], codes[ok]
    k = len(order)
    counts = np.bincount(codes, minlength=k).astype(float)
    counts[counts == 0] = np.nan
    means = np.bincount(codes, weights=y, minlength=k) / counts
    obs = float(np.nanmax(means) - np.nanmin(means))

    if strata is None:
        blocks = [np.arange(len(y))]
    else:
        st = np.asarray(strata)[ok]
        blocks = [np.flatnonzero(st == s) for s in pd.unique(st)]

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    yy = np.empty_like(y)
    for i in range(n_perm):
        for b in blocks:
            yy[b] = rng.permutation(y[b])
        m = np.bincount(codes, weights=yy, minlength=k) / counts
        null[i] = np.nanmax(m) - np.nanmin(m)
    p = float((1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1))

    grand = y.mean()
    ss_b = float(np.nansum(counts * (means - grand) ** 2))
    ss_t = float(np.sum((y - grand) ** 2))
    return {"observed_gap": obs, "means": means, "p_perm": p,
            "null_mean_gap": float(null.mean()),
            "null_q95": float(np.percentile(null, 95)),
            "eta2": ss_b / ss_t if ss_t > 0 else np.nan,
            "gap_over_sd": obs / y.std() if y.std() > 0 else np.nan,
            "n": int(len(y)), "null": null}


def signflip_test(x, n_perm=5000, seed=0):
    """Exact test that E[x] = 0 under sign exchangeability (seat swap)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    obs = float(x.mean())
    rng = np.random.default_rng(seed)
    n = len(x)
    null = np.empty(n_perm)
    for i in range(n_perm):
        s = rng.integers(0, 2, n) * 2 - 1
        null[i] = float(np.mean(x * s))
    p = float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-15)) / (n_perm + 1))
    sd = x.std(ddof=1)
    return {"mean": obs, "p_perm": p, "n": n, "sd": sd,
            "cohens_d": obs / sd if sd > 0 else np.nan,
            "lo": float(np.percentile(null, 2.5)), "hi": float(np.percentile(null, 97.5))}


def _short(m: str) -> str:
    """Compact model label for crowded tick axes; colour still carries identity."""
    return (m.replace("-Non-Reasoning", "").replace("-Flash-Lite", "-FL")
            .replace("Qwen3-235B-A22B", "Qwen3-235B"))


def _ci(sub, col, n_boot=1200, seed=0):
    return C.cluster_boot_ci(sub[col].to_numpy(), sub["cell"].to_numpy(),
                             n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------
def run(rounds, ag, dy):
    C.use_style()
    print("\n== 05 fairness ==")

    dy = dy.copy()
    dy["n_asym"] = np.round(dy["p_exploit"] * 10).astype(int)
    dy["n_gap"] = np.round(dy["utility_gap"] * 10).astype(int)
    dy["n_bidir"] = np.round(np.minimum(dy["p_CD"], dy["p_DC"]) * 10).astype(int)
    dy["equal"] = (dy["utility_gap"] == 0).astype(int)

    # ======================================================================
    # FR01  how unequal is a dyad, and how often is it exactly equal
    # ======================================================================
    gap0 = float(dy["equal"].mean())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))

    ax = axes[0]
    ax.hist(dy["utility_gap"], bins=np.arange(-.05, 1.06, .1), color=C.C_COOP,
            rwidth=.85)
    ax.set_xlabel("within-dyad utility gap  |$u_1 - u_2$| (utility units)")
    ax.set_ylabel("dyads (of 12,000)")
    ax.set_title("a  outcome fairness is spiky")
    C.annotate(ax, f"exactly equal split: {gap0:.1%}\n"
                   f"mean gap {dy.utility_gap.mean():.3f}\n"
                   f"median {dy.utility_gap.median():.2f}, "
                   f"max {dy.utility_gap.max():.1f}", loc="upper right")

    ax = axes[1]
    x = np.sort(dy["gini"].to_numpy())
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=C.OUTCOME_COL["DC"], lw=1.6)
    ax.axvline(0, color=C.MUTED, lw=.7, ls=":")
    ax.axhline(gap0, color=C.MUTED, lw=.7, ls=":")
    ax.set_xlabel("within-dyad Gini of utility")
    ax.set_ylabel("ECDF over dyads")
    ax.set_title("b  a third of games split evenly")
    q = dy["gini"].quantile([.5, .75, .9])
    C.annotate(ax, f"Gini = 0 for {gap0:.1%} of dyads\n"
                   f"median {q.iloc[0]:.2f}, p75 {q.iloc[1]:.2f}, p90 {q.iloc[2]:.2f}",
               loc="lower right")

    ax = axes[2]
    eq = dy.groupby("model", observed=True)["equal"].agg(["mean", "size"]).reindex(
        C.MODEL_ORDER)
    lo, hi = C.wilson(eq["mean"].to_numpy(), eq["size"].to_numpy())
    yy = np.arange(len(eq))
    ax.barh(yy, eq["mean"], color=[C.MODEL_COL[m] for m in eq.index], height=.66)
    ax.errorbar(eq["mean"], yy, xerr=[eq["mean"] - lo, hi - eq["mean"]], fmt="none",
                ecolor=C.INK, lw=1.1)
    ax.set_yticks(yy, [m.replace("-Non-Reasoning", "") for m in eq.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("share of games with a perfectly equal split")
    ax.set_title("c  equality is model-specific")
    ax.set_xlim(0, .62)
    for i, (v, h) in enumerate(zip(eq["mean"], hi)):
        ax.text(h + .015, i, f"{v:.0%}", va="center", fontsize=6.8, color=C.INK2)
    C.save(fig, "fairness", "FR01_outcome_gap_distribution",
           "How unequally do two identical agents split the proceeds of one game, "
           "and how often is the split exactly equal?",
           f"Outcome fairness is a spiky, boundary-heavy distribution rather than a "
           f"smooth one: {gap0:.1%} of the 12,000 dyads end in a perfectly equal split "
           f"(gap = 0, Gini = 0), the median gap is {dy.utility_gap.median():.2f} "
           f"utility units and the mean is {dy.utility_gap.mean():.3f}. The equal-split "
           f"share is strongly model-specific, from "
           f"{eq['mean'].min():.0%} to {eq['mean'].max():.0%}.",
           "histogram + ECDF + bar with Wilson CI",
           "utility_gap, gini, model")

    # ======================================================================
    # FR02  the gap is mechanical: it is net exploitation, re-labelled
    # ======================================================================
    resid = float((dy["utility_gap"] - (dy["p_DC"] - dy["p_CD"]).abs()).abs().max())
    r_gap_ex = float(dy["utility_gap"].corr(dy["p_exploit"]))
    sub = dy[dy.n_asym > 0]
    one_dir = float((sub["n_gap"] == sub["n_asym"]).mean())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    ax = axes[0]
    ax.scatter(dy["n_asym"] + np.random.default_rng(0).uniform(-.28, .28, len(dy)),
               dy["utility_gap"] + np.random.default_rng(1).uniform(-.022, .022, len(dy)),
               s=2.2, color=C.MUTED, alpha=.18, lw=0, rasterized=True)
    ms, los, his = [], [], []
    ks = list(range(11))
    for k in ks:
        s = dy[dy.n_asym == k]
        m, l, h = _ci(s, "utility_gap") if len(s) else (np.nan, np.nan, np.nan)
        ms.append(m); los.append(l); his.append(h)
    ax.plot(ks, np.array(ks) / 10, color=C.INK, lw=1.1, ls="--")
    ax.errorbar(ks, ms, yerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)],
                fmt="o-", color=C.C_COOP, ecolor=C.INK2, lw=1.5, zorder=4)
    ax.set_xlabel("asymmetric rounds in the game (CD or DC), of 10")
    ax.set_ylabel("within-dyad utility gap")
    ax.set_title("a  the gap is bounded by miscoordination")
    ax.set_xticks(ks)
    C.annotate(ax, "dashed line = algebraic ceiling\n(all exploitation one way)",
               loc="upper left")
    C.annotate(ax, f"r(gap, exploitation) = {r_gap_ex:.3f}", loc="lower right")

    ax = axes[1]
    ct = pd.crosstab(dy["n_asym"], dy["n_gap"])
    ct = ct.reindex(index=range(11), columns=range(11)).fillna(0)
    im = ax.imshow(np.log10(ct.values + 1), cmap=C.SEQ, origin="lower", aspect="auto")
    ax.plot([0, 10], [0, 10], color=C.C_DEFECT, lw=1.0, ls="--")
    ax.set_xlabel("utility gap $\\times$ 10 (rounds of net advantage)")
    ax.set_ylabel("asymmetric rounds")
    ax.set_title("b  a parity lattice, not a cloud")
    ax.set_xticks(range(11), [str(i) for i in range(11)], fontsize=6)
    ax.set_yticks(range(11), [str(i) for i in range(11)], fontsize=6)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.046, label="$\\log_{10}$(1 + dyads)")
    C.annotate(ax, "gap = asym $-$ 2 $\\times$ (two-way exploits),\nso only "
                   "same-parity cells\ncan be occupied; residual "
                   f"{resid:.0e}", loc="lower right", color=C.INK)

    ax = axes[2]
    parts = {
        "no asymmetric round\n(gap = 0 by construction)": float((dy.n_asym == 0).mean()),
        "one-way exploitation\n(gap = ceiling)": float(((dy.n_asym > 0) & (dy.n_gap == dy.n_asym)).mean()),
        "two-way exploitation\n(gap < ceiling)": float(((dy.n_asym > 0) & (dy.n_gap < dy.n_asym)).mean()),
    }
    cols = [C.OUTCOME_COL["CC"], C.OUTCOME_COL["DC"], C.OUTCOME_COL["CD"]]
    ax.barh(range(3), list(parts.values()), color=cols, height=.6)
    ax.set_yticks(range(3), list(parts.keys()), fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("share of the 12,000 dyads")
    ax.set_xlim(0, .72)
    ax.set_title("c  where inequality comes from")
    for i, v in enumerate(parts.values()):
        ax.text(v + .01, i, f"{v:.1%}", va="center", fontsize=7, color=C.INK2)
    C.annotate(ax, f"of games with any asymmetry,\n{one_dir:.0%} are exploited in one\n"
                   "direction only", loc="upper right")
    C.save(fig, "fairness", "FR02_gap_decomposition",
           "Is within-dyad inequality an independent behavioural construct, or is it "
           "an arithmetic re-description of exploitation events?",
           f"It is arithmetic. In a symmetric matrix the two agents earn identically in "
           f"any round they act alike, so the gap equals |p_DC - p_CD| exactly "
           f"(maximum residual {resid:.0e} over 12,000 dyads) and is bounded above by "
           f"the number of asymmetric rounds. {(dy.n_asym==0).mean():.1%} of dyads have "
           f"no asymmetric round at all and are equal by construction; among the rest, "
           f"{one_dir:.0%} are exploited in one direction only and sit exactly on the "
           f"ceiling. The correlation of gap with total exploitation is "
           f"{r_gap_ex:.3f}, and it falls short of 1 only because two-way exploitation "
           f"cancels.",
           "scatter with binned means + count heatmap + share bar",
           "utility_gap, p_exploit, p_CD, p_DC")

    C.savetab(pd.DataFrame({
        "n_asymmetric_rounds": ks,
        "n_dyads": [int((dy.n_asym == k).sum()) for k in ks],
        "mean_utility_gap": ms, "ci_lo": los, "ci_hi": his,
        "algebraic_ceiling": [k / 10 for k in ks],
        "share_at_ceiling": [float((dy[dy.n_asym == k].n_gap == k).mean())
                             if (dy.n_asym == k).any() else np.nan for k in ks],
    }), "fairness_gap_decomposition")

    # ======================================================================
    # FR03  fairness versus welfare: trade-off or complements?
    # ======================================================================
    ub = np.round(np.arange(.40, .81, .05), 3)
    dy["ubin"] = pd.cut(dy["joint_utility"], ub, include_lowest=True)
    grp = dy.groupby("ubin", observed=True)
    ctr = np.array([iv.mid for iv in grp.groups.keys()])
    fm, fl, fh = [], [], []
    for _, s in grp:
        m, l, h = _ci(s, "fairness")
        fm.append(m); fl.append(l); fh.append(h)
    comp = grp[["p_CC", "p_DD", "p_exploit"]].mean()

    rho = stats.spearmanr(dy["fairness"], dy["joint_utility"])
    prs = float(np.corrcoef(dy["fairness"], dy["joint_utility"])[0, 1])
    fq = dy["fairness"].quantile([1 / 3, 2 / 3]).to_numpy()
    lo_f = dy[dy.fairness <= fq[0]]
    hi_f = dy[dy.fairness >= fq[1]]
    d_tercile = float(hi_f.joint_utility.mean() - lo_f.joint_utility.mean())
    cl_tercile = C.cliffs_delta(hi_f.joint_utility.to_numpy(),
                                lo_f.joint_utility.to_numpy())

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.4))

    ax = axes[0]
    ax.hexbin(dy["joint_utility"], dy["fairness"], gridsize=34, cmap=C.SEQ,
              bins="log", mincnt=1, rasterized=True, linewidths=0)
    ax.errorbar(ctr, fm, yerr=[np.array(fm) - np.array(fl), np.array(fh) - np.array(fm)],
                fmt="o-", color=C.C_DEFECT, ecolor=C.C_DEFECT, lw=1.8, zorder=5,
                label="binned mean, clustered CI")
    top = grp["joint_utility"].max()
    ax.set_xlabel("dyad welfare  (mean utility of the pair)")
    ax.set_ylabel("fairness  (1 - Gini)")
    ax.set_title("a  the joint distribution is a V")
    ax.legend(loc="lower right", fontsize=6.5)
    C.annotate(ax, f"Spearman $\\rho$ = {rho.statistic:+.3f}\n"
                   f"Pearson r = {prs:+.3f}", loc="upper left")

    ax = axes[1]
    bot = np.zeros(len(ctr))
    for kcol, col, lab in [("p_CC", C.OUTCOME_COL["CC"], "CC mutual C"),
                           ("p_exploit", C.OUTCOME_COL["DC"], "CD or DC"),
                           ("p_DD", C.OUTCOME_COL["DD"], "DD mutual D")]:
        ax.bar(range(len(ctr)), comp[kcol], bottom=bot, color=col, width=.82, label=lab)
        bot += comp[kcol].to_numpy()
    ax.set_xticks(range(len(ctr)), [f"{c:.3f}" for c in ctr], rotation=90, fontsize=6.2)
    ax.set_xlabel("dyad welfare bin (bin centre)")
    ax.set_ylabel("share of rounds")
    ax.set_ylim(0, 1.46)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.set_title("b  why the V has a left arm")
    ax.legend(loc="upper center", ncol=3, fontsize=6.2)
    ax.text(.02, .80, "poorest dyads are DD-locked, so they are\n"
                      "equally poor and therefore perfectly fair;\n"
                      f"exploitation peaks at {comp['p_exploit'].max():.0%} of rounds "
                      "in the middle",
            transform=ax.transAxes, fontsize=6.3, color=C.INK2, va="top")

    ax = axes[2]
    rows = []
    for m in C.MODEL_ORDER:
        s = dy[dy.model == m]
        rows.append((m, stats.spearmanr(s.fairness, s.joint_utility).statistic))
    mrho = pd.DataFrame(rows, columns=["model", "rho"])
    _q = dy[dy.model == "Qwen3-235B-A22B"]
    qwen_low = float((_q.joint_utility <= .45).mean())
    ax.barh(range(6), mrho["rho"], color=[C.MODEL_COL[m] for m in mrho["model"]],
            height=.66)
    ax.axvline(0, color=C.INK, lw=.9)
    ax.axvline(rho.statistic, color=C.INK2, lw=.9, ls="--")
    ax.set_yticks(range(6), [m.replace("-Non-Reasoning", "") for m in mrho["model"]],
                  fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman $\\rho$ (fairness, welfare) within model")
    ax.set_title("c  one model reverses the sign")
    lo_r, hi_r = min(0.0, float(mrho["rho"].min())), float(mrho["rho"].max())
    pad = (hi_r - lo_r) * .22
    ax.set_xlim(lo_r - pad, hi_r + pad)
    for i, v in enumerate(mrho["rho"]):
        ax.text(v + (pad * .16 if v >= 0 else -pad * .16), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.8, color=C.INK2)
    ax.set_ylim(6.4, -0.6)
    C.annotate(ax, f"dashed = pooled {rho.statistic:+.2f}", loc="lower right")
    C.save(fig, "fairness", "FR03_fairness_welfare_tradeoff",
           "Is an equal split bought at the cost of total payoff, or do equality and "
           "welfare move together?",
           f"They move together on average, so there is no equality-efficiency "
           f"trade-off here: Spearman rho = {rho.statistic:+.3f}, and dyads in the top "
           f"fairness tercile earn {d_tercile:+.3f} more joint utility than the bottom "
           f"tercile (Cliff's delta {cl_tercile:+.2f}). The positive pooled association "
           f"nevertheless hides a V shape. The poorest dyads, locked into mutual "
           f"defection, are among the fairest ({fm[0]:.2f}) because they are equally "
           f"poor; fairness bottoms out at {min(fm):.2f} in the middle of the welfare "
           f"range where exploitation peaks at {comp['p_exploit'].max():.0%} of rounds, "
           f"then rises to {fm[-1]:.2f} at the top where mutual cooperation dominates. "
           f"The pooled sign is not universal: five models are positive but "
           f"Qwen3-235B reverses it (rho = "
           f"{float(mrho.set_index('model').loc['Qwen3-235B-A22B','rho']):+.2f}) "
           f"because {qwen_low:.0%} of its dyads are DD-locked at near-perfect "
           f"fairness, which puts its mass on the left arm of the V.",
           "hexbin + binned means, stacked composition, per-model correlation",
           "fairness, gini, joint_utility, p_CC, p_DD, p_exploit, model")

    tt = pd.DataFrame({"welfare_bin_centre": ctr, "mean_fairness": fm,
                       "ci_lo": fl, "ci_hi": fh,
                       "p_CC": comp["p_CC"].to_numpy(),
                       "p_exploit": comp["p_exploit"].to_numpy(),
                       "p_DD": comp["p_DD"].to_numpy(),
                       "max_joint_utility_in_bin": top.to_numpy(),
                       "n": grp.size().to_numpy()})
    C.savetab(tt, "fairness_welfare_frontier")
    C.record_test(section="fairness", test="Spearman correlation",
                  comparison="fairness vs joint_utility (dyads)",
                  statistic=rho.statistic, p_value=rho.pvalue, n=len(dy),
                  effect_size_note=f"top-vs-bottom fairness tercile differ by "
                                   f"{d_tercile:+.3f} joint utility, "
                                   f"Cliff's delta {cl_tercile:+.2f}")

    # ======================================================================
    # FR04  fairness versus cooperation
    # ======================================================================
    cb = np.round(np.arange(0, 1.01, .1), 2)
    dy["cbin"] = pd.cut(dy["joint_coop"], cb, include_lowest=True)
    g2 = dy.groupby("cbin", observed=True)
    cctr = np.array([iv.mid for iv in g2.groups.keys()])
    cm, cl_, ch = [], [], []
    for _, s in g2:
        m, l, h = _ci(s, "fairness")
        cm.append(m); cl_.append(l); ch.append(h)
    ex = g2["p_exploit"].mean()
    rho_c = stats.spearmanr(dy["fairness"], dy["joint_coop"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))

    ax = axes[0]
    ax.fill_between(cctr, cl_, ch, color=C.C_COOP, alpha=.2, lw=0)
    ax.plot(cctr, cm, "o-", color=C.C_COOP, zorder=3)
    ax.axvline(.5, color=C.MUTED, lw=.8, ls=":")
    ax.set_xlabel("dyad cooperation rate")
    ax.set_ylabel("fairness  (1 - Gini)")
    ax.set_title("a  fairness is U-shaped in cooperation")
    imin = int(np.argmin(cm))
    C.annotate(ax, f"minimum {min(cm):.2f} at cooperation {cctr[imin]:.2f}\n"
                   f"ends {cm[0]:.2f} and {cm[-1]:.2f}", loc="upper center")

    ax = axes[1]
    ax.plot(cctr, ex.to_numpy(), "o-", color=C.OUTCOME_COL["DC"])
    ax.axvline(.5, color=C.MUTED, lw=.8, ls=":")
    ax.set_xlabel("dyad cooperation rate")
    ax.set_ylabel("share of rounds that are CD or DC")
    ax.set_title("b  the mirror image: miscoordination")
    C.annotate(ax, "half-cooperating pairs are the ones that disagree,\n"
                   "and disagreement is the only source of inequality",
               loc="lower center")

    ax = axes[2]
    for m in C.MODEL_ORDER:
        s = dy[dy.model == m]
        gg = s.groupby("cbin", observed=True)["fairness"].agg(["mean", "size"])
        gg = gg[gg["size"] >= 40]
        ax.plot([iv.mid for iv in gg.index], gg["mean"], "o-", ms=3,
                color=C.MODEL_COL[m], lw=1.2,
                label=m.replace("-Non-Reasoning", ""))
    ax.set_xlabel("dyad cooperation rate")
    ax.set_ylabel("fairness  (1 - Gini)")
    ax.set_title("c  the U holds in every model")
    ax.set_ylim(0, 1.30)
    ax.legend(loc="upper center", fontsize=5.6, ncol=3)
    C.save(fig, "fairness", "FR04_fairness_vs_cooperation",
           "Does more cooperation buy a more equal split?",
           f"Not monotonically. Fairness is U-shaped in dyad cooperation: it is "
           f"{cm[0]:.2f} where the pair almost never cooperates and {cm[-1]:.2f} where "
           f"it almost always does, and it collapses to {min(cm):.2f} at a cooperation "
           f"rate of {cctr[imin]:.2f}. The reason is visible in panel b: intermediate "
           f"dyad cooperation is produced mainly by rounds in which the two agents "
           f"disagreed, and disagreement is the only mechanism that can create a gap. "
           f"The pooled rank correlation, {rho_c.statistic:+.3f}, is therefore a poor "
           f"summary of a non-monotone relation. The U is reproduced by all six models.",
           "line with CI + mirror line + per-model curves",
           "fairness, joint_coop, p_exploit, model")

    # ======================================================================
    # FR05  outcome fairness by model, lambda, language, pairing
    # ======================================================================
    cond_rows = []
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.3))
    panels = [
        ("model", C.MODEL_ORDER, "a  by model", [C.MODEL_COL[m] for m in C.MODEL_ORDER],
         [_short(m) for m in C.MODEL_ORDER]),
        ("scale", C.SCALES, "b  by payoff scale $\\lambda$",
         [C.C_COOP] * 10, [f"{s:g}" for s in C.SCALES]),
        ("language", C.LANGS, "c  by language", [C.LANG_COL[l] for l in C.LANGS],
         [C.LANG_LABEL[l] for l in C.LANGS]),
        ("dyad", C.DYADS, "d  by persona pairing", [C.DYAD_COL[x] for x in C.DYADS],
         C.DYADS),
    ]
    for ax, (key, order, title, cols, ticks) in zip(axes, panels):
        ms_, ls_, hs_ = [], [], []
        for lev in order:
            s = dy[dy[key] == lev]
            m, l, h = _ci(s, "fairness", n_boot=1500)
            ms_.append(m); ls_.append(l); hs_.append(h)
            gm, gl, gh = _ci(s, "utility_gap", n_boot=1500)
            cond_rows.append({"factor": key, "level": str(lev), "n_dyads": len(s),
                              "mean_fairness": m, "fair_ci_lo": l, "fair_ci_hi": h,
                              "mean_utility_gap": gm, "gap_ci_lo": gl, "gap_ci_hi": gh,
                              "mean_gini": s.gini.mean(),
                              "share_equal_split": float(s.equal.mean()),
                              "mean_joint_utility": s.joint_utility.mean(),
                              "mean_joint_coop": s.joint_coop.mean()})
        xx = np.arange(len(order))
        ax.bar(xx, ms_, color=cols, width=.68)
        ax.errorbar(xx, ms_, yerr=[np.array(ms_) - np.array(ls_),
                                   np.array(hs_) - np.array(ms_)],
                    fmt="none", ecolor=C.INK, lw=1.1)
        ax.set_xticks(xx, ticks, rotation=90, fontsize=6.4)
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        C.annotate(ax, f"range {max(ms_) - min(ms_):.3f}", loc="upper right")
        if key == "model":
            ax.set_ylabel("fairness  (1 - Gini), clustered 95% CI")
    axes[1].set_xlabel("payoff scale $\\lambda$")
    cdf = pd.DataFrame(cond_rows)
    rng_of = lambda f: (cdf[cdf.factor == f]["mean_fairness"].min(),
                        cdf[cdf.factor == f]["mean_fairness"].max())
    mlo, mhi = rng_of("model")
    slo, shi = rng_of("scale")
    llo, lhi = rng_of("language")
    dlo, dhi = rng_of("dyad")
    C.save(fig, "fairness", "FR05_fairness_by_condition",
           "Which design factor moves outcome fairness, and which leaves it alone?",
           f"Model identity dominates. Mean fairness spans {mlo:.2f} to {mhi:.2f} "
           f"across the six models, a range of {mhi-mlo:.3f}, and splits them into two "
           f"families: four models settle nearly every game evenly and two do not. The "
           f"payoff scale, which theory says must be irrelevant, moves fairness by only "
           f"{shi-slo:.3f} with no monotone trend. Language moves it by {lhi-llo:.3f} "
           f"and persona pairing by {dhi-dlo:.3f}: mixed cooperative-versus-selfish "
           f"pairs are markedly less equal ({dlo:.2f}) than same-persona pairs "
           f"({dhi:.2f}), which is what a persona label that actually bites should do.",
           "grouped bars with clustered bootstrap CI",
           "fairness, model, scale, language, dyad")
    C.savetab(cdf, "fairness_by_condition")

    # ======================================================================
    # FR06  TREATMENT fairness: the FAIRGAME language-disparity statistic
    # ======================================================================
    disp_rows = []
    pooled = {}
    for outcome in ["joint_coop", "joint_utility"]:
        res = perm_group_gap(dy[outcome], dy["language"], C.LANGS,
                             strata=dy["model"].astype(str), n_perm=10000, seed=1)
        pooled[outcome] = res
        disp_rows.append({"grouping": "language", "scope": "pooled (6 models)",
                          "outcome": outcome, "k_groups": 5,
                          "observed_gap": res["observed_gap"], "p_perm": res["p_perm"],
                          "null_mean_gap": res["null_mean_gap"],
                          "null_q95": res["null_q95"], "eta2": res["eta2"],
                          "gap_over_sd": res["gap_over_sd"], "n": res["n"],
                          "argmax": C.LANGS[int(np.nanargmax(res["means"]))],
                          "argmin": C.LANGS[int(np.nanargmin(res["means"]))]})
    per_model = {}
    for outcome in ["joint_coop", "joint_utility"]:
        rs = []
        for m in C.MODEL_ORDER:
            s = dy[dy.model == m]
            r = perm_group_gap(s[outcome], s["language"], C.LANGS, n_perm=5000, seed=2)
            rs.append(r)
            disp_rows.append({"grouping": "language", "scope": m, "outcome": outcome,
                              "k_groups": 5, "observed_gap": r["observed_gap"],
                              "p_perm": r["p_perm"], "null_mean_gap": r["null_mean_gap"],
                              "null_q95": r["null_q95"], "eta2": r["eta2"],
                              "gap_over_sd": r["gap_over_sd"], "n": r["n"],
                              "argmax": C.LANGS[int(np.nanargmax(r["means"]))],
                              "argmin": C.LANGS[int(np.nanargmin(r["means"]))]})
        per_model[outcome] = rs

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.4))

    ax = axes[0]
    ms_, ls_, hs_ = [], [], []
    for lg in C.LANGS:
        s = dy[dy.language == lg]
        m, l, h = _ci(s, "joint_coop", n_boot=2000)
        ms_.append(m); ls_.append(l); hs_.append(h)
    xx = np.arange(5)
    ax.bar(xx, ms_, color=[C.LANG_COL[l] for l in C.LANGS], width=.66)
    ax.errorbar(xx, ms_, yerr=[np.array(ms_) - np.array(ls_),
                               np.array(hs_) - np.array(ms_)],
                fmt="none", ecolor=C.INK, lw=1.1)
    ax.set_xticks(xx, [C.LANG_LABEL[l] for l in C.LANGS], rotation=90, fontsize=6.6)
    ax.set_ylabel("dyad cooperation rate")
    ax.set_title("a  pooled language means")
    ax.set_ylim(0, .78)
    ax.annotate("", xy=(float(np.argmax(ms_)), max(ms_)), xytext=(float(np.argmin(ms_)), max(ms_)),
                arrowprops=dict(arrowstyle="<->", color=C.INK, lw=.9))
    ax.text(2, max(ms_) + .015, f"gap = {max(ms_)-min(ms_):.3f}", ha="center",
            fontsize=6.8, color=C.INK)
    C.annotate(ax, f"permutation p = {pooled['joint_coop']['p_perm']:.4f}\n"
                   f"$\\eta^2$ = {pooled['joint_coop']['eta2']:.4f}", loc="upper left")

    ax = axes[1]
    nl = pooled["joint_coop"]["null"]
    ax.hist(nl, bins=50, color=C.MUTED, rwidth=.9)
    ax.axvline(pooled["joint_coop"]["observed_gap"], color=C.C_DEFECT, lw=1.6)
    ax.set_xlabel("max - min of language means (cooperation)")
    ax.set_ylabel("permutations (of 10,000)")
    ax.set_title("b  permutation null, pooled")
    ax.set_xlim(0, max(pooled["joint_coop"]["observed_gap"] * 1.15, nl.max() * 1.05))
    C.annotate(ax, f"observed {pooled['joint_coop']['observed_gap']:.3f}\n"
                   f"null mean {nl.mean():.3f}  (the bias floor)\n"
                   f"p = {pooled['joint_coop']['p_perm']:.4f}", loc="upper center")

    for ax, outcome, lab in [(axes[2], "joint_coop", "cooperation"),
                             (axes[3], "joint_utility", "welfare")]:
        rs = per_model[outcome]
        gaps = np.array([r["observed_gap"] for r in rs])
        nulls = np.array([r["null_q95"] for r in rs])
        ps = C.bh_fdr(np.array([r["p_perm"] for r in rs]))
        yy = np.arange(6)
        ax.barh(yy, gaps, color=[C.MODEL_COL[m] for m in C.MODEL_ORDER], height=.62)
        ax.scatter(nulls, yy, marker="|", s=90, color=C.INK, zorder=4,
                   label="95th pct of null")
        ax.set_yticks(yy, [m.replace("-Non-Reasoning", "") for m in C.MODEL_ORDER],
                      fontsize=6.6)
        ax.invert_yaxis()
        ax.set_xlabel(f"language gap in {lab} (max - min)")
        ax.set_title(("c  " if outcome == "joint_coop" else "d  ") +
                     f"per-model gap, {lab}")
        ax.set_xlim(0, gaps.max() * 1.34)
        ax.set_ylim(6.9, -0.6)
        for i, (g, p) in enumerate(zip(gaps, ps)):
            ax.text(g + gaps.max() * .02, i,
                    f"{g:.3f} {'*' if p < .05 else 'ns'}", va="center", fontsize=6.3,
                    color=C.INK2)
        ax.legend(loc="lower right", fontsize=6)
        pooled_gap = pooled[outcome]["observed_gap"]
        ax.axvline(pooled_gap, color=C.INK2, lw=.9, ls="--")
        C.annotate(ax, f"dashed = pooled gap {pooled_gap:.3f}\n"
                       f"* = FDR p < 0.05", loc="lower left")
    top_lang_txt = ", ".join(
        f"{m.replace('-Non-Reasoning','')}: {C.LANG_LABEL[C.LANGS[int(np.nanargmax(r['means']))]]}"
        for m, r in zip(C.MODEL_ORDER, per_model["joint_coop"]))
    C.save(fig, "fairness", "FR06_language_disparity",
           "Does a model treat the five language versions of an identical game alike, "
           "and how large is the disparity relative to what label noise alone produces?",
           f"No. Pooled over models the language gap in dyad cooperation is only "
           f"{pooled['joint_coop']['observed_gap']:.3f} "
           f"(permutation p = {pooled['joint_coop']['p_perm']:.4f}), but pooling badly "
           f"understates the disparity because the models disagree about which "
           f"language is the cooperative one. Within models the gap runs from "
           f"{min(r['observed_gap'] for r in per_model['joint_coop']):.3f} to "
           f"{max(r['observed_gap'] for r in per_model['joint_coop']):.3f}, and "
           f"{int(sum(C.bh_fdr(np.array([r['p_perm'] for r in per_model['joint_coop']])) < .05))} "
           f"of 6 models exceed their own permutation null after FDR correction. The "
           f"most cooperative language is not the same one for every model "
           f"({top_lang_txt}), so this is not a single high-resource-language effect "
           f"but a set of model-specific disparities that cancel on average.",
           "bars with clustered CI + permutation null + per-model forest",
           "joint_coop, joint_utility, language, model")

    # other groupings, same statistic, for the disparity table
    for grouping, col, order, strata in [
            ("persona pairing", "dyad", C.DYADS, dy["model"].astype(str)),
            ("model", "model", C.MODEL_ORDER, None),
            ("payoff scale", "scale", C.SCALES, dy["model"].astype(str))]:
        for outcome in ["joint_coop", "joint_utility"]:
            r = perm_group_gap(dy[outcome], dy[col], order, strata=strata,
                               n_perm=5000, seed=3)
            disp_rows.append({"grouping": grouping, "scope": "pooled (6 models)",
                              "outcome": outcome, "k_groups": len(order),
                              "observed_gap": r["observed_gap"], "p_perm": r["p_perm"],
                              "null_mean_gap": r["null_mean_gap"],
                              "null_q95": r["null_q95"], "eta2": r["eta2"],
                              "gap_over_sd": r["gap_over_sd"], "n": r["n"],
                              "argmax": str(order[int(np.nanargmax(r["means"]))]),
                              "argmin": str(order[int(np.nanargmin(r["means"]))])})
    # Seat, at the agent-game level. Permuting the seat label within a game is
    # exactly a random sign flip of the paired difference, so that is how it is
    # done: with 12,000 two-element strata a generic label shuffle would be both
    # slower and identical in distribution.
    for outcome, col in [("coop_rate", "coop_rate"), ("utility", "utility")]:
        w1 = ag[ag.agent == 1].set_index("game_uid")[col]
        w2 = ag[ag.agent == 2].set_index("game_uid")[col]
        dif = (w1 - w2.reindex(w1.index)).dropna().to_numpy()
        t = signflip_test(dif, n_perm=5000, seed=4)
        null_abs = np.abs(np.array([np.mean(dif * (np.random.default_rng(40 + i)
                                                   .integers(0, 2, len(dif)) * 2 - 1))
                                    for i in range(400)]))
        y = ag[col].to_numpy()
        disp_rows.append({"grouping": "seat (agent 1 vs 2)", "scope": "pooled (6 models)",
                          "outcome": outcome, "k_groups": 2,
                          "observed_gap": abs(t["mean"]), "p_perm": t["p_perm"],
                          "null_mean_gap": float(null_abs.mean()),
                          "null_q95": float(np.percentile(null_abs, 95)),
                          "eta2": float((abs(t["mean"]) / 2) ** 2 * 2 * len(dif)
                                        / np.sum((y - y.mean()) ** 2)),
                          "gap_over_sd": abs(t["mean"]) / y.std(),
                          "n": int(2 * len(dif)),
                          "argmax": "1" if t["mean"] > 0 else "2",
                          "argmin": "2" if t["mean"] > 0 else "1"})
    disp = pd.DataFrame(disp_rows)
    disp["p_fdr"] = np.nan
    for (g, o), idx in disp.groupby(["grouping", "outcome"]).groups.items():
        disp.loc[idx, "p_fdr"] = C.bh_fdr(disp.loc[idx, "p_perm"].to_numpy())
    C.savetab(disp, "fairness_group_disparity")
    for _, rr_ in disp[disp.grouping == "language"].iterrows():
        C.record_test(section="fairness",
                      test="FAIRGAME max-min group gap, label permutation",
                      comparison=f"language disparity in {rr_.outcome} [{rr_.scope}]",
                      statistic=rr_.observed_gap, p_value=rr_.p_perm, n=rr_.n,
                      effect_size_note=f"gap/SD = {rr_.gap_over_sd:.3f}, "
                                       f"eta2 = {rr_.eta2:.4f}, null mean gap "
                                       f"{rr_.null_mean_gap:.3f}")

    # ======================================================================
    # FR07  seat / role symmetry: the design validation
    # ======================================================================
    sym = dy[dy.dyad.isin(["CvC", "SvS"])]
    seat_rows = []
    pooled_sym = signflip_test(sym["signed_gap"].to_numpy(), n_perm=5000, seed=5)
    seat_rows.append({"scope": "all models", "dyad": "CvC + SvS", **pooled_sym})
    for m in C.MODEL_ORDER:
        for dd in ["CvC", "SvS"]:
            s = sym[(sym.model == m) & (sym.dyad == dd)]
            t = signflip_test(s["signed_gap"].to_numpy(), n_perm=5000, seed=6)
            seat_rows.append({"scope": m, "dyad": dd, **t})
    seat = pd.DataFrame(seat_rows)
    seat["p_fdr"] = C.bh_fdr(seat["p_perm"].to_numpy())
    C.savetab(seat.drop(columns=["lo", "hi"]), "fairness_seat_symmetry")

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))

    ax = axes[0]
    ms_, ls_, hs_ = [], [], []
    for dd in C.DYADS:
        s = dy[dy.dyad == dd]
        m, l, h = _ci(s, "signed_gap", n_boot=2000)
        ms_.append(m); ls_.append(l); hs_.append(h)
    xx = np.arange(4)
    ax.axhline(0, color=C.INK, lw=1.0)
    ax.bar(xx, ms_, color=[C.DYAD_COL[x] for x in C.DYADS], width=.62)
    ax.errorbar(xx, ms_, yerr=[np.array(ms_) - np.array(ls_),
                               np.array(hs_) - np.array(ms_)],
                fmt="none", ecolor=C.INK, lw=1.2)
    ax.set_xticks(xx, C.DYADS)
    ax.set_ylabel("signed gap  $u_1 - u_2$")
    ax.set_xlabel("persona pairing (focal seat first)")
    ax.set_title("a  seat advantage by pairing")
    C.annotate(ax, "CvC and SvS have identical\npersonas, so zero is the\n"
                   "specification check", loc="upper left")
    C.annotate(ax, f"CvC {ms_[0]:+.3f}, SvS {ms_[3]:+.3f}", loc="lower left")

    ax = axes[1]
    yy = np.arange(6)
    ax.axvline(0, color=C.INK, lw=1.0)
    for j, (dd, mk) in enumerate([("CvC", "o"), ("SvS", "s")]):
        errs = []
        for m in C.MODEL_ORDER:
            s = sym[(sym.model == m) & (sym.dyad == dd)]
            mm, ll, hh = _ci(s, "signed_gap", n_boot=1200)
            errs.append((mm, ll, hh))
        errs = np.array(errs)
        ax.errorbar(errs[:, 0], yy + (j - .5) * .26,
                    xerr=[errs[:, 0] - errs[:, 1], errs[:, 2] - errs[:, 0]],
                    fmt=mk, ms=4, color=C.INK2 if j else C.C_COOP,
                    ecolor=C.INK2 if j else C.C_COOP, lw=1.1, label=dd)
    ax.set_yticks(yy, [m.replace("-Non-Reasoning", "") for m in C.MODEL_ORDER],
                  fontsize=6.6)
    ax.invert_yaxis()
    ax.set_ylim(7.2, -1.3)
    ax.set_xlabel("signed gap in same-persona games")
    ax.set_title("b  same-persona games by model")
    ax.legend(loc="lower right", fontsize=6.5)
    worst = seat[(seat.scope != "all models")].reindex(
        seat[(seat.scope != "all models")]["mean"].abs().sort_values(ascending=False).index)
    wv = worst.iloc[0]
    n_seat_sig = int((seat[seat.scope != "all models"].p_fdr < .05).sum())
    n_seat_big = int((seat[seat.scope != "all models"].cohens_d.abs() > .5).sum())
    C.annotate(ax, f"{n_seat_sig} of 12 cells reject symmetry (FDR),\n"
                   f"but only {n_seat_big} are larger than |d| = 0.5. Largest: "
                   f"{wv.scope.replace('-Non-Reasoning','')} "
                   f"{wv.dyad} {wv['mean']:+.3f}, d = {wv.cohens_d:+.2f}",
               loc="upper left")

    ax = axes[2]
    # mirror check: the cooperative persona's utility in each of the two seats
    cvs = dy[dy.dyad == "CvS"]
    svc = dy[dy.dyad == "SvC"]
    pairs = [("cooperative persona\nin seat 1 (CvS)", cvs["utility_1"]),
             ("cooperative persona\nin seat 2 (SvC)", svc["utility_2"]),
             ("selfish persona\nin seat 2 (CvS)", cvs["utility_2"]),
             ("selfish persona\nin seat 1 (SvC)", svc["utility_1"])]
    vals, los_, his_ = [], [], []
    for lab_, v in pairs:
        src = cvs if "CvS" in lab_ else svc
        col = v.name
        m, l, h = C.cluster_boot_ci(v.to_numpy(), src["cell"].to_numpy(), n_boot=1500)
        vals.append(m); los_.append(l); his_.append(h)
    xx = np.arange(4)
    cols = [C.C_COOP, C.C_COOP, C.C_DEFECT, C.C_DEFECT]
    ax.bar(xx, vals, color=cols, width=.62, alpha=.9)
    ax.errorbar(xx, vals, yerr=[np.array(vals) - np.array(los_),
                                np.array(his_) - np.array(vals)],
                fmt="none", ecolor=C.INK, lw=1.2)
    ax.set_xticks(xx, [p[0] for p in pairs], rotation=25, fontsize=5.8, ha="right")
    ax.set_ylabel("mean utility")
    ax.set_ylim(0.45, 0.68)
    ax.set_title("c  mirror test: CvS against SvC")
    mirror_c = abs(vals[0] - vals[1])
    mirror_s = abs(vals[2] - vals[3])
    C.annotate(ax, f"same persona, seats swapped:\ncooperative differs by {mirror_c:.3f}\n"
                   f"selfish differs by {mirror_s:.3f}", loc="upper right")
    C.save(fig, "fairness", "FR07_seat_symmetry",
           "Agent 1 and agent 2 receive structurally identical prompts, so is there a "
           "seat advantage that no feature of the game can justify?",
           f"Negligible in aggregate, large in two model-by-pairing cells. Pooled over "
           f"the same-persona pairings the signed gap is {pooled_sym['mean']:+.4f} "
           f"utility units: statistically distinguishable from zero (sign-flip "
           f"permutation p = {pooled_sym['p_perm']:.4f}) but negligible in size "
           f"(Cohen's d = {pooled_sym['cohens_d']:+.3f}), which is the design "
           f"validation the corpus needs. Broken out, {n_seat_sig} of 12 model-by-"
           f"pairing cells reject symmetry after FDR correction, yet only "
           f"{n_seat_big} exceed |d| = 0.5, and both of those are selfish-versus-"
           f"selfish games where the two agents are indistinguishable by construction: "
           f"{wv.scope.replace('-Non-Reasoning','')} reaches {wv['mean']:+.3f} "
           f"(d = {wv.cohens_d:+.2f}). The CvS against SvC mirror test agrees that the "
           f"persona, not the seat, carries the effect: swapping seats moves the "
           f"cooperative persona's utility by {mirror_c:.3f} and the selfish "
           f"persona's by {mirror_s:.3f}.",
           "bars with clustered CI + per-model forest + mirror comparison",
           "signed_gap, dyad, model, utility_1, utility_2")
    C.record_test(section="fairness", test="sign-flip permutation, mean signed gap = 0",
                  comparison="seat symmetry in same-persona dyads (CvC + SvS)",
                  statistic=pooled_sym["mean"], p_value=pooled_sym["p_perm"],
                  n=pooled_sym["n"],
                  effect_size_note=f"Cohen's d = {pooled_sym['cohens_d']:+.3f}, "
                                   "negligible")

    # ======================================================================
    # FR08  does the persona label actually change behaviour?
    # ======================================================================
    mx = ag[ag.dyad.isin(["CvS", "SvC"])]
    per_rows = []
    for scope, s in [("pooled", mx)] + [(m, mx[mx.model == m]) for m in C.MODEL_ORDER]:
        co = s[s.personality == "cooperative"]
        se = s[s.personality == "selfish"]
        # paired within game: the two agents of one mixed dyad
        pv = s.pivot_table(index="game_uid", columns="personality",
                           values="coop_rate", observed=True)
        pv = pv.dropna()
        dif = (pv["selfish"] - pv["cooperative"]).to_numpy()
        mdif, ldif, hdif = C.cluster_boot_ci(dif, np.arange(len(dif)), n_boot=1500)
        pvu = s.pivot_table(index="game_uid", columns="personality",
                            values="utility", observed=True).dropna()
        du = (pvu["selfish"] - pvu["cooperative"]).to_numpy()
        mu, lu, hu = C.cluster_boot_ci(du, np.arange(len(du)), n_boot=1500)
        # round 1 happens before either agent has seen anything, so this isolates
        # the prompt effect of the label from everything the game then does to it
        pvf = s.pivot_table(index="game_uid", columns="personality",
                            values="first_coop", observed=True).dropna()
        d1 = (pvf["selfish"] - pvf["cooperative"]).to_numpy()
        m1, l1, h1 = C.cluster_boot_ci(d1, np.arange(len(d1)), n_boot=1500)
        w = stats.wilcoxon(dif) if np.any(dif != 0) else None
        per_rows.append({
            "scope": scope, "n_dyads": len(pv),
            "coop_cooperative": co.coop_rate.mean(), "coop_selfish": se.coop_rate.mean(),
            "paired_delta_coop": mdif, "coop_ci_lo": ldif, "coop_ci_hi": hdif,
            "cliffs_delta_coop": C.cliffs_delta(se.coop_rate.to_numpy(),
                                                co.coop_rate.to_numpy()),
            "cohens_d_coop": C.cohens_d(se.coop_rate.to_numpy(), co.coop_rate.to_numpy()),
            "first_coop_cooperative": co.first_coop.mean(),
            "first_coop_selfish": se.first_coop.mean(),
            "paired_delta_first": m1, "first_ci_lo": l1, "first_ci_hi": h1,
            "utility_cooperative": co.utility.mean(), "utility_selfish": se.utility.mean(),
            "paired_delta_utility": mu, "util_ci_lo": lu, "util_ci_hi": hu,
            "wilcoxon_p": w.pvalue if w is not None else np.nan})
    per = pd.DataFrame(per_rows)
    per["p_fdr"] = C.bh_fdr(per["wilcoxon_p"].to_numpy())
    # In a symmetric two-player matrix the two agents differ in payoff only in the
    # rounds where they acted differently, so the utility difference is exactly the
    # negative of the cooperation difference. Verified rather than assumed.
    ident = float((per["paired_delta_utility"] + per["paired_delta_coop"]).abs().max())
    C.savetab(per, "fairness_persona_effect")

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))
    pm = per[per.scope != "pooled"].set_index("scope").reindex(C.MODEL_ORDER)
    pooled_row = per[per.scope == "pooled"].iloc[0]

    ax = axes[0]
    yy = np.arange(6)
    ax.plot([pm.coop_cooperative, pm.coop_selfish], [yy, yy], color=C.MUTED, lw=1.0,
            zorder=1)
    ax.scatter(pm.coop_cooperative, yy, s=42, color=C.C_COOP, zorder=3,
               label="'cooperative' persona")
    ax.scatter(pm.coop_selfish, yy, s=42, color=C.C_DEFECT, zorder=3,
               label="'selfish' persona")
    ax.set_yticks(yy, [m.replace("-Non-Reasoning", "") for m in pm.index], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_ylim(7.4, -1.1)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("cooperation rate in mixed pairings")
    ax.set_title("a  what the persona label does")
    ax.legend(loc="lower center", fontsize=6.2, ncol=2)
    C.annotate(ax, "paired within game, so the opponent and every\n"
                   "design factor are held fixed", loc="upper left")

    ax = axes[1]
    ax.axvline(0, color=C.INK, lw=1.0)
    err = np.array([pm.paired_delta_coop - pm.coop_ci_lo,
                    pm.coop_ci_hi - pm.paired_delta_coop])
    ax.errorbar(pm.paired_delta_coop, yy, xerr=err, fmt="o", ms=5,
                color=C.INK2, ecolor=C.INK2, lw=1.2)
    for i, m in enumerate(pm.index):
        ax.scatter(pm.paired_delta_coop.iloc[i], i, s=48, color=C.MODEL_COL[m], zorder=4)
    ax.axvline(pooled_row.paired_delta_coop, color=C.MUTED, lw=1.0, ls="--")
    ax.set_yticks(yy, [m.replace("-Non-Reasoning", "") for m in pm.index], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_ylim(7.4, -0.6)
    ax.set_xlabel("paired $\\Delta$ cooperation  (selfish $-$ cooperative)")
    ax.set_title("b  the pooled effect is an artefact")
    C.annotate(ax, f"pooled {pooled_row.paired_delta_coop:+.3f} (dashed)\n"
                   f"per-model range {pm.paired_delta_coop.min():+.2f} to "
                   f"{pm.paired_delta_coop.max():+.2f}\n"
                   "95% CIs are narrower than the markers", loc="lower left")

    ax = axes[2]
    ax.axvline(0, color=C.INK, lw=1.0)
    err = np.array([pm.paired_delta_first - pm.first_ci_lo,
                    pm.first_ci_hi - pm.paired_delta_first])
    ax.errorbar(pm.paired_delta_first, yy, xerr=err, fmt="o", ms=5,
                color=C.INK2, ecolor=C.INK2, lw=1.2)
    for i, m in enumerate(pm.index):
        ax.scatter(pm.paired_delta_first.iloc[i], i, s=48, color=C.MODEL_COL[m],
                   zorder=4)
    ax.set_yticks(yy, [m.replace("-Non-Reasoning", "") for m in pm.index], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_ylim(7.4, -0.6)
    ax.set_xlabel("paired $\\Delta$ round-1 cooperation  (selfish $-$ cooperative)")
    ax.set_title("c  the label acts on the opening move")
    r_cu = float(np.corrcoef(pm.paired_delta_coop, pm.paired_delta_utility)[0, 1])
    C.annotate(ax, "round 1 precedes any interaction, so this is the\n"
                   "prompt effect alone. Utility is not shown: it is the\n"
                   f"exact negative of panel b (residual {ident:.0e}).",
               loc="lower left")
    C.save(fig, "fairness", "FR08_persona_effect",
           "Does labelling one agent 'selfish' actually change how it plays and what "
           "share of the proceeds it takes?",
           f"Enormously in four models, and in opposite directions, so the pooled "
           f"effect is meaningless. Pooled over models the paired difference in "
           f"cooperation between the selfish-labelled and cooperative-labelled agent "
           f"of the same mixed dyad is only {pooled_row.paired_delta_coop:+.3f}, but "
           f"the per-model values run from {pm.paired_delta_coop.min():+.3f} to "
           f"{pm.paired_delta_coop.max():+.3f}. Gemini-3.1-Flash-Lite obeys the label "
           f"(the selfish agent cooperates {abs(pm.loc['Gemini-3.1-Flash-Lite','paired_delta_coop']):.2f} "
           f"less) while Qwen3-235B inverts it "
           f"({pm.loc['Qwen3-235B-A22B','paired_delta_coop']:+.2f}). The effect is "
           f"already present in round 1, before either agent has seen anything, so it "
           f"is a prompt effect and not something the game dynamics produce. Utility "
           f"needs no separate panel: in a symmetric matrix the utility difference is "
           f"the exact negative of the cooperation difference (residual {ident:.0e}), "
           f"so in this penalty-framed game obeying the cooperative label costs payoff "
           f"by construction.",
           "dumbbell + paired-difference forest + round-1 forest",
           "coop_rate, first_coop, utility, personality, dyad, model")

    # ======================================================================
    # findings
    # ======================================================================
    lang_sig = int(np.sum(C.bh_fdr(np.array([r["p_perm"]
                                             for r in per_model["joint_coop"]])) < .05))
    lang_gaps = [r["observed_gap"] for r in per_model["joint_coop"]]

    C.record_finding(
        id="F-gap-is-exploitation",
        finding="Within-dyad inequality in this corpus is not an independent construct: "
                "it is net exploitation re-expressed in utility units.",
        evidence=f"With a symmetric matrix the two agents earn the same in any round "
                 f"they act alike, so utility_gap = |p_DC - p_CD| exactly; the maximum "
                 f"residual over 12,000 dyads is {resid:.0e}. "
                 f"{(dy.n_asym == 0).mean():.1%} of dyads never miscoordinate and are "
                 f"equal by construction, {gap0:.1%} end perfectly equal in total, and "
                 f"the gap correlates with total exploitation at r = {r_gap_ex:.3f}, "
                 f"falling short of 1 only where exploitation runs both ways "
                 f"({1-one_dir:.0%} of asymmetric games).",
        figure="05_fairness/FR02_gap_decomposition.png, FR01_outcome_gap_distribution.png",
        strength="strong",
        robustness="algebraic identity, verified numerically on every dyad; holds in "
                   "every model, language, pairing and payoff scale",
        interpretation="Any claim that a model is 'fair' in outcome terms in this "
                       "design is a claim about how often its two copies fail to "
                       "coordinate, nothing more. Outcome fairness and treatment "
                       "fairness must not be reported as if they were two readings of "
                       "the same quantity.",
        caveat="This identity is a property of a symmetric two-player matrix with "
               "self-play; it would not hold with asymmetric payoffs or cross-model "
               "play.",
        claim="Within-dyad payoff inequality is an exact arithmetic function of "
              "one-directional exploitation, so it measures miscoordination rather "
              "than any distributional disposition of the model.")

    C.record_finding(
        id="F-no-equality-efficiency-tradeoff",
        finding="Equality and welfare are complements here, not a trade-off, but the "
                "pooled correlation hides a V-shaped relation.",
        evidence=f"Spearman rho(fairness, joint utility) = {rho.statistic:+.3f}; dyads "
                 f"in the top fairness tercile earn {d_tercile:+.3f} more joint utility "
                 f"than the bottom tercile (Cliff's delta {cl_tercile:+.2f}). The "
                 f"binned frontier is non-monotone: fairness is {fm[0]:.2f} in the "
                 f"poorest welfare bin, which is DD-locked and therefore equally poor, "
                 f"falls to {min(fm):.2f} in the middle where exploitation reaches "
                 f"{comp['p_exploit'].max():.0%} of rounds, and rises to {fm[-1]:.2f} "
                 f"at the top. Against cooperation the same U appears, with a minimum "
                 f"of {min(cm):.2f} at a dyad cooperation rate of {cctr[imin]:.2f}.",
        figure="05_fairness/FR03_fairness_welfare_tradeoff.png, "
               "FR04_fairness_vs_cooperation.png",
        strength="strong",
        robustness="the V shape reproduces within every model, but the pooled rank "
                   f"correlation does not: it runs {mrho['rho'].min():+.2f} to "
                   f"{mrho['rho'].max():+.2f} and is negative for Qwen3-235B, whose "
                   f"dyads concentrate on the left arm ({qwen_low:.0%} DD-locked at "
                   "fairness 0.98)",
        interpretation="Correlational, not causal: nothing here manipulates fairness. "
                       "The mechanism is compositional - mutual defection and mutual "
                       "cooperation are both perfectly equal splits, so equality peaks "
                       "at both ends of the welfare range and only the exploitative "
                       "middle is unequal. The Qwen3-235B reversal is the same "
                       "mechanism seen from a different sampling of the curve, not a "
                       "different relationship, which is why a pooled correlation is "
                       "the wrong summary statistic for this pair of variables.",
        caveat="Because the poorest and the richest dyads are both maximally fair, a "
               "single fairness number cannot be read as a welfare signal; it must be "
               "conditioned on the cooperation level, and the sign of the pooled "
               "correlation depends on where a model's dyads sit on the curve.",
        claim="More equal dyads are also richer on average, but the association is "
              "V-shaped rather than monotone because mutual defection is a perfectly "
              "equal outcome.")

    C.record_finding(
        id="F-language-disparity-cancels",
        finding="Pooling across models understates language disparity by roughly a "
                "factor of four, because models disagree about which language is the "
                "cooperative one.",
        evidence=f"The FAIRGAME max-minus-min gap in dyad cooperation is "
                 f"{pooled['joint_coop']['observed_gap']:.3f} pooled "
                 f"(permutation p = {pooled['joint_coop']['p_perm']:.4f}, 10,000 "
                 f"permutations stratified by model), against a permutation bias floor "
                 f"of {pooled['joint_coop']['null_mean_gap']:.3f}. Within models the "
                 f"gap runs {min(lang_gaps):.3f} to {max(lang_gaps):.3f} with a median "
                 f"of {float(np.median(lang_gaps)):.3f}, and {lang_sig} of 6 models are "
                 f"FDR-significant. The most cooperative language differs by model "
                 f"({top_lang_txt}).",
        figure="05_fairness/FR06_language_disparity.png",
        strength="strong",
        robustness="permutation test at the game level with group sizes fixed by the "
                   "balanced design; FDR across the six models; the same pattern "
                   "appears for welfare as for cooperation",
        interpretation="A single cross-model fairness audit would conclude that these "
                       "systems are close to language-neutral. They are not; the "
                       "disparities are large and simply point in different "
                       "directions, so they cancel in the average. Auditing must be "
                       "per model.",
        caveat="The Arabic and Chinese prompts carry an inherited FAIRGAME translation "
               "quirk - their closing sentence says 'maximise your rewards' while the "
               "payoff sentences say 'penalty' - so part of any ar or cn effect is "
               "prompt inconsistency rather than language. English, French and "
               "Vietnamese are internally consistent, and the largest single-model "
               "gaps involve those three as well.",
        claim="Language disparity in LLM game play is large within each model but "
              "nearly cancels when models are pooled, so aggregate fairness audits "
              "systematically understate it.")

    C.record_finding(
        id="F-persona-label-inverted",
        finding="The persona label is honoured, ignored or inverted depending on the "
                "model, which is why it has almost no pooled effect.",
        evidence=f"In mixed cooperative-versus-selfish dyads the paired difference in "
                 f"cooperation (selfish minus cooperative agent of the same game) is "
                 f"{pooled_row.paired_delta_coop:+.3f} pooled but ranges from "
                 f"{pm.paired_delta_coop.min():+.3f} (Gemini-3.1-Flash-Lite, the label "
                 f"obeyed) to {pm.paired_delta_coop.max():+.3f} (Qwen3-235B, the label "
                 f"inverted). GPT-5.4-Nano is essentially insensitive "
                 f"({pm.loc['GPT-5.4-Nano','paired_delta_coop']:+.3f}). The split is "
                 f"already there in round 1 (paired difference "
                 f"{pm.paired_delta_first.min():+.2f} to "
                 f"{pm.paired_delta_first.max():+.2f}), before any interaction. Utility "
                 f"is the exact negative of cooperation here (residual {ident:.0e}, "
                 f"r = {r_cu:+.3f}), and the two persona-sensitive models are exactly "
                 f"the two with the lowest outcome fairness.",
        figure="05_fairness/FR08_persona_effect.png, FR05_fairness_by_condition.png",
        strength="strong",
        robustness="paired within game, so opponent and every design factor are held "
                   "fixed; Wilcoxon FDR-significant for four of six models",
        interpretation="Persona prompting is not a reliable behavioural lever across "
                       "models. In a penalty-framed game the label appears to be "
                       "resolved against the numbers rather than with them in at least "
                       "one model, and because cooperating more means being exploited "
                       "more, obeying the cooperative label costs payoff.",
        caveat="Only two persona words in one prompt family, and Qwen's inversion "
               "could reflect a translation or framing interaction this design cannot "
               "separate from a genuine dispositional inversion.",
        claim="Assigning an LLM a 'selfish' persona changes its cooperation by up to "
              "0.8 in one model and by the same amount in the opposite direction in "
              "another, so persona effects reported as a single pooled number are "
              "uninterpretable.")

    C.record_finding(
        id="F-seat-symmetry-mostly-holds",
        finding="Seat position is negligible overall, which validates the design, "
                "but two models take a large systematic seat advantage in games where "
                "the two agents are indistinguishable.",
        evidence=f"Across CvC and SvS games, where both agents carry the same persona "
                 f"and structurally identical prompts, the mean signed gap is "
                 f"{pooled_sym['mean']:+.4f} utility units: sign-flip permutation "
                 f"p = {pooled_sym['p_perm']:.4f} but Cohen's d = "
                 f"{pooled_sym['cohens_d']:+.3f}, so it is detectable and negligible "
                 f"at once. {n_seat_sig} of 12 model-by-pairing cells reject symmetry "
                 f"after FDR correction and {n_seat_big} exceed |d| = 0.5; the largest "
                 f"is {wv.scope.replace('-Non-Reasoning','')} in {wv.dyad} at "
                 f"{wv['mean']:+.3f} (d = {wv.cohens_d:+.2f}, p = {wv.p_perm:.4f}). The "
                 f"CvS against SvC mirror test moves the cooperative persona's utility "
                 f"by only {mirror_c:.3f} when the seats are swapped.",
        figure="05_fairness/FR07_seat_symmetry.png",
        strength="moderate",
        robustness="exact sign-flip permutation, 5,000 draws, FDR across the twelve "
                   "model-by-pairing cells",
        interpretation="The corpus-level null is a genuine specification check: "
                       "position in the prompt does not confound the main results. "
                       "The per-model violations are a within-game symmetry-breaking "
                       "that then locks in, since both agents open the game "
                       "identically and diverge later, so this is emergent asymmetry "
                       "rather than a round-1 prompt artefact.",
        caveat="Agent names differ between the two seats in the FAIRGAME prompts, so "
               "a name effect cannot be separated from a pure position effect with "
               "this corpus.",
        claim="Seat position carries a negligible advantage in aggregate (d = 0.03), "
              "confirming that the prompts are symmetric, yet individual models still "
              "break the symmetry within a game and one of them does so by nearly half "
              "a utility unit.")

    return disp, per, seat
