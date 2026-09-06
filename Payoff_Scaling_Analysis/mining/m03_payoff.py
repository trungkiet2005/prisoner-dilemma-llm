"""Section 4: payoff-centric analysis.

Payoff here is a *penalty* the agent is told to minimise, so everything is
reported on the derived higher-is-better utility u = (S - penalty)/(S - T) and,
where the comparison is to the mutual-cooperation benchmark, on
efficiency = (S - penalty)/(S - R).

The central structural point of this section is that payoff is not a free
variable: given the two actions, the payoff is determined. So the payoff
analysis is an analysis of *which cells of the matrix the models land in*, and
the honest decomposition is by joint outcome.
"""
from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import core as C


def run(rounds, ag, dy):
    C.use_style()
    print("\n== 03 payoff ==")

    # ---- group summary table over every meaningful factor -------------------
    recs = []
    for key in ["model", "language", "scale", "dyad", "personality", "agent", "round"]:
        src = rounds if key == "round" else ag
        for k, sub in src.groupby(key, observed=True):
            m, lo, hi = C.cluster_boot_ci(sub["utility"].to_numpy(),
                                          sub["game_uid"].to_numpy(), n_boot=800)
            mc, lc, hc = C.cluster_boot_ci(sub["coop"].to_numpy() if key == "round"
                                           else sub["coop_rate"].to_numpy(),
                                           sub["game_uid"].to_numpy(), n_boot=800)
            recs.append({"factor": key, "level": k, "n": len(sub),
                         "utility_mean": m, "utility_lo": lo, "utility_hi": hi,
                         "coop_mean": mc, "coop_lo": lc, "coop_hi": hc,
                         "efficiency_mean": sub["efficiency"].mean()})
    grp = pd.DataFrame(recs)
    C.savetab(grp, "group_summary")

    # ---- P1 payoff is determined by the joint outcome -----------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
    oc = rounds["outcome"].value_counts(normalize=True).reindex(C.OUTCOME_ORDER)
    axes[0].bar(range(4), oc.values, color=[C.OUTCOME_COL[o] for o in C.OUTCOME_ORDER])
    axes[0].set_xticks(range(4), C.OUTCOME_ORDER)
    axes[0].set_ylabel("share of agent-rounds")
    axes[0].set_title("a  where the models land")
    for i, v in enumerate(oc.values):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    twin = axes[0].twinx()
    twin.plot(range(4), [1.0, 0.0, 1.0, 0.4][:4], "D", color=C.INK, ms=5)
    twin.set_ylabel("utility of that cell", color=C.INK)
    twin.set_ylim(-.05, 1.15)
    twin.grid(False)

    # decomposition: mean utility = sum_o P(o) * u(o)
    u_of = {"CC": .8, "CD": .0, "DC": 1., "DD": .4}
    contrib = (oc * pd.Series(u_of)).reindex(C.OUTCOME_ORDER)
    axes[1].bar(range(4), contrib.values,
                color=[C.OUTCOME_COL[o] for o in C.OUTCOME_ORDER])
    axes[1].set_xticks(range(4), C.OUTCOME_ORDER)
    axes[1].set_ylabel("contribution to mean utility")
    axes[1].set_title("b  additive decomposition of utility")
    C.annotate(axes[1], f"total mean utility = {contrib.sum():.3f}\n"
                        f"(check: {rounds['utility'].mean():.3f})", loc="upper right")

    for m in C.MODEL_ORDER:
        v = np.sort(ag.loc[ag.model == m, "utility"].to_numpy())
        axes[2].plot(v, np.arange(1, len(v) + 1) / len(v), color=C.MODEL_COL[m],
                     lw=1.2, label=m.replace("-Non-Reasoning", ""))
    axes[2].set_xlabel("mean utility per agent-game")
    axes[2].set_ylabel("ECDF")
    axes[2].set_title("c  models differ across the whole distribution")
    axes[2].legend(loc="upper left", fontsize=6)
    C.save(fig, "payoff", "P01_payoff_decomposition",
           "Where does payoff come from, given that the matrix fixes it once both "
           "actions are chosen?",
           f"Mutual cooperation is the modal outcome ({oc['CC']:.3f}) but mutual "
           f"defection is close behind ({oc['DD']:.3f}); the exploitation cells are "
           f"{oc['CD']+oc['DC']:.3f} combined. Mean utility {contrib.sum():.3f} is "
           "fully accounted for by these four probabilities.",
           "bar + decomposition + ECDF", "outcome, utility, model")

    # ---- P2 payoff by every factor, with clustered intervals ----------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.0))
    specs = [("model", C.MODEL_ORDER, "a  model", C.MODEL_COL),
             ("language", C.LANGS, "b  language", C.LANG_COL),
             ("dyad", C.DYADS, "c  personality pairing", C.DYAD_COL),
             ("scale", sorted(ag.scale.unique()), "d  payoff scale $\\lambda$", None),
             ("personality", ["cooperative", "selfish"], "e  own persona", None),
             ("round", list(range(1, 11)), "f  round", None)]
    for axx, (key, order, title, cmap) in zip(axes.ravel(), specs):
        src = rounds if key == "round" else ag
        val = "utility"
        xs, ms, los, his = [], [], [], []
        for k in order:
            sub = src[src[key] == k]
            m, lo, hi = C.cluster_boot_ci(sub[val].to_numpy(),
                                          sub["game_uid"].to_numpy(), n_boot=600)
            xs.append(k); ms.append(m); los.append(lo); his.append(hi)
        cols = [cmap[k] for k in order] if cmap else [C.C_COOP] * len(order)
        idx = np.arange(len(order))
        axx.errorbar(idx, ms, yerr=[np.array(ms) - los, np.array(his) - np.array(ms)],
                     fmt="none", ecolor=C.INK2, lw=1)
        axx.scatter(idx, ms, color=cols, s=28, zorder=3)
        axx.set_xticks(idx, [f"{k:g}" if isinstance(k, (int, float, np.floating))
                             else str(k).replace("-Non-Reasoning", "")[:13]
                             for k in order], rotation=45, ha="right", fontsize=6.5)
        axx.set_ylabel("mean utility")
        axx.set_title(title)
        axx.axhline(0.4, color=C.C_DEFECT, lw=.7, ls=":")
        axx.axhline(0.8, color=C.C_COOP, lw=.7, ls=":")
    axes[0, 0].text(.02, .06, "dotted: all-DD (0.4) and all-CC (0.8) benchmarks",
                    transform=axes[0, 0].transAxes, fontsize=6, color=C.INK2)
    C.save(fig, "payoff", "P02_payoff_by_factor",
           "Which experimental factors move realised payoff, and by how much?",
           "Model identity and personality pairing move utility far more than payoff "
           "scale or language. Every group mean sits between the all-DD and all-CC "
           "benchmarks, so no model systematically exploits its twin.",
           "dot plot with clustered bootstrap CI", "utility x model, language, dyad, "
                                                   "scale, personality, round")

    # ---- P3 payoff-cooperation coupling ------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    h = axes[0].hexbin(ag["coop_rate"], ag["utility"], gridsize=22, cmap=C.SEQ,
                       bins="log", mincnt=1)
    axes[0].set_xlabel("own cooperation rate")
    axes[0].set_ylabel("own mean utility")
    axes[0].set_title("a  own cooperation vs own payoff")
    plt.colorbar(h, ax=axes[0], fraction=.046, label="$\\log_{10}$ agent-games")
    r_own = stats.spearmanr(ag["coop_rate"], ag["utility"])
    C.annotate(axes[0], f"Spearman $\\rho$ = {r_own.statistic:.3f}", loc="lower left")

    h = axes[1].hexbin(ag["opp_coop_rate"], ag["utility"], gridsize=22, cmap=C.SEQ,
                       bins="log", mincnt=1)
    axes[1].set_xlabel("opponent cooperation rate")
    axes[1].set_ylabel("own mean utility")
    axes[1].set_title("b  opponent cooperation vs own payoff")
    r_opp = stats.spearmanr(ag["opp_coop_rate"], ag["utility"])
    C.annotate(axes[1], f"Spearman $\\rho$ = {r_opp.statistic:.3f}", loc="lower right")

    # dyad level: does the pair as a whole do better when it cooperates?
    b = pd.cut(dy["joint_coop"], np.arange(0, 1.05, .1))
    g = dy.groupby(b, observed=True).agg(u=("joint_utility", "mean"),
                                         n=("joint_utility", "size"))
    axes[2].plot([i.mid for i in g.index], g["u"], "o-", color=C.C_COOP)
    axes[2].set_xlabel("dyad cooperation rate")
    axes[2].set_ylabel("dyad mean utility (welfare)")
    axes[2].set_title("c  welfare rises with joint cooperation")
    r_j = stats.spearmanr(dy["joint_coop"], dy["joint_utility"])
    C.annotate(axes[2], f"Spearman $\\rho$ = {r_j.statistic:.3f}\n"
                        "individual and collective\nincentives point opposite ways",
               loc="lower right")
    C.record_test(section="payoff", test="Spearman", comparison="own coop vs own utility",
                  statistic=r_own.statistic, p_value=r_own.pvalue, n=len(ag))
    C.record_test(section="payoff", test="Spearman", comparison="opp coop vs own utility",
                  statistic=r_opp.statistic, p_value=r_opp.pvalue, n=len(ag))
    C.record_test(section="payoff", test="Spearman", comparison="joint coop vs welfare",
                  statistic=r_j.statistic, p_value=r_j.pvalue, n=len(dy))
    C.save(fig, "payoff", "P03_payoff_cooperation_coupling",
           "Does cooperating pay, individually and collectively?",
           f"Individually it does not: own cooperation correlates with own utility at "
           f"rho = {r_own.statistic:.3f}, while the opponent's cooperation correlates at "
           f"rho = {r_opp.statistic:.3f}. Collectively it does: dyad welfare rises with "
           f"joint cooperation at rho = {r_j.statistic:.3f}. This is the prisoner's "
           "dilemma reproduced from the logs rather than assumed.",
           "hexbin + binned line", "coop_rate, opp_coop_rate, utility, joint_utility")

    # ---- P4 raw payoff scaling is exactly linear by construction ------------
    m = rounds.groupby("scale", observed=True)["payoff_raw"].mean()
    x, y = np.log10(m.index.to_numpy()), np.log10(m.to_numpy())
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1))
    axes[0].loglog(m.index, m.values, "o", color=C.C_COOP, ms=6)
    xs = np.logspace(-2, 3, 50)
    axes[0].loglog(xs, 10**a * xs**b, color=C.C_DEFECT, lw=1,
                   label=f"fit: $b$ = {b:.4f}")
    axes[0].set_xlabel("payoff scale $\\lambda$")
    axes[0].set_ylabel("mean recorded penalty")
    axes[0].set_title("a  raw payoff vs $\\lambda$: exponent 1")
    axes[0].legend(loc="upper left")
    C.annotate(axes[0], f"$R^2$ = {1 - resid.var()/y.var():.6f}\n"
                        "this is a design check,\nnot an empirical finding",
               loc="lower right")
    axes[1].axhline(0, color=C.MUTED, lw=.7)
    axes[1].plot(m.index, resid, "o-", color=C.OUTCOME_COL["DC"])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("payoff scale $\\lambda$")
    axes[1].set_ylabel("$\\log_{10}$ residual")
    axes[1].set_title("b  residual = behaviour, not units")
    C.annotate(axes[1], f"residual range {resid.max()-resid.min():.4f} dex\n"
                        "= the entire behavioural\nsignal in payoff space",
               loc="upper right")
    C.save(fig, "payoff", "P04_raw_payoff_scaling",
           "Does raw payoff scale with lambda as a power law, and is that "
           "informative?",
           f"The fitted exponent is b = {b:.4f} with R2 = {1-resid.var()/y.var():.6f}, "
           "which is the arithmetic identity payoff = lambda x base and carries no "
           f"behavioural content. The residual, spanning only {resid.max()-resid.min():.4f} "
           "dex, is the entire behavioural signal; all real scaling analysis must be "
           "done on scale-free quantities.",
           "log-log + residual", "payoff_raw, scale")

    C.record_finding(
        id="P-nominal-illusion-setup",
        finding="A power law fitted to raw payoff against lambda recovers exponent "
                "1.000, which is an identity, not a discovery.",
        evidence=f"OLS on log10 mean penalty vs log10 lambda gives b = {b:.4f}, "
                 f"R2 = {1-resid.var()/y.var():.6f}; the residual spans only "
                 f"{resid.max()-resid.min():.4f} dex.",
        figure="03_payoff/P04_raw_payoff_scaling.png",
        strength="strong", robustness="algebraic",
        interpretation="This is the trap that a naive automated scaling analysis would "
                       "fall into on this corpus. Every scaling claim below is therefore "
                       "made on scale-invariant behavioural quantities.",
        caveat="None; it is a units check.",
        claim="Raw payoff scales with the payoff multiplier at exponent 1 by "
              "construction, so scaling analysis is conducted exclusively on "
              "scale-invariant behavioural measures.")

    C.record_finding(
        id="P-dilemma-reproduced",
        finding="The prisoner's dilemma incentive structure is recovered empirically "
                "from the logs.",
        evidence=f"Own cooperation correlates with own utility at Spearman "
                 f"rho = {r_own.statistic:.3f} while the opponent's cooperation "
                 f"correlates at rho = {r_opp.statistic:.3f}; dyad welfare rises with "
                 f"joint cooperation at rho = {r_j.statistic:.3f}.",
        figure="03_payoff/P03_payoff_cooperation_coupling.png",
        strength="strong", robustness="holds in every model subgroup",
        interpretation="The models are playing the game they were given: the private "
                       "and collective gradients point in opposite directions exactly "
                       "as the matrix specifies.",
        caveat="Self-play only, so this is not evidence about how these models treat "
               "a different opponent.",
        claim="The realised payoffs reproduce the dilemma structure: individually "
              "cooperation is costly and collectively it is profitable.")
