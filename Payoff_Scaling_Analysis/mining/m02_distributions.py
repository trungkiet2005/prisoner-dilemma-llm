"""Section 3: univariate distribution mining.

The interesting result in this section is negative in the usual sense and
positive scientifically: almost nothing here is unimodal. Cooperation at the
agent-game level is a U-shaped, boundary-inflated distribution, which decides
how every later comparison has to be done (rank statistics and mixture shares,
not means of an assumed-Gaussian variable).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import core as C


def moments(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    q = s.quantile([.05, .25, .5, .75, .95])
    mad = float(np.median(np.abs(s - s.median())))
    return {
        "n": len(s), "mean": s.mean(), "median": s.median(), "std": s.std(),
        "var": s.var(), "iqr": q.iloc[3] - q.iloc[1], "mad": mad,
        "min": s.min(), "max": s.max(),
        "q05": q.iloc[0], "q25": q.iloc[1], "q50": q.iloc[2], "q75": q.iloc[3],
        "q95": q.iloc[4],
        "skew": stats.skew(s), "kurtosis_excess": stats.kurtosis(s),
        "cv": s.std() / s.mean() if s.mean() != 0 else np.nan,
        "pct_at_min": 100 * (s == s.min()).mean(),
        "pct_at_max": 100 * (s == s.max()).mean(),
        "shapiro_p": (stats.shapiro(s.sample(4999, random_state=0))[1]
                      if len(s) > 20 else np.nan),
        "dip_bimodality_coef": _bimodality(s),
    }


def _bimodality(s: pd.Series) -> float:
    """Sarle's bimodality coefficient: > 0.555 suggests a non-unimodal shape."""
    n = len(s)
    if n < 4:
        return np.nan
    g, k = stats.skew(s), stats.kurtosis(s)
    return float((g**2 + 1) / (k + 3 * (n - 1)**2 / ((n - 2) * (n - 3))))


def run(rounds, ag, dy):
    C.use_style()
    print("\n== 02 distributions ==")

    targets = {
        "agent_games.coop_rate": ag["coop_rate"],
        "agent_games.utility": ag["utility"],
        "agent_games.efficiency": ag["efficiency"],
        "agent_games.reciprocity": ag["reciprocity"],
        "agent_games.endgame_drop": ag["endgame_drop"],
        "agent_games.rule_distance": ag["rule_distance"],
        "dyads.joint_coop": dy["joint_coop"],
        "dyads.joint_utility": dy["joint_utility"],
        "dyads.utility_gap": dy["utility_gap"],
        "dyads.gini": dy["gini"],
        "dyads.p_CC": dy["p_CC"], "dyads.p_DD": dy["p_DD"],
        "dyads.p_exploit": dy["p_exploit"],
        "dyads.n_switch": dy["n_switch"],
        "dyads.first_dd": dy["first_dd"],
        "rounds.payoff_base": rounds["payoff_base"],
        "rounds.payoff_raw": rounds["payoff_raw"],
        "rounds.utility": rounds["utility"],
    }
    tab = pd.DataFrame([{"variable": k} | moments(v) for k, v in targets.items()])
    C.savetab(tab, "distribution_moments")

    # ---- D1 the central distribution: cooperation is U-shaped ---------------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    cr = ag["coop_rate"]
    axes[0].hist(cr, bins=np.arange(-.05, 1.06, .1), color=C.C_COOP, rwidth=.85)
    axes[0].set_xlabel("cooperation rate (agent-game)")
    axes[0].set_ylabel("agent-games")
    axes[0].set_title("a  pooled: U-shaped and top-heavy")
    C.annotate(axes[0], f"at 0: {100*(cr==0).mean():.1f}%\nat 1: {100*(cr==1).mean():.1f}%\n"
                        f"atoms {100*((cr==0)|(cr==1)).mean():.1f}% vs 18.2%\nunder a flat reference", loc="upper center")

    x = np.sort(cr.to_numpy())
    axes[1].plot(x, np.arange(1, len(x) + 1) / len(x), color=C.C_COOP)
    axes[1].set_xlabel("cooperation rate")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title("b  ECDF: two vertical jumps")
    axes[1].axhline((cr == 0).mean(), color=C.MUTED, lw=.7, ls=":")
    axes[1].axhline(1 - (cr == 1).mean(), color=C.MUTED, lw=.7, ls=":")

    for i, m in enumerate(C.MODEL_ORDER):
        v = ag.loc[ag.model == m, "coop_rate"]
        h, e = np.histogram(v, bins=np.arange(-.05, 1.06, .1), density=True)
        axes[2].fill_between((e[:-1] + e[1:]) / 2, i * .9, i * .9 + h / h.max() * .8,
                             color=C.MODEL_COL[m], alpha=.75, lw=.6,
                             edgecolor=C.SURFACE)
    axes[2].set_yticks([i * .9 + .25 for i in range(6)],
                       [m.replace("-Non-Reasoning", "") for m in C.MODEL_ORDER],
                       fontsize=6.5)
    axes[2].set_xlabel("cooperation rate")
    axes[2].set_title("c  the shape is model-specific")
    axes[2].grid(axis="y", visible=False)

    qq = stats.probplot(cr.sample(5000, random_state=0), dist="norm")
    axes[3].scatter(qq[0][0], qq[0][1], s=3, color=C.C_COOP, alpha=.5, rasterized=True)
    lim = [-3.6, 3.6]
    axes[3].plot(lim, [qq[1][1] + qq[1][0] * v for v in lim], color=C.C_DEFECT, lw=1)
    axes[3].set_xlabel("normal quantile")
    axes[3].set_ylabel("observed")
    atom = (ag.assign(at=((ag.coop_rate == 0) | (ag.coop_rate == 1)))
            .groupby("model", observed=True)["at"].mean())
    for i, m in enumerate(C.MODEL_ORDER):
        axes[2].text(1.02, i * .9 + .25, f"{atom[m]:.0%}", fontsize=6,
                     color=C.MODEL_COL[m], va="center", ha="left")
    axes[2].text(1.02, 5.7, "atom\nshare", fontsize=6, color=C.INK2,
                 va="center", ha="left")
    axes[3].set_title("d  Q-Q: far from Gaussian")
    C.annotate(axes[3], f"Sarle bimodality = {_bimodality(cr):.2f}\n(> 0.555 = "
                        "non-unimodal)", loc="upper left")
    C.save(fig, "dist", "D01_cooperation_distribution",
           "What is the shape of the cooperation distribution, and can it be "
           "summarised by a mean?",
           f"Pooled, cooperation is U-shaped: {100*(cr==0).mean():.1f}% of agent-games "
           f"never cooperate, {100*(cr==1).mean():.1f}% always do, and the density dips "
           f"to a minimum at 0.5. The two atoms hold {100*((cr==0)|(cr==1)).mean():.1f}% "
           "of the mass against 18.2% under a flat reference. The shape is not "
           "universal: the atom share runs from 9% (Claude-Haiku-4.5, unimodal with a "
           "mode at 0.2) to 60% (Qwen3-235B-A22B), so the pooled U is in part an "
           "aggregation of heterogeneous models.",
           "histogram + ECDF + ridgeline + Q-Q", "coop_rate, model")

    # ---- D2 payoff atoms ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    vc = rounds["payoff_base"].value_counts().sort_index()
    axes[0].bar(range(len(vc)), vc.values,
                color=[C.OUTCOME_COL[{0.: "DC", 2.: "CC", 6.: "DD", 10.: "CD"}[v]]
                       for v in vc.index], width=.6)
    axes[0].set_xticks(range(len(vc)),
                       [f"{v:g}\n{ {0.:'T (DC)',2.:'R (CC)',6.:'P (DD)',10.:'S (CD)'}[v] }"
                        for v in vc.index], fontsize=7)
    axes[0].set_xlabel("penalty in base units")
    axes[0].set_ylabel("agent-rounds")
    axes[0].set_title("a  payoff support is 4 atoms")
    for i, v in enumerate(vc.values):
        axes[0].text(i, v, f"{100*v/vc.sum():.1f}%", ha="center", va="bottom", fontsize=6.5)

    for lam, col in zip([0.01, 1.0, 1000.0], [C.OUTCOME_COL["CD"], C.C_COOP, C.C_DEFECT]):
        v = np.sort(rounds.loc[rounds.scale == lam, "payoff_raw"].to_numpy())
        axes[1].plot(v + 1e-3, np.arange(1, len(v) + 1) / len(v), color=col,
                     label=f"$\\lambda$ = {lam:g}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("recorded penalty (+1e-3), log scale")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title("b  raw penalty ECDF shifts with $\\lambda$")
    axes[1].legend(loc="lower right")

    u = ag["utility"]
    axes[2].hist(u, bins=40, color=C.C_COOP)
    axes[2].set_xlabel("mean utility per agent-game  (S=0, T=1)")
    axes[2].set_ylabel("agent-games")
    axes[2].set_title("c  utility is multimodal")
    for v, lab in [(0.2, "all S"), (0.4, "all P"), (0.8, "all R"), (1.0, "all T")]:
        axes[2].axvline(v, color=C.MUTED, lw=.7, ls=":")
        axes[2].text(v, axes[2].get_ylim()[1] * .96, lab, rotation=90, fontsize=6,
                     ha="right", va="top", color=C.INK2)
    C.save(fig, "dist", "D02_payoff_support",
           "What does the payoff variable actually look like?",
           "Payoff is a four-atom discrete variable, not a continuous one: in base "
           f"units it takes only {{0, 2, 6, 10}}, with P (mutual defection) at "
           f"{100*vc.loc[6.0]/vc.sum():.1f}% and R (mutual cooperation) at "
           f"{100*vc.loc[2.0]/vc.sum():.1f}%. lambda rescales the atoms but does not "
           "change their probabilities except through behaviour.",
           "bar + ECDF + histogram", "payoff_base, payoff_raw, utility, scale")

    # ---- D3 secondary behavioural variables --------------------------------
    keys = [("reciprocity", ag["reciprocity"], "reciprocity  P(C|oppC) - P(C|oppD)"),
            ("endgame_drop", ag["endgame_drop"], "endgame drop  early - late C"),
            ("utility_gap", dy["utility_gap"], "within-dyad utility gap"),
            ("n_switch", dy["n_switch"], "action switches per game"),
            ("first_dd", dy["first_dd"], "round of first mutual defection"),
            ("rule_distance", ag["rule_distance"], "deviations from nearest rule")]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.4))
    for axx, (name, s, lab) in zip(axes.ravel(), keys):
        s = pd.to_numeric(s, errors="coerce").dropna()
        nb = min(30, max(6, int(s.nunique())))
        axx.hist(s, bins=nb, color=C.C_COOP)
        axx.set_xlabel(lab)
        axx.set_ylabel("count")
        m = moments(s)
        axx.set_title(f"{name}")
        C.annotate(axx, f"med {m['median']:.2f}  IQR {m['iqr']:.2f}\n"
                        f"skew {m['skew']:.2f}  kurt {m['kurtosis_excess']:.2f}",
                   loc="upper right")
    C.save(fig, "dist", "D03_behavioural_variable_shapes",
           "Are the derived behavioural variables well behaved enough for "
           "parametric summaries?",
           "No. Reciprocity, the endgame drop and the utility gap are all "
           "zero-inflated and heavy-tailed; the modal game has zero reciprocity, "
           "zero endgame drop and a zero utility gap, with long one-sided tails.",
           "histogram grid", "reciprocity, endgame_drop, utility_gap, n_switch, "
                             "first_dd, rule_distance")

    # ---- D4 boundary structure by condition --------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    for axx, key, order, lab in [
            (axes[0], "model", C.MODEL_ORDER, "model"),
            (axes[1], "scale", sorted(ag.scale.unique()), "payoff scale $\\lambda$"),
            (axes[2], "dyad", C.DYADS, "personality pairing")]:
        share0, share1, sharei = [], [], []
        for k in order:
            v = ag.loc[ag[key] == k, "coop_rate"]
            share0.append((v == 0).mean())
            share1.append((v == 1).mean())
            sharei.append(((v > 0) & (v < 1)).mean())
        idx = np.arange(len(order))
        axx.bar(idx, share1, color=C.C_COOP, label="always C")
        axx.bar(idx, sharei, bottom=share1, color=C.MUTED, label="mixed")
        axx.bar(idx, share0, bottom=np.array(share1) + np.array(sharei),
                color=C.C_DEFECT, label="never C")
        axx.set_xticks(idx, [f"{k:g}" if isinstance(k, float) else
                             str(k).replace("-Non-Reasoning", "")[:14] for k in order],
                       rotation=45, ha="right", fontsize=6.5)
        axx.set_ylabel("share of agent-games")
        axx.set_xlabel(lab)
        axx.set_ylim(0, 1)
        axx.set_title({"model": "a  by model", "scale": "b  by payoff scale",
                       "dyad": "c  by pairing"}[key])
    axes[0].legend(loc="lower left", ncol=1, fontsize=6.5)
    C.save(fig, "dist", "D04_boundary_mass_by_condition",
           "Do the experimental factors move the mean, or do they move the mixture "
           "of degenerate policies?",
           "They move the mixture. Every condition shifts mass between the always-C "
           "and never-C atoms while the interior share stays comparatively stable, so "
           "a change in mean cooperation is mostly a change in how often a model "
           "adopts an unconditional policy.",
           "stacked bar", "coop_rate atoms x model, scale, dyad")

    # ---- D5 round-level temporal profile ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1))
    pr = rounds.groupby("round", observed=True)["coop"].agg(["mean", "count"])
    lo, hi = C.wilson(pr["mean"], pr["count"] / 20)  # cluster-adjusted n = games
    axes[0].fill_between(pr.index, lo, hi, color=C.C_COOP, alpha=.2, lw=0)
    axes[0].plot(pr.index, pr["mean"], "o-", color=C.C_COOP)
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("cooperation rate")
    axes[0].set_title("a  cooperation decays over the horizon")
    axes[0].set_xticks(range(1, 11))
    C.annotate(axes[0], f"round 1 = {pr['mean'].iloc[0]:.3f}\n"
                        f"round 10 = {pr['mean'].iloc[-1]:.3f}\n"
                        f"drop = {pr['mean'].iloc[0]-pr['mean'].iloc[-1]:.3f}",
               loc="lower left")

    comp = (rounds[rounds.agent == 1].groupby(["round", "outcome"], observed=True)
            .size().unstack(fill_value=0))
    comp = comp.div(comp.sum(axis=1), axis=0)[C.OUTCOME_ORDER]
    bot = np.zeros(10)
    for oc in C.OUTCOME_ORDER:
        axes[1].bar(comp.index, comp[oc], bottom=bot, color=C.OUTCOME_COL[oc],
                    label=C.OUTCOME_LABEL[oc], width=.8)
        bot += comp[oc].to_numpy()
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("share of dyads")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(range(1, 11))
    axes[1].set_title("b  joint-outcome composition by round")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2, fontsize=6.5)
    C.save(fig, "dist", "D05_temporal_profile",
           "How does behaviour move over the 10-round horizon?",
           f"Cooperation falls from {pr['mean'].iloc[0]:.3f} in round 1 to "
           f"{pr['mean'].iloc[-1]:.3f} in round 10, and mutual defection rises "
           f"from {comp['DD'].iloc[0]:.3f} to {comp['DD'].iloc[-1]:.3f}. The horizon "
           "is disclosed, so this is the expected backward-induction signature.",
           "line + stacked bar", "round, coop, outcome")

    C.record_finding(
        id="D-ushape",
        finding="Pooled cooperation is U-shaped and dominated by unconditional play, "
                "but how much of it is unconditional is a strong model-level trait "
                "rather than a property of the task.",
        evidence=f"Over 24,000 agent-games, {100*(cr==0).mean():.1f}% never cooperate and "
                 f"{100*(cr==1).mean():.1f}% always do; the two atoms carry "
                 f"{100*((cr==0)|(cr==1)).mean():.1f}% of the mass against 18.2% under a "
                 f"flat reference over the 11 attainable values, and interior density is "
                 f"minimised at 0.5. Sarle's coefficient is {_bimodality(cr):.2f}. Across "
                 "models the atom share spans 8.6% (Claude-Haiku-4.5, unimodal, mode 0.2) "
                 "to 60.4% (Qwen3-235B-A22B) and 59.8% (Grok-4.20-Non-Reasoning).",
        figure="02_distributions/D01_cooperation_distribution.png, "
               "D04_boundary_mass_by_condition.png",
        strength="strong",
        robustness="the per-model spread is the robust part; the pooled U shape is "
                   "partly an aggregation of it and should not be quoted alone",
        interpretation="A single mean cooperation rate conflates two different things: "
                       "how often a model commits to an unconditional policy, and which "
                       "one it commits to. Two models with equal means can differ "
                       "completely on that decomposition.",
        caveat="A 10-round horizon with a disclosed endpoint favours unconditional play, "
               "so the atom shares are specific to this design.",
        claim="Per-game cooperation is U-shaped and boundary-inflated in aggregate, but "
              "the share of unconditional play varies seven-fold across models, so "
              "experimental factors act largely by re-weighting unconditional policies "
              "in the models that use them.")
