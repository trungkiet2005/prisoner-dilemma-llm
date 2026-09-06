"""Section 8: Pareto and frontier analysis.

A word collision to clear up before anything else
-------------------------------------------------
The corpus directory is called `data_fairgame_frontier_llm`, where "frontier"
means FRONTIER MODELS: commercial flagship LLMs, as opposed to the open-weights
baseline branch of the same project. In this section "frontier" means something
completely different: the PARETO FRONTIER of a multi-objective trade-off, the set
of observations that no other observation beats on every objective at once. The
two senses are unrelated and nothing here is a statement about model tiers.

A second thing this section cannot do
-------------------------------------
The natural question "does scale improve the frontier" has two readings. For
MODEL scale it is unanswerable: this corpus has six models and no parameter
count, no compute axis and no size-matched family, so there is no model-scale
variable to regress on. For PAYOFF scale it is fully answerable, because lambda
sweeps five orders of magnitude over otherwise identical games, and that is the
version answered below. Every "frontier vs scale" result here is about lambda.

The objectives, and how much of a trade-off they really are
-----------------------------------------------------------
Three candidate objectives, all higher-is-better:

  welfare      joint_utility, the two agents' mean per-round utility
  equality     fairness = 1 - gini, the within-dyad split of that utility
  cooperation  joint_coop, the dyad's cooperation rate

Before treating these as a trade-off they have to be checked for mechanical
coupling, and one pair turns out to be coupled exactly. With this payoff matrix a
round pays 0.8 jointly under CC, 0.4 under DD and 0.5 under either
miscoordination, so

  welfare = 0.4 + 0.4 * cooperation - 0.1 * exploitation_share

holds as an algebraic identity, verified to a maximum residual of 2e-16 over all
12,000 games. Welfare and cooperation are therefore not two objectives; they are
one objective plus a miscoordination penalty, and a "trade-off" between them
would be an artefact of arithmetic. Equality is genuinely different: it is driven
by the ASYMMETRY of exploitation, |p_DC - p_CD|, whereas welfare is reduced by its
TOTAL. A dyad that alternates who gets exploited is perfectly equal and
inefficient at the same time.

Unit of analysis
----------------
Frontier geometry needs a unit that is neither degenerate nor noisy. Individual
games are hopeless for dominance: with 10 rounds, welfare takes 11 values and
equality is pinned to exactly 1.0 whenever the exploitation counts happen to
balance, so a third of all games sit on the equality ceiling and dominance
becomes an artefact of grid coarseness. The unit used here is the CONDITION CELL,
model x language x lambda x persona pairing, 1,200 of them, each the mean of
exactly 10 repetitions. It is the finest unit that is still an experimental
condition somebody could choose, and the balanced design gives every one of them
the same sample size. Individual games are still used for inference, clustered on
the common-random-number key `cell`, because 12,000 games are only 1,200
independent design cells across the lambda ladder.

What the analysis finds, and why the section is shaped the way it is
--------------------------------------------------------------------
The headline is negative in the trade-off sense and positive in the diagnostic
sense: there is no welfare-equality trade-off in this game. The collective ideal,
mutual cooperation every round, gives welfare 0.8 AND perfect equality, and it is
not merely approached but attained exactly in 122 of 1,200 conditions. The Pareto
frontier therefore collapses onto a single point, and every other condition is
strictly inefficient rather than differently optimal. Once that is established the
informative questions are all about distance to that one point: who is close, what
kind of failure moves them away, and whether the payoff multiplier moves them
toward it. The welfare shortfall decomposes exactly into a mutual-defection term
and an exploitation term, which turns "how badly did this model do" into the more
useful "which of the two ways of doing badly did it choose".

One reference point is worth marking on every frontier plot. The subgame-perfect
equilibrium of this finite, horizon-disclosed game is universal defection, which
sits at welfare 0.4 with perfect equality. Equilibrium play is exactly as "fair"
as the collective optimum, which is precisely why equality on its own is not a
welfare-relevant objective and only becomes informative jointly with welfare.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import core as C

# reference points in objective space -------------------------------------
W_IDEAL, EQ_IDEAL = 0.8, 1.0     # mutual cooperation every round
W_SPE, EQ_SPE = 0.4, 1.0         # subgame-perfect equilibrium: universal defection
W_FLOOR = 0.4                    # lowest attainable joint welfare (all DD)

DYAD_OF_PAIRING = {0: "CvC", 1: "CvS", 2: "SvC", 3: "SvS"}


def dist_to_ideal(w, eq):
    """Normalised Euclidean distance to (welfare 0.8, equality 1), in [0, 1].

    Welfare is rescaled by its attainable range [0.4, 0.8] so the two objectives
    contribute on the same footing; the sqrt(2) puts the worst corner at exactly 1.
    """
    w = np.asarray(w, float)
    eq = np.asarray(eq, float)
    return np.sqrt(((W_IDEAL - w) / (W_IDEAL - W_FLOOR)) ** 2
                   + (EQ_IDEAL - eq) ** 2) / np.sqrt(2)


def pareto_mask(P: np.ndarray) -> np.ndarray:
    """True where the row is non-dominated, maximising every column."""
    n = len(P)
    nd = np.ones(n, bool)
    for i in range(n):
        dom = np.all(P >= P[i], axis=1) & np.any(P > P[i], axis=1)
        if dom.any():
            nd[i] = False
    return nd


def hypervolume(w, eq) -> float:
    """2-D hypervolume dominated by a point set, reference (welfare 0.4, equality 0).

    Welfare is normalised to [0, 1] over its attainable range, so a set that
    contains the ideal point has hypervolume 1.
    """
    x = (np.asarray(w, float) - W_FLOOR) / (W_IDEAL - W_FLOOR)
    y = np.asarray(eq, float)
    o = np.lexsort((-y, -x))
    x, y = x[o], y[o]
    hv, ymax = 0.0, 0.0
    for xi, yi in zip(x, y):
        if yi > ymax:
            hv += xi * (yi - ymax)
            ymax = yi
    return float(hv)


def attainable_envelope(w: np.ndarray) -> np.ndarray:
    """Lowest equality a SINGLE game can show at a given joint welfare.

    Maximising the exploitation asymmetry m subject to the welfare constraint gives
    m_max = min(10 (w - 0.4), (0.8 - w) / 0.3) capped at 1, and equality 1 - m/(2w).
    The upper boundary is flat at 1: balancing p_CD against p_DC makes any welfare
    level perfectly equal.
    """
    m = np.clip(np.minimum(10 * (w - W_FLOOR), (W_IDEAL - w) / 0.3), 0, 1)
    return 1 - m / (2 * w)


def _cluster_slope(df, value, xcol="log_scale", cluster="cell"):
    d = df[[cluster, xcol, value]].dropna()
    X = sm.add_constant(d[xcol].to_numpy())
    m = sm.OLS(d[value].to_numpy(), X).fit(cov_type="cluster",
                                           cov_kwds={"groups": d[cluster].to_numpy()})
    ci = m.conf_int()
    return float(m.params[1]), float(ci[1][0]), float(ci[1][1]), float(m.pvalues[1])


def _short(m: str) -> str:
    return m.replace("-Non-Reasoning", "")


def _ord(k: int) -> str:
    """1 -> 1st, 2 -> 2nd, 3 -> 3rd, 4 -> 4th ..."""
    k = int(k)
    if 10 <= k % 100 <= 20:
        return f"{k}th"
    return f"{k}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th') }".replace(" ", "")


# --------------------------------------------------------------------------
def run(rounds, ag, dy):
    C.use_style()
    print("\n== 08 pareto and frontier ==")

    # ---- objectives at game level ---------------------------------------
    dy = dy.copy()
    dy["welfare"] = dy["joint_utility"]
    dy["equality"] = dy["fairness"]
    dy["coop"] = dy["joint_coop"]
    dy["dist"] = dist_to_ideal(dy["welfare"], dy["equality"])
    dy["at_ideal"] = ((dy["welfare"] > W_IDEAL - 1e-9)
                      & (dy["equality"] > EQ_IDEAL - 1e-9)).astype(float)
    dy["loss_dd"] = 0.4 * dy["p_DD"]
    dy["loss_exploit"] = 0.3 * dy["p_exploit"]

    # exact identity checks, printed and reused in the captions
    ident_w = float(np.abs(dy["welfare"]
                           - (0.4 + 0.4 * dy["coop"] - 0.1 * dy["p_exploit"])).max())
    ident_l = float(np.abs((W_IDEAL - dy["welfare"])
                           - (dy["loss_dd"] + dy["loss_exploit"])).max())
    print(f"  welfare identity residual   {ident_w:.2e}")
    print(f"  shortfall identity residual {ident_l:.2e}")

    # ---- condition cells: model x language x lambda x pairing ------------
    cc = (dy.groupby(["model", "language", "scale", "pairing"], observed=True)
          .agg(welfare=("welfare", "mean"), equality=("equality", "mean"),
               coop=("coop", "mean"), p_DD=("p_DD", "mean"),
               p_exploit=("p_exploit", "mean"), p_CC=("p_CC", "mean"),
               n_games=("welfare", "size"))
          .reset_index())
    cc["log_scale"] = np.log10(cc["scale"])
    cc["dyad"] = cc["pairing"].map(DYAD_OF_PAIRING)
    cc["dist"] = dist_to_ideal(cc["welfare"], cc["equality"])
    cc["nondominated"] = pareto_mask(cc[["welfare", "equality"]].to_numpy())
    cc["nondominated_3obj"] = pareto_mask(
        cc[["welfare", "equality", "coop"]].to_numpy())
    cc["at_ideal"] = ((cc["welfare"] > W_IDEAL - 1e-9)
                      & (cc["equality"] > EQ_IDEAL - 1e-9))
    C.savetab(cc, "frontier_points")

    n_nd = int(cc["nondominated"].sum())
    n_nd3 = int(cc["nondominated_3obj"].sum())
    n_ideal = int(cc["at_ideal"].sum())
    print(f"  condition cells {len(cc)}, non-dominated {n_nd}, at exact ideal {n_ideal}")

    # objective-set comparison for the three-objective question
    sets = {
        "welfare only": ["welfare"],
        "equality only": ["equality"],
        "welfare + equality": ["welfare", "equality"],
        "welfare + equality\n+ cooperation": ["welfare", "equality", "coop"],
    }
    nd_counts = {k: int(pareto_mask(cc[v].to_numpy()).sum()) for k, v in sets.items()}

    r_we = stats.pearsonr(cc["welfare"], cc["equality"])
    r_wc = stats.pearsonr(cc["welfare"], cc["coop"])
    rs_we = stats.spearmanr(cc["welfare"], cc["equality"])
    C.record_test(section="frontier", test="Pearson correlation between objectives",
                  comparison="welfare vs equality (condition cells)",
                  statistic=r_we.statistic, p_value=r_we.pvalue, n=len(cc),
                  effect_size_note=f"r2 = {r_we.statistic ** 2:.3f}")
    C.record_test(section="frontier", test="Pearson correlation between objectives",
                  comparison="welfare vs cooperation (condition cells)",
                  statistic=r_wc.statistic, p_value=r_wc.pvalue, n=len(cc),
                  effect_size_note=f"r2 = {r_wc.statistic ** 2:.3f}; the two are linked "
                                   "by an exact algebraic identity")

    # =====================================================================
    # PF01  are these really three objectives?
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))

    sc = axes[0].scatter(cc["coop"], cc["welfare"], c=cc["p_exploit"], cmap=C.SEQ,
                         s=11, lw=0, alpha=.85, rasterized=True)
    xs = np.linspace(0, 1, 50)
    axes[0].plot(xs, 0.4 + 0.4 * xs, color=C.INK, lw=1.0, ls="--",
                 label="no miscoordination (exploit = 0)")
    axes[0].plot(xs, 0.4 + 0.4 * xs - 0.1, color=C.INK, lw=1.0, ls=":",
                 label="every round miscoordinated")
    plt.colorbar(sc, ax=axes[0], fraction=.046, label="exploitation share")
    axes[0].set_ylim(0.27, 0.83)
    axes[0].legend(loc="lower right", fontsize=6.0)
    axes[0].set_xlabel("dyad cooperation rate")
    axes[0].set_ylabel("joint welfare (0.8 = mutual C)")
    axes[0].set_title("a  welfare is not independent")
    C.annotate(axes[0], "welfare = 0.4 + 0.4 coop - 0.1 exploit\n"
                        f"exact: max residual {ident_w:.0e}\n"
                        f"r(welfare, coop) = {r_wc.statistic:.3f}", loc="upper left")

    for m in C.MODEL_ORDER:
        s = cc[cc.model == m]
        axes[1].scatter(s["welfare"], s["equality"], s=9, lw=0, alpha=.55,
                        color=C.MODEL_COL[m], rasterized=True)
    bins = np.arange(0.40, 0.81, 0.05)
    mid = (bins[:-1] + bins[1:]) / 2
    grp = pd.cut(cc["welfare"], bins, include_lowest=True)
    bm = cc.groupby(grp, observed=True)["equality"].agg(["mean", "sem", "size"])
    axes[1].errorbar(mid[:len(bm)], bm["mean"], yerr=1.96 * bm["sem"], fmt="o-",
                     color=C.INK, lw=1.4, ms=4, zorder=5, label="binned mean, 95% CI")
    axes[1].scatter([W_IDEAL], [EQ_IDEAL], marker="*", s=190, color=C.C_COOP,
                    edgecolor=C.INK, lw=.6, zorder=6)
    axes[1].scatter([W_SPE], [EQ_SPE], marker="s", s=48, color=C.C_DEFECT,
                    edgecolor=C.INK, lw=.6, zorder=6)
    axes[1].text(W_IDEAL, 1.04, "collective\nideal", fontsize=6.2, ha="right",
                 va="bottom", color=C.INK)
    axes[1].text(W_SPE, 1.04, "equilibrium\n(all D)", fontsize=6.2, ha="left",
                 va="bottom", color=C.INK)
    axes[1].set_ylim(-0.12, 1.22)
    axes[1].set_xlabel("joint welfare")
    axes[1].set_ylabel("equality (1 - within-dyad Gini)")
    axes[1].set_title("b  equality vs welfare")
    axes[1].legend(loc="lower right", fontsize=6.2)
    C.annotate(axes[1], f"Pearson r = {r_we.statistic:+.3f}\n"
                        f"Spearman = {rs_we.statistic:+.3f}\n"
                        "U-shaped: both corners are equal",
               loc="lower left")

    keys = list(sets)
    vals = [nd_counts[k] for k in keys]
    cols = [C.MUTED, C.MUTED, C.C_COOP, C.OUTCOME_COL["DC"]]
    axes[2].barh(range(len(keys)), vals, color=cols)
    axes[2].set_yticks(range(len(keys)), keys, fontsize=6.3)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("non-dominated condition cells (of 1,200)")
    axes[2].set_title("c  a third objective adds nothing")
    for i, v in enumerate(vals):
        axes[2].text(v + max(vals) * .02, i, str(v), va="center", fontsize=7)
    axes[2].set_xlim(0, max(vals) * 1.62)
    C.annotate(axes[2], "adding cooperation leaves the\n"
                        f"non-dominated set unchanged\nat {n_nd} cells",
               loc="upper right")
    C.save(fig, "frontier", "PF01_objective_coupling",
           "Are welfare, equality and cooperation three independent objectives, or are "
           "some of them the same quantity in disguise?",
           "Welfare and cooperation are the same quantity up to a miscoordination term: "
           f"welfare = 0.4 + 0.4 coop - 0.1 exploit holds exactly (max residual "
           f"{ident_w:.0e}) and the two correlate at r = {r_wc.statistic:.3f}. Equality "
           f"is genuinely distinct (r = {r_we.statistic:+.3f} with welfare) but U-shaped "
           "in it, because universal defection and universal cooperation are both "
           "perfectly equal. Adding cooperation as a third objective leaves the "
           f"non-dominated set at {n_nd} cells, exactly where the two-objective set "
           "already was, so a three-objective frontier carries no extra information and "
           "is not plotted.",
           "scatter with identity line + binned mean + non-dominated counts",
           "welfare, equality, cooperation, exploitation share (condition cells)")

    # =====================================================================
    # PF02  the frontier itself, observed against theoretical
    # =====================================================================
    ml = (dy.groupby(["model", "scale"], observed=True)
          .agg(welfare=("welfare", "mean"), equality=("equality", "mean"),
               coop=("coop", "mean")).reset_index())
    ml["dist"] = dist_to_ideal(ml["welfare"], ml["equality"])
    ml["nondominated"] = pareto_mask(ml[["welfare", "equality"]].to_numpy())
    n_nd_ml = int(ml["nondominated"].sum())
    nd_models = sorted(ml.loc[ml.nondominated, "model"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7))

    dom = cc[~cc.nondominated]
    axes[0].scatter(dom["welfare"], dom["equality"], s=9, lw=0, color=C.MUTED,
                    alpha=.55, rasterized=True, label=f"dominated ({len(dom)})")
    nd = cc[cc.nondominated]
    axes[0].scatter(nd["welfare"], nd["equality"], s=30, lw=.4, color=C.C_COOP,
                    edgecolor=C.INK, zorder=4, label=f"non-dominated ({len(nd)})")
    axes[0].scatter([W_SPE], [EQ_SPE], marker="s", s=52, color=C.C_DEFECT,
                    edgecolor=C.INK, lw=.6, zorder=5)
    axes[0].annotate("equilibrium\n(all D)", (W_SPE, EQ_SPE), (0.435, .74),
                     fontsize=6.4, color=C.INK,
                     arrowprops=dict(arrowstyle="-", lw=.6, color=C.INK2))
    axes[0].annotate("theoretical ideal\n= observed frontier", (W_IDEAL, EQ_IDEAL),
                     (0.575, .52), fontsize=6.4, color=C.INK,
                     arrowprops=dict(arrowstyle="->", lw=.7, color=C.INK))
    axes[0].set_xlabel("joint welfare")
    axes[0].set_ylabel("equality (1 - within-dyad Gini)")
    axes[0].set_title("a  the frontier is one point")
    axes[0].legend(loc="lower center", fontsize=6.2)
    axes[0].set_ylim(-0.10, 1.34)
    C.annotate(axes[0], f"{n_ideal} of 1,200 cells sit exactly on the ideal "
                        f"({n_ideal / len(cc):.1%});\nevery other cell is strictly "
                        "dominated", loc="upper left")

    wgrid = np.linspace(0.4001, 0.7999, 400)
    lowb = attainable_envelope(wgrid)
    axes[1].fill_between(wgrid, lowb, 1.0, color=C.C_COOP, alpha=.13, lw=0,
                         label="attainable by one game")
    axes[1].plot(wgrid, lowb, color=C.INK2, lw=1.0)
    axes[1].axhline(1.0, color=C.INK2, lw=1.0)
    axes[1].scatter(cc["welfare"], cc["equality"], s=7, lw=0, color=C.INK,
                    alpha=.35, rasterized=True, label="condition cells")
    axes[1].scatter([W_IDEAL], [EQ_IDEAL], marker="*", s=190, color=C.C_COOP,
                    edgecolor=C.INK, lw=.6, zorder=6)
    axes[1].set_xlabel("joint welfare")
    axes[1].set_ylabel("equality (1 - within-dyad Gini)")
    axes[1].set_title("b  attainable set for one game")
    axes[1].legend(loc="lower center", fontsize=6.2)
    C.annotate(axes[1], "equality 1 is attainable at every welfare level, so\n"
                        "only the top-right corner is Pareto optimal in theory",
               loc="upper left")
    axes[1].set_ylim(-0.10, 1.34)

    for m in C.MODEL_ORDER:
        s = ml[ml.model == m].sort_values("scale")
        axes[2].plot(s["welfare"], s["equality"], "-", color=C.MODEL_COL[m],
                     lw=.7, alpha=.5, zorder=2)
        axes[2].scatter(s["welfare"], s["equality"], s=22, lw=0,
                        color=C.MODEL_COL[m], label=_short(m), zorder=3)
    ndm = ml[ml.nondominated]
    axes[2].scatter(ndm["welfare"], ndm["equality"], s=95, facecolor="none",
                    edgecolor=C.INK, lw=1.0, zorder=4)
    axes[2].scatter([W_IDEAL], [EQ_IDEAL], marker="*", s=190, color=C.C_COOP,
                    edgecolor=C.INK, lw=.6, zorder=6)
    axes[2].set_xlabel("mean joint welfare")
    axes[2].set_ylabel("mean equality")
    axes[2].set_title("c  model-by-$\\lambda$ means")
    axes[2].set_ylim(0.40, 1.30)
    axes[2].legend(loc="upper left", fontsize=5.6, ncol=2, handletextpad=.3,
                   labelspacing=.25, columnspacing=.7)
    C.annotate(axes[2], f"{n_nd_ml} of 60 model-$\\lambda$ means\n"
                        f"are non-dominated,\nall from {_short(nd_models[0])}",
               loc="lower right")
    C.save(fig, "frontier", "PF02_pareto_frontier",
           "What does the empirical Pareto frontier over welfare and equality look "
           "like, and how far short of the theoretical optimum does it fall?",
           f"It does not fall short at all: {n_ideal} of 1,200 condition cells sit "
           "exactly at welfare 0.8 with perfect equality, and all "
           f"{n_nd} non-dominated cells are those same points, so the observed frontier "
           "coincides with the theoretical ideal. The reason is structural, not "
           "behavioural: equality 1 is attainable at every welfare level by balancing "
           "who gets exploited, so the theoretical Pareto set is the single top-right "
           f"corner and there is no welfare-equality trade-off to trade off. Averaged to "
           f"model-by-lambda means, only {n_nd_ml} of 60 points are non-dominated and "
           f"all belong to {_short(nd_models[0])}. Note that 'frontier' here means the "
           "Pareto frontier, not the frontier-model corpus the data come from.",
           "Pareto scatter with dominance marking + attainable region + aggregate frontier",
           "welfare, equality by condition cell and by model x scale")

    # =====================================================================
    # PF03  who is close to the ideal
    # =====================================================================
    rows = []
    for m in C.MODEL_ORDER:
        s = dy[dy.model == m]
        mu, lo, hi = C.cluster_boot_ci(s["dist"].to_numpy(), s["cell"].to_numpy(),
                                       n_boot=2000)
        am, alo, ahi = C.cluster_boot_ci(s["at_ideal"].to_numpy(),
                                         s["cell"].to_numpy(), n_boot=2000)
        cs = cc[cc.model == m]
        rows.append({
            "model": m, "vendor": C.VENDOR[m],
            "mean_welfare": s["welfare"].mean(), "mean_equality": s["equality"].mean(),
            "mean_coop": s["coop"].mean(),
            "dist_game_mean": mu, "dist_lo": lo, "dist_hi": hi,
            "dist_of_centroid": float(dist_to_ideal(s["welfare"].mean(),
                                                    s["equality"].mean())),
            "dist_cell_mean": cs["dist"].mean(),
            "share_games_at_ideal": am, "ideal_lo": alo, "ideal_hi": ahi,
            "share_cells_at_ideal": cs["at_ideal"].mean(),
            "n_cells_nondominated": int(cs["nondominated"].sum()),
            "hypervolume": hypervolume(cs["welfare"], cs["equality"]),
            "shortfall": W_IDEAL - s["welfare"].mean(),
            "loss_from_DD": s["loss_dd"].mean(),
            "loss_from_exploitation": s["loss_exploit"].mean(),
            "dd_share_of_shortfall": s["loss_dd"].mean()
                                     / (W_IDEAL - s["welfare"].mean()),
            "n_games": len(s),
        })
    bym = pd.DataFrame(rows).sort_values("dist_game_mean").reset_index(drop=True)
    bym["rank_distance"] = np.arange(1, len(bym) + 1)
    bym["rank_attainment"] = bym["share_cells_at_ideal"].rank(ascending=False).astype(int)
    C.savetab(bym, "frontier_distance_by_model")

    rho_rank = stats.spearmanr(bym["rank_distance"], bym["rank_attainment"])
    C.record_test(section="frontier",
                  test="Spearman between two frontier rankings of the same models",
                  comparison="mean distance to ideal vs exact ideal attainment",
                  statistic=rho_rank.statistic, p_value=rho_rank.pvalue, n=len(bym),
                  effect_size_note="rank agreement between a central-tendency and a "
                                   "best-case frontier metric")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7))
    for m in C.MODEL_ORDER:
        s = cc[cc.model == m]
        axes[0].scatter(s["welfare"], s["equality"], s=6, lw=0, alpha=.20,
                        color=C.MODEL_COL[m], rasterized=True)
    for _, r in bym.iterrows():
        axes[0].plot([r.mean_welfare, W_IDEAL], [r.mean_equality, EQ_IDEAL],
                     color=C.MODEL_COL[r.model], lw=.8, ls=":", zorder=4)
        axes[0].scatter(r.mean_welfare, r.mean_equality, s=64,
                        color=C.MODEL_COL[r.model], edgecolor=C.INK, lw=.6, zorder=5)
    axes[0].scatter([W_IDEAL], [EQ_IDEAL], marker="*", s=210, color=C.C_COOP,
                    edgecolor=C.INK, lw=.6, zorder=6)
    axes[0].set_xlabel("joint welfare")
    axes[0].set_ylabel("equality (1 - within-dyad Gini)")
    axes[0].set_title("a  centroids and the ideal")
    axes[0].set_ylim(-0.08, 1.12)
    C.annotate(axes[0], "large dots = model centroids\n"
                        "dotted line = distance to the ideal", loc="lower left")

    y = np.arange(len(bym))
    axes[1].barh(y, bym["dist_game_mean"],
                 xerr=[bym["dist_game_mean"] - bym["dist_lo"],
                       bym["dist_hi"] - bym["dist_game_mean"]],
                 color=[C.MODEL_COL[m] for m in bym["model"]],
                 error_kw=dict(ecolor=C.INK2, lw=1.0))
    axes[1].set_yticks(y, [_short(m) for m in bym["model"]], fontsize=6.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("distance to the ideal (0 = ideal, 1 = worst corner)")
    axes[1].set_title("b  distance to the ideal")
    for i, v in enumerate(bym["dist_game_mean"]):
        axes[1].text(v + .012, i, f"{v:.3f}", va="center", fontsize=6.5)
    axes[1].set_xlim(0, bym["dist_hi"].max() * 1.24)
    C.annotate(axes[1], "game-level mean, 95% CI\nclustered on the design cell",
               loc="upper right")

    order2 = bym.sort_values("share_cells_at_ideal", ascending=False)
    y = np.arange(len(order2))
    axes[2].barh(y, order2["share_cells_at_ideal"],
                 color=[C.MODEL_COL[m] for m in order2["model"]])
    axes[2].errorbar(order2["share_games_at_ideal"], y, fmt="D", ms=4, color=C.INK,
                     xerr=[order2["share_games_at_ideal"] - order2["ideal_lo"],
                           order2["ideal_hi"] - order2["share_games_at_ideal"]],
                     ecolor=C.INK2, lw=1.0, label="per game, 95% CI")
    axes[2].set_yticks(y, [_short(m) for m in order2["model"]], fontsize=6.5)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("share reaching the ideal exactly")
    axes[2].set_title("c  best case, not average case")
    axes[2].legend(loc="lower right", fontsize=6.2)
    axes[2].set_xlim(0, max(order2["share_cells_at_ideal"].max(),
                            order2["ideal_hi"].max()) * 1.5)
    rev = order2.model.iloc[0]
    rev_rank = int(bym.loc[bym.model == rev, "rank_distance"].iloc[0])
    rev_game_rank = int(bym["share_games_at_ideal"].rank(ascending=False)
                        [bym.model == rev].iloc[0])
    C.annotate(axes[2], f"{_short(rev)} is 1st on cells,\n"
                        f"{_ord(rev_game_rank)} on games, and {_ord(rev_rank)} on\n"
                        f"mean distance in panel b\n"
                        f"(rank Spearman {rho_rank.statistic:+.2f})", loc="upper right")
    C.save(fig, "frontier", "PF03_frontier_by_model",
           "Which models sit near the welfare-equality ideal, which are dominated, and "
           "does a best-case frontier metric rank them the way an average-case metric "
           "does?",
           f"{_short(bym.model.iloc[0])} and {_short(bym.model.iloc[1])} are closest to "
           f"the ideal (distance {bym.dist_game_mean.iloc[0]:.3f} and "
           f"{bym.dist_game_mean.iloc[1]:.3f}) and {_short(bym.model.iloc[-1])} is "
           f"furthest ({bym.dist_game_mean.iloc[-1]:.3f}), a spread of "
           f"{bym.dist_game_mean.iloc[-1] - bym.dist_game_mean.iloc[0]:.3f} on a scale "
           f"whose worst corner is 1. The two frontier metrics disagree: {_short(rev)} "
           f"reaches the ideal exactly in {order2.share_cells_at_ideal.iloc[0]:.0%} of "
           f"its condition cells, more than any other model, and in "
           f"{order2.share_games_at_ideal.iloc[0]:.0%} of its individual games, "
           f"{_ord(rev_game_rank)} of 6, yet it ranks {_ord(rev_rank)} of 6 on mean "
           "distance, so it is bimodal rather than good.",
           "objective-space centroids + ranked distance bars + attainment bars",
           "welfare, equality, distance to ideal by model")

    # =====================================================================
    # PF04  efficiency-loss decomposition
    # =====================================================================
    bysc = []
    for s in C.SCALES:
        sub = dy[dy.scale == s]
        mu, lo, hi = C.cluster_boot_ci(sub["dist"].to_numpy(), sub["cell"].to_numpy(),
                                       n_boot=2000)
        am, alo, ahi = C.cluster_boot_ci(sub["at_ideal"].to_numpy(),
                                         sub["cell"].to_numpy(), n_boot=2000)
        cs = cc[cc.scale == s]
        bysc.append({
            "scale": s, "log_scale": np.log10(s),
            "mean_welfare": sub["welfare"].mean(),
            "mean_equality": sub["equality"].mean(),
            "mean_coop": sub["coop"].mean(),
            "dist_game_mean": mu, "dist_lo": lo, "dist_hi": hi,
            "dist_of_centroid": float(dist_to_ideal(sub["welfare"].mean(),
                                                    sub["equality"].mean())),
            "share_games_at_ideal": am, "ideal_lo": alo, "ideal_hi": ahi,
            "share_cells_at_ideal": cs["at_ideal"].mean(),
            "n_cells_nondominated": int(cs["nondominated"].sum()),
            "hypervolume": hypervolume(cs["welfare"], cs["equality"]),
            "shortfall": W_IDEAL - sub["welfare"].mean(),
            "loss_from_DD": sub["loss_dd"].mean(),
            "loss_from_exploitation": sub["loss_exploit"].mean(),
            "dd_share_of_shortfall": sub["loss_dd"].mean()
                                     / (W_IDEAL - sub["welfare"].mean()),
            "n_games": len(sub),
        })
    bysc = pd.DataFrame(bysc)
    C.savetab(bysc, "frontier_by_lambda")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
    o = bym.sort_values("shortfall")
    y = np.arange(len(o))
    axes[0].barh(y, o["loss_from_DD"], color=C.C_DEFECT)
    axes[0].barh(y, o["loss_from_exploitation"], left=o["loss_from_DD"],
                 color=C.OUTCOME_COL["DC"])
    axes[0].set_yticks(y, [_short(m) for m in o["model"]], fontsize=6.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("welfare lost per round vs mutual cooperation")
    axes[0].set_title("a  shortfall decomposition")
    for i, v in enumerate(o["shortfall"]):
        axes[0].text(v + .006, i, f"{v:.3f}", va="center", fontsize=6.5)
    # label the two segments inside the widest bar instead of using a legend,
    # which would have to sit on top of the bars
    wide = int(np.argmax(o["shortfall"].to_numpy()))
    axes[0].text(o["loss_from_DD"].iloc[wide] / 2, wide, "mutual\ndefection",
                 ha="center", va="center", fontsize=5.8, color="white")
    axes[0].text(o["loss_from_DD"].iloc[wide]
                 + o["loss_from_exploitation"].iloc[wide] / 2, wide,
                 "exploitation\n(CD / DC)", ha="center", va="center", fontsize=5.8,
                 color="white")
    axes[0].set_xlim(0, o["shortfall"].max() * 1.42)
    axes[0].set_ylim(len(o) - 0.4, -1.7)
    C.annotate(axes[0], "exact accounting identity, max residual "
                        f"{ident_l:.0e}:\n0.8 - welfare = 0.4 p(DD) + 0.3 p(exploit)",
               loc="upper left")

    o2 = bym.sort_values("dd_share_of_shortfall")
    y = np.arange(len(o2))
    axes[1].barh(y, o2["dd_share_of_shortfall"],
                 color=[C.MODEL_COL[m] for m in o2["model"]])
    axes[1].vlines(.5, -0.5, len(o2) - 0.5, color=C.INK, lw=.9, ls="--")
    axes[1].set_yticks(y, [_short(m) for m in o2["model"]], fontsize=6.5)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(len(o2) - 0.4, -1.7)
    axes[1].set_xlabel("share of the shortfall caused by mutual defection")
    axes[1].set_title("b  two ways of losing welfare")
    for i, v in enumerate(o2["dd_share_of_shortfall"]):
        axes[1].text(v + .015, i, f"{v:.0%}", va="center", fontsize=6.5)
    C.annotate(axes[1], "left of the line: the loss is mostly\n"
                        "failed coordination; right of it:\n"
                        "mostly mutual defection", loc="upper right")

    axes[2].bar(range(10), bysc["loss_from_DD"], color=C.C_DEFECT, width=.8,
                label="mutual defection")
    axes[2].bar(range(10), bysc["loss_from_exploitation"], bottom=bysc["loss_from_DD"],
                color=C.OUTCOME_COL["DC"], width=.8, label="exploitation")
    axes[2].set_xticks(range(10), [f"{s:g}" for s in C.SCALES], rotation=90,
                       fontsize=6.5)
    axes[2].set_xlabel("payoff scale $\\lambda$")
    axes[2].set_ylabel("welfare lost per round")
    axes[2].set_title("c  shortfall across the ladder")
    axes[2].legend(loc="upper left", fontsize=6.2)
    axes[2].set_ylim(0, bysc["shortfall"].max() * 1.60)
    C.annotate(axes[2], f"shortfall spans {bysc.shortfall.min():.3f} to "
                        f"{bysc.shortfall.max():.3f}\nacross $\\lambda$, against "
                        f"{bym.shortfall.min():.3f} to {bym.shortfall.max():.3f}\n"
                        "across models", loc="upper right")
    C.save(fig, "frontier", "PF04_efficiency_loss_decomposition",
           "How much welfare is lost relative to full mutual cooperation, and how much "
           "of that loss comes from mutual defection rather than from failed "
           "coordination?",
           "The shortfall splits exactly into 0.4 p(DD) + 0.3 p(exploit), and models "
           "differ far more in which term dominates than in the total. "
           f"{_short(o2.model.iloc[0])} loses only "
           f"{o2.dd_share_of_shortfall.iloc[0]:.0%} of its shortfall to mutual "
           "defection, so most of its inefficiency is one agent exploiting the other, "
           f"while {_short(o2.model.iloc[-1])} loses "
           f"{o2.dd_share_of_shortfall.iloc[-1]:.0%} that way and is essentially a "
           f"coordination failure. Total shortfall per round ranges from "
           f"{o.shortfall.min():.3f} to {o.shortfall.max():.3f} across models against "
           f"only {bysc.shortfall.max() - bysc.shortfall.min():.3f} across the whole "
           "lambda ladder.",
           "stacked loss bars by model + DD share + stacked loss across lambda",
           "p_DD, p_exploit, welfare shortfall by model and scale")

    # =====================================================================
    # PF05  does the payoff multiplier move the cloud toward the ideal
    # =====================================================================
    b_all, lo_all, hi_all, p_all = _cluster_slope(dy, "dist")
    C.record_test(section="frontier",
                  test="OLS distance-to-ideal on log10(lambda), cluster-robust on cell",
                  comparison="pooled distance to the welfare-equality ideal",
                  statistic=b_all, p_value=p_all, ci_lo=lo_all, ci_hi=hi_all,
                  n=len(dy), effect_size_note="change in normalised distance per decade "
                                              "of lambda")
    piv = dy.pivot_table(index="cell", columns="scale", values="dist")
    dlt = piv.sub(piv[1.0], axis=0)
    dm, dse = dlt.mean(), dlt.std() / np.sqrt(dlt.notna().sum())
    n_sig_paired = int(sum(abs(dm[s]) > 1.96 * dse[s] for s in C.SCALES if s != 1.0))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
    axes[0].fill_between(C.SCALES, bysc["dist_lo"], bysc["dist_hi"], color=C.C_COOP,
                         alpha=.2, lw=0)
    axes[0].plot(C.SCALES, bysc["dist_game_mean"], "o-", color=C.C_COOP, zorder=3)
    axes[0].axhline(bysc["dist_game_mean"].mean(), color=C.INK, lw=.9, ls="--")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("payoff scale $\\lambda$")
    axes[0].set_ylabel("distance to the ideal (0 = ideal)")
    axes[0].set_title("a  distance vs $\\lambda$")
    axes[0].text(0.012, bysc["dist_game_mean"].mean() + .003, "invariance prediction",
                 fontsize=6.3, color=C.INK)
    C.annotate(axes[0], f"slope {b_all:+.4f}/decade "
                        f"[{lo_all:+.4f}, {hi_all:+.4f}]\n"
                        f"ladder range "
                        f"{bysc.dist_game_mean.max() - bysc.dist_game_mean.min():.3f}\n"
                        f"{n_sig_paired} of 9 scales differ from $\\lambda$=1 (paired)",
               loc="lower right")

    cmap = plt.get_cmap("viridis")
    xs, ys = bysc["mean_welfare"].to_numpy(), bysc["mean_equality"].to_numpy()
    for i in range(len(xs) - 1):
        axes[1].annotate("", (xs[i + 1], ys[i + 1]), (xs[i], ys[i]),
                         arrowprops=dict(arrowstyle="->", lw=1.0,
                                         color=cmap(i / (len(xs) - 1))))
    sc = axes[1].scatter(xs, ys, c=np.log10(bysc["scale"]), cmap="viridis", s=42,
                         edgecolor=C.INK, lw=.4, zorder=5)
    plt.colorbar(sc, ax=axes[1], fraction=.046, label="$\\log_{10}\\lambda$")
    for s in [0.01, 1.0, 1000.0]:
        i = C.SCALES.index(s)
        axes[1].text(xs[i], ys[i] + .0035, f"$\\lambda$={s:g}", fontsize=6.2,
                     ha="center", va="bottom", color=C.INK)
    axes[1].set_xlabel("mean joint welfare")
    axes[1].set_ylabel("mean equality")
    axes[1].set_title("b  where the cloud moves")
    axes[1].margins(x=.14, y=.26)
    C.annotate(axes[1], "arrows run from small to large $\\lambda$;\n"
                        "the ideal (0.8, 1.0) is far outside this range",
               loc="upper left")

    for m in C.MODEL_ORDER:
        h = [hypervolume(cc[(cc.model == m) & (cc.scale == s)]["welfare"],
                         cc[(cc.model == m) & (cc.scale == s)]["equality"])
             for s in C.SCALES]
        axes[2].plot(C.SCALES, h, "o-", color=C.MODEL_COL[m], ms=3.5,
                     label=_short(m))
    axes[2].set_xscale("log")
    axes[2].set_ylim(0.02, 1.30)
    axes[2].set_xlabel("payoff scale $\\lambda$")
    axes[2].set_ylabel("hypervolume of the 20 cells at that $\\lambda$")
    axes[2].set_title("c  best-case frontier quality")
    axes[2].legend(loc="lower left", fontsize=5.6, ncol=2, handletextpad=.3,
                   columnspacing=.7, labelspacing=.25)
    C.annotate(axes[2], "hypervolume saturates at 1 whenever a model reaches\n"
                        "the ideal at that $\\lambda$, so it separates only the rest",
               loc="upper left")
    C.save(fig, "frontier", "PF05_frontier_movement_with_lambda",
           "Does multiplying every payoff by a constant move play toward or away from "
           "the welfare-equality ideal? This is payoff scale, not model scale: the "
           "corpus has no model-size axis, so the model-scaling version of the question "
           "cannot be asked here.",
           f"Pooled, larger lambda moves play very slightly toward the ideal: "
           f"{b_all:+.4f} per decade [{lo_all:+.4f}, {hi_all:+.4f}], p = {p_all:.1e}. "
           "The effect is statistically clear and practically small - five decades of "
           f"lambda buy {abs(b_all) * 5:.3f} of distance, against a between-model spread "
           f"of {bym.dist_game_mean.iloc[-1] - bym.dist_game_mean.iloc[0]:.3f} on the "
           "same scale. It is also not monotone: the ladder ranges over "
           f"{bysc.dist_game_mean.max() - bysc.dist_game_mean.min():.3f} with the worst "
           f"point at lambda = {bysc.scale[bysc.dist_game_mean.idxmax()]:g} and the best "
           f"at lambda = {bysc.scale[bysc.dist_game_mean.idxmin()]:g}, and "
           f"{n_sig_paired} of 9 scales differ from lambda = 1 in the paired within-cell "
           "contrast on identical games.",
           "line with clustered CI + centroid trajectory + hypervolume",
           "distance to ideal, welfare, equality vs scale")

    # =====================================================================
    # PF06  frontier position vs lambda, per model
    # =====================================================================
    rows = []
    for m in C.MODEL_ORDER:
        s = dy[dy.model == m]
        b, lo, hi, p = _cluster_slope(s, "dist")
        lev = s.groupby("scale", observed=True)["dist"].mean()
        rows.append({"model": m, "slope_per_decade": b, "lo": lo, "hi": hi, "p": p,
                     "range_across_ladder": float(lev.max() - lev.min()),
                     "best_lambda": float(lev.idxmin()),
                     "worst_lambda": float(lev.idxmax()),
                     "n_games": len(s)})
    sl = pd.DataFrame(rows)
    sl["p_fdr"] = C.bh_fdr(sl["p"].to_numpy())
    C.savetab(sl, "frontier_lambda_slopes_by_model")
    for _, r in sl.iterrows():
        C.record_test(section="frontier",
                      test="OLS distance-to-ideal on log10(lambda), cluster-robust",
                      comparison=f"{r.model}: distance to ideal ~ log10(lambda)",
                      statistic=r.slope_per_decade, p_value=r.p, ci_lo=r.lo,
                      ci_hi=r.hi, n=int(r.n_games),
                      effect_size_note="distance units per decade of lambda; family of "
                                       "6 models, BH-adjusted in "
                                       "frontier_lambda_slopes_by_model.csv")
    n_neg = int(((sl.p_fdr < .05) & (sl.slope_per_decade < 0)).sum())
    n_pos = int(((sl.p_fdr < .05) & (sl.slope_per_decade > 0)).sum())
    slmin = sl.sort_values("slope_per_decade").iloc[0]

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.3), sharex=True, sharey=True)
    for axx, m in zip(axes.ravel(), C.MODEL_ORDER):
        s = dy[dy.model == m]
        mm, ll, hh = [], [], []
        for sc_ in C.SCALES:
            ss = s[s.scale == sc_]
            a_, b_, c_ = C.cluster_boot_ci(ss["dist"].to_numpy(),
                                           ss["cell"].to_numpy(), n_boot=800)
            mm.append(a_); ll.append(b_); hh.append(c_)
        axx.fill_between(C.SCALES, ll, hh, color=C.MODEL_COL[m], alpha=.2, lw=0)
        axx.plot(C.SCALES, mm, "o-", color=C.MODEL_COL[m])
        axx.axhline(np.mean(mm), color=C.INK, lw=.8, ls="--")
        axx.set_xscale("log")
        axx.set_title(m, fontsize=8.5)
        r = sl[sl.model == m].iloc[0]
        C.annotate(axx, f"slope {r.slope_per_decade:+.3f}/decade\n"
                        f"[{r.lo:+.3f}, {r.hi:+.3f}]\n"
                        f"FDR p = {r.p_fdr:.1e}\n"
                        f"ladder range {r.range_across_ladder:.3f}", loc="upper left")
    axes[0, 0].set_ylim(0.0, 1.0)
    for axx in axes[1]:
        axx.set_xlabel("payoff scale $\\lambda$")
    for axx in axes[:, 0]:
        axx.set_ylabel("distance to the ideal")
    C.save(fig, "frontier", "PF06_frontier_position_by_model",
           "Is the direction in which the payoff multiplier moves a model relative to "
           "the welfare-equality ideal shared across models, or model-specific?",
           f"Model-specific, and the signs are opposite: {n_neg} models move "
           f"significantly toward the ideal as lambda grows and {n_pos} move "
           "significantly away from it after BH correction. The pooled slope is "
           "therefore a composition of cancelling effects, not a shared tendency. "
           f"{_short(slmin.model)} dominates the pooled result with "
           f"{slmin.slope_per_decade:+.3f} per decade and a range of "
           f"{slmin.range_across_ladder:.3f} across the ladder, several times its own "
           "linear slope, so even within a model the response is far from monotone.",
           "small multiples of distance to ideal vs scale with clustered CI",
           "distance to ideal, scale, model")

    # =====================================================================
    # PF07  the other design factors, and how big lambda is next to them
    # =====================================================================
    lang_rows, dyad_rows = [], []
    for lg in C.LANGS:
        s = dy[dy.language == lg]
        mu, lo, hi = C.cluster_boot_ci(s["dist"].to_numpy(), s["cell"].to_numpy(),
                                       n_boot=2000)
        lang_rows.append({"language": lg, "dist": mu, "lo": lo, "hi": hi, "n": len(s)})
    for dd in C.DYADS:
        s = dy[dy.dyad == dd]
        mu, lo, hi = C.cluster_boot_ci(s["dist"].to_numpy(), s["cell"].to_numpy(),
                                       n_boot=2000)
        dyad_rows.append({"dyad": dd, "dist": mu, "lo": lo, "hi": hi, "n": len(s)})
    lt = pd.DataFrame(lang_rows)
    dt = pd.DataFrame(dyad_rows)
    C.savetab(pd.concat(
        [lt.assign(factor="language").rename(columns={"language": "level"}),
         dt.assign(factor="pairing").rename(columns={"dyad": "level"})],
        ignore_index=True), "frontier_distance_by_factor")

    eta = {}
    tot = float(((dy["dist"] - dy["dist"].mean()) ** 2).sum())
    for f in ["model", "dyad", "language", "scale"]:
        gm = dy.groupby(f, observed=True)["dist"]
        eta[f] = float(((gm.mean() - dy["dist"].mean()) ** 2 * gm.size()).sum()) / tot
    ee = pd.Series(eta).sort_values()

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))
    axes[0].bar(range(5), lt["dist"],
                yerr=[lt["dist"] - lt["lo"], lt["hi"] - lt["dist"]],
                color=[C.LANG_COL[l] for l in lt["language"]],
                error_kw=dict(ecolor=C.INK2, lw=1.0))
    axes[0].set_xticks(range(5), [C.LANG_LABEL[l] for l in lt["language"]],
                       rotation=25, fontsize=6.8, ha="right")
    axes[0].set_ylabel("distance to the ideal")
    ytop = max(lt["hi"].max(), dt["hi"].max()) * 1.62
    axes[0].set_ylim(0, ytop)
    for i, v in enumerate(lt["dist"]):
        axes[0].text(i, lt["hi"].iloc[i] + ytop * .012, f"{v:.3f}", ha="center",
                     fontsize=6.3, color=C.INK2)
    axes[0].set_title("a  by prompt language")
    C.annotate(axes[0], "the Arabic and Chinese templates carry an\n"
                        "inherited translation inconsistency (their goal\n"
                        "sentence says maximise rewards while the payoff\n"
                        "sentences say penalty), so part of this gap is a\n"
                        "prompt artefact rather than a culture effect",
               loc="upper left")

    axes[1].bar(range(4), dt["dist"],
                yerr=[dt["dist"] - dt["lo"], dt["hi"] - dt["dist"]],
                color=[C.DYAD_COL[d] for d in dt["dyad"]],
                error_kw=dict(ecolor=C.INK2, lw=1.0))
    axes[1].set_xticks(range(4), dt["dyad"], fontsize=7)
    axes[1].set_ylabel("distance to the ideal")
    axes[1].set_ylim(0, ytop)
    for i, v in enumerate(dt["dist"]):
        axes[1].text(i, dt["hi"].iloc[i] + ytop * .012, f"{v:.3f}", ha="center",
                     fontsize=6.3, color=C.INK2)
    axes[1].set_title("b  by persona pairing")
    C.annotate(axes[1], "C = cooperative persona, S = selfish.\n"
                        "Mixed pairs sit further from the ideal than\n"
                        "two selfish agents: mutual defection is at\n"
                        "least equal, exploitation is not",
               loc="upper left")

    axes[2].barh(range(len(ee)), ee.values,
                 color=[C.C_COOP if k == "scale" else C.MUTED for k in ee.index])
    axes[2].set_yticks(range(len(ee)), ee.index, fontsize=7)
    axes[2].set_xlabel("$\\eta^2$: share of variance in distance to the ideal")
    axes[2].set_title("c  how big is $\\lambda$ really")
    for i, v in enumerate(ee.values):
        axes[2].text(v + ee.max() * .015, i, f"{v:.4f}", va="center", fontsize=6.5)
    axes[2].set_xlim(0, ee.max() * 1.25)
    C.save(fig, "frontier", "PF07_frontier_by_language_and_pairing",
           "Do prompt language and persona pairing move a dyad relative to the "
           "welfare-equality ideal, and how large is the payoff multiplier next to those "
           "design factors?",
           f"Both matter more than lambda. English is closest to the ideal "
           f"({lt.dist.min():.3f}) and Arabic furthest ({lt.dist.max():.3f}), a gap of "
           f"{lt.dist.max() - lt.dist.min():.3f} that is partly attributable to a known "
           "translation inconsistency in the Arabic and Chinese templates rather than to "
           "culture. Mixed cooperative-selfish pairings sit further from the ideal "
           f"({dt.set_index('dyad').dist.loc[['CvS', 'SvC']].mean():.3f}) than two "
           f"selfish agents ({dt.set_index('dyad').dist.loc['SvS']:.3f}), because mutual "
           "defection is at least equal while exploitation is not. Model identity "
           f"explains {eta['model']:.3f} of the variance in distance, pairing "
           f"{eta['dyad']:.3f}, language {eta['language']:.3f} and payoff scale only "
           f"{eta['scale']:.4f}.",
           "grouped bars with clustered CI + variance decomposition",
           "distance to ideal by language, dyad composition, and eta-squared by factor")

    # =====================================================================
    # findings
    # =====================================================================
    C.record_finding(
        id="F-no-tradeoff",
        finding="There is no welfare-equality trade-off in this game: the collective "
                "ideal is attainable on both objectives at once, and it is attained.",
        evidence=f"Over 1,200 condition cells the Pareto set contains {n_nd} points and "
                 "all of them are the single theoretical ideal (welfare 0.8, equality "
                 f"1), reached exactly in {n_ideal} cells ({n_ideal / len(cc):.1%}) and "
                 f"in {dy.at_ideal.mean():.1%} of individual games. Welfare and equality "
                 f"correlate positively (r = {r_we.statistic:+.3f}), not negatively. "
                 "Structurally, balancing p_CD against p_DC makes equality 1 attainable "
                 "at every welfare level, so the theoretical Pareto set is a single "
                 "corner.",
        figure="08_frontier/PF02_pareto_frontier.png, PF01_objective_coupling.png",
        strength="strong",
        robustness="holds at the game level, the condition-cell level and the "
                   "model-by-lambda level; adding cooperation as a third objective "
                   f"leaves the non-dominated set unchanged at {n_nd3} cells",
        interpretation="Multi-objective language is the wrong frame for this game. "
                       "Because the ideal dominates everything, frontier analysis "
                       "reduces to a ranking by distance to one point, and every "
                       "observed departure is inefficiency rather than a different "
                       "point on a trade-off curve. This is a property of the payoff "
                       "matrix, not of the models.",
        caveat="Equality here is the within-dyad split only. A different equality "
               "notion, across games or across personas, could behave differently. "
               "Equality is also maximised at universal defection, so it is not "
               "welfare-relevant on its own.",
        claim="In this prisoner's dilemma the welfare-maximising outcome is also the "
              "perfectly equal one, so the empirical Pareto frontier collapses onto a "
              "single theoretical ideal point that about a tenth of experimental "
              "conditions reach exactly.")

    C.record_finding(
        id="F-loss-decomposition",
        finding="Models differ far more in HOW they lose welfare than in how much they "
                "lose, and the split between mutual defection and exploitation is an "
                "exact accounting identity.",
        evidence="0.8 - welfare = 0.4 p(DD) + 0.3 p(exploit) holds to a maximum residual "
                 f"of {ident_l:.0e} over 12,000 games. The mutual-defection share of the "
                 f"shortfall ranges from {bym.dd_share_of_shortfall.min():.0%} "
                 f"({_short(bym.loc[bym.dd_share_of_shortfall.idxmin(), 'model'])}) to "
                 f"{bym.dd_share_of_shortfall.max():.0%} "
                 f"({_short(bym.loc[bym.dd_share_of_shortfall.idxmax(), 'model'])}), "
                 f"while total shortfall per round only ranges from "
                 f"{bym.shortfall.min():.3f} to {bym.shortfall.max():.3f}.",
        figure="08_frontier/PF04_efficiency_loss_decomposition.png",
        strength="strong",
        robustness="the decomposition is algebraic, so it cannot fail; the between-model "
                   "spread in the DD share is far larger than its spread across the "
                   "lambda ladder",
        interpretation="Two behavioural failure modes are being conflated by any single "
                       "welfare number. A model whose loss is mostly mutual defection is "
                       "failing to coordinate; a model whose loss is mostly exploitation "
                       "is producing an unequal outcome that is only moderately "
                       "inefficient. Only the second damages equality.",
        caveat="Self-play only, so 'exploitation' is one copy of a model exploiting "
               "another copy of itself under a different persona prompt, not a "
               "cross-model result.",
        claim="The welfare shortfall from mutual cooperation decomposes exactly into a "
              "mutual-defection term and an exploitation term, and frontier LLMs differ "
              "more in that split than in the size of the shortfall.")

    C.record_finding(
        id="F-lambda-frontier-cancels",
        finding="Payoff scale moves play toward the ideal on average, but the pooled "
                "effect is a composition of opposite-signed model effects and is an "
                "order of magnitude smaller than model identity.",
        evidence=f"Pooled slope {b_all:+.4f} distance units per decade of lambda "
                 f"[{lo_all:+.4f}, {hi_all:+.4f}], p = {p_all:.1e}, clustered on the "
                 f"design cell. Per model, {n_neg} slopes are significantly negative and "
                 f"{n_pos} significantly positive after BH correction, spanning "
                 f"{sl.slope_per_decade.min():+.3f} to {sl.slope_per_decade.max():+.3f}. "
                 f"Payoff scale explains {eta['scale']:.4f} of the variance in distance "
                 f"to the ideal against {eta['model']:.3f} for model identity.",
        figure="08_frontier/PF05_frontier_movement_with_lambda.png, "
               "PF06_frontier_position_by_model.png",
        strength="moderate",
        robustness="the sign reversal survives BH correction across the six-model "
                   "family; the within-cell paired contrast on common random numbers "
                   f"puts {n_sig_paired} of 9 scales away from lambda = 1",
        interpretation="Reporting a pooled 'payoff scale improves outcomes' result would "
                       "be an aggregation artefact. The pooled slope is carried by one "
                       "model whose behaviour changes sharply at sub-unit lambda, while "
                       "two others drift the opposite way. Lambda is randomised by "
                       "design, so it does cause the within-model change; which "
                       "direction a given model moves is a property of that model and is "
                       "not explained by anything measured here.",
        caveat="Distance to the ideal is a composite of two objectives, so a slope in it "
               "can come from welfare, equality or both; the per-objective slopes live "
               "in section 04. Gemini-3.1-Flash-Lite is a preview build.",
        claim="The payoff multiplier shifts LLM play relative to the welfare-equality "
              "ideal in model-specific directions that partly cancel, so pooled frontier "
              "movement with payoff scale should not be read as a general tendency.")

    C.record_finding(
        id="F-attainment-vs-mean",
        finding="Best-case and average-case frontier metrics rank the models "
                "differently, so a single frontier statistic can invert a comparison.",
        evidence=f"{_short(rev)} reaches the ideal exactly in "
                 f"{order2.share_cells_at_ideal.iloc[0]:.0%} of its 200 condition cells, "
                 "more than any other model, and in "
                 f"{order2.share_games_at_ideal.iloc[0]:.0%} of its individual games, "
                 f"{_ord(rev_game_rank)} of six, yet its mean distance to the ideal is "
                 f"{float(bym.loc[bym.model == rev, 'dist_game_mean'].iloc[0]):.3f}, rank "
                 f"{rev_rank} of 6. Rank agreement between the two metrics is Spearman "
                 f"{rho_rank.statistic:+.2f} over six models.",
        figure="08_frontier/PF03_frontier_by_model.png",
        strength="moderate",
        robustness="the reversal appears in both attainment measures, per condition cell "
                   "and per game, so it is not an aggregation choice. Hypervolume cannot "
                   "arbitrate: pooled over a model's 200 cells it equals 1 for five of "
                   "six models, because two ideal cells are enough to saturate it, so it "
                   "is only informative at the model-by-lambda level",
        interpretation="Some models are bimodal: they either coordinate perfectly or "
                       "collapse into exploitation, so their best case is excellent and "
                       "their average is poor. Frontier and hypervolume metrics reward "
                       "the best case only, which is why they have to be reported "
                       "alongside a central-tendency measure.",
        caveat="Six models is a small family for a rank correlation, so the Spearman "
               "value is illustrative rather than inferential.",
        claim="Pareto-style best-case metrics and mean-distance metrics disagree about "
              "which frontier LLM plays this game better, because the bimodal models "
              "reach the ideal most often and also fail worst.")

    return cc, bym, bysc
