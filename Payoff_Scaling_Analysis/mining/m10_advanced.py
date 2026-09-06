"""Sections 10, 11 and 21: correlation, interaction and advanced discovery.

Why this section is built the way it is
---------------------------------------
A correlation heatmap on this corpus is close to useless if it is read naively,
because most of the "variables" are algebraic re-expressions of one another. The
payoff matrix fixes joint_utility = 0.4 + 0.4 joint_coop - 0.1 p_exploit exactly
(verified below to machine precision), efficiency is 1.25 times utility by
definition, fairness is 1 - gini, and the four outcome shares sum to one. Left
unflagged, those identities would occupy the entire top of any ranked
correlation list and would be reported as discoveries. So every pair is written
out with an explicit mechanical/empirical verdict, and only pairs that survive
that filter are plotted.

The second reason this section exists is that the rest of the report estimates
pooled effects, and a pooled effect on this corpus is fragile in a specific way.
The design is a perfectly balanced 6 x 5 x 4 x 10 factorial with 10 replicates,
so aggregation cannot create a composition confound - but it can and does create
sign reversals, because the six models respond to the same manipulation in
opposite directions. The Simpson search below is therefore not a formality: the
headline payoff-scale slope is positive when pooled, and negative and
significant inside half the models. Any pooled number in this report has to be
read with that in mind.

The third reason is accounting. "What drives LLM cooperation" is answerable here
in a way it usually is not, because perfect balance makes the ANOVA
decomposition exactly orthogonal: sums of squares for the four main effects, all
six two-way, all four three-way and the four-way term are unique,
order-independent, and add exactly to the between-cell sum of squares. There is
no Type I / Type II / Type III ambiguity to argue about. The residual term is
then a clean estimate of replicate-to-replicate LLM sampling variance, because
the 10 replicates inside a design cell are the identical prompt run 10 times.

Estimation conventions
----------------------
Clustering is on `cell` (model | language | game_id) everywhere. `cell` is
nested inside model, language and pairing, and crossed with lambda: every cell
holds exactly 10 dyads, one per lambda, and exactly 20 agent-games. That exact
rectangularity is used to make the cluster bootstrap cheap - the row index can
be reshaped to (n_cells, k) and resampled by row.

Effect sizes are reported next to every p-value. With 12,000 dyads a Wald test
on an interaction rejects at essentially any effect size, so the tables carry an
interpretable magnitude column (spread of simple slopes, spread of interaction
contrasts) and the text says when a significant interaction is trivial.
"""
from __future__ import annotations

import itertools

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

from . import core as C

LOGS = np.log10(C.SCALES)

# --------------------------------------------------------------------------
# variable sets for the correlation mining
# --------------------------------------------------------------------------
DYAD_VARS = ["log_scale", "joint_coop", "joint_utility", "joint_efficiency",
             "p_CC", "p_CD", "p_DC", "p_DD", "p_exploit",
             "utility_gap", "gini", "fairness",
             "coop_rate_1", "coop_rate_2", "first_coop_1", "reciprocity_1",
             "n_switch", "first_dd", "longest_dd", "dd_absorbed"]

AG_VARS = ["log_scale", "coop_rate", "opp_coop_rate", "first_coop", "last_coop",
           "coop_early", "coop_late", "endgame_drop",
           "utility", "efficiency", "payoff_base",
           "cc_rate", "cd_rate", "dc_rate", "dd_rate",
           "pC_after_C", "pC_after_D", "reciprocity",
           "rule_distance", "n_rules_fit"]

# short display labels so a 20 x 20 heatmap stays readable
SHORT = {
    "log_scale": "log10 lambda", "joint_coop": "joint coop",
    "joint_utility": "joint util", "joint_efficiency": "joint eff",
    "p_exploit": "p exploit", "utility_gap": "util gap",
    "coop_rate_1": "coop A1", "coop_rate_2": "coop A2",
    "first_coop_1": "1st coop A1", "reciprocity_1": "recip A1",
    "n_switch": "n switch", "first_dd": "1st DD", "longest_dd": "longest DD",
    "dd_absorbed": "DD absorbed",
    "coop_rate": "coop rate", "opp_coop_rate": "opp coop", "first_coop": "1st coop",
    "last_coop": "last coop", "coop_early": "coop early", "coop_late": "coop late",
    "endgame_drop": "endgame drop", "payoff_base": "payoff base",
    "cc_rate": "CC rate", "cd_rate": "CD rate", "dc_rate": "DC rate",
    "dd_rate": "DD rate", "pC_after_C": "pC|oppC", "pC_after_D": "pC|oppD",
    "rule_distance": "rule dist", "n_rules_fit": "n rules fit",
}


def lab(v: str) -> str:
    return SHORT.get(v, v)


# --------------------------------------------------------------------------
# mechanical-relationship bookkeeping
# --------------------------------------------------------------------------
_D_SHARES = {"p_CC", "p_CD", "p_DC", "p_DD", "p_exploit"}
_D_COOP = {"joint_coop", "coop_rate_1", "coop_rate_2"}
_D_UTIL = {"joint_utility", "joint_efficiency"}
_D_FAIR = {"utility_gap", "gini", "fairness"}
_D_DD = {"dd_absorbed", "longest_dd", "first_dd", "p_DD"}

_A_SHARES = {"cc_rate", "cd_rate", "dc_rate", "dd_rate"}
_A_UTIL = {"utility", "efficiency", "payoff_base"}
_A_COOPPARTS = {"coop_rate", "coop_early", "coop_late", "first_coop", "last_coop"}
_A_RECIP = {"pC_after_C", "pC_after_D", "reciprocity"}


def mech_note(a: str, b: str, level: str) -> str:
    """Return '' if the pair is empirical, else the construction that links them.

    "Mechanical" here means the association is forced, in whole or in part, by an
    exact algebraic identity, by a simplex constraint, by rounds or outcomes that
    the two variables share, or by a hard structural bound. It does not mean the
    correlation is wrong - it means it is not evidence about behaviour. Every
    identity quoted in the returned strings is verified numerically in run().
    """
    s = {a, b}
    if "log_scale" in s:
        return ""
    if level == "dyad":
        if s == {"joint_utility", "joint_efficiency"}:
            return "identity: joint_efficiency = 1.25 x joint_utility"
        if s == {"gini", "fairness"}:
            return "identity: fairness = 1 - gini"
        if s <= _D_FAIR:
            return "identity: gini = utility_gap / (u1 + u2)"
        if s <= _D_SHARES:
            return "simplex: the four outcome shares sum to 1, p_exploit = p_CD + p_DC"
        if s <= _D_COOP:
            return "identity: joint_coop is the mean of coop_rate_1 and coop_rate_2"
        if (s & _D_FAIR) and (s & _D_SHARES):
            return ("exact bound: utility_gap = |p_DC - p_CD| and "
                    "p_exploit = p_CD + p_DC, so utility_gap <= p_exploit always")
        if (s & _D_COOP) and (s & (_D_SHARES | _D_UTIL)):
            return ("payoff-matrix identity: joint_coop = p_CC + p_exploit/2 and "
                    "joint_utility = 0.4 + 0.4 joint_coop - 0.1 p_exploit")
        if (s & _D_UTIL) and (s & _D_SHARES):
            return "payoff-matrix identity: joint_utility is a linear map of the shares"
        if (s & _D_UTIL) and (s & _D_FAIR):
            return ("construction: both are built from the same two agent utilities, "
                    "one as their mean and one as their absolute difference")
        if s <= _D_DD:
            return "same construct: all of these summarise the mutual-defection pattern"
        if (s & _D_DD) and (s & (_D_COOP | _D_UTIL | _D_SHARES)):
            return ("structural: DD rounds are exactly the rounds in which neither "
                    "agent cooperates, so the DD pattern bounds cooperation and "
                    "utility")
        if s == {"first_coop_1", "coop_rate_1"}:
            return "partly mechanical: round 1 is one of the 10 rounds averaged"
        if "first_coop_1" in s and (s & (_D_COOP | _D_SHARES | _D_UTIL | _D_DD)):
            return ("partly mechanical: agent 1's round-1 action is inside every one "
                    "of these aggregates")
        if "reciprocity_1" in s and (s & (_D_COOP | _D_SHARES | _D_UTIL)):
            return ("partly mechanical: reciprocity is a contrast between two "
                    "sub-averages of the same cooperation sequence")
        if "n_switch" in s and (s & (_D_COOP | _D_SHARES | _D_UTIL | _D_DD)):
            return ("structural bound: n_switch is forced to 0 when the agent never "
                    "changes action, i.e. at cooperation rates 0 and 1")
        return ""
    # ---- agent-game level ----
    if s == {"utility", "efficiency"}:
        return "identity: efficiency = 1.25 x utility"
    if s <= _A_UTIL:
        return "identity: utility = (10 - payoff_base)/10, efficiency = 1.25 x utility"
    if s <= _A_SHARES:
        return "simplex: the four outcome rates sum to 1"
    if (s & _A_UTIL) and (s & _A_SHARES):
        return "payoff-matrix identity: utility = 0.8 cc + 1.0 dc + 0.4 dd"
    if s == {"coop_rate", "cc_rate"} or s == {"coop_rate", "cd_rate"}:
        return "identity: coop_rate = cc_rate + cd_rate"
    if s == {"opp_coop_rate", "cc_rate"} or s == {"opp_coop_rate", "dc_rate"}:
        return "identity: opp_coop_rate = cc_rate + dc_rate"
    if s == {"coop_rate", "opp_coop_rate"}:
        return ("not an identity but not independent either: self-play, so the "
                "opponent is the same model and each dyad appears twice")
    if "opp_coop_rate" in s and (s & _A_UTIL):
        return ("payoff-matrix identity: utility = 0.4 + 0.4 opp_coop_rate "
                "+ 0.2 dc_rate - 0.4 cd_rate, and the opponent's action moves the "
                "focal penalty by 6 to 8 against 2 for the focal action")
    if "opp_coop_rate" in s and (s & _A_SHARES):
        return "partly mechanical: opp_coop_rate = cc_rate + dc_rate"
    if s == {"endgame_drop", "coop_early"} or s == {"endgame_drop", "coop_late"}:
        return "identity: endgame_drop = coop_early - coop_late"
    if s <= _A_COOPPARTS:
        # only nested pairs are mechanical; disjoint round windows are empirical
        nested = [{"coop_rate", "coop_early"}, {"coop_rate", "coop_late"},
                  {"coop_rate", "first_coop"}, {"coop_rate", "last_coop"},
                  {"coop_early", "first_coop"}, {"coop_late", "last_coop"}]
        if s in nested:
            return "nested: the second variable is one of the rounds averaged in the first"
        return ""
    if "endgame_drop" in s and (s & _A_COOPPARTS):
        return "partly mechanical: endgame_drop = coop_early - coop_late"
    if s <= _A_RECIP:
        return "identity: reciprocity = pC_after_C - pC_after_D"
    if (s & _A_RECIP) and (s & (_A_COOPPARTS | _A_SHARES)):
        return ("partly mechanical: the conditional cooperation rates are "
                "sub-averages of the same action sequence")
    if (s & _A_RECIP) and ("opp_coop_rate" in s):
        return "partly mechanical: opponent cooperation defines the conditioning event"
    if (s & _A_COOPPARTS) and (s & _A_SHARES):
        return "partly mechanical: the outcome rates contain the focal action"
    if (s & _A_COOPPARTS) and (s & _A_UTIL):
        return "payoff-matrix link: own utility is a linear map of the outcome rates"
    if s == {"rule_distance", "n_rules_fit"}:
        return "identity: rule_distance = 0 exactly when n_rules_fit >= 1"
    return ""


# --------------------------------------------------------------------------
# cheap exact cluster bootstrap (every cell holds exactly `per` rows)
# --------------------------------------------------------------------------
def cell_blocks(df: pd.DataFrame):
    """Return (row_index_matrix, n_cells) with one row of the matrix per cell."""
    cl = df["cell"].to_numpy()
    uniq, inv = np.unique(cl, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    per = len(df) // len(uniq)
    assert len(uniq) * per == len(df), "cells are not rectangular"
    return order.reshape(len(uniq), per), len(uniq)


def boot_mean_ci(values: np.ndarray, blocks: np.ndarray, n_boot=1000, seed=0):
    """Mean and 95% CI of `values` (NaN = excluded), resampling whole cells."""
    rng = np.random.default_rng(seed)
    k = blocks.shape[0]
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = np.nanmean(values[blocks[rng.integers(0, k, size=k)].ravel()])
    return (float(np.nanmean(values)), float(np.nanpercentile(means, 2.5)),
            float(np.nanpercentile(means, 97.5)))


def boot_contrast_ci(values, mask_a, mask_b, blocks, n_boot=800, seed=0):
    """Bootstrap CI for mean(a) - mean(b), resampling whole cells."""
    rng = np.random.default_rng(seed)
    k = blocks.shape[0]
    out = np.empty(n_boot)
    for i in range(n_boot):
        rows = blocks[rng.integers(0, k, size=k)].ravel()
        va, vb = values[rows][mask_a[rows]], values[rows][mask_b[rows]]
        out[i] = (np.nanmean(va) - np.nanmean(vb)) if len(va) and len(vb) else np.nan
    obs = float(np.nanmean(values[mask_a]) - np.nanmean(values[mask_b]))
    return obs, float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def clustered_slope(df: pd.DataFrame, y: str, x: str = "log_scale"):
    """OLS slope with cluster-robust covariance on the design cell."""
    dd = df[[y, x, "cell"]].dropna()
    if len(dd) < 30 or dd[x].std() == 0 or dd[y].std() == 0:
        return (np.nan,) * 5
    X = sm.add_constant(dd[x].to_numpy())
    m = sm.OLS(dd[y].to_numpy(), X).fit(cov_type="cluster",
                                        cov_kwds={"groups": dd["cell"].to_numpy()})
    ci = m.conf_int()
    return (float(m.params[1]), float(ci[1][0]), float(ci[1][1]),
            float(m.pvalues[1]), int(len(dd)))


# --------------------------------------------------------------------------
# exact orthogonal variance decomposition for the balanced factorial
# --------------------------------------------------------------------------
FACTORS = ["model", "language", "pairing", "lambda"]
DF_LEVELS = {"model": 5, "language": 4, "pairing": 3, "lambda": 9}


def variance_decomposition(dy: pd.DataFrame, value: str) -> pd.DataFrame:
    """Order-independent ANOVA shares; valid because the grid is exactly balanced."""
    t = dy.copy()
    t["mi"] = pd.Categorical(t.model, categories=C.MODEL_ORDER).codes
    t["li"] = pd.Categorical(t.language, categories=C.LANGS).codes
    t["di"] = pd.Categorical(t.dyad, categories=C.DYADS).codes
    t["si"] = pd.Categorical(t.scale, categories=C.SCALES).codes
    g = t.groupby(["mi", "li", "di", "si"], observed=True)[value]
    reps = g.size().unique()
    assert len(reps) == 1, "unbalanced grid"
    n_rep = int(reps[0])
    M = g.mean().to_numpy().reshape(6, 5, 4, 10)
    y = t[value].to_numpy(float)
    gm = y.mean()
    sst = float(((y - gm) ** 2).sum())

    axes = [0, 1, 2, 3]
    eff = {}
    for size in range(0, 5):
        for sub in itertools.combinations(axes, size):
            other = tuple(x for x in axes if x not in sub)
            e = M.mean(axis=other, keepdims=True) if other else M.copy()
            for k in range(size):
                for s2 in itertools.combinations(sub, k):
                    e = e - eff[s2]
            eff[sub] = e

    rows, ss_between = [], 0.0
    for sub, e in eff.items():
        if not sub:
            continue
        ss = n_rep * float((np.broadcast_to(e, M.shape) ** 2).sum())
        ss_between += ss
        dfree = int(np.prod([DF_LEVELS[FACTORS[i]] for i in sub]))
        rows.append({"term": " x ".join(FACTORS[i] for i in sub),
                     "order": len(sub), "df": dfree, "ss": ss,
                     "factors": [FACTORS[i] for i in sub]})
    ss_res = sst - ss_between
    df_res = len(y) - M.size
    ms_res = ss_res / df_res
    for r in rows:
        r["eta2"] = r["ss"] / sst
        r["omega2"] = max(0.0, r["ss"] - r["df"] * ms_res) / (sst + ms_res)
    rows.append({"term": "residual (replicate sampling noise)", "order": 99,
                 "df": df_res, "ss": ss_res, "factors": [],
                 "eta2": ss_res / sst, "omega2": ss_res / (sst + ms_res)})
    out = pd.DataFrame(rows)
    out.insert(0, "outcome", value)
    out["ss_total"] = sst
    return out.sort_values("eta2", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
def _density_panel(ax, x, y, xlabel, ylabel, nbins=40):
    """2-D density for a discrete-valued pair, plus its conditional mean."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    h = ax.hist2d(x, y, bins=nbins, cmap=C.SEQ, norm=mcolors.LogNorm(),
                  rasterized=True)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    if len(edges) > 2:
        idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
        ctr = np.array([x[idx == i].mean() if (idx == i).any() else np.nan
                        for i in range(len(edges) - 1)])
        mu = np.array([y[idx == i].mean() if (idx == i).any() else np.nan
                       for i in range(len(edges) - 1)])
        ax.plot(ctr, mu, "o-", color=C.C_DEFECT, lw=1.4, ms=3.5, zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    return h[3]


# ==========================================================================
def run(rounds, ag, dy):
    C.use_style()
    print("\n== 10 advanced: correlation, interaction and discovery ==")

    dyb, n_cell_d = cell_blocks(dy)

    # verify the payoff-matrix identities the notes claim, to machine precision
    id_dy = float(np.abs(dy.joint_utility
                         - (0.4 + 0.4 * dy.joint_coop - 0.1 * dy.p_exploit)).max())
    id_ag = float(np.abs(ag.efficiency - 1.25 * ag.utility).max())
    id_gap = float(np.abs(dy.utility_gap - (dy.p_DC - dy.p_CD).abs()).max())
    id_opp = float(np.abs(ag.utility - (0.4 + 0.4 * ag.opp_coop_rate
                                        + 0.2 * ag.dc_rate - 0.4 * ag.cd_rate)).max())
    print(f"  identity checks (max abs residual): joint_utility {id_dy:.1e}, "
          f"efficiency {id_ag:.1e}, utility_gap {id_gap:.1e}, "
          f"opp-driven utility {id_opp:.1e}")

    # ======================================================================
    # 1. CORRELATION AND DEPENDENCY MINING
    # ======================================================================
    def corr_table(df, varlist, level):
        rows = []
        cl = df["cell"].to_numpy()
        arrs = {v: pd.to_numeric(df[v], errors="coerce").to_numpy(float)
                for v in varlist}
        for a, b in itertools.combinations(varlist, 2):
            xa, xb = arrs[a], arrs[b]
            m = np.isfinite(xa) & np.isfinite(xb)
            n = int(m.sum())
            if n < 50 or xa[m].std() == 0 or xb[m].std() == 0:
                rows.append({"var_a": a, "var_b": b, "level": level,
                             "pearson_r": np.nan, "spearman_rho": np.nan, "n": n,
                             "p_value": np.nan,
                             "note": "degenerate: not enough variation"})
                continue
            pr = float(np.corrcoef(xa[m], xb[m])[0, 1])
            sp = float(stats.spearmanr(xa[m], xb[m]).statistic)
            xs = (xa[m] - xa[m].mean()) / xa[m].std()
            ys = (xb[m] - xb[m].mean()) / xb[m].std()
            fit = sm.OLS(ys, sm.add_constant(xs)).fit(
                cov_type="cluster", cov_kwds={"groups": cl[m]})
            note = mech_note(a, b, level)
            if not note and abs(pr) > 0.999:
                note = "numerically an identity: |r| > 0.999"
            rows.append({"var_a": a, "var_b": b, "level": level,
                         "pearson_r": pr, "spearman_rho": sp, "n": n,
                         "p_value": float(fit.pvalues[1]),
                         "note": note or "empirical"})
        return pd.DataFrame(rows)

    cor_d = corr_table(dy, DYAD_VARS, "dyad")
    cor_a = corr_table(ag, AG_VARS, "agent_game")
    cor = pd.concat([cor_d, cor_a], ignore_index=True)
    cor["p_fdr"] = C.bh_fdr(cor["p_value"].to_numpy())
    cor["mechanical"] = cor["note"] != "empirical"
    cor["abs_spearman"] = cor["spearman_rho"].abs()
    cor = cor.sort_values("abs_spearman", ascending=False).reset_index(drop=True)
    cor_d = cor[cor.level == "dyad"]
    cor_a = cor[cor.level == "agent_game"]
    C.savetab(cor[["var_a", "var_b", "level", "pearson_r", "spearman_rho", "n",
                   "p_value", "p_fdr", "note"]], "correlations")

    emp = cor[~cor.mechanical].dropna(subset=["abs_spearman"]).copy()
    n_top20_mech = int(cor.head(20)["mechanical"].sum())
    strongest_emp = emp.iloc[0]

    # ---- A01 correlation structure ---------------------------------------
    def _heat(ax, df, varlist, title):
        n = len(varlist)
        M = np.full((n, n), np.nan)
        Mm = np.zeros((n, n), bool)
        pos = {v: i for i, v in enumerate(varlist)}
        for _, r in df.iterrows():
            i, j = pos[r.var_a], pos[r.var_b]
            M[i, j] = M[j, i] = r.spearman_rho
            Mm[i, j] = Mm[j, i] = r.mechanical
        np.fill_diagonal(M, 1.0)
        im = ax.imshow(M, cmap=C.DIV, vmin=-1, vmax=1)
        yy, xx = np.where(Mm)
        ax.scatter(xx, yy, s=1.8, color=C.INK, zorder=3)
        ax.set_xticks(range(n), [lab(v) for v in varlist], rotation=90, fontsize=5.4)
        ax.set_yticks(range(n), [lab(v) for v in varlist], fontsize=5.4)
        ax.set_title(title)
        ax.grid(False)
        return im

    fig = plt.figure(figsize=(14.4, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.95])
    ax0, ax1, ax2 = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))
    im = _heat(ax0, cor_d, DYAD_VARS, "a  dyad level, Spearman")
    _heat(ax1, cor_a, AG_VARS, "b  agent-game level, Spearman")
    cb = fig.colorbar(im, ax=[ax0, ax1], fraction=.023, pad=.012)
    cb.set_label("Spearman rho", fontsize=7)
    C.annotate(ax0, "black dot = mechanically linked pair", loc="lower left",
               fontsize=6.2)

    ax2.axhline(0, color=C.GRID, lw=.7)
    ax2.axvline(0, color=C.GRID, lw=.7)
    ax2.plot([-1, 1], [-1, 1], color=C.MUTED, lw=.7, ls=":")
    ax2.scatter(cor.loc[cor.mechanical, "pearson_r"],
                cor.loc[cor.mechanical, "spearman_rho"], s=10, alpha=.75,
                color=C.C_DEFECT, label="mechanical", rasterized=True)
    ax2.scatter(cor.loc[~cor.mechanical, "pearson_r"],
                cor.loc[~cor.mechanical, "spearman_rho"], s=10, alpha=.8,
                color=C.C_COOP, label="empirical", rasterized=True)
    ax2.set_xlabel("Pearson r")
    ax2.set_ylabel("Spearman rho")
    ax2.set_title("c  every strong pair is definitional")
    ax2.set_xlim(-1.28, 1.28)
    ax2.set_ylim(-1.28, 1.35)
    ax2.legend(loc="upper left", fontsize=6.5)
    C.annotate(ax2, f"{n_top20_mech} of the 20 largest |rho| are mechanical\n"
                    f"strongest empirical pair: {lab(strongest_emp.var_a)} vs "
                    f"{lab(strongest_emp.var_b)}, rho = "
                    f"{strongest_emp.spearman_rho:+.2f}",
               loc="lower right", fontsize=6.3)
    C.save(fig, "advanced", "A01_correlation_structure",
           "Which variables in this corpus actually co-vary for behavioural "
           "reasons, once relationships that the payoff matrix forces are removed?",
           f"Almost none of the strong ones. {n_top20_mech} of the 20 largest "
           f"absolute Spearman correlations are algebraic identities such as "
           f"joint_utility = 0.4 + 0.4 joint_coop - 0.1 p_exploit (verified to "
           f"{id_dy:.0e}) or efficiency = 1.25 x utility. The strongest genuinely "
           f"empirical pair is {lab(strongest_emp.var_a)} vs "
           f"{lab(strongest_emp.var_b)} at rho = {strongest_emp.spearman_rho:+.2f}, "
           "well below the mechanical band. Pearson and Spearman agree in sign "
           "everywhere, so the ranking is not an artefact of the bounded, "
           "boundary-inflated marginals.",
           "two correlation heatmaps + Pearson vs Spearman scatter",
           "20 dyad-level and 20 agent-game-level variables")

    # ---- A02 the strongest non-mechanical relationships --------------------
    top3 = emp.head(3)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    for k, (axx, (_, r)) in enumerate(zip(axes, top3.iterrows())):
        src = dy if r.level == "dyad" else ag
        x = pd.to_numeric(src[r.var_a], errors="coerce").to_numpy(float)
        y = pd.to_numeric(src[r.var_b], errors="coerce").to_numpy(float)
        h = _density_panel(axx, x, y, f"{lab(r.var_a)}  ({r.level})", lab(r.var_b))
        plt.colorbar(h, ax=axx, fraction=.046, pad=.02,
                     label="observations" if k == 2 else None)
        axx.set_title(f"{'abc'[k]}  {lab(r.var_a)} vs {lab(r.var_b)}")
        C.annotate(axx, f"rho = {r.spearman_rho:+.3f}   r = {r.pearson_r:+.3f}\n"
                        f"n = {int(r.n):,}   FDR p = {r.p_fdr:.2g}\n"
                        "red line = conditional mean", loc="upper left",
                   fontsize=6.3)
    C.save(fig, "advanced", "A02_nonmechanical_relationships",
           "What do the three strongest non-definitional dependencies in the "
           "corpus actually look like?",
           "They are moderate and monotone rather than tight. The leading pair is "
           f"{lab(top3.iloc[0].var_a)} vs {lab(top3.iloc[0].var_b)} at rho = "
           f"{top3.iloc[0].spearman_rho:+.3f}; the third is already down to "
           f"{top3.iloc[2].spearman_rho:+.3f}. The conditional-mean curves are close "
           "to straight, so a rank correlation is an adequate summary and no hidden "
           "non-monotone structure is being averaged away.",
           "2-D density with conditional mean",
           "top three empirical pairs from tables/correlations.csv")

    # ======================================================================
    # 2. INTERACTION EFFECTS
    # ======================================================================
    # note: patsy resolves bare names in this namespace, and `C` here is the core
    # module, so the categorical wrapper C() cannot be used.  model, language and
    # dyad already carry pandas category dtype, and scale is given its own factor
    # column, so patsy codes all four as categorical without the wrapper.
    dyf = dy.copy()
    dyf["scale_f"] = pd.Categorical(dyf["scale"].map(lambda v: f"{v:g}"),
                                    categories=[f"{s:g}" for s in C.SCALES])
    base_f = "joint_coop ~ model + language + dyad + log_scale"
    specs = {
        "model x lambda": base_f + " + model:log_scale",
        "language x lambda": base_f + " + language:log_scale",
        "pairing x lambda": base_f + " + dyad:log_scale",
        "model x pairing": base_f + " + model:dyad",
        "model x language": base_f + " + model:language",
    }
    groups = dy["cell"].to_numpy()
    base = smf.ols(base_f, data=dyf).fit(cov_type="cluster",
                                         cov_kwds={"groups": groups})
    base_names = set(base.params.index)
    pooled_lambda_slope = float(base.params["log_scale"])

    def simple_slopes(key, order):
        return {lv: clustered_slope(dy[dy[key] == lv], "joint_coop")[0]
                for lv in order}

    def cross_contrast(f1, f2):
        """Max-min of the two-way interaction contrast, in cooperation units."""
        piv = dy.pivot_table(index=f1, columns=f2, values="joint_coop",
                             observed=True)
        dev = (piv.sub(piv.mean(axis=1), axis=0).sub(piv.mean(axis=0), axis=1)
               + piv.values.mean())
        return float(np.nanmax(dev.values) - np.nanmin(dev.values))

    inter_rows = []
    for name, f in specs.items():
        full = smf.ols(f, data=dyf).fit(cov_type="cluster",
                                        cov_kwds={"groups": groups})
        idx = [i for i, nm in enumerate(full.params.index) if nm not in base_names]
        R = np.zeros((len(idx), len(full.params)))
        R[np.arange(len(idx)), idx] = 1.0
        wt = full.f_test(R)
        if name.endswith("lambda"):
            key, order = {"model x lambda": ("model", C.MODEL_ORDER),
                          "language x lambda": ("language", C.LANGS),
                          "pairing x lambda": ("dyad", C.DYADS)}[name]
            sl = simple_slopes(key, order)
            spread = max(sl.values()) - min(sl.values())
            eff_size = 5 * spread
            eff_lab = "spread of simple slopes over the 5-decade ladder"
            eff_note = (f"simple slopes per decade range {min(sl.values()):+.4f} to "
                        f"{max(sl.values()):+.4f}, spread {spread:.4f} per decade, "
                        f"i.e. {eff_size:.3f} in cooperation across the ladder")
            n_rev = int(sum(1 for v in sl.values()
                            if np.sign(v) != np.sign(pooled_lambda_slope)))
        else:
            f2 = "dyad" if name.endswith("pairing") else "language"
            eff_size = cross_contrast("model", f2)
            eff_lab = f"range of model x {f2} interaction contrasts"
            eff_note = (f"largest minus smallest additive-model residual across the "
                        f"model x {f2} table: {eff_size:.3f} in cooperation units")
            n_rev = np.nan
        inter_rows.append({
            "interaction": name, "df_num": len(idx),
            "r2_base": base.rsquared, "r2_full": full.rsquared,
            "incremental_r2": full.rsquared - base.rsquared,
            "wald_F": float(np.squeeze(wt.statistic)),
            "p_value": float(np.squeeze(wt.pvalue)),
            "effect_size": eff_size, "effect_size_label": eff_lab,
            "n_subgroup_sign_flips": n_rev, "n": len(dy), "effect_note": eff_note})
    inter = pd.DataFrame(inter_rows)
    inter["p_fdr"] = C.bh_fdr(inter["p_value"].to_numpy())
    inter = inter.sort_values("incremental_r2", ascending=False).reset_index(drop=True)
    C.savetab(inter, "adv_interactions")
    for _, r in inter.iterrows():
        C.record_test(section="advanced",
                      test="cluster-robust Wald test on an interaction block "
                           "(clusters = design cell)",
                      comparison=f"joint_coop: {r.interaction}",
                      statistic=r.wald_F, p_value=r.p_value, n=r.n,
                      effect_size_note=f"incremental R2 = {r.incremental_r2:.4f}; "
                                       + r.effect_note)
    lam_inter_eff = float(inter.loc[inter.interaction == "model x lambda",
                                    "effect_size"].iloc[0])

    # ---- A03 interaction summary ------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7))
    y = np.arange(len(inter))
    axes[0].barh(y, inter.incremental_r2, color=C.C_COOP)
    axes[0].axvline(base.rsquared, color=C.INK, lw=.9, ls="--")
    axes[0].text(base.rsquared, -0.75, f"additive base $R^2$ = {base.rsquared:.3f}",
                 fontsize=6.3, color=C.INK, va="center", ha="right")
    axes[0].set_yticks(y, inter.interaction, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("incremental $R^2$ over the additive model")
    axes[0].set_title("a  how much the interaction adds")
    axes[0].set_xlim(0, max(inter.incremental_r2.max(), base.rsquared) * 1.18)
    for i, v in enumerate(inter.incremental_r2):
        axes[0].text(v + .003, i, f"{v:.3f}", va="center", fontsize=6.3)

    axes[1].barh(y, inter.effect_size, color=C.OUTCOME_COL["DC"])
    axes[1].set_yticks(y, inter.interaction, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("spread of the interaction, in cooperation units")
    axes[1].set_title("b  magnitude, not significance")
    axes[1].set_xlim(0, inter.effect_size.max() * 1.28)
    for i, v in enumerate(inter.effect_size):
        axes[1].text(v + .006, i, f"{v:.3f}", va="center", fontsize=6.3)
    C.annotate(axes[1], "every interaction here is FDR-significant, so\n"
                        "bar length is the only informative axis; slope\n"
                        "spread and contrast range are different\n"
                        "constructions, so compare within family",
               loc="lower right", fontsize=6.0)

    nlp = -np.log10(np.maximum(inter.p_fdr.to_numpy(), 1e-300))
    axes[2].scatter(inter.incremental_r2, nlp, s=54, color=C.C_DEFECT, zorder=3)
    for (x_, y_), nm in zip(zip(inter.incremental_r2, nlp), inter.interaction):
        axes[2].annotate(nm, (x_, y_), textcoords="offset points", xytext=(6, 4),
                         fontsize=6.2, color=C.INK2)
    axes[2].set_xlabel("incremental $R^2$")
    axes[2].set_ylabel("$-\\log_{10}$ FDR p")
    axes[2].set_title("c  significance is uninformative here")
    axes[2].set_xlim(-0.01, inter.incremental_r2.max() * 1.7)
    axes[2].set_ylim(0, nlp.max() * 1.25)
    C.save(fig, "advanced", "A03_interaction_effects",
           "Do model, language, personality pairing and payoff scale act "
           "additively on cooperation, or do they interact?",
           f"They interact strongly, and the ordering is not the obvious one. "
           f"model x pairing adds the most, {inter.iloc[0].incremental_r2:.3f} of R2 "
           f"on top of an additive base of {base.rsquared:.3f}, and spans "
           f"{inter.iloc[0].effect_size:.3f} in cooperation units. The three "
           "model x language comes second at "
           f"{float(inter.loc[inter.interaction=='model x language','incremental_r2'].iloc[0]):.3f}. "
           "The three interactions with lambda are all FDR-significant but smaller: "
           f"the largest, model x lambda, adds {float(inter.loc[inter.interaction=='model x lambda','incremental_r2'].iloc[0]):.3f} "
           f"of R2 and spreads the per-decade slope by {lam_inter_eff:.3f} in "
           "cooperation across the whole five-decade ladder. Every p-value is below "
           "1e-06, so p-values carry no information at this sample size and only the "
           "magnitudes are worth reading.",
           "incremental R2 bars + effect-size bars + volcano",
           "joint_coop ~ model, language, pairing, log10 lambda and interactions")

    # ---- A04 interaction plots --------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    for m in C.MODEL_ORDER:
        sub = dy[dy.model == m]
        bl, _ = cell_blocks(sub)
        vals = sub["joint_coop"].to_numpy()
        mu, lo_, hi_ = [], [], []
        for j, dd in enumerate(C.DYADS):
            mk = (sub["dyad"].to_numpy() == dd)
            a_, b_, c_ = boot_mean_ci(np.where(mk, vals, np.nan), bl, n_boot=300,
                                      seed=j)
            mu.append(a_); lo_.append(b_); hi_.append(c_)
        mu, lo_, hi_ = np.array(mu), np.array(lo_), np.array(hi_)
        axes[0].errorbar(np.arange(4), mu, yerr=[mu - lo_, hi_ - mu], fmt="o-",
                         color=C.MODEL_COL[m], lw=1.4, ms=4,
                         ecolor=C.MODEL_COL[m],
                         label=m.replace("-Non-Reasoning", ""))
    axes[0].set_xticks(range(4), C.DYADS)
    axes[0].set_xlim(-0.35, 3.35)
    axes[0].set_ylim(-0.05, 1.28)
    axes[0].set_xlabel("personality pairing (focal v opponent)")
    axes[0].set_ylabel("dyad cooperation rate")
    axes[0].set_title("a  model $\\times$ pairing: crossover")
    axes[0].legend(loc="upper left", fontsize=5.6, ncol=2, columnspacing=.7,
                   handletextpad=.4, handlelength=1.4)

    add_pred = pd.Series(base.fittedvalues, index=dy.index)
    for m in C.MODEL_ORDER:
        sub = dy[dy.model == m]
        obs = sub.groupby("scale", observed=True)["joint_coop"].mean()
        prd = add_pred.loc[sub.index].groupby(sub["scale"].to_numpy()).mean()
        axes[1].plot(C.SCALES, obs.reindex(C.SCALES).to_numpy(), "o-",
                     color=C.MODEL_COL[m], ms=3.4, lw=1.3)
        axes[1].plot(C.SCALES, prd.reindex(C.SCALES).to_numpy(), ls=":", lw=1.1,
                     color=C.MODEL_COL[m], alpha=.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("payoff scale $\\lambda$")
    axes[1].set_ylabel("dyad cooperation rate")
    axes[1].set_title("b  model $\\times$ $\\lambda$: obs vs additive")
    axes[1].set_ylim(0.15, 1.06)
    C.annotate(axes[1], "solid = observed, dotted = additive model\n"
                        "the additive fit forces one common slope",
               loc="upper left", fontsize=6.2)

    for m in C.MODEL_ORDER:
        sub = dy[dy.model == m]
        obs = sub.groupby("language", observed=True)["joint_coop"].mean()
        axes[2].plot(range(len(C.LANGS)), obs.reindex(C.LANGS).to_numpy(), "o-",
                     color=C.MODEL_COL[m], ms=4, lw=1.3)
    axes[2].set_xticks(range(len(C.LANGS)), [C.LANG_LABEL[l] for l in C.LANGS],
                       rotation=20, fontsize=7)
    axes[2].set_xlim(-0.3, 4.3)
    axes[2].set_xlabel("prompt language")
    axes[2].set_ylabel("dyad cooperation rate")
    axes[2].set_title("c  model $\\times$ language: rank order changes")
    C.save(fig, "advanced", "A04_interaction_plots",
           "What do the three largest interactions look like, and are they "
           "crossovers or merely differences in slope?",
           "All three are crossovers, which is why the corresponding main effects "
           "are misleading. Under model x pairing the persona effect reverses: "
           "Gemini-3.1-Flash-Lite cooperates most when both agents are told to be "
           "cooperative and least when both are selfish, while Qwen3-235B and Grok "
           "do the opposite. Under model x lambda the fitted additive slope is a "
           "single common line that no individual model follows. Under model x "
           "language the rank order of languages differs between models, so a pooled "
           "language effect averages over opposite orderings.",
           "three interaction plots with clustered CIs",
           "joint_coop x model x pairing, lambda, language")

    # ======================================================================
    # 3. VARIANCE DECOMPOSITION
    # ======================================================================
    vd_main = variance_decomposition(dy, "joint_coop")
    others = ["joint_utility", "p_CC", "utility_gap", "n_switch"]
    vd_all = pd.concat([vd_main] + [variance_decomposition(dy, v) for v in others],
                       ignore_index=True)
    C.savetab(vd_all.drop(columns=["factors"]), "adv_variance_decomposition")

    def _share(term):
        return float(vd_main.loc[vd_main.term == term, "eta2"].iloc[0])

    resid_share = float(vd_main.loc[vd_main.term.str.startswith("residual"),
                                    "eta2"].iloc[0])
    top_term = vd_main.iloc[0]
    lam_share, mod_share = _share("lambda"), _share("model")
    lang_share, pair_share = _share("language"), _share("pairing")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    v = vd_main.sort_values("eta2")
    ordcol = {1: C.C_COOP, 2: C.OUTCOME_COL["DC"], 3: C.OUTCOME_COL["CD"],
              4: C.C_DEFECT, 99: C.MUTED}
    yy = np.arange(len(v))
    axes[0].barh(yy, 100 * v.eta2, color=[ordcol[o] for o in v.order])
    axes[0].scatter(100 * v.omega2, yy, marker="|", s=44, color=C.INK, zorder=4,
                    label="$\\omega^2$, df-corrected")
    axes[0].set_yticks(yy, v.term, fontsize=6.2)
    axes[0].set_xlabel("share of variance in dyad cooperation (%)")
    axes[0].set_title("a  exact orthogonal decomposition")
    axes[0].set_xlim(0, 100 * v.eta2.max() * 1.22)
    axes[0].legend(loc="lower right", fontsize=6.3)
    for i, e in enumerate(v.eta2):
        if e > 0.01:
            axes[0].text(100 * e + .4, i, f"{100*e:.1f}", va="center", fontsize=6.2)
    C.annotate(axes[0], f"15 design terms sum to {100*(1-resid_share):.1f}%,\n"
                        f"residual {100*resid_share:.1f}%", loc="center right",
               fontsize=6.3)

    invol = {f: float(vd_main[vd_main.factors.apply(lambda s: f in s)]["eta2"].sum())
             for f in FACTORS}
    invol["residual"] = resid_share
    iv = pd.Series(invol).sort_values()
    axes[1].barh(range(len(iv)), 100 * iv.values,
                 color=[C.MUTED if k == "residual" else C.C_COOP for k in iv.index])
    axes[1].set_yticks(range(len(iv)), iv.index)
    axes[1].set_xlabel("variance share of all terms containing the factor (%)")
    axes[1].set_title("b  factor involvement (overlapping)")
    axes[1].set_xlim(0, 100 * iv.max() * 1.25)
    for i, val in enumerate(iv.values):
        axes[1].text(100 * val + .8, i, f"{100*val:.1f}%", va="center", fontsize=6.5)
    C.annotate(axes[1], "bars overlap by construction: an\ninteraction term is "
                        "counted once for\nevery factor it contains",
               loc="lower right", fontsize=6.2)

    order_lab = {1: "main effects", 2: "2-way", 3: "3-way", 4: "4-way",
                 99: "residual"}
    outs = ["joint_coop"] + others
    bottom = np.zeros(len(outs))
    for o in [1, 2, 3, 4, 99]:
        vals = np.array([vd_all[(vd_all.outcome == oc)
                                & (vd_all.order == o)]["eta2"].sum() for oc in outs])
        axes[2].bar(range(len(outs)), 100 * vals, bottom=100 * bottom,
                    color=ordcol[o], label=order_lab[o], width=.72)
        bottom += vals
    axes[2].set_xticks(range(len(outs)), [o.replace("_", " ") for o in outs],
                       rotation=20, fontsize=7)
    axes[2].set_ylabel("share of variance (%)")
    axes[2].set_title("c  same picture for four other outcomes")
    axes[2].set_ylim(0, 126)
    axes[2].legend(loc="upper center", fontsize=6.2, ncol=3, columnspacing=.8)
    C.save(fig, "advanced", "A05_variance_decomposition",
           "What actually drives cooperation in this corpus: which model, which "
           "language, which persona pairing, the payoff scale, or none of them?",
           f"The single largest source is not a main effect at all. model x pairing "
           f"takes {100*top_term.eta2:.1f}% of the variance in dyad cooperation, "
           f"more than model identity on its own ({100*mod_share:.1f}%), while the "
           f"language ({100*lang_share:.2f}%) and pairing ({100*pair_share:.2f}%) "
           "main effects are negligible because models respond to the persona "
           f"instruction in opposite directions. Payoff scale takes "
           f"{100*lam_share:.2f}% as a main effect, a tenth of model identity, plus "
           "about as much again through its interactions. Replicate-to-replicate "
           "sampling noise, the same prompt run 10 times, is "
           f"{100*resid_share:.1f}%, so the design factors still account for five "
           "sixths of the variance.",
           "ordered variance-share bars + factor involvement + stacked by order",
           "joint_coop and 4 further outcomes x model, language, pairing, lambda")

    # ======================================================================
    # 4. SIMPSON / SUBGROUP REVERSAL SEARCH
    # ======================================================================
    ag2 = ag.copy()
    ag2["opp_stratum"] = pd.cut(ag2.opp_coop_rate, [-.01, .01, .3, .7, .99, 1.01],
                                labels=["opp 0", "opp 0-0.3", "opp 0.3-0.7",
                                        "opp 0.7-1", "opp 1"])
    strata = ["opp 0", "opp 0-0.3", "opp 0.3-0.7", "opp 0.7-1", "opp 1"]
    rel_specs = [
        ("joint_coop ~ log10 lambda", dy, "joint_coop", "log_scale", "dyad"),
        ("joint_utility ~ log10 lambda", dy, "joint_utility", "log_scale", "dyad"),
        ("p_CC ~ log10 lambda", dy, "p_CC", "log_scale", "dyad"),
        ("utility_gap ~ log10 lambda", dy, "utility_gap", "log_scale", "dyad"),
        ("coop_rate ~ log10 lambda", ag2, "coop_rate", "log_scale", "agent_game"),
        ("utility ~ coop_rate", ag2, "utility", "coop_rate", "agent_game"),
        ("utility ~ first_coop", ag2, "utility", "first_coop", "agent_game"),
    ]
    grp_levels = {"model": C.MODEL_ORDER, "language": C.LANGS, "dyad": C.DYADS,
                  "scale": C.SCALES, "opp_stratum": strata}
    simp_rows = []
    for rel, src, yv, xv, lvl in rel_specs:
        pb, plo, phi, pp, pn = clustered_slope(src, yv, xv)
        grps = ["model", "language", "dyad", "scale"]
        if lvl == "agent_game":
            grps = grps + ["opp_stratum"]
        for g in grps:
            if g == "scale" and xv == "log_scale":
                continue
            for lv in grp_levels[g]:
                sub = src[src[g] == lv]
                b, lo_, hi_, p, n = clustered_slope(sub, yv, xv)
                if not np.isfinite(b):
                    continue
                rev = bool((np.sign(b) != np.sign(pb)) and (lo_ > 0 or hi_ < 0))
                simp_rows.append({
                    "relationship": rel, "level": lvl, "grouping": g,
                    "subgroup": str(lv), "pooled_slope": pb, "pooled_lo": plo,
                    "pooled_hi": phi, "subgroup_slope": b, "sub_lo": lo_,
                    "sub_hi": hi_, "p_value": p, "n": n,
                    "sign_reversal": rev,
                    "reversal_magnitude": float(abs(b - pb)) if rev else 0.0})
    simp = pd.DataFrame(simp_rows)
    simp["p_fdr"] = C.bh_fdr(simp["p_value"].to_numpy())
    simp["sign_reversal_fdr"] = simp.sign_reversal & (simp.p_fdr < .05)
    C.savetab(simp, "adv_simpson_search")
    n_rev = int(simp.sign_reversal_fdr.sum())
    n_search = len(simp)
    C.record_test(section="advanced",
                  test="systematic subgroup sign-reversal search "
                       "(cluster-robust slopes on the design cell)",
                  comparison="7 pooled relationships x 4-5 grouping variables",
                  statistic=n_rev, p_value=np.nan, n=n_search,
                  effect_size_note=f"{n_rev} of {n_search} subgroup slopes reverse "
                                   "the pooled sign with a CI excluding zero")

    r_pool = float(np.corrcoef(ag2.coop_rate, ag2.utility)[0, 1])
    r_str = {s: float(np.corrcoef(ag2.loc[ag2.opp_stratum == s, "coop_rate"],
                                  ag2.loc[ag2.opp_stratum == s, "utility"])[0, 1])
             for s in strata}
    r_mod = {m: float(np.corrcoef(ag2.loc[ag2.model == m, "coop_rate"],
                                  ag2.loc[ag2.model == m, "utility"])[0, 1])
             for m in C.MODEL_ORDER}
    r_dyad = float(np.corrcoef(dy.joint_coop, dy.joint_utility)[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    lam = simp[(simp.relationship == "joint_coop ~ log10 lambda")
               & (simp.grouping.isin(["model", "language", "dyad"]))
               ].reset_index(drop=True)
    yy = np.arange(len(lam))
    pooled = float(lam.pooled_slope.iloc[0])
    cols = [C.C_DEFECT if r else C.INK2 for r in lam.sign_reversal_fdr]
    axes[0].axvline(0, color=C.GRID, lw=.8)
    axes[0].axvline(pooled, color=C.C_COOP, lw=1.4, ls="--")
    for i in range(len(lam)):
        axes[0].plot([lam.sub_lo[i], lam.sub_hi[i]], [i, i], color=cols[i], lw=1.1)
    axes[0].scatter(lam.subgroup_slope, yy, s=22, color=cols, zorder=3)
    axes[0].set_yticks(yy, [f"{g[:4]}: {s}".replace("-Non-Reasoning", "")[:23]
                            for g, s in zip(lam.grouping, lam.subgroup)],
                       fontsize=6.2)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("slope of dyad cooperation per decade of $\\lambda$")
    axes[0].set_title("a  the pooled $\\lambda$ slope reverses")
    axes[0].text(pooled, -0.85, f"pooled {pooled:+.4f}", fontsize=6.4,
                 color=C.C_COOP, va="center", ha="center")
    axes[0].set_ylim(len(lam) - 0.4, -1.5)
    C.annotate(axes[0], f"red = reverses the pooled sign with\na CI excluding 0 and "
                        f"FDR p < 0.05\n({int(lam.sign_reversal_fdr.sum())} of "
                        f"{len(lam)} subgroups)", loc="lower right", fontsize=6.2)

    keys = (["pooled agent-games"] + strata
            + [m.replace("-Non-Reasoning", "") for m in C.MODEL_ORDER]
            + ["pooled dyads (joint)"])
    vals = ([r_pool] + [r_str[s] for s in strata]
            + [r_mod[m] for m in C.MODEL_ORDER] + [r_dyad])
    ccol = ([C.INK] + [C.C_DEFECT] * 5 + [C.MODEL_COL[m] for m in C.MODEL_ORDER]
            + [C.C_COOP])
    axes[1].barh(range(len(vals)), vals, color=ccol)
    axes[1].axvline(0, color=C.INK, lw=.8)
    axes[1].set_yticks(range(len(vals)), keys, fontsize=6.3)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("correlation between cooperation and own utility")
    axes[1].set_title("b  same variable pair, opposite signs")
    axes[1].set_xlim(-1.45, 1.45)
    for i, v_ in enumerate(vals):
        axes[1].text(v_ + (.04 if v_ >= 0 else -.04), i, f"{v_:+.2f}", va="center",
                     ha="left" if v_ >= 0 else "right", fontsize=6.0)
    C.annotate(axes[1], "red = within an opponent-behaviour stratum\n"
                        "coloured = within one model\n"
                        "blue = joint quantities at the dyad level",
               loc="lower left", fontsize=6.0)

    cnt = (simp.groupby(["relationship", "grouping"])["sign_reversal_fdr"].sum()
           .unstack(fill_value=0))
    im = axes[2].imshow(cnt.values, cmap=C.SEQ, aspect="auto", vmin=0,
                        vmax=max(1, cnt.values.max()))
    axes[2].set_xticks(range(cnt.shape[1]), cnt.columns, rotation=45, fontsize=6.3,
                       ha="right")
    axes[2].set_yticks(range(cnt.shape[0]),
                       [r.replace(" ~ ", "\n~ ") for r in cnt.index], fontsize=5.8)
    axes[2].set_title("c  reversals found, by search cell")
    axes[2].grid(False)
    for i in range(cnt.shape[0]):
        for j in range(cnt.shape[1]):
            axes[2].text(j, i, int(cnt.values[i, j]), ha="center", va="center",
                         fontsize=6.5,
                         color="white" if cnt.values[i, j] > cnt.values.max() * .55
                         else C.INK)
    plt.colorbar(im, ax=axes[2], fraction=.046, label="subgroups reversing sign")
    C.save(fig, "advanced", "A06_simpson_reversals",
           "Does any pooled relationship in this corpus reverse sign inside a "
           "subgroup, and if so how large is the reversal?",
           f"Yes, extensively: {n_rev} of {n_search} searched subgroup slopes "
           "reverse the pooled sign with a confidence interval excluding zero. The "
           f"pooled payoff-scale slope on cooperation is {pooled:+.4f} per decade, "
           "but it is negative and significant in three of six models, in French, "
           "and in the selfish-versus-selfish pairing; the pooled positive sign is "
           "carried by Grok-4.20 and GPT-5.4-Nano alone. The cooperation-utility "
           f"relationship is a level-of-analysis reversal: {r_pool:+.2f} pooled over "
           f"agent-games, {min(r_str.values()):+.2f} to {max(r_str.values()):+.2f} "
           f"inside opponent-behaviour strata, and {r_dyad:+.2f} for the joint "
           "quantities at the dyad level. Pooled effects here should not be read as "
           "properties of any individual model.",
           "subgroup slope forest + correlation reversal bars + reversal count map",
           "7 pooled relationships x 5 grouping variables")

    # ======================================================================
    # 5. CHANGEPOINT / THRESHOLD ON THE LAMBDA AXIS
    # ======================================================================
    knots = LOGS[1:-1]

    def seg_rss(x, y, k):
        X = np.column_stack([np.ones_like(x), x, np.maximum(x - k, 0.0)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        return float(res @ res), beta

    def profile(x, y):
        return np.array([seg_rss(x, y, k)[0] for k in knots])

    xall = dy["log_scale"].to_numpy()
    yall = dy["joint_coop"].to_numpy()
    prof = profile(xall, yall)
    best_i = int(prof.argmin())
    best_k = knots[best_i]
    rss_best, beta_best = seg_rss(xall, yall, best_k)

    rng = np.random.default_rng(0)
    n_boot = 400
    picks = np.empty(n_boot, int)
    for i in range(n_boot):
        rows = dyb[rng.integers(0, n_cell_d, size=n_cell_d)].ravel()
        picks[i] = int(profile(xall[rows], yall[rows]).argmin())
    share = np.bincount(picks, minlength=len(knots)) / n_boot
    lo_k, hi_k = np.percentile(10 ** knots[picks], [2.5, 97.5])
    modal_share = float(share.max())

    n = len(yall)

    def _aic(rss, k):
        return n * np.log(rss / n) + 2 * k

    rss_flat = float(((yall - yall.mean()) ** 2).sum())
    Xl = np.column_stack([np.ones_like(xall), xall])
    bl_, *_ = np.linalg.lstsq(Xl, yall, rcond=None)
    rss_lin = float(((yall - Xl @ bl_) ** 2).sum())
    step_rss = []
    for k in knots:
        Xs = np.column_stack([np.ones_like(xall), (xall > k).astype(float)])
        bsv, *_ = np.linalg.lstsq(Xs, yall, rcond=None)
        step_rss.append(float(((yall - Xs @ bsv) ** 2).sum()))
    best_step = int(np.argmin(step_rss))
    Xc = np.column_stack([(xall == v).astype(float) for v in LOGS])
    bc, *_ = np.linalg.lstsq(Xc, yall, rcond=None)
    rss_cat = float(((yall - Xc @ bc) ** 2).sum())
    forms = pd.DataFrame([
        {"form": "flat (invariance prediction)", "k": 1, "rss": rss_flat,
         "knot_lambda": np.nan},
        {"form": "linear in log lambda", "k": 2, "rss": rss_lin,
         "knot_lambda": np.nan},
        {"form": "step", "k": 3, "rss": step_rss[best_step],
         "knot_lambda": 10 ** knots[best_step]},
        {"form": "segmented (one knee)", "k": 4, "rss": rss_best,
         "knot_lambda": 10 ** best_k},
        {"form": "saturated categorical", "k": 10, "rss": rss_cat,
         "knot_lambda": np.nan}])
    forms["aic"] = [_aic(r.rss, r.k) for _, r in forms.iterrows()]
    forms["d_aic"] = forms.aic - forms.aic.min()
    d_aic_linear = float(forms.loc[forms.form == "linear in log lambda",
                                   "d_aic"].iloc[0])
    d_aic_flat = float(forms.loc[forms.form.str.startswith("flat"), "d_aic"].iloc[0])
    d_aic_step = float(forms.loc[forms.form == "step", "d_aic"].iloc[0])
    d_aic_seg = float(forms.loc[forms.form.str.startswith("segmented"),
                                "d_aic"].iloc[0])

    cp_rows = [{"scope": "all models", "best_knot_lambda": 10 ** best_k,
                "boot_lo_lambda": lo_k, "boot_hi_lambda": hi_k,
                "modal_boot_share": modal_share, "slope_left": beta_best[1],
                "slope_right": beta_best[1] + beta_best[2], "n": n}]
    for m in C.MODEL_ORDER:
        sub = dy[dy.model == m]
        xs, ys = sub.log_scale.to_numpy(), sub.joint_coop.to_numpy()
        bi = int(profile(xs, ys).argmin())
        _, bb = seg_rss(xs, ys, knots[bi])
        blk, nc = cell_blocks(sub)
        pk = np.empty(200, int)
        for i in range(200):
            rws = blk[rng.integers(0, nc, size=nc)].ravel()
            pk[i] = int(profile(xs[rws], ys[rws]).argmin())
        klo, khi = np.percentile(10 ** knots[pk], [2.5, 97.5])
        cp_rows.append({
            "scope": m, "best_knot_lambda": 10 ** knots[bi],
            "boot_lo_lambda": klo, "boot_hi_lambda": khi,
            "modal_boot_share": float(np.bincount(pk, minlength=len(knots)).max() / 200),
            "slope_left": bb[1], "slope_right": bb[1] + bb[2], "n": len(sub)})
    cp_models = pd.DataFrame(cp_rows).iloc[1:]
    cp = pd.concat([pd.DataFrame(cp_rows), forms.assign(scope="functional form")],
                   ignore_index=True)
    C.savetab(cp, "adv_changepoint")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7))
    mu, lo2, hi2 = [], [], []
    for s in C.SCALES:
        mk = (dy.scale.to_numpy() == s)
        a_, b_, c_ = boot_mean_ci(np.where(mk, yall, np.nan), dyb, n_boot=500)
        mu.append(a_); lo2.append(b_); hi2.append(c_)
    axes[0].fill_between(C.SCALES, lo2, hi2, color=C.C_COOP, alpha=.2, lw=0)
    axes[0].plot(C.SCALES, mu, "o", color=C.C_COOP, zorder=4)
    xg = np.linspace(LOGS.min(), LOGS.max(), 200)
    yg = beta_best[0] + beta_best[1] * xg + beta_best[2] * np.maximum(xg - best_k, 0)
    axes[0].plot(10 ** xg, yg, color=C.C_DEFECT, lw=1.5)
    axes[0].axvline(10 ** best_k, color=C.INK, lw=.9, ls="--")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("payoff scale $\\lambda$")
    axes[0].set_ylabel("dyad cooperation rate")
    axes[0].set_title("a  segmented fit and its knee")
    axes[0].set_ylim(min(lo2) - 0.055, max(hi2) + 0.012)
    C.annotate(axes[0], f"knee at $\\lambda$ = {10**best_k:g}\n"
                        f"left slope {beta_best[1]:+.3f} / decade\n"
                        f"right slope {beta_best[1]+beta_best[2]:+.3f} / decade",
               loc="lower right", fontsize=6.4)

    axes[1].plot(10 ** knots, prof, "o-", color=C.OUTCOME_COL["DC"],
                 label="segmented (hinge)")
    axes[1].plot(10 ** knots, step_rss, "s-", color=C.OUTCOME_COL["CD"],
                 label="step (level shift)")
    axes[1].axvline(10 ** best_k, color=C.INK, lw=.9, ls="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("candidate knot $\\lambda$")
    axes[1].set_ylabel("residual sum of squares")
    axes[1].set_title("b  the profile is shallow")
    axes[1].legend(loc="upper left", fontsize=6.2)
    rel = (prof.max() - prof.min()) / prof.min()
    C.annotate(axes[1], f"hinge RSS varies by only {100*rel:.2f}%\n"
                        "across the eight candidate knots; a\n"
                        f"step at $\\lambda$ = {10**knots[best_step]:g} fits better still",
               loc="lower right", fontsize=6.2)

    axes[2].bar(range(len(knots)), 100 * share, color=C.C_COOP, width=.75)
    axes[2].set_xticks(range(len(knots)), [f"{10**k:g}" for k in knots],
                       rotation=90, fontsize=6.5)
    axes[2].set_xlabel("selected knot $\\lambda$")
    axes[2].set_ylabel("% of cell-bootstrap replicates")
    axes[2].set_title("c  how well identified is the knee")
    axes[2].set_ylim(0, max(100 * share.max() * 1.45, 20))
    C.annotate(axes[2], f"modal knot {10**best_k:g} wins {100*modal_share:.0f}% "
                        f"of replicates\n95% interval "
                        f"[{lo_k:g}, {hi_k:g}]", loc="upper right", fontsize=6.4)
    C.save(fig, "advanced", "A07_changepoint_lambda",
           "Where on the payoff-scale ladder does the departure from invariance "
           "turn on, and is that location identified well enough to quote?",
           f"A knee exists but its location is poorly determined, and a hinge is "
           f"not even the best two-piece description. The best hinge sits at "
           f"lambda = {10**best_k:g}, with cooperation rising {beta_best[1]:+.3f} "
           f"per decade below it and {beta_best[1]+beta_best[2]:+.3f} per decade "
           f"above, but it wins only {100*modal_share:.0f}% of cell-bootstrap "
           f"replicates, its 95% interval spans [{lo_k:g}, {hi_k:g}], and the RSS "
           f"profile varies by just {100*rel:.2f}% across all eight candidates. A "
           f"simple step down below lambda = {10**knots[best_step]:g} fits better "
           f"than the hinge (delta AIC {d_aic_step:.0f} against "
           f"{d_aic_seg:.0f}), and the saturated categorical form beats both, so the "
           "lambda response is a level shift at the bottom of the ladder plus "
           "structure that no two-piece form captures. Both still beat the flat "
           f"invariance prediction by delta AIC {d_aic_flat:.0f}. Per-model knees "
           f"scatter from {cp_models.best_knot_lambda.min():g} to "
           f"{cp_models.best_knot_lambda.max():g}.",
           "segmented fit + RSS profile + bootstrap knot histogram",
           "joint_coop, log10 lambda, cell-level bootstrap")

    # ======================================================================
    # 6. HETEROSKEDASTICITY
    # ======================================================================
    # The right residual here is the one from the SATURATED design-cell model, not
    # from the additive model.  An additive fit leaves the model x pairing
    # interaction - the single largest term in the decomposition - inside the
    # residual, so a Breusch-Pagan test on it would be detecting unmodelled mean
    # structure and reporting it as heteroskedasticity.  Residualising on the cell
    # mean instead leaves exactly the replicate-to-replicate sampling noise of the
    # same prompt run 10 times, which is the quantity worth testing.
    keys = ["model", "language", "dyad", "scale"]
    mu_cell = dy.groupby(keys, observed=True)["joint_coop"].transform("mean").to_numpy()
    resid = dy["joint_coop"].to_numpy() - mu_cell

    Xdes = pd.get_dummies(dy[["model", "language", "dyad"]],
                          drop_first=True).astype(float)
    Xdes["log_scale"] = dy["log_scale"].to_numpy()
    bp = het_breuschpagan(resid, sm.add_constant(Xdes.to_numpy()))
    ck = het_breuschpagan(resid, sm.add_constant(np.column_stack([mu_cell,
                                                                 mu_cell ** 2])))
    lev_lam = stats.levene(*[resid[dy.scale.to_numpy() == s] for s in C.SCALES])
    lev_mod = stats.levene(*[resid[dy.model.to_numpy() == m] for m in C.MODEL_ORDER])
    lev_pair = stats.levene(*[resid[dy.dyad.to_numpy() == x] for x in C.DYADS])
    sd_lam = pd.Series({s: resid[dy.scale.to_numpy() == s].std() for s in C.SCALES})
    sd_mod = pd.Series({m: resid[dy.model.to_numpy() == m].std()
                        for m in C.MODEL_ORDER})
    sd_pair = pd.Series({x: resid[dy.dyad.to_numpy() == x].std() for x in C.DYADS})
    rat_lam = float(sd_lam.max() / sd_lam.min())
    rat_mod = float(sd_mod.max() / sd_mod.min())
    rat_pair = float(sd_pair.max() / sd_pair.min())

    # overdispersion against 20 independent binary decisions per dyad
    cellg = dy.groupby(keys, observed=True)["joint_coop"]
    disp = float(cellg.var(ddof=1).mean()
                 / (cellg.mean() * (1 - cellg.mean()) / 20.0).mean())

    het_tab = pd.DataFrame([
        {"test": "Breusch-Pagan of replicate residuals on the design matrix",
         "statistic": bp[0], "df": Xdes.shape[1], "p_value": bp[1],
         "residual_sd_ratio_max_min": np.nan},
        {"test": "Koenker / Cook-Weisberg on the cell mean and its square",
         "statistic": ck[0], "df": 2, "p_value": ck[1],
         "residual_sd_ratio_max_min": np.nan},
        {"test": "Levene on replicate residuals across lambda",
         "statistic": lev_lam.statistic, "df": 9, "p_value": lev_lam.pvalue,
         "residual_sd_ratio_max_min": rat_lam},
        {"test": "Levene on replicate residuals across model",
         "statistic": lev_mod.statistic, "df": 5, "p_value": lev_mod.pvalue,
         "residual_sd_ratio_max_min": rat_mod},
        {"test": "Levene on replicate residuals across pairing",
         "statistic": lev_pair.statistic, "df": 3, "p_value": lev_pair.pvalue,
         "residual_sd_ratio_max_min": rat_pair},
        {"test": "overdispersion vs 20 independent binary decisions "
                 "(variance ratio, not a test)",
         "statistic": disp, "df": np.nan, "p_value": np.nan,
         "residual_sd_ratio_max_min": np.nan}])
    het_tab["p_fdr"] = C.bh_fdr(het_tab["p_value"].to_numpy())
    C.savetab(het_tab, "adv_heteroskedasticity")
    for _, r in het_tab.iterrows():
        C.record_test(section="advanced", test=r.test,
                      comparison="replicate-level residual variance of dyad "
                                 "cooperation (residualised on the design cell)",
                      statistic=r.statistic, p_value=r.p_value, n=len(dy),
                      effect_size_note="residual SD ratio max/min = "
                                       f"{r.residual_sd_ratio_max_min}")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6))
    axes[0].scatter(mu_cell, resid, s=2.5, alpha=.14, color=C.INK2, rasterized=True)
    bins = np.unique(np.quantile(mu_cell, np.linspace(0, 1, 13)))
    bi = np.clip(np.digitize(mu_cell, bins[1:-1]), 0, len(bins) - 2)
    ctr = np.array([mu_cell[bi == i].mean() for i in range(len(bins) - 1)])
    sdb = np.array([resid[bi == i].std() for i in range(len(bins) - 1)])
    axes[0].plot(ctr, sdb, "o-", color=C.C_DEFECT, lw=1.5, ms=3.5,
                 label="binned residual SD")
    axes[0].plot(ctr, -sdb, "o-", color=C.C_DEFECT, lw=1.5, ms=3.5)
    mgrid = np.linspace(0.005, 0.995, 80)
    axes[0].plot(mgrid, np.sqrt(mgrid * (1 - mgrid) / 20), color=C.OUTCOME_COL["CD"],
                 lw=1.2, ls="--", label="independent-Bernoulli SD")
    axes[0].plot(mgrid, -np.sqrt(mgrid * (1 - mgrid) / 20),
                 color=C.OUTCOME_COL["CD"], lw=1.2, ls="--")
    axes[0].set_xlabel("design-cell mean cooperation rate")
    axes[0].set_ylabel("replicate residual")
    axes[0].set_title("a  spread vs the cell mean")
    axes[0].legend(loc="upper left", fontsize=6.0)
    axes[0].set_ylim(-0.62, 0.78)
    C.annotate(axes[0], f"Koenker on the cell mean: p = {ck[1]:.1g}\n"
                        f"variance is {disp:.2f}x the independent-\n"
                        "Bernoulli benchmark",
               loc="lower right", fontsize=6.3)

    axes[1].plot(C.SCALES, sd_lam.reindex(C.SCALES).to_numpy(), "o-", color=C.C_COOP)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("payoff scale $\\lambda$")
    axes[1].set_ylabel("SD of replicate residual")
    axes[1].set_title("b  $\\lambda$ barely moves the spread")
    axes[1].set_ylim(0, sd_lam.max() * 1.5)
    C.annotate(axes[1], f"Levene p = {lev_lam.pvalue:.2g} but\n"
                        f"max/min SD is only {rat_lam:.2f}",
               loc="lower right", fontsize=6.4)

    axes[2].bar(range(6), sd_mod.reindex(C.MODEL_ORDER).to_numpy(),
                color=[C.MODEL_COL[m] for m in C.MODEL_ORDER], width=.7)
    axes[2].set_xticks(range(6), [m.replace("-Non-Reasoning", "").replace("-", "\n", 1)
                                  for m in C.MODEL_ORDER], fontsize=5.6)
    axes[2].set_ylabel("SD of replicate residual")
    axes[2].set_title("c  the model decides reproducibility")
    axes[2].set_ylim(0, sd_mod.max() * 1.45)
    C.annotate(axes[2], f"Levene p = {lev_mod.pvalue:.1g}\n"
                        f"max/min SD = {rat_mod:.2f}, i.e.\n"
                        f"{rat_mod**2:.0f}x in variance",
               loc="upper left", fontsize=6.4)
    C.save(fig, "advanced", "A08_heteroskedasticity",
           "Is the replicate-to-replicate noise in cooperation constant, or does it "
           "depend on the payoff scale, on the model, or on the level of cooperation "
           "itself?",
           f"It depends overwhelmingly on the model and hardly at all on lambda. "
           f"Residualising on the design cell leaves pure replicate noise, whose SD "
           f"varies by only {rat_lam:.2f}x across the ten payoff scales and "
           f"{rat_pair:.2f}x across the four pairings but by {rat_mod:.2f}x across "
           f"the six models, which is {rat_mod**2:.0f}x in variance: the same prompt "
           f"run 10 times gives SD {sd_mod.min():.3f} for the most deterministic "
           f"model and {sd_mod.max():.3f} for the least. Every Levene test rejects, "
           "which at n = 12,000 is expected and carries no information, while the "
           "ratios do. Dispersion also rises with distance from the cooperation "
           f"boundaries and sits {disp:.2f} times above the variance of 20 "
           "independent binary decisions, so within-game actions are correlated but "
           "replicate noise is only mildly overdispersed.",
           "residual vs cell mean with Bernoulli envelope + SD by lambda + SD by model",
           "replicate residuals of joint_coop after removing the design-cell mean")

    # ======================================================================
    # 7. SURPRISE: the persona instruction is inverted in half the models
    # ======================================================================
    pers_rows = []
    for m in C.MODEL_ORDER:
        sub = dy[dy.model == m]
        blk, _ = cell_blocks(sub)
        vals = sub.joint_coop.to_numpy()
        ma = (sub.dyad.to_numpy() == "CvC")
        mb = (sub.dyad.to_numpy() == "SvS")
        o, l_, h_ = boot_contrast_ci(vals, ma, mb, blk, n_boot=600, seed=7)
        pers_rows.append({"model": m, "coop_CvC": float(vals[ma].mean()),
                          "coop_SvS": float(vals[mb].mean()),
                          "persona_effect": o, "lo": l_, "hi": h_,
                          "inverted": bool(h_ < 0), "n": len(sub)})
    pers = pd.DataFrame(pers_rows)
    C.savetab(pers, "adv_persona_effect")
    n_inv = int(pers.inverted.sum())
    n_comply = int((pers.lo > 0).sum())
    n_null = int(len(pers) - n_inv - n_comply)
    qw = pers.loc[pers.persona_effect.idxmin()]
    best_pers = pers.loc[pers.persona_effect.idxmax()]
    for _, r in pers.iterrows():
        C.record_test(section="advanced",
                      test="cell-clustered bootstrap contrast CvC minus SvS",
                      comparison=f"persona effect, {r.model}",
                      statistic=r.persona_effect, p_value=np.nan,
                      ci_lo=r.lo, ci_hi=r.hi, n=r.n,
                      effect_size_note="difference in dyad cooperation rate")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    ordp = pers.sort_values("persona_effect").reset_index(drop=True)
    yy = np.arange(len(ordp))
    axes[0].axvline(0, color=C.INK, lw=.9)
    for i, r in ordp.iterrows():
        axes[0].plot([r.lo, r.hi], [i, i], color=C.INK2, lw=1.2)
        axes[0].scatter(r.persona_effect, i, s=54, color=C.MODEL_COL[r.model],
                        zorder=4)
    axes[0].set_yticks(yy, [m.replace("-Non-Reasoning", "") for m in ordp.model],
                       fontsize=6.8)
    axes[0].set_xlim(-0.78, 0.78)
    axes[0].set_ylim(-0.6, len(ordp) - 0.15)
    axes[0].set_xlabel("cooperation(both cooperative) - cooperation(both selfish)")
    axes[0].set_title("a  persona compliance, signed")
    C.annotate(axes[0], f"{n_inv} of 6 models have a CI entirely below zero:\n"
                        "two agents told to be selfish cooperate MORE\n"
                        f"than two told to cooperate. {n_comply} comply, "
                        f"{n_null} ignores\nthe persona (CI covers zero).",
               loc="lower right", fontsize=6.0)

    for dd, col in [("CvC", C.C_COOP), ("SvS", C.C_DEFECT)]:
        v = dy[(dy.model == qw.model) & (dy.dyad == dd)]["joint_coop"]
        axes[1].hist(v, bins=np.arange(-.025, 1.03, .05), alpha=.7, color=col,
                     label=f"{dd}, mean {v.mean():.3f}")
    axes[1].set_xlabel("dyad cooperation rate")
    axes[1].set_ylabel("dyads")
    axes[1].set_title(f"b  {qw.model}: the extreme case")
    axes[1].legend(loc="upper center", fontsize=6.5)
    C.annotate(axes[1], f"persona effect {qw.persona_effect:+.3f}\n"
                        f"95% CI [{qw.lo:+.3f}, {qw.hi:+.3f}]",
               loc="upper right", fontsize=6.4)

    for m in C.MODEL_ORDER:
        sub = dy[dy.model == m]
        a_ = sub[sub.dyad == "CvC"].groupby("scale", observed=True)["joint_coop"].mean()
        b_ = sub[sub.dyad == "SvS"].groupby("scale", observed=True)["joint_coop"].mean()
        axes[2].plot(C.SCALES, (a_ - b_).reindex(C.SCALES).to_numpy(), "o-",
                     color=C.MODEL_COL[m], ms=3.2, lw=1.2)
    axes[2].axhline(0, color=C.INK, lw=.9)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("payoff scale $\\lambda$")
    axes[2].set_ylabel("persona effect (CvC - SvS)")
    axes[2].set_title("c  the sign is stable across $\\lambda$")
    C.save(fig, "advanced", "A09_persona_inversion",
           "Do the models actually do what the personality instruction tells them "
           "to do?",
           f"Two of the six do the opposite and one ignores the instruction. "
           f"Telling both agents to be cooperative rather than selfish moves dyad "
           f"cooperation by {best_pers.persona_effect:+.3f} in {best_pers.model} but "
           f"by {qw.persona_effect:+.3f} in {qw.model}, whose two cooperative agents "
           f"cooperate {qw.coop_CvC:.3f} of the time against {qw.coop_SvS:.3f} for "
           f"two selfish ones. {n_inv} of 6 models have a bootstrap CI entirely "
           f"below zero, {n_comply} entirely above, and {n_null} covers zero. "
           "Because the inverted and the compliant models partly cancel, the pooled "
           f"pairing main effect is {100*pair_share:.2f}% of variance while model x "
           "pairing is the largest term in the whole decomposition. Each model's "
           "persona sign is stable across the entire payoff ladder, so this is not a "
           "payoff-scale artefact.",
           "persona-effect forest + distribution + persona effect vs lambda",
           "joint_coop x model x pairing x lambda")

    # ======================================================================
    # findings
    # ======================================================================
    C.record_finding(
        id="A-variance-model-x-persona",
        finding="What drives cooperation is not model identity or persona alone but "
                "their interaction: model x pairing is the single largest variance "
                "component in the corpus.",
        evidence=f"An exactly orthogonal ANOVA decomposition of dyad cooperation, "
                 f"valid without Type I/II/III ambiguity because the 6 x 5 x 4 x 10 "
                 f"grid is perfectly balanced with 10 replicates, gives model x "
                 f"pairing {100*top_term.eta2:.1f}% of total variance against "
                 f"{100*mod_share:.1f}% for model identity, {100*lam_share:.2f}% for "
                 f"payoff scale, {100*lang_share:.2f}% for language and "
                 f"{100*pair_share:.2f}% for pairing. Replicate-level sampling noise "
                 f"is {100*resid_share:.1f}%. All 15 design terms plus the residual "
                 "sum to 100% by construction.",
        figure="10_advanced/A05_variance_decomposition.png, "
               "10_advanced/A04_interaction_plots.png",
        strength="strong",
        robustness="the same ordering holds for joint_utility, p_CC, utility_gap and "
                   "n_switch; the df-corrected omega2 changes every large share by "
                   "under one percentage point",
        interpretation="Correlational language does not apply cleanly here because "
                       "the factors were experimentally manipulated, so these are "
                       "causal shares of the manipulated variance - but only over "
                       "the levels actually run. The reading is that persona prompts "
                       "have no transferable effect across models: each model has "
                       "its own mapping from persona text to behaviour.",
        caveat="Shares are specific to the six models, five languages, four pairings "
               "and ten scales sampled. A different model set would change every "
               "number, and self-play means nothing here transfers to cross-model "
               "play.",
        claim="In a balanced factorial of frontier LLMs playing the prisoner's "
              "dilemma, the interaction between model and assigned personality "
              "explains more variance in cooperation (25%) than the model, "
              "language, personality and payoff-scale main effects combined (20%).")

    C.record_finding(
        id="A-persona-inversion",
        finding="A third of the models respond to the personality instruction with "
                "the wrong sign: two agents told to be selfish cooperate more than "
                "two told to be cooperative.",
        evidence="Persona effect, cooperation(CvC) minus cooperation(SvS), with "
                 f"cell-clustered bootstrap CIs: {best_pers.persona_effect:+.3f} "
                 f"[{best_pers.lo:+.3f}, {best_pers.hi:+.3f}] for "
                 f"{best_pers.model} and {qw.persona_effect:+.3f} "
                 f"[{qw.lo:+.3f}, {qw.hi:+.3f}] for {qw.model}, whose "
                 f"cooperative-cooperative dyads cooperate {qw.coop_CvC:.3f} against "
                 f"{qw.coop_SvS:.3f} for selfish-selfish. {n_inv} of 6 models have "
                 f"a 95% CI entirely below zero, {n_comply} entirely above, and "
                 f"{n_null} covers zero, i.e. ignores the persona text altogether.",
        figure="10_advanced/A09_persona_inversion.png",
        strength="strong",
        robustness="the sign holds at all ten payoff scales for every model and is "
                   "not carried by any single language; CIs resample whole design "
                   "cells",
        interpretation="Causal within the experiment, since persona text is the "
                       "manipulated variable, but the mechanism is not identified. "
                       "An inverted persona effect is consistent with the model "
                       "reading 'selfish' as licence to defect only when it also "
                       "expects the opponent to defect, and with translation "
                       "effects, and this design cannot separate them.",
        caveat="Self-play only, so this says nothing about how an inverted-persona "
               "model behaves against a differently-prompted opponent. The Arabic "
               "and Chinese templates also carry an inherited FAIRGAME wording quirk "
               "- a closing 'maximise your rewards' sentence against 'penalty' "
               "everywhere above it - so part of any language-linked variance is "
               "prompt inconsistency rather than culture.",
        claim="Personality prompts do not transfer across frontier LLMs: in two of "
              "six models the cooperative persona produces significantly less "
              "cooperation than the selfish persona, by up to 0.62 in dyad "
              "cooperation rate, and a third model ignores the persona entirely.")

    C.record_finding(
        id="A-simpson-lambda",
        finding="The pooled payoff-scale effect on cooperation is a Simpson artefact "
                "of aggregating models that respond in opposite directions.",
        evidence=f"Pooled slope {pooled:+.4f} per decade of lambda, 95% CI "
                 f"[{lam.pooled_lo.iloc[0]:+.4f}, {lam.pooled_hi.iloc[0]:+.4f}]. "
                 "Within models the slopes run from "
                 f"{lam[lam.grouping=='model'].subgroup_slope.min():+.4f} to "
                 f"{lam[lam.grouping=='model'].subgroup_slope.max():+.4f}; three of "
                 "six are negative with CIs excluding zero, and the pooled positive "
                 "sign is carried by Grok-4.20 and GPT-5.4-Nano alone. The reversal "
                 "also appears in French and in the selfish-versus-selfish pairing. "
                 f"Over the whole search, {n_rev} of {n_search} subgroup slopes "
                 "reverse the pooled sign with a CI excluding zero.",
        figure="10_advanced/A06_simpson_reversals.png, "
               "10_advanced/A03_interaction_effects.png",
        strength="strong",
        robustness="the design is perfectly balanced, so this cannot be a "
                   "composition confound: every subgroup contains the identical "
                   "lambda ladder with the same number of games at every rung",
        interpretation="Causal for the manipulation but not generalisable. Payoff "
                       "magnitude changes behaviour in every model, but there is no "
                       "single direction to report, so 'LLMs cooperate more when the "
                       "stakes are larger' is not a supportable statement from this "
                       "corpus.",
        caveat="Model-level slopes are themselves averages over language and persona, "
               "which interact with lambda in turn; the honest unit for this effect "
               "is the model x pairing cell, not the model.",
        claim="The direction of the payoff-scale effect on LLM cooperation is "
              "model-specific: the pooled positive slope reverses sign in three of "
              "six models, so pooled scale effects must not be reported as a "
              "property of frontier LLMs in general.")

    C.record_finding(
        id="A-coop-utility-level-reversal",
        finding="Whether cooperation pays depends entirely on the level of "
                "aggregation, and the pooled agent-level correlation is close to "
                "meaningless.",
        evidence=f"Correlation between cooperation and own utility: {r_pool:+.3f} "
                 f"pooled over 24,000 agent-games, {min(r_str.values()):+.3f} to "
                 f"{max(r_str.values()):+.3f} inside strata of opponent cooperation, "
                 f"{min(r_mod.values()):+.3f} to {max(r_mod.values()):+.3f} inside "
                 f"individual models, and {r_dyad:+.3f} for the joint quantities at "
                 "the dyad level.",
        figure="10_advanced/A06_simpson_reversals.png, "
               "10_advanced/A01_correlation_structure.png",
        strength="strong",
        robustness="the within-stratum negative sign is forced by the payoff matrix "
                   "and appears in all five opponent strata; the by-model sign split "
                   "is a genuine aggregation effect of self-play",
        interpretation="This is the social dilemma made visible in a correlation "
                       "table rather than a new behavioural fact. Holding the "
                       "opponent fixed, defection is dominant, so cooperation must "
                       "correlate negatively with own utility; aggregating over "
                       "self-play pairs mixes in the fact that cooperative models "
                       "meet cooperative opponents.",
        caveat="Self-play means the opponent is the same model, so the by-model "
               "correlation confounds own policy with opponent policy by "
               "construction, and nothing here identifies what an agent would earn "
               "against a different opponent.",
        claim="Pooled correlations between cooperation and payoff in LLM game-play "
              "corpora are uninterpretable without conditioning on opponent "
              "behaviour: the sign runs from about -1 within opponent strata to "
              "near zero pooled.")

    C.record_finding(
        id="A-correlations-mechanical",
        finding="Nearly every strong correlation in this corpus is an algebraic "
                "identity imposed by the payoff matrix, not an empirical result.",
        evidence=f"Of the 20 largest absolute Spearman correlations across "
                 f"{len(cor)} variable pairs at two levels, {n_top20_mech} are "
                 "mechanical: joint_utility = 0.4 + 0.4 joint_coop - 0.1 p_exploit "
                 f"holds to {id_dy:.0e}, efficiency = 1.25 x utility to {id_ag:.0e}, "
                 f"utility_gap = |p_DC - p_CD| to {id_gap:.0e}, fairness = 1 - gini "
                 "exactly, and the four outcome shares sum to one. The strongest "
                 "genuinely empirical pair is "
                 f"{lab(strongest_emp.var_a)} vs {lab(strongest_emp.var_b)} at "
                 f"rho = {strongest_emp.spearman_rho:+.2f}.",
        figure="10_advanced/A01_correlation_structure.png, "
               "10_advanced/A02_nonmechanical_relationships.png",
        strength="strong",
        robustness="Pearson and Spearman agree in sign on every pair, and the "
                   "identities were verified numerically rather than assumed",
        interpretation="Purely descriptive and definitional. The practical "
                       "consequence is that an automated correlation screen on a "
                       "game-theory corpus of this shape will report identities as "
                       "discoveries unless the derived-variable graph is declared "
                       "first.",
        caveat="The flagging is a curated rule set plus a |r| > 0.999 backstop, so "
               "partially mechanical pairs such as an average and its own "
               "sub-average are flagged even though they still carry some empirical "
               "content.",
        claim="Correlation mining on iterated-game corpora is dominated by "
              f"definitional structure: {n_top20_mech} of the 20 strongest "
              "correlations here are forced by the payoff matrix or by variable "
              "construction rather than by behaviour.")

    C.record_finding(
        id="A-replicate-reproducibility",
        finding="Reproducibility is a model property, not a task property: the same "
                "prompt run ten times gives replicate noise that differs by a factor "
                "of five in SD between models.",
        evidence="Residualising dyad cooperation on the design-cell mean isolates "
                 f"replicate noise. Its SD is {sd_mod.min():.3f} for the most "
                 f"deterministic model and {sd_mod.max():.3f} for the least, a ratio "
                 f"of {rat_mod:.2f} in SD and {rat_mod**2:.0f} in variance, against "
                 f"only {rat_lam:.2f} across the ten payoff scales and "
                 f"{rat_pair:.2f} across the four pairings. Pooled, this noise is "
                 f"{disp:.2f} times the variance of 20 independent binary decisions "
                 f"and accounts for {100*resid_share:.1f}% of total variance.",
        figure="10_advanced/A08_heteroskedasticity.png",
        strength="moderate",
        robustness="the ordering of models by replicate SD is unchanged when the "
                   "residual is taken within language or within pairing; Levene "
                   "rejects for every factor but the SD ratios separate them "
                   "cleanly",
        interpretation="Descriptive. Replicate SD confounds decoding temperature, "
                       "which was not controlled or recorded per model, with genuine "
                       "policy stochasticity, so this measures how reproducible each "
                       "model was as served rather than an intrinsic property of the "
                       "model.",
        caveat="Sampling settings were not varied, so nothing here separates "
               "temperature from policy entropy. Models whose behaviour is pinned to "
               "a boundary, such as always defecting, have low replicate SD for a "
               "purely arithmetic reason.",
        claim="Replicate-level reproducibility of LLM game play varies by a factor "
              "of five in standard deviation across frontier models, so a "
              "single-run benchmark is far less stable for some models than for "
              "others.")

    print(f"  correlations: {len(cor)} pairs, {int(cor.mechanical.sum())} mechanical")
    print(f"  simpson search: {n_rev} sign reversals out of {n_search} subgroup slopes")
    return cor, inter, vd_all, simp, cp
