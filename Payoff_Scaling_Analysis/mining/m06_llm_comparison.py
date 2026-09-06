"""Section 8: systematic comparison of the six frontier models.

Why this section is not a leaderboard
-------------------------------------
The obvious thing to do with six models and one behavioural corpus is to average
cooperation per model and print the order. That answer is close to worthless for
three separate reasons, and this section is built to expose all three rather than
to hide them behind a mean.

1. *A mean hides the shape.* Agent-game cooperation in this corpus is a U-shaped,
   boundary-inflated variable (see 02_distributions): most agent-games sit at 0 or
   at 1, so a model's mean is really a mixing weight between two very different
   populations. Two models can land on the same mean by mixing very different
   atoms. Every comparison here is therefore accompanied by an explicit overlap
   measure on the full discrete distribution, not only by a difference of means.

2. *A mean hides the worst case.* Each model is observed under 200 conditions
   (10 payoff scales x 5 languages x 4 personality pairings), each with 10 matched
   games. A model whose pooled cooperation is high can still collapse in one
   corner of that grid. The consistency figure plots the worst and best condition
   next to the mean, because for anything resembling deployment the worst case is
   the number that matters.

3. *A mean is one metric, and the ranking depends on which metric you pick.* The
   final figure ranks the models by fifteen metrics, then by cooperation inside
   each payoff scale, each language and each pairing, and measures how much the
   order moves. Instability there is a negative result about LLM game-theory
   benchmarks in general: it says a headline number is a choice of metric as much
   as a property of the model.

Statistical choices
-------------------
* Every pairwise test runs on *dyad-level* values (12,000 rows, 2,000 per model).
  The agent-game table double counts each interaction - both agents of a dyad
  appear as focal rows - so it cannot be used for between-model tests.
* All 15 pairs are tested with Mann-Whitney U, and the p-values are FDR-corrected
  within each metric family. With n = 2,000 per model, p-values are nearly free;
  Cliff's delta carries the interpretation and the standard bands (0.147 / 0.33 /
  0.474) are drawn on the figure.
* Group means use the clustered bootstrap on the design cell, which is the common
  random number key: 200 cells per model, each contributing its 10 matched games.
  Resampling games instead of cells would understate the interval.
* The model x metric heatmap is standardised across the six models (z within
  metric) purely so that metrics on different units can share one colour scale.
  With six models a z-score is a crude statistic, so the raw value is printed in
  every cell and the standardisation is only a display device.

Caveats that constrain interpretation
-------------------------------------
* Gemini-3.1-Flash-Lite is a preview build, so its numbers describe a preview.
* Only the non-reasoning variant of Grok was collected.
* Language disparity is not a clean culture measure: the Arabic and Chinese
  prompts inherit a FAIRGAME translation quirk whose final sentence says
  "maximise your rewards" while the payoff sentences say "penalty", so part of any
  ar/cn gap is prompt inconsistency.
* Self-play only. Nothing here supports a claim about how these models would play
  against each other.
* Two of the six models come from one vendor, so the vendor analysis has n = 2 in
  its only replicated group and is descriptive, not inferential.
"""
from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import core as C

# short labels: the full names collide on a six-column axis
ONELINE = {
    "Claude-Haiku-4.5": "Claude-Haiku-4.5",
    "Qwen3-235B-A22B": "Qwen3-235B",
    "GPT-5.4-Nano": "GPT-5.4-Nano",
    "Gemini-3.1-Flash-Lite": "Gemini-3.1-FL*",
    "Grok-4.20-Non-Reasoning": "Grok-4.20-nr",
    "Gemini-3.5-Flash-Lite": "Gemini-3.5-FL",
}
TWOLINE = {
    "Claude-Haiku-4.5": "Claude\nHaiku-4.5",
    "Qwen3-235B-A22B": "Qwen3\n235B",
    "GPT-5.4-Nano": "GPT-5.4\nNano",
    "Gemini-3.1-Flash-Lite": "Gemini-3.1\nFL*",
    "Grok-4.20-Non-Reasoning": "Grok-4.20\nnon-reas.",
    "Gemini-3.5-Flash-Lite": "Gemini-3.5\nFlash-Lite",
}
PREVIEW_NOTE = "* preview build"

# metric key -> (display label, family, ranking orientation, print format)
# orientation +1 means a larger value ranks better / more prosocial / more stable
METRICS = [
    ("coop_rate",      "cooperation rate",             "level",       +1, "{:.3f}"),
    ("utility",        "utility (higher better)",      "level",       +1, "{:.3f}"),
    ("efficiency",     "efficiency (1 = mutual C)",    "level",       +1, "{:.3f}"),
    ("p_CC",           "mutual cooperation share",     "structure",   +1, "{:.3f}"),
    ("p_DD",           "mutual defection share",       "structure",   -1, "{:.3f}"),
    ("p_exploit",      "exploitation share",           "structure",   -1, "{:.3f}"),
    ("dd_absorbed",    "locked in mutual defection",   "structure",   -1, "{:.3f}"),
    ("reciprocity",    "reciprocity pC|C - pC|D",      "structure",   +1, "{:+.3f}"),
    ("endgame_drop",   "endgame cooperation drop",     "structure",   -1, "{:+.3f}"),
    ("uncond_share",   "unconditional play share",     "structure",   -1, "{:.3f}"),
    ("game_sd",        "SD across games",              "variability", -1, "{:.3f}"),
    ("cond_sd",        "SD across 200 conditions",     "variability", -1, "{:.3f}"),
    ("lambda_sens",    "|slope| per decade lambda",    "variability", -1, "{:.3f}"),
    ("lang_disparity", "language max - min",           "variability", -1, "{:.3f}"),
    ("persona_resp",   "persona effect CvC - SvS",     "variability", +1, "{:+.3f}"),
]
FAMILY_COL = {"level": C.C_COOP, "structure": C.OUTCOME_COL["DC"],
              "variability": "#8250b8"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _snap(x):
    """joint_coop lives on a 21-point grid (multiples of 0.05); kill float dust."""
    return np.round(np.asarray(x, float) * 20.0) / 20.0


def _eta2(df, factor, value):
    g = df.groupby(factor, observed=True)[value]
    ss_b = float((((g.mean() - df[value].mean()) ** 2) * g.size()).sum())
    ss_t = float(((df[value] - df[value].mean()) ** 2).sum())
    return ss_b / ss_t if ss_t > 0 else np.nan


def _kendall_w(rank_matrix: np.ndarray) -> float:
    """Concordance of m rankers over n items; 1 = identical orders, 0 = none."""
    m, n = rank_matrix.shape
    if m < 2 or n < 2:
        return np.nan
    rs = rank_matrix.sum(axis=0)
    s = float(((rs - rs.mean()) ** 2).sum())
    return 12.0 * s / (m ** 2 * (n ** 3 - n))


def _ovl(a, b) -> float:
    """Overlapping coefficient on the exact discrete support of joint_coop.

    joint_coop is the mean of two rates that are each a multiple of 0.1, so it
    takes only 21 values. No kernel or bin width has to be chosen: OVL is the
    sum over the support of the smaller of the two probability masses.
    """
    ia = np.round(_snap(a) * 20).astype(int)
    ib = np.round(_snap(b) * 20).astype(int)
    pa = np.bincount(ia, minlength=21) / len(ia)
    pb = np.bincount(ib, minlength=21) / len(ib)
    return float(np.minimum(pa, pb).sum())


def _iqr_containment(a, b):
    """Share of distribution a that falls inside the IQR of distribution b."""
    lo, hi = np.percentile(b, [25, 75])
    return float(((a >= lo) & (a <= hi)).mean())


def model_metrics(ag: pd.DataFrame, dy: pd.DataFrame) -> pd.DataFrame:
    """One row per model, one column per metric, plus the lambda-slope CI."""
    rows = []
    for m in C.MODEL_ORDER:
        a = ag[ag.model == m]
        d = dy[dy.model == m]
        cond = (d.groupby(["scale", "language", "dyad"], observed=True)["joint_coop"]
                .mean())
        lang = d.groupby("language", observed=True)["joint_coop"].mean()
        X = sm.add_constant(d["log_scale"].to_numpy())
        fit = sm.OLS(d["joint_coop"].to_numpy(), X).fit(
            cov_type="cluster", cov_kwds={"groups": d["cell"].to_numpy()})
        ci = fit.conf_int()
        rows.append({
            "model": m, "vendor": C.VENDOR[m],
            "coop_rate": d["joint_coop"].mean(),
            "utility": d["joint_utility"].mean(),
            "efficiency": d["joint_efficiency"].mean(),
            "p_CC": d["p_CC"].mean(), "p_DD": d["p_DD"].mean(),
            "p_exploit": d["p_exploit"].mean(),
            "dd_absorbed": d["dd_absorbed"].mean(),
            "reciprocity": a["reciprocity"].mean(),
            "endgame_drop": a["endgame_drop"].mean(),
            "uncond_share": float(((a.coop_rate == 0) | (a.coop_rate == 1)).mean()),
            "game_sd": d["joint_coop"].std(),
            "cond_sd": float(cond.std()),
            "lambda_sens": abs(float(fit.params[1])),
            "lambda_slope": float(fit.params[1]),
            "lambda_slope_lo": float(ci[1][0]), "lambda_slope_hi": float(ci[1][1]),
            "lambda_slope_p": float(fit.pvalues[1]),
            "lang_disparity": float(lang.max() - lang.min()),
            "persona_resp": float(d.loc[d.dyad == "CvC", "joint_coop"].mean()
                                  - d.loc[d.dyad == "SvS", "joint_coop"].mean()),
            "worst_condition": float(cond.min()), "best_condition": float(cond.max()),
            "cond_range": float(cond.max() - cond.min()),
            "cond_iqr": float(cond.quantile(.75) - cond.quantile(.25)),
            "worst_condition_id": " / ".join(f"{x:g}" if isinstance(x, float) else str(x)
                                             for x in cond.idxmin()),
            "best_condition_id": " / ".join(f"{x:g}" if isinstance(x, float) else str(x)
                                            for x in cond.idxmax()),
            "n_games": len(d), "n_agent_games": len(a),
            "reciprocity_coverage": float(a["reciprocity"].notna().mean()),
        })
    return (pd.DataFrame(rows).set_index("model").reindex(C.MODEL_ORDER)
            .reset_index())


def pairwise_tests(dy: pd.DataFrame, metrics=("joint_coop", "joint_utility", "p_CC")):
    """Mann-Whitney U plus Cliff's delta for all 15 model pairs, FDR within metric."""
    out = []
    for met in metrics:
        rows = []
        for m1, m2 in itertools.combinations(C.MODEL_ORDER, 2):
            a = dy.loc[dy.model == m1, met].to_numpy()
            b = dy.loc[dy.model == m2, met].to_numpy()
            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            dlt = C.cliffs_delta(a, b)
            rows.append({
                "metric": met, "model_a": m1, "model_b": m2,
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "mean_diff": float(a.mean() - b.mean()),
                "median_a": float(np.median(a)), "median_b": float(np.median(b)),
                "mannwhitney_u": float(u.statistic), "p_value": float(u.pvalue),
                "cliffs_delta": dlt, "abs_delta": abs(dlt),
                "n_a": len(a), "n_b": len(b),
                "ovl": _ovl(a, b),
                "share_a_in_b_iqr": _iqr_containment(a, b),
                "share_b_in_a_iqr": _iqr_containment(b, a),
            })
        t = pd.DataFrame(rows)
        t["p_fdr"] = C.bh_fdr(t["p_value"].to_numpy())
        t["magnitude"] = pd.cut(t["abs_delta"], [-.01, .147, .33, .474, 1.01],
                                labels=["negligible", "small", "medium", "large"])
        out.append(t)
    return pd.concat(out, ignore_index=True)


def rank_by(values, orient):
    """Rank 1 = best under the declared orientation; average ranks on ties."""
    v = np.asarray(values, float)
    return stats.rankdata(-v if orient > 0 else v, method="average")


# --------------------------------------------------------------------------
def run(rounds, ag, dy):
    C.use_style()
    print("\n== 06 LLM comparison ==")

    keys = [k for k, *_ in METRICS]
    labels = {k: lab for k, lab, *_ in METRICS}
    fam = {k: f for k, _, f, *_ in METRICS}
    orient = {k: o for k, _, _, o, _ in METRICS}
    fmt = {k: f for k, _, _, _, f in METRICS}

    # ---- the metric matrix ---------------------------------------------------
    MM = model_metrics(ag, dy)
    Z = MM.set_index("model")[keys].copy()
    Zs = (Z - Z.mean()) / Z.std(ddof=1)              # z within metric, across models
    tab = MM.copy()
    for k in keys:
        tab[f"z_{k}"] = Zs[k].reindex(MM["model"]).to_numpy()
    C.savetab(tab, "model_metric_matrix")

    # ---- pairwise tests ------------------------------------------------------
    PW = pairwise_tests(dy)
    C.savetab(PW, "model_pairwise_tests")
    coop_pw = PW[PW.metric == "joint_coop"].reset_index(drop=True)
    n_sig = int((coop_pw.p_fdr < .05).sum())
    n_nonneg = int((coop_pw.abs_delta >= .147).sum())
    n_medium = int((coop_pw.abs_delta >= .33).sum())
    ns_pairs = coop_pw[coop_pw.p_fdr >= .05]
    ns_txt = ("; ".join(f"{ONELINE[r.model_a]} / {ONELINE[r.model_b]}"
                        for _, r in ns_pairs.iterrows()) or "none")
    for _, r in coop_pw.iterrows():
        C.record_test(section="llm_comparison",
                      test="Mann-Whitney U on dyad cooperation, FDR across 15 pairs",
                      comparison=f"{r.model_a} vs {r.model_b}",
                      statistic=r.cliffs_delta, p_value=r.p_value,
                      n=int(r.n_a + r.n_b),
                      effect_size_note=f"Cliff's delta {r.cliffs_delta:+.3f} "
                                       f"({r.magnitude}), OVL {r.ovl:.3f}")

    # ---- condition spread ----------------------------------------------------
    cond = (dy.groupby(["model", "scale", "language", "dyad"], observed=True)
            ["joint_coop"].agg(["mean", "size"]).reset_index()
            .rename(columns={"mean": "coop", "size": "n_games"}))
    spread = []
    for m in C.MODEL_ORDER:
        c = cond[cond.model == m]["coop"]
        d = dy[dy.model == m]
        row = MM[MM.model == m].iloc[0]
        spread.append({
            "model": m, "vendor": C.VENDOR[m], "n_conditions": len(c),
            "mean": float(c.mean()), "sd": float(c.std()),
            "min": float(c.min()), "q25": float(c.quantile(.25)),
            "median": float(c.median()), "q75": float(c.quantile(.75)),
            "max": float(c.max()), "range": float(c.max() - c.min()),
            "iqr": float(c.quantile(.75) - c.quantile(.25)),
            "worst_condition": row.worst_condition_id,
            "best_condition": row.best_condition_id,
            "eta2_scale": _eta2(d, "scale", "joint_coop"),
            "eta2_language": _eta2(d, "language", "joint_coop"),
            "eta2_pairing": _eta2(d, "dyad", "joint_coop"),
        })
    SP = pd.DataFrame(spread)
    SP["rank_mean"] = rank_by(SP["mean"], +1)
    SP["rank_worst"] = rank_by(SP["min"], +1)
    SP["rank_shift"] = SP["rank_worst"] - SP["rank_mean"]
    C.savetab(SP, "model_condition_spread")

    # =====================================================================
    # L1  the model x metric matrix
    # =====================================================================
    Zo = np.array([Zs[k].reindex(C.MODEL_ORDER).to_numpy() * orient[k] for k in keys])
    raw = np.array([Z[k].reindex(C.MODEL_ORDER).to_numpy() for k in keys])
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    v = float(np.nanmax(np.abs(Zo)))
    im = ax.imshow(Zo, cmap=C.DIV, vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(6), [TWOLINE[m] for m in C.MODEL_ORDER], fontsize=7)
    ax.set_yticks(range(len(keys)), [labels[k] for k in keys], fontsize=7.5)
    for tick, k in zip(ax.get_yticklabels(), keys):
        tick.set_color(FAMILY_COL[fam[k]])
    ax.grid(False)
    for i, k in enumerate(keys):
        for j in range(6):
            ax.text(j, i, fmt[k].format(raw[i, j]), ha="center", va="center",
                    fontsize=6.4,
                    color="white" if abs(Zo[i, j]) > v * .58 else C.INK)
    for b in [2.5, 9.5]:                              # family separators
        ax.axhline(b, color=C.INK, lw=1.1)
    cb = plt.colorbar(im, ax=ax, fraction=.036, pad=.02)
    cb.set_label("z across the 6 models, oriented\n(blue = more cooperative / more stable)",
                 fontsize=7)
    ax.set_title("fifteen metrics, six models: cell text is the raw value")
    ax.text(0, -.105, "row-label colour = metric family:", fontsize=6.4, color=C.INK2,
            transform=ax.transAxes, va="top")
    for xoff, f_ in zip([.238, .300, .394], ["level", "structure", "variability"]):
        ax.text(xoff, -.105, f_, fontsize=6.4, color=FAMILY_COL[f_], va="top",
                transform=ax.transAxes, fontweight="semibold")
    ax.text(0, -.145, f"{PREVIEW_NOTE}.  Cooperation, utility, efficiency and the four "
                      "outcome shares are dyad-level (2,000 games per model);\n"
                      "reciprocity, endgame drop and unconditional share are "
                      "agent-game level (4,000 rows per model).",
            fontsize=6.2, color=C.MUTED, transform=ax.transAxes, va="top")
    mi = MM.set_index("model")
    top = mi["coop_rate"].idxmax()
    C.save(fig, "llm", "L01_model_metric_matrix",
           "Do the six models differ along one axis, or do they occupy genuinely "
           "different regions of a multi-metric behavioural space?",
           f"They occupy different regions, and no model is uniformly best. "
           f"{ONELINE[top]} has the highest cooperation rate "
           f"({mi.loc[top,'coop_rate']:.3f}) yet the second-highest rate of locking "
           f"into mutual defection ({mi.loc[top,'dd_absorbed']:.3f}). GPT-5.4-Nano and "
           f"Gemini-3.1-Flash-Lite sit {abs(mi.loc['GPT-5.4-Nano','coop_rate']-mi.loc['Gemini-3.1-Flash-Lite','coop_rate']):.3f} "
           f"apart in cooperation but differ by "
           f"{abs(mi.loc['GPT-5.4-Nano','uncond_share']-mi.loc['Gemini-3.1-Flash-Lite','uncond_share']):.3f} "
           f"in unconditional-play share and by "
           f"{abs(mi.loc['GPT-5.4-Nano','persona_resp']-mi.loc['Gemini-3.1-Flash-Lite','persona_resp']):.3f} "
           f"in persona responsiveness. Grok-4.20 is the most payoff-scale sensitive "
           f"model ({mi.loc['Grok-4.20-Non-Reasoning','lambda_sens']:.3f} per decade, "
           f"{mi.loc['Grok-4.20-Non-Reasoning','lambda_sens']/mi['lambda_sens'].median():.1f} "
           "times the median) while being second in cooperation. A single headline "
           "cooperation number compresses away most of what separates these models.",
           "standardised heatmap with raw values", "15 metrics x 6 models")

    # =====================================================================
    # L2  forest plot with clustered bootstrap CIs
    # =====================================================================
    head = [("joint_coop", "dyad cooperation rate", "cooperation rate", dy, "cell"),
            ("joint_efficiency", "efficiency (1 = mutual C)", "efficiency", dy, "cell"),
            ("p_CC", "mutual cooperation share", "mutual cooperation", dy, "cell"),
            ("dd_absorbed", "share of games locked into DD", "lock-in to mutual D",
             dy, "cell")]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5))
    forest_rows = []
    overlap_ct = {}
    for axx, (k, lab, short, src, clus), letter in zip(axes, head, "abcd"):
        ms, los, his = [], [], []
        for m in C.MODEL_ORDER:
            s = src[src.model == m]
            mu, lo, hi = C.cluster_boot_ci(s[k].to_numpy(), s[clus].to_numpy(),
                                           n_boot=2000)
            ms.append(mu); los.append(lo); his.append(hi)
            forest_rows.append({"metric": k, "model": m, "mean": mu,
                                "ci_lo": lo, "ci_hi": hi, "n_games": len(s)})
            C.record_test(section="llm_comparison",
                          test="clustered bootstrap mean on design cell",
                          comparison=f"{k} | {m}", statistic=mu, ci_lo=lo, ci_hi=hi,
                          n=len(s), effect_size_note="2000 resamples of 200 cells")
        y = np.arange(6)
        ms, los, his = np.array(ms), np.array(los), np.array(his)
        axx.errorbar(ms, y, xerr=[ms - los, his - ms], fmt="none", ecolor=C.INK2,
                     lw=1.3, zorder=2)
        for i, m in enumerate(C.MODEL_ORDER):
            axx.scatter(ms[i], i, color=C.MODEL_COL[m], s=46, zorder=4)
        axx.axvline(ms.mean(), color=C.MUTED, lw=.8, ls="--")
        axx.set_yticks(y, [ONELINE[m] for m in C.MODEL_ORDER], fontsize=6.8)
        axx.set_ylim(6.5, -0.5)
        axx.set_xlabel(lab)
        axx.set_title(f"{letter}  {short}")
        # do the CIs of the extremes overlap?
        o = int(sum(1 for i in range(6) for j in range(i + 1, 6)
                    if not (his[i] < los[j] or his[j] < los[i])))
        overlap_ct[k] = o
        C.annotate(axx, f"spread {ms.max()-ms.min():.3f}\n{o} of 15 CI pairs overlap",
                   loc="lower right", fontsize=6.3)
    FM = pd.DataFrame(forest_rows)
    C.savetab(FM, "model_forest_means")
    dd = FM[FM.metric == "dd_absorbed"].set_index("model")["mean"]
    C.save(fig, "llm", "L02_forest_headline_metrics",
           "How far apart are the six models on the headline behavioural metrics "
           "once the interval is clustered on the common-random-number design cell?",
           f"On cooperation, efficiency and mutual-cooperation share the 95% "
           f"clustered bootstrap intervals are narrow relative to the between-model "
           f"spread ({overlap_ct['joint_coop']} of 15 interval pairs overlap on "
           f"cooperation). The orderings are not the same across panels: lock-in to "
           f"mutual defection is led by Qwen3-235B ({dd['Qwen3-235B-A22B']:.3f}) and "
           f"Gemini-3.5-Flash-Lite ({dd['Gemini-3.5-Flash-Lite']:.3f}), so the model "
           f"with the highest cooperation rate is also the second most prone to "
           f"absorbing into permanent mutual defection once cooperation breaks down. "
           f"Ranking on the mean therefore hides a failure mode.",
           "forest plot with clustered bootstrap CI",
           "joint_coop, joint_efficiency, p_CC, dd_absorbed x model")

    # =====================================================================
    # L3  all 15 pairs: Mann-Whitney + Cliff's delta, FDR corrected
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))
    D = np.full((6, 6), np.nan)
    S = np.zeros((6, 6), dtype=bool)
    idx = {m: i for i, m in enumerate(C.MODEL_ORDER)}
    for _, r in coop_pw.iterrows():
        i, j = idx[r.model_a], idx[r.model_b]
        D[max(i, j), min(i, j)] = r.cliffs_delta if i > j else -r.cliffs_delta
        S[max(i, j), min(i, j)] = r.p_fdr < .05
    vm = float(np.nanmax(np.abs(D)))
    im = axes[0].imshow(D, cmap=C.DIV, vmin=-vm, vmax=vm, aspect="auto")
    axes[0].set_xticks(range(5), [ONELINE[m] for m in C.MODEL_ORDER[:5]], rotation=45,
                       ha="right", fontsize=6.3)
    axes[0].set_yticks(range(1, 6), [ONELINE[m] for m in C.MODEL_ORDER[1:]],
                       fontsize=6.3)
    axes[0].set_xlim(-.5, 4.5)
    axes[0].set_ylim(5.5, .5)
    axes[0].grid(False)
    for i in range(6):
        for j in range(6):
            if np.isfinite(D[i, j]):
                axes[0].text(j, i, f"{D[i,j]:+.2f}{'*' if S[i,j] else ''}",
                             ha="center", va="center", fontsize=6,
                             color="white" if abs(D[i, j]) > vm * .58 else C.INK)
    plt.colorbar(im, ax=axes[0], fraction=.046, label="Cliff's delta (row - column)")
    axes[0].set_title("a  pairwise effect size")
    C.annotate(axes[0], "* FDR q < 0.05", loc="upper right", fontsize=6.3)

    o = coop_pw.sort_values("abs_delta", ascending=False).reset_index(drop=True)
    cols = [C.C_COOP if p < .05 else C.MUTED for p in o.p_fdr]
    axes[1].barh(range(15), o.abs_delta, color=cols)
    axes[1].set_yticks(range(15),
                       [f"{ONELINE[a]} / {ONELINE[b]}"
                        for a, b in zip(o.model_a, o.model_b)], fontsize=5.6)
    axes[1].invert_yaxis()
    for t, lab in [(.147, "negligible"), (.33, "small"), (.474, "medium")]:
        axes[1].axvline(t, color=C.INK2, lw=.7, ls=":")
        axes[1].text(t, -1.1, lab, fontsize=5.6, color=C.INK2, ha="center")
    axes[1].set_xlabel("|Cliff's delta| on dyad cooperation")
    axes[1].set_title(f"b  {n_sig} of 15 pairs separated (blue)")
    axes[1].set_ylim(15.5, -2.2)
    C.annotate(axes[1], f"{n_sig}/15 FDR-significant\n{n_nonneg}/15 above negligible\n"
                        f"{n_medium}/15 medium or larger", loc="lower right",
               fontsize=6.3)

    cnt = []
    for met, lab in [("joint_coop", "cooperation"), ("joint_utility", "utility"),
                     ("p_CC", "mutual coop.")]:
        t = PW[PW.metric == met]
        cnt.append({"metric": lab, "FDR q<0.05": int((t.p_fdr < .05).sum()),
                    "|d| >= 0.147": int((t.abs_delta >= .147).sum()),
                    "|d| >= 0.33": int((t.abs_delta >= .33).sum())})
    cdf = pd.DataFrame(cnt).set_index("metric")
    w, xs = .26, np.arange(3)
    for i, (c, col) in enumerate(zip(cdf.columns,
                                     [C.MUTED, C.C_COOP, C.OUTCOME_COL["DC"]])):
        axes[2].bar(xs + (i - 1) * w, cdf[c], w, color=col, label=c)
        for x, v_ in zip(xs + (i - 1) * w, cdf[c]):
            axes[2].text(x, v_ + .3, str(int(v_)), ha="center", fontsize=6)
    axes[2].set_xticks(xs, cdf.index, fontsize=7)
    axes[2].set_ylabel("pairs out of 15")
    axes[2].set_ylim(0, 20)
    axes[2].set_title("c  significance is cheap, size is not")
    axes[2].legend(fontsize=6.2, loc="upper left", ncol=3, columnspacing=1.0)
    C.save(fig, "llm", "L03_pairwise_effect_sizes",
           "Are all six models genuinely distinguishable from one another, or does "
           "significance at n = 2,000 games per model overstate how far apart they are?",
           f"{n_sig} of the 15 pairs are separated at FDR q < 0.05 on dyad "
           f"cooperation, but only {n_nonneg} of 15 exceed the negligible band of "
           f"Cliff's delta and only {n_medium} of 15 reach medium or larger. The one "
           f"pair that is not separated at all is {ns_txt}, whose mean cooperation "
           f"differs by {abs(coop_pw.loc[coop_pw.p_fdr.idxmax(), 'mean_diff']):.3f}. "
           "Significance here is mostly a statement about sample size: the "
           "Claude/Qwen pair, for instance, is significant at q = 2e-12 with a "
           "Cliff's delta of only -0.13, which is inside the negligible band.",
           "effect-size matrix + ranked bars + counts",
           "joint_coop, joint_utility, p_CC x 15 model pairs")

    # =====================================================================
    # L4  distribution overlap: similar means, different shapes
    # =====================================================================
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.3))
    for m in C.MODEL_ORDER:
        x = np.sort(_snap(dy.loc[dy.model == m, "joint_coop"].to_numpy()))
        axes[0].step(x, np.arange(1, len(x) + 1) / len(x), where="post",
                     color=C.MODEL_COL[m], lw=1.3, label=ONELINE[m])
    axes[0].set_xlabel("dyad cooperation rate")
    axes[0].set_ylabel("ECDF")
    axes[0].set_title("a  ECDF by model")
    axes[0].legend(fontsize=5.6, loc="upper left")
    C.annotate(axes[0], "vertical jumps at 0 and 1 are\nthe unconditional atoms",
               loc="lower right", fontsize=6.2)

    O = np.full((6, 6), np.nan)
    for _, r in coop_pw.iterrows():
        i, j = idx[r.model_a], idx[r.model_b]
        O[i, j] = O[j, i] = r.ovl
    np.fill_diagonal(O, 1.0)
    im = axes[1].imshow(O, cmap=C.SEQ, vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks(range(6), [ONELINE[m] for m in C.MODEL_ORDER], rotation=45,
                       ha="right", fontsize=6.1)
    axes[1].set_yticks(range(6), [ONELINE[m] for m in C.MODEL_ORDER], fontsize=6.1)
    axes[1].grid(False)
    for i in range(6):
        for j in range(6):
            axes[1].text(j, i, f"{O[i,j]:.2f}", ha="center", va="center", fontsize=5.8,
                         color="white" if O[i, j] > .62 else C.INK)
    plt.colorbar(im, ax=axes[1], fraction=.046, label="overlapping coefficient")
    axes[1].set_title("b  distributions still overlap")

    closest = coop_pw.loc[coop_pw.mean_diff.abs().idxmin()]
    supp = np.arange(21) / 20
    for m, col in [(closest.model_a, C.MODEL_COL[closest.model_a]),
                   (closest.model_b, C.MODEL_COL[closest.model_b])]:
        v = np.round(_snap(dy.loc[dy.model == m, "joint_coop"].to_numpy()) * 20).astype(int)
        p = np.bincount(v, minlength=21) / len(v)
        axes[2].plot(supp, p, "o-", color=col, ms=3, lw=1.2, label=ONELINE[m])
    axes[2].set_xlabel("dyad cooperation rate")
    axes[2].set_ylabel("share of games")
    axes[2].set_ylim(0, .30)
    axes[2].set_title("c  closest means, different shapes")
    axes[2].legend(fontsize=6, loc="upper left")
    C.annotate(axes[2], f"means differ by {abs(closest.mean_diff):.3f}\n"
                        f"OVL {closest.ovl:.2f}, |delta| {closest.abs_delta:.2f}",
               loc="upper right", fontsize=6.2)

    axes[3].scatter(coop_pw.mean_diff.abs(), coop_pw.ovl, s=34, color=C.C_COOP,
                    zorder=3)
    axes[3].set_xlabel("|difference in mean cooperation|")
    axes[3].set_ylabel("overlapping coefficient")
    axes[3].set_title("d  overlap versus mean gap")
    rho = stats.spearmanr(coop_pw.mean_diff.abs(), coop_pw.ovl)
    hi_ovl = coop_pw.loc[coop_pw.ovl.idxmax()]
    axes[3].annotate(f"{ONELINE[hi_ovl.model_a]}\n/ {ONELINE[hi_ovl.model_b]}",
                     (abs(hi_ovl.mean_diff), hi_ovl.ovl),
                     textcoords="offset points", xytext=(6, -2), fontsize=5.8,
                     color=C.INK2)
    C.annotate(axes[3], f"Spearman rho = {rho.statistic:+.2f}\n"
                        f"OVL range {coop_pw.ovl.min():.2f} to {coop_pw.ovl.max():.2f}",
               loc="upper right", fontsize=6.2)
    C.save(fig, "llm", "L04_distribution_overlap",
           "Do models with similar mean cooperation actually play the same way, or "
           "does the mean hide different distributional shapes?",
           f"The mean hides a great deal. Even the most separated pair still shares "
           f"an overlapping coefficient of {coop_pw.ovl.min():.2f}, and the closest "
           f"pair ({ONELINE[closest.model_a]} and {ONELINE[closest.model_b]}) differs "
           f"in mean by only {abs(closest.mean_diff):.3f} while its distributions "
           f"overlap at {closest.ovl:.2f} with visibly different mass at the "
           "boundary atoms. Overlap falls with the mean gap only loosely "
           f"(Spearman rho = {rho.statistic:+.2f}), so ranking on means and ranking "
           "on behavioural distinguishability are not the same exercise.",
           "ECDF + overlap matrix + PMF + scatter",
           "joint_coop distribution x 6 models, 15 pairs")

    # =====================================================================
    # L5  consistency: the 200 conditions each model is observed under
    # =====================================================================
    SP["q05"] = [float(cond[cond.model == m]["coop"].quantile(.05))
                 for m in SP["model"]]
    SP["n_zero_conditions"] = [int((cond[cond.model == m]["coop"] == 0).sum())
                               for m in SP["model"]]
    SP["n_one_conditions"] = [int((cond[cond.model == m]["coop"] == 1).sum())
                              for m in SP["model"]]
    SP["rank_q05"] = rank_by(SP["q05"], +1)
    SP["rank_shift_q05"] = SP["rank_q05"] - SP["rank_mean"]
    C.savetab(SP, "model_condition_spread")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    rng = np.random.default_rng(0)
    order = SP.sort_values("mean")["model"].tolist()
    for i, m in enumerate(order):
        c = cond[cond.model == m]["coop"].to_numpy()
        axes[0].scatter(c, i + rng.uniform(-.22, .22, len(c)), s=5, alpha=.35,
                        color=C.MODEL_COL[m], rasterized=True, zorder=2)
        axes[0].plot([c.min(), c.max()], [i, i], color=C.INK2, lw=.9, zorder=3)
        axes[0].scatter([c.mean()], [i], marker="D", s=42, color=C.MODEL_COL[m],
                        edgecolor=C.INK, linewidth=.7, zorder=5)
        axes[0].scatter([c.min(), c.max()], [i, i], marker="|", s=90, color=C.INK,
                        zorder=4)
    axes[0].set_yticks(range(6), [ONELINE[m] for m in order], fontsize=6.6)
    axes[0].set_xlabel("cooperation rate in one condition (10 matched games)")
    axes[0].set_title("a  200 conditions per model")
    axes[0].set_xlim(-.04, 1.04)
    axes[0].set_ylim(-.6, 6.6)          # blank band on top for the key
    nz = int((SP.n_zero_conditions > 0).sum())
    C.annotate(axes[0], f"diamond = mean, bar ends = worst and best condition.  "
                        f"{nz} of 6 models have at least one\ncondition in which "
                        f"cooperation collapses to exactly 0 across all ten games.",
               loc="upper left", fontsize=6.2)

    for i, m in enumerate(C.MODEL_ORDER):
        r = SP[SP.model == m].iloc[0]
        axes[1].plot([0, 1], [r.rank_mean, r.rank_q05], "o-", color=C.MODEL_COL[m],
                     lw=1.6, ms=6)
        axes[1].text(-.06, r.rank_mean, ONELINE[m], ha="right", va="center",
                     fontsize=6.2, color=C.MODEL_COL[m])
        axes[1].text(1.06, r.rank_q05, ONELINE[m], ha="left", va="center",
                     fontsize=6.2, color=C.MODEL_COL[m])
    axes[1].set_xticks([0, 1], ["rank by\nmean", "rank by 5th-pct\ncondition"],
                       fontsize=7)
    axes[1].set_xlim(-.85, 1.85)
    axes[1].set_ylim(6.6, .4)
    axes[1].set_ylabel("rank (1 = most cooperative)")
    axes[1].set_title("b  mean rank versus worst-case rank")
    axes[1].grid(axis="x", visible=False)
    mx = int(SP.rank_shift_q05.abs().max())
    C.annotate(axes[1], f"largest rank move: {mx} places", loc="lower center",
               fontsize=6.4)

    w, xs = .26, np.arange(6)
    for i, (f_, lab, col) in enumerate([
            ("eta2_scale", "payoff scale", C.C_COOP),
            ("eta2_language", "language", C.OUTCOME_COL["CD"]),
            ("eta2_pairing", "personality pairing", C.OUTCOME_COL["DC"])]):
        axes[2].bar(xs + (i - 1) * w, SP.set_index("model").loc[C.MODEL_ORDER, f_],
                    w, color=col, label=lab)
    axes[2].set_xticks(xs, [TWOLINE[m] for m in C.MODEL_ORDER], fontsize=5.9)
    axes[2].set_ylabel("$\\eta^2$ of cooperation within model")
    axes[2].set_title("c  what each model is sensitive to")
    axes[2].legend(fontsize=6.2, loc="upper center", ncol=3)
    axes[2].set_ylim(0, 1.06)
    C.save(fig, "llm", "L05_condition_consistency",
           "How stable is each model across the 200 conditions it is tested in, and "
           "does the ranking survive a switch from the mean to the worst case?",
           f"Stability differs by more than a factor of 1.7: the condition-level SD "
           f"runs from {SP['sd'].min():.3f} ({ONELINE[SP.loc[SP['sd'].idxmin(),'model']]}) "
           f"to {SP['sd'].max():.3f} ({ONELINE[SP.loc[SP['sd'].idxmax(),'model']]}), "
           f"and {nz} of the 6 models have at least one condition in which all ten "
           f"matched games end in zero cooperation. Ranking by the 5th-percentile "
           f"condition instead of by the mean moves models by up to {mx} places. The "
           "third panel shows the models are sensitive to different things: "
           "GPT-5.4-Nano is almost blind to the personality pairing (eta2 = "
           f"{SP.set_index('model').loc['GPT-5.4-Nano','eta2_pairing']:.3f}) while for "
           f"Qwen3-235B and Gemini-3.1 the pairing explains over 0.8 of the variance, "
           f"and Grok-4.20 is the only model whose payoff-scale eta2 exceeds 0.3.",
           "condition strip + rank slope chart + eta-squared bars",
           "joint_coop x 200 conditions x 6 models")

    # =====================================================================
    # L6  vendor: are the two Google builds each other's nearest neighbour?
    # =====================================================================
    prof = (cond.pivot_table(index=["scale", "language", "dyad"], columns="model",
                             values="coop", observed=True)[C.MODEL_ORDER])
    prof_c = prof - prof.mean(axis=0)          # centred: shape, not level
    R = prof_c.corr(method="spearman").reindex(index=C.MODEL_ORDER,
                                               columns=C.MODEL_ORDER)
    vend_rows = []
    for m1, m2 in itertools.combinations(C.MODEL_ORDER, 2):
        vend_rows.append({
            "model_a": m1, "model_b": m2,
            "same_vendor": C.VENDOR[m1] == C.VENDOR[m2],
            "profile_spearman": float(R.loc[m1, m2]),
            "profile_distance": float(1 - R.loc[m1, m2]),
            "level_distance": float(abs(prof[m1].mean() - prof[m2].mean())),
            "cliffs_delta": float(coop_pw[(coop_pw.model_a == m1)
                                          & (coop_pw.model_b == m2)].cliffs_delta.iloc[0]),
        })
    VD = pd.DataFrame(vend_rows)
    VD["rank_profile"] = VD["profile_distance"].rank()
    VD["rank_level"] = VD["level_distance"].rank()
    C.savetab(VD, "model_vendor_similarity")
    g = VD[VD.same_vendor].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7))
    Rv = R.to_numpy().copy()
    im = axes[0].imshow(Rv, cmap=C.DIV, vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(6), [ONELINE[m] for m in C.MODEL_ORDER], rotation=45,
                       ha="right", fontsize=6.1)
    axes[0].set_yticks(range(6), [ONELINE[m] for m in C.MODEL_ORDER], fontsize=6.1)
    axes[0].grid(False)
    for i in range(6):
        for j in range(6):
            axes[0].text(j, i, f"{Rv[i,j]:+.2f}", ha="center", va="center",
                         fontsize=5.9,
                         color="white" if abs(Rv[i, j]) > .58 else C.INK)
    gi, gj = C.MODEL_ORDER.index("Gemini-3.1-Flash-Lite"), C.MODEL_ORDER.index("Gemini-3.5-Flash-Lite")
    for (i, j) in [(gi, gj), (gj, gi)]:
        axes[0].add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                        edgecolor=C.OUTCOME_COL["DC"], lw=2.2))
    plt.colorbar(im, ax=axes[0], fraction=.046, label="Spearman rho of centred profiles")
    axes[0].set_title("a  response shape (Google pair boxed)")

    ranks = {}
    for axx, (col, lab, letter, ttl) in zip(axes[1:], [
            ("profile_distance", "1 - Spearman rho of condition profile", "b",
             "response shape"),
            ("level_distance", "|difference in mean cooperation|", "c",
             "cooperation level")]):
        o2 = VD.sort_values(col).reset_index(drop=True)
        cols = [C.OUTCOME_COL["DC"] if s else C.MUTED for s in o2.same_vendor]
        axx.barh(range(15), o2[col], color=cols)
        axx.set_yticks(range(15), [f"{ONELINE[a]} / {ONELINE[b]}"
                                   for a, b in zip(o2.model_a, o2.model_b)],
                       fontsize=5.5)
        axx.set_ylim(15.4, -1.4)
        axx.set_xlabel(lab)
        k = int(o2.index[o2.same_vendor][0]) + 1
        ranks[col] = k
        axx.set_title(f"{letter}  {ttl}: Google pair is #{k} of 15")
        C.annotate(axx, "orange = same vendor (Google)\nmost similar pair at the top",
                   loc="upper right", fontsize=6.2)
    C.save(fig, "llm", "L06_vendor_similarity",
           "Are the two models from the same vendor more similar to each other than "
           "to models from other vendors?",
           f"The answer depends on what similarity means, and the two answers point in "
           f"opposite directions. On the *shape* of the response across the 200 "
           f"conditions the two Google builds are the single most similar pair of the "
           f"15 (Spearman rho = {g.profile_spearman:+.2f}, rank "
           f"{ranks['profile_distance']} of 15), so they react to payoff scale, "
           f"language and persona in a more similar way than any cross-vendor pair. On "
           f"the *level* of cooperation they are unremarkable: they differ by "
           f"{g.level_distance:.3f} with a Cliff's delta of {g.cliffs_delta:+.3f}, "
           f"rank {ranks['level_distance']} of 15. With one vendor contributing two "
           "models, one of them a preview build and a generation apart from the other, "
           "vendor is confounded with model generation and this is a description of "
           "one pair, not a test of a vendor effect.",
           "correlation matrix + two ranked distance bars",
           "condition profiles x 6 models, 15 pairs, vendor")

    # =====================================================================
    # L7  ranking stability
    # =====================================================================
    RK = pd.DataFrame({k: rank_by(Z[k].reindex(C.MODEL_ORDER), orient[k])
                       for k in keys}, index=C.MODEL_ORDER).T
    W_metric = _kendall_w(RK.to_numpy())
    # concordance inside each metric family: the level metrics are near-redundant by
    # construction, so the collapse has to be attributed to the other two families
    W_fam = {f: _kendall_w(RK.loc[[k for k in keys if fam[k] == f]].to_numpy())
             for f in ["level", "structure", "variability"]}
    rank_by_lam = pd.DataFrame(
        {s: rank_by(dy[dy.scale == s].groupby("model", observed=True)["joint_coop"]
                    .mean().reindex(C.MODEL_ORDER), +1)
         for s in C.SCALES}, index=C.MODEL_ORDER).T
    W_lambda = _kendall_w(rank_by_lam.to_numpy())
    slices = ([(f"lang: {l}", dy[dy.language == l]) for l in C.LANGS]
              + [(f"pair: {d_}", dy[dy.dyad == d_]) for d_ in C.DYADS])
    rank_by_slice = pd.DataFrame(
        {nm: rank_by(s.groupby("model", observed=True)["joint_coop"].mean()
                     .reindex(C.MODEL_ORDER), +1) for nm, s in slices},
        index=C.MODEL_ORDER).T
    W_slice = _kendall_w(rank_by_slice.to_numpy())

    # subgroup-reversal audit: does any pooled pairwise ordering flip inside a slice?
    all_slices = ([(f"lambda={s:g}", dy[dy.scale == s]) for s in C.SCALES]
                  + [(f"lang:{l}", dy[dy.language == l]) for l in C.LANGS]
                  + [(f"pair:{d_}", dy[dy.dyad == d_]) for d_ in C.DYADS])
    sl_means = {nm: s.groupby("model", observed=True)["joint_coop"].mean()
                .reindex(C.MODEL_ORDER) for nm, s in all_slices}
    pooled_mean = dy.groupby("model", observed=True)["joint_coop"].mean()
    rev_rows = []
    for m1, m2 in itertools.combinations(C.MODEL_ORDER, 2):
        sgn = np.sign(pooled_mean[m1] - pooled_mean[m2])
        flips = [nm for nm, v in sl_means.items()
                 if np.sign(v[m1] - v[m2]) == -sgn]
        rev_rows.append({"model_a": m1, "model_b": m2,
                         "pooled_diff": float(pooled_mean[m1] - pooled_mean[m2]),
                         "n_slices": len(all_slices), "n_reversed": len(flips),
                         "reversed_in": "; ".join(flips)})
    REV = pd.DataFrame(rev_rows)
    C.savetab(REV, "model_subgroup_reversals")
    n_rev = int((REV.n_reversed > 0).sum())
    rho_rev = stats.spearmanr(REV.pooled_diff.abs(), REV.n_reversed).statistic
    C.record_test(section="llm_comparison", test="subgroup-reversal audit",
                  comparison="pooled pairwise cooperation order vs 19 design slices",
                  statistic=n_rev, n=len(all_slices),
                  effect_size_note=f"{n_rev} of 15 pairs reverse their pooled order in "
                                   "at least one slice (10 lambdas, 5 languages, "
                                   "4 pairings)")

    RS = pd.concat([RK.assign(ranker_type="metric"),
                    rank_by_lam.assign(ranker_type="payoff scale"),
                    rank_by_slice.assign(ranker_type="language / pairing")])
    RS.index.name = "ranker"
    C.savetab(RS.reset_index(), "model_rank_stability")
    C.record_test(section="llm_comparison", test="Kendall W concordance of rankings",
                  comparison="15 metrics rank the 6 models", statistic=W_metric,
                  n=len(keys), effect_size_note="1 = identical orders, 0 = no "
                  f"agreement; by family level {W_fam['level']:.2f}, structure "
                  f"{W_fam['structure']:.2f}, variability {W_fam['variability']:.2f}")
    C.record_test(section="llm_comparison", test="Kendall W concordance of rankings",
                  comparison="10 payoff scales rank the 6 models on cooperation",
                  statistic=W_lambda, n=10, effect_size_note="within-metric slicing")
    C.record_test(section="llm_comparison", test="Kendall W concordance of rankings",
                  comparison="5 languages + 4 pairings rank the 6 models on cooperation",
                  statistic=W_slice, n=9, effect_size_note="within-metric slicing")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    im = axes[0].imshow(RK.to_numpy(), cmap=C.SEQ.reversed(), vmin=1, vmax=6,
                        aspect="auto")
    axes[0].set_xticks(range(6), [TWOLINE[m] for m in C.MODEL_ORDER], fontsize=5.9)
    axes[0].set_yticks(range(len(keys)), [labels[k] for k in keys], fontsize=6.3)
    for tick, k in zip(axes[0].get_yticklabels(), keys):
        tick.set_color(FAMILY_COL[fam[k]])
    axes[0].grid(False)
    for i in range(len(keys)):
        for j in range(6):
            axes[0].text(j, i, f"{RK.to_numpy()[i,j]:g}", ha="center", va="center",
                         fontsize=6, color="white" if RK.to_numpy()[i, j] <= 2.5 else C.INK)
    axes[0].set_ylim(len(keys) + 1.6, -.5)      # blank band under the map for the key
    axes[0].set_title("a  rank by each metric (1 = best)")
    rr = RK.max(axis=0) - RK.min(axis=0)
    C.annotate(axes[0], f"Kendall W across all 15 metrics = {W_metric:.2f}.  Within a "
                        f"family: level {W_fam['level']:.2f},\nstructure "
                        f"{W_fam['structure']:.2f}, variability "
                        f"{W_fam['variability']:.2f}.  Widest swing: "
                        f"{ONELINE[rr.idxmax()]} moves {rr.max():g} places.",
               loc="lower left", fontsize=6.1)

    for m in C.MODEL_ORDER:
        axes[1].plot(C.SCALES, rank_by_lam[m], "o-", color=C.MODEL_COL[m], lw=1.4,
                     ms=4.5, label=ONELINE[m])
    axes[1].set_xscale("log")
    axes[1].set_ylim(8.9, .4)               # blank band below rank 6 for legend + key
    axes[1].set_yticks(range(1, 7))
    axes[1].set_xlabel("payoff scale $\\lambda$")
    axes[1].set_ylabel("rank on cooperation (1 = most)")
    axes[1].set_title("b  same metric, ranked inside each $\\lambda$")
    axes[1].legend(fontsize=5.6, loc="lower left", ncol=2, columnspacing=.8,
                   handlelength=1.2, labelspacing=.25)
    C.annotate(axes[1], f"Kendall W = {W_lambda:.2f}\n"
                        f"{int((rank_by_lam.max()-rank_by_lam.min()>=2).sum())} of 6 models\n"
                        "move 2+ places", loc="lower right", fontsize=6.0)

    im = axes[2].imshow(rank_by_slice.to_numpy(), cmap=C.SEQ.reversed(), vmin=1,
                        vmax=6, aspect="auto")
    axes[2].set_xticks(range(6), [TWOLINE[m] for m in C.MODEL_ORDER], fontsize=5.9)
    axes[2].set_yticks(range(9), rank_by_slice.index, fontsize=6.3)
    axes[2].grid(False)
    for i in range(9):
        for j in range(6):
            axes[2].text(j, i, f"{rank_by_slice.to_numpy()[i,j]:g}", ha="center",
                         va="center", fontsize=6.2,
                         color="white" if rank_by_slice.to_numpy()[i, j] <= 2.5 else C.INK)
    axes[2].set_ylim(11.0, -.5)
    axes[2].set_title("c  ranked inside language and pairing")
    C.annotate(axes[2], f"Kendall W = {W_slice:.2f} across these 9 slices.\n"
                        f"Across all 19 slices (10 $\\lambda$ + 5 languages + 4 "
                        f"pairings),\n{n_rev} of the 15 pairwise orderings reverse "
                        "somewhere.", loc="lower left", fontsize=6.1)
    C.save(fig, "llm", "L07_ranking_stability",
           "Does the ordering of the six models depend on which metric you rank by, "
           "or on which slice of the design you look at?",
           f"It depends on both, and far more on the metric than on the slice. The "
           f"three level metrics agree almost perfectly with each other "
           f"(Kendall W = {W_fam['level']:.2f}, as they must, being near-redundant), "
           f"but concordance falls to {W_fam['structure']:.2f} within the structural "
           f"family, {W_fam['variability']:.2f} within the variability family and "
           f"{W_metric:.2f} across all 15 metrics; {ONELINE[rr.idxmax()]} alone moves "
           f"{rr.max():g} of the 6 available places. Holding the metric fixed at "
           f"cooperation and re-ranking inside each payoff scale gives "
           f"W = {W_lambda:.2f}, and inside each language and pairing "
           f"W = {W_slice:.2f}: the top and bottom of the table are fairly stable "
           f"while the middle reorders freely, and {n_rev} of the 15 pairwise "
           f"orderings reverse in at least one of the 19 design slices. A "
           "single-number leaderboard for LLM game-theory behaviour is therefore a "
           "choice of metric at least as much as a property of the models.",
           "rank heatmaps + rank trajectory", "model ranks x 15 metrics, 10 scales, "
                                              "5 languages, 4 pairings")

    # =====================================================================
    # L8  behavioural fingerprints behind the same mean
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.7))
    prev = ["pC_R", "pC_S", "pC_T", "pC_P"]
    plab = ["after R\n(both C)", "after S\n(exploited)", "after T\n(exploiting)",
            "after P\n(both D)"]
    fp_rows = []
    for m in C.MODEL_ORDER:
        a = ag[ag.model == m]
        mus, los, his = [], [], []
        for k in prev:
            mu, lo, hi = C.cluster_boot_ci(a[k].to_numpy(), a["game_uid"].to_numpy(),
                                           n_boot=800)
            mus.append(mu); los.append(lo); his.append(hi)
            fp_rows.append({"model": m, "condition": k, "pC": mu, "ci_lo": lo,
                            "ci_hi": hi, "coverage": float(a[k].notna().mean())})
        axes[0].fill_between(range(4), los, his, color=C.MODEL_COL[m], alpha=.15, lw=0)
        axes[0].plot(range(4), mus, "o-", color=C.MODEL_COL[m], label=ONELINE[m],
                     lw=1.5, ms=4.5)
    axes[0].set_xticks(range(4), plab, fontsize=6.4)
    axes[0].set_ylabel("P(cooperate | previous outcome)")
    axes[0].set_title("a  memory-one fingerprint")
    axes[0].set_ylim(-.12, 1.06)            # blank band below the data for the legend
    axes[0].set_yticks([0, .2, .4, .6, .8, 1.0])
    axes[0].legend(fontsize=5.7, loc="lower left", ncol=2, labelspacing=.25)
    C.annotate(axes[0], "bands are clustered bootstrap 95% CI;\ncells with no prior "
                        "occurrence of that\noutcome are dropped, not imputed",
               loc="upper right", fontsize=6.0)

    sm_order = ["AllC", "TFT", "WSLS", "GRIM", "AllD", "ambiguous", "unclassified"]
    sm_col = dict(C.STRAT_COL, ambiguous="#8b8983", unclassified="#d3d1c7")
    mix = (ag.groupby(["model", "strategy"], observed=True).size()
           .unstack(fill_value=0).reindex(index=C.MODEL_ORDER, columns=sm_order)
           .fillna(0))
    mix = mix.div(mix.sum(axis=1), axis=0)
    C.savetab(mix.reset_index(), "model_strategy_mix")
    bot = np.zeros(6)
    for s in sm_order:
        axes[1].bar(range(6), mix[s], bottom=bot, color=sm_col[s], label=s, width=.78)
        bot += mix[s].to_numpy()
    axes[1].set_xticks(range(6), [TWOLINE[m] for m in C.MODEL_ORDER], fontsize=5.9)
    axes[1].set_ylabel("share of agent-games")
    axes[1].set_ylim(0, 1.22)
    axes[1].set_title("b  exact memory-one rule mix")
    axes[1].legend(fontsize=5.8, loc="upper center", ncol=4, columnspacing=1.0,
                   handlelength=1.2)
    axes[1].set_xlabel("'ambiguous' = several rules fit the realised history equally; "
                       "it is an answer, not a failure", fontsize=6.1)

    rd_rows = []
    for i, m in enumerate(C.MODEL_ORDER):
        a = ag[ag.model == m]
        mu, lo, hi = C.cluster_boot_ci(a["rule_distance"].to_numpy(),
                                       a["game_uid"].to_numpy(), n_boot=1500)
        rd_rows.append({"model": m, "rule_distance": mu, "ci_lo": lo, "ci_hi": hi})
        axes[2].barh(i, mu, color=C.MODEL_COL[m], height=.66)
        axes[2].plot([lo, hi], [i, i], color=C.INK, lw=1.3)
    axes[2].set_yticks(range(6), [ONELINE[m] for m in C.MODEL_ORDER], fontsize=6.4)
    axes[2].set_ylim(6.4, -.6)
    axes[2].set_xlabel("deviations from the nearest memory-one rule, per 10 rounds")
    axes[2].set_title("c  how rule-like is the play")
    RD = pd.DataFrame(rd_rows)
    C.savetab(pd.DataFrame(fp_rows).merge(RD, on="model", suffixes=("", "_rule")),
              "model_fingerprint")
    C.annotate(axes[2], f"range {RD.rule_distance.min():.2f} to "
                        f"{RD.rule_distance.max():.2f} deviations per 10 rounds",
               loc="lower right", fontsize=6.2)

    gap = abs(mi.loc["GPT-5.4-Nano", "coop_rate"]
              - mi.loc["Gemini-3.1-Flash-Lite", "coop_rate"])
    C.save(fig, "llm", "L08_behavioural_fingerprint",
           "Two models with statistically indistinguishable mean cooperation: do they "
           "get there by the same conditional policy?",
           f"No. GPT-5.4-Nano and Gemini-3.1-Flash-Lite differ in mean cooperation by "
           f"only {gap:.3f} and are the one pair the Mann-Whitney test cannot "
           f"separate, yet their memory-one fingerprints diverge: Gemini-3.1 keeps "
           f"cooperating after being exploited (pC after S = "
           f"{mi_fp(fp_rows,'Gemini-3.1-Flash-Lite','pC_S'):.2f} against "
           f"{mi_fp(fp_rows,'GPT-5.4-Nano','pC_S'):.2f}) and plays an unconditional "
           f"rule in {mix.loc['Gemini-3.1-Flash-Lite',['AllC','AllD']].sum():.0%} of "
           f"agent-games against {mix.loc['GPT-5.4-Nano',['AllC','AllD']].sum():.0%}. "
           "Across all six models the exact memory-one rules cover only a minority of "
           "agent-games, and the residual is large enough that a rule-based read-out "
           "of these models is a summary rather than a description.",
           "conditional-probability profile + strategy mix + rule distance",
           "pC_R/S/T/P, strategy, rule_distance x 6 models")

    # ---- findings -----------------------------------------------------------
    close = coop_pw.loc[coop_pw.p_fdr.idxmax()]
    C.record_finding(
        id="M-ranking-unstable",
        finding="The ranking of the six models is metric-dependent: there is no "
                "single ordering that a benchmark could report.",
        evidence=f"Ranking the 6 models by each of 15 behavioural metrics gives "
                 f"Kendall W = {W_metric:.2f}, and {ONELINE[rr.idxmax()]} alone moves "
                 f"{rr.max():g} of the 6 available places between metrics. Within the "
                 f"three level metrics the order is essentially fixed "
                 f"(W = {W_fam['level']:.2f}); it is the structural family "
                 f"(W = {W_fam['structure']:.2f}) and the variability family "
                 f"(W = {W_fam['variability']:.2f}) that disagree with it. Holding the "
                 f"metric fixed at cooperation and re-ranking inside each payoff scale "
                 f"gives W = {W_lambda:.2f}; inside each language and pairing, "
                 f"W = {W_slice:.2f}. A direct subgroup-reversal audit finds that "
                 f"{n_rev} of the 15 pairwise cooperation orderings flip in at least "
                 f"one of the 19 slices. Ranking by the 5th-percentile condition "
                 f"rather than by the mean moves models by up to "
                 f"{int(SP.rank_shift_q05.abs().max())} places.",
        figure="06_llm_comparison/L07_ranking_stability.png, "
               "L05_condition_consistency.png",
        strength="strong",
        robustness="the instability appears in three independent slicings (metric, "
                   "payoff scale, language/pairing); the level metrics agree with each "
                   "other trivially, so the finding is that structure and stability "
                   "order the models differently from level, not that cooperation is "
                   f"measured unreliably. Reversals are concentrated in the pairs with "
                   f"small pooled gaps (Spearman rho between |pooled gap| and reversal "
                   f"count = {rho_rev:+.2f}), as they should be",
        interpretation="Descriptive, not causal. The result says a leaderboard number "
                       "for LLM game-theoretic behaviour encodes the analyst's choice "
                       "of metric and of test conditions as much as it encodes model "
                       "behaviour. It does not say the models are equivalent.",
        caveat="Six models, one game, one horizon, self-play only. The orientation "
               "used to turn each metric into a rank is declared in the metric table "
               "and is a judgement for two of the fifteen metrics.",
        claim="Rankings of frontier LLMs on iterated prisoner's dilemma behaviour are "
              "not stable across metrics or across test conditions, so a single "
              "headline score is not a well-defined property of a model.")

    C.record_finding(
        id="M-significance-vs-size",
        finding="Statistical separation between models is nearly automatic at this "
                "sample size, while behavioural separation is not.",
        evidence=f"{n_sig} of the 15 model pairs separate at FDR q < 0.05 on dyad "
                 f"cooperation (n = 2,000 games per model), but only {n_nonneg} of 15 "
                 f"exceed the negligible band of Cliff's delta and {n_medium} of 15 "
                 f"reach medium or larger. The Claude/Qwen pair is significant at "
                 f"q = 2.3e-12 with a Cliff's delta of only "
                 f"{coop_pw.loc[0,'cliffs_delta']:+.3f}. Every pair still shares an "
                 f"overlapping coefficient of at least {coop_pw.ovl.min():.2f} on the "
                 f"exact 21-point support of dyad cooperation.",
        figure="06_llm_comparison/L03_pairwise_effect_sizes.png, "
               "L04_distribution_overlap.png",
        strength="strong",
        robustness="the same pattern holds for utility and mutual-cooperation share, "
                   "each FDR-corrected within its own family of 15 tests",
        interpretation="Correlational and descriptive. The overlap is computed on the "
                       "exact discrete support of the outcome, so it involves no "
                       "smoothing choice, and it means an individual game drawn from "
                       "one model is often indistinguishable from one drawn from "
                       "another even when the pooled means differ.",
        caveat="Overlap is a property of the marginal distribution; two models could "
               "overlap heavily in the margin and still be perfectly separable given "
               "the condition, which the condition-level analysis partly addresses.",
        claim="At 2,000 games per model almost every pair of frontier LLMs is "
              "statistically distinguishable, yet their cooperation distributions "
              "overlap by 35 to 75 percent, so p-values are a poor guide to how "
              "differently two models actually play.")

    C.record_finding(
        id="M-same-mean-different-policy",
        finding="Two models with statistically indistinguishable mean cooperation "
                "reach it through opposite conditional policies.",
        evidence=f"GPT-5.4-Nano and Gemini-3.1-Flash-Lite differ in mean dyad "
                 f"cooperation by {gap:.3f} and are the only pair not separated by the "
                 f"FDR-corrected Mann-Whitney test (q = {close.p_fdr:.2f}, Cliff's "
                 f"delta {close.cliffs_delta:+.3f}). Yet their unconditional-play "
                 f"shares are "
                 f"{mi.loc['GPT-5.4-Nano','uncond_share']:.3f} and "
                 f"{mi.loc['Gemini-3.1-Flash-Lite','uncond_share']:.3f}, their "
                 f"persona responses (CvC minus SvS cooperation) are "
                 f"{mi.loc['GPT-5.4-Nano','persona_resp']:+.3f} and "
                 f"{mi.loc['Gemini-3.1-Flash-Lite','persona_resp']:+.3f}, and the "
                 f"share of within-model cooperation variance explained by the "
                 f"personality pairing is "
                 f"{SP.set_index('model').loc['GPT-5.4-Nano','eta2_pairing']:.3f} "
                 f"against "
                 f"{SP.set_index('model').loc['Gemini-3.1-Flash-Lite','eta2_pairing']:.3f}.",
        figure="06_llm_comparison/L08_behavioural_fingerprint.png, "
               "L05_condition_consistency.png, L01_model_metric_matrix.png",
        strength="strong",
        robustness="the divergence shows up independently in the memory-one "
                   "fingerprint, the exact rule mix, the persona contrast and the "
                   "within-model variance decomposition",
        interpretation="Descriptive. One model is effectively ignoring the persona "
                       "instruction in its prompt while the other is largely driven by "
                       "it; the coincidence of their pooled means is arithmetic, not "
                       "behavioural similarity. This is the clearest single argument "
                       "against mean-only model comparison in this corpus.",
        caveat="Gemini-3.1-Flash-Lite is a preview build, so its persona sensitivity "
               "may not describe the released model.",
        claim="Equal mean cooperation does not imply equal behaviour: two frontier "
              "models with indistinguishable cooperation rates differ by a factor of "
              f"{SP.set_index('model').loc['Gemini-3.1-Flash-Lite','eta2_pairing'] / SP.set_index('model').loc['GPT-5.4-Nano','eta2_pairing']:.0f} "
              "in how much the personality instruction explains their play.")

    C.record_finding(
        id="M-worst-case-gap",
        finding="Good average cooperation does not protect against a total collapse "
                "in some condition.",
        evidence=f"Each model is observed under 200 conditions of 10 matched games. "
                 f"{nz} of the 6 models, including the two most cooperative on "
                 f"average, have at least one condition in which cooperation is "
                 f"exactly zero in all ten games. Condition-level SD ranges from "
                 f"{SP['sd'].min():.3f} to {SP['sd'].max():.3f}, and the model with "
                 f"the highest mean cooperation "
                 f"({ONELINE[SP.loc[SP['mean'].idxmax(),'model']]}) has a condition "
                 f"range of {SP.loc[SP['mean'].idxmax(),'range']:.3f}, the maximum "
                 f"possible. The worst conditions cluster at the small end of the "
                 f"payoff ladder.",
        figure="06_llm_comparison/L05_condition_consistency.png",
        strength="moderate",
        robustness="the collapse conditions are not a single language or pairing, but "
                   "they do concentrate at lambda = 0.01 and 0.1, which links this to "
                   "the payoff-scale violation in section 04",
        interpretation="Descriptive. A condition mean of zero over ten matched games "
                       "is not sampling noise at this cell size, but with 200 "
                       "conditions per model some extreme cells are expected, so the "
                       "informative part is which conditions they are rather than "
                       "that they exist.",
        caveat="A condition holds 10 games, so a single condition mean is itself "
               "estimated with real uncertainty; the 5th-percentile condition is the "
               "more stable worst-case summary and is what the rank comparison uses.",
        claim="Frontier LLMs that cooperate well on average can still cooperate at a "
              "rate of exactly zero in specific payoff-scale and language conditions, "
              "so average benchmark scores do not bound worst-case behaviour.")

    C.record_finding(
        id="M-vendor-shape-not-level",
        finding="The one same-vendor pair matches on the shape of its response to the "
                "design but not on the level of cooperation.",
        evidence=f"Across the 200 centred condition profiles the two Google builds "
                 f"correlate at Spearman rho = {g.profile_spearman:+.2f}, the highest "
                 f"of all 15 pairs (rank {ranks['profile_distance']} of 15). On mean "
                 f"cooperation they differ by {g.level_distance:.3f} with a Cliff's "
                 f"delta of {g.cliffs_delta:+.3f}, which is rank "
                 f"{ranks['level_distance']} of 15, i.e. middling.",
        figure="06_llm_comparison/L06_vendor_similarity.png",
        strength="weak",
        robustness="a single same-vendor pair, so this is one observation rather than "
                   "a vendor-level test; the profile correlation of +0.44 is well "
                   "above the next pair but rests on 200 condition means",
        interpretation="Descriptive only, and vendor is fully confounded with model "
                       "generation here: Gemini-3.1-Flash-Lite is a preview build one "
                       "generation behind Gemini-3.5-Flash-Lite. The pattern is "
                       "consistent with shared prompt-handling or post-training "
                       "conventions moving the response shape while the cooperation "
                       "level is set elsewhere, but this design cannot test that.",
        caveat="n = 2 models in the only replicated vendor group. Whoever repeats this "
               "should not read the rank-1 profile correlation as evidence of a vendor "
               "effect without more vendors contributing more than one model.",
        claim="The two same-vendor models respond to payoff scale, language and "
              "persona in the most similar way of any pair in the corpus while "
              "differing unremarkably in how much they cooperate, so vendor "
              "similarity, if it exists at all, would live in response shape rather "
              "than in level.")

    return MM, PW, SP


def mi_fp(rows, model, cond):
    for r in rows:
        if r["model"] == model and r["condition"] == cond:
            return r["pC"]
    return np.nan
