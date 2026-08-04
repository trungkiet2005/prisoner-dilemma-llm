"""FR13-FR17: payoffs, bimodality, openings, who the model reacts to, and
whether the measurement itself is trustworthy.

FR17 is the audit figure.  Role, replicate and split-half invariance are
properties the *task* guarantees, not results, so any departure is a defect in
the measurement rather than a finding about the model -- which is exactly why
it belongs in the paper rather than in a lab notebook.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdlib.ingest import cluster_bootstrap_ci, payoff_matrix
from pdlib.metrics import grouped_ci
from pdlib.natstyle import (DATADIR, FRONTIER, INK, INK2, LANG_ORDER, MODEL_C,
                            MODEL_LABEL, MODEL_M, MUTED, PAGE, PERSONALITY_C,
                            PERSONALITY_ORDER, RULE, SCALE_ORDER, SPINE,
                            TABDIR, W2, bars, caption, figure, finalize, hgrid,
                            model_legend, save, shared_model_legend,
                            use_journal_style)

use_journal_style()

SEED = 0
N_BOOT = 2000
PM = payoff_matrix("frontier")


def _boot_diff(a, b, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI for mean(a) - mean(b) over two independent samples."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    d = np.empty(n_boot)
    for i in range(n_boot):
        d[i] = (a[rng.integers(0, len(a), len(a))].mean()
                - b[rng.integers(0, len(b), len(b))].mean())
    return a.mean() - b.mean(), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ==========================================================================
# FR13 -- payoffs, and who profits from a mismatched pairing
# ==========================================================================
def fig_payoffs(games):
    fig = figure(W2, 2.55)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    # (a) distribution of per-round payoff -----------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    data = [games.loc[games.model == m, "payoff_per_round"].to_numpy()
            for m in FRONTIER]
    parts = ax.violinplot(data, positions=np.arange(len(FRONTIER)),
                          widths=0.72, showextrema=False, showmedians=False)
    for body, mdl in zip(parts["bodies"], FRONTIER):
        body.set_facecolor(MODEL_C[mdl])
        body.set_edgecolor(PAGE)
        body.set_linewidth(0.5)
        body.set_alpha(0.85)
    for i, (v, mdl) in enumerate(zip(data, FRONTIER)):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, q1, q3, color=INK, lw=2.2, zorder=5, capstyle="round")
        ax.plot(i, med, "o", ms=3.0, mfc=PAGE, mec=INK, mew=0.7, zorder=6)

    # reference lines stop before the label column, so nothing is struck through
    x_end = len(FRONTIER) - 0.45
    for val, lab in ((PM["R"], "$R$ = 6  both cooperate"),
                     ((PM["T"] + PM["S"]) / 2, "$(T{+}S)/2$ = 5  anti-coordinated"),
                     (PM["P"], "$P$ = 2  both defect")):
        ax.hlines(val, -0.6, x_end, color=MUTED, lw=0.5, ls=(0, (2.5, 2)),
                  zorder=1)
        ax.text(x_end + 0.10, val, lab, ha="left", va="center", fontsize=5.4,
                color=MUTED)
    ax.set_xticks(range(len(FRONTIER)))
    ax.set_xticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in FRONTIER],
                       fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("payoff per round (base units)")
    ax.set_ylim(0, 10.4)
    ax.set_xlim(-0.6, len(FRONTIER) + 1.75)
    ax.set_title("What a game actually earns", pad=6)

    # (b) within-dyad inequality ---------------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax, axis="x")
    rows = []
    for mdl in FRONTIER:
        g = games[games.model == mdl]
        mixed = g[g.dyad.isin(["CvS", "SvC"])]
        s = mixed.loc[mixed.personality == "selfish", "payoff_per_round"]
        c = mixed.loc[mixed.personality == "cooperative", "payoff_per_round"]
        d, lo, hi = _boot_diff(s, c)
        same = g[g.dyad.isin(["CvC", "SvS"])]
        a1 = same.loc[same.agent == 1, "payoff_per_round"]
        a2 = same.loc[same.agent == 2, "payoff_per_round"]
        d0, lo0, hi0 = _boot_diff(a1, a2)
        rows.append({"model": mdl, "gap": d, "lo": lo, "hi": hi,
                     "control_gap": d0, "control_lo": lo0, "control_hi": hi0})
    ineq = pd.DataFrame(rows)
    ineq.to_csv(TABDIR / "T_FR19_within_dyad_inequality.csv", index=False)

    y = np.arange(len(ineq))[::-1].astype(float)
    ax.axvline(0, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    off = 0.17
    for yi, r in zip(y, ineq.itertuples()):
        c = MODEL_C[r.model]
        ax.hlines(yi + off, r.lo, r.hi, color=c, lw=0.8, zorder=3)
        ax.plot(r.gap, yi + off, marker=MODEL_M[r.model], ms=4.4, mfc=c,
                mec=PAGE, mew=0.6, ls="none", zorder=4)
        ax.hlines(yi - off, r.control_lo, r.control_hi, color=c, lw=0.8, zorder=3)
        ax.plot(r.control_gap, yi - off, marker=MODEL_M[r.model], ms=4.4,
                mfc=PAGE, mec=c, mew=1.0, ls="none", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in ineq.model],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.95, len(ineq) - 0.30)
    ax.set_xlabel("payoff advantage (base units per round)")
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=4.0, mfc=MUTED,
                          mec=PAGE, mew=0.6, label="selfish $-$ cooperative,\nmixed pairings"),
               plt.Line2D([], [], ls="none", marker="o", ms=4.0, mfc=PAGE,
                          mec=MUTED, mew=1.0, label="agent 1 $-$ agent 2,\nmatched pairings")]
    ax.legend(handles=handles, ncol=1, loc="upper right",
              bbox_to_anchor=(1.02, 1.03), handlelength=0.8, labelspacing=0.8)
    ax.set_title("Who profits, 95% CI", pad=6)

    finalize(fig, [pa, pb], ["a", "b"])
    caption(fig, "In a the bar is the interquartile range and the open circle "
                 "the median. The open markers in b are a control: agents 1 and "
                 "2 of a matched pairing are given identical prompts, so that "
                 "interval should straddle zero. For Claude and Gemini it does "
                 "not, which is a position artefact rather than a result about "
                 "personas -- FR17a audits it directly.")
    save(fig, "FR13_payoffs")
    return ineq


# ==========================================================================
# FR14 -- game-level cooperation is bimodal
# ==========================================================================
def fig_bimodality(games):
    fig = figure(W2, 2.4)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])

    # (a) ridgeline of the game-level cooperation rate ------------------------
    # Ten rounds means the rate can only be k/10, so this is a discrete
    # distribution over eleven values, not a histogram: binning it would merge
    # adjacent values and invent empty bins where none exist.
    ax = pa = fig.add_subplot(gs[0, 0])
    n_r = int(games.n_rounds.iloc[0])
    vals = np.arange(n_r + 1) / n_r
    step = 1.0
    for k, mdl in enumerate(FRONTIER):
        v = games.loc[games.model == mdl, "coop_rate"].to_numpy()
        h = np.array([(np.isclose(v, x)).mean() for x in vals])
        base = (len(FRONTIER) - 1 - k) * step
        ax.bar(vals, h / h.max() * 0.86, bottom=base, width=1 / n_r * 0.82,
               color=MODEL_C[mdl], edgecolor=PAGE, linewidth=0.4, zorder=3)
        ax.hlines(base, -0.02, 1.02, color=RULE, lw=0.6, zorder=2)
        ax.text(-0.045, base + 0.34, MODEL_LABEL[mdl].replace(" ", "\n", 1),
                ha="right", va="center", fontsize=6.0, color=INK)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.12, len(FRONTIER) * step)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("cooperation rate within a game")
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("The mean describes almost no game", pad=6)

    # (b) how much of the mass sits at the corners ---------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    rows = []
    for mdl in FRONTIER:
        v = games.loc[games.model == mdl, "coop_rate"].to_numpy()
        rows.append({"model": mdl,
                     "all_D": float((v == 0).mean()),
                     "mixed": float(((v > 0) & (v < 1)).mean()),
                     "all_C": float((v == 1).mean()),
                     "mean": float(v.mean()), "sd": float(v.std(ddof=1))})
    pol = pd.DataFrame(rows)
    pol.to_csv(TABDIR / "T_FR20_polarisation.csv", index=False)

    y = np.arange(len(pol))[::-1]
    seg = [("all_D", "#d55e00", "every round D"),
           ("mixed", "#e8e8e8", "mixed"),
           ("all_C", "#0072b2", "every round C")]
    left = np.zeros(len(pol))
    for col, colr, lab in seg:
        v = pol[col].to_numpy()
        ax.barh(y, v, left=left, height=0.6, color=colr, edgecolor=PAGE,
                linewidth=0.5, zorder=3, label=lab)
        for yi, (l, vv) in zip(y, zip(left, v)):
            if vv > 0.07:
                ax.text(l + vv / 2, yi, f"{vv:.2f}", ha="center", va="center",
                        fontsize=5.6, zorder=4,
                        color=INK if col == "mixed" else PAGE)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in pol.model],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("share of games")
    ax.set_ylim(-0.75, len(pol) - 0.25)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.11),
              handlelength=0.9, columnspacing=0.8)
    ax.set_title("Absorbing corners", pad=16)

    finalize(fig, [pa, pb], ["a", "b"])
    caption(fig, "Each row in a is normalised to its own maximum, so shapes are "
                 "comparable but heights are not counts. GPT-4o's mean of 0.55 "
                 "sits in the emptiest part of its own distribution: the modal "
                 "game is all-C. Every mean in this suite should be read "
                 "against this panel.")
    save(fig, "FR14_bimodality")
    return pol


# ==========================================================================
# FR15 -- openings
# ==========================================================================
def fig_openings(games, rounds):
    fig = figure(W2, 2.45)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])

    # (a) opening move by persona --------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    ci = grouped_ci(games, ["model", "personality"], "first_move_coop",
                    n_boot=N_BOOT)
    ci.to_csv(TABDIR / "T_FR21_openings.csv", index=False)
    x = np.arange(len(FRONTIER))
    w = 0.34
    for k, pers in enumerate(PERSONALITY_ORDER):
        sub = ci[ci.personality == pers].set_index("model").reindex(FRONTIER)
        xs = x + (k - 0.5) * w
        bars(ax, xs, sub["mean"], PERSONALITY_C[pers], width=w * 0.9,
             label=f"{pers} persona")
        ax.vlines(xs, sub["lo"], sub["hi"], color=INK, lw=0.6, zorder=5)
    ax.axhline(0.5, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in FRONTIER],
                       fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("P(cooperate on round 1)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, len(FRONTIER) - 0.4)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.06),
              handlelength=0.9)
    ax.set_title("The opening move is where the persona lands", pad=16)

    # (b) does a cooperative opening pay? ------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax, axis="x")
    rows = []
    for mdl in FRONTIER:
        g = games[games.model == mdl]
        oc = g.loc[g.first_move_coop == 1, "payoff_per_round"]
        od = g.loc[g.first_move_coop == 0, "payoff_per_round"]
        d, lo, hi = _boot_diff(oc, od)
        rows.append({"model": mdl, "payoff_open_C": oc.mean(),
                     "payoff_open_D": od.mean(), "gap": d, "lo": lo, "hi": hi,
                     "n_open_C": len(oc), "n_open_D": len(od)})
    op = pd.DataFrame(rows)
    op.to_csv(TABDIR / "T_FR22_opening_pays.csv", index=False)

    y = np.arange(len(op))[::-1]
    ax.axvline(0, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    ax.text(0.0, -0.72, " opening does not matter", ha="left", va="center",
            fontsize=5.6, color=MUTED)
    for yi, r in zip(y, op.itertuples()):
        ax.hlines(yi, r.lo, r.hi, color=MODEL_C[r.model], lw=1.0, zorder=3)
        ax.plot(r.gap, yi, marker=MODEL_M[r.model], ms=4.4,
                mfc=MODEL_C[r.model], mec=PAGE, mew=0.6, ls="none", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in op.model],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1.05, len(op) - 0.35)
    ax.set_xlabel("payoff after opening C $-$ after opening D\n"
                  "(base units per round)")
    ax.set_title("What the opening buys, 95% CI", pad=16)

    finalize(fig, [pa, pb], ["a", "b"])
    caption(fig, "Round 1 carries no history, so panel a isolates the prompt's "
                 "effect from anything learned in play; compare it with the "
                 "whole-game persona effect in FR5b. Panel b conditions on the "
                 "focal agent's own opening only, and the opponent is another "
                 "instance of the same model.")
    save(fig, "FR15_openings")
    return op


# ==========================================================================
# FR16 -- does the model react to the opponent, or to itself?
# ==========================================================================
def fig_reactivity(rounds):
    fig = figure(W2, 2.45)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])

    lag = rounds.dropna(subset=["prev_action", "prev_opp_action"])
    rows = []
    for mdl in FRONTIER:
        d = lag[lag.model == mdl]
        self_p = float((d.action == d.prev_action).mean())
        opp_m = float((d.action == d.prev_opp_action).mean())
        # influence of each history channel on the next move
        s_c = d.loc[d.prev_action == "C", "coop"]
        s_d = d.loc[d.prev_action == "D", "coop"]
        o_c = d.loc[d.prev_opp_action == "C", "coop"]
        o_d = d.loc[d.prev_opp_action == "D", "coop"]
        sd, slo, shi = _boot_diff(s_c, s_d)
        od, olo, ohi = _boot_diff(o_c, o_d)
        rows.append({"model": mdl, "self_persist": self_p, "opp_match": opp_m,
                     "self_influence": sd, "self_lo": slo, "self_hi": shi,
                     "opp_influence": od, "opp_lo": olo, "opp_hi": ohi})
    react = pd.DataFrame(rows)
    react.to_csv(TABDIR / "T_FR23_reactivity.csv", index=False)

    # (a) the plane -----------------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    anchors = {"AllC / AllD": (1.0, 0.5, -5, 4), "TFT": (0.5, 1.0, 5, -2),
               "Alternator": (0.0, 0.5, 5, 4)}
    for name, (ax_, ay, ox, oy) in anchors.items():
        ax.plot(ax_, ay, marker="*", ms=6.5, mfc=RULE, mec=SPINE, mew=0.5,
                ls="none", zorder=2)
        ax.annotate(name, (ax_, ay), textcoords="offset points",
                    xytext=(ox, oy), ha="right" if ox < 0 else "left",
                    fontsize=5.8, color=MUTED)
    ax.axhline(0.5, color=RULE, lw=0.5, zorder=1)
    ax.axvline(0.5, color=RULE, lw=0.5, zorder=1)
    for r in react.itertuples():
        ax.plot(r.self_persist, r.opp_match, marker=MODEL_M[r.model], ms=5.0,
                mfc=MODEL_C[r.model], mec=PAGE, mew=0.7, ls="none", zorder=5)
    ax.set_xlabel("perseveration  P($a_t = a_{t-1}$)")
    ax.set_ylabel("imitation  P($a_t$ = opponent's $a_{t-1}$)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.20, 1.05)
    ax.set_title("Who is being copied", pad=6)

    # (b) influence of each channel ------------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    x = np.arange(len(react))
    w = 0.34
    for k, (val, lo, hi, lab, alpha) in enumerate((
            ("self_influence", "self_lo", "self_hi", "own previous move", 1.0),
            ("opp_influence", "opp_lo", "opp_hi", "opponent's previous move", 0.45))):
        xs = x + (k - 0.5) * w
        cols = [MODEL_C[m] for m in react.model]
        cont = ax.bar(xs, react[val], width=w * 0.9, color=cols,
                      edgecolor=PAGE, linewidth=0.5, alpha=alpha, zorder=3)
        ax.vlines(xs, react[lo], react[hi], color=INK, lw=0.6, zorder=5)
    ax.axhline(0, color=SPINE, lw=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in react.model],
                       fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("$\\Delta$ P(cooperate) after C vs after D")
    ax.set_xlim(-0.6, len(x) - 0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=PAGE,
                             lw=0.5, label="own previous move"),
               plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=PAGE,
                             lw=0.5, alpha=0.45, label="opponent's previous move")]
    ax.legend(handles=handles, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.06), handlelength=0.9, columnspacing=0.8)
    ax.set_title("Which channel drives the next move", pad=16)

    finalize(fig, [pa, pb], ["a", "b"])
    leg = shared_model_legend(fig, [pa, pb], ncol=4, lines=False)
    caption(fig, "A pure tit-for-tat sits at imitation 1 and inherits whatever "
                 "perseveration the opponent's sequence induces; AllC and AllD "
                 "sit at perseveration 1. In b a positive bar means the channel "
                 "pulls towards cooperation; the two channels are correlated in "
                 "play, so these are marginal associations, not a decomposition.",
                 below=leg)
    save(fig, "FR16_reactivity")
    return react


# ==========================================================================
# FR17 -- invariances the measurement should respect
# ==========================================================================
CELL = ["model", "scale_nominal", "language", "dyad"]


def fig_invariance(games):
    fig = figure(W2, 2.5)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0])

    # (a) role symmetry -------------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    # Only matched pairings: in a mixed pairing the two agents hold different
    # personas, so a difference between them is the persona effect (FR5), not
    # a violation of role symmetry.  CvC and SvS give the two agents identical
    # prompts, so there the roles really are interchangeable.
    matched = games[games.dyad.isin(["CvC", "SvS"])]
    cells = (matched.groupby(CELL + ["agent"]).coop_rate.mean()
             .unstack("agent").dropna())
    cells.columns = [f"agent{c}" for c in cells.columns]
    for mdl in FRONTIER:
        sub = cells.xs(mdl, level="model")
        ax.plot(sub.agent1, sub.agent2, marker=MODEL_M[mdl], ms=3.2,
                mfc=MODEL_C[mdl], mec=PAGE, mew=0.4, ls="none", alpha=0.8,
                zorder=3)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    r_role = float(np.corrcoef(cells.agent1, cells.agent2)[0, 1])
    bias = float((cells.agent2 - cells.agent1).mean())
    role = pd.DataFrame([{
        "model": m,
        "bias": float((cells.xs(m, level="model").agent2
                       - cells.xs(m, level="model").agent1).mean()),
        "r": float(np.corrcoef(cells.xs(m, level="model").agent1,
                               cells.xs(m, level="model").agent2)[0, 1]),
    } for m in FRONTIER])
    role.to_csv(TABDIR / "T_FR26_role_symmetry.csv", index=False)
    ax.text(0.04, 0.96, f"matched pairings only\n$r$ = {r_role:.2f}\n"
            f"mean bias = {bias:+.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.8,
            color=INK2)
    ax.set_xlabel("agent 1 cooperation rate")
    ax.set_ylabel("agent 2 cooperation rate")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Role symmetry", pad=6)

    # (b) replicate spread against the binomial floor -------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    rows = []
    for mdl in FRONTIER:
        g = games[games.model == mdl]
        grp = g.groupby(CELL + ["agent"]).coop_rate
        sd = grp.std(ddof=1)
        mu = grp.mean()
        n_rounds = int(g.n_rounds.iloc[0])
        # if every round were an independent coin at the cell mean, the
        # replicate-to-replicate SD of a 10-round rate would be this
        floor = np.sqrt(mu * (1 - mu) / n_rounds)
        rows.append({"model": mdl, "observed_sd": float(sd.mean()),
                     "binomial_sd": float(floor.mean()),
                     "ratio": float((sd / floor.replace(0, np.nan)).mean())})
    rep = pd.DataFrame(rows)
    rep.to_csv(TABDIR / "T_FR24_replicate_spread.csv", index=False)

    x = np.arange(len(rep))
    w = 0.34
    bars(ax, x - w / 2, rep.observed_sd, [MODEL_C[m] for m in rep.model],
         width=w * 0.9)
    bars(ax, x + w / 2, rep.binomial_sd, "#d0d0d0", width=w * 0.9)
    for xi, r in zip(x, rep.itertuples()):
        ax.text(xi, max(r.observed_sd, r.binomial_sd) + 0.008,
                f"{r.ratio:.1f}$\\times$", ha="center", va="bottom",
                fontsize=5.8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m].split()[0] for m in rep.model],
                       fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("SD of cooperation rate across the 10 replicates")
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, max(rep.observed_sd.max(), rep.binomial_sd.max()) * 1.35)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=PAGE,
                             lw=0.5, label="observed"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#d0d0d0",
                             edgecolor=PAGE, lw=0.5, label="independent-round floor")]
    ax.legend(handles=handles, ncol=1, loc="upper right",
              bbox_to_anchor=(1.02, 1.03), handlelength=0.9)
    ax.set_title("Replicate spread", pad=6)

    # (c) split-half reliability of the cell means ----------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax, axis="x")
    rng = np.random.default_rng(SEED)
    rows = []
    for mdl in FRONTIER:
        g = games[games.model == mdl].copy()
        # split whole dyads, not agent-rows: the two agents of a game share a
        # history, so letting them fall on opposite sides of the split would
        # correlate the halves and inflate the reliability
        uids = g.game_uid.unique()
        rs = []
        for _ in range(200):
            assign = pd.Series(rng.integers(0, 2, len(uids)), index=uids)
            g["half"] = g.game_uid.map(assign)
            piv = (g.groupby(CELL + ["agent", "half"]).coop_rate.mean()
                   .unstack("half").dropna())
            if piv.shape[1] < 2 or len(piv) < 3:
                continue
            r = np.corrcoef(piv.iloc[:, 0], piv.iloc[:, 1])[0, 1]
            # Spearman-Brown steps a half-length correlation up to full length
            rs.append(2 * r / (1 + r))
        rows.append({"model": mdl, "reliability": float(np.mean(rs)),
                     "lo": float(np.percentile(rs, 2.5)),
                     "hi": float(np.percentile(rs, 97.5))})
    rel = pd.DataFrame(rows)
    rel.to_csv(TABDIR / "T_FR25_split_half.csv", index=False)

    y = np.arange(len(rel))[::-1]
    for yi, r in zip(y, rel.itertuples()):
        ax.barh(yi, r.reliability, height=0.55, color=MODEL_C[r.model],
                edgecolor=PAGE, linewidth=0.5, zorder=3)
        ax.hlines(yi, r.lo, r.hi, color=INK, lw=0.6, zorder=5)
        ax.text(r.reliability + 0.02, yi, f"{r.reliability:.2f}", ha="left",
                va="center", fontsize=6.0, color=INK2)
    ax.axvline(0.7, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    ax.text(0.7, -0.70, " 0.7", ha="left", va="center", fontsize=5.6,
            color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].split()[0] for m in rel.model])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1.0, len(rel) - 0.3)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("split-half reliability of the\ncell means (Spearman-Brown)")
    ax.set_title("Reproducibility", pad=6)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"])
    leg = shared_model_legend(fig, [pa, pb, pc], ncol=4, lines=False)
    caption(fig, f"None of these three is a result: agents 1 and 2 of a matched "
                 f"pairing get identical prompts, the ten replicates share a "
                 f"prompt, and the cell means should reproduce across a random "
                 f"split of the dyads. Agent 2 cooperates {abs(bias):.3f} less "
                 f"than agent 1 on average, a position artefact worth carrying "
                 f"into any claim that rests on a single role. Panel b's ratio "
                 f"is the observed spread over what independent rounds would "
                 f"give; above 1 means rounds within a game are correlated, "
                 f"which is why every interval in this suite resamples whole "
                 f"dyads.", below=leg)
    save(fig, "FR17_invariance")
    return rep, rel, role


def main():
    games = pd.read_parquet(DATADIR / "frontier_games.parquet")
    rounds = pd.read_parquet(DATADIR / "frontier_rounds.parquet")
    ineq = fig_payoffs(games)
    pol = fig_bimodality(games)
    op = fig_openings(games, rounds)
    react = fig_reactivity(rounds)
    rep, rel, role = fig_invariance(games)
    for name, t in (("within-dyad inequality", ineq), ("polarisation", pol),
                    ("opening pays", op), ("reactivity", react),
                    ("replicate spread", rep), ("split-half", rel),
                    ("role symmetry", role)):
        print(f"\n{name}")
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
