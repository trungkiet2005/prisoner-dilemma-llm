"""FR27-FR28: the payoff plane, and distance to each canonical rule.

These are the two questions FR13 and FR7 answer only in summary.  FR13 reports
a signed advantage between personas; FR27 puts every dyad in the (payoff_1,
payoff_2) plane, where the game's landmarks -- the Nash point, mutual
cooperation, the anti-coordination line -- are visible and a dyad's position
relative to them is the whole story.  FR7 reports the distance to the *nearest*
canonical rule; FR28 gives the distance to each of the five, because "nearest"
hides whether the runner-up was a close second or nowhere near.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from pdlib.ingest import payoff_matrix
from pdlib.metrics import reciprocity, state_transitions
from pdlib.natstyle import (CMAP_SEQ, DATADIR, FRONTIER, INK, INK2, MEMORY1,
                            MODEL_C, MODEL_LABEL, MODEL_M, MUTED, PAGE, RULE,
                            SCALE_ORDER, SPINE, TABDIR, W2, annotate_heatmap,
                            caption, colorbar, figure, finalize, hgrid, save,
                            shared_model_legend, use_journal_style)

use_journal_style()

SEED = 0
PM = payoff_matrix("frontier")
STATE_ORDER = ["R", "S", "T", "P"]
RULE_ORDER = ["AllC", "TFT", "WSLS", "GRIM", "AllD"]


# ==========================================================================
# FR27 -- where a dyad lands in payoff space
# ==========================================================================
def fig_payoff_space(games):
    dy = (games.pivot_table(index=["model", "game_uid"], columns="agent",
                            values="payoff_per_round")
          .rename(columns={1: "p1", 2: "p2"}).dropna().reset_index())
    dy["total"] = dy.p1 + dy.p2
    dy["gap"] = (dy.p1 - dy.p2).abs()

    fig = figure(W2, 2.6)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0])

    # (a) the payoff plane ----------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax, axis="both")
    rng = np.random.default_rng(SEED)
    for mdl in FRONTIER:
        s = dy[dy.model == mdl]
        j = rng.normal(0, 0.07, (len(s), 2))
        ax.plot(s.p1 + j[:, 0], s.p2 + j[:, 1], "o", ms=1.6,
                mfc=MODEL_C[mdl], mec="none", alpha=0.40, ls="none", zorder=3)

    T, R, P, S = PM["T"], PM["R"], PM["P"], PM["S"]
    ax.plot([S, T], [T, S], color=MUTED, lw=0.6, ls=(0, (2.5, 2)), zorder=4)
    # label parked in the empty corner beyond the line, with a leader, so the
    # dashes do not run through the text
    ax.annotate(f"anti-coordination\ntotal = $T+S$ = {T + S:g}",
                xy=(8.2, 1.8), xytext=(8.6, 6.6), ha="center", va="bottom",
                fontsize=5.4, color=MUTED,
                arrowprops=dict(arrowstyle="-", lw=0.45, color=MUTED,
                                shrinkA=2, shrinkB=2))
    for (px, py), mark, lab in (((R, R), "*", f"mutual C\n({R:g}, {R:g})"),
                                ((P, P), "X", f"Nash: mutual D\n({P:g}, {P:g})")):
        ax.plot(px, py, marker=mark, ms=8 if mark == "*" else 6, mfc=PAGE,
                mec=INK, mew=0.9, ls="none", zorder=7)
        ax.annotate(lab, (px, py), textcoords="offset points", xytext=(6, 6),
                    ha="left", va="bottom", fontsize=5.6, color=INK, zorder=7)
    ax.set_xlabel("agent 1 payoff per round")
    ax.set_ylabel("agent 2 payoff per round")
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(-0.4, 10.4)
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks(range(0, 11, 2))
    ax.set_title("Where dyads land", pad=6)

    # (b) within-dyad inequality ----------------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    data = [dy.loc[dy.model == m, "gap"].to_numpy() for m in FRONTIER]
    parts = ax.violinplot(data, positions=np.arange(len(FRONTIER)), widths=0.72,
                          showextrema=False, showmedians=False)
    for body, mdl in zip(parts["bodies"], FRONTIER):
        body.set_facecolor(MODEL_C[mdl])
        body.set_edgecolor(PAGE)
        body.set_linewidth(0.5)
        body.set_alpha(0.85)
    rows = []
    for i, (v, mdl) in enumerate(zip(data, FRONTIER)):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, q1, q3, color=INK, lw=2.0, zorder=5)
        ax.plot(i, med, "o", ms=2.8, mfc=PAGE, mec=INK, mew=0.6, zorder=6)
        rows.append({"model": mdl, "median_gap": med, "q25": q1, "q75": q3,
                     "share_equal": float((v < 0.05).mean()),
                     "mean_total": float(dy.loc[dy.model == mdl, "total"].mean())})
    ineq = pd.DataFrame(rows)
    ineq.to_csv(TABDIR / "T_FR44_dyad_inequality.csv", index=False)
    for i, r in enumerate(ineq.itertuples()):
        ax.text(i, 10.2, f"{r.share_equal:.0%}\nequal", ha="center", va="top",
                fontsize=5.4, color=MUTED)
    ax.set_xticks(range(len(FRONTIER)))
    ax.set_xticklabels([MODEL_LABEL[m].split()[0] for m in FRONTIER],
                       fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("|payoff$_1$ $-$ payoff$_2$| per round")
    ax.set_ylim(0, 10.6)
    ax.set_xlim(-0.6, len(FRONTIER) - 0.4)
    ax.set_title("Within-dyad inequality", pad=6)

    # (c) efficiency by model and scale --------------------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    piv = (games.pivot_table(index="model", columns="scale_nominal",
                             values="efficiency", aggfunc="mean")
           .reindex(index=FRONTIER, columns=SCALE_ORDER))
    piv.to_csv(TABDIR / "T_FR45_efficiency_grid.csv")
    im = ax.imshow(piv.to_numpy(), cmap=CMAP_SEQ, vmin=0.5, vmax=0.9,
                   aspect="auto")
    annotate_heatmap(ax, piv.to_numpy(), thresh=0.74, size=6.4)
    ax.set_xticks(range(len(SCALE_ORDER)))
    ax.set_xticklabels([f"$\\times${s:g}" for s in SCALE_ORDER])
    ax.set_yticks(range(len(FRONTIER)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in FRONTIER], fontsize=6.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("payoff scale, $\\lambda$")
    colorbar(fig, im, ax, label="efficiency (1.0 = all-CC)")
    ax.set_title("Payoff efficiency", pad=6)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"])
    leg = shared_model_legend(fig, [pa, pb, pc], ncol=4, lines=False)
    caption(fig, f"One point per dyad in a ({len(dy):,} dyads), jittered by "
                 f"0.07 units so overlapping dyads stay visible. The dashed "
                 f"line is every split of the {T + S:g} units an "
                 f"anti-coordinated round pays; mutual cooperation sits above "
                 f"it at a total of {2 * R:g}, the Nash point far below at "
                 f"{2 * P:g}. The percentage above each violin in b is the "
                 f"share of dyads whose two agents earned within 0.05 of each "
                 f"other.", below=leg)
    save(fig, "FR27_payoff_space")
    return dy, ineq


# ==========================================================================
# FR28 -- distance to each canonical rule, not just the nearest
# ==========================================================================
def fig_canonical_distance(rounds):
    fp, rec = {}, {}
    for mdl in FRONTIER:
        r = rounds[rounds.model == mdl]
        fp[mdl] = state_transitions(r)
        rec[mdl] = reciprocity(r)
    fpd = pd.DataFrame(fp).T
    recd = pd.DataFrame(rec).T

    fig = figure(W2, 2.55)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])

    # (a) the reactive plane at model level -----------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=(0, (2.5, 2)), zorder=1)
    # short label, parked on the empty stretch of the diagonal near AllC; the
    # full reading is in the caption
    ax.text(0.83, 0.86, "no reciprocity", rotation=45, rotation_mode="anchor",
            ha="center", va="bottom", fontsize=5.4, color=MUTED)
    anchors = {"AllC": (1.0, 1.0, -6, -2), "TFT": (1.0, 0.0, -6, 4),
               "AllD": (0.0, 0.0, 6, 2)}
    for name, (px, qy, ox, oy) in anchors.items():
        ax.plot(px, qy, marker="*", ms=7.5, mfc=PAGE, mec=INK, mew=0.8,
                ls="none", zorder=6)
        ax.annotate(name, (px, qy), textcoords="offset points",
                    xytext=(ox, oy), ha="right" if ox < 0 else "left",
                    fontsize=6.0, color=INK)
    for mdl in FRONTIER:
        ax.plot(recd.loc[mdl, "p_c_after_c"], recd.loc[mdl, "p_c_after_d"],
                marker=MODEL_M[mdl], ms=5.2, mfc=MODEL_C[mdl], mec=PAGE,
                mew=0.7, ls="none", zorder=7)
    ax.set_xlabel("P(cooperate | opponent cooperated last round)")
    ax.set_ylabel("P(cooperate | opponent defected last round)")
    ax.set_xlim(-0.06, 1.10)
    ax.set_ylim(-0.06, 1.10)
    ax.set_title("Above the line is inverted reciprocity", pad=6)

    # (b) RMS distance to every canonical rule --------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    V = np.array([[fpd.loc[m, f"pC_{s}"] for s in STATE_ORDER]
                  for m in FRONTIER])
    Rref = np.array([MEMORY1[k] for k in RULE_ORDER], dtype=float)
    D = np.sqrt(((V[:, None, :] - Rref[None, :, :]) ** 2).mean(axis=2))
    dist = pd.DataFrame(D, index=FRONTIER, columns=RULE_ORDER)
    dist.to_csv(TABDIR / "T_FR46_rule_distance_matrix.csv")

    im = ax.imshow(D, cmap=CMAP_SEQ, vmin=0.2, vmax=0.75, aspect="auto")
    annotate_heatmap(ax, D, thresh=0.55, size=6.4)
    for i in range(D.shape[0]):
        j = int(D[i].argmin())
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none",
                               edgecolor=INK, linewidth=1.2, zorder=6))
    ax.set_xticks(range(len(RULE_ORDER)))
    ax.set_xticklabels(RULE_ORDER, fontsize=6.2)
    ax.set_yticks(range(len(FRONTIER)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in FRONTIER], fontsize=6.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    colorbar(fig, im, ax, label="RMS distance in fingerprint space")
    ax.set_title("Distance to every canonical rule, nearest boxed", pad=6)

    finalize(fig, [pa, pb], ["a", "b"])
    leg = shared_model_legend(fig, [pa, pb], ncol=4, lines=False)
    spread = float(np.ptp(np.sort(D, axis=1)[:, :2], axis=1).mean())
    caption(fig, f"A point on the diagonal in a ignores the opponent entirely; "
                 f"below it is ordinary reciprocity, above it is inverted -- "
                 f"cooperating more after being defected on, which is where "
                 f"Mistral sits. Panel b is why FR7c's 'nearest rule' needs "
                 f"reading with care: the gap between each model's nearest and "
                 f"second-nearest rule averages only {spread:.2f}, so the "
                 f"label is a ranking among near-ties, not an identification.",
            below=leg)
    save(fig, "FR28_canonical_distance")
    return fpd, recd, dist


def main():
    games = pd.read_parquet(DATADIR / "frontier_games.parquet")
    rounds = pd.read_parquet(DATADIR / "frontier_rounds.parquet")
    dy, ineq = fig_payoff_space(games)
    fpd, recd, dist = fig_canonical_distance(rounds)
    print()
    print(ineq.round(3).to_string(index=False))
    print()
    print(recd[["p_c_after_c", "p_c_after_d", "reciprocity"]].round(3).to_string())
    print()
    print(dist.round(3).to_string())


if __name__ == "__main__":
    main()
