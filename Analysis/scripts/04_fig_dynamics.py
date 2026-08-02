"""F06 -- round dynamics, F07 -- outcome composition over time,
F08 -- reciprocity and memory-one fingerprints, F09 -- payoffs and inequality."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdlib.ingest import wilson
from pdlib.metrics import MEMORY1_REFERENCE, reciprocity, state_transitions
from pdlib.style import (CMAP_DIV, CMAP_SEQ, C_COOP, C_DEFECT, DATADIR,
                         FRONTIER_MODELS, GRID, INK, INK2, MODEL, MODEL_ORDER,
                         MUTED, OUTCOME, OUTCOME_LABEL, OUTCOME_ORDER,
                         PERSONALITY, SMALL_MODELS, STRATEGY, SURFACE, TABDIR,
                         panel_tag, savefig, use_paper_style)

use_paper_style()


# --------------------------------------------------------------------------
def fig_round_dynamics(rounds):
    fig = plt.figure(figsize=(11.2, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.70, wspace=0.45)

    def coop_curve(ax, sub, models, title, nmax):
        for mdl in models:
            d = sub[sub.model == mdl].groupby("round").coop.agg(["mean", "size"])
            lo, hi = wilson(d["mean"], d["size"])
            ax.fill_between(d.index, lo, hi, color=MODEL[mdl], alpha=0.15, lw=0)
            ax.plot(d.index, d["mean"], "-o", color=MODEL[mdl], ms=3.6,
                    mec=SURFACE, mew=0.9, label=mdl)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.set_xlabel("round")
        ax.set_ylabel("cooperation rate")
        ax.set_xlim(0.5, nmax + 0.5)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(ncol=1, fontsize=6.5, loc="upper right")

    # (a) frontier, 10 rounds, horizon unknown -------------------------------
    ax = fig.add_subplot(gs[0, 0])
    coop_curve(ax, rounds[rounds.family == "frontier"], FRONTIER_MODELS,
               "Frontier · horizon hidden", 10)
    panel_tag(ax, "a", dx=-0.24)

    # (b) small, 30 rounds, horizon announced --------------------------------
    ax = fig.add_subplot(gs[0, 1])
    coop_curve(ax, rounds[rounds.family == "small"], SMALL_MODELS,
               "Open-weight · horizon known", 30)
    panel_tag(ax, "b", dx=-0.24)

    # (c) normalised game clock so the two families are comparable -----------
    ax = fig.add_subplot(gs[0, 2])
    r = rounds.copy()
    r["bin"] = np.clip((r.round_frac * 10).astype(int), 0, 9)
    for mdl in MODEL_ORDER:
        d = r[r.model == mdl].groupby("bin").coop.mean()
        ax.plot((d.index + 0.5) / 10, d.values, "-o", color=MODEL[mdl], ms=3.6,
                mec=SURFACE, mew=0.9, label=mdl)
    ax.set_xlabel("position in the game (0 = start, 1 = end)")
    ax.set_ylabel("cooperation rate")
    ax.set_title("Rescaled game clock")
    ax.legend(ncol=2, fontsize=5.8, loc="lower left")
    panel_tag(ax, "c", dx=-0.26)

    # (d) endgame effect ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    first = rounds[rounds["round"] == 1].groupby("model").coop.mean()
    mid = rounds[(rounds.round_frac > 0.3) & (rounds.round_frac < 0.7)] \
        .groupby("model").coop.mean()
    last = rounds[rounds.round_frac == 1.0].groupby("model").coop.mean()
    x = np.arange(len(MODEL_ORDER))
    for k, (ser, lab, col) in enumerate(((first, "first round", C_COOP),
                                         (mid, "middle", MUTED),
                                         (last, "last round", C_DEFECT))):
        ax.bar(x + (k - 1) * 0.27, ser.reindex(MODEL_ORDER), width=0.25,
               color=col, edgecolor=SURFACE, linewidth=1.0, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("-")[0] for m in MODEL_ORDER], rotation=20,
                       ha="right")
    ax.set_ylabel("cooperation rate")
    ax.set_title("Endgame unravelling")
    ax.legend(ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    panel_tag(ax, "d", dx=-0.24)

    # (e) round dynamics by dyad (pooled) ------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    r = rounds.copy()
    r["bin"] = np.clip((r.round_frac * 10).astype(int), 0, 9)
    styles = {"CvC": ("-", C_COOP), "CvS": ((0, (4, 2)), C_COOP),
              "SvC": ((0, (4, 2)), C_DEFECT), "SvS": ("-", C_DEFECT)}
    for dy, (ls, col) in styles.items():
        d = r[r.dyad == dy].groupby("bin").coop.mean()
        ax.plot((d.index + 0.5) / 10, d.values, linestyle=ls, color=col,
                marker="o", ms=3.6, mec=SURFACE, mew=0.9, label=dy)
    ax.set_xlabel("position in the game")
    ax.set_ylabel("cooperation rate")
    ax.set_title("Persona dyads")
    ax.legend(ncol=2, fontsize=6.5)
    panel_tag(ax, "e", dx=-0.24)

    # (f) absorbing mutual defection -----------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    games = pd.read_parquet(DATADIR / "games.parquet")
    ab = games[games.agent == 1].groupby("model").dd_absorbed.mean().reindex(MODEL_ORDER)
    y = np.arange(len(ab))[::-1]
    ax.barh(y, ab.values, color=[MODEL[m] for m in ab.index], height=0.62,
            edgecolor=SURFACE, linewidth=1.0)
    for yi, v in zip(y, ab.values):
        ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(ab.index)
    ax.set_xlim(0, max(ab.values) * 1.28)
    ax.set_xlabel("share of games ending trapped in DD")
    ax.set_title("Locked into mutual defection")
    ax.grid(True, axis="x")
    panel_tag(ax, "f", dx=-0.58)

    fig.suptitle("Temporal structure of play", x=0.02, ha="left",
                 fontweight="bold", color=INK)
    savefig(fig, "F06_round_dynamics")


# --------------------------------------------------------------------------
def fig_outcome_flow(rounds):
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.8))
    for ax, mdl in zip(axes.ravel(), MODEL_ORDER):
        sub = rounds[rounds.model == mdl]
        nmax = int(sub.max_rounds.iloc[0])
        share = (sub.groupby(["round", "outcome"]).size()
                 .unstack(fill_value=0))
        share = share.div(share.sum(axis=1), axis=0).reindex(
            columns=OUTCOME_ORDER, fill_value=0.0)
        ax.stackplot(share.index, *[share[c] for c in OUTCOME_ORDER],
                     colors=[OUTCOME[c] for c in OUTCOME_ORDER],
                     labels=OUTCOME_ORDER, edgecolor=SURFACE, linewidth=0.8)
        ax.set_xlim(1, nmax)
        ax.set_ylim(0, 1)
        ax.set_title(mdl)
        ax.set_xlabel("round")
        ax.grid(False)
        if ax in axes[:, 0]:
            ax.set_ylabel("share of dyads")
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME[c]) for c in OUTCOME_ORDER]
    fig.legend(handles, [OUTCOME_LABEL[c] for c in OUTCOME_ORDER], ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("How the four joint outcomes evolve over a game",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    savefig(fig, "F07_outcome_flow")


# --------------------------------------------------------------------------
def fig_reciprocity(rounds):
    fig = plt.figure(figsize=(11.2, 7.0))
    gs = fig.add_gridspec(2, 3, hspace=0.68, wspace=0.46)

    rec = (rounds.groupby("model").apply(reciprocity, include_groups=False)
           .reindex(MODEL_ORDER))
    rec.to_csv(TABDIR / "T07_reciprocity.csv")

    # (a) forgiveness-retaliation plane --------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#f7f7f5", edgecolor="none"))
    refs = {"AllC": (1, 1), "AllD": (0, 0), "TFT": (1, 0), "WSLS": (1, 0)}
    for name, (px, py) in {"AllC": (1, 1), "AllD": (0, 0), "TFT": (1, 0)}.items():
        ax.plot(px, py, "*", ms=15, color=STRATEGY[name], mec=SURFACE, mew=1.2,
                zorder=4)
        ax.annotate(name, (px, py), textcoords="offset points",
                    xytext=(-4 if px else 8, 10 if py < 0.5 else -16),
                    fontsize=8, color=STRATEGY[name], fontweight="bold")
    for mdl in MODEL_ORDER:
        r = rec.loc[mdl]
        ax.plot(r.p_c_after_c, r.p_c_after_d, "o", ms=11, color=MODEL[mdl],
                mec=SURFACE, mew=1.6, zorder=5)
        ax.annotate(mdl, (r.p_c_after_c, r.p_c_after_d),
                    textcoords="offset points", xytext=(10, -3), fontsize=7,
                    color=MODEL[mdl], fontweight="semibold")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(0.62, 0.66, "no reciprocity\n(behaviour ignores the opponent)",
            fontsize=6.8, color=MUTED, rotation=32, ha="center")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("P(cooperate | opponent cooperated last round)")
    ax.set_ylabel("P(cooperate | opponent defected last round)")
    ax.set_title("Weak reciprocity; inverted for Mistral")
    ax.grid(True, axis="both")
    panel_tag(ax, "a", dx=-0.10)

    # (b) reciprocity strength bars ------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    y = np.arange(len(rec))[::-1]
    ax.barh(y, rec.reciprocity, color=[MODEL[m] for m in rec.index], height=0.6,
            edgecolor=SURFACE, linewidth=1.0)
    for yi, v in zip(y, rec.reciprocity):
        ax.text(v + (0.012 if v >= 0 else -0.012), yi, f"{v:.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7, color=INK2)
    ax.axvline(1.0, color=STRATEGY["TFT"], lw=1.2, ls=(0, (3, 2)))
    ax.text(0.985, len(rec) - 0.45, "TFT ", color=STRATEGY["TFT"], fontsize=7,
            va="top", ha="right", fontweight="semibold")
    ax.axvline(0.0, color=INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(rec.index)
    ax.set_xlim(-0.30, 1.12)
    ax.set_xlabel("P(C|C) − P(C|D)")
    ax.set_title("Reciprocity strength")
    ax.grid(True, axis="x")
    panel_tag(ax, "b", dx=-0.62)

    # (c) memory-one fingerprint heatmap -------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    st = (rounds.groupby("model").apply(state_transitions, include_groups=False)
          .reindex(MODEL_ORDER))
    st.to_csv(TABDIR / "T08_memory1_fingerprints.csv")
    cols = ["pC_R", "pC_S", "pC_T", "pC_P"]
    mat = st[cols].to_numpy()
    ref_names = list(MEMORY1_REFERENCE)
    ref_mat = np.array([MEMORY1_REFERENCE[k] for k in ref_names], dtype=float)
    full = np.vstack([mat, np.full((1, 4), np.nan), ref_mat])
    labels = list(st.index) + [""] + ref_names
    im = ax.imshow(full, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(full.shape[0]):
        for j in range(4):
            v = full[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.55 else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["after R\n(both C)", "after S\n(betrayed)",
                        "after T\n(exploited them)", "after P\n(both D)"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(False)
    ax.set_title("Memory-one fingerprint")
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.outline.set_visible(False)
    panel_tag(ax, "c", dx=-0.20)

    # (d) distance to each canonical strategy --------------------------------
    ax = fig.add_subplot(gs[1, 2])
    d = np.zeros((len(MODEL_ORDER), len(ref_names)))
    for i, mdl in enumerate(MODEL_ORDER):
        v = st.loc[mdl, cols].to_numpy(dtype=float)
        for j, k in enumerate(ref_names):
            d[i, j] = np.sqrt(np.nanmean((v - np.array(MEMORY1_REFERENCE[k])) ** 2))
    im = ax.imshow(d, cmap=CMAP_SEQ.reversed(), aspect="auto")
    for i in range(d.shape[0]):
        j = int(np.argmin(d[i]))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=INK, linewidth=1.6))
        for jj in range(d.shape[1]):
            ax.text(jj, i, f"{d[i, jj]:.2f}", ha="center", va="center", fontsize=6.5,
                    color=INK if d[i, jj] > 0.35 else "white")
    ax.set_xticks(range(len(ref_names)))
    ax.set_xticklabels(ref_names, rotation=30, ha="right")
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([m.split("-")[0] for m in MODEL_ORDER])
    ax.grid(False)
    ax.set_title("RMS distance to canonical rules")
    panel_tag(ax, "d", dx=-0.42)

    fig.suptitle("Conditional behaviour: how models react to what just happened",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F08_reciprocity")
    return rec, st


# --------------------------------------------------------------------------
def fig_payoffs(rounds, games):
    fig = plt.figure(figsize=(11.2, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.68, wspace=0.46)

    # (a) Pareto plane of dyad payoffs ---------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    a1 = games[games.agent == 1].set_index("game_uid")
    a2 = games[games.agent == 2].set_index("game_uid")
    joint = a1[["model", "payoff_per_round"]].join(
        a2[["payoff_per_round"]], rsuffix="_2", how="inner")
    joint = joint.join(a1[["family"]])
    rng = np.random.default_rng(0)
    # only two series here: a scatter puts every pair of colours on screen at
    # once, and the all-pairs CVD floor only clears for a small palette
    for fam, col in (("frontier", C_COOP), ("small", C_DEFECT)):
        s = joint[joint.family == fam]
        j = rng.normal(0, 0.09, (len(s), 2))
        ax.scatter(s.payoff_per_round + j[:, 0], s.payoff_per_round_2 + j[:, 1],
                   s=5, alpha=0.22, color=col, linewidths=0, label=f"{fam} runs")
    for fam, R in (("frontier", 6), ("small", 8)):
        ax.plot(R, R, "*", ms=16, color=INK, mec=SURFACE, mew=1.2, zorder=6)
        ax.annotate(f"mutual C\n{fam}  (R={R})", (R, R),
                    textcoords="offset points", xytext=(9, 4), fontsize=6.6,
                    color=INK2, fontweight="semibold")
    ax.plot(2, 2, "X", ms=12, color=INK, mec=SURFACE, mew=1.2, zorder=6)
    ax.annotate("Nash equilibrium\nmutual D  (P=2)", (2, 2),
                textcoords="offset points", xytext=(11, -20), fontsize=6.6,
                color=INK2, fontweight="semibold")
    ax.plot([0, 10], [10, 0], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.set_xlabel("agent 1 mean payoff per round (base units)")
    ax.set_ylabel("agent 2 mean payoff")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Where dyads land")
    ax.legend(ncol=2, fontsize=7, loc="upper right", markerscale=3.0,
              framealpha=0.0)
    ax.grid(True, axis="both")
    panel_tag(ax, "a", dx=-0.10)

    # (b) efficiency vs cooperation ------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    agg = games.groupby(["model", "scale_nominal"])[["coop_rate", "efficiency"]].mean()
    for mdl in MODEL_ORDER:
        s = agg.loc[mdl]
        ax.plot(s.coop_rate, s.efficiency, "-o", color=MODEL[mdl], ms=5,
                mec=SURFACE, mew=1.1, alpha=0.9)
    ax.set_xlabel("cooperation rate")
    ax.set_ylabel("payoff efficiency")
    ax.set_title("More cooperation, more payoff")
    panel_tag(ax, "b", dx=-0.36)

    # (c) inequality within a dyad -------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    joint["gap"] = (joint.payoff_per_round - joint.payoff_per_round_2).abs()
    parts = [joint.loc[joint.model == m, "gap"].to_numpy() for m in MODEL_ORDER]
    vp = ax.violinplot(parts, showextrema=False, widths=0.85)
    for b, mdl in zip(vp["bodies"], MODEL_ORDER):
        b.set_facecolor(MODEL[mdl])
        b.set_alpha(0.75)
        b.set_edgecolor(SURFACE)
        b.set_linewidth(1.0)
    for i, p in enumerate(parts):
        ax.plot(i + 1, np.median(p), "o", ms=5, color=INK, mec=SURFACE, mew=1.0)
    ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
    ax.set_xticklabels([m.split("-")[0] for m in MODEL_ORDER], rotation=20,
                       ha="right")
    ax.set_ylabel("|payoff₁ − payoff₂| per round")
    ax.set_title("Within-dyad inequality")
    panel_tag(ax, "c", dx=-0.30)

    # (d) who wins: cooperative or selfish persona? --------------------------
    ax = fig.add_subplot(gs[1, 1])
    mixed = games[games.dyad.isin(["CvS", "SvC"])]
    tab = mixed.groupby(["model", "personality"]).payoff_per_round.mean().unstack()
    tab = tab.reindex(MODEL_ORDER)
    x = np.arange(len(tab))
    for k, pers in enumerate(["cooperative", "selfish"]):
        ax.bar(x + (k - 0.5) * 0.36, tab[pers], width=0.34,
               color=PERSONALITY[pers], edgecolor=SURFACE, linewidth=1.0,
               label=f"{pers} persona")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("-")[0] for m in tab.index], rotation=20, ha="right")
    ax.set_ylabel("mean payoff per round")
    ax.set_title("Mixed dyads")
    ax.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    panel_tag(ax, "d", dx=-0.30)

    # (e) price of anarchy ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    poa = games.groupby(["model", "scale_nominal"]).efficiency.mean().unstack()
    poa = poa.reindex(MODEL_ORDER)
    im = ax.imshow(poa.to_numpy(), cmap=CMAP_SEQ, aspect="auto", vmin=0.45, vmax=0.9)
    for i in range(poa.shape[0]):
        for j in range(poa.shape[1]):
            v = poa.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                        color="white" if v > 0.68 else INK)
    ax.set_xticks(range(poa.shape[1]))
    ax.set_xticklabels([f"×{c:g}" for c in poa.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(poa)))
    ax.set_yticklabels([m.split("-")[0] for m in poa.index])
    ax.grid(False)
    ax.set_title("Efficiency (1.0 = all-CC)")
    panel_tag(ax, "e", dx=-0.38)

    fig.suptitle("Payoff consequences: efficiency, inequality, and who profits",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F09_payoffs")


def main():
    rounds = pd.read_parquet(DATADIR / "rounds.parquet")
    games = pd.read_parquet(DATADIR / "games.parquet")
    fig_round_dynamics(rounds)
    fig_outcome_flow(rounds)
    rec, st = fig_reciprocity(rounds)
    fig_payoffs(rounds, games)
    print(rec.round(3).to_string())
    print()
    print(st.round(3).to_string())


if __name__ == "__main__":
    main()
