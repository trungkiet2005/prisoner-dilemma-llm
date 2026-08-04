"""FR5-FR7: persona steerability, within-game dynamics, reciprocity.

FR7 is the only three-panel figure in the suite; the third panel carries the
distance-to-archetype read-out that the first two set up, and splitting it out
would leave two figures that neither stands on its own.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdlib.ingest import cluster_bootstrap_ci
from pdlib.metrics import cohens_d, grouped_ci, reciprocity, state_transitions
from pdlib.natstyle import (DATADIR, DYAD_LABEL, DYAD_ORDER, FRONTIER, INK,
                            INK2, MEMORY1, MODEL_C, MODEL_LABEL, MODEL_M,
                            MUTED, PAGE, PERSONALITY_C, PERSONALITY_ORDER,
                            RULE, SPINE, TABDIR, W2, bars, caption, figure,
                            finalize, hgrid, model_legend, model_line, refline,
                            save, shared_model_legend, use_journal_style)

use_journal_style()

SEED = 0
N_BOOT = 2000


def _cluster_diff_ci(g: pd.DataFrame, flag: str, value: str = "coop_rate",
                     n_boot: int = N_BOOT, seed: int = SEED):
    """Mean(value | flag == 'cooperative') - mean(value | 'selfish'), with a
    bootstrap that resamples whole dyads.

    The two agent-rows of a dyad share a game history, so resampling rows
    would understate the interval by roughly sqrt(2).
    """
    d = g.sort_values(["game_uid", "agent"])
    uids = d.game_uid.to_numpy()
    n_per = pd.Series(uids).value_counts()
    if n_per.nunique() != 1:
        raise ValueError("ragged clusters; the reshape below assumes a fixed size")
    k = int(n_per.iloc[0])
    val = d[value].to_numpy().reshape(-1, k)
    is_c = (d[flag].to_numpy() == "cooperative").reshape(-1, k)
    obs = val[is_c].mean() - val[~is_c].mean()

    rng = np.random.default_rng(seed)
    n = val.shape[0]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        j = rng.integers(0, n, n)
        v, c = val[j], is_c[j]
        boots[i] = v[c].mean() - v[~c].mean() if c.any() and (~c).any() else np.nan
    return obs, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


# ==========================================================================
# FR5 -- does the assigned persona steer behaviour?
# ==========================================================================
def fig_persona(games):
    fig = figure(W2, 2.55)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0])

    # (a) cooperation by the four persona pairings ---------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    ci = grouped_ci(games, ["model", "dyad"], "coop_rate", n_boot=N_BOOT)
    ci.to_csv(TABDIR / "T_FR09_persona_cells.csv", index=False)

    x = np.arange(len(DYAD_ORDER))
    w = 0.19
    for k, mdl in enumerate(FRONTIER):
        sub = ci[ci.model == mdl].set_index("dyad").reindex(DYAD_ORDER)
        xs = x + (k - 1.5) * w
        bars(ax, xs, sub["mean"], MODEL_C[mdl], width=w * 0.88)
        ax.vlines(xs, sub["lo"], sub["hi"], color=INK, lw=0.6, zorder=5)
    refline(ax, 0.5, "indifference")
    lab = {"C": "cooperative", "S": "selfish"}
    ax.set_xticks(x)
    ax.set_xticklabels([f"own {lab[d[0]]}\nopp. {lab[d[2]]}" for d in DYAD_ORDER],
                       fontsize=6.0, linespacing=1.5)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("cooperation rate")
    ax.set_ylim(0, 0.85)
    ax.set_xlim(-0.55, len(x) - 0.45)
    model_legend(ax, ncol=4, loc="upper center", bbox=(0.5, 1.03))
    ax.set_title("Cooperation by persona pairing", pad=16)

    # (b) how much of that is the prompt? ------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax, axis="x")
    rows = []
    for mdl in FRONTIER:
        g = games[games.model == mdl]
        own, own_lo, own_hi = _cluster_diff_ci(g, "personality")
        opp, opp_lo, opp_hi = _cluster_diff_ci(g, "opp_personality")
        rows.append({
            "model": mdl,
            "own_effect": own, "own_lo": own_lo, "own_hi": own_hi,
            "own_d": cohens_d(g[g.personality == "cooperative"].coop_rate,
                              g[g.personality == "selfish"].coop_rate),
            "opp_effect": opp, "opp_lo": opp_lo, "opp_hi": opp_hi,
            "opp_d": cohens_d(g[g.opp_personality == "cooperative"].coop_rate,
                              g[g.opp_personality == "selfish"].coop_rate)})
    eff = pd.DataFrame(rows)
    eff.to_csv(TABDIR / "T_FR10_persona_effects.csv", index=False)

    y = np.arange(len(eff))[::-1].astype(float)
    ax.axvline(0, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    off = 0.17
    for yi, r in zip(y, eff.itertuples()):
        c = MODEL_C[r.model]
        ax.hlines(yi + off, r.own_lo, r.own_hi, color=c, lw=0.8, zorder=3)
        ax.plot(r.own_effect, yi + off, marker=MODEL_M[r.model], ms=4.4,
                mfc=c, mec=PAGE, mew=0.6, ls="none", zorder=4)
        ax.hlines(yi - off, r.opp_lo, r.opp_hi, color=c, lw=0.8, zorder=3)
        ax.plot(r.opp_effect, yi - off, marker=MODEL_M[r.model], ms=4.4,
                mfc=PAGE, mec=c, mew=1.0, ls="none", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in eff.model],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1.05, len(eff) - 0.30)
    ax.set_xlabel("cooperative $-$ selfish persona\n($\\Delta$ cooperation rate)")
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=4.0, mfc=MUTED,
                          mec=PAGE, mew=0.6, label="own persona (stated)"),
               plt.Line2D([], [], ls="none", marker="o", ms=4.0, mfc=PAGE,
                          mec=MUTED, mew=1.0, label="opponent's (hidden)")]
    ax.legend(handles=handles, ncol=1, loc="upper right",
              bbox_to_anchor=(1.0, 1.02), handlelength=0.8)
    ax.set_title("Steerability, 95% CI", pad=16)

    finalize(fig, [pa, pb], ["a", "b"])
    caption(fig, "Persona is stated in the system prompt; the opponent's persona "
                 "is never disclosed, so a non-zero open marker in b would mean "
                 "the pairing leaked through play rather than through the prompt. "
                 "Intervals in a are 95% bootstrap CIs over dyads.")
    save(fig, "FR5_persona")
    return eff


# ==========================================================================
# FR6 -- what happens inside a game
# ==========================================================================
def fig_rounds(rounds, games):
    fig = figure(W2, 2.5)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])

    # (a) cooperation by round ------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    rows = []
    for mdl in FRONTIER:
        sub = rounds[rounds.model == mdl]
        for t, d in sub.groupby("round"):
            lo, hi = cluster_bootstrap_ci(d, "coop", n_boot=800, seed=SEED)
            rows.append({"model": mdl, "round": t, "mean": d.coop.mean(),
                         "lo": lo, "hi": hi})
    tr = pd.DataFrame(rows)
    tr.to_csv(TABDIR / "T_FR11_round_profile.csv", index=False)
    for mdl in FRONTIER:
        s = tr[tr.model == mdl].sort_values("round")
        model_line(ax, s["round"], s["mean"], mdl, lo=s["lo"], hi=s["hi"])
    refline(ax, 0.5, "indifference")
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0.05, 0.85)
    ax.set_xlabel("round")
    ax.set_ylabel("cooperation rate")
    model_legend(ax, ncol=2, loc="upper center", bbox=(0.5, 1.02))
    ax.set_title("Trajectories diverge after the opening", pad=16)

    # (b) opening vs closing move --------------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    ends = (games.groupby("model")
            .agg(first=("first_move_coop", "mean"),
                 last=("last_move_coop", "mean"))
            .reindex(FRONTIER))
    # signed, so the direction is readable without consulting the sign column
    ends["delta_last_minus_first"] = ends["last"] - ends["first"]
    ends.to_csv(TABDIR / "T_FR12_endgame.csv")

    x = np.arange(len(ends))
    for xi, mdl in zip(x, FRONTIER):
        f, l = ends.loc[mdl, "first"], ends.loc[mdl, "last"]
        ax.plot([xi, xi], [f, l], color=MODEL_C[mdl], lw=1.4, alpha=0.5,
                solid_capstyle="round", zorder=2)
        ax.plot(xi, f, marker=MODEL_M[mdl], ms=4.6, mfc=MODEL_C[mdl], mec=PAGE,
                mew=0.6, ls="none", zorder=4)
        ax.plot(xi, l, marker=MODEL_M[mdl], ms=4.6, mfc=PAGE, mec=MODEL_C[mdl],
                mew=1.0, ls="none", zorder=4)
        ax.text(xi + 0.16, (f + l) / 2,
                ("$-$" if l < f else "$+$") + f"{abs(l - f):.2f}", ha="left",
                va="center", fontsize=5.8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in FRONTIER],
                       fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("cooperation rate")
    ax.set_ylim(0, 0.95)
    ax.set_xlim(-0.55, len(x) - 0.20)
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=4.2, mfc=MUTED,
                          mec=PAGE, mew=0.6, label="round 1 (opening)"),
               plt.Line2D([], [], ls="none", marker="o", ms=4.2, mfc=PAGE,
                          mec=MUTED, mew=1.0, label="round 10 (final)")]
    ax.legend(handles=handles, ncol=1, loc="upper center",
              bbox_to_anchor=(0.5, 1.03), handlelength=0.8)
    ax.set_title("Opening versus final round", pad=16)

    finalize(fig, [pa, pb], ["a", "b"])
    caption(fig, "Backward induction from a finite horizon predicts decay towards "
                 "mutual defection. Only Claude does that ($-$0.27 from round 1 to "
                 "round 10); GPT-4o, Mistral and Gemini all end more cooperative "
                 "than they start. Gemini is the one arm that was told the round "
                 "count, and it warms up rather than unravelling, so the pattern "
                 "is not driven by horizon knowledge.")
    save(fig, "FR6_round_dynamics")
    return ends


# ==========================================================================
# FR7 -- the memory-one fingerprint
# ==========================================================================
STATE_ORDER = ["R", "S", "T", "P"]
STATE_LABEL = {"R": "after CC\n($R$)", "S": "after CD\n($S$)",
               "T": "after DC\n($T$)", "P": "after DD\n($P$)"}


def fig_reciprocity(rounds):
    fig = figure(W2, 2.6)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.05, 0.85])

    # (a) P(cooperate | previous joint outcome) ------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    fp = {}
    for mdl in FRONTIER:
        fp[mdl] = state_transitions(rounds[rounds.model == mdl])
    fpd = pd.DataFrame(fp).T
    fpd.to_csv(TABDIR / "T_FR13_memory1_fingerprint.csv")

    x = np.arange(4)
    w = 0.19
    for k, mdl in enumerate(FRONTIER):
        v = [fpd.loc[mdl, f"pC_{s}"] for s in STATE_ORDER]
        bars(ax, x + (k - 1.5) * w, v, MODEL_C[mdl], width=w * 0.88)
    for s_i, s in enumerate(STATE_ORDER):
        tft = MEMORY1["TFT"][s_i]
        ax.hlines(tft, s_i - 0.44, s_i + 0.44, color=INK, lw=0.7,
                  ls=(0, (2, 1.6)), zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABEL[s] for s in STATE_ORDER], fontsize=6.0,
                       linespacing=1.35)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("P(cooperate next round)")
    ax.set_ylim(0, 1.06)
    ax.set_xlim(-0.55, 3.55)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.plot([], [], color=INK, lw=0.7, ls=(0, (2, 1.6)),
            label="tit-for-tat reference")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), handlelength=1.6)
    ax.set_title("Memory-one fingerprint", pad=14)

    # (b) niceness-reciprocity plane -----------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    rec = {}
    for mdl in FRONTIER:
        rec[mdl] = reciprocity(rounds[rounds.model == mdl])
    recd = pd.DataFrame(rec).T
    recd.to_csv(TABDIR / "T_FR14_reciprocity.csv")

    # (niceness, reciprocity) of the canonical strategies.  TFT and GRIM share
    # a corner in this projection, so they are drawn once with a joint label.
    anchors = {"AllD": (0.0, 0.0, 3, -9), "AllC": (1.0, 0.0, -4, -9),
               "TFT / GRIM": (1.0, 1.0, -4, 4)}
    for name, (nx, ny, ox, oy) in anchors.items():
        ax.plot(nx, ny, marker="*", ms=6.5, mfc=RULE, mec=SPINE, mew=0.5,
                ls="none", zorder=2)
        ax.annotate(name, (nx, ny), textcoords="offset points",
                    xytext=(ox, oy), ha="right" if ox < 0 else "left",
                    fontsize=5.8, color=MUTED)
    for mdl in FRONTIER:
        ax.plot(recd.loc[mdl, "nice"], recd.loc[mdl, "reciprocity"],
                marker=MODEL_M[mdl], ms=5.0, mfc=MODEL_C[mdl], mec=PAGE,
                mew=0.7, ls="none", zorder=5)
    ax.set_xlabel("niceness  P(C) on round 1")
    ax.set_ylabel("reciprocity  P(C|C$_{-1}$) $-$ P(C|D$_{-1}$)")
    ax.set_xlim(-0.10, 1.14)
    ax.set_ylim(-0.24, 1.12)
    ax.axhline(0.0, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    ax.text(0.16, 0.012, "no reciprocity", ha="left", va="bottom", fontsize=5.6,
            color=MUTED)
    ax.set_title("Against the canonical strategies", pad=14)

    # (c) distance to the nearest memory-one archetype ------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax, axis="x")
    rows = []
    for mdl in FRONTIER:
        v = np.array([fpd.loc[mdl, f"pC_{s}"] for s in STATE_ORDER])
        d = {name: float(np.linalg.norm(v - np.array(ref)))
             for name, ref in MEMORY1.items()}
        best = min(d, key=d.get)
        rows.append({"model": mdl, "nearest": best, "distance": d[best],
                     **{f"d_{k}": val for k, val in d.items()}})
    near = pd.DataFrame(rows)
    near.to_csv(TABDIR / "T_FR15_archetype_distance.csv", index=False)

    y = np.arange(len(near))[::-1]
    for yi, r in zip(y, near.itertuples()):
        ax.barh(yi, r.distance, height=0.5, color=MODEL_C[r.model],
                edgecolor=PAGE, linewidth=0.5, zorder=3)
        ax.text(r.distance + 0.015, yi, r.nearest, ha="left", va="center",
                fontsize=6.0, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].split()[0] for m in near.model])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(near) - 0.3)
    ax.set_xlim(0, near.distance.max() * 1.55)
    ax.set_xlabel("Euclidean distance in\nfingerprint space")
    ax.set_title("Nearest archetype", pad=14)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"])
    leg = shared_model_legend(fig, [pa, pb, pc], ncol=4, lines=False)
    caption(fig, "The four conditional probabilities in a are the complete "
                 "description of a memory-one strategy; the dashes mark "
                 "tit-for-tat, which is 1 after CC and DC and 0 after CD and DD. "
                 "Panel c gives the distance from the closest of AllC, AllD, TFT, "
                 "WSLS and GRIM in that four-dimensional space: all four models "
                 "sit 0.7 to 1.0 away, so no memory-one strategy describes their "
                 "play.", below=leg)
    save(fig, "FR7_reciprocity")
    return fpd, recd, near


def main():
    games = pd.read_parquet(DATADIR / "frontier_games.parquet")
    rounds = pd.read_parquet(DATADIR / "frontier_rounds.parquet")
    eff = fig_persona(games)
    ends = fig_rounds(rounds, games)
    fpd, recd, near = fig_reciprocity(rounds)
    print()
    print(eff.to_string(index=False))
    print()
    print(ends.to_string())
    print()
    print(fpd[[f"pC_{s}" for s in STATE_ORDER]].to_string())
    print()
    print(near[["model", "nearest", "distance"]].to_string(index=False))


if __name__ == "__main__":
    main()
