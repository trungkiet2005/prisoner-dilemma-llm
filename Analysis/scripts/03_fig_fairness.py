"""F04 -- language fairness, F05 -- personality conditioning, F06 -- first-mover.

FAIRGAME's premise is that a fair agent should play the *same* game the same
way regardless of the language it is prompted in or the persona label attached
to it.  These panels quantify how far from that the six models are.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdlib.metrics import (cohens_d, fairness_gap, grouped_ci,
                           permutation_gap_p)
from pdlib.style import (CMAP_DIV, CMAP_SEQ, C_COOP, C_DEFECT, DATADIR, GRID,
                         INK, INK2, LANG_LABEL, LANG_ORDER, MODEL, MODEL_ORDER,
                         MUTED, PERSONALITY, SURFACE, TABDIR, panel_tag,
                         savefig, use_paper_style)

use_paper_style()


# --------------------------------------------------------------------------
def fig_language(games):
    fig = plt.figure(figsize=(11.2, 7.0))
    gs = fig.add_gridspec(2, 3, hspace=0.70, wspace=0.44,
                          height_ratios=[1.15, 1])

    # (a) model x language heatmap ------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    piv = (games.pivot_table(index="model", columns="language", values="coop_rate")
           .reindex(index=MODEL_ORDER, columns=LANG_ORDER))
    im = ax.imshow(piv.to_numpy(), cmap=CMAP_SEQ, vmin=0.25, vmax=0.7, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.50 else INK)
    ax.set_xticks(range(len(LANG_ORDER)))
    ax.set_xticklabels([LANG_LABEL[l] for l in LANG_ORDER])
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.grid(False)
    ax.set_title("Cooperation rate by prompt language")
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.outline.set_visible(False)
    panel_tag(ax, "a", dx=-0.19)

    # (b) deviation from each model's own mean (diverging) -------------------
    ax = fig.add_subplot(gs[0, 2])
    dev = piv.sub(piv.mean(axis=1), axis=0)
    m = np.nanmax(np.abs(dev.to_numpy()))
    im = ax.imshow(dev.to_numpy(), cmap=CMAP_DIV, vmin=-m, vmax=m, aspect="auto")
    for i in range(dev.shape[0]):
        for j in range(dev.shape[1]):
            v = dev.iat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.62 * m else INK)
    ax.set_xticks(range(len(LANG_ORDER)))
    ax.set_xticklabels([l.upper() for l in LANG_ORDER])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Deviation from the model's own mean")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.outline.set_visible(False)
    panel_tag(ax, "b", dx=-0.14)

    # (c) disparity per model with permutation p ----------------------------
    ax = fig.add_subplot(gs[1, 0])
    rows = []
    for mdl in MODEL_ORDER:
        sub = games[games.model == mdl]
        gap, p = permutation_gap_p(sub, "language", "coop_rate", n_perm=2000)
        gapp, pp = permutation_gap_p(sub, "personality", "coop_rate", n_perm=2000)
        rows.append({"model": mdl, "language_gap": gap, "language_p": p,
                     "personality_gap": gapp, "personality_p": pp})
    fair = pd.DataFrame(rows)
    fair.to_csv(TABDIR / "T05_fairness_gaps.csv", index=False)
    y = np.arange(len(fair))[::-1]
    b = ax.barh(y, fair.language_gap, color=[MODEL[m] for m in fair.model],
                edgecolor=SURFACE, linewidth=1.0, height=0.62)
    for yi, r in zip(y, fair.itertuples()):
        star = "∗∗" if r.language_p < 0.01 else ("∗" if r.language_p < 0.05 else "n.s.")
        ax.text(r.language_gap + 0.006, yi, f"{r.language_gap:.2f} {star}",
                va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(fair.model)
    ax.set_xlim(0, fair.language_gap.max() * 1.42)
    ax.set_xlabel("max − min cooperation across languages")
    ax.set_title("Language disparity")
    ax.grid(True, axis="x")
    panel_tag(ax, "c", dx=-0.55)

    # (d) per-language profile lines ----------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ci = grouped_ci(games, ["model", "language"], "coop_rate", n_boot=500)
    for mdl in MODEL_ORDER:
        sub = ci[ci.model == mdl].set_index("language").reindex(LANG_ORDER)
        ax.plot(range(len(LANG_ORDER)), sub["mean"], "-o", color=MODEL[mdl], ms=4,
                mec=SURFACE, mew=1.0, label=mdl)
    ax.set_xticks(range(len(LANG_ORDER)))
    ax.set_xticklabels([l.upper() for l in LANG_ORDER])
    ax.set_ylabel("cooperation rate")
    ax.set_title("Profiles do not rank alike")
    ax.legend(ncol=3, fontsize=5.6, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    panel_tag(ax, "d", dx=-0.30)

    # (e) does the language effect survive at every scale? ------------------
    ax = fig.add_subplot(gs[1, 2])
    gp = (games.groupby(["model", "scale_nominal"])
          .apply(lambda d: fairness_gap(d, "language", "coop_rate"),
                 include_groups=False)
          .rename("gap").reset_index())
    for mdl in MODEL_ORDER:
        sub = gp[gp.model == mdl].sort_values("scale_nominal")
        ax.plot(np.log10(sub.scale_nominal), sub.gap, "-o", color=MODEL[mdl],
                ms=4, mec=SURFACE, mew=1.0)
    ax.set_xticks([-2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1k"])
    ax.set_xlabel("payoff scale")
    ax.set_ylabel("language disparity")
    ax.set_title("Disparity at every stake")
    panel_tag(ax, "e", dx=-0.32)

    fig.suptitle("Language changes how the same game is played", x=0.02,
                 ha="left", fontweight="bold", color=INK)
    savefig(fig, "F04_language_fairness")
    return fair


# --------------------------------------------------------------------------
def fig_personality(games):
    fig = plt.figure(figsize=(11.2, 6.8))
    gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.44)

    dyads = ["CvC", "CvS", "SvC", "SvS"]
    dyad_lbl = ["coop\nvs coop", "coop\nvs selfish", "selfish\nvs coop",
                "selfish\nvs selfish"]

    # (a) coop rate per model per dyad --------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    ci = grouped_ci(games, ["model", "dyad"], "coop_rate", n_boot=600)
    w = 0.13
    for k, mdl in enumerate(MODEL_ORDER):
        sub = ci[ci.model == mdl].set_index("dyad").reindex(dyads)
        x = np.arange(len(dyads)) + (k - 2.5) * w
        ax.bar(x, sub["mean"], width=w * 0.92, color=MODEL[mdl],
               edgecolor=SURFACE, linewidth=1.0, label=mdl, zorder=3)
        ax.vlines(x, sub["lo"], sub["hi"], color=INK, lw=1.0, zorder=4)
    ax.set_xticks(range(len(dyads)))
    ax.set_xticklabels(dyad_lbl)
    ax.set_ylabel("cooperation rate of the focal agent")
    ax.set_ylim(0, 1)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=7)
    ax.set_title("Own label and opponent label both move behaviour")
    panel_tag(ax, "a", dx=-0.085)

    # (b) own vs opponent persona effect size -------------------------------
    ax = fig.add_subplot(gs[0, 2])
    rows = []
    for mdl in MODEL_ORDER:
        s = games[games.model == mdl]
        d_own = cohens_d(s.loc[s.personality == "cooperative", "coop_rate"],
                         s.loc[s.personality == "selfish", "coop_rate"])
        d_opp = cohens_d(s.loc[s.opp_personality == "cooperative", "coop_rate"],
                         s.loc[s.opp_personality == "selfish", "coop_rate"])
        rows.append({"model": mdl, "own": d_own, "opponent": d_opp})
    eff = pd.DataFrame(rows)
    eff.to_csv(TABDIR / "T06_personality_effects.csv", index=False)
    for r in eff.itertuples():
        ax.plot(r.own, r.opponent, "o", ms=9, color=MODEL[r.model], mec=SURFACE,
                mew=1.5)
        ax.annotate(r.model.split("-")[0], (r.own, r.opponent),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=6.5, color=MODEL[r.model], fontweight="semibold")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("own-persona effect (Cohen's d)")
    ax.set_ylabel("opponent-persona effect")
    ax.margins(0.16)
    ax.set_title("Whose label matters?")
    ax.grid(True, axis="both")
    panel_tag(ax, "b", dx=-0.32)

    # (c) exploitation asymmetry: what a cooperative agent earns -------------
    ax = fig.add_subplot(gs[1, 0])
    sub = (games.groupby(["model", "dyad"]).efficiency.mean().unstack()
           .reindex(index=MODEL_ORDER, columns=dyads))
    im = ax.imshow(sub.to_numpy(), cmap=CMAP_SEQ, aspect="auto")
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > np.nanmean(sub.to_numpy()) else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["CvC", "CvS", "SvC", "SvS"])
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.grid(False)
    ax.set_title("Payoff efficiency by dyad")
    panel_tag(ax, "c", dx=-0.60)

    # (d) persona x scale interaction ---------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    # the two families use different payoff matrices, so they are drawn as
    # separate lines rather than pooled
    for fam, ls, mk in (("frontier", "-", "o"), ("small", (0, (3, 2)), "s")):
        for pers in ("cooperative", "selfish"):
            sub = (games[(games.personality == pers) & (games.family == fam)]
                   .groupby("scale_nominal").coop_rate.mean())
            ax.plot(np.log10(sub.index.to_numpy()), sub.to_numpy(),
                    linestyle=ls, color=PERSONALITY[pers], marker=mk, ms=4.5,
                    mec=SURFACE, mew=1.1,
                    label=f"{pers}, {fam}")
    ax.set_xticks([-2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1k"])
    ax.set_xlabel("payoff scale")
    ax.set_ylabel("cooperation rate")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=6.0,
              ncol=2)
    ax.set_title("Persona gap at every stake")
    panel_tag(ax, "d", dx=-0.30)

    # (e) is the persona instruction obeyed at all? -------------------------
    ax = fig.add_subplot(gs[1, 2])
    obey = (games.groupby(["model", "personality"]).coop_rate.mean().unstack())
    obey = obey.reindex(MODEL_ORDER)
    y = np.arange(len(obey))[::-1]
    for yi, mdl in zip(y, obey.index):
        c, s = obey.loc[mdl, "cooperative"], obey.loc[mdl, "selfish"]
        ax.plot([s, c], [yi, yi], color=MODEL[mdl], lw=2.6, alpha=0.55,
                solid_capstyle="round")
        ax.plot(s, yi, "o", ms=7, color=PERSONALITY["selfish"], mec=SURFACE, mew=1.3,
                label="selfish persona" if yi == y[0] else None)
        ax.plot(c, yi, "o", ms=7, color=PERSONALITY["cooperative"], mec=SURFACE, mew=1.3,
                label="cooperative persona" if yi == y[0] else None)
        ax.text(max(c, s) + 0.012, yi, f"Δ={c - s:+.2f}", va="center", fontsize=7,
                color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(obey.index)
    ax.set_xlim(0.2, 0.85)
    ax.set_xlabel("cooperation rate")
    ax.set_title("Persona compliance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=6.5)
    ax.grid(True, axis="x")
    panel_tag(ax, "e", dx=-0.60)

    fig.suptitle("Persona labels steer play, but never fully determine it",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F05_personality")
    return eff


def main():
    games = pd.read_parquet(DATADIR / "games.parquet")
    fair = fig_language(games)
    eff = fig_personality(games)
    print(fair.to_string(index=False))
    print()
    print(eff.to_string(index=False))


if __name__ == "__main__":
    main()
