"""F21 -- variance decomposition, F22 -- game-level distributions,
F23 -- language x scale small multiples, F24 -- openings and replicate noise."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pdlib.ingest import wilson
from pdlib.style import (CMAP_DIV, CMAP_SEQ, C_COOP, C_DEFECT, DATADIR, INK,
                         INK2, LANG_LABEL, LANG_ORDER, MODEL, MODEL_ORDER,
                         MUTED, PERSONALITY, SURFACE, TABDIR, panel_tag,
                         savefig, use_paper_style)

use_paper_style()

FACTORS = ["model", "language", "personality", "opp_personality", "scale_f"]
FACTOR_LABEL = {
    "model": "which model",
    "language": "prompt language",
    "personality": "own persona",
    "opp_personality": "opponent persona",
    "scale_f": "payoff scale",
}


# --------------------------------------------------------------------------
def fig_variance(games):
    """How much of the variation in cooperation each design factor explains.

    A partial eta-squared from a main-effects ANOVA on the game-level
    cooperation rate.  Under the game-theoretic reading only the payoff
    structure should matter, and the payoff *scale* is not part of that
    structure -- so every bar here is variation a strategically consistent
    agent would not produce.
    """
    g = games.copy()
    g["scale_f"] = g.scale_nominal.astype(str)

    fig = plt.figure(figsize=(11.2, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1.1, 1, 1])

    rows = []
    for scope, sub in (("all runs", g),
                       ("frontier", g[g.family == "frontier"]),
                       ("open-weight", g[g.family == "small"])):
        formula = "coop_rate ~ " + " + ".join(f"C({f})" for f in FACTORS)
        mdl = smf.ols(formula, data=sub).fit()
        aov = sm.stats.anova_lm(mdl, typ=2)
        ss_total = aov["sum_sq"].sum()
        for f in FACTORS:
            key = f"C({f})"
            if key in aov.index:
                rows.append({"scope": scope, "factor": f,
                             "eta2": aov.loc[key, "sum_sq"] / ss_total,
                             "p": aov.loc[key, "PR(>F)"]})
        rows.append({"scope": scope, "factor": "residual",
                     "eta2": aov.loc["Residual", "sum_sq"] / ss_total,
                     "p": np.nan})
    var = pd.DataFrame(rows)
    var.to_csv(TABDIR / "T16_variance_decomposition.csv", index=False)

    ax = fig.add_subplot(gs[0, 0])
    order = FACTORS
    scopes = ["all runs", "frontier", "open-weight"]
    w = 0.26
    palette = [C_COOP, "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
    for k, sc in enumerate(scopes):
        sub = var[var.scope == sc].set_index("factor").reindex(order)
        x = np.arange(len(order)) + (k - 1) * w
        ax.bar(x, sub.eta2, width=w * 0.9,
               color=palette, alpha=[1.0, 0.72, 0.45][k],
               edgecolor=SURFACE, linewidth=0.9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([FACTOR_LABEL[f] for f in order], rotation=32,
                       ha="right", fontsize=7)
    ax.set_ylabel("share of total variance")
    res = var[var.factor == "residual"].set_index("scope").eta2
    ax.set_title("Variance decomposition\n"
                 "bar triplet = all runs · frontier · open-weight",
                 fontsize=8.6)
    ax.text(0.98, 0.96, "residual (run-to-run):\n"
            + "  ·  ".join(f"{k.split()[0]} {v:.2f}" for k, v in res.items()),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.4,
            color=INK2)
    panel_tag(ax, "a", dx=-0.24, dy=1.12)

    # (b) same, excluding the residual, as a share of explained variance ----
    ax = fig.add_subplot(gs[0, 1])
    expl = var[var.factor != "residual"].copy()
    expl["share"] = expl.groupby("scope").eta2.transform(lambda s: s / s.sum())
    piv = expl.pivot(index="scope", columns="factor", values="share") \
              .reindex(index=scopes, columns=FACTORS)
    bottom = np.zeros(len(piv))
    for f, col in zip(FACTORS, palette):
        v = piv[f].to_numpy()
        ax.bar(np.arange(len(piv)), v, bottom=bottom, color=col,
               edgecolor=SURFACE, linewidth=1.2, width=0.62,
               label=FACTOR_LABEL[f])
        for xi, (b, q) in enumerate(zip(bottom, v)):
            if q > 0.07:
                ax.text(xi, b + q / 2, f"{q:.2f}", ha="center", va="center",
                        fontsize=6.4, color="white", fontweight="semibold")
        bottom = bottom + v
    ax.set_xticks(range(len(piv)))
    ax.set_xticklabels(piv.index, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of explained variance")
    ax.grid(False)
    ax.legend(ncol=2, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    ax.set_title("Composition of the explained part")
    panel_tag(ax, "b", dx=-0.26)

    # (c) irreducible run-to-run noise ---------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    cell = (g.groupby(["model", "language", "scale_nominal", "dyad"])
            .coop_rate.agg(["mean", "std", "size"]).reset_index())
    for mdl in MODEL_ORDER:
        s = cell[cell.model == mdl]
        ax.scatter(s["mean"], s["std"], s=7, alpha=0.5, color=MODEL[mdl],
                   linewidths=0, label=mdl)
    p = np.linspace(0.001, 0.999, 200)
    ax.plot(p, np.sqrt(p * (1 - p)), color=INK, lw=1.4, ls=(0, (4, 3)),
            label="Bernoulli(p) upper bound")
    ax.set_xlabel("mean cooperation rate of the condition")
    ax.set_ylabel("s.d. across replicate games")
    ax.set_title("Replicate-to-replicate spread")
    ax.set_ylim(-0.02, 0.58)
    ax.legend(ncol=2, fontsize=5.8, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), markerscale=1.8)
    ax.grid(True, axis="both")
    panel_tag(ax, "c", dx=-0.26)

    fig.suptitle("What actually drives the variation in cooperation?", x=0.02,
                 ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig(fig, "F21_variance_decomposition")
    return var


# --------------------------------------------------------------------------
def fig_distributions(games):
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.8), sharex=True)
    bins = np.linspace(0, 1, 21)
    for ax, mdl in zip(axes.ravel(), MODEL_ORDER):
        s = games[games.model == mdl]
        for pers, ls in (("cooperative", "-"), ("selfish", (0, (3, 2)))):
            v = s.loc[s.personality == pers, "coop_rate"]
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.9,
                    linestyle=ls, color=PERSONALITY[pers], label=f"{pers} persona")
        ax.axvline(s.coop_rate.mean(), color=MODEL[mdl], lw=2.0)
        ax.set_title(mdl)
        ax.set_xlim(0, 1)
        if ax in axes[1]:
            ax.set_xlabel("cooperation rate within a game")
        if ax in axes[:, 0]:
            ax.set_ylabel("density")
    axes[0, 0].legend(fontsize=6.5, loc="upper center")
    fig.suptitle("Game-level cooperation is bimodal: dyads mostly settle at an "
                 "extreme, not in the middle", x=0.02, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, "F22_game_distributions")


# --------------------------------------------------------------------------
def fig_lang_scale(games):
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.6))
    vmin, vmax = 0.1, 0.9
    for ax, mdl in zip(axes.ravel(), MODEL_ORDER):
        piv = (games[games.model == mdl]
               .pivot_table(index="language", columns="scale_nominal",
                            values="coop_rate")
               .reindex(index=LANG_ORDER))
        im = ax.imshow(piv.to_numpy(), cmap=CMAP_SEQ, vmin=vmin, vmax=vmax,
                       aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.4,
                            color="white" if v > 0.55 else INK)
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"×{c:g}" for c in piv.columns], fontsize=6.5,
                           rotation=30, ha="right")
        ax.set_yticks(range(len(LANG_ORDER)))
        ax.set_yticklabels([l.upper() for l in LANG_ORDER], fontsize=7)
        ax.grid(False)
        ax.set_title(mdl, color=MODEL[mdl])
    cb = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("cooperation rate", fontsize=7)
    fig.suptitle("Language and stake interact: the bias is not a constant offset",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F23_language_scale_grid")


# --------------------------------------------------------------------------
def fig_openings(rounds, games):
    fig = plt.figure(figsize=(11.2, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.44)

    # (a) opening move by persona and model ----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    first = rounds[rounds["round"] == 1]
    tab = first.groupby(["model", "personality"]).coop.agg(["mean", "size"])
    x = np.arange(len(MODEL_ORDER))
    for k, pers in enumerate(("cooperative", "selfish")):
        m = tab.xs(pers, level=1).reindex(MODEL_ORDER)
        lo, hi = wilson(m["mean"], m["size"])
        xx = x + (k - 0.5) * 0.36
        ax.bar(xx, m["mean"], width=0.34, color=PERSONALITY[pers],
               edgecolor=SURFACE, linewidth=1.0, label=f"{pers} persona")
        ax.vlines(xx, lo, hi, color=INK, lw=1.0)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("-")[0] for m in MODEL_ORDER], rotation=25,
                       ha="right")
    ax.set_ylabel("P(cooperate) on round 1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    ax.set_title("Opening move")
    panel_tag(ax, "a", dx=-0.26)

    # (b) opening by language -------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    piv = (first.pivot_table(index="model", columns="language", values="coop")
           .reindex(index=MODEL_ORDER, columns=LANG_ORDER))
    im = ax.imshow(piv.to_numpy(), cmap=CMAP_SEQ, vmin=0.1, vmax=0.95,
                   aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                    color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(LANG_ORDER)))
    ax.set_xticklabels([l.upper() for l in LANG_ORDER])
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([m.split("-")[0] for m in MODEL_ORDER])
    ax.grid(False)
    ax.set_title("Opening move by language")
    panel_tag(ax, "b", dx=-0.30)

    # (c) does a nice opening pay? -------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    for mdl in MODEL_ORDER:
        s = games[games.model == mdl]
        m = s.groupby("first_move_coop").payoff_per_round.mean()
        if len(m) == 2:
            ax.plot([0, 1], m.reindex([0, 1]).to_numpy(), "-o", color=MODEL[mdl],
                    ms=6, mec=SURFACE, mew=1.2, label=mdl)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["opened D", "opened C"])
    ax.set_ylabel("mean payoff per round (base units)")
    ax.set_title("Return on a cooperative opening")
    ax.legend(ncol=2, fontsize=6.2, loc="best")
    panel_tag(ax, "c", dx=-0.30)

    fig.suptitle("Openings: the first move is where the persona bites hardest",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig(fig, "F24_openings")


def main():
    rounds = pd.read_parquet(DATADIR / "rounds.parquet")
    games = pd.read_parquet(DATADIR / "games.parquet")
    var = fig_variance(games)
    fig_distributions(games)
    fig_lang_scale(games)
    fig_openings(rounds, games)
    print(var.pivot(index="factor", columns="scope", values="eta2").round(4).to_string())


if __name__ == "__main__":
    main()
