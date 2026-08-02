"""F17 -- behavioural signature radars, F18 -- perseveration vs reciprocity,
F19 -- the model card, F20 -- departures from game-theoretic invariances."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdlib.metrics import fairness_gap, reciprocity
from pdlib.style import (CMAP_DIV, CMAP_SEQ, C_COOP, C_DEFECT, DATADIR,
                         FRONTIER_MODELS, INK, INK2, MODEL, MODEL_ORDER, MUTED,
                         SMALL_MODELS, STRATEGY, STRATEGY_ORDER, SURFACE,
                         TABDIR, panel_tag, savefig, use_paper_style)

use_paper_style()

METRICS = [
    ("coop_rate", "cooperation"),
    ("nice", "opens with C"),
    ("reciprocity", "reciprocity"),
    ("self_persistence", "self-persistence"),
    ("endgame_drop", "endgame drop"),
    ("efficiency", "payoff efficiency"),
    ("dd_absorbed", "locked in DD"),
    ("language_gap", "language disparity"),
    ("persona_gap", "persona gap"),
    ("scale_swing", "scale swing"),
    ("rule_exact", "exact-rule share"),
    ("min_deviations", "distance to rule"),
]


def build_profile(rounds, games, arche):
    rows = []
    for mdl in MODEL_ORDER:
        r = rounds[rounds.model == mdl]
        g = games[games.model == mdl]
        a = arche[arche.model == mdl]
        rec = reciprocity(r)

        lag = r.dropna(subset=["prev_action"])
        self_pers = float((lag.action == lag.prev_action).mean())
        opp_match = float((lag.action == lag.prev_opp_action).mean())

        mid = r.loc[(r.round_frac > 0.3) & (r.round_frac < 0.7), "coop"].mean()
        last = r.loc[r.round_frac == 1.0, "coop"].mean()

        per = g.groupby("personality").coop_rate.mean()
        sc = g.groupby("scale_nominal").coop_rate.mean()

        rows.append({
            "model": mdl,
            "family": g.family.iloc[0],
            "coop_rate": g.coop_rate.mean(),
            "nice": rec["nice"],
            "p_c_after_c": rec["p_c_after_c"],
            "p_c_after_d": rec["p_c_after_d"],
            "reciprocity": rec["reciprocity"],
            "self_persistence": self_pers,
            "opp_matching": opp_match,
            "endgame_drop": float(mid - last),
            "efficiency": g.efficiency.mean(),
            "dd_absorbed": g[g.agent == 1].dd_absorbed.mean(),
            "language_gap": fairness_gap(g, "language", "coop_rate"),
            "persona_gap": abs(per.get("cooperative", np.nan)
                               - per.get("selfish", np.nan)),
            "scale_swing": float(sc.max() - sc.min()),
            "canonicality": a.confidence.mean(),
            # how far the play is from *provably* following a canonical rule
            "rule_exact": float((a.assignment == "exact").mean()),
            "rule_ambiguous": float((a.assignment == "ambiguous").mean()),
            "min_deviations": float(a.min_deviations.mean()),
            "share_AllC": (a.archetype == "AllC").mean(),
            "share_TFT": (a.archetype == "TFT").mean(),
            "share_WSLS": (a.archetype == "WSLS").mean(),
            "share_AllD": (a.archetype == "AllD").mean(),
            "share_Ambiguous": (a.archetype == "Ambiguous").mean(),
        })
    prof = pd.DataFrame(rows).set_index("model")
    prof.to_csv(TABDIR / "T15_model_profiles.csv")
    return prof


# --------------------------------------------------------------------------
def fig_radar(prof):
    keys = [k for k, _ in METRICS]
    labels = [l for _, l in METRICS]
    # min-max scale each axis across the six models so the radar is readable
    z = prof[keys].copy()
    z = (z - z.min()) / (z.max() - z.min() + 1e-12)

    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.2),
                             subplot_kw=dict(projection="polar"))
    for ax, group, title in ((axes[0], FRONTIER_MODELS, "Frontier models"),
                             (axes[1], SMALL_MODELS, "Open-weight models")):
        for mdl in group:
            v = z.loc[mdl].to_numpy()
            v = np.concatenate([v, v[:1]])
            ax.plot(ang, v, color=MODEL[mdl], lw=2.0, label=mdl)
            ax.fill(ang, v, color=MODEL[mdl], alpha=0.12)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(labels, fontsize=6.8, color=INK2)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.set_ylim(0, 1.05)
        ax.grid(color="#e1e0d9", lw=0.7)
        ax.spines["polar"].set_color("#e1e0d9")
        ax.set_title(title, pad=22)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3,
                  fontsize=6.8)
    fig.suptitle("Behavioural signatures  ·  each axis min–max scaled across all six models",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    savefig(fig, "F17_signature_radar")


# --------------------------------------------------------------------------
def fig_perseveration(prof, rounds):
    fig = plt.figure(figsize=(11.2, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.42)

    # (a) self-persistence vs opponent-matching -----------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=(0, (4, 3)))
    for mdl in MODEL_ORDER:
        p = prof.loc[mdl]
        ax.plot(p.self_persistence, p.opp_matching, "o", ms=11,
                color=MODEL[mdl], mec=SURFACE, mew=1.5)
        ax.annotate(mdl.split("-")[0], (p.self_persistence, p.opp_matching),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=6.8, color=MODEL[mdl], fontweight="semibold")
    # canonical rules for reference
    ax.plot(1.0, 0.5, "*", ms=14, color=INK, mec=SURFACE, mew=1.0)
    ax.annotate("AllC / AllD", (1.0, 0.5), textcoords="offset points",
                xytext=(-6, -16), ha="right", fontsize=6.6, color=INK2)
    ax.plot(0.5, 1.0, "*", ms=14, color=STRATEGY["TFT"], mec=SURFACE, mew=1.0)
    ax.annotate("TFT", (0.5, 1.0), textcoords="offset points", xytext=(8, -4),
                fontsize=6.6, color=STRATEGY["TFT"])
    ax.set_xlabel("P(repeat own previous move)")
    ax.set_ylabel("P(copy opponent's previous move)")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.grid(True, axis="both")
    ax.set_title("Perseveration vs imitation")
    panel_tag(ax, "a", dx=-0.26)

    # (b) cooperation-reciprocity plane, bubbles = efficiency ---------------
    ax = fig.add_subplot(gs[0, 1])
    for mdl in MODEL_ORDER:
        p = prof.loc[mdl]
        ax.scatter(p.coop_rate, p.reciprocity, s=p.efficiency * 900,
                   color=MODEL[mdl], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(mdl.split("-")[0], (p.coop_rate, p.reciprocity),
                    textcoords="offset points", xytext=(0, 15), ha="center",
                    fontsize=6.8, color=MODEL[mdl], fontweight="semibold")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.axvline(0.5, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.set_xlabel("cooperation rate")
    ax.set_ylabel("reciprocity  P(C|C) − P(C|D)")
    ax.set_title("Bubble area ∝ payoff efficiency")
    ax.grid(True, axis="both")
    panel_tag(ax, "b", dx=-0.28)

    # (c) transition to defection conditional on own vs opponent -----------
    ax = fig.add_subplot(gs[0, 2])
    lag = rounds.dropna(subset=["prev_action"])
    tab = (lag.groupby(["model", "prev_action", "prev_opp_action"]).coop.mean()
           .unstack(level=[1, 2]))
    cols = [("C", "C"), ("C", "D"), ("D", "C"), ("D", "D")]
    tab = tab.reindex(index=MODEL_ORDER, columns=cols)
    im = ax.imshow(tab.to_numpy(), cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.55 else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["own C\nopp C", "own C\nopp D", "own D\nopp C",
                        "own D\nopp D"], fontsize=6.8)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([m.split("-")[0] for m in MODEL_ORDER])
    ax.grid(False)
    ax.set_title("P(cooperate | last round)")
    panel_tag(ax, "c", dx=-0.34)

    fig.suptitle("Do the models react to the opponent, or just to themselves?",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig(fig, "F18_perseveration")


# --------------------------------------------------------------------------
def fig_model_card(prof):
    keys = [k for k, _ in METRICS]
    labels = [l for _, l in METRICS]
    v = prof[keys]
    z = (v - v.mean()) / (v.std(ddof=0) + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6),
                             gridspec_kw={"width_ratios": [1.5, 1],
                                          "wspace": 0.34})

    ax = axes[0]
    m = np.nanmax(np.abs(z.to_numpy()))
    im = ax.imshow(z.to_numpy(), cmap=CMAP_DIV, vmin=-m, vmax=m, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{v.iat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.4,
                    color="white" if abs(z.iat[i, j]) > 0.62 * m else INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.grid(False)
    ax.set_title("Model card  ·  cell shows the raw value, colour is the z-score")
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.outline.set_visible(False)
    cb.set_label("z across models", fontsize=7)
    panel_tag(ax, "a", dx=-0.22)

    ax = axes[1]
    cats = STRATEGY_ORDER + ["Ambiguous"]
    cat_color = dict(STRATEGY, Ambiguous=MUTED)
    share = prof[[f"share_{s}" for s in cats]]
    bottom = np.zeros(len(share))
    x = np.arange(len(share))
    for s in cats:
        vv = share[f"share_{s}"].to_numpy()
        ax.bar(x, vv, bottom=bottom, color=cat_color[s], edgecolor=SURFACE,
               linewidth=1.2, width=0.68, label=s)
        for xi, (b, q) in enumerate(zip(bottom, vv)):
            if q > 0.10:
                ax.text(xi, b + q / 2, f"{q:.2f}", ha="center", va="center",
                        fontsize=6.4, color="white", fontweight="semibold")
        bottom = bottom + vv
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("-")[0] for m in share.index], rotation=25,
                       ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of agent-games")
    ax.grid(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=6.5)
    ax.set_title("Archetype mix")
    panel_tag(ax, "b", dx=-0.24)

    fig.suptitle("One-page summary of every model's play", x=0.02, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig(fig, "F19_model_card")


# --------------------------------------------------------------------------
def fig_invariances(prof):
    """Three invariances a game-theoretically consistent agent should satisfy,
    and how far each model departs from them."""
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    specs = [
        ("scale_swing", "Payoff-scale invariance",
         "a positive rescaling leaves the game identical", "cooperation swing across scales"),
        ("language_gap", "Language invariance",
         "the same game described in another language", "cooperation gap across languages"),
        ("persona_gap", "Persona sensitivity",
         "a label should not rewrite the payoff matrix", "cooperation gap across personas"),
    ]
    for ax, (key, title, sub, xlab), tag in zip(axes, specs, "abc"):
        s = prof[key].reindex(MODEL_ORDER)
        y = np.arange(len(s))[::-1]
        ax.barh(y, s.to_numpy(), color=[MODEL[m] for m in s.index], height=0.6,
                edgecolor=SURFACE, linewidth=1.0)
        for yi, vv in zip(y, s.to_numpy()):
            ax.text(vv + 0.008, yi, f"{vv:.2f}", va="center", fontsize=7,
                    color=INK2)
        ax.axvline(0, color=INK, lw=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels([m.split("-")[0] for m in s.index])
        ax.set_xlim(0, float(np.nanmax(s.to_numpy())) * 1.32)
        ax.set_xlabel(xlab, fontsize=7)
        ax.set_title(title + "\n" + sub, fontsize=8.6)
        ax.grid(True, axis="x")
        panel_tag(ax, tag, dx=-0.30, dy=1.10)
    fig.suptitle("Departures from invariances the prisoner's dilemma should respect",
                 x=0.02, y=0.99, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, "F20_invariance_violations")


def main():
    rounds = pd.read_parquet(DATADIR / "rounds.parquet")
    games = pd.read_parquet(DATADIR / "games.parquet")
    arche = pd.read_parquet(DATADIR / "llm_archetypes.parquet")
    prof = build_profile(rounds, games, arche)
    fig_radar(prof)
    fig_perseveration(prof, rounds)
    fig_model_card(prof)
    fig_invariances(prof)
    print(prof.round(3).to_string())


if __name__ == "__main__":
    main()
