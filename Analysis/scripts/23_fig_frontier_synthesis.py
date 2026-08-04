"""FR8: the behavioural signature of each frontier model, and where the
variance in cooperation actually lives.

Panel b is the load-bearing result of the suite.  If model identity dominates
the design factors, then "how an LLM plays the prisoner's dilemma" is a
property of the model; if the design factors dominate, it is a property of the
prompt.  The decomposition is a sequential (Type I) sum of squares on the
dyad-level cooperation rate, entered in the order model, scale, language,
persona pairing, which credits shared variance to whichever factor is entered
first -- so model identity is given every advantage here, and the number it
gets is an upper bound on its share.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdlib.metrics import reciprocity, state_transitions
from pdlib.natstyle import (CMAP_DIV, DATADIR, FRONTIER, INK, INK2, MODEL_C,
                            MODEL_LABEL, MUTED, PAGE, RULE, SPINE, TABDIR, W2,
                            annotate_heatmap, caption, colorbar, figure,
                            finalize, hgrid, save, use_journal_style)

use_journal_style()

# metric key -> (label, source table, higher-is-what)
METRICS = [
    ("coop_rate", "cooperation rate"),
    ("cc_rate", "mutual cooperation"),
    ("dd_rate", "mutual defection"),
    ("efficiency", "payoff efficiency"),
    ("nice", "niceness (round 1)"),
    ("reciprocity", "reciprocity"),
    ("forgiveness", "forgiveness"),
    ("dd_absorbed", "locked into DD"),
    ("scale_sensitivity", "scale sensitivity"),
    ("language_gap", "language disparity"),
    ("persona_effect", "persona steerability"),
]


def build_signature(games, rounds):
    rows = {}
    for mdl in FRONTIER:
        g = games[games.model == mdl]
        r = rounds[rounds.model == mdl]
        rec = reciprocity(r)
        per_scale = g.groupby("scale_nominal").coop_rate.mean()
        per_lang = g.groupby("language").coop_rate.mean()
        rows[mdl] = {
            "coop_rate": g.coop_rate.mean(),
            "cc_rate": g.cc_rate.mean(),
            "dd_rate": g.dd_rate.mean(),
            "efficiency": g.efficiency.mean(),
            "nice": rec["nice"],
            "reciprocity": rec["reciprocity"],
            "forgiveness": rec["forgiveness"],
            "dd_absorbed": g.dd_absorbed.mean(),
            "scale_sensitivity": per_scale.max() - per_scale.min(),
            "language_gap": per_lang.max() - per_lang.min(),
            "persona_effect": (g[g.personality == "cooperative"].coop_rate.mean()
                               - g[g.personality == "selfish"].coop_rate.mean()),
        }
    return pd.DataFrame(rows).T.reindex(FRONTIER)


def variance_decomposition(games):
    """Sequential (Type I) R-squared for coop_rate ~ model + scale + language
    + dyad.

    The unit is the agent-game, not the dyad: `dyad` records the focal agent's
    persona *and* its opponent's, so it differs between the two agents of a
    mixed-persona game and collapsing to the dyad would either lose the factor
    or split those games in two.  The two rows of a game are not independent,
    which matters for a standard error but not for the descriptive share of
    variance reported here.
    """
    d = games[["model", "scale_nominal", "language", "dyad", "coop_rate"]].copy()
    y = d.coop_rate.to_numpy()
    ss_tot = float(((y - y.mean()) ** 2).sum())

    factors = [("model", "model"), ("scale_nominal", "payoff scale"),
               ("language", "language"), ("dyad", "persona pairing")]
    X = np.ones((len(d), 1))
    prev_r2 = 0.0
    out = []
    for col, label in factors:
        dummies = pd.get_dummies(d[col], drop_first=True, dtype=float).to_numpy()
        X = np.hstack([X, dummies])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - float(((y - X @ beta) ** 2).sum()) / ss_tot
        out.append({"factor": label, "share": r2 - prev_r2, "cumulative_r2": r2,
                    "df": dummies.shape[1]})
        prev_r2 = r2
    out.append({"factor": "unexplained", "share": 1 - prev_r2,
                "cumulative_r2": 1.0, "df": len(d) - X.shape[1]})
    return pd.DataFrame(out), len(d)


def fig_synthesis(games, rounds):
    sig = build_signature(games, rounds)
    sig.to_csv(TABDIR / "T_FR16_signature_raw.csv")
    var, n_dyads = variance_decomposition(games)
    var.to_csv(TABDIR / "T_FR17_variance_decomposition.csv", index=False)

    fig = figure(W2, 2.75)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0])

    # (a) z-scored signature --------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    keys = [k for k, _ in METRICS]
    raw = sig[keys]
    # z across the four models: the question is how they differ from each
    # other, and the metrics are on incomparable scales
    z = (raw - raw.mean()) / raw.std(ddof=0)
    z.to_csv(TABDIR / "T_FR16b_signature_z.csv")

    im = ax.imshow(z.to_numpy(), cmap=CMAP_DIV, vmin=-1.6, vmax=1.6,
                   aspect="auto")
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v, zz = raw.iat[i, j], z.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                    color=PAGE if abs(zz) > 1.05 else INK)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([lab for _, lab in METRICS], rotation=38, ha="right",
                       rotation_mode="anchor", fontsize=5.8)
    ax.set_yticks(range(len(FRONTIER)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in FRONTIER], fontsize=6.4)
    ax.tick_params(length=0)
    ax.tick_params(axis="y", pad=7)
    for s in ax.spines.values():
        s.set_visible(False)
    # colour key sits between the row label and the first cell
    for i in range(len(FRONTIER)):
        ax.add_patch(plt.Rectangle((-0.60, i - 0.46), 0.10, 0.92,
                                   facecolor=MODEL_C[FRONTIER[i]],
                                   clip_on=False, zorder=5))
    colorbar(fig, im, ax, label="z across models", ticks=[-1.5, 0, 1.5])
    ax.set_title("Behavioural signature (cell values are raw units)", pad=6)

    # (b) variance decomposition ---------------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax, axis="x")
    fill = {"model": MODEL_C["Gemini-3.5-Flash-Lite"], "payoff scale": "#7fb2d4",
            "language": "#a1b792", "persona pairing": "#be9976",
            "unexplained": "#dcdcdc"}
    y = np.arange(len(var))[::-1]
    for yi, r in zip(y, var.itertuples()):
        ax.barh(yi, r.share, height=0.62, color=fill[r.factor], edgecolor=PAGE,
                linewidth=0.5, zorder=3)
        ax.text(r.share + 0.012, yi, f"{r.share:.1%}", ha="left", va="center",
                fontsize=6.2, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(var.factor)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.65, len(var) - 0.35)
    ax.set_xlim(0, max(0.55, var.share.max() * 1.32))
    ax.set_xlabel("share of variance in dyad cooperation rate")
    ax.set_title("Where the variance lives", pad=6)

    finalize(fig, [pa, pb], ["a", "b"])
    caption(fig, f"Panel b is a sequential (Type I) decomposition over "
                 f"{n_dyads:,} agent-game cooperation rates, entered in the "
                 f"order shown, so each factor is credited only with what the "
                 f"factors above it leave unexplained; entering model first "
                 f"makes its {var.share.iloc[0]:.1%} an upper bound. The four "
                 f"design factors together account for "
                 f"{1 - var.share.iloc[-1]:.1%} of the variance. The remaining "
                 f"{var.share.iloc[-1]:.1%} is within-cell: replicates of the "
                 f"identical prompt diverge more than the design factors move "
                 f"the mean.")
    save(fig, "FR8_synthesis")
    return sig, var


def main():
    games = pd.read_parquet(DATADIR / "frontier_games.parquet")
    rounds = pd.read_parquet(DATADIR / "frontier_rounds.parquet")
    sig, var = fig_synthesis(games, rounds)
    print()
    print(sig.round(3).to_string())
    print()
    print(var.to_string(index=False))


if __name__ == "__main__":
    main()
