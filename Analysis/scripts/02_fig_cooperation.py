"""F02 -- the cooperation landscape, and F03 -- payoff-scale (non-)invariance."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdlib.metrics import grouped_ci, mean_with_ci, scale_slope
from pdlib.style import (CMAP_DIV, CMAP_SEQ, C_COOP, C_DEFECT, DATADIR, GRID,
                         INK, INK2, MODEL, MODEL_ORDER, MUTED, SURFACE, TABDIR,
                         panel_tag, savefig, use_paper_style)

use_paper_style()


def fig_landscape(games):
    fig = plt.figure(figsize=(11.2, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1],
                          hspace=0.58, wspace=0.46)

    # (a) coop rate per model x scale, with game-clustered bootstrap CI ------
    ax = fig.add_subplot(gs[0, :])
    ci = grouped_ci(games, ["model", "scale_nominal"], "coop_rate", n_boot=800)
    scales = sorted(games.scale_nominal.unique())
    xslot = {s: i for i, s in enumerate(scales)}
    w = 0.13
    for k, mdl in enumerate(MODEL_ORDER):
        sub = ci[ci.model == mdl]
        x = np.array([xslot[s] for s in sub.scale_nominal]) + (k - 2.5) * w
        ax.bar(x, sub["mean"], width=w * 0.92, color=MODEL[mdl], edgecolor=SURFACE,
               linewidth=1.0, label=mdl, zorder=3)
        ax.vlines(x, sub["lo"], sub["hi"], color=INK, lw=1.0, zorder=4)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.45, 0.505, "indifference", fontsize=6.8, color=MUTED,
            va="bottom", ha="left")
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f"×{s:g}" for s in scales])
    ax.set_xlabel("payoff scale multiplier")
    ax.set_ylabel("cooperation rate")
    ax.set_ylim(0, 1)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    ax.set_title("Cooperation rate by model and payoff scale  ·  bars = 95% game-clustered bootstrap CI")
    panel_tag(ax, "a", dx=-0.065)

    # (b) heatmap model x scale ---------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    piv = (games.pivot_table(index="model", columns="scale_nominal",
                             values="coop_rate", aggfunc="mean")
           .reindex(MODEL_ORDER))
    im = ax.imshow(piv.to_numpy(), cmap=CMAP_SEQ, vmin=0.2, vmax=0.75,
                   aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.52 else INK)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"×{c:g}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("payoff scale")
    ax.grid(False)
    ax.set_title("Cooperation rate")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, color=MUTED)
    panel_tag(ax, "b", dx=-0.30)

    # (c) outcome composition per model -------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    from pdlib.style import OUTCOME, OUTCOME_ORDER
    comp = (games.groupby("model")[["cc_rate", "cd_rate", "dc_rate", "dd_rate"]]
            .mean().reindex(MODEL_ORDER))
    left = np.zeros(len(comp))
    y = np.arange(len(comp))
    for code, col in zip(OUTCOME_ORDER, ["cc_rate", "cd_rate", "dc_rate", "dd_rate"]):
        v = comp[col].to_numpy()
        ax.barh(y, v, left=left, color=OUTCOME[code], edgecolor=SURFACE,
                linewidth=1.2, label=code)
        for yi, (l, vv) in enumerate(zip(left, v)):
            if vv > 0.09:
                ax.text(l + vv / 2, yi, f"{vv:.2f}", ha="center", va="center",
                        fontsize=6.8, color="white", fontweight="semibold")
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels(comp.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of rounds")
    ax.grid(False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    ax.set_title("Joint-outcome composition")
    panel_tag(ax, "c", dx=-0.34)

    fig.suptitle("Cooperation across models and payoff magnitudes", x=0.02,
                 ha="left", fontweight="bold", color=INK)
    savefig(fig, "F02_cooperation_landscape")


def fig_scale_invariance(games):
    """A positive rescaling of a PD leaves the game strategically identical.
    Any dependence of behaviour on the multiplier is therefore a departure
    from the game-theoretic reading of the task."""
    fig = plt.figure(figsize=(11.2, 6.8))
    gs = fig.add_gridspec(2, 3, hspace=0.60, wspace=0.42)

    # (a) coop vs log scale ---------------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    ci = grouped_ci(games, ["model", "scale_nominal"], "coop_rate", n_boot=800)
    ax.axvspan(-0.35, 1.35, color="#f0efec", zorder=0)
    ax.text(0.5, 0.845, "cooperation trough", ha="center", fontsize=7, color=MUTED)
    for mdl in MODEL_ORDER:
        sub = ci[ci.model == mdl].sort_values("scale_nominal")
        x = np.log10(sub.scale_nominal)
        ax.fill_between(x, sub["lo"], sub["hi"], color=MODEL[mdl], alpha=0.16, lw=0)
        ax.plot(x, sub["mean"], "-o", color=MODEL[mdl], mec=SURFACE, mew=1.2,
                label=mdl)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks([-2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(["×0.01", "×0.1", "×1", "×10", "×100", "×1000"])
    ax.set_xlim(-2.35, 3.35)
    ax.set_ylim(0.15, 0.88)
    ax.set_xlabel("payoff scale (log axis)")
    ax.set_ylabel("cooperation rate")
    ax.legend(ncol=3, loc="lower left", fontsize=7)
    ax.set_title("Cooperation is non-monotone in stake size")
    panel_tag(ax, "a", dx=-0.075)

    # (b) scale-induced swing -------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    rows = []
    for mdl in MODEL_ORDER:
        s = scale_slope(games[games.model == mdl], "coop_rate", n_boot=1500)
        per = ci[ci.model == mdl].set_index("scale_nominal")["mean"]
        rows.append({"model": mdl, **s, "swing": per.max() - per.min(),
                     "argmin": per.idxmin(), "argmax": per.idxmax()})
    sl = pd.DataFrame(rows)
    sl.to_csv(TABDIR / "T04_scale_sensitivity.csv", index=False)
    y = np.arange(len(sl))[::-1]
    for yi, r in zip(y, sl.itertuples()):
        lo = ci[(ci.model == r.model)].set_index("scale_nominal")["mean"]
        ax.plot([lo.min(), lo.max()], [yi, yi], color=MODEL[r.model], lw=3.0,
                solid_capstyle="round", alpha=0.5, zorder=2)
        ax.plot(lo.min(), yi, "o", ms=7, color=MODEL[r.model], mec=SURFACE,
                mew=1.4, zorder=3)
        ax.plot(lo.max(), yi, "o", ms=7, color=MODEL[r.model], mec=SURFACE,
                mew=1.4, zorder=3)
        ax.text(lo.max() + 0.012, yi, f"{r.swing:.2f}", va="center", fontsize=7,
                color=INK2, fontweight="semibold")
        ax.text(lo.min(), yi - 0.34, f"×{r.argmin:g}", ha="center", va="top",
                fontsize=6, color=MUTED)
        ax.text(lo.max(), yi - 0.34, f"×{r.argmax:g}", ha="center", va="top",
                fontsize=6, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(sl.model)
    ax.set_xlim(0.2, 0.86)
    ax.set_ylim(-0.75, len(sl) - 0.25)
    ax.set_xlabel("cooperation rate")
    ax.set_title("Swing across scales")
    ax.grid(True, axis="x")
    panel_tag(ax, "b", dx=-0.62)

    # (c) mutual cooperation and (d) mutual defection vs scale ---------------
    for k, (col, name, tag) in enumerate((("cc_rate", "mutual cooperation (CC)", "c"),
                                          ("dd_rate", "mutual defection (DD)", "d"),
                                          ("efficiency", "payoff efficiency  (mean payoff / R)", "e"))):
        ax = fig.add_subplot(gs[1, k])
        ci = grouped_ci(games, ["model", "scale_nominal"], col, n_boot=600)
        for mdl in MODEL_ORDER:
            sub = ci[ci.model == mdl].sort_values("scale_nominal")
            x = np.log10(sub.scale_nominal)
            ax.plot(x, sub["mean"], "-o", color=MODEL[mdl], ms=4, mec=SURFACE,
                    mew=1.0, label=mdl)
        ax.set_xticks([-2, -1, 0, 1, 2, 3])
        ax.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1k"])
        ax.set_xlabel("payoff scale")
        ax.set_ylabel(name.split("(")[0].strip())
        ax.set_title(name)
        if k == 0:
            handles, labels = ax.get_legend_handles_labels()
        panel_tag(ax, tag, dx=-0.30)
    fig.legend(handles, labels, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.035), fontsize=7.5)

    fig.suptitle("Payoff-scale invariance is violated: identical games, different behaviour",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F03_scale_invariance")
    return sl


def main():
    games = pd.read_parquet(DATADIR / "games.parquet")
    fig_landscape(games)
    sl = fig_scale_invariance(games)
    print(sl.to_string(index=False))


if __name__ == "__main__":
    main()
