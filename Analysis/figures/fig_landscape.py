"""Cooperation against the payoff scale, drawn so that the models stay apart.

The claim is that cooperation does move with the payoff scale, that it moves
differently in every model, and that pooling the five models therefore reports
a movement no individual model makes.  The earlier version of this figure drew
five curves with confidence bands on one pair of axes, where the bands overlap
almost everywhere and the reader can recover neither the level of any model nor
the shape of any model's response.  Nothing here is new data; the same numbers
are simply given a form in which the disagreement is the thing you see first.

Panels
  a  The whole design as a table of colour: mean cooperation for each of the
     five models at each of the ten payoff scales, with the number printed in
     every tile.  A row is one model read left to right, and the rows plainly
     disagree, in level and in direction.  The row labels carry the model
     colours used by the rest of the paper, so the figure needs no legend.
  b  Small multiples, one panel per model.  The model that owns the panel is
     drawn at full strength with its own marker; the other four sit behind it
     in grey as context.  At most one salient line per panel is what keeps the
     comparison readable, and the facet title in the model's colour is what
     replaces the legend a spaghetti plot would have needed.
  c  The pooling argument reduced to one number per model: the within-model
     range, meaning the highest minus the lowest cooperation rate across the
     ten scales.  The five-model average has a range of 0.15, drawn as the
     vertical rule.

A caveat the panel c claim is written around.  The pooled range is not smaller
than every within-model range, and the figure does not pretend otherwise.  It
is smaller than three of the five, and four times smaller than the largest, but
it is larger than the two flattest models.  The stronger and more defensible
statement is about direction rather than size: the pooled curve rises with the
scale, Spearman rho of +0.27 against log lambda, while two of the five models
fall, at rho -0.89 for Qwen3 and -0.50 for Gemini 3.5, and the pooled shape
tracks Grok almost perfectly, at a correlation of 0.98, because Grok is the
model that swings most.  The average is not a compromise between the five, it
is a portrait of the loudest one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S      # noqa: E402
import data as D       # noqa: E402

REPS = 2000
SEED = 20260906


def _cube(g):
    """Dyad means as (model, scale, dyad), preserving paired agents."""
    rows = []
    for m in S.MODEL_ORDER:
        row = [g.loc[(g.model == m) & (g.scale_nominal == s)]
               .groupby("game_id", sort=False).coop_rate.mean().to_numpy(float)
               for s in S.SCALES]
        rows.append(row)
    n = min(len(x) for row in rows for x in row)
    return np.array([[x[:n] for x in row] for row in rows]), n


def _range_cis(cube):
    """Percentile bootstrap of each within-model range and of the pooled range.

    Whole dyads are resampled inside their own model-by-scale cell, preserving
    the paired agents in each game and the sampling noise of the cell means.
    """
    rng = np.random.default_rng(SEED)
    nm, ns, ng = cube.shape
    per = np.empty((REPS, nm))
    pooled = np.empty(REPS)
    done = 0
    while done < REPS:
        b = min(200, REPS - done)
        idx = rng.integers(0, ng, size=(b, nm, ns, ng))
        means = np.take_along_axis(cube[None], idx, axis=3).mean(axis=3)
        per[done:done + b] = means.max(axis=2) - means.min(axis=2)
        p = means.mean(axis=1)
        pooled[done:done + b] = p.max(axis=1) - p.min(axis=1)
        done += b
    q = lambda a: (np.quantile(a, 0.025, axis=0), np.quantile(a, 0.975, axis=0))
    return q(per), q(pooled)


def main():
    g = D.games()
    g = g[g.model.isin(S.MODEL_ORDER)]
    mat = D.coop_matrix(g)
    arr = mat.to_numpy(float)
    lam = np.array(S.SCALES, dtype=float)
    loglam = np.log10(lam)
    pooled = arr.mean(axis=0)

    cube, n_per_cell = _cube(g)
    (lo, hi), (plo, phi) = _range_cis(cube)
    rng_by_model = arr.max(axis=1) - arr.min(axis=1)
    pooled_range = pooled.max() - pooled.min()

    print(f"  {len(g):,} agent-games, {n_per_cell} dyads per model and payoff scale")
    for i, m in enumerate(S.MODEL_ORDER):
        rho = spearmanr(loglam, arr[i]).statistic
        r_pool = np.corrcoef(arr[i], pooled)[0, 1]
        print(f"  {S.MODEL_SHORT[m]:<11} coop {arr[i].min():.3f}-{arr[i].max():.3f}  "
              f"range {rng_by_model[i]:.3f} [{lo[i]:.3f}, {hi[i]:.3f}]  "
              f"rho(log lambda) {rho:+.2f}  r(pooled) {r_pool:+.2f}")
    print(f"  {'pooled':<11} coop {pooled.min():.3f}-{pooled.max():.3f}  "
          f"range {pooled_range:.3f} [{plo:.3f}, {phi:.3f}]  "
          f"rho(log lambda) {spearmanr(loglam, pooled).statistic:+.2f}")
    n_wider = int((rng_by_model > pooled_range).sum())
    print(f"  pooled range is narrower than {n_wider} of {len(S.MODEL_ORDER)} "
          "within-model ranges")

    fig = plt.figure(figsize=(S.FULL, 3.52))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.18, 1.0], hspace=0.86)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.66, 1.0], wspace=0.30)
    bot = outer[1].subgridspec(1, 5, wspace=0.16)
    axA = fig.add_subplot(top[0, 0])
    axC = fig.add_subplot(top[0, 1])
    facets = [fig.add_subplot(bot[0, i]) for i in range(5)]

    # --- a: the design as a table of colour --------------------------------
    cols = ["0.01", "0.1", "0.25", "0.5", "1", "2", "5", "10", "100", "1000"]
    S.heat_tiles(axA, arr, [S.MODEL_SHORT[m] for m in S.MODEL_ORDER], cols,
                 cmap="viridis", vmin=0.25, vmax=0.92, fmt="{:.2f}",
                 textcolor_flip=0.55)
    for t, m in zip(axA.get_yticklabels(), S.MODEL_ORDER):
        t.set_color(S.MODEL_C[m])
        t.set_fontweight("bold")
    axA.set_xlabel(r"payoff scale $\lambda$", labelpad=2)
    axA.tick_params(axis="x", labelsize=S.FS_NOTE, pad=1.5)
    axA.tick_params(axis="y", labelsize=S.FS_NOTE)
    S.panel(axA, "a", "every row moves differently")

    # --- c: within-model range against the pooled range --------------------
    ys = np.arange(len(S.MODEL_ORDER))[::-1]
    axC.axvspan(plo, phi, color=S.BAND, lw=0, zorder=0)
    axC.axvline(pooled_range, color=S.INK_2, lw=0.9, linestyle=(0, (3, 2)),
                zorder=2)
    axC.text(pooled_range + 0.018, -0.72,
             f"five-model average {pooled_range:.2f}", fontsize=S.FS_NOTE,
             color=S.INK_2, ha="left", va="center")
    for y, m, v, a, b in zip(ys, S.MODEL_ORDER, rng_by_model, lo, hi):
        axC.hlines(y, a, b, color=S.MODEL_C[m], lw=1.1, zorder=3)
        S.dot(axC, v, y, color=S.MODEL_C[m], marker=S.MODEL_M[m], size=20)
        axC.text(0.815, y, f"{v:.2f}", fontsize=S.FS_NOTE,
                 color=S.INK_2, ha="right", va="center")
    axC.set_yticks(ys)
    axC.set_yticklabels([S.MODEL_SHORT[m] for m in S.MODEL_ORDER])
    for t, m in zip(axC.get_yticklabels(), S.MODEL_ORDER):
        t.set_color(S.MODEL_C[m])
    axC.tick_params(axis="y", length=0, labelsize=S.FS_NOTE)
    axC.set_ylim(-1.0, 4.7)
    axC.set_xlim(0, 0.82)
    axC.set_xticks([0, 0.2, 0.4, 0.6])
    axC.set_xlabel("cooperation range over the ten scales", labelpad=2)
    S.strip(axC, grid_axis="x")
    S.panel(axC, "c")
    axC.text(0.062, 1.0, "the average is nobody's curve",
             transform=axC.transAxes, ha="left", va="bottom",
             fontsize=S.FS_CLAIM, color=S.INK_2)

    # --- b: one salient line per panel -------------------------------------
    for i, (ax, m) in enumerate(zip(facets, S.MODEL_ORDER)):
        for j, other in enumerate(S.MODEL_ORDER):
            if j != i:
                ax.plot(lam, arr[j], color=S.HAIRLINE, lw=0.8, zorder=2)
        ax.plot(lam, arr[i], color=S.MODEL_C[m], lw=1.4, zorder=4,
                marker=S.MODEL_M[m], ms=2.8, markeredgecolor=S.SURFACE,
                markeredgewidth=0.4)
        S.scale_axis(ax, label=None, ticks=False)
        ax.set_xticks([0.01, 1, 1000])
        ax.set_xticklabels(["0.01", "1", "1000"])
        ax.set_ylim(0.20, 0.96)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        S.strip(ax)
        ax.tick_params(labelsize=S.FS_NOTE)
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("cooperation rate", labelpad=2)
        ax.set_title(S.MODEL_SHORT[m], fontsize=S.FS_NOTE,
                     color=S.MODEL_C[m], fontweight="bold", pad=2.6)
        if i == 2:
            ax.set_xlabel(r"payoff scale $\lambda$", labelpad=1.5)
    # The facet titles already occupy the title line, so the panel letter for
    # the row is set in figure coordinates just above them.
    box = facets[0].get_position()
    fig.text(box.x0, box.y1 + 0.052, "b", ha="left", va="bottom",
             fontsize=S.FS_PANEL, color=S.INK, fontweight="bold")
    fig.text(box.x0 + 0.017, box.y1 + 0.052, "no two models agree", ha="left",
             va="bottom", fontsize=S.FS_CLAIM, color=S.INK_2)

    S.caption(
        fig,
        f"{len(g):,} agent-games from 10,000 dyads, {n_per_cell} dyads per model and payoff scale; "
        "shading in b marks the sub-unit regime; intervals in c are 95% percentile "
        "bootstrap intervals over whole dyads.",
        y=-0.07,
    )

    S.save(fig, "f_landscape")


if __name__ == "__main__":
    main()
