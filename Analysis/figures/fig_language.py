"""The language the game is written in sets the size of the payoff-scale effect.

The manuscript's central result is that shrinking every printed payoff below one
unit suppresses cooperation.  That result was established on English prompts.
This figure asks whether it survives translation, and the answer is that it
survives in four languages out of five and dies in the fifth.  What changes
across languages is not the level of cooperation, which moves by only about
eight points from the least to the most cooperative language, but the size of
the scale effect itself, which moves from eighteen points in English to nothing
at all in French.  That is a language-by-scale interaction rather than a main
effect of language, and it matters because a main effect of language would be a
nuisance to control for whereas an interaction means the scale effect is not a
property of the payoff numbers alone.

The scale effect is measured the way the rest of the paper measures it, as the
difference between the sub-unit regime, the two scales 0.01 and 0.1 at which
every printed payoff is at most one, and the supra-unit regime, the remaining
eight scales.  The maximum-minus-minimum range over the ten scales is reported
in the printed output as well, but it is not the quantity plotted, because a
range is bounded below by zero and is inflated by sampling noise, so it cannot
show the one language whose scale effect is absent.

Panels
  a  Mean cooperation for every language and every payoff scale.  The dashed
     rule separates the sub-unit regime on the left from the supra-unit regime
     on the right.  Reading along the French row, the two regimes are the same
     shade; reading along the English row they are not.
  b  The scale effect per language, supra-unit minus sub-unit, ranked, with a
     percentile bootstrap interval on the pooled estimate and the five models
     shown individually so that the ranking is not read as the work of one
     model.  Only French has an interval that contains zero.
  c  Each language's cooperation curve after its own mean has been subtracted,
     so that any pure level difference between languages is removed and only
     the shape remains.  The other four curves are drawn behind each facet in
     grey.  The shapes stay different, which is the interaction.

The shaded band in panel c and the dashed rule in panel a both mark the same
boundary, the one between 0.1 and 0.25, because that is where the statistics
tables put it.  The shared helper shades to a payoff scale of one instead, so
this figure draws its own band rather than calling for the default one.

Caveats.  The five models disagree about how large the scale effect is in any
given language, and Grok, whose scale effect is the largest in the corpus,
pulls the pooled ranking upward in every language.  Qwen has a small negative
scale effect everywhere, so the ranking is a statement about the average model
and not about each model separately.  Language and payoff scale are crossed by
design, so the interaction is not confounded with model or persona, but the
translations were produced once and not back-translated, so a prompt-wording
artefact in a single language cannot be excluded by this design.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S      # noqa: E402
import data as D       # noqa: E402

SUBUNIT = [0.01, 0.1]


def regime_shift(d):
    lo = d[d.scale_nominal.isin(SUBUNIT)].coop_rate
    hi = d[~d.scale_nominal.isin(SUBUNIT)].coop_rate
    return float(hi.mean()) - float(lo.mean())


def shift_ci(d, *, n=2000, seed=20260906):
    rng = np.random.default_rng(seed)
    lo = d[d.scale_nominal.isin(SUBUNIT)].coop_rate.to_numpy(float)
    hi = d[~d.scale_nominal.isin(SUBUNIT)].coop_rate.to_numpy(float)
    a = rng.choice(lo, size=(n, len(lo)), replace=True).mean(axis=1)
    b = rng.choice(hi, size=(n, len(hi)), replace=True).mean(axis=1)
    d_ = b - a
    return float(np.quantile(d_, 0.025)), float(np.quantile(d_, 0.975))


def main():
    g = D.games()
    g = g[g.model.isin(S.MODEL_ORDER)]
    print(f"  {len(g)} games, {g.language.nunique()} languages, "
          f"{g.scale_nominal.nunique()} scales")

    mat = (g.groupby(["language", "scale_nominal"], observed=True)
             .coop_rate.mean().unstack("scale_nominal")
             .reindex(index=S.LANG_ORDER, columns=S.SCALES))
    arr = mat.to_numpy()

    shift, ci, rng_, lvl = {}, {}, {}, {}
    per_model = {}
    for lang in S.LANG_ORDER:
        d = g[g.language == lang]
        shift[lang] = regime_shift(d)
        ci[lang] = shift_ci(d)
        rng_[lang] = float(mat.loc[lang].max() - mat.loc[lang].min())
        lvl[lang] = float(mat.loc[lang].mean())
        per_model[lang] = {m: regime_shift(d[d.model == m]) for m in S.MODEL_ORDER}

    print("  language   level   range   scale effect (supra - sub)  95% CI")
    for lang in sorted(S.LANG_ORDER, key=lambda k: -shift[k]):
        lo, hi = ci[lang]
        print(f"  {S.LANG_LABEL[lang]:<11}{lvl[lang]:.3f}   {rng_[lang]:.3f}   "
              f"{shift[lang]:+.3f}   [{lo:+.3f}, {hi:+.3f}]"
              + ("   contains zero" if lo <= 0 <= hi else ""))
    spread = max(shift.values()) - min(shift.values())
    print(f"  level spread across languages {max(lvl.values()) - min(lvl.values()):.3f}; "
          f"scale-effect spread {spread:.3f}")
    for lang in S.LANG_ORDER:
        print(f"  {lang}: " + ", ".join(f"{S.MODEL_SHORT[m]} {v:+.3f}"
                                        for m, v in per_model[lang].items()))

    order = sorted(S.LANG_ORDER, key=lambda k: shift[k])

    fig = plt.figure(figsize=(S.FULL, 3.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.24, 1.00], hspace=0.60)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.30)
    bot = outer[1].subgridspec(1, 5, wspace=0.22)
    axA = fig.add_subplot(top[0, 0])
    axB = fig.add_subplot(top[0, 1])
    axC = [fig.add_subplot(bot[0, i]) for i in range(5)]

    # --- a: the language by scale field ------------------------------------
    S.heat_tiles(axA, arr,
                 [S.LANG_LABEL[l] for l in S.LANG_ORDER],
                 ["0.01", "0.1", "0.25", "0.5", "1", "2", "5", "10", "100", "1000"],
                 vmin=0.33, vmax=0.73, fmt="{:.2f}", textcolor_flip=0.60)
    axA.axvline(1.5, color=S.INK, lw=1.0, linestyle=(0, (2.6, 1.8)), zorder=6)
    axA.text(0.5, -0.62, "sub-unit", ha="center", va="bottom",
             fontsize=S.FS_NOTE, color=S.INK_2)
    axA.text(6.0, -0.62, "supra-unit", ha="center", va="bottom",
             fontsize=S.FS_NOTE, color=S.INK_2)
    axA.set_ylim(4.5, -1.05)
    axA.set_xlabel(r"payoff scale $\lambda$")
    S.panel(axA, "a", "cooperation, language by scale", pad=13)

    # --- b: the scale effect, ranked ---------------------------------------
    ys = np.arange(len(order))
    for y, lang in zip(ys, order):
        lo, hi = ci[lang]
        axB.hlines(y, lo, hi, color=S.INK, lw=1.1, zorder=4)
        for m in S.MODEL_ORDER:
            axB.scatter([per_model[lang][m]], [y + 0.30], s=9,
                        marker=S.MODEL_M[m], color=S.MODEL_C[m],
                        linewidths=0, zorder=3, alpha=0.9)
        S.dot(axB, shift[lang], y, color=S.INK, size=22, zorder=5)
    S.zero_rule(axB, 0.0, vertical=True, lw=0.8)
    axB.set_yticks(ys)
    axB.set_yticklabels([S.LANG_LABEL[l] for l in order])
    axB.set_ylim(-0.6, len(order) - 0.35)
    axB.set_xlim(-0.28, 0.78)
    axB.set_xticks([0.0, 0.25, 0.5, 0.75])
    axB.set_xlabel("supra-unit minus sub-unit")
    S.strip(axB, grid_axis="x")
    axB.tick_params(axis="y", length=0)
    S.panel(axB, "b", "how big the effect is", pad=13)

    # --- c: level removed, shape kept --------------------------------------
    cen = {l: mat.loc[l].to_numpy() - lvl[l] for l in S.LANG_ORDER}
    for ax, lang in zip(axC, S.LANG_ORDER):
        for other in S.LANG_ORDER:
            if other == lang:
                continue
            ax.plot(S.SCALES, cen[other], color=S.HAIRLINE, lw=0.8, zorder=2)
        ax.plot(S.SCALES, cen[lang], color=S.INK, lw=1.4, zorder=4)
        ax.scatter(S.SCALES, cen[lang], s=7, color=S.INK, linewidths=0, zorder=5)
        S.zero_rule(ax, 0.0, lw=0.7)
        S.scale_axis(ax, band=False, label=None, ticks=False)
        ax.axvspan(0.006, 0.158, color=S.BAND, lw=0, zorder=0)
        ax.set_xticks([0.01, 1, 1000])
        ax.set_xticklabels(["0.01", "1", "1000"])
        ax.set_ylim(-0.20, 0.15)
        S.strip(ax, grid_axis="y")
        ax.text(0.5, 0.995, S.LANG_LABEL[lang], transform=ax.transAxes,
                ha="center", va="top", fontsize=S.FS_NOTE, color=S.INK)
    for ax in axC[1:]:
        ax.set_yticklabels([])
    axC[0].set_ylabel("centred")
    axC[2].set_xlabel(r"payoff scale $\lambda$")
    S.panel(axC[0], "c", None, pad=7)
    axC[0].text(0.16, 1.005, "level removed, the shapes still differ",
                transform=axC[0].transAxes, ha="left", va="bottom",
                fontsize=S.FS_CLAIM, color=S.INK_2)

    handles = [Line2D([], [], color=S.MODEL_C[m], marker=S.MODEL_M[m], lw=0,
                      ms=3.8, label=S.MODEL_SHORT[m]) for m in S.MODEL_ORDER]
    handles.append(Line2D([], [], color=S.INK, marker="o", lw=1.1, ms=4.0,
                          markerfacecolor=S.INK, label="all five pooled"))
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.055), fontsize=S.FS_NOTE)
    fig.text(0.5, -0.100,
             f"{len(g):,} games from five models, 400 per language and scale; "
             "intervals are 95% percentile bootstrap over games",
             ha="center", va="top", fontsize=S.FS_NOTE, color=S.MUTED)

    S.save(fig, "f_language")


if __name__ == "__main__":
    main()
