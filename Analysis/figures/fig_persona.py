"""Telling a model it is cooperative does not have a fixed effect.

Every agent is given a persona in its prompt, either that it is a cooperative
player or that it is a selfish one, and the natural expectation is that the
cooperative persona raises cooperation by some amount that is a property of the
model.  It does not.  The persona effect, defined here and in the manuscript's
tables as mean cooperation under the cooperative persona minus mean cooperation
under the selfish one, changes with the payoff scale in every model, and in two
of the five it changes sign.  When every printed payoff is at most one unit, the
cooperative persona in Claude and in Gemini 3.5 produces less cooperation than
the selfish persona does, which is the opposite of the instruction.  Once the
payoffs are printed at unit scale or larger, both models obey the instruction.
The persona is therefore not an independent knob; it is gated by how the numbers
in the prompt are written.

The two regimes are the ones the statistics tables use.  The sub-unit regime is
the two payoff scales 0.01 and 0.1, at which every payoff printed in the prompt
is at most one, and the supra-unit regime is the remaining eight scales.  All
values in panels a and b were checked against T07_persona_gating.csv and all
ten agree to three decimal places; the per-scale values in panel c were checked
against T07_persona_by_scale.csv.

Panels
  a  A slopegraph in the sub-unit regime.  The left column is the cooperative
     persona and the right column the selfish one, and a line that falls to the
     right is a model that obeyed its instruction.  In the sub-unit regime all
     five lines rise, which is disobedience by every model.
  b  The same slopegraph in the supra-unit regime.  Claude and Gemini 3.5 have
     turned over and now fall; the other three keep the direction they had.
     Comparing a line between the two panels is what shows the reversal.
  c  The persona effect against payoff scale, with a rule at zero.  Claude and
     Gemini 3.5 cross the rule between 0.1 and 0.25, which is exactly the
     regime boundary.  GPT sits close to zero throughout and crosses several
     times, so its reversal in panel a and panel b is small and should not be
     read as a sign flip; the gating table agrees, marking only Claude and
     Gemini 3.5 as flips.

Caveats.  The intervals are percentile bootstrap over games and therefore treat
games as independent, which slightly understates uncertainty because games share
a repetition seed and an opponent.  The two regimes are unbalanced by design,
two scales against eight, so the sub-unit estimates are the noisier ones.  The
supplementary sixth model, Gemini 3.1 Flash-Lite Preview, is not drawn here; it
has the largest and the most stable persona effect in the corpus and would
compress the vertical range of every panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S      # noqa: E402
import data as D       # noqa: E402

SUBUNIT = [0.01, 0.1]
PERSONAS = ["cooperative", "selfish"]


def persona_means(d):
    return {p: float(d[d.personality == p].coop_rate.mean()) for p in PERSONAS}


def effect_ci(d, *, n=2000, seed=20260906):
    rng = np.random.default_rng(seed)
    a = d[d.personality == "cooperative"].coop_rate.to_numpy(float)
    b = d[d.personality == "selfish"].coop_rate.to_numpy(float)
    draws = (rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
             - rng.choice(b, size=(n, len(b)), replace=True).mean(axis=1))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def slopegraph(ax, means, *, order, ylim):
    for m in order:
        c, s = means[m]["cooperative"], means[m]["selfish"]
        ax.plot([0, 1], [c, s], color=S.MODEL_C[m], lw=1.3, zorder=3,
                solid_capstyle="round")
        S.dot(ax, 0, c, color=S.MODEL_C[m], marker=S.MODEL_M[m], size=19)
        S.dot(ax, 1, s, color=S.MODEL_C[m], marker=S.MODEL_M[m], size=19)
    ax.set_xlim(-0.72, 1.72)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["told\ncooperative", "told\nselfish"])
    ax.tick_params(axis="x", length=0)
    S.strip(ax, grid_axis="y")
    ax.spines["bottom"].set_visible(False)


def nudge(vals, minsep, lo=None, hi=None):
    """Push labels apart just enough that two adjacent ones stay readable.

    Separation is enforced upward, then the whole stack is slid back down if it
    has run past the top of the axes, so a direct label never leaves the panel.
    """
    idx = np.argsort(vals)
    out = np.array(vals, dtype=float)
    for k in range(1, len(idx)):
        i, j = idx[k - 1], idx[k]
        if out[j] - out[i] < minsep:
            out[j] = out[i] + minsep
    if hi is not None and out.max() > hi:
        out -= out.max() - hi
    if lo is not None and out.min() < lo:
        out += lo - out.min()
    return out


def main():
    g = D.games()
    g = g[g.model.isin(S.MODEL_ORDER)]
    gate = D.table("T07_persona_gating.csv").set_index("model")
    by_scale = D.table("T07_persona_by_scale.csv")

    means = {"sub": {}, "sup": {}}
    eff = {"sub": {}, "sup": {}}
    ci = {"sub": {}, "sup": {}}
    for m in S.MODEL_ORDER:
        d = g[g.model == m]
        parts = {"sub": d[d.scale_nominal.isin(SUBUNIT)],
                 "sup": d[~d.scale_nominal.isin(SUBUNIT)]}
        for k, dd in parts.items():
            means[k][m] = persona_means(dd)
            eff[k][m] = means[k][m]["cooperative"] - means[k][m]["selfish"]
            ci[k][m] = effect_ci(dd)

    print("  model        sub-unit    supra-unit   change    sign flip")
    bad = []
    for m in S.MODEL_ORDER:
        # A flip is a claim about two means, so it needs an interval on each.
        # Where one interval straddles zero the honest reading is that the
        # effect is abolished, not that it is reversed, which is the rule the
        # gating table uses.
        (sl, sh), (pl, ph) = ci["sub"][m], ci["sup"][m]
        flip = bool(sh < 0 < pl or ph < 0 < sl)
        print(f"  {S.MODEL_SHORT[m]:<12}{eff['sub'][m]:+.3f}      "
              f"{eff['sup'][m]:+.3f}      {eff['sup'][m] - eff['sub'][m]:+.3f}    "
              f"{'yes' if flip else 'no'}")
        for k, col in (("sub", "persona_effect_subunit"),
                       ("sup", "persona_effect_suprunit")):
            if abs(eff[k][m] - gate.loc[m, col]) > 5e-4:
                bad.append((m, k, eff[k][m], gate.loc[m, col]))
        if flip != bool(gate.loc[m, "sign_flip"]):
            bad.append((m, "sign_flip", flip, gate.loc[m, "sign_flip"]))
    if bad:
        print("  DISAGREEMENT with T07_persona_gating.csv:")
        for row in bad:
            print("   ", row)
    else:
        print("  all ten values and both sign flips match T07_persona_gating.csv")

    curves = {m: (by_scale[by_scale.model == m]
                  .set_index("scale").persona_effect.reindex(S.SCALES))
              for m in S.MODEL_ORDER}
    for m in S.MODEL_ORDER:
        d = g[g.model == m]
        mine = np.array([persona_means(d[d.scale_nominal == s])["cooperative"]
                         - persona_means(d[d.scale_nominal == s])["selfish"]
                         for s in S.SCALES])
        gap = float(np.nanmax(np.abs(mine - curves[m].to_numpy())))
        print(f"  {S.MODEL_SHORT[m]:<12}per-scale max gap to T07_persona_by_scale "
              f"{gap:.4f}")
        curves[m] = mine

    fig = plt.figure(figsize=(S.FULL, 3.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.86, 0.86, 1.42], wspace=0.50)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    ylim = (-0.06, 1.06)
    for ax, key, letter, claim in ((axA, "sub", "a", "sub-unit: all five rise"),
                                   (axB, "sup", "b", "supra-unit: two turn over")):
        slopegraph(ax, means[key], order=S.MODEL_ORDER, ylim=ylim)
        if key == "sub":
            ax.axvspan(-0.72, 1.72, color=S.BAND, lw=0, zorder=0)
            ax.set_xlim(-0.72, 1.72)
        left = nudge([means[key][m]["cooperative"] for m in S.MODEL_ORDER],
                     0.075, ylim[0] + 0.03, ylim[1] - 0.07)
        right = nudge([means[key][m]["selfish"] for m in S.MODEL_ORDER],
                      0.075, ylim[0] + 0.03, ylim[1] - 0.07)
        for i, m in enumerate(S.MODEL_ORDER):
            ax.text(-0.12, left[i], S.MODEL_SHORT[m], ha="right", va="center",
                    fontsize=S.FS_NOTE, color=S.MODEL_C[m])
            ax.text(1.12, right[i], S.MODEL_SHORT[m], ha="left", va="center",
                    fontsize=S.FS_NOTE, color=S.MODEL_C[m])
        S.panel(ax, letter, claim, pad=8)
    axA.set_ylabel("cooperation rate")
    axB.set_yticklabels([])

    # --- c: the effect against scale ---------------------------------------
    for m in S.MODEL_ORDER:
        axC.plot(S.SCALES, curves[m], color=S.MODEL_C[m], lw=1.2, zorder=3)
        axC.scatter(S.SCALES, curves[m], s=10, marker=S.MODEL_M[m],
                    color=S.MODEL_C[m], linewidths=0, zorder=4)
    S.zero_rule(axC, 0.0, lw=0.9)
    S.scale_axis(axC, band=False)
    axC.axvspan(0.006, 0.158, color=S.BAND, lw=0, zorder=0)
    axC.set_ylim(-1.02, 0.52)
    axC.set_ylabel("told cooperative minus told selfish")
    S.strip(axC, grid_axis="y")
    ends = nudge([curves[m][-1] for m in S.MODEL_ORDER], 0.115, -0.98, 0.48)
    for i, m in enumerate(S.MODEL_ORDER):
        axC.text(2300, ends[i], S.MODEL_SHORT[m], ha="left", va="center",
                 fontsize=S.FS_NOTE, color=S.MODEL_C[m])
    axC.set_xlim(0.006, 2100)
    for m in ("Claude-Haiku-4.5", "Gemini-3.5-Flash-Lite"):
        below = curves[m][1]
        above = curves[m][2]
        x = 0.1 * (0.25 / 0.1) ** (-below / (above - below))
        S.dot(axC, x, 0.0, color=S.MODEL_C[m], marker=S.MODEL_M[m], size=26,
              filled=False, zorder=6)
    axC.text(0.0105, 0.47,
             "open rings: Claude and Gemini 3.5\ncross zero at the regime edge",
             fontsize=S.FS_NOTE, color=S.INK_2, ha="left", va="top",
             linespacing=1.25)
    S.panel(axC, "c", "the instruction is obeyed only above the unit", pad=8)

    S.caption(fig,
              f"{len(g):,} games, {len(g) // len(S.MODEL_ORDER) // 2:,} per "
              "model and persona; the shaded ground marks the sub-unit regime, "
              "the two payoff scales at which every payoff printed in the "
              "prompt is at most one",
              y=-0.115)

    S.save(fig, "f_persona")


if __name__ == "__main__":
    main()
