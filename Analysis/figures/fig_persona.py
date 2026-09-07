"""Telling a model it is cooperative does not have a fixed effect.

Every agent is given a persona in its prompt, either that it is a cooperative
player or that it is a selfish one, and the natural expectation is that the
cooperative persona raises cooperation by some amount that is a property of the
model. It does not. The persona effect, defined as mean cooperation under the
cooperative persona minus mean cooperation under the selfish one, changes with the
payoff scale in every model, and in two of the five it changes sign. When every
printed payoff is at most one unit, the cooperative persona in Claude and in
Gemini 3.5 produces less cooperation than the selfish persona does, which is the
opposite of the instruction. Once the payoffs are printed at unit scale or larger,
both models obey the instruction. The persona is therefore not an independent
knob; it is gated by how the numbers in the prompt are written.

Panels:
  a  The persona effect against payoff scale, with the sub-unit regime (lambda <= 0.1)
     shaded in grey and a zero rule. Claude and Gemini 3.5 cross zero at the regime edge.
  b  Regime comparison: sub-unit (lambda <= 0.1, open markers) vs supra-unit
     (lambda >= 0.25, filled markers) with 95% bootstrap confidence intervals for each model.
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


def nudge(vals, minsep, lo=None, hi=None):
    """Push labels apart just enough that two adjacent ones stay readable."""
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
    gate = D.table("T07_persona_gating.csv").set_index("model")
    by_scale = D.table("T07_persona_by_scale.csv")

    curves = {m: (by_scale[by_scale.model == m]
                  .set_index("scale").persona_effect.reindex(S.SCALES))
              for m in S.MODEL_ORDER}

    fig = plt.figure(figsize=(S.FULL, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.46,
                          left=0.075, right=0.985, top=0.88, bottom=0.15)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # --- a: the effect against scale ---------------------------------------
    axA.axvspan(0.007, 0.158, color=S.BAND, lw=0, zorder=0)
    axA.text(0.033, -0.65, "sub-unit regime\n" + r"($\lambda \leq 0.1$)",
             fontsize=S.FS_NOTE, color=S.MUTED, ha="center", va="center",
             linespacing=1.15)

    S.zero_rule(axA, 0.0, lw=0.9)
    for m in S.MODEL_ORDER:
        vals = curves[m].to_numpy()
        axA.plot(S.SCALES, vals, color=S.MODEL_C[m], lw=1.3, zorder=3)
        axA.scatter(S.SCALES, vals, s=16, marker=S.MODEL_M[m],
                    color=S.MODEL_C[m], linewidths=0.6, edgecolors=S.SURFACE, zorder=4)

    # zero crossings for Claude and Gemini 3.5
    for m in ("Claude-Haiku-4.5", "Gemini-3.5-Flash-Lite"):
        below = curves[m].loc[0.1]
        above = curves[m].loc[0.25]
        x = 0.1 * (0.25 / 0.1) ** (-below / (above - below))
        S.dot(axA, x, 0.0, color=S.MODEL_C[m], marker=S.MODEL_M[m], size=32,
              filled=False, zorder=6)

    axA.text(0.012, 0.44,
             "open markers at zero:\nClaude and Gemini 3.5 cross\nzero at regime boundary",
             fontsize=5.8, color=S.INK_2, ha="left", va="top", linespacing=1.2)

    S.scale_axis(axA, band=False)
    axA.set_ylim(-1.02, 0.52)
    axA.set_xlim(0.007, 1400)
    axA.set_ylabel("persona effect\n(cooperative $-$ selfish)")
    S.strip(axA, grid_axis="y")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=S.MODEL_C[m], marker=S.MODEL_M[m], lw=1.2, ms=4.2,
                      label=S.MODEL_SHORT[m]) for m in S.MODEL_ORDER]
    axA.legend(handles=handles, loc="upper right", ncol=2, frameon=True,
               framealpha=0.92, edgecolor=S.GRID, fontsize=5.8, handletextpad=0.3,
               columnspacing=0.8)
    S.panel(axA, "a", "Claude and Gemini 3.5 switch sign above the unit", pad=7)

    # --- b: regime shift dumbbell / forest plot ----------------------------
    order = S.MODEL_ORDER[::-1]  # display top-to-bottom
    ys = np.arange(len(order))

    S.zero_rule(axB, 0.0, vertical=True, lw=0.9)
    for y, m in zip(ys, order):
        row = gate.loc[m]
        sub = row.persona_effect_subunit
        sup = row.persona_effect_suprunit
        # line connecting the two regimes
        axB.plot([sub, sup], [y, y], color=S.MODEL_C[m], lw=1.2, zorder=2)
        # CI for sub-unit
        axB.plot([row.sub_lo, row.sub_hi], [y, y], color=S.MODEL_C[m], lw=2.4,
                 alpha=0.32, zorder=1)
        # CI for supra-unit
        axB.plot([row.sup_lo, row.sup_hi], [y, y], color=S.MODEL_C[m], lw=2.4,
                 alpha=0.32, zorder=1)
        # sub-unit point (open)
        axB.scatter([sub], [y], s=30, marker=S.MODEL_M[m], facecolor=S.SURFACE,
                    edgecolor=S.MODEL_C[m], linewidth=1.2, zorder=4)
        # supra-unit point (filled)
        axB.scatter([sup], [y], s=30, marker=S.MODEL_M[m], facecolor=S.MODEL_C[m],
                    edgecolor=S.MODEL_C[m], linewidth=1.2, zorder=4)

        # shift label
        axB.text(max(sub, sup) + 0.045, y, f"{row['shift']:+.2f}",
                 fontsize=5.8, color=S.INK_2, ha="left", va="center")

    axB.set_yticks(ys)
    axB.set_yticklabels([S.MODEL_SHORT[m] for m in order])
    for lab, m in zip(axB.get_yticklabels(), order):
        lab.set_color(S.MODEL_C[m])
        lab.set_fontweight("bold")

    axB.set_xlim(-1.05, 0.52)
    axB.set_xticks([-1.0, -0.5, 0.0, 0.5])
    axB.set_ylim(-0.7, len(order) - 0.2)
    axB.set_xlabel("persona effect (cooperative $-$ selfish)")
    S.strip(axB, grid_axis="x")
    axB.tick_params(axis="y", length=0)

    # Header legend on b
    axB.scatter([-0.95], [len(order) - 0.45], s=26, marker="o", facecolor=S.SURFACE,
                edgecolor=S.INK_2, linewidth=1.1)
    axB.text(-0.90, len(order) - 0.45, "sub-unit", fontsize=5.8, color=S.INK_2,
             ha="left", va="center")
    axB.scatter([-0.50], [len(order) - 0.45], s=26, marker="o", facecolor=S.INK_2,
                edgecolor=S.INK_2, linewidth=1.1)
    axB.text(-0.45, len(order) - 0.45, "supra-unit (shift shown)", fontsize=5.8, color=S.INK_2,
             ha="left", va="center")

    S.panel(axB, "b", "regime shift: sub-unit vs supra-unit", pad=7)

    S.caption(fig,
              "20,000 agent-games from 10,000 dyads across five models and ten payoff scales; "
              "shading on a and open markers on b denote the sub-unit regime (lambda <= 0.1);\n"
              "thick bars on b are 95% bootstrap intervals; values right of markers are the supra minus sub-unit shift",
              y=-0.01)

    S.save(fig, "f_persona")


if __name__ == "__main__":
    main()
