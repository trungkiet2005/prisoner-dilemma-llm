"""Which canonical rule the transcripts match, against the payoff scale.

Multiplying every payoff by lambda leaves the game unchanged, so the reading
the models give it should be unchanged too.  It is not, and the earlier figures
already show that the level of cooperation moves.  This figure makes the
stronger point: what moves is the composition.  The scale changes which of the
four canonical memory-one rules the play most resembles, and it does so by a
different route in each model, which is why the pooled curve understates it.

Panel a draws one stacked band per model over the ten scales, each stack the
four label shares summing to one hundred per cent of that model's four hundred
agent-games at that scale.  Reading a stack top to bottom is reading the
composition; a band that swells or shrinks from left to right is the scale
acting on the reading of the matrix.  Grok 4.20 turns over almost completely,
from two thirds AllD at the smallest scale to four fifths AllC at the largest,
while Qwen3 235B-A22B holds the same mixture throughout.

Panel b puts one number on that movement.  The total variation distance between
a model's mix at the smallest scale and its mix at the largest is the share of
agent-games that would have to be relabelled to turn one mixture into the
other, so it is on the same footing as the shares themselves.  The dashed rule
is the sampling floor: with four hundred games per cell, two draws from one
fixed mixture land this far apart five per cent of the time, so a model at or
under the rule has not been shown to move at all.

Panel c carries the caveat the reading rests on, and it must be read before
panel a.  A label is deduced when exactly one canonical rule reproduces the
whole trajectory, ambiguous when several do, and attributed when none does and
the label comes from the learned read-out instead.  The AllC and AllD bands are
anchored in part by exact matching.  The Tit-for-Tat and Win-Stay-Lose-Shift
bands are almost entirely attributions, so a movement inside them says the play
came to resemble that rule more closely, not that the model was running it.

Shares are read from the published tables; nothing here recomputes them.  The
noise floor and the interval on panel b are the one thing computed here, by
resampling the four hundred agent-games behind each cell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S      # noqa: E402
import data as D       # noqa: E402

COLS = {"ALLC": "AllC", "TFT": "TFT", "WSLS": "WSLS", "ALLD": "AllD"}
N_CELL = 400           # agent-games behind one model-by-scale cell
LO, HI = 0.01, 1000.0
SEED = 20260906


def mix_matrix(t, model):
    s = t[t.model == model].set_index("scale")
    return {k: s.loc[S.SCALES, COLS[k]].to_numpy() / 100.0 for k in S.STRAT_ORDER}


def movement(t, model, *, n=6000):
    """Total variation distance between the extreme scales, and its noise floor."""
    s = t[t.model == model].set_index("scale")
    p = s.loc[LO, [COLS[k] for k in S.STRAT_ORDER]].to_numpy() / 100.0
    q = s.loc[HI, [COLS[k] for k in S.STRAT_ORDER]].to_numpy() / 100.0
    obs = 0.5 * np.abs(p - q).sum()
    rng = np.random.default_rng(SEED)
    a = rng.multinomial(N_CELL, p, size=n) / N_CELL
    b = rng.multinomial(N_CELL, q, size=n) / N_CELL
    draws = 0.5 * np.abs(a - b).sum(axis=1)
    pooled = (p + q) / 2
    c = rng.multinomial(N_CELL, pooled, size=n) / N_CELL
    d = rng.multinomial(N_CELL, pooled, size=n) / N_CELL
    null = 0.5 * np.abs(c - d).sum(axis=1)
    return (obs, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)),
            float(np.quantile(null, 0.95)))


def main():
    t = D.table("T12_label_mix.csv")
    t = t[t.model.isin(S.MODEL_ORDER)]
    prov = D.table("T16_label_provenance.csv")
    prov = prov[prov.panel == "main_five"].set_index("label")

    move = {m: movement(t, m) for m in S.MODEL_ORDER}
    floor = float(np.mean([v[3] for v in move.values()]))
    print("  total variation between the extreme scales, per model")
    for m in S.MODEL_ORDER:
        o, lo, hi, nl = move[m]
        print(f"    {S.MODEL_SHORT[m]:11s} TVD={o:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"noise floor {nl:.3f}  {'moves' if lo > nl else 'at the floor'}")
    print(f"  pooled noise floor {floor:.3f}")
    print("  label provenance, five main models")
    for k in S.STRAT_ORDER:
        row = prov.loc[COLS[k]]
        print(f"    {COLS[k]:5s} n={int(row.n):5d}  deduced {row.deduced:5.1f}%  "
              f"ambiguous {row.ambiguous:5.1f}%  attributed {row.unmatched:5.1f}%")

    fig = plt.figure(figsize=(S.FULL, 3.05))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.86], hspace=0.80,
                          wspace=0.24, left=0.062, right=0.995,
                          top=0.905, bottom=0.135)
    tops = [fig.add_subplot(gs[0, i]) for i in range(5)]
    axB = fig.add_subplot(gs[1, 0:2])
    axC = fig.add_subplot(gs[1, 3:5])

    # --- a: the mix of each model, stacked, one panel per model ------------
    lam = np.array(S.SCALES, dtype=float)
    for i, (ax, m) in enumerate(zip(tops, S.MODEL_ORDER)):
        mix = mix_matrix(t, m)
        base = np.zeros_like(lam)
        for k in S.STRAT_ORDER:
            ax.fill_between(lam, base, base + mix[k], color=S.STRAT_C[k],
                            lw=0, zorder=2)
            base = base + mix[k]
        S.scale_axis(ax, band=False, label=None, ticks=False)
        ax.set_xticks([0.01, 1, 1000])
        ax.set_xticklabels(["0.01", "1", "1000"])
        ax.set_ylim(0, 1)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        if i == 0:
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_yticklabels(["0", "50", "100%"])
            ax.set_ylabel("share of agent-games")
        else:
            ax.set_yticks([])
        ax.set_xlabel(S.MODEL_SHORT[m], color=S.MODEL_C[m], fontsize=S.FS_NOTE,
                      fontweight="bold", labelpad=8.0)
        # the two scales at which every payoff printed is at most one
        ax.plot([0.006, 1.0], [-0.20, -0.20], color=S.HAIRLINE, lw=2.2,
                solid_capstyle="butt", clip_on=False, zorder=5)

    S.panel(tops[0], "a", "the mix moves, model by model")
    tops[2].text(0.5, -0.46, r"payoff scale $\lambda$", transform=tops[2].transAxes,
                 ha="center", va="top", fontsize=S.FS_NOTE, color=S.MUTED)

    # Name the four bands in place, in the panel where all four are legible.
    key = mix_matrix(t, "GPT-5.4-Nano")
    acc = 0.0
    for k in S.STRAT_ORDER:
        h = key[k][-1]
        tops[1].text(900, acc + h / 2, S.STRAT_LABEL[k], fontsize=S.FS_NOTE,
                     color=S.SURFACE, ha="right", va="center", zorder=6,
                     fontweight="bold")
        acc += h

    # --- b: how far the mix travels between the extreme scales -------------
    order = sorted(S.MODEL_ORDER, key=lambda m: move[m][0])
    ys = np.arange(len(order))
    for y, m in zip(ys, order):
        o, lo, hi, _ = move[m]
        axB.hlines(y, 0, o, color=S.MODEL_C[m], lw=1.1, zorder=2)
        axB.hlines(y, lo, hi, color=S.MODEL_C[m], lw=2.6, alpha=0.30, zorder=3)
        S.dot(axB, o, y, color=S.MODEL_C[m], marker=S.MODEL_M[m], size=24)
        axB.text(o + 0.022, y, f"{o:.2f}", fontsize=S.FS_NOTE,
                 color=S.INK_2, ha="left", va="center")
    axB.axvline(floor, color=S.INK_2, lw=0.9, linestyle=(0, (3, 2)), zorder=4)
    axB.text(floor + 0.012, len(order) - 0.42, "sampling floor", fontsize=5.9,
             color=S.INK_2, ha="left", va="center")
    axB.set_yticks(ys)
    axB.set_yticklabels([S.MODEL_SHORT[m] for m in order])
    for lab, m in zip(axB.get_yticklabels(), order):
        lab.set_color(S.MODEL_C[m])
    S.strip(axB, grid_axis="x")
    axB.tick_params(axis="y", length=0)
    axB.set_ylim(-0.7, len(order) - 0.3)
    axB.set_xlim(0, 0.78)
    axB.set_xticks([0, 0.2, 0.4, 0.6])
    axB.set_xlabel("share relabelled between the extreme scales")
    S.panel(axB, "b", "four of five move, Qwen3 does not")

    # --- c: where the labels come from -------------------------------------
    for j, k in enumerate(S.STRAT_ORDER):
        row = prov.loc[COLS[k]]
        y = len(S.STRAT_ORDER) - 1 - j
        segs = [(row.deduced, dict(color=S.STRAT_C[k])),
                (row.ambiguous, dict(color=S.STRAT_C[k], alpha=0.38)),
                (row.unmatched, dict(color=S.SURFACE, hatch="////",
                                     edgecolor=S.STRAT_C[k], lw=0.5))]
        left = 0.0
        for w, kw in segs:
            axC.barh(y, w, left=left, height=0.62, zorder=3, **kw)
            left += w
        axC.text(101, y, f"n={int(row.n):,}", fontsize=5.9, color=S.MUTED,
                 ha="left", va="center")
    axC.set_yticks(range(len(S.STRAT_ORDER)))
    axC.set_yticklabels([S.STRAT_LABEL[k] for k in reversed(S.STRAT_ORDER)])
    for lab, k in zip(axC.get_yticklabels(), reversed(S.STRAT_ORDER)):
        lab.set_color(S.STRAT_C[k])
        lab.set_fontweight("bold")
    S.strip(axC, grid_axis="x")
    axC.tick_params(axis="y", length=0)
    axC.set_xlim(0, 100)
    axC.set_xticks([0, 25, 50, 75, 100])
    axC.set_xticklabels(["0", "25", "50", "75", "100%"])
    axC.set_ylim(-0.7, len(S.STRAT_ORDER) - 0.05)
    axC.set_xlabel("how the label was arrived at")
    S.panel(axC, "c", "TFT and WSLS are resemblances")

    top = prov.loc["AllC"]
    marks = [(top.deduced / 2, "one rule fits"),
             (top.deduced + top.ambiguous / 2, "several fit"),
             (top.deduced + top.ambiguous + top.unmatched / 2, "none fits")]
    for x, txt in marks:
        axC.text(x, len(S.STRAT_ORDER) - 1 + 0.44, txt, fontsize=5.9,
                 color=S.INK_2, ha="center", va="bottom")

    fig.text(0.5, -0.005,
             "five models, ten payoff scales, 400 agent-games per cell, 20,000 in all; "
             "the grey rule under a marks the scales at which every payoff printed is "
             "at most 1\n"
             "the pale bar on b is a 95% bootstrap interval, the dashed rule the 95th "
             "percentile of the distance between two draws from one fixed mixture",
             ha="center", va="top", fontsize=S.FS_NOTE, color=S.MUTED,
             linespacing=1.5)

    S.save(fig, "f_strategy_mix")


if __name__ == "__main__":
    main()
