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

    fig = plt.figure(figsize=(S.FULL, 3.50))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.30], hspace=0.76,
                          left=0.065, right=0.985, top=0.91, bottom=0.12)
    gs_top = gs[0, 0].subgridspec(1, 5, wspace=0.24)
    tops = [fig.add_subplot(gs_top[0, i]) for i in range(5)]
    gs_bot = gs[1, 0].subgridspec(1, 2, wspace=0.44, width_ratios=[1.08, 0.92])
    axB = fig.add_subplot(gs_bot[0, 0])
    axC = fig.add_subplot(gs_bot[0, 1])

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

    # --- b: game-theoretic composition shift (lambda=1000 minus lambda=0.01) ---
    order = sorted(S.MODEL_ORDER, key=lambda m: move[m][0], reverse=True)
    ys = np.arange(len(order))
    bar_h = 0.16
    offsets = [0.24, 0.08, -0.08, -0.24]  # AllC, TFT, WSLS, AllD top to bottom

    S.zero_rule(axB, 0.0, vertical=True, lw=0.8)
    for y, m in zip(ys, order):
        s_lo = t[(t.model == m) & (t.scale == LO)].iloc[0]
        s_hi = t[(t.model == m) & (t.scale == HI)].iloc[0]
        o = move[m][0]
        for off, k in zip(offsets, S.STRAT_ORDER):
            col = COLS[k]
            delta = float(s_hi[col] - s_lo[col])
            axB.barh(y + off, delta, height=bar_h, color=S.STRAT_C[k],
                     edgecolor="none", zorder=3)

    axB.set_yticks(ys)
    axB.set_yticklabels([f"{S.MODEL_SHORT[m]} ({move[m][0]:.2f})" for m in order])
    for lab, m in zip(axB.get_yticklabels(), order):
        lab.set_color(S.MODEL_C[m])
        lab.set_fontweight("bold")
    S.strip(axB, grid_axis="x")
    axB.tick_params(axis="y", length=0)
    axB.set_ylim(-0.65, len(order) - 0.05)
    axB.set_xlim(-68, 68)
    axB.set_xticks([-60, -30, 0, 30, 60])
    axB.set_xticklabels(["-60", "-30", "0", "+30", "+60%"])
    axB.set_xlabel(r"change in share: $\lambda=1000$ minus $\lambda=0.01$ ($D_{\mathrm{TV}}$ in parens)")
    S.panel(axB, "b", "composition shift between extreme scales")

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=S.STRAT_C[k], edgecolor="none",
                               label=S.STRAT_LABEL[k]) for k in S.STRAT_ORDER]
    axB.legend(handles=handles, loc="upper left", ncol=2, frameon=False,
               fontsize=5.8, handlelength=0.9, handletextpad=0.3, columnspacing=0.8)

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

    S.caption(fig,
             "five models, ten payoff scales, 400 agent-games per cell, 20,000 in all; "
             "the grey rule under a marks the scales at which every payoff printed is at most 1;\n"
             r"b shows net strategy share changes between $\lambda=1000$ and $\lambda=0.01$ with total variation distance in parentheses; "
             "c shows attribution provenance", y=-0.005)

    S.save(fig, "f_strategy_mix")


if __name__ == "__main__":
    main()
