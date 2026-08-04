"""FR1-FR4: design, payoff-scale invariance, outcome composition, language.

Every figure here is at most three panels wide, sized to 183 mm, and carries
its own qualifier line.  See `pdlib/natstyle.py` for the house rules.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from pdlib.ingest import payoff_matrix
from pdlib.metrics import grouped_ci, scale_slope
from pdlib.natstyle import (CMAP_SEQ, DATADIR, FRONTIER, HORIZON, INK, INK2,
                            LANG_ORDER, LANG_SHORT, MODEL_C, MODEL_LABEL,
                            MODEL_M, MUTED, OUTCOME_C, OUTCOME_LABEL,
                            OUTCOME_ORDER, PAGE, RULE, SCALE_ORDER, SPINE,
                            STACK_C, STACK_COLS, STACK_LABEL, STACK_ORDER,
                            TABDIR, W2, annotate_heatmap, bars, caption,
                            colorbar, errorbars, figure, finalize, hgrid,
                            model_legend, model_line, refline, save,
                            scale_axis, use_journal_style)

use_journal_style()

SEED = 0
N_BOOT = 2000


# ==========================================================================
# FR1 -- the game and the design
# ==========================================================================
def fig_design(games):
    m = payoff_matrix("frontier")
    fig = figure(W2, 2.35)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35])

    # (a) the stage game -----------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-1.30, 2.0)
    ax.set_ylim(-0.95, 2.05)
    ax.axis("off")

    cell = {("C", "C"): (m["R"], m["R"]), ("C", "D"): (m["S"], m["T"]),
            ("D", "C"): (m["T"], m["S"]), ("D", "D"): (m["P"], m["P"])}
    tint = {("C", "C"): "#dcebf6", ("C", "D"): "#f6f6f6",
            ("D", "C"): "#f6f6f6", ("D", "D"): "#f8e2d3"}
    for i, own in enumerate(("C", "D")):
        for j, opp in enumerate(("C", "D")):
            x, y = j, 1 - i
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1.0, 1.0,
                                   facecolor=tint[(own, opp)], edgecolor=PAGE,
                                   linewidth=1.2, zorder=1))
            a, b = cell[(own, opp)]
            ax.text(x, y + 0.13, f"{a:g},  {b:g}", ha="center", va="center",
                    fontsize=8.5, color=INK, zorder=3)
            ax.text(x, y - 0.20, own + opp, ha="center", va="center",
                    fontsize=6, color=MUTED, zorder=3)

    for j, lab in enumerate(("cooperate", "defect")):
        ax.text(j, 1.66, lab, ha="center", va="center", fontsize=6.5, color=INK2)
    ax.text(0.5, 1.98, "opponent", ha="center", va="center", fontsize=6.5,
            color=MUTED, style="italic")
    for i, lab in enumerate(("cooperate", "defect")):
        ax.text(-0.60, 1 - i, lab, ha="right", va="center", fontsize=6.5,
                color=INK2)
    ax.text(-1.20, 0.5, "focal player", ha="center", va="center", fontsize=6.5,
            color=MUTED, style="italic", rotation=90)

    note = (f"$T$>$R$>$P$>$S$ and $2R$>$T$+$S$    "
            f"greed $=(T-R)/(T-S)=${m['greed']:.2f}    "
            f"fear $=(P-S)/(T-S)=${m['fear']:.2f}")
    ax.text(-1.30, -0.60, note, ha="left", va="center", fontsize=5.8, color=MUTED)
    ax.text(-1.30, -0.88, "payoffs shown at $\\lambda=1$; every cell is "
            "multiplied by $\\lambda \\in \\{0.1, 1, 10\\}$",
            ha="left", va="center", fontsize=5.8, color=MUTED)
    ax.set_title("Stage game (one round)", color=INK, pad=12)

    # (b) the factorial ladder ----------------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    # (name, cell labels, fills, label colours, multiplier, half-height)
    rows = [
        ("model", [MODEL_LABEL[m_].split()[0] for m_ in FRONTIER],
         [MODEL_C[m_] for m_ in FRONTIER], [PAGE] * 4, "4", 0.24),
        ("horizon", [HORIZON[m_] for m_ in FRONTIER],
         ["#ededed" if HORIZON[m_] == "unknown" else "#4d4d4d" for m_ in FRONTIER],
         [INK2 if HORIZON[m_] == "unknown" else PAGE for m_ in FRONTIER],
         None, 0.15),
        ("payoff scale $\\lambda$", [f"$\\times${s:g}" for s in SCALE_ORDER],
         ["#9fbfd6", "#6b9dbe", "#3c7ea6"], [INK, INK, PAGE], "3", 0.24),
        ("language", [LANG_SHORT[l] for l in LANG_ORDER],
         ["#cfd9c8", "#b8c8ad", "#a1b792", "#8aa678", "#73955d"],
         [INK, INK, INK, INK, PAGE], "5", 0.24),
        ("persona pairing", ["C-C", "C-S", "S-C", "S-S"],
         ["#dcc9b0", "#cdb193", "#be9976", "#af8159"],
         [INK, INK, INK, PAGE], "4", 0.24),
        ("replicate", [""] * 10, ["#e8e8e8", "#dedede"] * 5, [INK] * 10,
         "10", 0.24),
    ]
    ypos = np.cumsum([0.0] + [0.62 if h > 0.2 else 0.34 for _, _, _, _, _, h in rows])
    ax.set_xlim(0, 1)
    ax.set_ylim(ypos[-1] + 0.10, -0.52)
    ax.axis("off")
    for (name, labels, cols, tcols, k, hh), y in zip(rows, ypos):
        w = 1 / len(labels)
        for i, (lab, c, tc) in enumerate(zip(labels, cols, tcols)):
            ax.add_patch(Rectangle((i * w, y - hh), w, 2 * hh, facecolor=c,
                                   edgecolor=PAGE, linewidth=0.8, zorder=2))
            if lab:
                ax.text(i * w + w / 2, y, lab, ha="center", va="center",
                        fontsize=5.4 if hh < 0.2 else 5.8, color=tc, zorder=3)
        ax.text(-0.015, y, name, ha="right", va="center", fontsize=6.4,
                color=INK2)
        if k:
            ax.text(1.015, y, f"$\\times${k}", ha="left", va="center",
                    fontsize=6.4, color=MUTED)
        else:
            ax.text(1.015, y, "confound", ha="left", va="center", fontsize=5.6,
                    color=MUTED, style="italic")
    ax.text(0.5, ypos[-1] - 0.10,
            "2,400 dyads  $\\times$  10 rounds  $\\times$  2 agents  =  "
            "48,000 binary decisions",
            ha="center", va="center", fontsize=6.4, color=INK)
    ax.set_title("Fully crossed design", color=INK, pad=12)

    finalize(fig, [pa, pb], ["a", "b"], dx=-0.005)
    caption(fig, "Horizon condition differs by arm: the Gemini games were run "
                 "with the round count disclosed to the agents, the other three "
                 "arms with the horizon hidden.")
    save(fig, "FR1_design")


# ==========================================================================
# FR2 -- payoff-scale invariance
# ==========================================================================
def fig_scale(games):
    fig = figure(W2, 2.6)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])

    # (a) cooperation vs lambda ---------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    ci = grouped_ci(games, ["model", "scale_nominal"], "coop_rate", n_boot=N_BOOT)
    for mdl in FRONTIER:
        sub = ci[ci.model == mdl].sort_values("scale_nominal")
        model_line(ax, np.log10(sub.scale_nominal), sub["mean"], mdl,
                   lo=sub["lo"], hi=sub["hi"])
    refline(ax, 0.5, "indifference")
    scale_axis(ax)
    ax.set_ylabel("cooperation rate")
    ax.set_ylim(0.10, 0.72)
    ax.set_yticks(np.arange(0.1, 0.75, 0.1))
    model_legend(ax, ncol=2, loc="upper center", bbox=(0.5, 1.015))
    ax.set_title("A rescaling that cannot matter, but does", pad=16)

    # (b) slope on log10(lambda), with the null at zero -----------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax, axis="x")
    rows = []
    for mdl in FRONTIER:
        s = scale_slope(games[games.model == mdl], "coop_rate", n_boot=N_BOOT,
                        seed=SEED)
        per = ci[ci.model == mdl].set_index("scale_nominal")["mean"]
        rows.append({"model": mdl, "slope": s["slope"], "lo": s["lo"],
                     "hi": s["hi"], "swing": per.max() - per.min(),
                     "argmin": per.idxmin(), "argmax": per.idxmax(),
                     "horizon": HORIZON[mdl]})
    sl = pd.DataFrame(rows)
    sl.to_csv(TABDIR / "T_FR05_scale_sensitivity.csv", index=False)

    y = np.arange(len(sl))[::-1]
    ax.axvline(0.0, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    ax.text(0.0, -0.62, " no scale effect", ha="left", va="center",
            fontsize=5.6, color=MUTED)
    for yi, r in zip(y, sl.itertuples()):
        ax.hlines(yi, r.lo, r.hi, color=MODEL_C[r.model], lw=1.1, zorder=3)
        ax.plot(r.slope, yi, marker=MODEL_M[r.model], ms=4.2,
                mfc=MODEL_C[r.model], mec=PAGE, mew=0.6, ls="none", zorder=4)
        # right-aligned in a fixed column: anchoring to each interval's own
        # end makes the four labels ragged and can butt a label against a
        # line cap
        ax.text(0.995, yi, f"swing {r.swing:.2f}",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=5.6, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m] for m in sl.model])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.95, len(sl) - 0.35)
    ax.set_xlim(sl.lo.min() - 0.012, 0.075)
    ax.set_xticks([-0.15, -0.10, -0.05, 0.0])
    ax.set_xlabel("change in cooperation rate per decade of $\\lambda$")
    ax.set_title("Slope, 95% bootstrap CI", pad=16)

    finalize(fig, [pa, pb], ["a", "b"], dx=-0.010)
    caption(fig, "Bands and intervals are 95% bootstrap CIs resampling whole "
                 "dyads (n = 200 per model $\\times$ $\\lambda$ cell). A positive "
                 "affine rescaling leaves the game strategically identical, so "
                 "any non-zero slope is a departure from the game-theoretic reading.")
    save(fig, "FR2_payoff_scale")
    return sl


# ==========================================================================
# FR3 -- what the dyads actually produce
# ==========================================================================
def fig_outcomes(games):
    fig = figure(W2, 2.75)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])

    # (a) stacked joint-outcome composition ----------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    comp = (games.groupby(["model", "scale_nominal"])
            [["cc_rate", "cd_rate", "dc_rate", "dd_rate"]].mean())
    # one row per (model, lambda), with a gap between models
    cells, ypos = [], []
    y = 0.0
    for mdl in FRONTIER:
        for s in SCALE_ORDER:
            cells.append((mdl, s))
            ypos.append(y)
            y += 1
        y += 0.65
    ypos = np.array(ypos)
    ylab = [f"$\\times${s:g}" for _, s in cells]

    left = np.zeros(len(ypos))
    for code in STACK_ORDER:
        v = np.array([comp.loc[key, STACK_COLS[code]].sum() for key in cells])
        ax.barh(ypos, v, left=left, height=0.72, color=STACK_C[code],
                edgecolor=PAGE, linewidth=0.6, zorder=3,
                label=STACK_LABEL[code])
        for yi, (l, vv) in enumerate(zip(left, v)):
            if vv > 0.09:
                ax.text(l + vv / 2, ypos[yi], f"{vv:.2f}", ha="center",
                        va="center", fontsize=5.4, zorder=4,
                        color=PAGE if code in ("CC", "DD") else INK)
        left += v

    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("share of rounds")
    ax.set_ylim(ypos.max() + 0.75, -0.75)
    for k, mdl in enumerate(FRONTIER):
        yc = ypos[k * 3: k * 3 + 3].mean()
        ax.text(-0.175, yc, MODEL_LABEL[mdl].split()[0],
                transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=6.4, color=INK, rotation=90)
        ax.plot([-0.125, -0.125], [ypos[k * 3] - 0.42, ypos[k * 3 + 2] + 0.42],
                transform=ax.get_yaxis_transform(), color=MODEL_C[mdl], lw=1.6,
                clip_on=False, solid_capstyle="butt")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.135),
              handlelength=1.0, columnspacing=0.9)
    ax.set_title("Joint-outcome composition", pad=24)

    # (b) the cooperation-efficiency plane -----------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    pm = payoff_matrix("frontier")

    def bench(p):
        """Efficiency of a dyad that cooperates at rate p but never mis-times:
        every round is either CC or DD, so payoff = pR + (1-p)P."""
        return (p * pm["R"] + (1 - p) * pm["P"]) / pm["R"]

    pts = (games.groupby(["model", "scale_nominal"])
           .agg(coop=("coop_rate", "mean"), eff=("efficiency", "mean"))
           .reset_index())
    pts["benchmark"] = bench(pts.coop)
    pts["premium"] = pts.eff - pts.benchmark
    pts.to_csv(TABDIR / "T_FR06_coop_efficiency.csv", index=False)

    xlo, xhi = pts.coop.min() - 0.05, pts.coop.max() + 0.05
    xx = np.linspace(xlo, xhi, 50)
    ax.plot(xx, bench(xx), color=MUTED, lw=0.6, ls=(0, (2.5, 2)), zorder=1)

    size = {0.1: 2.6, 1.0: 4.0, 10.0: 5.6}
    for mdl in FRONTIER:
        sub = pts[pts.model == mdl].sort_values("scale_nominal")
        ax.plot(sub.coop, sub.eff, color=MODEL_C[mdl], lw=0.7, alpha=0.5,
                zorder=2)
        ax.vlines(sub.coop, sub.benchmark, sub.eff, color=MODEL_C[mdl],
                  lw=0.6, alpha=0.6, zorder=2)
        for r in sub.itertuples():
            ax.plot(r.coop, r.eff, marker=MODEL_M[mdl], ms=size[r.scale_nominal],
                    mfc=MODEL_C[mdl], mec=PAGE, mew=0.6, ls="none", zorder=4)

    ax.set_xlabel("cooperation rate")
    ax.set_ylabel("payoff efficiency (mean payoff / $R$)")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(bench(xlo) - 0.03, pts.eff.max() + 0.055)
    # both notes go in the two empty corners of this plane
    ax.annotate("coordinated benchmark\n(every round CC or DD)",
                xy=(0.60, bench(0.60)), xytext=(0.63, bench(0.60) - 0.075),
                ha="center", va="top", fontsize=5.6, color=MUTED,
                arrowprops=dict(arrowstyle="-", lw=0.45, color=MUTED,
                                shrinkA=1, shrinkB=1))
    ax.text(xlo + 0.012, pts.eff.max() + 0.030,
            "vertical drop = anti-coordination premium", ha="left", va="top",
            fontsize=5.6, color=INK2)
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=v, mfc=MUTED,
                          mec=PAGE, mew=0.5, label=f"$\\times${k:g}")
               for k, v in size.items()]
    ax.legend(handles=handles, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), handlelength=0.8, columnspacing=0.7)
    ax.set_title("Payoff runs ahead of cooperation", pad=24)

    finalize(fig, [pa, pb], ["a", "b"], dx=-0.010)
    caption(fig, "Marker area in b encodes $\\lambda$. The dashed curve is what a "
                 "dyad would earn if it reached the same cooperation rate without "
                 "ever mis-timing, so every round were CC or DD. All twelve cells "
                 "sit above it: the frontier models' payoff advantage comes from "
                 "anti-coordinated CD/DC rounds, which pay $(T+S)/2 = 5$ against "
                 "$P = 2$ for mutual defection, not from mutual cooperation.")
    save(fig, "FR3_outcomes")


# ==========================================================================
# FR4 -- language
# ==========================================================================
def _perm_gap(values, labels, n_perm=5000, seed=SEED):
    """Observed max-min group gap and its permutation null (vectorised)."""
    codes, uniq = pd.factorize(labels)
    k = len(uniq)
    cnt = np.bincount(codes, minlength=k).astype(float)
    obs_means = np.bincount(codes, weights=values, minlength=k) / cnt
    obs = obs_means.max() - obs_means.min()
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    v = np.asarray(values, dtype=float)
    for i in range(n_perm):
        c = rng.permutation(codes)
        mu = np.bincount(c, weights=v, minlength=k) / np.bincount(c, minlength=k)
        null[i] = mu.max() - mu.min()
    return obs, float((null >= obs).mean()), float(np.percentile(null, 95))


def fig_language(games):
    fig = figure(W2, 2.5)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    # (a) model x language heatmap -------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    piv = (games.pivot_table(index="model", columns="language",
                             values="coop_rate", aggfunc="mean")
           .reindex(index=FRONTIER, columns=LANG_ORDER))
    piv.to_csv(TABDIR / "T_FR07_language_means.csv")
    vmin, vmax = 0.10, 0.70
    im = ax.imshow(piv.to_numpy(), cmap=CMAP_SEQ, vmin=vmin, vmax=vmax,
                   aspect="auto")
    annotate_heatmap(ax, piv.to_numpy(), thresh=0.47, size=6.2)
    ax.set_xticks(range(len(LANG_ORDER)))
    ax.set_xticklabels([LANG_LABEL_SHORT[l] for l in LANG_ORDER], fontsize=6.2)
    ax.set_yticks(range(len(FRONTIER)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in FRONTIER], fontsize=6.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    colorbar(fig, im, ax, label="cooperation rate")
    ax.set_title("Same game, five prompt languages", pad=6)

    # (b) disparity vs its permutation null ----------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    rows = []
    for mdl in FRONTIER:
        sub = games[games.model == mdl]
        obs, p, null95 = _perm_gap(sub.coop_rate.to_numpy(),
                                   sub.language.to_numpy(), n_perm=5000)
        lo = piv.loc[mdl].idxmin()
        hi = piv.loc[mdl].idxmax()
        rows.append({"model": mdl, "gap": obs, "p_perm": p, "null_q95": null95,
                     "min_lang": lo, "max_lang": hi})
    gaps = pd.DataFrame(rows)
    gaps.to_csv(TABDIR / "T_FR08_language_disparity.csv", index=False)

    x = np.arange(len(gaps))
    bars(ax, x, gaps.gap, [MODEL_C[m] for m in gaps.model], width=0.56)
    for xi, r in zip(x, gaps.itertuples()):
        ax.hlines(r.null_q95, xi - 0.34, xi + 0.34, color=INK, lw=0.7,
                  ls=(0, (2, 1.6)), zorder=5)
        star = ("$P$ < 0.001" if r.p_perm < 1e-3 else f"$P$ = {r.p_perm:.3f}")
        ax.text(xi, r.gap + 0.020, star, ha="center", va="bottom", fontsize=5.6,
                color=INK2)
        # widest-vs-narrowest language pair, inside the bar so it cannot
        # collide with the model tick labels below the axis
        ax.text(xi, r.gap - 0.006,
                f"{LANG_SHORT[r.max_lang]}$-${LANG_SHORT[r.min_lang]}",
                ha="center", va="top", fontsize=5.6, color=PAGE, zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in gaps.model],
                       fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("language disparity (max $-$ min)")
    ax.set_ylim(0, max(0.24, gaps.gap.max() * 1.30))
    ax.set_xlim(-0.62, len(gaps) - 0.38)
    ax.plot([], [], color=INK, lw=0.7, ls=(0, (2, 1.6)),
            label="95th pct. of permutation null")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), handlelength=1.6)
    ax.set_title("Language disparity against chance", pad=14)

    finalize(fig, [pa, pb], ["a", "b"], dx=-0.010)
    caption(fig, "Permutation null shuffles the language label across dyads "
                 "within a model (5,000 draws), holding the number of dyads per "
                 "language fixed. The pair inside each bar names that model's "
                 "highest- and lowest-cooperation language.")
    save(fig, "FR4_language")
    return gaps


LANG_LABEL_SHORT = {"en": "English", "fr": "French", "vn": "Viet.",
                    "cn": "Chinese", "ar": "Arabic"}


def main():
    games = pd.read_parquet(DATADIR / "frontier_games.parquet")
    fig_design(games)
    sl = fig_scale(games)
    fig_outcomes(games)
    gaps = fig_language(games)
    print()
    print(sl.to_string(index=False))
    print()
    print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
