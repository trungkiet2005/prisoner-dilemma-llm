"""The three main-text figures for the Interface Focus manuscript.

The frontier suite (`run_frontier.py`, FR1-FR26) is the full record and stays
the supplementary material.  A paper gets three figures, so each one here has
to carry a whole step of the argument rather than a single measurement:

    fig1_design       what the game is and what was crossed with what
    fig2_invariance   three manipulations that cannot matter, and all three do
    fig3_strategy     what the play is not: no unravelling, no memory-one rule

Panel-level numbers are read from `tables/T_FR*.csv`, which the frontier
pipeline writes, so a panel here can never drift from the corresponding
supplementary figure.  Run `python Analysis/run_frontier.py` first.

Captions are *not* baked into the canvas: in a manuscript the caption belongs
to the LaTeX float, and duplicating it inside the PDF would print it twice.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from pdlib.ingest import payoff_matrix
from pdlib.natstyle import (DATADIR, FRONTIER, HORIZON, INK, INK2, LANG_ORDER,
                            LANG_SHORT, MEMORY1, MODEL_C, MODEL_LABEL, MODEL_M,
                            MUTED, PAGE, SCALE_ORDER, TABDIR, W2, bars, figure,
                            finalize, hgrid, model_legend, model_line, refline,
                            shared_model_legend, use_journal_style)

use_journal_style()

PAPERFIG = Path(__file__).resolve().parents[2] / "paper" / "figures"
PAPERFIG.mkdir(parents=True, exist_ok=True)

STATE_ORDER = ["R", "S", "T", "P"]
STATE_LABEL = {"R": "after CC\n($R$)", "S": "after CD\n($S$)",
               "T": "after DC\n($T$)", "P": "after DD\n($P$)"}

ASSIGN_ORDER = ["exact", "ambiguous", "approx"]
ASSIGN_C = {"exact": "#0072b2", "ambiguous": "#9ecae1", "approx": "#d9d9d9"}
ASSIGN_LABEL = {"exact": "exactly one rule fits",
                "ambiguous": "several rules fit",
                "approx": "no rule fits: nearest neighbour"}


# The FAIRGAME templates do not state the objective identically in every
# language: en/fr/vn ask the agent to *minimise a penalty*, ar/cn ask it to
# *maximise a reward* while still calling the outcomes penalties.  Language is
# therefore confounded with objective framing, and a disparity measured over
# all five languages cannot separate the two.  MINIMISE is the subset that
# shares one framing, so the gap measured inside it is a language effect with
# framing held fixed.
MINIMISE = ("en", "fr", "vn")


def _perm_gap(values, labels, n_perm=5000, seed=0):
    """Observed max-min group gap and its permutation null."""
    codes, uniq = pd.factorize(labels)
    k = len(uniq)
    v = np.asarray(values, dtype=float)
    cnt = np.bincount(codes, minlength=k).astype(float)
    mu = np.bincount(codes, weights=v, minlength=k) / cnt
    obs = mu.max() - mu.min()
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        c = rng.permutation(codes)
        m = np.bincount(c, weights=v, minlength=k) / np.bincount(c, minlength=k)
        null[i] = m.max() - m.min()
    return obs, float((null >= obs).mean()), float(np.percentile(null, 95))


def language_controlled():
    """Language disparity over all five languages and within one framing."""
    g = pd.read_parquet(DATADIR / "frontier_games.parquet")
    rows = []
    for mdl in FRONTIER:
        s = g[g.model == mdl]
        f = s[s.language.isin(MINIMISE)]
        gap5, p5, q5 = _perm_gap(s.coop_rate.to_numpy(), s.language.to_numpy())
        gap3, p3, q3 = _perm_gap(f.coop_rate.to_numpy(), f.language.to_numpy())
        by = s.groupby(s.language.isin(MINIMISE)).coop_rate.mean()
        rows.append({"model": mdl,
                     "gap_all5": gap5, "p_all5": p5, "null95_all5": q5,
                     "gap_fixed_framing": gap3, "p_fixed_framing": p3,
                     "null95_fixed_framing": q3,
                     "mean_minimise": by[True], "mean_maximise": by[False],
                     "framing_gap": by[True] - by[False]})
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_PAPER_language_framing.csv", index=False)
    return out


def save(fig, name: str) -> None:
    """Vector PDF for the typesetter plus a 600 dpi PNG for drafts."""
    fig.savefig(PAPERFIG / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(PAPERFIG / f"{name}.png", dpi=600)
    plt.close(fig)
    print(f"  [fig] paper/figures/{name}.pdf + .png")


# ==========================================================================
# Figure 1 -- the stage game and the crossed design
# ==========================================================================
def fig_design():
    m = payoff_matrix("frontier")
    fig = figure(W2, 2.35)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35])

    # (a) the stage game -----------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-1.30, 2.0)
    ax.set_ylim(-0.95, 2.05)
    ax.axis("off")

    # The agent is shown penalties and told to minimise them, so the
    # cooperative action is the one with the *lower* symmetric number.  Rows
    # and columns are labelled with the option names the agent actually sees,
    # because the gap between those labels and the game-theoretic roles is
    # itself one of the findings.
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

    for j, lab in enumerate(("Option B\ncooperate", "Option A\ndefect")):
        ax.text(j, 1.72, lab, ha="center", va="center", fontsize=6.2,
                color=INK2, linespacing=1.35)
    ax.text(0.5, 2.10, "opponent", ha="center", va="center", fontsize=6.5,
            color=MUTED, style="italic")
    for i, lab in enumerate(("Option B\ncooperate", "Option A\ndefect")):
        ax.text(-0.58, 1 - i, lab, ha="right", va="center", fontsize=6.2,
                color=INK2, linespacing=1.35)
    ax.text(-1.24, 0.5, "focal player", ha="center", va="center", fontsize=6.5,
            color=MUTED, style="italic", rotation=90)

    ax.text(-1.30, -0.58,
            f"cells are $\\bf{{penalties}}$ to be minimised: $T$<$R$<$P$<$S$    "
            f"greed $=${m['greed']:.2f}    fear $=${m['fear']:.2f}",
            ha="left", va="center", fontsize=5.8, color=MUTED)
    ax.text(-1.30, -0.86, "shown at $\\lambda=1$; every cell is multiplied by "
            "$\\lambda \\in \\{0.1, 1, 10\\}$, which leaves the game unchanged",
            ha="left", va="center", fontsize=5.8, color=MUTED)
    ax.set_title("Stage game (one round)", color=INK, pad=16)

    # (b) the factorial ladder ----------------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
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
    ypos = np.cumsum([0.0] + [0.62 if h > 0.2 else 0.34 for *_, h in rows])
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
        ax.text(1.015, y, f"$\\times${k}" if k else "confound", ha="left",
                va="center", fontsize=6.4 if k else 5.6, color=MUTED,
                style="normal" if k else "italic")
    ax.text(0.5, ypos[-1] - 0.10,
            "2,400 dyads  $\\times$  10 rounds  $\\times$  2 agents  =  "
            "48,000 binary decisions",
            ha="center", va="center", fontsize=6.4, color=INK)
    ax.set_title("Fully crossed design", color=INK, pad=12)

    finalize(fig, [pa, pb], ["a", "b"], dx=-0.005)
    save(fig, "fig1_design")


# ==========================================================================
# Figure 2 -- three manipulations that cannot matter, and all three do
# ==========================================================================
def fig_invariance():
    fig = figure(W2, 2.45)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.00, 1.16, 1.08])

    # (a) payoff scale --------------------------------------------------------
    sl = pd.read_csv(TABDIR / "T_FR05_scale_sensitivity.csv")
    sl = sl.set_index("model").reindex(FRONTIER).reset_index()

    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax, axis="x")
    y = np.arange(len(sl))[::-1].astype(float)
    ax.axvline(0.0, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    for yi, r in zip(y, sl.itertuples()):
        c = MODEL_C[r.model]
        ax.hlines(yi, r.lo, r.hi, color=c, lw=1.1, zorder=3)
        ax.plot(r.slope, yi, marker=MODEL_M[r.model], ms=4.2, mfc=c, mec=PAGE,
                mew=0.6, ls="none", zorder=4)
        ax.text(0.995, yi, f"swing {r.swing:.2f}",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=5.6, color=MUTED)
    ax.text(0.0, -0.78, " no effect", ha="left", va="center", fontsize=5.6,
            color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in sl.model],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1.05, len(sl) - 0.35)
    lo, hi = min(sl.lo.min(), 0.0), max(sl.hi.max(), 0.0)
    pad = 0.06 * (hi - lo)
    ax.set_xlim(lo - pad, hi + 5.5 * pad)   # right margin holds the swing labels
    ax.set_xlabel("$\\Delta$ cooperation per\ndecade of $\\lambda$")
    ax.set_title("Payoff magnitude", pad=13)

    # (b) prompt language, with objective framing held fixed ------------------
    lang = language_controlled()

    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    x = np.arange(len(lang))
    w = 0.30
    for xi, r in zip(x, lang.itertuples()):
        c = MODEL_C[r.model]
        # ghost bar: all five languages, objective framing varying with them
        ax.bar(xi - w / 2, r.gap_all5, width=w * 0.92, color=c, alpha=0.30,
               edgecolor=PAGE, linewidth=0.5, zorder=3)
        # solid bar: the three languages that share one objective framing
        ax.bar(xi + w / 2, r.gap_fixed_framing, width=w * 0.92, color=c,
               edgecolor=PAGE, linewidth=0.5, zorder=3)
        for dx, gap, q in ((-w / 2, r.gap_all5, r.null95_all5),
                           (w / 2, r.gap_fixed_framing, r.null95_fixed_framing)):
            ax.hlines(q, xi + dx - w * 0.46, xi + dx + w * 0.46, color=INK,
                      lw=0.7, ls=(0, (2, 1.6)), zorder=5)
        p = r.p_fixed_framing
        if p < 1e-3:
            lab = "$P$<.001"
        else:
            lab = "$P$=" + (f"{p:.3f}" if p < 0.01 else f"{p:.2f}").lstrip("0")
        ax.text(xi + w / 2, max(r.gap_fixed_framing, r.null95_fixed_framing) + 0.008,
                lab, ha="center", va="bottom", fontsize=5.4,
                color=INK2 if p < 0.05 else MUTED)
    # panels a and c already spell the models out; a second full name here
    # would need two lines and the four would collide at this panel width
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m].split()[0] for m in lang.model],
                       fontsize=6.2)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("language disparity (max $-$ min)")
    ax.set_ylim(0, 0.215)
    ax.set_xlim(-0.60, len(lang) - 0.40)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, alpha=0.30,
                             edgecolor=PAGE, lw=0.5, label="all 5 languages"),
               plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=PAGE,
                             lw=0.5, label="3 sharing one objective framing"),
               plt.Line2D([], [], color=INK, lw=0.7, ls=(0, (2, 1.6)),
                          label="95th pct. of permutation null")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.46, 1.20),
              handlelength=1.1, labelspacing=0.28)
    ax.set_title("Prompt language", pad=27)

    # (c) assigned persona ----------------------------------------------------
    eff = pd.read_csv(TABDIR / "T_FR10_persona_effects.csv")
    eff = eff.set_index("model").reindex(FRONTIER).reset_index()

    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax, axis="x")
    y = np.arange(len(eff))[::-1].astype(float)
    ax.axvline(0, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    off = 0.17
    for yi, r in zip(y, eff.itertuples()):
        c = MODEL_C[r.model]
        ax.hlines(yi + off, r.own_lo, r.own_hi, color=c, lw=0.8, zorder=3)
        ax.plot(r.own_effect, yi + off, marker=MODEL_M[r.model], ms=4.4,
                mfc=c, mec=PAGE, mew=0.6, ls="none", zorder=4)
        ax.hlines(yi - off, r.opp_lo, r.opp_hi, color=c, lw=0.8, zorder=3)
        ax.plot(r.opp_effect, yi - off, marker=MODEL_M[r.model], ms=4.4,
                mfc=PAGE, mec=c, mew=1.0, ls="none", zorder=4)
    ax.text(0.0, -0.78, " no effect", ha="left", va="center", fontsize=5.6,
            color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in eff.model],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1.05, len(eff) - 0.35)
    lo = min(eff.own_lo.min(), eff.opp_lo.min(), 0.0)
    hi = max(eff.own_hi.max(), eff.opp_hi.max(), 0.0)
    pad = 0.06 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("$\\Delta$ cooperation,\ncooperative $-$ selfish persona")
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=4.0, mfc=MUTED,
                          mec=PAGE, mew=0.6, label="own persona (stated)"),
               plt.Line2D([], [], ls="none", marker="o", ms=4.0, mfc=PAGE,
                          mec=MUTED, mew=1.0, label="opponent's (hidden)")]
    ax.legend(handles=handles, ncol=1, loc="upper right",
              bbox_to_anchor=(1.02, 1.05), handlelength=0.8)
    ax.set_title("Assigned persona", pad=13)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"], dx=-0.008)
    save(fig, "fig2_invariance")


# ==========================================================================
# Figure 3 -- what the play is not
# ==========================================================================
def fig_strategy():
    fig = figure(W2, 2.55)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.20, 1.15, 1.10])

    # (a) cooperation by round ------------------------------------------------
    tr = pd.read_csv(TABDIR / "T_FR11_round_profile.csv")
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    for mdl in FRONTIER:
        s = tr[tr.model == mdl].sort_values("round")
        model_line(ax, s["round"], s["mean"], mdl, lo=s["lo"], hi=s["hi"])
    refline(ax, 0.5, "indifference")
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0.05, 0.80)
    ax.set_xlabel("round")
    ax.set_ylabel("cooperation rate")
    ax.set_title("Endgame decay, with one exception", pad=6)

    # (b) the memory-one fingerprint -----------------------------------------
    fpd = pd.read_csv(TABDIR / "T_FR13_memory1_fingerprint.csv", index_col=0)
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    x = np.arange(4)
    w = 0.19
    for k, mdl in enumerate(FRONTIER):
        v = [fpd.loc[mdl, f"pC_{s}"] for s in STATE_ORDER]
        bars(ax, x + (k - 1.5) * w, v, MODEL_C[mdl], width=w * 0.88)
    for s_i in range(4):
        ax.hlines(MEMORY1["TFT"][s_i], s_i - 0.44, s_i + 0.44, color=INK,
                  lw=0.7, ls=(0, (2, 1.6)), zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABEL[s] for s in STATE_ORDER], fontsize=6.0,
                       linespacing=1.35)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("P(cooperate next round)")
    ax.set_ylim(0, 1.06)
    ax.set_xlim(-0.55, 3.55)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.plot([], [], color=INK, lw=0.7, ls=(0, (2, 1.6)),
            label="tit-for-tat")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), handlelength=1.6)
    ax.set_title("No memory-one rule", pad=13)

    # (c) how a label is actually reached -------------------------------------
    arche = pd.read_parquet(DATADIR / "frontier_archetypes.parquet")
    prov = (arche.groupby(["model", "assignment"]).size()
            .unstack("assignment").reindex(FRONTIER)
            .reindex(columns=ASSIGN_ORDER).fillna(0))
    prov = prov.div(prov.sum(axis=1), axis=0)
    dev = arche.groupby("model").min_deviations.mean().reindex(FRONTIER)

    ax = pc = fig.add_subplot(gs[0, 2])
    y = np.arange(len(prov))[::-1].astype(float)
    left = np.zeros(len(prov))
    for key in ASSIGN_ORDER:
        v = prov[key].to_numpy()
        ax.barh(y, v, left=left, height=0.62, color=ASSIGN_C[key],
                edgecolor=PAGE, linewidth=0.5, zorder=3, label=ASSIGN_LABEL[key])
        for yi, l, vv in zip(y, left, v):
            if vv > 0.07:
                ax.text(l + vv / 2, yi, f"{vv:.2f}", ha="center", va="center",
                        fontsize=5.4, zorder=4,
                        color=PAGE if key == "exact" else INK)
        left += v
    for yi, mdl in zip(y, prov.index):
        ax.text(1.015, yi, f"{dev[mdl]:.1f}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.0, color=INK2)
    ax.text(1.015, len(prov) - 0.52, "mean\ndeviations",
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=5.6, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in prov.index],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("share of agent-games")
    ax.set_ylim(-1.30, len(prov) - 0.25)
    ax.legend(ncol=1, loc="lower center", bbox_to_anchor=(0.44, -0.010),
              handlelength=0.9, labelspacing=0.30)
    ax.set_title("How a label is reached", pad=6)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"], dx=-0.008)
    shared_model_legend(fig, [pa, pb, pc], ncol=4, lines=True, gap=0.028)
    save(fig, "fig3_strategy")


def main():
    fig_design()
    fig_invariance()
    fig_strategy()
    print(f"\nwrote 3 main-text figures to {PAPERFIG}")


if __name__ == "__main__":
    main()
