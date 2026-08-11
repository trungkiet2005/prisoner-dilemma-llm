"""The main-text figures for the Interface Focus manuscript.

The frontier suite (`run_frontier.py`, FR1-FR29) is the full record and stays
the supplementary material.  A main-text figure has to carry a whole step of
the argument rather than a single measurement, and the argument has one step
per contribution claimed in the abstract:

    fig1_design       what the game is and what was crossed with what
    fig2_aggregate    what the pooled cooperation rate hides           (new)
    fig2_invariance   three manipulations that cannot matter, and all three do
    fig3_strategy     what the play is not: no unravelling, no memory-one rule
    fig6_misread      one account of all three: the frame is read backwards (new)

`fig2_aggregate` and `fig6_misread` cover the two claims that the manuscript
made in prose with no figure at all: that the pooled mean describes almost no
game and that nearly all of the structure lives in interactions rather than in
main effects, and that the models are optimising a reward reading of a penalty
table.  The file names of the three older figures are left alone so
`paper/main.tex` keeps compiling; renumber them once the final lineup is fixed.

Panel-level numbers are read from `tables/T_FR*.csv`, which the frontier
pipeline writes, plus `tables/T_PAPER_*.csv` from `32_paper_revision_stats.py`
for the three quantities the FR suite reports in a form the main text cannot
use: the variance decomposition with interactions separated from replicate
noise, the payoff scale as two adjacent contrasts rather than one slope, and
the persona effect resolved by payoff scale.  Run `python
Analysis/run_frontier.py` and then `scripts/32_paper_revision_stats.py` first.

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
                            MUTED, PAGE, PERSONALITY_C, PERSONALITY_ORDER,
                            RULE, SCALE_ORDER, SPINE, TABDIR, W2, bars, figure,
                            finalize, hgrid, model_legend, model_line, refline,
                            shared_model_legend, use_journal_style)

use_journal_style()

PAPERFIG = Path(__file__).resolve().parents[2] / "paper" / "figures"
PAPERFIG.mkdir(parents=True, exist_ok=True)

STATE_ORDER = ["R", "S", "T", "P"]
STATE_LABEL = {"R": "after CC\n($R$)", "S": "after CD\n($S$)",
               "T": "after DC\n($T$)", "P": "after DD\n($P$)"}

# nice-to-nasty, so the bar groups read left to right as a gradient rather
# than in the arbitrary order the dictionary happens to hold
RULE_ORDER = ["AllC", "TFT", "WSLS", "GRIM", "AllD"]

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
    # Two adjacent contrasts, not one slope.  Cooperation is a step in lambda:
    # everything happens between x0.1 and x1 and nothing between x1 and x10, so
    # a line fitted through the step reports a per-decade movement that no
    # model produced, and for Gemini and Mistral the step is not even monotone.
    sl = pd.read_csv(TABDIR / "T_PAPER_scale_contrasts.csv")
    sl = sl.set_index("model").reindex(FRONTIER).reset_index()

    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax, axis="x")
    y = np.arange(len(sl))[::-1].astype(float)
    ax.axvline(0.0, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    off = 0.17
    for yi, r in zip(y, sl.itertuples()):
        c = MODEL_C[r.model]
        ax.hlines(yi + off, r.step1_lo, r.step1_hi, color=c, lw=0.8, zorder=3)
        ax.plot(r.step1, yi + off, marker=MODEL_M[r.model], ms=4.4, mfc=c,
                mec=PAGE, mew=0.6, ls="none", zorder=4)
        ax.hlines(yi - off, r.step2_lo, r.step2_hi, color=c, lw=0.8, zorder=3)
        ax.plot(r.step2, yi - off, marker=MODEL_M[r.model], ms=4.4, mfc=PAGE,
                mec=c, mew=1.0, ls="none", zorder=4)
    ax.text(0.0, -0.78, " no effect", ha="left", va="center", fontsize=5.6,
            color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in sl.model],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1.05, len(sl) - 0.35)
    lo = min(sl.step1_lo.min(), sl.step2_lo.min(), 0.0)
    hi = max(sl.step1_hi.max(), sl.step2_hi.max(), 0.0)
    pad = 0.06 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("$\\Delta$ cooperation between\nadjacent payoff scales")
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=4.0, mfc=MUTED,
                          mec=PAGE, mew=0.6,
                          label="$\\times 0.1 \\rightarrow \\times 1$"),
               plt.Line2D([], [], ls="none", marker="o", ms=4.0, mfc=PAGE,
                          mec=MUTED, mew=1.0,
                          label="$\\times 1 \\rightarrow \\times 10$")]
    ax.legend(handles=handles, ncol=1, loc="upper right",
              bbox_to_anchor=(1.04, 1.05), handlelength=0.8, labelspacing=0.28)
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

    # (c) assigned persona, resolved by payoff scale --------------------------
    # Pooling the three scales averages an effect of -0.49 against one of
    # +0.06 and reports the mean of the two as though it described either.  It
    # is also the wrong way round to plot: the panel exists to show that a
    # manipulation which cannot matter decides the sign of one that could.
    eff = pd.read_csv(TABDIR / "T_PAPER_persona_by_scale.csv")
    eff = eff[eff.model.isin(FRONTIER) & eff.scale.notna()]
    eff["model"] = pd.Categorical(eff.model, FRONTIER, ordered=True)
    eff = eff.sort_values(["model", "scale"])

    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax, axis="x")
    y = np.arange(len(FRONTIER))[::-1].astype(float)
    ax.axvline(0, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    # one row per model, three rungs per row: darkest is the smallest scale
    alpha = {0.1: 1.00, 1.0: 0.62, 10.0: 0.34}
    for yi, mdl in zip(y, FRONTIER):
        c = MODEL_C[mdl]
        for k, s in enumerate(SCALE_ORDER):
            r = eff[(eff.model == mdl) & (eff.scale == s)].iloc[0]
            yy = yi + 0.24 - 0.24 * k
            ax.hlines(yy, r.lo, r.hi, color=c, lw=0.8, alpha=alpha[s], zorder=3)
            ax.plot(r.effect, yy, marker=MODEL_M[mdl], ms=3.8, mfc=c, mec=PAGE,
                    mew=0.5, alpha=alpha[s], ls="none", zorder=4)
    ax.text(0.0, -0.80, " no effect", ha="left", va="center", fontsize=5.6,
            color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in FRONTIER],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1.10, len(FRONTIER) - 0.30)
    lo, hi = min(eff.lo.min(), 0.0), max(eff.hi.max(), 0.0)
    pad = 0.06 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("$\\Delta$ cooperation,\ncooperative $-$ selfish persona")
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=3.6, mfc=MUTED,
                          mec=PAGE, mew=0.5, alpha=alpha[s],
                          label=f"$\\lambda={s:g}$") for s in SCALE_ORDER]
    ax.legend(handles=handles, ncol=1, loc="upper left",
              bbox_to_anchor=(-0.02, 1.06), handlelength=0.8,
              labelspacing=0.24)
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

    # (b) distance to every canonical rule, not just the nearest --------------
    # The fingerprint itself now belongs to fig6_misread, where it is evidence
    # for the reward reading.  What this figure needs from that space is the
    # negative result, and "nearest archetype" hides whether the runner-up was
    # a close second or nowhere near -- so all five distances are shown.
    # T_FR15 is Euclidean, matching the numbers quoted in the manuscript;
    # T_FR46 is the same quantity as an RMS (that is, halved), and mixing the
    # two would silently double or halve every distance in the text.
    near = pd.read_csv(TABDIR / "T_FR15_archetype_distance.csv")
    near = near.set_index("model").reindex(FRONTIER)
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    x = np.arange(len(RULE_ORDER))
    w = 0.19
    for k, mdl in enumerate(FRONTIER):
        v = [near.loc[mdl, f"d_{r}"] for r in RULE_ORDER]
        bars(ax, x + (k - 1.5) * w, v, MODEL_C[mdl], width=w * 0.88)
        # ring the nearest rule, which is the one a label would name
        j = int(np.argmin(v))
        ax.plot(j + (k - 1.5) * w, v[j] + 0.055, marker="v", ms=3.0,
                mfc=MODEL_C[mdl], mec=PAGE, mew=0.4, ls="none", zorder=6)
    # the canonical rules are at most 2.0 from one another in this space, so
    # the scale is what makes 0.7-1.0 a miss rather than a near miss
    refline(ax, 2.0, "greatest distance between two canonical rules", tx=0.5,
            ha="center", size=5.4)
    ax.set_xticks(x)
    ax.set_xticklabels(RULE_ORDER, fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("Euclidean distance in\nfingerprint space")
    ax.set_ylim(0, 2.32)
    ax.set_xlim(-0.55, len(RULE_ORDER) - 0.45)
    ax.set_yticks(np.arange(0, 2.01, 0.5))
    ax.set_title("No memory-one rule is close", pad=6)

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


# ==========================================================================
# Figure 2 -- what the pooled cooperation rate hides
# ==========================================================================
# The manuscript opens the results by quoting four pooled rates and then
# spending a section explaining why they carry almost nothing.  That section
# has three claims and no figure, which is the wrong way round: the reader is
# asked to discount the headline number on the strength of prose.  The three
# panels are the three claims, in the order the section makes them.
# A main-effects fit leaves every interaction among the four factors in the
# residual, so calling that residual "within-cell" credits the design with a
# fifth of the variance it actually accounts for.  The rows below separate the
# three things the old "unexplained" bar was hiding: interactions among the
# same four factors, the asymmetry between the two seats of a dyad, and the
# replicate spread that alone deserves the name.
VAR_FILL = {"model (main effect)": "#0072b2",
            "payoff scale (main effect)": "#7fb2d4",
            "language (main effect)": "#a1b792",
            "persona pairing (main effect)": "#be9976",
            "interactions among the four": "#41618a",
            "position within the dyad": "#b0b0b0",
            "replicate (identical prompt)": "#dcdcdc"}
VAR_SHORT = {"model (main effect)": "model",
             "payoff scale (main effect)": "payoff scale",
             "language (main effect)": "language",
             "persona pairing (main effect)": "persona pairing",
             "interactions among the four": "their interactions",
             "position within the dyad": "seat in the dyad",
             "replicate (identical prompt)": "replicate spread"}


def fig_aggregate():
    games = pd.read_parquet(DATADIR / "frontier_games.parquet")
    pol = pd.read_csv(TABDIR / "T_FR20_polarisation.csv").set_index("model")
    var = pd.read_csv(TABDIR / "T_PAPER_variance_full.csv")
    rep = pd.read_csv(TABDIR / "T_FR24_replicate_spread.csv").set_index("model")

    fig = figure(W2, 2.60)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.05, 0.92])

    # (a) the within-game distribution ---------------------------------------
    # Ten rounds means a game's rate can only be k/10, so this is a discrete
    # distribution over eleven values.  Binning it into a histogram would merge
    # adjacent values and invent empty bins, which is exactly the smoothing
    # that makes a bimodal mixture look unimodal.
    ax = pa = fig.add_subplot(gs[0, 0])
    n_r = int(games.n_rounds.iloc[0])
    vals = np.arange(n_r + 1) / n_r
    step = 1.0
    for k, mdl in enumerate(FRONTIER):
        v = games.loc[games.model == mdl, "coop_rate"].to_numpy()
        h = np.array([np.isclose(v, x).mean() for x in vals])
        base = (len(FRONTIER) - 1 - k) * step
        # 0.70 of the row, not the full step: the mean label above each ridge
        # needs clearance, and a taller ridge would put it under the bars of
        # the row above
        ax.bar(vals, h / h.max() * 0.70, bottom=base, width=1 / n_r * 0.80,
               color=MODEL_C[mdl], edgecolor=PAGE, linewidth=0.4, zorder=3)
        ax.hlines(base, -0.02, 1.02, color=RULE, lw=0.6, zorder=2)
        # the pooled mean, drawn *through* the distribution it is supposed to
        # summarise -- for three of four models it lands in a trough
        mu = float(v.mean())
        ax.vlines(mu, base, base + 0.76, color=INK, lw=0.8, zorder=6)
        ax.plot(mu, base + 0.76, marker="v", ms=3.0, mfc=INK, mec=PAGE,
                mew=0.4, ls="none", zorder=6)
        ax.text(mu, base + 0.79, f"mean {mu:.2f}", ha="center", va="bottom",
                fontsize=5.4, color=INK, zorder=6)
        ax.text(-0.075, base + 0.30, MODEL_LABEL[mdl].replace(" ", "\n", 1),
                ha="right", va="center", fontsize=6.0, color=INK)
        # the two corners, labelled above their own bar so the number never
        # sits on a fill it cannot be read against
        for xc, col in ((0.0, "all_D"), (1.0, "all_C")):
            top = h[0 if xc == 0.0 else -1] / h.max() * 0.70
            ax.text(xc, base + top + 0.025, f"{pol.loc[mdl, col]:.0%}",
                    ha="center", va="bottom", fontsize=5.4,
                    color=MODEL_C[mdl], zorder=6)
    ax.text(0.0, -0.10, "all D", ha="center", va="top", fontsize=5.4,
            color=MUTED)
    ax.text(1.0, -0.10, "all C", ha="center", va="top", fontsize=5.4,
            color=MUTED)
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.30, len(FRONTIER) * step + 0.02)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("cooperation rate within one game")
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("The mean describes almost no game", pad=6)

    # (b) where the variance lives -------------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax, axis="x")
    y = np.arange(len(var))[::-1].astype(float)
    for yi, r in zip(y, var.itertuples()):
        ax.barh(yi, r.share, height=0.58, color=VAR_FILL[r.component],
                edgecolor=PAGE, linewidth=0.5, zorder=3)
        ax.text(r.share + 0.016, yi, f"{r.share:.1%}", ha="left", va="center",
                fontsize=6.0, color=INK2)
    # a bracket over the five rows the four crossed factors account for: the
    # comparison the panel exists to make is design against replicate, and the
    # main-effect rows on their own invite exactly the wrong reading
    designed = float(var.share.iloc[:5].sum())
    xb = 0.62
    ax.plot([xb] * 2, [y[4] - 0.30, y[0] + 0.30], color=SPINE, lw=0.6, zorder=4)
    for yy in (y[0] + 0.30, y[4] - 0.30):
        ax.plot([xb - 0.02, xb], [yy] * 2, color=SPINE, lw=0.6, zorder=4)
    ax.text(xb + 0.018, (y[0] + y[4]) / 2,
            f"the four\ncrossed factors\nand their\ninteractions:\n{designed:.1%}",
            ha="left", va="center", fontsize=5.6, color=INK2, linespacing=1.35)
    ax.set_yticks(y)
    ax.set_yticklabels([VAR_SHORT[c] for c in var.component], fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.60, len(var) - 0.40)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("share of variance in agent-game\ncooperation rate")
    ax.set_title("Where the variance lives", pad=6)

    # (c) what the replicate share is made of --------------------------------
    # Panel b puts 28% on replicates of a byte-identical prompt but not what
    # that spread is: the observed SD among the ten is 1.3-1.6 times the floor
    # that ten independent rounds would already impose, so the row is neither
    # pure arithmetic nor pure caprice -- the floor is 42% of it.
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax)
    rep = rep.reindex(FRONTIER)
    x = np.arange(len(FRONTIER))
    w = 0.34
    bars(ax, x - w / 2, rep.observed_sd, [MODEL_C[m] for m in FRONTIER],
         width=w * 0.9)
    bars(ax, x + w / 2, rep.binomial_sd, "#d6d6d6", width=w * 0.9)
    for xi, mdl in zip(x, FRONTIER):
        r = rep.loc[mdl]
        ax.text(xi, max(r.observed_sd, r.binomial_sd) + 0.006,
                f"{r.ratio:.1f}$\\times$", ha="center", va="bottom",
                fontsize=5.8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m].split()[0] for m in FRONTIER],
                       fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("SD across the 10 replicates\nof an identical prompt")
    ax.set_xlim(-0.62, len(x) - 0.38)
    ax.set_ylim(0, max(rep.observed_sd.max(), rep.binomial_sd.max()) * 1.42)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=PAGE,
                             lw=0.5, label="observed"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#d6d6d6", edgecolor=PAGE,
                             lw=0.5, label="independent-round floor")]
    ax.legend(handles=handles, ncol=1, loc="upper right",
              bbox_to_anchor=(1.03, 1.04), handlelength=0.9, labelspacing=0.28)
    ax.set_title("The last row is not only arithmetic", pad=6)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"], dx=-0.008)
    save(fig, "fig2_aggregate")


# ==========================================================================
# Figure 6 -- one account of all three: the frame is read backwards
# ==========================================================================
# The frame-misreading hypothesis is the paper's own explanation and the only
# claim in it that is mechanistic rather than descriptive, yet in the current
# lineup it borrows a panel from fig3.  The argument has a shape -- hypothesis,
# then two independent predictions it gets right -- and the figure follows it:
# (a) states the two readings of one table, (b) tests the prediction at the
# opening move, (c) tests it inside the memory-one fingerprint.
#
# Rows are the focal player's action, columns the opponent's, and the number
# shown is what the *focal* player receives.  Option B is cooperation.
CELL_LABEL = [["BB", "BA"], ["AB", "AA"]]
GOOD_C = ["#f4f4f4", "#dceaf7", "#a8cbe4", "#5c9bce", "#0072b2"]


def _reading_grid(ax, x0, y0, cw, ch, mat, *, maximise, title, sub, best):
    """One 2x2 payoff grid, shaded by how good each cell is under a reading."""
    v = np.asarray(mat, dtype=float)
    lo, hi = v.min(), v.max()
    rank = (v - lo) / (hi - lo)
    if not maximise:                      # lower penalty is the better outcome
        rank = 1.0 - rank
    for i in range(2):
        for j in range(2):
            # four discrete tints rather than a continuous ramp: the panel is
            # an ordering statement, and a ramp invites the reader to compare
            # magnitudes across two grids whose scales are deliberately flipped
            c = GOOD_C[int(round(rank[i, j] * (len(GOOD_C) - 1)))]
            x, y = x0 + j * cw, y0 - i * ch
            ax.add_patch(Rectangle((x, y - ch), cw, ch, facecolor=c,
                                   edgecolor=PAGE, linewidth=1.1, zorder=2))
            ax.text(x + cw / 2, y - ch / 2 + 0.055, f"{v[i, j]:g}",
                    ha="center", va="center", fontsize=8.5, zorder=3,
                    color=PAGE if rank[i, j] > 0.72 else INK)
            ax.text(x + cw / 2, y - ch / 2 - 0.115, CELL_LABEL[i][j],
                    ha="center", va="center", fontsize=5.2, zorder=3,
                    color=PAGE if rank[i, j] > 0.72 else MUTED)
    for j, lab in enumerate(("opp. B", "opp. A")):
        ax.text(x0 + j * cw + cw / 2, y0 + 0.055, lab, ha="center", va="bottom",
                fontsize=5.4, color=MUTED)
    for i, lab in enumerate(("B", "A")):
        ax.text(x0 - 0.045, y0 - i * ch - ch / 2, lab, ha="right", va="center",
                fontsize=5.8, color=INK2)
    ax.text(x0, y0 + 0.30, title, ha="left", va="bottom", fontsize=6.4,
            color=INK)
    ax.text(x0, y0 + 0.155, sub, ha="left", va="bottom", fontsize=5.6,
            color=MUTED)
    # Everything is left-anchored at a fixed offset rather than right-aligned
    # against the cell block: at 5-6 pt the two ends of a 0.92-wide block
    # collide, and matplotlib reports nothing when they do.
    for k, (who, act, note) in enumerate(best):
        yy = y0 - 2 * ch - 0.22 - k * 0.30
        ax.text(x0, yy, f"told {who}", ha="left", va="center", fontsize=5.6,
                color=INK2)
        ax.text(x0 + 0.70, yy, f"→  Option {act}", ha="left", va="center",
                fontsize=5.6, color=INK, fontweight="bold")
        ax.text(x0, yy - 0.135, note, ha="left", va="center", fontsize=5.0,
                color=MUTED)


def fig_misread():
    m = payoff_matrix("frontier")
    op = pd.read_csv(TABDIR / "T_FR21_openings.csv")
    fpd = pd.read_csv(TABDIR / "T_FR13_memory1_fingerprint.csv", index_col=0)

    fig = figure(W2, 2.95)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 0.98, 1.16])

    # (a) the same table, read two ways --------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.16, 2.60)
    ax.set_ylim(-1.28, 1.02)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")

    focal = [[m["R"], m["S"]],       # focal plays B: (BB) = R, (BA) = S
             [m["T"], m["P"]]]       # focal plays A: (AB) = T, (AA) = P
    cw = ch = 0.46
    _reading_grid(
        ax, 0.0, 0.62, cw, ch, focal, maximise=False,
        title="as stated: minimise",
        sub="the objective the prompt gives",
        best=[("cooperative", "B", "mutual B costs 2, mutual A costs 6"),
              ("selfish", "A", "A dominates; exploiting costs 0")])
    _reading_grid(
        ax, 1.42, 0.62, cw, ch, focal, maximise=True,
        title="as read: maximise",
        sub="the canonical reward-framed dilemma",
        best=[("cooperative", "A", "mutual A pays 6, mutual B pays 2"),
              ("selfish", "B", "only B can reach the 10")])
    ax.annotate("", xy=(1.36, 0.62 - ch), xytext=(1.00, 0.62 - ch),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7,
                                shrinkA=0, shrinkB=0, mutation_scale=6))
    ax.text(1.18, 0.62 - ch + 0.04, "same\nnumbers", ha="center", va="bottom",
            fontsize=5.0, color=MUTED, linespacing=1.2)
    ax.plot([0.0, 2.34], [-1.06, -1.06], color=RULE, lw=0.5, zorder=1)
    ax.text(0.0, -1.19, "every predicted action inverts between the two readings",
            ha="left", va="center", fontsize=5.8, color=INK)
    ax.set_title("One table, two readings", pad=6)

    # (b) prediction 1: the opening move -------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    x = np.arange(len(FRONTIER))
    w = 0.34
    for k, pers in enumerate(PERSONALITY_ORDER):
        sub = op[op.personality == pers].set_index("model").reindex(FRONTIER)
        xs = x + (k - 0.5) * w
        bars(ax, xs, sub["mean"], PERSONALITY_C[pers], width=w * 0.9,
             label=f"told {pers}")
        ax.vlines(xs, sub["lo"], sub["hi"], color=INK, lw=0.6, zorder=5)
    refline(ax, 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m_].split()[0] for m_ in FRONTIER],
                       fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("P(cooperate on round 1)")
    ax.set_ylim(0, 1.30)
    ax.set_xlim(-0.62, len(x) - 0.38)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    # the legend lives in the empty band above the bars, not above the axes,
    # where it would land on the title
    ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(0.0, 0.99),
              handlelength=0.9, labelspacing=0.28)
    ax.set_title("Prediction 1: the opening inverts", pad=6)

    # (c) prediction 2: the memory-one fingerprint ---------------------------
    # The state labels carry the number the focal agent actually received, and
    # that is the entire panel: p(C | S) is behaviour after collecting the
    # largest entry on the table, which the reward reading says to repeat and
    # every canonical rule except AllC says to punish.
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax)
    recv = {"R": m["R"], "S": m["S"], "T": m["T"], "P": m["P"]}
    # the band covers both extreme states, because the hypothesis makes a
    # prediction about each: repeat after the 10, abandon after the 0
    ax.add_patch(Rectangle((0.56, 0), 1.88, 1.06, facecolor="#f3f3f3",
                           edgecolor="none", zorder=0))
    x = np.arange(4)
    w = 0.19
    for k, mdl in enumerate(FRONTIER):
        v = [fpd.loc[mdl, f"pC_{s}"] for s in STATE_ORDER]
        bars(ax, x + (k - 1.5) * w, v, MODEL_C[mdl], width=w * 0.88)
    for s_i in range(4):
        ax.hlines(MEMORY1["TFT"][s_i], s_i - 0.44, s_i + 0.44, color=INK,
                  lw=0.7, ls=(0, (2, 1.6)), zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"after {'CC' if s == 'R' else 'CD' if s == 'S' else 'DC' if s == 'T' else 'DD'}"
         f"\nreceives {recv[s]:g}" for s in STATE_ORDER],
        fontsize=5.8, linespacing=1.45)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("P(cooperate next round)")
    ax.set_ylim(0, 1.30)
    ax.set_xlim(-0.55, 3.55)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.text(1.50, 1.085, "the largest and smallest entries on the table",
            ha="center", va="bottom", fontsize=5.4, color=INK2)
    ax.annotate("", xy=(0.56, 1.075), xytext=(2.44, 1.075),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.plot([], [], color=INK, lw=0.7, ls=(0, (2, 1.6)), label="tit-for-tat")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=1.6)
    ax.set_title("Prediction 2: repeat the 10, abandon the 0", pad=6)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"], dx=-0.008)
    shared_model_legend(fig, [pc], ncol=4, lines=False, gap=0.030)
    save(fig, "fig6_misread")


# ==========================================================================
# Figure 7 -- what the unmatched play actually is
# ==========================================================================
# fig3_strategy establishes that half to seven-eighths of agent-games match no
# memory-one rule.  That is a negative result, and on its own it invites the
# obvious dismissal: the residual is noise, and a label was never going to fit
# noise.  This figure is the reply.  Each panel widens the hypothesis class in
# a different direction -- more rules, more history, finer resolution -- and
# each answer is that the residual is real but is not in the class.
LADDER_ORDER = ["canonical4", "memory1_32", "two_regime", "unexplained"]
LADDER_LABEL = {"canonical4": "one of the 4 canonical rules",
                "memory1_32": "+ any of the 32 memory-one rules",
                "two_regime": "+ one regime switch",
                "unexplained": "still unexplained"}
LADDER_C = {"canonical4": "#0072b2", "memory1_32": "#6aa8cf",
            "two_regime": "#b8d4e6", "unexplained": "#e0e0e0"}

DEPTH_COLS = ["memoryless", "memory-one", "memory-two", "memory-one + phase"]
DEPTH_TICK = ["memory-\nless", "memory-\none", "memory-\ntwo",
              "memory-one\n+ phase"]


def fig_residual():
    lad = pd.read_csv(TABDIR / "T_FR35_hypothesis_ladder.csv", index_col=0)
    lad = lad.reindex(FRONTIER)
    mem = pd.read_csv(TABDIR / "T_FR38_memory_depth.csv").set_index("model")
    mem = mem.reindex(FRONTIER)
    per = pd.read_csv(TABDIR / "T_FR36b_rules_per_game.csv", index_col=0)

    fig = figure(W2, 2.60)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 0.98, 0.92])

    # (a) how far a wider vocabulary gets ------------------------------------
    # The classes are nested, so the segments are disjoint and stack to one.
    # The tick is the same three classes applied to shuffled play: coverage to
    # the left of it is what a two-segment fit over ten rounds buys by chance,
    # and it is large enough that raw coverage is not evidence of anything.
    ax = pa = fig.add_subplot(gs[0, 0])
    y = np.arange(len(lad))[::-1].astype(float)
    left = np.zeros(len(lad))
    for cls in LADDER_ORDER:
        v = lad[cls].to_numpy()
        ax.barh(y, v, left=left, height=0.58, color=LADDER_C[cls],
                edgecolor=PAGE, linewidth=0.5, zorder=3,
                label=LADDER_LABEL[cls])
        for yi, l, vv in zip(y, left, v):
            if vv > 0.10:
                ax.text(l + vv / 2, yi, f"{vv:.2f}", ha="center", va="center",
                        fontsize=5.4, zorder=4,
                        color=PAGE if cls == "canonical4" else INK)
        left += v
    cum_null = lad[["canonical4_null", "memory1_32_null",
                    "two_regime_null"]].sum(axis=1).to_numpy()
    ax.plot(cum_null, y, marker="|", ms=8, mew=1.2, color=INK, ls="none",
            zorder=6, label="same classes on shuffled play")
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in lad.index],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("share of agent-games")
    ax.set_ylim(-1.55, len(lad) - 0.30)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.46, -0.015),
              handlelength=0.9, columnspacing=0.8, labelspacing=0.28)
    ax.set_title("Widening the vocabulary does not close the gap", pad=6)

    # (b) how much history the next move uses --------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    x = np.arange(len(DEPTH_COLS))
    for mdl in FRONTIER:
        v = mem.loc[mdl, DEPTH_COLS].to_numpy(dtype=float)
        ax.plot(x, v, "-", color=MODEL_C[mdl], lw=1.0, marker=MODEL_M[mdl],
                ms=3.4, mfc=MODEL_C[mdl], mec=PAGE, mew=0.5, zorder=4)
        j = int(np.argmin(v))
        ax.plot(x[j], v[j], "o", ms=7.0, mfc="none", mec=MODEL_C[mdl], mew=1.0,
                zorder=5)
    ax.axhline(0, color=SPINE, lw=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(DEPTH_TICK, fontsize=5.6, linespacing=1.3)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("$\\Delta$BIC against a memoryless model\n(lower is better)")
    ax.set_xlim(-0.35, len(x) - 0.65)
    ax.set_title("The residual uses two rounds of history", pad=6)

    # (c) even where a rule fits, which rule is undecidable -------------------
    # Ten rounds rarely visit all four conditioning states, so rules that
    # differ only in an unvisited state are indistinguishable on that game.
    # The mass at 4 and 8 is that degeneracy, not a finding about the model.
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax)
    n_tot = int(per["agent_games"].sum())
    lv = [str(i) for i in per.index]
    share = per["agent_games"].to_numpy() / n_tot
    colr = ["#e0e0e0" if i == 0 else "#0072b2" if i == 1 else "#9ecae1"
            for i in per.index]
    bars(ax, np.arange(len(lv)), share, colr, width=0.68)
    for xi, (s, n) in enumerate(zip(share, per["agent_games"])):
        ax.text(xi, s + 0.012, f"{n:,}", ha="center", va="bottom",
                fontsize=5.4, color=INK2)
    ax.annotate(f"only {per['agent_games'].get(1, 0):,} of {n_tot:,}\n"
                f"games name one rule",
                xy=(1, share[list(per.index).index(1)] + 0.02), xytext=(1.9, 0.34),
                fontsize=5.4, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_xticks(np.arange(len(lv)))
    ax.set_xticklabels(lv, fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlabel("memory-one rules fitting\nthe same game")
    ax.set_ylabel("share of agent-games")
    ax.set_xlim(-0.6, len(lv) - 0.4)
    ax.set_ylim(0, 0.66)
    ax.set_title("A fit rarely names a rule", pad=6)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"], dx=-0.008)
    shared_model_legend(fig, [pb], ncol=4, lines=True, gap=0.030)
    save(fig, "fig7_residual")


def main():
    fig_design()
    fig_aggregate()
    fig_invariance()
    fig_misread()
    fig_strategy()
    fig_residual()
    print(f"\nwrote 6 main-text figures to {PAPERFIG}")


if __name__ == "__main__":
    main()
