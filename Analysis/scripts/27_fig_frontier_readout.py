"""FR18-FR22: calibrating the strategy read-out, then applying it to frontier
play.

FR18 is about the *instrument*, on synthetic play where the ground truth is
known.  FR19-FR22 point it at the LLM transcripts.  Keeping the two apart is
the point: an archetype label on LLM play is only worth what the instrument
scores on data whose answer is known, and its ceiling there is set by
identifiability, not by the network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdlib.lstm import predict_prefixes
from pdlib.natstyle import (CMAP_SEQ, DATADIR, DYAD_LABEL, DYAD_ORDER,
                            FRONTIER, INK, INK2, MODEL_C, MODEL_LABEL,
                            MODEL_M, MUTED, PAGE, RULE, SCALE_ORDER, SPINE,
                            TABDIR, W2, annotate_heatmap, bars, caption,
                            colorbar, figure, finalize, hgrid, model_line,
                            save, shared_model_legend, use_journal_style)
from pdlib.rulebase import TOKEN_OK
from pdlib.seqcode import STRATEGIES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

use_journal_style()

SEED = 0

# archetype palette: the four canonical rules plus the honest "several fit"
ARCH_ORDER = ["AllC", "TFT", "WSLS", "AllD", "Ambiguous"]
ARCH_C = {"AllC": "#0072b2", "TFT": "#56b4e9", "WSLS": "#009e73",
          "AllD": "#d55e00", "Ambiguous": "#c8c8c8"}

ASSIGN_ORDER = ["exact", "ambiguous", "approx"]
ASSIGN_C = {"exact": "#0072b2", "ambiguous": "#9ecae1", "approx": "#d9d9d9"}
# kept short so the key fits on one row above the panel; the caption carries
# the full reading
ASSIGN_LABEL = {"exact": "exact rule",
                "ambiguous": "several rules",
                "approx": "no rule: LSTM"}


def _bayes_ceiling(X, y, k):
    """Best possible single-label accuracy from the first k tokens alone.

    Identical prefixes carrying different labels cannot be separated by any
    method, so the ceiling is the share of rows whose label is the modal one
    within its own prefix group.
    """
    keys = [tuple(row) for row in X[:, :k]]
    df = pd.DataFrame({"k": keys, "y": y})
    top = df.groupby("k").y.agg(lambda s: s.value_counts().iloc[0]).sum()
    return top / len(df)


# ==========================================================================
# FR18 -- what the instrument can and cannot do
# ==========================================================================
def fig_calibration():
    test = np.load(DATADIR / "clf_test_h10.npz", allow_pickle=True)
    X, y, proba = test["X"], test["y"], test["proba"]

    readout = import_module("26_frontier_readout")
    model = readout.load_readout()
    pref = predict_prefixes(model, X)

    fig = figure(W2, 2.55)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0])

    # (a) identifiability vs rounds observed ---------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    ks = np.arange(1, X.shape[1] + 1)
    ceil = np.array([_bayes_ceiling(X, y, k) for k in ks])
    lstm = np.array([(pref[:, k - 1, :].argmax(1) == y).mean() for k in ks])
    ident = pd.DataFrame({"rounds": ks, "bayes_ceiling": ceil, "lstm": lstm})
    ident.to_csv(TABDIR / "T_FR29_identifiability.csv", index=False)

    ax.fill_between(ks, lstm, ceil, color="#f0c05a", alpha=0.35, lw=0, zorder=2)
    ax.plot(ks, ceil, "-", color=INK, lw=1.0, zorder=4, label="Bayes ceiling")
    ax.plot(ks, lstm, "-o", color="#0072b2", lw=1.0, ms=2.8, mec=PAGE, mew=0.5,
            zorder=5, label="read-out LSTM")
    ax.set_xticks(ks)
    ax.set_xlim(1, ks[-1])
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("rounds observed")
    ax.set_ylabel("single-label accuracy")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), handlelength=1.4)
    ax.set_title("Identifiability, not capacity, is the limit", pad=6)

    # (b) confusion on the held-out split ------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    pred = proba.argmax(1)
    cm = np.zeros((len(STRATEGIES), len(STRATEGIES)))
    for t, p in zip(y, pred):
        cm[t, p] += 1
    cm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
    annotate_heatmap(ax, cm, thresh=0.55, size=6.0)
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels(STRATEGIES, fontsize=6.0)
    ax.set_yticks(range(len(STRATEGIES)))
    ax.set_yticklabels(STRATEGIES, fontsize=6.0)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true strategy")
    colorbar(fig, im, ax, label="row-normalised share", ticks=[0, 0.5, 1])
    ax.set_title(f"Confusion, held-out ({(pred == y).mean():.3f})", pad=6)

    # (c) unseen execution noise ---------------------------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax)
    rows = []
    for name, noise, seen in (("NoNoise", 0.0, True), ("Noise005", 0.05, True),
                              ("Noise01", 0.10, False), ("Noise02", 0.20, False)):
        if seen:
            m = test["src"] == name
            acc = float((pred[m] == y[m]).mean())
            n = int(m.sum())
        else:
            d = np.load(DATADIR / f"clf_ood_h10_{name}.npz", allow_pickle=True)
            acc = float((d["proba"].argmax(1) == d["y"]).mean())
            n = len(d["y"])
        rows.append({"split": name, "noise": noise, "in_training": seen,
                     "accuracy": acc, "n": n})
    ood = pd.DataFrame(rows)
    ood.to_csv(TABDIR / "T_FR30_noise_robustness.csv", index=False)

    x = np.arange(len(ood))
    cols = ["#0072b2" if s else "#d9d9d9" for s in ood.in_training]
    bars(ax, x, ood.accuracy, cols, width=0.6)
    for xi, r in zip(x, ood.itertuples()):
        ax.text(xi, r.accuracy + 0.008, f"{r.accuracy:.3f}", ha="center",
                va="bottom", fontsize=5.8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0%}" for v in ood.noise])
    ax.set_xlabel("execution noise in the corpus")
    ax.set_ylabel("single-label accuracy")
    ax.set_ylim(0.7, 1.03)
    ax.set_xlim(-0.6, len(x) - 0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#0072b2", edgecolor=PAGE,
                             lw=0.5, label="in training"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#d9d9d9", edgecolor=PAGE,
                             lw=0.5, label="unseen")]
    ax.legend(handles=handles, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handlelength=0.9, columnspacing=0.9)
    ax.set_title("Degradation off-distribution", pad=16)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"])
    gapmax = float(np.max(ceil - lstm))
    caption(fig, f"Synthetic play only, where the strategy that generated each "
                 f"trajectory is known. The shaded gap in a is everything the "
                 f"network leaves on the table, and it never exceeds "
                 f"{gapmax:.3f}: the curve's shape is set by identifiability -- "
                 f"distinct strategies that produced identical play -- not by "
                 f"the model, and no method can lift it. Frontier games are 10 "
                 f"rounds, the right-hand end of a.")
    save(fig, "FR18_calibration")
    return ident, ood


# ==========================================================================
# FR19 -- what the read-out says about frontier play
# ==========================================================================
def _stacked(ax, frame, order, colours, labels, y, height=0.62,
             min_label=0.07, dark=()):
    left = np.zeros(len(frame))
    for key in order:
        v = frame[key].to_numpy() if key in frame else np.zeros(len(frame))
        ax.barh(y, v, left=left, height=height, color=colours[key],
                edgecolor=PAGE, linewidth=0.5, zorder=3, label=labels[key])
        for yi, (l, vv) in zip(y, zip(left, v)):
            if vv > min_label:
                ax.text(l + vv / 2, yi, f"{vv:.2f}", ha="center", va="center",
                        fontsize=5.4, zorder=4,
                        color=PAGE if key in dark else INK)
        left += v
    return left


def fig_archetypes(arche):
    fig = figure(W2, 2.5)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

    # (a) archetype mix -------------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    share = (arche.groupby(["model", "archetype"]).size()
             .unstack("archetype").reindex(FRONTIER)
             .reindex(columns=ARCH_ORDER).fillna(0))
    share = share.div(share.sum(axis=1), axis=0)
    y = np.arange(len(share))[::-1]
    _stacked(ax, share, ARCH_ORDER, ARCH_C,
             {k: k for k in ARCH_ORDER}, y, dark=("AllC", "AllD", "WSLS"))
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in share.index],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("share of agent-games")
    ax.set_ylim(-0.75, len(share) - 0.25)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12),
              handlelength=0.9, columnspacing=0.8)
    ax.set_title("Nearest canonical archetype", pad=16)

    # (b) how the label was reached, and how far it is -------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    prov = (arche.groupby(["model", "assignment"]).size()
            .unstack("assignment").reindex(FRONTIER)
            .reindex(columns=ASSIGN_ORDER).fillna(0))
    prov = prov.div(prov.sum(axis=1), axis=0)
    dev = arche.groupby("model").min_deviations.mean().reindex(FRONTIER)
    y = np.arange(len(prov))[::-1]
    _stacked(ax, prov, ASSIGN_ORDER, ASSIGN_C, ASSIGN_LABEL, y, dark=("exact",))
    for yi, mdl in zip(y, prov.index):
        ax.text(1.015, yi, f"{dev[mdl]:.1f}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.0, color=INK2)
    ax.text(1.015, len(prov) - 0.55, "mean\ndeviations",
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=5.6, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in prov.index],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("share of agent-games")
    ax.set_ylim(-0.75, len(prov) - 0.25)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12),
              handlelength=0.9, columnspacing=0.8)
    ax.set_title("How the label was reached", pad=16)

    finalize(fig, [pa, pb], ["a", "b"])
    caption(fig, "A label in a is the nearest canonical rule, not an identity "
                 "claim. In b, dark = exactly one canonical rule reproduces the "
                 "whole trajectory, mid = several do and the honest answer is "
                 "the set, grey = none does and the LSTM names the nearest. "
                 "The right-hand column of b is the mean number of rounds out "
                 "of ten on which play departs from its own nearest rule, so "
                 "Claude's 2.0 means its modal 'archetype' is violated a fifth "
                 "of the time.")
    save(fig, "FR19_archetypes")
    return share, prov


# ==========================================================================
# FR20 -- archetype by condition
# ==========================================================================
def fig_archetype_conditions(arche):
    fig = figure(W2, 2.75)
    gs = fig.add_gridspec(1, 2)

    for panel, (col, order, lab, title) in enumerate((
            ("scale_nominal", SCALE_ORDER, lambda v: f"$\\times${v:g}",
             "By payoff scale"),
            ("dyad", DYAD_ORDER, lambda v: DYAD_LABEL[v].replace(" vs ", " / "),
             "By persona pairing"))):
        ax = fig.add_subplot(gs[0, panel])
        rows, ypos, ylab = [], [], []
        yv = 0.0
        for mdl in FRONTIER:
            for v in order:
                sub = arche[(arche.model == mdl) & (arche[col] == v)]
                s = sub.archetype.value_counts(normalize=True)
                rows.append(s.reindex(ARCH_ORDER).fillna(0))
                ypos.append(yv)
                ylab.append(lab(v))
                yv += 1
            yv += 0.7
        frame = pd.DataFrame(rows).reset_index(drop=True)
        ypos = np.array(ypos)
        _stacked(ax, frame, ARCH_ORDER, ARCH_C, {k: k for k in ARCH_ORDER},
                 ypos, height=0.72, min_label=0.11,
                 dark=("AllC", "AllD", "WSLS"))
        ax.set_yticks(ypos)
        ax.set_yticklabels(ylab, fontsize=5.8)
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("share of agent-games")
        ax.set_ylim(ypos.max() + 0.75, -0.75)
        n = len(order)
        for k, mdl in enumerate(FRONTIER):
            block = ypos[k * n: (k + 1) * n]
            off = -0.30 if panel == 0 else -0.46
            ax.text(off - 0.045, block.mean(), MODEL_LABEL[mdl].split()[0],
                    transform=ax.get_yaxis_transform(), ha="center",
                    va="center", fontsize=6.2, color=INK, rotation=90)
            ax.plot([off, off], [block[0] - 0.42, block[-1] + 0.42],
                    transform=ax.get_yaxis_transform(), color=MODEL_C[mdl],
                    lw=1.6, clip_on=False, solid_capstyle="butt")
        if panel == 0:
            ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0.0, 1.10),
                      handlelength=0.9, columnspacing=0.8)
            pa = ax
        else:
            pb = ax
        ax.set_title(title, pad=16)

    finalize(fig, [pa, pb], ["a", "b"])
    caption(fig, "Same read-out as FR19, split by condition. If an LLM had a "
                 "strategy in the game-theoretic sense, these blocks would be "
                 "flat within a model: the payoff multiplier and the persona "
                 "label do not change what the optimal play is.")
    save(fig, "FR20_archetype_conditions")


# ==========================================================================
# FR21 -- how the label crystallises
# ==========================================================================
def fig_crystallisation(arche):
    z = np.load(DATADIR / "frontier_prefixes.npz", allow_pickle=True)
    pref, keys = z["pref"], z["keys"]
    idx = {k: i for i, k in enumerate(keys)}
    order = np.array([idx[f"{u}|{a}"] for u, a in
                      zip(arche.game_uid, arche.agent)])
    P = pref[order]                                   # (N, 10, 4)
    final = P[:, -1, :].argmax(1)
    conf_path = P[np.arange(len(P)), :, final]        # posterior of the final call

    fig = figure(W2, 2.45)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    # (a) posterior of the eventual label, by round --------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    ks = np.arange(1, P.shape[1] + 1)
    # No dispersion band here: four interquartile ribbons over this range
    # overlap into a single grey wash that hides the very curves it qualifies.
    # Panel b carries the spread, per model, where it can be read.
    rows = []
    for mdl in FRONTIER:
        m = (arche.model == mdl).to_numpy()
        mu = conf_path[m].mean(axis=0)
        model_line(ax, ks, mu, mdl)
        rows += [{"model": mdl, "round": k, "mean": v,
                  "q25": np.percentile(conf_path[m], 25, axis=0)[k - 1],
                  "q75": np.percentile(conf_path[m], 75, axis=0)[k - 1]}
                 for k, v in zip(ks, mu)]
    pd.DataFrame(rows).to_csv(TABDIR / "T_FR31_crystallisation.csv", index=False)
    ax.axhline(0.9, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    ax.text(1.05, 0.905, "confidence floor 0.90", ha="left", va="bottom",
            fontsize=5.6, color=MUTED)
    ax.set_xticks(ks)
    ax.set_xlim(1, ks[-1])
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("rounds observed")
    ax.set_ylabel("mean posterior on the eventual label")
    ax.set_title("The posterior climbs steadily", pad=6)

    # (b) when it first crosses the floor -------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    rows = []
    for mdl in FRONTIER:
        m = (arche.model == mdl).to_numpy()
        c = conf_path[m]
        crossed = c >= 0.90
        first = np.where(crossed.any(axis=1), crossed.argmax(axis=1) + 1, np.nan)
        rows.append({"model": mdl, "never": float(np.isnan(first).mean()),
                     "median_round": float(np.nanmedian(first)),
                     "q25": float(np.nanpercentile(first, 25)),
                     "q75": float(np.nanpercentile(first, 75))})
    cr = pd.DataFrame(rows)
    cr.to_csv(TABDIR / "T_FR32_crossing_round.csv", index=False)

    x = np.arange(len(cr))
    for xi, r in zip(x, cr.itertuples()):
        ax.vlines(xi, r.q25, r.q75, color=MODEL_C[r.model], lw=3.0,
                  zorder=3, capstyle="round")
        ax.plot(xi, r.median_round, "o", ms=3.4, mfc=PAGE,
                mec=MODEL_C[r.model], mew=1.0, zorder=5)
        ax.text(xi, 10.85, f"{r.never:.0%}\nnever", ha="center", va="top",
                fontsize=5.6, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m].split()[0] for m in cr.model],
                       fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks(range(1, 11))
    ax.set_ylim(0.4, 11.4)
    ax.set_ylabel("first round the posterior reaches 0.90")
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_title("Median and IQR", pad=6)

    finalize(fig, [pa, pb], ["a", "b"])
    leg = shared_model_legend(fig, [pa, pb], ncol=4)
    caption(fig, "The eventual label is the read-out's call after all ten "
                 "rounds; a traces the mean posterior it assigned to that same "
                 "label after each prefix, and the quartiles are in "
                 "T_FR31. The mean never reaches the 0.90 floor, but the median "
                 "game crosses it by round 2-4 (b) -- the average is held down "
                 "by the fifth of games that never cross at all, which are the "
                 "abstention set mined in FR25.", below=leg)
    save(fig, "FR21_crystallisation")
    return cr


# ==========================================================================
# FR22 -- how far frontier play is from any canonical rule
# ==========================================================================
def fig_rule_distance(arche):
    z = np.load(DATADIR / "frontier_prefixes.npz", allow_pickle=True)
    X, keys = z["X"], z["keys"]
    idx = {k: i for i, k in enumerate(keys)}
    order = np.array([idx[f"{u}|{a}"] for u, a in
                      zip(arche.game_uid, arche.agent)])
    Xa = X[order]

    fig = figure(W2, 2.5)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0])

    # (a) rule survival: share still consistent after k rounds ----------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)
    ks = np.arange(1, Xa.shape[1] + 1)
    rows = []
    for mdl in FRONTIER:
        m = (arche.model == mdl).to_numpy()
        surv = []
        for k in ks:
            ok = TOKEN_OK[:, Xa[m][:, :k]].all(axis=2).T.any(axis=1)
            surv.append(ok.mean())
        model_line(ax, ks, surv, mdl)
        rows += [{"model": mdl, "round": k, "surviving": s}
                 for k, s in zip(ks, surv)]
    surv_t = pd.DataFrame(rows)
    surv_t.to_csv(TABDIR / "T_FR33_rule_survival.csv", index=False)
    ax.set_xticks(ks)
    ax.set_xlim(1, ks[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("rounds observed")
    ax.set_ylabel("share still matching some canonical rule")
    ax.set_title("Rule survival on LLM play", pad=6)

    # (b) distance to the nearest rule ---------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    maxd = int(arche.min_deviations.max())
    edges = np.arange(-0.5, maxd + 1.5)
    for k, mdl in enumerate(FRONTIER):
        v = arche.loc[arche.model == mdl, "min_deviations"].to_numpy()
        h, _ = np.histogram(v, bins=edges)
        ax.step(np.arange(maxd + 1), h / h.sum(), where="mid",
                color=MODEL_C[mdl], lw=1.0, zorder=3 + k)
    ax.set_xticks(range(0, maxd + 1))
    ax.set_xlim(-0.5, maxd + 0.5)
    ax.set_xlabel("rounds violating the nearest rule")
    ax.set_ylabel("share of agent-games")
    ax.set_title("Distance to the nearest rule", pad=6)

    # (c) what the set-valued answer buys ------------------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax, axis="x")
    rows = []
    for mdl in FRONTIER:
        a = arche[arche.model == mdl]
        rows.append({"model": mdl,
                     "single": float((a.n_rule_fits == 1).mean()),
                     "set": float((a.n_rule_fits >= 1).mean()),
                     "mean_set_size": float(a.loc[a.n_rule_fits >= 1,
                                                  "n_rule_fits"].mean())})
    cov = pd.DataFrame(rows)
    cov.to_csv(TABDIR / "T_FR34_provable_coverage.csv", index=False)

    y = np.arange(len(cov))[::-1]
    for yi, r in zip(y, cov.itertuples()):
        ax.barh(yi, r.set, height=0.55, color=MODEL_C[r.model], alpha=0.35,
                edgecolor=PAGE, linewidth=0.5, zorder=2)
        ax.barh(yi, r.single, height=0.55, color=MODEL_C[r.model],
                edgecolor=PAGE, linewidth=0.5, zorder=3)
        ax.text(r.set + 0.015, yi, f"{r.set:.2f}", ha="left", va="center",
                fontsize=6.0, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].split()[0] for m in cov.model])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(cov) - 0.3)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("share with a provable rule")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=PAGE,
                             lw=0.5, label="exactly one rule"),
               plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, alpha=0.35,
                             edgecolor=PAGE, lw=0.5, label="at least one rule")]
    ax.legend(handles=handles, ncol=1, loc="lower right",
              bbox_to_anchor=(1.02, -0.02), handlelength=0.9)
    ax.set_title("Provable coverage", pad=6)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"])
    leg = shared_model_legend(fig, [pa, pb, pc], ncol=4)
    caption(fig, "Panel a is the exact matcher run on prefixes: every model "
                 "starts fully explainable, because one round cannot "
                 "contradict a rule, and most of the corpus falls out of the "
                 "canonical vocabulary within four rounds. What remains is the "
                 "residual that FR23-FR26 mine.", below=leg)
    save(fig, "FR22_rule_distance")
    return surv_t, cov


def main():
    arche = pd.read_parquet(DATADIR / "frontier_archetypes.parquet")
    ident, ood = fig_calibration()
    share, prov = fig_archetypes(arche)
    fig_archetype_conditions(arche)
    cr = fig_crystallisation(arche)
    surv, cov = fig_rule_distance(arche)
    print()
    print(ident.round(3).to_string(index=False))
    print()
    print(share.round(3).to_string())
    print()
    print(cr.round(2).to_string(index=False))
    print()
    print(cov.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
