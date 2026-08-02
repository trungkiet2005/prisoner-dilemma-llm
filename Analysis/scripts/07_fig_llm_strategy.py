"""Apply the trained read-out model to the LLM transcripts.

F14 -- which archetype each model's play resembles, by payoff scale
F15 -- archetype by language and by persona
F16 -- how the archetype crystallises over the course of a game
F17 -- how canonical LLM play is, and what each archetype earns
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

from pdlib.lstm import StrategyLSTM, predict, predict_prefixes
from pdlib.rulebase import (consistent_mask, deviation_counts,
                            set_name)
from pdlib.seqcode import LTOI, PAD, STOI, STRATEGIES
from pdlib.style import (CMAP_DIV, CMAP_SEQ, C_COOP, C_DEFECT, DATADIR, INK,
                         INK2, LANG_LABEL, LANG_ORDER, MODEL, MODEL_ORDER,
                         MODELDIR, MUTED, PERSONALITY, STRATEGY,
                         STRATEGY_ORDER, SURFACE, TABDIR, panel_tag, savefig,
                         use_paper_style)

use_paper_style()

FAM_MODEL = {"frontier": ("h10", 10), "small": ("h30", 30)}


def load_model(tag):
    ck = torch.load(MODELDIR / f"strategy_lstm_{tag}.pt", weights_only=False)
    m = StrategyLSTM()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def build_sequences(rounds: pd.DataFrame, max_len: int):
    """One padded token sequence per (game, focal agent)."""
    r = rounds.sort_values(["game_uid", "agent", "round"])
    keys, seqs = [], []
    for (uid, ag), d in r.groupby(["game_uid", "agent"], sort=False):
        toks = d.token.tolist()[:max_len]
        keys.append((uid, ag))
        seqs.append([STOI[t] for t in toks])
    X = np.full((len(seqs), max_len), PAD, dtype=np.int64)
    L = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s
        L[i] = len(s)
    return keys, X, L


def classify_all(rounds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    out = []
    prefix_store = {}
    for fam, (tag, ml) in FAM_MODEL.items():
        model = load_model(tag)
        sub = rounds[rounds.family == fam]
        keys, X, L = build_sequences(sub, ml)
        proba = predict(model, X, L)
        pref = predict_prefixes(model, X)
        prefix_store[fam] = (keys, pref, L)
        ent = -(proba * np.log(np.clip(proba, 1e-12, 1))).sum(1) / np.log(4)

        # --- hybrid assignment ------------------------------------------
        # Exact rule matching first.  When several rules fit the observed
        # history equally well the honest answer is the set, not an argmax
        # tie-break; only trajectories that no rule explains are handed to
        # the network, and then the label means "nearest rule", not "is".
        mask = consistent_mask(X)
        n_fit = mask.sum(axis=1)
        sets = [frozenset(np.array(STRATEGIES)[row]) for row in mask]
        dev = deviation_counts(X, L)

        arche, kind = [], []
        for i in range(len(X)):
            if n_fit[i] == 1:
                arche.append(next(iter(sets[i])))
                kind.append("exact")
            elif n_fit[i] > 1:
                arche.append("Ambiguous")
                kind.append("ambiguous")
            else:
                arche.append(STRATEGIES[int(proba[i].argmax())])
                kind.append("approx")

        df = pd.DataFrame(keys, columns=["game_uid", "agent"])
        df["family"] = fam
        df["archetype"] = arche
        df["assignment"] = kind
        df["rule_set"] = [set_name(s) for s in sets]
        df["n_rule_fits"] = n_fit
        df["min_deviations"] = dev.min(axis=1)
        df["lstm_archetype"] = [STRATEGIES[i] for i in proba.argmax(1)]
        df["confidence"] = proba.max(1)
        df["entropy"] = ent
        for i, s in enumerate(STRATEGIES):
            df[f"p_{s}"] = proba[:, i]
        out.append(df)
    res = pd.concat(out, ignore_index=True)
    res = res.merge(games[["game_uid", "agent", "model", "language",
                           "scale_nominal", "personality", "opp_personality",
                           "dyad", "coop_rate", "efficiency",
                           "payoff_per_round"]],
                    on=["game_uid", "agent"], how="left")
    res.to_parquet(DATADIR / "llm_archetypes.parquet", index=False)
    np.savez_compressed(
        DATADIR / "llm_prefix_posteriors.npz",
        **{f"{fam}_pref": v[1] for fam, v in prefix_store.items()},
        **{f"{fam}_len": v[2] for fam, v in prefix_store.items()},
        frontier_keys=np.array([f"{a}|{b}" for a, b in prefix_store["frontier"][0]]),
        small_keys=np.array([f"{a}|{b}" for a, b in prefix_store["small"][0]]))
    return res, prefix_store


AMBIGUOUS = "Ambiguous"
CATS = STRATEGY_ORDER + [AMBIGUOUS]
CAT_COLOR = dict(STRATEGY, **{AMBIGUOUS: MUTED})


def share_table(df, by):
    """Share of each category, split into exact-rule and nearest-rule parts.

    Columns are a MultiIndex (category, kind) so the plot can keep the two
    apart: a solid block is a trajectory that provably follows the rule, a
    hatched block is only the network's nearest guess.
    """
    d = df.copy()
    d["kind"] = np.where(d.assignment == "approx", "approx", "exact")
    t = (d.groupby(by + ["archetype", "kind"] if isinstance(by, list)
                   else [by, "archetype", "kind"]).size()
         .unstack(["archetype", "kind"], fill_value=0))
    t = t.div(t.sum(axis=1), axis=0)
    full = pd.MultiIndex.from_product([CATS, ["exact", "approx"]])
    return t.reindex(columns=full, fill_value=0.0)


def stacked(ax, tab, xlabels, xlabel=None, annotate=0.10, rot=0, legend=False):
    bottom = np.zeros(len(tab))
    x = np.arange(len(tab))
    handles = []
    for s in CATS:
        for kind in ("exact", "approx"):
            v = tab[(s, kind)].to_numpy()
            b = ax.bar(x, v, bottom=bottom, color=CAT_COLOR[s],
                       edgecolor=SURFACE, linewidth=1.2, width=0.72,
                       hatch="" if kind == "exact" else "///",
                       alpha=1.0 if kind == "exact" else 0.72)
            if kind == "exact":
                handles.append((b, s))
            bottom = bottom + v
        # annotate the category total, not the two halves
        tot = tab[(s, "exact")].to_numpy() + tab[(s, "approx")].to_numpy()
        for xi, (top, q) in enumerate(zip(bottom, tot)):
            if q >= annotate:
                ax.text(xi, top - q / 2, f"{q:.2f}", ha="center", va="center",
                        fontsize=6.2, color="white", fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=rot, ha="right" if rot else "center")
    ax.set_ylim(0, 1)
    ax.grid(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if legend:
        ax.legend([h for h, _ in handles], [n for _, n in handles],
                  ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.20),
                  fontsize=7)
    return handles


# --------------------------------------------------------------------------
def fig_archetypes(res):
    fig = plt.figure(figsize=(11.2, 7.0))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.36,
                          height_ratios=[1, 1.05])

    # (a) overall per model ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    t = share_table(res, "model").reindex(MODEL_ORDER)
    stacked(ax, t, [m.split("-")[0] for m in t.index], rot=25, legend=True)
    ax.set_ylabel("share of agent-games")
    ax.set_title("Archetype mix per model")
    panel_tag(ax, "a", dx=-0.26)

    # (b) per scale, frontier -------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    t = share_table(res[res.family == "frontier"], "scale_nominal")
    stacked(ax, t, [f"×{s:g}" for s in t.index], "payoff scale")
    ax.set_title("Frontier models")
    panel_tag(ax, "b", dx=-0.24)

    # (c) per scale, open-weight ----------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    t = share_table(res[res.family == "small"], "scale_nominal")
    stacked(ax, t, [f"×{s:g}" for s in t.index], "payoff scale")
    ax.set_title("Open-weight models")
    panel_tag(ax, "c", dx=-0.24)

    # (d) model x scale grid ---------------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    labels, cols = [], []
    rows = []
    for mdl in MODEL_ORDER:
        sub = res[res.model == mdl]
        for sc in sorted(sub.scale_nominal.unique()):
            t = share_table(sub[sub.scale_nominal == sc], "model")
            rows.append(t.iloc[0])
            labels.append(f"×{sc:g}")
            cols.append(mdl)
    tab = pd.DataFrame(rows).reset_index(drop=True)
    stacked(ax, tab, labels, annotate=0.14, rot=45)
    # model separators + names
    start = 0
    for mdl in MODEL_ORDER:
        n = sum(1 for c in cols if c == mdl)
        ax.text(start + (n - 1) / 2, 1.04, mdl, ha="center", fontsize=7.5,
                color=MODEL[mdl], fontweight="bold")
        if start > 0:
            ax.axvline(start - 0.5, color=MUTED, lw=0.8)
        start += n
    ax.set_ylabel("share of agent-games")
    ax.set_xlabel("payoff scale, within each model")
    ax.set_ylim(0, 1.10)
    ax.set_title("Archetype mix shifts with the stake")
    ax.text(1.0, -0.30, "solid = exact rule match   ·   hatched = LSTM nearest rule",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
            color=MUTED)
    panel_tag(ax, "d", dx=-0.055)

    fig.suptitle("What canonical strategy does each LLM's play resemble?",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F14_archetype_by_scale")


# --------------------------------------------------------------------------
def fig_archetype_context(res):
    fig = plt.figure(figsize=(11.2, 6.8))
    gs = fig.add_gridspec(2, 3, hspace=0.66, wspace=0.38)

    # (a) by language ---------------------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    rows, labels, cols = [], [], []
    for mdl in MODEL_ORDER:
        sub = res[res.model == mdl]
        for lg in LANG_ORDER:
            t = share_table(sub[sub.language == lg], "model")
            rows.append(t.iloc[0] if len(t) else pd.Series(0.0, index=t.columns))
            labels.append(lg.upper())
            cols.append(mdl)
    stacked(ax, pd.DataFrame(rows).reset_index(drop=True), labels, annotate=2.0)
    start = 0
    for mdl in MODEL_ORDER:
        n = sum(1 for c in cols if c == mdl)
        ax.text(start + (n - 1) / 2, 1.04, mdl.split("-")[0], ha="center",
                fontsize=7.5, color=MODEL[mdl], fontweight="bold")
        if start > 0:
            ax.axvline(start - 0.5, color=MUTED, lw=0.8)
        start += n
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("share")
    ax.set_title("Prompt language rewrites the strategy")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=CAT_COLOR[c]) for c in CATS]
              + [plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch="///",
                               edgecolor=MUTED)],
              CATS + ["LSTM nearest rule"], ncol=6,
              loc="lower center", bbox_to_anchor=(0.5, -0.32), fontsize=6.8)
    panel_tag(ax, "a", dx=-0.09)

    # (b) by persona ----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    rows, labels = [], []
    for pers in ("cooperative", "selfish"):
        for opp in ("cooperative", "selfish"):
            t = share_table(res[(res.personality == pers) &
                                (res.opp_personality == opp)], "personality")
            rows.append(t.iloc[0])
            labels.append(f"{pers[0].upper()} vs {opp[0].upper()}")
    stacked(ax, pd.DataFrame(rows).reset_index(drop=True), labels)
    ax.set_title("Persona dyad")
    panel_tag(ax, "b", dx=-0.28)

    # (c) archetype vs cooperation rate --------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    parts = [res.loc[res.archetype == s, "coop_rate"].to_numpy() for s in CATS]
    vp = ax.violinplot(parts, showextrema=False, widths=0.82)
    for b, s in zip(vp["bodies"], CATS):
        b.set_facecolor(CAT_COLOR[s])
        b.set_alpha(0.8)
        b.set_edgecolor(SURFACE)
    for i, p in enumerate(parts):
        ax.plot(i + 1, np.median(p), "o", ms=5, color=INK, mec=SURFACE, mew=1.0)
    ax.set_xticks(range(1, len(CATS) + 1))
    ax.set_xticklabels(CATS, rotation=25, ha="right")
    ax.set_ylabel("cooperation rate")
    ax.set_title("Sanity check on the read-out")
    panel_tag(ax, "c", dx=-0.28)

    # (d) archetype vs payoff -------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    tab = res.groupby(["family", "archetype"]).efficiency.mean().unstack()
    tab = tab.reindex(columns=CATS)
    x = np.arange(len(CATS))
    for k, fam in enumerate(("frontier", "small")):
        ax.bar(x + (k - 0.5) * 0.38, tab.loc[fam], width=0.36,
               color=[CAT_COLOR[s] for s in CATS],
               alpha=1.0 if k == 0 else 0.55, edgecolor=SURFACE, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(CATS, rotation=25, ha="right")
    ax.set_ylabel("payoff efficiency")
    ax.set_ylim(0, float(np.nanmax(tab.to_numpy())) * 1.16)
    ax.set_title("What each archetype earns\n"
                 "solid = frontier, faded = open-weight", fontsize=8.6)
    panel_tag(ax, "d", dx=-0.28, dy=1.14)

    # (e) archetype pairing matrix -------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    a1 = res[res.agent == 1].set_index("game_uid").archetype
    a2 = res[res.agent == 2].set_index("game_uid").archetype
    pair = pd.crosstab(a1, a2.reindex(a1.index), normalize=True)
    pair = pair.reindex(index=CATS, columns=CATS).fillna(0)
    im = ax.imshow(pair.to_numpy(), cmap=CMAP_SEQ, aspect="equal")
    for i in range(len(CATS)):
        for j in range(len(CATS)):
            v = pair.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                    color="white" if v > pair.to_numpy().max() * 0.55 else INK)
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels(CATS, rotation=30, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(CATS)))
    ax.set_yticklabels(CATS, fontsize=6.5)
    ax.set_xlabel("agent 2")
    ax.set_ylabel("agent 1")
    ax.grid(False)
    ax.set_title("Who meets whom")
    panel_tag(ax, "e", dx=-0.34)

    fig.suptitle("Context determines the archetype an LLM enacts", x=0.02,
                 ha="left", fontweight="bold", color=INK)
    savefig(fig, "F15_archetype_context")


# --------------------------------------------------------------------------
def fig_crystallisation(res, prefix_store):
    fig = plt.figure(figsize=(11.2, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.74, wspace=0.44)

    key2model = res.set_index(["game_uid", "agent"]).model.to_dict()

    for k, (fam, ml) in enumerate((("frontier", 10), ("small", 30))):
        keys, pref, lens = prefix_store[fam]
        mdls = np.array([key2model[(u, a)] for u, a in keys])
        ax = fig.add_subplot(gs[0, k])
        mean = pref.mean(axis=0)
        bottom = np.zeros(ml)
        for s in STRATEGY_ORDER:
            v = mean[:, LTOI[s]]
            ax.fill_between(np.arange(1, ml + 1), bottom, bottom + v,
                            color=STRATEGY[s], label=s, lw=0.8,
                            edgecolor=SURFACE)
            bottom = bottom + v
        ax.set_xlim(1, ml)
        ax.set_ylim(0, 1)
        ax.set_xlabel("rounds observed")
        ax.set_ylabel("mean posterior")
        ax.grid(False)
        ax.set_title(f"Belief · {'frontier, 10 rounds' if fam == 'frontier' else 'open-weight, 30 rounds'}")
        if k == 0:
            ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22),
                      fontsize=7)
        panel_tag(ax, "ab"[k], dx=-0.26)

    # (c) how decisive the read-out becomes ----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    for fam, ml, col in (("frontier", 10, C_COOP), ("small", 30, C_DEFECT)):
        keys, pref, lens = prefix_store[fam]
        ent = -(pref * np.log(np.clip(pref, 1e-12, 1))).sum(2) / np.log(4)
        ax.plot(np.arange(1, ml + 1), ent.mean(0), "-o", color=col, ms=3.4,
                mec=SURFACE, mew=0.8, label=fam)
    ax.set_xlabel("rounds observed")
    ax.set_ylabel("normalised posterior entropy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Ambiguity never vanishes")
    panel_tag(ax, "c", dx=-0.30)

    # (d) per-model crystallisation for the open-weight family ----------------
    ax = fig.add_subplot(gs[1, :2])
    keys, pref, lens = prefix_store["small"]
    mdls = np.array([key2model[(u, a)] for u, a in keys])
    for mdl in MODEL_ORDER:
        m = mdls == mdl
        if not m.any():
            continue
        ax.plot(np.arange(1, 31), pref[m][:, :, LTOI["AllD"]].mean(0), "-",
                color=MODEL[mdl], label=mdl)
    keysf, preff, _ = prefix_store["frontier"]
    mdlf = np.array([key2model[(u, a)] for u, a in keysf])
    for mdl in MODEL_ORDER:
        m = mdlf == mdl
        if not m.any():
            continue
        ax.plot(np.arange(1, 11), preff[m][:, :, LTOI["AllD"]].mean(0),
                linestyle=(0, (3, 2)), color=MODEL[mdl], label=mdl)
    ax.set_xlabel("rounds observed")
    ax.set_ylabel("posterior mass on AllD")
    ax.set_title("Posterior mass on AllD over the game")
    ax.text(0.0, 1.02, "solid = open-weight (30 rounds), dashed = frontier (10 rounds)",
            transform=ax.transAxes, fontsize=6.4, color=MUTED, va="bottom")
    ax.legend(ncol=6, fontsize=6.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.19))
    panel_tag(ax, "d", dx=-0.075)

    # (e) canonicality of LLM play vs the synthetic corpus -------------------
    ax = fig.add_subplot(gs[1, 2])
    ref = np.load(DATADIR / "clf_test_h10.npz", allow_pickle=True)["proba"].max(1)
    ax.hist(ref, bins=40, range=(0.25, 1), density=True, histtype="stepfilled",
            color=MUTED, alpha=0.35, label="synthetic corpus")
    for fam, col in (("frontier", C_COOP), ("small", C_DEFECT)):
        ax.hist(res.loc[res.family == fam, "confidence"], bins=40,
                range=(0.25, 1), density=True, histtype="step", lw=1.9,
                color=col, label=f"{fam} LLM play")
    ax.set_xlabel("posterior of the assigned archetype")
    ax.set_ylabel("density")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("LLM play is less canonical")
    panel_tag(ax, "e", dx=-0.32)

    fig.suptitle("Reading the strategy as the game unfolds", x=0.02, ha="left",
                 fontweight="bold", color=INK)
    savefig(fig, "F16_crystallisation")


# --------------------------------------------------------------------------
def summary_table(res):
    t = (res.groupby(["family", "model", "scale_nominal"])
         .archetype.value_counts(normalize=True).unstack()
         .reindex(columns=CATS).fillna(0).round(3).reset_index())
    t.to_csv(TABDIR / "T13_archetype_shares.csv", index=False)
    conf = (res.groupby("model")
            .agg(confidence=("confidence", "mean"),
                 entropy=("entropy", "mean"),
                 exact_rule=("assignment", lambda s: (s == "exact").mean()),
                 ambiguous=("assignment", lambda s: (s == "ambiguous").mean()),
                 lstm_fallback=("assignment", lambda s: (s == "approx").mean()),
                 min_dev=("min_deviations", "mean"))
            .reindex(MODEL_ORDER).round(3))
    conf.to_csv(TABDIR / "T14_archetype_confidence.csv")
    return t, conf


def main():
    rounds = pd.read_parquet(DATADIR / "rounds.parquet")
    games = pd.read_parquet(DATADIR / "games.parquet")
    res, prefix_store = classify_all(rounds, games)
    fig_archetypes(res)
    fig_archetype_context(res)
    fig_crystallisation(res, prefix_store)
    t, conf = summary_table(res)
    print(share_table(res, "model").reindex(MODEL_ORDER).round(3).to_string())
    print()
    print(conf.to_string())


if __name__ == "__main__":
    main()
