"""FR23-FR26: what the canonical vocabulary misses.

Two thirds of frontier play matches no canonical rule.  The question is
whether that residual is *structure the vocabulary is too small to name* or
simply noise, and the only way to answer it is to widen the hypothesis class
in controlled steps and pair every coverage number with a permutation null.

The null shuffles the focal player's own action sequence in place, holding its
cooperation rate and the opponent's realised sequence fixed.  It matters: a
two-segment fit over ten rounds has enough freedom to explain a large share of
shuffled play by chance, so raw coverage is not evidence of anything.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdlib.natstyle import (CMAP_SEQ, DATADIR, FRONTIER, INK, INK2, MODEL_C,
                            MODEL_LABEL, MODEL_M, MUTED, PAGE, RULE, SPINE,
                            TABDIR, W2, bars, caption, colorbar, figure,
                            finalize, hgrid, save, shared_model_legend,
                            use_journal_style)
from pdlib.residual import (CANONICAL_4, FULL_NAMES, NAMED, encode,
                            hypothesis_coverage, shuffle_null)
from pdlib.unclassified import (CORNERS, THRESHOLD, add_buckets,
                                corner_distance, posterior_geometry,
                                reactive_coordinates, split_half_reactive)

use_journal_style()

SEED = 0
MAX_LEN = 10

CLASS_ORDER = ["canonical4", "memory1_32", "two_regime", "unexplained"]
CLASS_LABEL = {"canonical4": "4 canonical rules",
               "memory1_32": "+ all 32 memory-one",
               "two_regime": "+ one regime switch",
               "unexplained": "still unexplained"}
CLASS_C = {"canonical4": "#0072b2", "memory1_32": "#6aa8cf",
           "two_regime": "#b8d4e6", "unexplained": "#e0e0e0"}

BUCKET_ORDER = ["exact", "ambiguous", "confident", "unclassified"]
BUCKET_C = {"exact": "#0072b2", "ambiguous": "#9ecae1",
            "confident": "#f0c05a", "unclassified": "#d55e00"}
BUCKET_LABEL = {"exact": "exact rule", "ambiguous": "several rules",
                "confident": f"LSTM $\\geq$ {THRESHOLD:.2f}",
                "unclassified": f"LSTM < {THRESHOLD:.2f}"}


def action_sequences(rounds: pd.DataFrame):
    """(keys, own, opponent) action lists, one per (game, focal agent)."""
    r = rounds.sort_values(["game_uid", "agent", "round"])
    keys, own, opp = [], [], []
    for (uid, ag), d in r.groupby(["game_uid", "agent"], sort=False):
        keys.append((uid, ag))
        own.append(d.action.tolist())
        opp.append(d.opp_action.tolist())
    return keys, own, opp


# ==========================================================================
# FR23 -- widening the vocabulary, against its own null
# ==========================================================================
def fig_vocabulary(rounds):
    keys, own, opp = action_sequences(rounds)
    kf = pd.DataFrame(keys, columns=["game_uid", "agent"])
    kf = kf.merge(rounds[["game_uid", "agent", "model"]].drop_duplicates(),
                  on=["game_uid", "agent"], how="left")

    X, _ = encode(own, opp, MAX_LEN)
    cov = hypothesis_coverage(X, MAX_LEN)
    null = shuffle_null(own, opp, MAX_LEN, seed=SEED)

    def nested(c):
        """Nested classes -> disjoint shares that stack to 1."""
        return pd.DataFrame({
            "canonical4": c["canonical4"],
            "memory1_32": c["memory1_32"] & ~c["canonical4"],
            "two_regime": c["two_regime"] & ~c["memory1_32"],
            "unexplained": ~c["two_regime"] & ~c["memory1_32"],
        })

    obs, nul = nested(cov), nested(null)
    obs["model"], nul["model"] = kf.model.to_numpy(), kf.model.to_numpy()
    obs_s = obs.groupby("model")[CLASS_ORDER].mean().reindex(FRONTIER)
    nul_s = nul.groupby("model")[CLASS_ORDER].mean().reindex(FRONTIER)
    ladder = obs_s.join(nul_s, rsuffix="_null")
    ladder.to_csv(TABDIR / "T_FR35_hypothesis_ladder.csv")

    fig = figure(W2, 2.6)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.15, 0.95])

    # (a) the coverage ladder --------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    y = np.arange(len(obs_s))[::-1]
    left = np.zeros(len(obs_s))
    for cls in CLASS_ORDER:
        v = obs_s[cls].to_numpy()
        ax.barh(y, v, left=left, height=0.6, color=CLASS_C[cls],
                edgecolor=PAGE, linewidth=0.5, zorder=3, label=CLASS_LABEL[cls])
        for yi, (l, vv) in zip(y, zip(left, v)):
            if vv > 0.09:
                ax.text(l + vv / 2, yi, f"{vv:.2f}", ha="center", va="center",
                        fontsize=5.4, zorder=4,
                        color=PAGE if cls == "canonical4" else INK)
        left += v
    # cumulative null: where a shuffled sequence would already reach
    cum_null = nul_s[["canonical4", "memory1_32", "two_regime"]].sum(axis=1)
    ax.plot(cum_null.to_numpy(), y, marker="|", ms=7, mew=1.1, color=INK,
            ls="none", zorder=6, label="shuffled null, same classes")
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in obs_s.index],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("share of agent-games")
    ax.set_ylim(-0.7, len(obs_s) - 0.3)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22),
              handlelength=0.9, columnspacing=0.8)
    ax.set_title("How far a wider vocabulary gets", pad=34)

    # (b) which memory-one rules uniquely explain a game ----------------------
    # Rules that differ only in states the game never visited are
    # indistinguishable on that game -- an all-D trajectory matches every rule
    # prescribing D after T and P, whatever it says about R and S.  Counting
    # raw matches would therefore report the same share for a dozen rules, so
    # this panel keeps only the games where exactly one of the 32 fits.
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    per = cov["per_rule"]                       # (N, 32)
    n_match = per.sum(axis=1)
    levels = sorted(pd.unique(n_match))
    x = np.arange(len(levels))
    w = 0.19
    for k, mdl in enumerate(FRONTIER):
        m = (kf.model == mdl).to_numpy()
        v = [float((n_match[m] == lv).mean()) for lv in levels]
        bars(ax, x + (k - 1.5) * w, v, MODEL_C[mdl], width=w * 0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([str(lv) for lv in levels])
    ax.set_xlabel("memory-one rules matching the same game")
    ax.set_ylabel("share of agent-games")
    ax.set_xlim(-0.55, len(x) - 0.45)
    ax.set_title("The wider vocabulary is under-identified", pad=34)

    # the named rules are still worth recording, just not worth a panel: over
    # ten rounds almost no game is pinned to exactly one of the 32
    uniq = per.sum(axis=1) == 1
    which = np.where(uniq, per.argmax(axis=1), -1)
    share = np.array([(which == i).mean() for i in range(per.shape[1])])
    top = np.argsort(-share)[:10]
    pd.DataFrame({"rule": [FULL_NAMES[i] for i in top],
                  "name": [NAMED.get(FULL_NAMES[i], FULL_NAMES[i]) for i in top],
                  "share_unique": share[top],
                  "canonical": [FULL_NAMES[i] in CANONICAL_4 for i in top]
                  }).to_csv(TABDIR / "T_FR36_memory1_rules.csv", index=False)
    pd.Series(n_match).value_counts().sort_index().rename("agent_games").to_csv(
        TABDIR / "T_FR36b_rules_per_game.csv")

    # (c) excess over the null -------------------------------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax)
    classes = ["canonical4", "memory1_32", "two_regime"]
    cum_obs = obs_s[classes].cumsum(axis=1)
    cum_nul = nul_s[classes].cumsum(axis=1)
    excess = cum_obs - cum_nul
    excess.to_csv(TABDIR / "T_FR37_excess_over_null.csv")
    x = np.arange(len(classes))
    for mdl in FRONTIER:
        ax.plot(x, excess.loc[mdl], color=MODEL_C[mdl], lw=1.0, ms=3.2,
                marker=MODEL_M[mdl], mfc=MODEL_C[mdl], mec=PAGE, mew=0.5,
                zorder=4)
    ax.axhline(0, color=SPINE, lw=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["canonical\n4", "memory-one\n32", "one\nswitch"],
                       fontsize=5.8)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("coverage above the shuffled null")
    ax.set_xlim(-0.4, len(x) - 0.6)
    ax.set_title("Excess over chance", pad=34)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"])
    leg = shared_model_legend(fig, [pa, pb, pc], ncol=4, lines=False)
    caption(fig, "The classes in a are nested, so the segments are disjoint and "
                 "stack to one. The tick is where a shuffled version of the same "
                 "play already reaches with the same three classes; coverage "
                 "below that tick is not evidence of structure. Ten rounds "
                 "rarely visit all four conditioning states, so rules that "
                 "differ only in an unvisited state are indistinguishable: in b "
                 "most matched games are matched by 4 or 8 of the 32 rules at "
                 "once, and only 89 of 4,800 are pinned to exactly one. Panel c "
                 "is the cumulative excess, and it is what makes the widening "
                 "worth doing at all.", below=leg)
    save(fig, "FR23_vocabulary")
    return kf, cov, null, obs_s, excess


# ==========================================================================
# FR24 -- how much history the residual actually uses
# ==========================================================================
def fig_memory(rounds, kf, cov):
    r = rounds.sort_values(["game_uid", "agent", "round"]).copy()
    r["prev2"] = r.groupby(["game_uid", "agent"], sort=False)["prev_letter"].shift(1)
    r = r.dropna(subset=["prev2"])
    r = r[r.prev_letter != "E"]

    fig = figure(W2, 2.4)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

    # (a) BIC of nested predictors of the next move ---------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax)

    def bic(labels, y):
        ll, k = 0.0, 0
        for v in pd.unique(labels):
            yy = y[labels == v]
            if not len(yy):
                continue
            p = np.clip(yy.mean(), 1e-9, 1 - 1e-9)
            ll += float((yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum())
            k += 1
        return -2 * ll + k * np.log(len(y))

    rows = []
    for mdl in FRONTIER:
        d = r[r.model == mdl]
        y = d.coop.to_numpy()
        p1 = d.prev_letter.to_numpy().astype(str)
        p2 = d.prev2.to_numpy().astype(str)
        late = (d.round_frac.to_numpy() > 0.5).astype(str)
        b0 = bic(np.zeros(len(y)), y)
        rows.append({"model": mdl, "n": len(y),
                     "memoryless": 0.0,
                     "memory-one": bic(p1, y) - b0,
                     "memory-two": bic(np.char.add(p2, p1), y) - b0,
                     "memory-one + phase": bic(np.char.add(p1, late), y) - b0})
    mem = pd.DataFrame(rows)
    mem.to_csv(TABDIR / "T_FR38_memory_depth.csv", index=False)

    cols = ["memoryless", "memory-one", "memory-two", "memory-one + phase"]
    tick = ["memoryless", "memory-one", "memory-two", "memory-one\n+ game phase"]
    x = np.arange(len(cols))
    for mdl in FRONTIER:
        v = mem.loc[mem.model == mdl, cols].iloc[0].to_numpy(dtype=float)
        ax.plot(x, v, "-", color=MODEL_C[mdl], lw=1.0, marker=MODEL_M[mdl],
                ms=3.4, mfc=MODEL_C[mdl], mec=PAGE, mew=0.5, zorder=4)
        best = int(np.argmin(v))
        ax.plot(x[best], v[best], "o", ms=6.5, mfc="none", mec=MODEL_C[mdl],
                mew=1.0, zorder=5)
    ax.axhline(0, color=SPINE, lw=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(tick, fontsize=5.8, linespacing=1.3)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("$\\Delta$BIC against a memoryless model\n(lower is better)")
    ax.set_xlim(-0.35, len(x) - 0.65)
    ax.set_title("How much history the next move uses", pad=6)

    # (b) where a two-regime fit puts the switch ------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    two_only = cov["two_regime"] & ~cov["memory1_32"]
    cut = cov["switch_round"]
    rows = []
    for mdl in FRONTIER:
        m = (kf.model == mdl).to_numpy() & two_only & (cut > 0)
        h = np.bincount(cut[m], minlength=MAX_LEN + 1)[1:MAX_LEN + 1]
        h = h / max(h.sum(), 1)
        ax.plot(np.arange(1, MAX_LEN + 1), h, "-", color=MODEL_C[mdl], lw=1.0,
                marker=MODEL_M[mdl], ms=3.0, mfc=MODEL_C[mdl], mec=PAGE,
                mew=0.5, zorder=4)
        rows.append({"model": mdl, "n_two_regime_only": int(m.sum()),
                     "median_switch": float(np.median(cut[m])) if m.any() else np.nan})
    sw = pd.DataFrame(rows)
    sw.to_csv(TABDIR / "T_FR39_switch_round.csv", index=False)
    ax.set_xticks(range(1, MAX_LEN + 1))
    ax.set_xlim(1, MAX_LEN)
    ax.set_xlabel("round at which the regime switches")
    ax.set_ylabel("share of two-regime fits")
    ax.set_title("Where the regime switches", pad=6)

    finalize(fig, [pa, pb], ["a", "b"])
    leg = shared_model_legend(fig, [pa, pb], ncol=4)
    n_txt = ", ".join(f"{MODEL_LABEL[r.model].split()[0]} {r.n_two_regime_only:,}"
                      for r in sw.itertuples())
    caption(fig, f"Panel a scores all four predictors on identical rows, so the "
                 f"comparison is about how much history the behaviour uses, not "
                 f"about sample size; the ringed marker is each model's selected "
                 f"depth, and all four select memory-two. Panel b covers only "
                 f"the games a single memory-one rule cannot explain but two "
                 f"consecutive ones can ({n_txt}). The mode is round 2-3 for "
                 f"every model, with a second Gemini mode at round 9.",
            below=leg)
    save(fig, "FR24_memory_depth")
    return mem, sw


# ==========================================================================
# FR25 -- the set the pipeline declines to name
# ==========================================================================
def fig_abstention(arche):
    a = add_buckets(arche)
    a = posterior_geometry(a)

    fig = figure(W2, 2.5)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.05, 1.0])

    # (a) four-way composition ------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    comp = (a.groupby(["model", "bucket"]).size().unstack("bucket")
            .reindex(FRONTIER).reindex(columns=BUCKET_ORDER).fillna(0))
    comp = comp.div(comp.sum(axis=1), axis=0)
    comp.to_csv(TABDIR / "T_FR40_bucket_composition.csv")
    y = np.arange(len(comp))[::-1]
    left = np.zeros(len(comp))
    for b in BUCKET_ORDER:
        v = comp[b].to_numpy()
        ax.barh(y, v, left=left, height=0.6, color=BUCKET_C[b], edgecolor=PAGE,
                linewidth=0.5, zorder=3, label=BUCKET_LABEL[b])
        for yi, (l, vv) in zip(y, zip(left, v)):
            if vv > 0.09:
                ax.text(l + vv / 2, yi, f"{vv:.2f}", ha="center", va="center",
                        fontsize=5.4, zorder=4,
                        color=PAGE if b in ("exact", "unclassified") else INK)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m].replace(" ", "\n", 1) for m in comp.index],
                       fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("share of agent-games")
    ax.set_ylim(-0.7, len(comp) - 0.3)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18),
              handlelength=0.9, columnspacing=0.8)
    ax.set_title("What the pipeline will and will not name", pad=34)

    # (b) where the confidence floor falls ------------------------------------
    # The 0.90 floor is inherited from the 30-round corpus, where the posterior
    # is strongly bimodal and the cut lands in a valley.  On these 10-round
    # games it does not: there is a sharp mode at 1.0 and a broad low shoulder,
    # so the floor is a convention and its cost has to be shown, not asserted.
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    res = a[a.assignment == "approx"]
    ax.hist(res.confidence, bins=np.linspace(0.25, 1.0, 31), color="#b8d4e6",
            edgecolor=PAGE, linewidth=0.4, zorder=3)
    ax.axvline(THRESHOLD, color=INK, lw=0.8, ls=(0, (2.5, 2)), zorder=5)
    ax.text(THRESHOLD - 0.012, 0.97, f"floor {THRESHOLD:.2f}",
            transform=ax.get_xaxis_transform(), rotation=90, ha="right",
            va="top", fontsize=5.6, color=INK2)
    sens = [(t, float((res.confidence < t).sum() / len(a))) for t in
            (0.80, 0.90, 0.95)]
    pd.DataFrame(sens, columns=["floor", "unclassified_share"]).to_csv(
        TABDIR / "T_FR40b_floor_sensitivity.csv", index=False)
    txt = "abstained if floor were\n" + "\n".join(
        f"   {t:.2f}   {s:.2f}" for t, s in sens)
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=5.6, color=INK2, linespacing=1.4)
    ax.set_xlabel("LSTM top posterior, games no rule explains")
    ax.set_ylabel("agent-games")
    ax.set_xlim(0.25, 1.0)
    ax.set_title("The floor is a convention", pad=6)

    # (c) is the abstention set lost, or torn between two rules? ---------------
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax)
    unc = a[a.bucket == "unclassified"]
    conf = a[a.bucket == "confident"]
    for name, sub, colr in (("abstained", unc, "#d55e00"),
                            ("named", conf, "#f0c05a")):
        ax.hist(sub.top2_mass, bins=np.linspace(0.5, 1.0, 26), density=True,
                histtype="step", color=colr, lw=1.1, zorder=4, label=name)
    ax.axvline(0.5, color=MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    ax.set_xlabel("posterior mass on the top two labels")
    ax.set_ylabel("density")
    ax.set_xlim(0.5, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), handlelength=1.3)
    ax.set_title("Abstention is a two-way tie", pad=6)

    pair = (unc.pair.value_counts(normalize=True).head(6).rename("share")
            .to_frame())
    pair.to_csv(TABDIR / "T_FR41_abstention_pairs.csv")

    finalize(fig, [pa, pb, pc], ["a", "b", "c"])
    caption(fig, f"The 0.90 floor is inherited from the 30-round corpus, where "
                 f"the posterior is bimodal and the cut lands in a valley. Over "
                 f"ten rounds it does not: {(res.confidence >= 0.95).mean():.0%} "
                 f"of the residual sits above 0.95 and the rest is a broad "
                 f"shoulder, so moving the floor from 0.80 to 0.95 moves the "
                 f"abstained share from 0.16 to 0.28. Panel c separates two "
                 f"readings of an abstention -- a network lost among four "
                 f"labels would put about half its mass on the top two, one "
                 f"caught between two rules puts nearly all of it there. The "
                 f"abstained games sit at {unc.top2_mass.median():.2f}, so they "
                 f"are interpolations between two rules, not noise.")
    save(fig, "FR25_abstention")
    return a, comp


# ==========================================================================
# FR26 -- reactive geometry, and whether it holds still
# ==========================================================================
def fig_geometry(rounds, a):
    react = reactive_coordinates(rounds)
    react = corner_distance(react)
    react = react.merge(a[["game_uid", "agent", "model", "bucket"]],
                        on=["game_uid", "agent"], how="inner").dropna(
        subset=["p_CgivenC", "q_CgivenD"])

    fig = figure(W2, 2.55)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0])

    # (a) the reactive square --------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(SEED)
    for b, colr, alpha, size in (("exact", "#9ecae1", 0.5, 2.0),
                                 ("confident", "#f0c05a", 0.6, 2.2),
                                 ("unclassified", "#d55e00", 0.8, 2.6)):
        s = react[react.bucket == b]
        j = rng.normal(0, 0.012, (len(s), 2))
        ax.plot(s.p_CgivenC + j[:, 0], s.q_CgivenD + j[:, 1], "o", ms=size,
                mfc=colr, mec="none", alpha=alpha, ls="none",
                zorder=3 if b == "exact" else 5, label=b)
    for name, (px, qy) in CORNERS.items():
        ax.plot(px, qy, marker="*", ms=7, mfc=PAGE, mec=INK, mew=0.7,
                ls="none", zorder=6)
        ax.annotate(name, (px, qy), textcoords="offset points",
                    xytext=(-4 if px > 0.5 else 4, 5),
                    ha="right" if px > 0.5 else "left", fontsize=5.8, color=INK)
    ax.set_xlabel("$p$ = P(C | opponent cooperated)")
    ax.set_ylabel("$q$ = P(C | opponent defected)")
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10),
              handlelength=0.6, columnspacing=0.7, markerscale=2.2)
    ax.set_title("Reactive square", pad=16)

    # (b) distance to the nearest corner, per bucket ---------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    order = ["exact", "ambiguous", "confident", "unclassified"]
    data = [react.loc[react.bucket == b, "d_nearest_corner"].to_numpy()
            for b in order]
    data = [d for d in data if len(d) > 5]
    keep = [b for b, d in zip(order, [react.loc[react.bucket == b,
                                                "d_nearest_corner"].to_numpy()
                                      for b in order]) if len(d) > 5]
    parts = ax.violinplot(data, positions=np.arange(len(keep)), widths=0.7,
                          showextrema=False, showmedians=False)
    for body, b in zip(parts["bodies"], keep):
        body.set_facecolor(BUCKET_C[b])
        body.set_edgecolor(PAGE)
        body.set_linewidth(0.5)
        body.set_alpha(0.85)
    for i, d in enumerate(data):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(i, q1, q3, color=INK, lw=2.0, zorder=5)
        ax.plot(i, med, "o", ms=2.8, mfc=PAGE, mec=INK, mew=0.6, zorder=6)
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels([BUCKET_LABEL[b].replace(" ", "\n", 1) for b in keep],
                       fontsize=5.8)
    ax.tick_params(axis="x", length=0, pad=3)
    for i, d in enumerate(data):
        ax.text(i, -0.055, f"n = {len(d):,}", ha="center", va="top",
                fontsize=5.4, color=MUTED)
    ax.set_ylabel("distance to the nearest corner")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.6, len(keep) - 0.4)
    dist = (react.groupby("bucket").d_nearest_corner
            .agg(["mean", "median", "size"]))
    dist.to_csv(TABDIR / "T_FR42_corner_distance.csv")
    ax.set_title("How far from the vocabulary", pad=6)

    # (c) does the geometry reproduce across replicates? -----------------------
    # A within-game split-half is the natural test but ten rounds cannot
    # support it: it needs each conditioning state twice in each half, which
    # only a handful of games satisfy.  The replicate-level split asks the
    # same question with the power the design actually has -- do two disjoint
    # halves of the ten replicates of one condition land in the same place?
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax)
    cellcols = ["scale_nominal", "language", "dyad", "agent"]
    rc = react.merge(
        a[["game_uid", "agent"] + [c for c in cellcols if c != "agent"]],
        on=["game_uid", "agent"], how="left")
    rng = np.random.default_rng(SEED)
    rows = []
    for mdl in FRONTIER:
        s = rc[rc.model == mdl].dropna(subset=cellcols)
        uids = s.game_uid.unique()
        rp_l, rq_l = [], []
        for _ in range(200):
            assign = pd.Series(rng.integers(0, 2, len(uids)), index=uids)
            s = s.assign(half=s.game_uid.map(assign))
            piv = (s.groupby(cellcols + ["half"])[["p_CgivenC", "q_CgivenD"]]
                   .mean().unstack("half").dropna())
            if piv.shape[0] < 3 or piv.shape[1] < 4:
                continue
            for col, store in (("p_CgivenC", rp_l), ("q_CgivenD", rq_l)):
                r = np.corrcoef(piv[(col, 0)], piv[(col, 1)])[0, 1]
                if np.isfinite(r) and r > -1:
                    store.append(2 * r / (1 + r))       # Spearman-Brown
        rows.append({"model": mdl, "n_cells": int(piv.shape[0]),
                     "r_p": float(np.mean(rp_l)) if rp_l else np.nan,
                     "r_q": float(np.mean(rq_l)) if rq_l else np.nan})
    coh = pd.DataFrame(rows)
    coh.to_csv(TABDIR / "T_FR43_replicate_coherence.csv", index=False)

    x = np.arange(len(coh))
    w = 0.34
    bars(ax, x - w / 2, coh.r_p, [MODEL_C[m] for m in coh.model], width=w * 0.9)
    cont = ax.bar(x + w / 2, coh.r_q, width=w * 0.9,
                  color=[MODEL_C[m] for m in coh.model], edgecolor=PAGE,
                  linewidth=0.5, alpha=0.45, zorder=3)
    ax.axhline(0, color=SPINE, lw=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m].split()[0] for m in coh.model],
                       fontsize=6.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("split-half reliability across replicates")
    ax.set_ylim(-0.15, 1.0)
    ax.set_xlim(-0.6, len(x) - 0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=PAGE,
                             lw=0.5, label="$p$ = P(C | opp. C)"),
               plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, alpha=0.45,
                             edgecolor=PAGE, lw=0.5, label="$q$ = P(C | opp. D)")]
    ax.legend(handles=handles, ncol=1, loc="upper right",
              bbox_to_anchor=(1.02, 1.03), handlelength=0.9)
    ax.set_title("Replicate coherence", pad=6)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"])
    caption(fig, f"Both coordinates need the opponent to have played C and D at "
                 f"least once, so the {len(react):,} agent-games plotted are "
                 f"those where both are estimable; the lattice in a is small "
                 f"denominators over ten rounds, not clustering. WSLS is "
                 f"deliberately not a corner: it needs the player's own last "
                 f"action, so anything in the interior is stochastic, deeper "
                 f"than memory-one, or both. Panel c splits the ten replicates "
                 f"of each condition in half and re-estimates both coordinates; "
                 f"a position that is a property of the condition reproduces, "
                 f"one that is a property of the run does not.")
    save(fig, "FR26_geometry")
    return coh


def main():
    rounds = pd.read_parquet(DATADIR / "frontier_rounds.parquet")
    arche = pd.read_parquet(DATADIR / "frontier_archetypes.parquet")
    kf, cov, null, obs_s, excess = fig_vocabulary(rounds)
    mem, sw = fig_memory(rounds, kf, cov)
    a, comp = fig_abstention(arche)
    coh = fig_geometry(rounds, a)
    print()
    print(obs_s.round(3).to_string())
    print()
    print(excess.round(3).to_string())
    print()
    print(mem.round(0).to_string(index=False))
    print()
    print(comp.round(3).to_string())
    print()
    print(coh.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
