"""Section 7: game theory, strategy attribution and temporal dynamics.

Why this section is built the way it is
---------------------------------------
Every other section asks what the models *produce*.  This one asks what policy
they are *running*, which is a harder question because a policy is not observed.
All that is observed is ten binary choices per agent-game against an opponent
whose own choices those ten decisions helped cause.  Three consequences shape
the whole section.

**1.  A strategy label is an inference, and its identifiability depends on the
opponent.**  The corpus labels each agent-game by exact consistency with the
memory-one rules AllC, AllD, TFT, WSLS and GRIM.  Exact consistency is a strong
test, and it fails to single out a rule in two different ways: 4,245 agent-games
are consistent with several rules at once ("ambiguous") and 14,192 with none
("unclassified").  Together those are 77% of the corpus, and the honest reading
is that the rule vocabulary, not the corpus, is the thing that is small.
Ambiguity in particular is a *correct* answer: an agent that cooperated in all
ten rounds against an opponent who never defected has produced a history that
AllC, TFT, WSLS and GRIM all predict exactly, so no experiment inside this game
could separate them.  We verify below that this description accounts for
4,171 of the 4,245 ambiguous cases exactly.

That has a sharp statistical consequence which drives G08.  A label is only
resolved when the opponent supplies a discriminating event, so label and
outcome are not independent: "AllC" is distinguishable from TFT only when the
opponent defected, which mechanically hands unique-AllC the worst payoffs in
the corpus.  Comparing mean payoff across labels without conditioning on what
the opponent did is therefore an aggregation artefact, and it reverses once you
condition.  We show both.

**2.  The residual is the result, not the leftover.**  Fifty-nine per cent of
agent-games fit no canonical rule exactly.  Reporting only the 23% that do
would be reporting the subset that a four-rule vocabulary happens to cover.  So
G02 mines the residual directly: how far from a rule it is, which rule it is
nearest, *where in the game* the deviation sits, and what nameable shapes the
sequences take.  The answer that comes out is specific and was not anticipated:
deviations are concentrated at the two boundaries of the game, round 1 and
round 10, and are lowest in the middle.  Memory-one rules describe the middle
of an LLM game well and fail at its edges.

**3.  Rounds are not observations.**  240,000 rounds are 12,000 independent
games.  Every mean here carries a bootstrap CI resampling whole games, and the
endgame test is a cluster-robust regression on the game.  The memory-one
fingerprint is aggregated over *rounds within a model*, not by averaging the
per-game pC_* columns, because those columns are NaN whenever the conditioning
state never occurred in that game (50.8% of games never see state S or T), and
averaging over the games where a state did occur would silently condition on a
selected subsample.  Pooling rounds keeps the estimand well defined.

What theory predicts and what the corpus says
---------------------------------------------
The horizon is disclosed to both agents, so backward induction predicts
unravelling: defection in round 10, hence in round 9, hence everywhere.  The
corpus shows nothing of the kind.  Pooled cooperation in round 10 sits within
0.001 of the linear trend fitted to rounds 2 to 9, and the per-model deviations
that are significant go in both directions.  This cannot be turned into a
"horizon effect" estimate, because there is no unknown-horizon arm in this
corpus to contrast against; the claim available here is the weaker and still
useful one that finite-horizon unravelling does not happen.

Finally, self-play.  Both agents in a game are the same model, so every payoff
comparison below is a within-population comparison, not a tournament.  Nothing
here licenses a statement about how one model would fare against another.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import core as C

# ordering and colours for the seven label classes actually present
STRAT_PLOT = ["AllC", "TFT", "WSLS", "GRIM", "AllD", "ambiguous", "unclassified"]
STRAT_PLOT_COL = dict(C.STRAT_COL)
STRAT_PLOT_COL["ambiguous"] = "#5f7d8c"

# canonical memory-one fingerprints in the order (R, S, T, P)
RULE_FP = {"AllC": (1, 1, 1, 1), "AllD": (0, 0, 0, 0), "TFT": (1, 0, 1, 0),
           "WSLS": (1, 0, 0, 1), "GRIM": (1, 0, 0, 0)}
STATES = ["R", "S", "T", "P"]
STATE_OUTCOME = {"R": "CC", "S": "CD", "T": "DC", "P": "DD"}


def _short(m: str) -> str:
    return m.replace("-Non-Reasoning", "").replace("-Flash-Lite", "-FL")


# --------------------------------------------------------------------------
# sequence extraction
# --------------------------------------------------------------------------
def _sequences(rounds: pd.DataFrame):
    """(game_uid, agent) -> own and opponent action strings of length 10."""
    srt = rounds.sort_values(["game_uid", "agent", "round"], kind="mergesort")
    assert len(srt) % 10 == 0, "expected exactly 10 rounds per agent-game"
    own = srt["action"].astype(str).to_numpy().reshape(-1, 10)
    opp = srt["opp_action"].astype(str).to_numpy().reshape(-1, 10)
    rnd = srt["round"].to_numpy().reshape(-1, 10)
    assert (rnd == np.arange(1, 11)).all(), "rounds are not 1..10 in every game"
    key = srt[["game_uid", "agent"]].to_numpy()[::10]
    out = pd.DataFrame({
        "game_uid": key[:, 0], "agent": key[:, 1].astype(int),
        "own_seq": ["".join(r) for r in own],
        "opp_seq": ["".join(r) for r in opp]})
    return out, own, opp


def _rule_prediction(rule: str, own: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Per-round action a rule would have taken given the realised history."""
    n, T = own.shape
    if rule == "GRIM":
        trig = np.zeros((n, T), dtype=bool)
        trig[:, 1:] = np.maximum.accumulate(opp[:, :-1] == "D", axis=1)
        return np.where(trig, "D", "C")
    prev = np.empty((n, T), dtype="<U1")
    prev[:, 0] = "E"
    oc, pc = own[:, :-1] == "C", opp[:, :-1] == "C"
    prev[:, 1:] = np.where(oc & pc, "R",
                           np.where(oc & ~pc, "S", np.where(~oc & pc, "T", "P")))
    out = np.empty_like(prev)
    for k, v in C.RULES[rule].items():
        out[prev == k] = v
    return out


def _blocks(s: str) -> int:
    return 1 + sum(s[i] != s[i + 1] for i in range(len(s) - 1))


def _taxon(s: str) -> str:
    """Nameable shape of an own-action sequence."""
    nb = _blocks(s)
    nD = s.count("D")
    if nb == 1:
        return "constant (AllC / AllD)"
    if nb == 2 and s[0] == "C":
        tail = len(s) - s.index("D")
        return ("cooperate then defect at horizon" if tail <= 3
                else "cooperate then defect, early")
    if nb == 2 and s[0] == "D":
        return ("defect once then cooperate" if s.index("C") == 1
                else "defect block then cooperate")
    if nb == 3 and nD == 1:
        return "one isolated defection"
    if nb == 3 and nD == 9:
        return "one isolated cooperation"
    if nb == 3:
        return "three blocks, other"
    if nb <= 5:
        return "4-5 blocks (multi-switch)"
    return "6+ blocks (oscillating)"


def _boot(values, clusters, n_boot=1000, seed=0):
    """Pooled (observation-weighted) mean with a cluster bootstrap CI.

    C.cluster_boot_ci returns the mean of the per-cluster means, which is a
    different estimand: it weights every game equally regardless of how many
    rounds of the conditioning state that game contained, and so silently
    conditions on the games where the state occurred at all.  The estimand this
    section needs is the pooled probability over rounds, so the point estimate
    is sum(values) / n while whole games are still resampled for the interval.
    """
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    codes, uniq = pd.factorize(c)
    k = len(uniq)
    s = np.bincount(codes, weights=v, minlength=k)
    n = np.bincount(codes, minlength=k).astype(float)
    mean = float(s.sum() / n.sum())
    if k < 2:
        return mean, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    step = max(1, int(2_000_000 // k))
    done = 0
    while done < n_boot:
        m = min(step, n_boot - done)
        idx = rng.integers(0, k, size=(m, k))
        boots[done:done + m] = s[idx].sum(1) / n[idx].sum(1)
        done += m
    return mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _clustered_slope(df, y, x="log_scale", cluster="cell"):
    d = df[[cluster, x, y]].dropna()
    X = sm.add_constant(d[x].to_numpy(float))
    m = sm.OLS(d[y].to_numpy(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster].to_numpy()})
    ci = m.conf_int()
    return float(m.params[1]), float(ci[1][0]), float(ci[1][1]), float(m.pvalues[1])


# ==========================================================================
def run(rounds, ag, dy):
    C.use_style()
    print("\n== 07 game theory, strategy and temporal dynamics ==")

    seqdf, own_arr, opp_arr = _sequences(rounds)
    ag = ag.merge(seqdf, on=["game_uid", "agent"], how="left").reset_index(drop=True)
    assert ag["own_seq"].notna().all()
    # _sequences returns rows in (game_uid, agent) sort order; align the arrays to ag
    order = (seqdf.assign(_i=np.arange(len(seqdf)))
             .set_index(["game_uid", "agent"])["_i"]
             .reindex(pd.MultiIndex.from_frame(ag[["game_uid", "agent"]])).to_numpy())
    own_arr, opp_arr = own_arr[order], opp_arr[order]
    assert (np.array(["".join(r) for r in own_arr]) == ag["own_seq"].to_numpy()).all()

    # ----------------------------------------------------------------------
    # G01  strategy composition by model, by lambda, by pairing
    # ----------------------------------------------------------------------
    comp_rows = []
    for scope, col, order_ in [("model", "model", C.MODEL_ORDER),
                               ("scale", "scale", C.SCALES),
                               ("pairing", "dyad", C.DYADS),
                               ("language", "language", C.LANGS)]:
        ct = pd.crosstab(ag[col], ag["strategy"]).reindex(index=order_)
        ct = ct.reindex(columns=STRAT_PLOT, fill_value=0)
        for lev in order_:
            n = int(ct.loc[lev].sum())
            for s in STRAT_PLOT:
                comp_rows.append({"scope": scope, "level": lev, "strategy": s,
                                  "n": int(ct.loc[lev, s]), "n_total": n,
                                  "share": ct.loc[lev, s] / n})
    comp = pd.DataFrame(comp_rows)
    C.savetab(comp, "egt_strategy_composition")

    # the crisp statement about what "ambiguous" is
    allC_seq = ag["own_seq"] == "C" * 10
    opp_clean = ag["opp_seq"].str[:9].str.count("D") == 0
    amb_explained = int((allC_seq & opp_clean).sum())
    n_amb = int((ag.strategy == "ambiguous").sum())
    print(f"  ambiguous = {n_amb}; 'all-C vs never-defecting opponent' = {amb_explained}")

    def _stack(ax, scope, order_, xlabels, title, xlab):
        sub = comp[comp.scope == scope]
        bot = np.zeros(len(order_))
        for s in STRAT_PLOT:
            v = (sub[sub.strategy == s].set_index("level")
                 .reindex(order_)["share"].to_numpy())
            ax.bar(range(len(order_)), v, bottom=bot, width=.82,
                   color=STRAT_PLOT_COL[s], label=s, lw=.3, edgecolor=C.SURFACE)
            bot += v
        ax.set_xticks(range(len(order_)), xlabels, rotation=90, fontsize=6.5)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.grid(axis="x", visible=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    _stack(axes[0], "model", C.MODEL_ORDER, [_short(m) for m in C.MODEL_ORDER],
           "a  by model", "")
    axes[0].set_ylabel("share of agent-games")
    _stack(axes[1], "scale", C.SCALES, [f"{s:g}" for s in C.SCALES],
           "b  by payoff scale $\\lambda$", "payoff scale $\\lambda$")
    _stack(axes[2], "pairing", C.DYADS, C.DYADS, "c  by persona pairing",
           "focal vs opponent persona")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=7,
                   title="memory-one label", title_fontsize=7)
    res_share = (ag.strategy == "unclassified").mean()
    uniq = 1 - res_share - n_amb / len(ag)
    C.annotate(axes[0], f"unclassified {res_share:.0%}, ambiguous "
                        f"{n_amb/len(ag):.0%},\nuniquely labelled {uniq:.0%}",
               loc="upper center", color=C.INK)
    C.save(fig, "egt", "G01_strategy_composition",
           "Which memory-one strategies do frontier LLMs actually play, and does the "
           "mix depend on the model, on the payoff scale or on the assigned personas?",
           f"No canonical rule describes most of the corpus: {res_share:.0%} of "
           f"agent-games fit no rule exactly and {n_amb/len(ag):.0%} fit several at "
           f"once, so only {uniq:.0%} carry a unique label. The identified mix is "
           "strongly model-specific (Claude-Haiku-4.5 is 90% unclassified, Grok is "
           "39% ambiguous), varies little across the five decades of lambda, and "
           "shifts sharply with persona pairing: cooperative-versus-cooperative pairs "
           "produce 37% ambiguous and 24% AllD while mixed pairs produce far less of "
           "either.",
           "stacked composition bars", "strategy label x model, scale, dyad")

    # ----------------------------------------------------------------------
    # G02  mining the residual
    # ----------------------------------------------------------------------
    mask_res = (ag.strategy == "unclassified").to_numpy()
    res = ag[mask_res].copy()
    own_r, opp_r = own_arr[mask_res], opp_arr[mask_res]

    mismatch = np.zeros((len(res), 10), dtype=bool)
    for rule in RULE_FP:
        m = (res["nearest_rule"] == rule).to_numpy()
        if m.sum():
            mismatch[m] = _rule_prediction(rule, own_r[m], opp_r[m]) != own_r[m]
    res["taxon"] = res["own_seq"].map(_taxon)
    ov = mismatch.mean(0)

    tax = (res.groupby("taxon").size().rename("n").reset_index()
           .sort_values("n", ascending=False))
    tax["share_of_residual"] = tax["n"] / len(res)
    tax["share_of_corpus"] = tax["n"] / len(ag)
    tax["median_rule_distance"] = tax["taxon"].map(
        res.groupby("taxon")["rule_distance"].median())
    tax["mean_coop_rate"] = tax["taxon"].map(res.groupby("taxon")["coop_rate"].mean())
    tax["mean_utility"] = tax["taxon"].map(res.groupby("taxon")["utility"].mean())
    tax["mean_opp_coop_rate"] = tax["taxon"].map(
        res.groupby("taxon")["opp_coop_rate"].mean())
    C.savetab(tax, "egt_residual_taxonomy")

    n_horizon = int((res.taxon == "cooperate then defect at horizon").sum())
    frac_single = float((res.own_seq.map(_blocks) == 2).mean())

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.2))
    # a  how far from a rule, split by which rule is nearest
    ax = axes[0, 0]
    bot = np.zeros(5)
    xs = np.arange(1, 6)
    for rule in ["AllC", "TFT", "WSLS", "GRIM", "AllD"]:
        v = np.array([((res.nearest_rule == rule) & (res.rule_distance == k)).sum()
                      for k in xs], float)
        ax.bar(xs, v, bottom=bot, color=C.STRAT_COL[rule], label=rule, width=.8,
               lw=.3, edgecolor=C.SURFACE)
        bot += v
    ax.set_xlabel("Hamming distance to the nearest rule (out of 10 moves)")
    ax.set_ylabel("agent-games")
    ax.set_title("a  most residuals are 1-2 moves off")
    ax.legend(fontsize=6.5, ncol=2, loc="upper right", title="nearest rule",
              title_fontsize=6.5)
    ax.set_ylim(0, bot.max() * 1.42)
    C.annotate(ax, f"{(res.rule_distance <= 2).mean():.0%} of the residual is within 2 "
                   f"moves\nof a canonical rule (median "
                   f"{res.rule_distance.median():.0f})", loc="upper left")

    # b  where in the game the rules fail
    ax = axes[0, 1]
    ax.plot(range(1, 11), ov, "o-", color=C.INK, lw=2.0, zorder=5, label="all residual")
    for rule in ["AllC", "AllD", "TFT", "WSLS", "GRIM"]:
        m = (res["nearest_rule"] == rule).to_numpy()
        if m.sum() > 300:
            ax.plot(range(1, 11), mismatch[m].mean(0), "-", lw=1.0, alpha=.85,
                    color=C.STRAT_COL[rule], label=rule)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("round")
    ax.set_ylabel("P(move differs from nearest rule)")
    ax.set_title("b  rules fail at the edges of the game")
    ax.legend(fontsize=6.5, ncol=3, loc="upper center")
    ax.set_ylim(0, .78)
    C.annotate(ax, f"round 1 {ov[0]:.2f}  |  mid-game min {ov[4:8].min():.2f}  |  "
                   f"round 10 {ov[9]:.2f}", loc="lower left")

    # c  nameable shapes
    ax = axes[1, 0]
    t = tax.sort_values("n")
    ax.barh(range(len(t)), t["n"], color=C.MUTED)
    hl = np.flatnonzero(t["taxon"].to_numpy() == "cooperate then defect at horizon")
    ax.barh(hl, t["n"].to_numpy()[hl], color=C.C_DEFECT)
    ax.set_yticks(range(len(t)), t["taxon"], fontsize=6.5)
    ax.set_xlabel("agent-games in the residual")
    ax.set_title("c  what the residual actually does")
    for i, v in enumerate(t["n"]):
        ax.text(v + len(res) * .012, i, f"{v/len(res):.0%}", va="center", fontsize=6)
    ax.set_xlim(0, t["n"].max() * 1.24)
    ax.grid(axis="y", visible=False)

    # d  the single-switch family in detail
    ax = axes[1, 1]
    cd = res[res.own_seq.str.match(r"^C+D+$")]
    dc = res[res.own_seq.str.match(r"^D+C+$")]
    sw_cd = cd["own_seq"].str.index("D") + 1
    sw_dc = dc["own_seq"].str.index("C") + 1
    w = .4
    hc = np.array([(sw_cd == k).sum() for k in range(2, 11)], float)
    hd = np.array([(sw_dc == k).sum() for k in range(2, 11)], float)
    ax.axvspan(7.5, 10.5, color=C.C_DEFECT, alpha=.07, lw=0)
    ax.bar(np.arange(2, 11) - w / 2, hc, width=w, color=C.C_DEFECT,
           label="C block then D block")
    ax.bar(np.arange(2, 11) + w / 2, hd, width=w, color=C.C_COOP,
           label="D block then C block")
    ax.set_xticks(range(2, 11))
    ax.set_xlabel("round at which the single switch happens")
    ax.set_ylabel("agent-games")
    ax.set_title("d  single-switch sequences")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_ylim(0, max(hc.max(), hd.max()) * 1.75)
    C.annotate(ax, f"'AllC except the final 1-3 rounds'\n(shaded): {n_horizon} "
                   f"agent-games = {n_horizon/len(res):.1%}\nof the residual, "
                   f"{n_horizon/len(ag):.1%} of the corpus", loc="upper left")
    C.save(fig, "egt", "G02_residual_taxonomy",
           "Fifty-nine per cent of agent-games fit no canonical memory-one rule "
           "exactly. What are those games actually doing, and is the deviation "
           "structured or noise?",
           f"The residual is structured and close to the rules: "
           f"{(res.rule_distance <= 2).mean():.0%} of it lies within two moves of a "
           f"canonical rule. The deviations are concentrated at the two boundaries of "
           f"the game, {ov[0]:.2f} in round 1 and {ov[9]:.2f} in round 10 against "
           f"{ov[4:8].min():.2f} in mid-game, so memory-one rules describe the middle "
           f"of an LLM game and fail at its opening and its close. The nameable "
           f"horizon strategy 'AllC except the final one to three rounds' exists but "
           f"is rare: {n_horizon} agent-games, {n_horizon/len(res):.1%} of the "
           f"residual. Only {frac_single:.0%} of residual sequences are a single "
           "switch at all; the modal residual oscillates.",
           "stacked histogram + per-round deviation + taxonomy bars",
           "rule_distance, nearest_rule, own action sequence, round")

    # ----------------------------------------------------------------------
    # G03  memory-one fingerprint
    # ----------------------------------------------------------------------
    rr = rounds[rounds.prev_letter != "E"]
    fp_rows = []
    for scope, col, order_ in [("model", "model", C.MODEL_ORDER),
                               ("language", "language", C.LANGS),
                               ("pairing", "dyad", C.DYADS)]:
        for lev in order_:
            sub = rr[rr[col] == lev]
            row = {"scope": scope, "level": lev, "n_rounds": len(sub)}
            for L in STATES:
                v = sub[sub.prev_letter == L]
                m, lo, hi = _boot(v["coop"].to_numpy(), v["game_uid"].to_numpy(), 800)
                row[f"pC_{L}"] = m
                row[f"pC_{L}_lo"], row[f"pC_{L}_hi"] = lo, hi
                row[f"n_{L}"] = len(v)
            row["recip_partial"] = .5 * ((row["pC_R"] - row["pC_S"])
                                         + (row["pC_T"] - row["pC_P"]))
            row["persist_partial"] = .5 * ((row["pC_R"] - row["pC_T"])
                                           + (row["pC_S"] - row["pC_P"]))
            fp_rows.append(row)
    fp = pd.DataFrame(fp_rows)
    rule_rows = []
    for rule, v in RULE_FP.items():
        rule_rows.append({
            "scope": "canonical rule", "level": rule, "n_rounds": np.nan,
            "pC_R": v[0], "pC_S": v[1], "pC_T": v[2], "pC_P": v[3],
            "recip_partial": .5 * ((v[0] - v[1]) + (v[2] - v[3])),
            "persist_partial": .5 * ((v[0] - v[2]) + (v[1] - v[3]))})
    fp = pd.concat([fp, pd.DataFrame(rule_rows)], ignore_index=True)
    C.savetab(fp, "egt_fingerprint")

    fpm = fp[fp.scope == "model"].set_index("level").reindex(C.MODEL_ORDER)
    fpr = fp[fp.scope == "canonical rule"].set_index("level")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    ax = axes[0]
    rows = list(C.MODEL_ORDER) + list(RULE_FP)
    M = np.vstack([fpm[[f"pC_{L}" for L in STATES]].to_numpy(float),
                   fpr.reindex(list(RULE_FP))[[f"pC_{L}" for L in STATES]]
                   .to_numpy(float)])
    im = ax.imshow(M, cmap=C.SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), [f"{L}\n{STATE_OUTCOME[L]}" for L in STATES], fontsize=7.5)
    ax.set_yticks(range(len(rows)), [_short(r) for r in rows], fontsize=6.8)
    ax.axhline(5.5, color=C.INK, lw=1.2)
    for i in range(M.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.4,
                    color="white" if M[i, j] > .55 else C.INK)
    ax.set_title("a  P(cooperate | previous state)")
    ax.set_xlabel("previous joint outcome (models above, rules below)")
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.046, label="P(cooperate)")

    ax = axes[1]
    w = .2
    for k, L in enumerate(STATES):
        x = np.arange(6) + (k - 1.5) * w
        m = fpm[f"pC_{L}"].to_numpy(float)
        lo = fpm[f"pC_{L}_lo"].to_numpy(float)
        hi = fpm[f"pC_{L}_hi"].to_numpy(float)
        ax.bar(x, m, width=w, color=C.OUTCOME_COL[STATE_OUTCOME[L]],
               label=f"after {L} ({STATE_OUTCOME[L]})")
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor=C.INK2, lw=.8)
    ax.set_xticks(range(6), [_short(m) for m in C.MODEL_ORDER], rotation=25,
                  ha="right", fontsize=6.5)
    ax.set_ylabel("P(cooperate | previous state)")
    ax.set_ylim(0, 1.38)
    ax.set_title("b  with game-clustered 95% CIs")
    ax.legend(fontsize=6, ncol=2, loc="upper center")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    D = np.zeros((6, 5))
    for i, m in enumerate(C.MODEL_ORDER):
        vec = fpm.loc[m, [f"pC_{L}" for L in STATES]].to_numpy(float)
        for j, rule in enumerate(RULE_FP):
            D[i, j] = float(np.sqrt(((vec - np.array(RULE_FP[rule], float)) ** 2).sum()))
    im = ax.imshow(D, cmap=C.SEQ.reversed(), aspect="auto", vmin=0, vmax=D.max())
    ax.set_xticks(range(5), list(RULE_FP), fontsize=7.5)
    ax.set_yticks(range(6), [_short(m) for m in C.MODEL_ORDER], fontsize=6.8)
    for i in range(6):
        j = int(np.argmin(D[i]))
        for k in range(5):
            ax.text(k, i, f"{D[i, k]:.2f}", ha="center", va="center", fontsize=6.4,
                    color="white" if D[i, k] < D.max() * .42 else C.INK,
                    fontweight="bold" if k == j else "normal")
    nearest_rules = [list(RULE_FP)[int(np.argmin(D[i]))] for i in range(6)]
    top_rule = max(set(nearest_rules), key=nearest_rules.count)
    ax.set_title("c  distance to each canonical rule")
    ax.set_xlabel(f"canonical rule (bold = nearest). Closest match anywhere is "
                  f"{D.min():.2f};\n{nearest_rules.count(top_rule)} of 6 models are "
                  f"nearest {top_rule}, none is exact", fontsize=7.5)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.046, label="Euclidean distance in (R,S,T,P)")
    C.save(fig, "egt", "G03_memory_one_fingerprint",
           "Does the aggregate conditional-cooperation fingerprint of each model sit "
           "near any canonical memory-one strategy?",
           "No. Every model's (P(C|R), P(C|S), P(C|T), P(C|P)) vector is at least "
           f"{D.min():.2f} in Euclidean distance from the nearest canonical rule, so "
           "the aggregate policy of a model is a genuine mixture rather than a noisy "
           "copy of TFT, WSLS or GRIM. The fingerprints separate the models sharply: "
           "Grok cooperates after mutual cooperation with probability 1.00 but only "
           "0.12 after mutual defection, while Qwen3 and Gemini-3.1 cooperate more "
           "after being exploited (S) than after exploiting (T), the opposite of every "
           "reciprocal rule. Probabilities are pooled over rounds within a model, not "
           "averaged over per-game rates, because a per-game rate is undefined "
           "whenever that state never occurred in the game.",
           "heatmap + grouped bars with CI + distance heatmap",
           "P(C | previous joint outcome) x model, canonical rules")

    # ----------------------------------------------------------------------
    # G04  reciprocity and conditional cooperation
    # ----------------------------------------------------------------------
    rr2 = rr.copy()
    rr2["opp_prev_C"] = (rr2["prev_opp_action"] == "C")
    rec_rows = []
    for m in C.MODEL_ORDER:
        s = rr2[rr2.model == m]
        a1 = s[s.opp_prev_C]
        a0 = s[~s.opp_prev_C]
        m1, l1, h1 = _boot(a1["coop"].to_numpy(), a1["game_uid"].to_numpy(), 1200)
        m0, l0, h0 = _boot(a0["coop"].to_numpy(), a0["game_uid"].to_numpy(), 1200)
        pc1 = (s[s.opp_prev_C].groupby(["game_uid", "agent"], observed=True)["coop"]
               .mean().rename("c1"))
        pc0 = (s[~s.opp_prev_C].groupby(["game_uid", "agent"], observed=True)["coop"]
               .mean().rename("c0"))
        pair = pd.concat([pc1, pc0], axis=1).dropna()
        pair["d"] = pair["c1"] - pair["c0"]
        t = stats.ttest_1samp(pair["d"].to_numpy(), 0.0)
        rec_rows.append({"model": m, "pC_after_oppC": m1, "lo_C": l1, "hi_C": h1,
                         "pC_after_oppD": m0, "lo_D": l0, "hi_D": h0,
                         "reciprocity_raw": m1 - m0,
                         "n_agent_games_both_states": len(pair),
                         "paired_mean": float(pair["d"].mean()),
                         "p_paired": float(t.pvalue),
                         "recip_partial": float(fpm.loc[m, "recip_partial"]),
                         "persist_partial": float(fpm.loc[m, "persist_partial"])})
    rec = pd.DataFrame(rec_rows)
    rec["p_fdr"] = C.bh_fdr(rec["p_paired"].to_numpy())
    C.savetab(rec, "egt_reciprocity")
    for _, r_ in rec.iterrows():
        C.record_test(section="game_theory",
                      test="paired t-test on within-agent-game reciprocity",
                      comparison=f"{r_.model}: P(C|opp C) - P(C|opp D)",
                      statistic=r_.paired_mean, p_value=r_.p_paired,
                      ci_lo=np.nan, ci_hi=np.nan, n=int(r_.n_agent_games_both_states),
                      effect_size_note="difference in probability, paired within "
                                       "agent-game")
    n_neg = int((rec.reciprocity_raw < 0).sum())

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))
    ax = axes[0]
    for i, m in enumerate(C.MODEL_ORDER):
        r_ = rec[rec.model == m].iloc[0]
        ax.plot([r_.pC_after_oppD, r_.pC_after_oppC], [i, i], color=C.MUTED, lw=1.6,
                zorder=1)
        ax.plot([r_.lo_D, r_.hi_D], [i, i], color=C.C_DEFECT, lw=3.4, alpha=.35)
        ax.plot([r_.lo_C, r_.hi_C], [i, i], color=C.C_COOP, lw=3.4, alpha=.35)
        ax.scatter([r_.pC_after_oppD], [i], color=C.C_DEFECT, s=44, zorder=3)
        ax.scatter([r_.pC_after_oppC], [i], color=C.C_COOP, s=44, zorder=3)
    ax.set_yticks(range(6), [_short(m) for m in C.MODEL_ORDER], fontsize=7)
    ax.set_ylim(5.6, -1.3)
    ax.set_xlabel("P(cooperate this round)")
    ax.set_xlim(0, 1)
    ax.set_title("a  after opponent C (blue) vs D (red)")
    ax.grid(axis="y", visible=False)
    C.annotate(ax, f"{n_neg} of 6 models cooperate MORE after the\nopponent defected; "
                   f"the gap runs {rec.reciprocity_raw.min():+.2f} to "
                   f"{rec.reciprocity_raw.max():+.2f}", loc="upper left")

    ax = axes[1]
    for m in C.MODEL_ORDER:
        v = []
        for s in C.SCALES:
            q = rr2[(rr2.model == m) & (rr2.scale == s)]
            v.append(q.loc[q.opp_prev_C, "coop"].mean()
                     - q.loc[~q.opp_prev_C, "coop"].mean())
        ax.plot(C.SCALES, v, "o-", color=C.MODEL_COL[m], ms=3.2, lw=1.1,
                label=_short(m))
    pooled_rec = []
    for s in C.SCALES:
        q = rr2[rr2.scale == s]
        pooled_rec.append(q.loc[q.opp_prev_C, "coop"].mean()
                          - q.loc[~q.opp_prev_C, "coop"].mean())
    ax.plot(C.SCALES, pooled_rec, "s--", color=C.INK, lw=1.6, ms=4, label="pooled")
    ax.axhline(0, color=C.INK, lw=.8)
    ax.set_xscale("log")
    ax.set_xlabel("payoff scale $\\lambda$")
    ax.set_ylabel("reciprocity: P(C|opp C) - P(C|opp D)")
    ax.set_title("b  reciprocity across $\\lambda$")
    ax.set_ylim(-.72, .95)
    ax.legend(fontsize=6, ncol=3, loc="lower center")
    b_, lo_, hi_, p_ = _clustered_slope(ag.assign(_r=ag["reciprocity"]), "_r")
    C.annotate(ax, f"agent-game reciprocity slope\n{b_:+.4f} per decade "
                   f"[{lo_:+.4f}, {hi_:+.4f}]", loc="upper center")
    C.record_test(section="game_theory", test="OLS slope on log10(lambda), "
                                              "cluster-robust on design cell",
                  comparison="agent-game reciprocity ~ log10(lambda)",
                  statistic=b_, p_value=p_, ci_lo=lo_, ci_hi=hi_,
                  n=int(ag["reciprocity"].notna().sum()),
                  effect_size_note="change in reciprocity index per decade of lambda")

    ax = axes[2]
    rec_sorted = rec.sort_values("recip_partial").reset_index(drop=True)
    for i, r_ in rec_sorted.iterrows():
        ax.scatter(r_.recip_partial, r_.persist_partial, s=72,
                   color=C.MODEL_COL[r_.model], zorder=4)
        left = r_.recip_partial < .05
        ax.annotate(_short(r_.model), (r_.recip_partial, r_.persist_partial),
                    fontsize=6, xytext=(-7 if left else 7, 5 if i % 2 == 0 else -11),
                    textcoords="offset points", color=C.INK2,
                    ha="right" if left else "left")
    # several canonical rules project to the same point in this plane, so group them
    rule_pts = {}
    for rule in RULE_FP:
        rp = fpr.loc[rule]
        rule_pts.setdefault((round(float(rp.recip_partial), 6),
                             round(float(rp.persist_partial), 6)), []).append(rule)
    for (rx, ry), names in rule_pts.items():
        ax.scatter(rx, ry, marker="*", s=150, color=C.STRAT_COL[names[0]], zorder=3,
                   edgecolor=C.INK, linewidth=.4)
        ax.annotate(" / ".join(names), (rx, ry), fontsize=6.5, xytext=(6, -12),
                    textcoords="offset points", color=C.INK)
    ax.axhline(0, color=C.INK, lw=.8)
    ax.axvline(0, color=C.INK, lw=.8)
    ax.plot([-.6, 1.15], [-.6, 1.15], ls=":", color=C.MUTED, lw=.9)
    ax.set_xlim(-.68, 1.25)
    ax.set_ylim(-.45, 1.25)
    ax.set_xlabel("reciprocity: response to the opponent's move")
    ax.set_ylabel("persistence: response to own last move")
    ax.set_title("c  persistence beats reciprocity")
    C.annotate(ax, "dotted line = equal weight; all six\nmodels sit above it. AllC, "
                   "AllD and WSLS\nall project onto the origin.", loc="lower right")
    C.save(fig, "egt", "G04_reciprocity",
           "Is any frontier model a genuine reciprocator - does it condition on what "
           "the opponent did rather than on what it did itself?",
           "No. Decomposing the memory-one fingerprint into a partial reciprocity term "
           "(holding own previous action fixed) and a partial persistence term "
           "(holding the opponent's previous action fixed) puts all six models far "
           f"above the equal-weight line: persistence runs {rec.persist_partial.min():.2f} "
           f"to {rec.persist_partial.max():.2f} while partial reciprocity runs "
           f"{rec.recip_partial.min():.2f} to {rec.recip_partial.max():.2f}. "
           f"{n_neg} of 6 models are net anti-reciprocal on the raw contrast, "
           "cooperating more after being defected on than after being cooperated with. "
           f"Reciprocity is flat across five decades of lambda (slope {b_:+.4f} per "
           "decade at the agent-game level). What looks like conditional cooperation "
           "in the raw contrast is largely an agent repeating its own last move.",
           "dumbbell + line across lambda + 2-D policy plane",
           "P(C|opp last action), reciprocity, persistence x model, scale")

    # ----------------------------------------------------------------------
    # G05  temporal trajectory and the endgame test
    # ----------------------------------------------------------------------
    end_rows = []
    for scope in ["all"] + list(C.MODEL_ORDER):
        g = rounds if scope == "all" else rounds[rounds.model == scope]
        g = g[g["round"] >= 2]
        X = np.column_stack([np.ones(len(g)), g["round"].to_numpy(float),
                             (g["round"].to_numpy() == 10).astype(float)])
        m = sm.OLS(g["coop"].to_numpy(float), X).fit(
            cov_type="cluster", cov_kwds={"groups": g["game_uid"].to_numpy()})
        ci = m.conf_int()
        end_rows.append({
            "scope": scope, "trend_per_round": float(m.params[1]),
            "trend_p": float(m.pvalues[1]), "round10_excess": float(m.params[2]),
            "lo": float(ci[2][0]), "hi": float(ci[2][1]), "p": float(m.pvalues[2]),
            "n_rounds": len(g), "n_games": int(g["game_uid"].nunique())})
    end = pd.DataFrame(end_rows)
    end["p_fdr"] = C.bh_fdr(end["p"].to_numpy())
    C.savetab(end, "egt_endgame_test")
    for _, r_ in end.iterrows():
        C.record_test(section="game_theory",
                      test="round-10 dummy over the linear rounds 2-9 trend, "
                           "cluster-robust on game",
                      comparison=f"{r_.scope}: endgame excess cooperation",
                      statistic=r_.round10_excess, p_value=r_.p, ci_lo=r_.lo,
                      ci_hi=r_.hi, n=int(r_.n_games),
                      effect_size_note="change in P(cooperate) at round 10 relative to "
                                       "the fitted rounds 2-9 trend")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    ax = axes[0]
    for m in C.MODEL_ORDER:
        mm, ll, hh = [], [], []
        for t in range(1, 11):
            q = rounds[(rounds.model == m) & (rounds["round"] == t)]
            a_, b2_, c_ = _boot(q["coop"].to_numpy(), q["game_uid"].to_numpy(), 500)
            mm.append(a_); ll.append(b2_); hh.append(c_)
        ax.fill_between(range(1, 11), ll, hh, color=C.MODEL_COL[m], alpha=.15, lw=0)
        ax.plot(range(1, 11), mm, "o-", color=C.MODEL_COL[m], ms=3, label=_short(m))
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("round (horizon of 10 is disclosed)")
    ax.set_ylabel("cooperation rate")
    ax.set_title("a  trajectories diverge, not decay")
    ax.set_ylim(.14, 1.02)
    ax.legend(fontsize=6, ncol=2, loc="upper left")

    ax = axes[1]
    for dd_ in C.DYADS:
        v = rounds[rounds.dyad == dd_].groupby("round", observed=True)["coop"].mean()
        ax.plot(range(1, 11), v.to_numpy(), "o-", color=C.DYAD_COL[dd_], ms=3,
                label=dd_)
    pooled_t = rounds.groupby("round", observed=True)["coop"].mean()
    ax.plot(range(1, 11), pooled_t.to_numpy(), "s--", color=C.INK, ms=3.5,
            label="pooled")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("round")
    ax.set_ylabel("cooperation rate")
    ax.set_title("b  by persona pairing")
    ax.set_ylim(.42, .74)
    ax.legend(fontsize=6.5, ncol=3, loc="lower center")
    C.annotate(ax, f"pooled: round 1 {pooled_t.iloc[0]:.3f} -> "
                   f"round 10 {pooled_t.iloc[-1]:.3f}", loc="upper left")

    ax = axes[2]
    sel = end.set_index("scope").reindex(["all"] + list(C.MODEL_ORDER))
    y = np.arange(len(sel))
    ax.axvline(0, color=C.INK, lw=.9, ls="--")
    ax.errorbar(sel.round10_excess, y,
                xerr=[sel.round10_excess - sel.lo, sel.hi - sel.round10_excess],
                fmt="none", ecolor=C.INK2, lw=1.1)
    ax.scatter(sel.round10_excess, y, s=48, zorder=4,
               color=[C.INK] + [C.MODEL_COL[m] for m in C.MODEL_ORDER])
    ax.set_yticks(y, ["POOLED"] + [_short(m) for m in C.MODEL_ORDER], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("round-10 excess cooperation vs the rounds 2-9 trend")
    ax.set_title("c  no backward-induction unravelling")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(sel.lo.min() - .035, sel.hi.max() + .035)
    for i, (_, r_) in enumerate(sel.iterrows()):
        if r_.p_fdr < .05:
            ax.text(r_.hi + .004, i, "*", fontsize=10, va="center", color=C.INK2)
    sig = sel[(sel.index != "all") & (sel.p_fdr < .05)].sort_values("round10_excess")
    sig_txt = ", ".join(f"{_short(i)} {r.round10_excess:+.3f}"
                        for i, r in sig.iterrows())
    C.annotate(ax, f"pooled {sel.loc['all','round10_excess']:+.4f} "
                   f"(p = {sel.loc['all','p']:.2f});\n* FDR-significant, and the signs "
                   "disagree", loc="lower right")
    C.save(fig, "egt", "G05_temporal_endgame",
           "The horizon is disclosed, so backward induction predicts defection in the "
           "final round and unravelling before it. Does that happen?",
           f"It does not. Pooled cooperation in round 10 is "
           f"{sel.loc['all','round10_excess']:+.4f} away from the linear trend fitted "
           f"to rounds 2 to 9 (p = {sel.loc['all','p']:.2f}), and the pooled trajectory "
           f"is flat from {pooled_t.iloc[0]:.3f} in round 1 to {pooled_t.iloc[-1]:.3f} "
           f"in round 10. {len(sig)} models do show an FDR-significant round-10 "
           f"deviation, but they disagree in sign ({sig_txt}), so the pooled null "
           "averages opposing model-level effects rather than showing their absence. "
           "Because every game here discloses the horizon, the size of the horizon "
           "effect itself is not identified from this corpus.",
           "trajectory lines with CI + forest of endgame coefficients",
           "coop x round x model, dyad")

    # ----------------------------------------------------------------------
    # G06  absorption and lock-in
    # ----------------------------------------------------------------------
    a1r = rounds[rounds.agent == 1].sort_values(["game_uid", "round"],
                                                kind="mergesort")
    out_arr = a1r["outcome"].astype(str).to_numpy().reshape(-1, 10)
    guid = a1r["game_uid"].to_numpy()[::10]
    dyx = dy.set_index("game_uid").loc[guid]

    def _absorb_first(arr, val):
        hit = arr == val
        any_hit = hit.any(1)
        first = np.where(any_hit, hit.argmax(1) + 1, 0)
        # absorbed if every round from the first hit onwards equals val
        tail_ok = np.array([bool(hit[i, first[i] - 1:].all()) if any_hit[i] else False
                            for i in range(len(arr))])
        return tail_ok.astype(int), np.where(any_hit, first, np.nan)

    dd_abs, dd_first = _absorb_first(out_arr, "DD")
    cc_abs, cc_first = _absorb_first(out_arr, "CC")
    assert (dd_abs == dyx["dd_absorbed"].to_numpy()).all(), "DD absorption mismatch"
    lock = pd.DataFrame({"game_uid": guid, "model": dyx["model"].to_numpy(),
                         "scale": dyx["scale"].to_numpy(),
                         "dd_absorbed": dd_abs, "cc_absorbed": cc_abs,
                         "dd_first": dd_first, "cc_first": cc_first})

    abs_rows = []
    for scope, col, order_ in [("model", "model", C.MODEL_ORDER),
                               ("scale", "scale", C.SCALES)]:
        for lev in order_:
            s = lock[lock[col] == lev]
            for kind in ["dd", "cc"]:
                p = float(s[f"{kind}_absorbed"].mean())
                lo, hi = C.wilson(p, len(s))
                abs_rows.append({"scope": scope, "level": lev, "state": kind.upper(),
                                 "n_dyads": len(s), "absorbed_share": p,
                                 "wilson_lo": float(lo), "wilson_hi": float(hi),
                                 "median_absorption_round": float(
                                     s.loc[s[f"{kind}_absorbed"] == 1,
                                           f"{kind}_first"].median())})
    absorb = pd.DataFrame(abs_rows)
    C.savetab(absorb, "egt_absorption")
    ddl = absorb[(absorb.scope == "scale") & (absorb.state == "DD")].set_index("level")
    ratio_small = ddl.loc[0.01, "absorbed_share"] / ddl.loc[10.0, "absorbed_share"]
    neither = 1 - lock.cc_absorbed.mean() - lock.dd_absorbed.mean()

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    ax = axes[0]
    for m in C.MODEL_ORDER:
        s = lock[lock.model == m]
        surv = [1 - float(((s.dd_absorbed == 1) & (s.dd_first <= t)).mean())
                for t in range(1, 11)]
        ax.plot(range(1, 11), surv, "o-", color=C.MODEL_COL[m], ms=3, label=_short(m))
    ax.set_xticks(range(1, 11))
    ax.set_ylim(.6, 1.005)
    ax.set_xlabel("round")
    ax.set_ylabel("share of dyads not yet locked into DD")
    ax.set_title("a  survival against DD lock-in")
    ax.legend(fontsize=6, ncol=2, loc="lower left")

    ax = axes[1]
    w = .38
    x = np.arange(6)
    for k, (kind, col, lab) in enumerate([("CC", C.C_COOP, "locked into CC"),
                                          ("DD", C.C_DEFECT, "locked into DD")]):
        s = (absorb[(absorb.scope == "model") & (absorb.state == kind)]
             .set_index("level").reindex(C.MODEL_ORDER))
        ax.bar(x + (k - .5) * w, s.absorbed_share, width=w, color=col, label=lab)
        ax.errorbar(x + (k - .5) * w, s.absorbed_share,
                    yerr=[s.absorbed_share - s.wilson_lo,
                          s.wilson_hi - s.absorbed_share],
                    fmt="none", ecolor=C.INK2, lw=.8)
    ax.set_xticks(x, [_short(m) for m in C.MODEL_ORDER], rotation=25, ha="right",
                  fontsize=6.5)
    ax.set_ylabel("share of dyads")
    ax.set_ylim(0, 1.0)
    ax.set_title("b  cooperation locks in more often")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(axis="x", visible=False)
    C.annotate(ax, f"pooled CC {lock.cc_absorbed.mean():.3f} vs DD "
                   f"{lock.dd_absorbed.mean():.3f};\n{neither:.0%} of dyads end in "
                   "neither", loc="upper left")

    ax = axes[2]
    for kind, col, lab in [("CC", C.C_COOP, "locked into CC"),
                           ("DD", C.C_DEFECT, "locked into DD")]:
        s = (absorb[(absorb.scope == "scale") & (absorb.state == kind)]
             .set_index("level").reindex(C.SCALES))
        ax.fill_between(C.SCALES, s.wilson_lo, s.wilson_hi, color=col, alpha=.18, lw=0)
        ax.plot(C.SCALES, s.absorbed_share, "o-", color=col, ms=3.5, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("payoff scale $\\lambda$")
    ax.set_ylabel("share of dyads")
    ax.set_ylim(0, .45)
    ax.set_title("c  lock-in across $\\lambda$")
    ax.legend(fontsize=6.5, loc="center right")
    C.annotate(ax, f"DD lock-in at $\\lambda$=0.01 is "
                   f"{ddl.loc[0.01,'absorbed_share']:.3f}, {ratio_small:.1f}x its "
                   f"value at $\\lambda$=10", loc="upper center")
    C.save(fig, "egt", "G06_absorption_lockin",
           "Mutual defection is the stage-game equilibrium and mutual cooperation is "
           "the efficient outcome. Which of the two absorbs dyads, and how fast?",
           f"Cooperation absorbs more than three times as often as defection: "
           f"{lock.cc_absorbed.mean():.1%} of dyads reach mutual cooperation and never "
           f"leave it against {lock.dd_absorbed.mean():.1%} for mutual defection, and "
           f"{neither:.0%} settle in neither. The split is strongly model-specific: "
           "Grok locks into CC in 77% of dyads while Qwen3 locks into DD in 23% and "
           "into CC in only 7%. Lock-in also carries the payoff-scale anomaly, with DD "
           f"absorption at lambda = 0.01 running {ratio_small:.1f} times its value at "
           "lambda = 10, so the scaling effect of section 04 shows up as a change in "
           "which attractor the dyad falls into.",
           "survival curves + paired bars with Wilson CI + lambda profile",
           "dd_absorbed, cc_absorbed, first-absorption round x model, scale")

    # ----------------------------------------------------------------------
    # G07  transition matrix and the empirical Markov chain
    # ----------------------------------------------------------------------
    frm = out_arr[:, :-1].ravel()
    to = out_arr[:, 1:].ravel()
    T = pd.crosstab(pd.Series(frm, name="from"), pd.Series(to, name="to"))
    T = T.reindex(index=C.OUTCOME_ORDER, columns=C.OUTCOME_ORDER).fillna(0)
    Tn = T.div(T.sum(1), axis=0)
    Mv = Tn.to_numpy()
    w_, v_ = np.linalg.eig(Mv.T)
    i0 = int(np.argmin(np.abs(w_ - 1)))
    stat = np.real(v_[:, i0])
    stat = stat / stat.sum()
    obs_all = (pd.Series(out_arr.ravel()).value_counts(normalize=True)
               .reindex(C.OUTCOME_ORDER).fillna(0).to_numpy())
    obs_r1 = (pd.Series(out_arr[:, 0]).value_counts(normalize=True)
              .reindex(C.OUTCOME_ORDER).fillna(0).to_numpy())
    obs_r10 = (pd.Series(out_arr[:, 9]).value_counts(normalize=True)
               .reindex(C.OUTCOME_ORDER).fillna(0).to_numpy())

    tt = Tn.copy()
    tt.index.name = "from_state"
    trans_tab = tt.reset_index().melt(id_vars="from_state", var_name="to_state",
                                      value_name="probability")
    trans_tab["n_from"] = trans_tab["from_state"].map(T.sum(1))
    extra = pd.DataFrame({"from_state": ["stationary", "observed_all_rounds",
                                         "observed_round1", "observed_round10"]})
    for j, s in enumerate(C.OUTCOME_ORDER):
        extra[s] = [stat[j], obs_all[j], obs_r1[j], obs_r10[j]]
    extra = extra.melt(id_vars="from_state", var_name="to_state",
                       value_name="probability")
    extra["n_from"] = np.nan
    C.savetab(pd.concat([trans_tab, extra], ignore_index=True), "egt_transitions")
    tvd = float(.5 * np.abs(obs_all - stat).sum())
    tvd1 = float(.5 * np.abs(obs_r1 - stat).sum())

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    ax = axes[0]
    pc = []
    for L in STATES:
        q = rr[rr.prev_letter == L]
        m_, lo_2, hi_2 = _boot(q["coop"].to_numpy(), q["game_uid"].to_numpy(), 1500)
        pc.append([m_, lo_2, hi_2, len(q)])
    pc = np.array(pc, float)
    ax.bar(range(4), pc[:, 0], width=.72,
           color=[C.OUTCOME_COL[STATE_OUTCOME[L]] for L in STATES])
    ax.errorbar(range(4), pc[:, 0], yerr=[pc[:, 0] - pc[:, 1], pc[:, 2] - pc[:, 0]],
                fmt="none", ecolor=C.INK2, lw=1.0)
    for k, rule in enumerate(["TFT", "WSLS", "GRIM"]):
        ax.scatter(np.arange(4) + (k - 1) * .17, RULE_FP[rule], marker="_", s=150,
                   color=C.STRAT_COL[rule], lw=1.8, zorder=5, label=rule)
    ax.set_xticks(range(4), [f"after {L}\n({STATE_OUTCOME[L]})" for L in STATES],
                  fontsize=7.5)
    ax.set_ylim(0, 1.34)
    ax.set_ylabel("P(cooperate next round)")
    ax.set_title("a  response to the last joint state")
    ax.legend(fontsize=6.5, ncol=3, loc="upper center", title="rule prediction",
              title_fontsize=6.5)
    ax.grid(axis="x", visible=False)
    for i in range(4):
        ax.text(i, pc[i, 2] + .03, f"n = {int(pc[i, 3]):,}", ha="center", va="bottom",
                fontsize=6, color=C.INK2)
    C.annotate(ax, f"after being exploited (S) the pooled\nresponse is "
                   f"{pc[1,0]:.2f}, while TFT, WSLS and\nGRIM all predict 0",
               loc="upper right")

    ax = axes[1]
    im = ax.imshow(Mv, cmap=C.SEQ, vmin=0, vmax=Mv.max(), aspect="auto")
    ax.set_xticks(range(4), C.OUTCOME_ORDER, fontsize=7.5)
    ax.set_yticks(range(4), C.OUTCOME_ORDER, fontsize=7.5)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{Mv[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if Mv[i, j] > Mv.max() * .55 else C.INK)
    ax.set_xlabel("joint outcome in round t+1")
    ax.set_ylabel("joint outcome in round t")
    ax.set_title("b  joint-state transition matrix")
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.046, label="transition probability")
    C.annotate(ax, f"mean diagonal {np.trace(Mv)/4:.2f}", loc="lower left",
               color="white")

    ax = axes[2]
    w2 = .27
    x = np.arange(4)
    for k, (vals, col, lab) in enumerate([
            (obs_r1, C.MUTED, "observed round 1"),
            (obs_all, C.C_COOP, "observed all rounds"),
            (stat, C.OUTCOME_COL["DC"], "chain stationary limit")]):
        ax.bar(x + (k - 1) * w2, vals, width=w2, color=col, label=lab)
    ax.set_xticks(x, [f"{o}\n{C.OUTCOME_LABEL[o].split('  ')[1].split(' ')[0]}"
                      for o in C.OUTCOME_ORDER], fontsize=7.5)
    ax.set_ylabel("share of dyad-rounds")
    ax.set_ylim(0, .55)
    ax.set_title("c  observed vs stationary")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(axis="x", visible=False)
    C.annotate(ax, f"total variation distance to the limit:\nround 1 {tvd1:.3f}, all "
                   f"rounds {tvd:.3f}", loc="upper left")
    C.save(fig, "egt", "G07_transition_markov",
           "Treating the dyad as a Markov chain on the four joint outcomes, what does "
           "the empirical one-step kernel look like and has the process reached its "
           "own stationary distribution inside ten rounds?",
           f"The kernel is strongly self-reinforcing: the diagonal averages "
           f"{np.trace(Mv)/4:.2f}, with CC persisting at {Mv[0,0]:.3f} and DD at "
           f"{Mv[3,3]:.3f}, so both mutual states are sticky while the two asymmetric "
           "states are the least stable. The observed outcome distribution is already "
           f"within {tvd:.3f} total variation of the chain's stationary limit and "
           f"round 1 alone is within {tvd1:.3f}, so the process starts essentially at "
           "its own fixed point. That agreement is the Markov restatement of the flat "
           "trajectory in G05 rather than independent evidence, since kernel and "
           "marginal are estimated from the same rounds.",
           "conditional-response bars + transition heatmap + stationary comparison",
           "joint outcome_t -> outcome_t+1, stationary distribution")

    # ----------------------------------------------------------------------
    # G08  strategy performance, and why the naive ranking is an artefact
    # ----------------------------------------------------------------------
    opp = ag[["game_uid", "agent", "strategy", "coop_rate"]].copy()
    opp["agent"] = 3 - opp["agent"]
    opp = opp.rename(columns={"strategy": "opp_strategy", "coop_rate": "opp_cr"})
    agx = ag.merge(opp, on=["game_uid", "agent"], how="left")
    bins = ["0-0.2", "0.2-0.5", "0.5-0.8", "0.8-1.0"]
    agx["opp_bin"] = pd.cut(agx["opp_cr"], [-.01, .2, .5, .8, 1.01], labels=bins)
    wts = agx["opp_bin"].value_counts(normalize=True)

    perf_rows = []
    for s in STRAT_PLOT:
        q = agx[agx.strategy == s]
        if len(q) < 5:
            continue
        m_, lo_2, hi_2 = _boot(q["utility"].to_numpy(), q["game_uid"].to_numpy(), 1500)
        cell = q.groupby("opp_bin", observed=True)["utility"].mean()
        cn = q.groupby("opp_bin", observed=True)["utility"].size()
        wsel = wts.reindex(cell.index)
        adj = float((cell * wsel).sum() / wsel.sum()) if wsel.sum() > 0 else np.nan
        row = {"strategy": s, "n": len(q), "share": len(q) / len(agx),
               "mean_utility": m_, "lo": lo_2, "hi": hi_2,
               "mean_efficiency": float(q["efficiency"].mean()),
               "mean_coop_rate": float(q["coop_rate"].mean()),
               "mean_opp_coop_rate": float(q["opp_cr"].mean()),
               "utility_std_by_opp": adj, "n_strata_covered": int((cn > 0).sum())}
        for b in bins:
            row[f"u_opp_{b}"] = float(cell[b]) if b in cell.index else np.nan
            row[f"n_opp_{b}"] = int(cn[b]) if b in cn.index else 0
        perf_rows.append(row)
    perf = pd.DataFrame(perf_rows)
    C.savetab(perf, "egt_strategy_payoff")
    pidx = perf.set_index("strategy")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    ax = axes[0]
    p = perf.sort_values("mean_utility")
    ax.barh(range(len(p)), p.mean_utility,
            color=[STRAT_PLOT_COL[s] for s in p.strategy])
    ax.errorbar(p.mean_utility, range(len(p)),
                xerr=[p.mean_utility - p.lo, p.hi - p.mean_utility], fmt="none",
                ecolor=C.INK2, lw=1.0)
    ax.set_yticks(range(len(p)), p.strategy, fontsize=7)
    ax.set_xlabel("mean utility (T = 1, R = 0.8, P = 0.4, S = 0)")
    ax.set_title("a  naive ranking by label")
    ax.grid(axis="y", visible=False)
    for i, (u, n) in enumerate(zip(p.mean_utility, p.n)):
        ax.text(u + .015, i, f"n={n:,}", va="center", fontsize=6)
    ax.set_xlim(0, 1.02)
    C.annotate(ax, "labels are identified by what the opponent\ndid, so this ranking "
                   "is confounded", loc="lower right")

    ax = axes[1]
    xb = np.arange(len(bins))
    for s in ["AllD", "AllC", "TFT", "ambiguous", "unclassified"]:
        if s not in pidx.index:
            continue
        r_ = pidx.loc[s]
        ax.plot(xb, [r_[f"u_opp_{b}"] for b in bins], "o-", color=STRAT_PLOT_COL[s],
                ms=4, label=s)
    ax.set_xticks(xb, bins, fontsize=7)
    ax.set_xlabel("opponent's realised cooperation rate")
    ax.set_ylabel("mean utility")
    ax.set_ylim(0, 1.15)
    ax.set_title("b  the ranking reverses on conditioning")
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    C.annotate(ax, "within every stratum AllD earns most,\nas dominance requires",
               loc="lower right")

    ax = axes[2]
    ax.scatter(perf.share, perf.mean_utility, s=88,
               color=[STRAT_PLOT_COL[s] for s in perf.strategy], zorder=3,
               label="naive mean")
    ax.scatter(perf.share, perf.utility_std_by_opp, s=88, marker="D",
               facecolor="none", zorder=4, linewidth=1.4,
               edgecolor=[STRAT_PLOT_COL[s] for s in perf.strategy],
               label="standardised on opponent")
    for _, r_ in perf.iterrows():
        ax.plot([r_.share, r_.share], [r_.mean_utility, r_.utility_std_by_opp],
                color=C.MUTED, lw=.8, zorder=1)
        ax.annotate(r_.strategy, (r_.share, max(r_.mean_utility,
                                                r_.utility_std_by_opp)),
                    fontsize=6, xytext=(0, 7), textcoords="offset points",
                    ha="center", color=C.INK2)
    ax.set_xscale("log")
    ax.set_xlabel("share of agent-games (log)")
    ax.set_ylabel("mean utility")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(perf.share.min() * .35, perf.share.max() * 3.2)
    ax.set_title("c  frequency against payoff")
    ax.legend(fontsize=6.5, loc="lower left")
    C.annotate(ax, "self-play: every opponent is drawn from the\nsame policy pool, so "
                   "this is not a tournament", loc="upper right")
    C.save(fig, "egt", "G08_strategy_payoff",
           "Which strategy labels earn the most, and does the ordering match "
           "evolutionary intuition?",
           "The naive ordering is an identifiability artefact and it reverses on "
           f"conditioning. Pooled, 'ambiguous' earns {pidx.loc['ambiguous','mean_utility']:.3f} "
           f"and unique-AllC earns {pidx.loc['AllC','mean_utility']:.3f}, but a label "
           "is only resolved by what the opponent did: AllC becomes distinguishable "
           "from TFT precisely when the opponent defected, which is also when "
           "cooperating pays worst. Stratifying on the opponent's realised cooperation "
           "rate puts AllD first in all four strata, which is what dominance in the "
           "penalty matrix requires, and direct standardisation on the common opponent "
           f"distribution moves ambiguous to {pidx.loc['ambiguous','utility_std_by_opp']:.3f} "
           f"and AllC to {pidx.loc['AllC','utility_std_by_opp']:.3f}. All of this is a "
           "within-population comparison under self-play, not a tournament result.",
           "ranked bars with CI + stratified lines + frequency-payoff scatter",
           "utility x strategy label x opponent cooperation rate")

    # ----------------------------------------------------------------------
    # G09  does lambda move the strategy mix?
    # ----------------------------------------------------------------------
    lam_rows = []
    for s in STRAT_PLOT:
        ind = ag.assign(_y=(ag.strategy == s).astype(float))
        b2_, lo2_, hi2_, p2_ = _clustered_slope(ind, "_y")
        v = (comp[(comp.scope == "scale") & (comp.strategy == s)]
             .set_index("level").reindex(C.SCALES)["share"])
        lam_rows.append({"strategy": s, "slope_per_decade": b2_, "lo": lo2_, "hi": hi2_,
                         "p": p2_, "share_min_lambda": float(v.iloc[0]),
                         "share_max_lambda": float(v.iloc[-1]),
                         "range_across_ladder": float(v.max() - v.min()),
                         "argmax_lambda": float(v.idxmax()),
                         "argmin_lambda": float(v.idxmin())})
    lam = pd.DataFrame(lam_rows)
    lam["p_fdr"] = C.bh_fdr(lam["p"].to_numpy())
    C.savetab(lam, "egt_lambda_strategy_mix")
    for _, r_ in lam.iterrows():
        C.record_test(section="game_theory",
                      test="linear probability slope on log10(lambda), cluster-robust "
                           "on design cell",
                      comparison=f"share of '{r_.strategy}' ~ log10(lambda)",
                      statistic=r_.slope_per_decade, p_value=r_.p, ci_lo=r_.lo,
                      ci_hi=r_.hi, n=len(ag),
                      effect_size_note="change in share per decade of lambda")

    ct = pd.crosstab(ag["scale"], ag["strategy"])
    chi2 = stats.chi2_contingency(ct)
    cramv = float(np.sqrt(chi2.statistic
                          / (ct.to_numpy().sum() * (min(ct.shape) - 1))))
    l2 = lam.set_index("strategy").reindex(STRAT_PLOT)
    n_sig_lam = int((l2.p_fdr < .05).sum())

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    ax = axes[0]
    for s in STRAT_PLOT:
        v = (comp[(comp.scope == "scale") & (comp.strategy == s)]
             .set_index("level").reindex(C.SCALES)["share"])
        ax.plot(C.SCALES, v.to_numpy(), "o-", color=STRAT_PLOT_COL[s], ms=3.2, label=s)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("payoff scale $\\lambda$")
    ax.set_ylabel("share of agent-games (log)")
    ax.set_ylim(1.2e-4, 1.4)
    ax.set_title("a  strategy shares across $\\lambda$")
    ax.legend(fontsize=6, ncol=4, loc="lower center")

    ax = axes[1]
    y = np.arange(len(l2))
    ax.axvline(0, color=C.INK, lw=.9, ls="--")
    ax.errorbar(l2.slope_per_decade, y,
                xerr=[l2.slope_per_decade - l2.lo, l2.hi - l2.slope_per_decade],
                fmt="none", ecolor=C.INK2, lw=1.1)
    ax.scatter(l2.slope_per_decade, y, s=48, zorder=4,
               color=[STRAT_PLOT_COL[s] for s in l2.index])
    ax.set_yticks(y, l2.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("change in share per decade of $\\lambda$")
    ax.set_title("b  slopes with 95% CI")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(l2.lo.min() - .006, l2.hi.max() + .009)
    for i, (_, r_) in enumerate(l2.iterrows()):
        if r_.p_fdr < .05:
            ax.text(r_.hi + .0012, i, "*", fontsize=10, va="center", color=C.INK2)
    C.annotate(ax, f"{n_sig_lam} of 7 labels move with $\\lambda$ (FDR);\nlargest "
                   f"|slope| {l2.slope_per_decade.abs().max():.4f} per decade",
               loc="lower left")

    ax = axes[2]
    jc = dy.groupby("scale", observed=True)["joint_coop"].mean().reindex(C.SCALES)
    alld = (comp[(comp.scope == "scale") & (comp.strategy == "AllD")]
            .set_index("level").reindex(C.SCALES)["share"])
    sc = ax.scatter(alld.to_numpy(), jc.to_numpy(), s=58, c=np.log10(C.SCALES),
                    cmap=C.SEQ, zorder=3, edgecolor=C.INK, linewidth=.4)
    for s in C.SCALES:
        ax.annotate(f"{s:g}", (alld[s], jc[s]), fontsize=6, xytext=(5, 3),
                    textcoords="offset points", color=C.INK2)
    rho = stats.spearmanr(alld.to_numpy(), jc.to_numpy())
    xs2 = np.linspace(alld.min(), alld.max(), 20)
    ax.plot(xs2, np.polyval(np.polyfit(alld.to_numpy(), jc.to_numpy(), 1), xs2),
            color=C.MUTED, lw=1.0, ls="--", zorder=1)
    plt.colorbar(sc, ax=ax, fraction=.046, label="$\\log_{10}\\lambda$")
    ax.margins(x=.13, y=.16)
    ax.set_xlabel("share of agent-games labelled AllD")
    ax.set_ylabel("dyad cooperation rate")
    ax.set_title("c  the $\\lambda$ effect runs through AllD")
    C.annotate(ax, f"across the 10 scales, Spearman $\\rho$ = {rho.statistic:+.2f}\n"
                   f"(p = {rho.pvalue:.1g}, n = 10 aggregate points)",
               loc="upper right")
    C.save(fig, "egt", "G09_lambda_strategy_mix",
           "Section 04 found that cooperation depends on a payoff multiplier that "
           "cannot matter in theory. Does that show up as a shift in which strategies "
           "are played?",
           f"Yes, and it is carried mainly by the unconditional-defection class. The "
           f"AllD share falls from {l2.loc['AllD','share_min_lambda']:.3f} at "
           f"lambda = 0.01 to about 0.07 in mid-ladder before rising again, a range of "
           f"{l2.loc['AllD','range_across_ladder']:.3f}, and across the ten scales the "
           f"AllD share tracks dyad cooperation at Spearman rho = {rho.statistic:+.2f}. "
           f"The overall dependence is nonetheless weak: Cramer's V for the full "
           f"scale-by-label table is {cramv:.3f}, and {n_sig_lam} of 7 label shares "
           f"have an FDR-significant linear slope, the largest of them "
           f"{l2.slope_per_decade.abs().max():.4f} per decade. The mix moves at the "
           "small-lambda end rather than trending across the ladder.",
           "share lines + slope forest + mechanism scatter",
           "strategy label share x scale, joint_coop")

    # ----------------------------------------------------------------------
    # G10  joint-state occupancy over the ten rounds, per model
    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.2), sharex=True, sharey=True)
    occ_rows = []
    for axx, m in zip(axes.ravel(), C.MODEL_ORDER):
        sub = a1r[a1r.model == m]
        piv = (pd.crosstab(sub["round"], sub["outcome"], normalize="index")
               .reindex(columns=C.OUTCOME_ORDER).fillna(0))
        bot = np.zeros(10)
        for o in C.OUTCOME_ORDER:
            v = piv[o].to_numpy()
            axx.fill_between(range(1, 11), bot, bot + v, color=C.OUTCOME_COL[o],
                             alpha=.92, lw=.4, edgecolor=C.SURFACE, label=o)
            bot += v
            for t in range(10):
                occ_rows.append({"model": m, "round": t + 1, "outcome": o,
                                 "share": float(v[t])})
        axx.set_xlim(1, 10)
        axx.set_ylim(0, 1)
        axx.set_xticks(range(1, 11))
        axx.set_title(_short(m), fontsize=8.5)
        axx.grid(False)
        d_cc = piv["CC"].iloc[-1] - piv["CC"].iloc[0]
        axx.text(.98, .04, f"CC {piv['CC'].iloc[0]:.2f} -> {piv['CC'].iloc[-1]:.2f} "
                           f"({d_cc:+.2f})", transform=axx.transAxes, ha="right",
                 va="bottom", fontsize=6.8, color="white")
    for axx in axes[1]:
        axx.set_xlabel("round")
    for axx in axes[:, 0]:
        axx.set_ylabel("share of dyads")
    h, lbl = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, [C.OUTCOME_LABEL[o] for o in lbl], loc="lower center", ncol=4,
               fontsize=7.5, bbox_to_anchor=(.5, -.035))
    occ = pd.DataFrame(occ_rows)
    C.savetab(occ, "egt_state_occupancy")
    cc_drift = (occ[occ.outcome == "CC"].pivot(index="model", columns="round",
                                               values="share"))
    cc_drift = (cc_drift[10] - cc_drift[1]).reindex(C.MODEL_ORDER)
    C.save(fig, "egt", "G10_state_occupancy",
           "How does the distribution over the four joint outcomes evolve across the "
           "ten rounds, and do the models converge on the same regime?",
           f"They diverge rather than converge. Between round 1 and round 10 the "
           f"mutual-cooperation share moves by {cc_drift.max():+.2f} in "
           f"{_short(cc_drift.idxmax())} and {cc_drift.min():+.2f} in "
           f"{_short(cc_drift.idxmin())}, and Qwen3 spends the whole game in a regime "
           "dominated by the asymmetric exploitation states. The pooled trajectory is "
           "flat only because these model-level drifts cancel, which is an aggregation "
           "artefact and the reason every claim in this section is made per model as "
           "well as pooled.",
           "stacked area small multiples", "joint outcome share x round x model")

    # ----------------------------------------------------------------------
    # findings
    # ----------------------------------------------------------------------
    e_all = end.set_index("scope")
    C.record_finding(
        id="G-no-endgame-unravelling",
        finding="Disclosing a ten-round horizon produces no backward-induction "
                "unravelling: pooled cooperation in the final round is "
                "indistinguishable from the trend of the preceding rounds.",
        evidence=f"A cluster-robust regression of cooperation on round plus a round-10 "
                 f"dummy over rounds 2 to 10 gives a pooled excess of "
                 f"{e_all.loc['all','round10_excess']:+.4f} "
                 f"[{e_all.loc['all','lo']:+.4f}, {e_all.loc['all','hi']:+.4f}], "
                 f"p = {e_all.loc['all','p']:.2f}, on 12,000 games. The pooled "
                 f"trajectory runs {pooled_t.iloc[0]:.3f} in round 1 to "
                 f"{pooled_t.iloc[-1]:.3f} in round 10. {len(sig)} models have an "
                 f"FDR-significant round-10 deviation and they disagree in sign "
                 f"({sig_txt}).",
        figure="07_game_theory/G05_temporal_endgame.png, G10_state_occupancy.png",
        strength="strong",
        robustness="holds in every persona pairing; the per-model deviations are all "
                   "under 0.07 in absolute value and cancel in the pooled estimate, "
                   "so the pooled null "
                   "is an average of opposing effects rather than a uniform absence",
        interpretation="Correlational. The models do not implement the backward "
                       "induction that the disclosed horizon licenses, but this corpus "
                       "cannot say whether they would behave differently under an "
                       "undisclosed horizon, because every game here discloses it.",
        caveat="No unknown-horizon arm exists in this corpus, so the horizon effect "
               "itself is not identified; only the absence of end-of-game unravelling "
               "under disclosure is measured. Ten rounds is short, and self-play means "
               "both sides fail to unravel together.",
        claim="Frontier LLMs told the game lasts exactly ten rounds do not defect more "
              "in the final round than the preceding trend predicts, so they do not "
              "perform the backward induction that finite-horizon equilibrium analysis "
              "requires.")

    C.record_finding(
        id="G-persistence-not-reciprocity",
        finding="What looks like conditional cooperation is mostly self-persistence: "
                "every model responds more to its own previous move than to the "
                "opponent's, and two models are net anti-reciprocal.",
        evidence="Decomposing the memory-one fingerprint into a partial reciprocity "
                 "term 0.5[(pC_R - pC_S) + (pC_T - pC_P)] and a partial persistence "
                 "term 0.5[(pC_R - pC_T) + (pC_S - pC_P)] gives persistence from "
                 f"{rec.persist_partial.min():.2f} to {rec.persist_partial.max():.2f} "
                 f"against reciprocity from {rec.recip_partial.min():.2f} to "
                 f"{rec.recip_partial.max():.2f}; all six models lie above the "
                 f"equal-weight diagonal. {n_neg} of 6 models have a negative raw "
                 "reciprocity contrast. Gemini-3.1 cooperates 0.86 of the time after "
                 "being exploited but only 0.07 after exploiting.",
        figure="07_game_theory/G04_reciprocity.png, G03_memory_one_fingerprint.png",
        strength="strong",
        robustness="the ordering persists in all five languages and all four persona "
                   "pairings (tables/egt_fingerprint.csv), and each conditional "
                   "probability carries a game-clustered bootstrap CI",
        interpretation="Correlational, and under self-play own and opponent histories "
                       "are correlated, which is exactly why the partial decomposition "
                       "is needed: it holds one arm fixed while varying the other. The "
                       "reading is that these agents largely restate their previous "
                       "commitment rather than run a reciprocal rule.",
        caveat="Self-play only, so the two conditioning arms are not independently "
               "manipulated; a designed-opponent experiment would identify this "
               "cleanly. The Arabic and Chinese prompts also carry an inherited "
               "translation inconsistency (a 'maximise your rewards' goal sentence "
               "over penalty payoffs), so language-level contrasts in the same table "
               "are partly prompt artefacts.",
        claim="Frontier LLMs in iterated prisoner's dilemma are action-persistent "
              "rather than reciprocal: they condition their next move mainly on their "
              "own previous move, and two of six cooperate more after being defected "
              "on than after being cooperated with.")

    C.record_finding(
        id="G-residual-fails-at-the-edges",
        finding="The 59% of agent-games that fit no canonical memory-one rule are not "
                "noise: they sit within a move or two of a rule and deviate "
                "specifically at the first and last rounds.",
        evidence=f"{(res.rule_distance <= 2).mean():.0%} of the residual is within "
                 f"Hamming distance 2 of a canonical rule (median "
                 f"{res.rule_distance.median():.0f} of 10 moves). The probability that "
                 f"a move departs from the nearest rule is {ov[0]:.2f} at round 1 and "
                 f"{ov[9]:.2f} at round 10 against a mid-game minimum of "
                 f"{ov[4:8].min():.2f}. The nameable horizon strategy 'AllC except the "
                 f"final one to three rounds' accounts for {n_horizon} agent-games, "
                 f"{n_horizon/len(res):.1%} of the residual and {n_horizon/len(ag):.1%} "
                 f"of the corpus, and only {frac_single:.0%} of residual sequences "
                 "contain a single switch at all.",
        figure="07_game_theory/G02_residual_taxonomy.png",
        strength="moderate",
        robustness="the U-shaped deviation profile appears separately for every "
                   "nearest-rule subgroup, including AllD-nearest games where a "
                   "round-1 spike cannot be a mechanical artefact of the rule's "
                   "opening move",
        interpretation="Descriptive. Memory-one rules are a good description of the "
                       "middle of an LLM game and a poor one of its opening and its "
                       "close, which suggests the models treat the first and last "
                       "moves as special rather than applying a stationary policy.",
        caveat="The rule vocabulary is only five rules; a richer hypothesis class "
               "(memory-two, or rules with an explicit round index) would reclassify "
               "much of this residual. The sequence taxonomy is descriptive rather "
               "than a fitted model.",
        claim="Rule-based strategy attribution in LLM prisoner's dilemma fails "
              "systematically at the opening and closing move rather than uniformly, "
              "so a stationary memory-one vocabulary is the wrong hypothesis class for "
              "the edges of the game.")

    C.record_finding(
        id="G-strategy-payoff-artefact",
        finding="Ranking strategy labels by mean payoff is an identifiability "
                "artefact, and the ranking reverses once the opponent's behaviour is "
                "held fixed.",
        evidence=f"Pooled, 'ambiguous' earns mean utility "
                 f"{pidx.loc['ambiguous','mean_utility']:.3f} and unique-AllC "
                 f"{pidx.loc['AllC','mean_utility']:.3f}. But a label is resolved only "
                 f"by a discriminating opponent move: {amb_explained} of the {n_amb} "
                 "ambiguous agent-games are exactly 'cooperated in all ten rounds "
                 "against an opponent who never defected in rounds 1 to 9'. "
                 "Stratifying on the opponent's realised cooperation rate puts AllD "
                 "first in all four strata, and direct standardisation on the common "
                 "opponent distribution moves ambiguous to "
                 f"{pidx.loc['ambiguous','utility_std_by_opp']:.3f} and AllC to "
                 f"{pidx.loc['AllC','utility_std_by_opp']:.3f}.",
        figure="07_game_theory/G08_strategy_payoff.png",
        strength="strong",
        robustness="the reversal holds in every one of the four opponent-cooperation "
                   "strata; TFT, WSLS and GRIM have fewer than 60 agent-games in some "
                   "strata and should not be ranked at all",
        interpretation="A causal reading is unavailable and would be wrong: the label "
                       "is a function of the realised history, so label and payoff "
                       "share a common cause in the opponent's play. This is a "
                       "textbook Simpson reversal and a warning about "
                       "strategy-attribution tables in the LLM game-theory literature.",
        caveat="Self-play: the opponent is drawn from the same policy pool, so none of "
               "these numbers are tournament payoffs. Standardisation adjusts for the "
               "opponent's cooperation rate only, not for its full policy.",
        claim="Payoff comparisons across inferred strategy labels are confounded by "
              "identifiability, because a label is resolved only when the opponent "
              "supplies a discriminating move; conditioning on the opponent reverses "
              "the apparent ranking and restores the dominance ordering.")

    C.record_finding(
        id="G-cooperation-is-the-attractor",
        finding="Mutual cooperation, not the stage-game equilibrium, is the dominant "
                "absorbing state, and the payoff-scale anomaly acts by switching which "
                "attractor a dyad falls into.",
        evidence=f"{lock.cc_absorbed.mean():.1%} of the 12,000 dyads reach mutual "
                 f"cooperation and never leave it, against "
                 f"{lock.dd_absorbed.mean():.1%} for mutual defection; {neither:.0%} "
                 f"end in neither. The empirical joint-state kernel has CC persisting "
                 f"at {Mv[0,0]:.3f} and DD at {Mv[3,3]:.3f}. DD lock-in is "
                 f"{ddl.loc[0.01,'absorbed_share']:.3f} at lambda = 0.01 against "
                 f"{ddl.loc[10.0,'absorbed_share']:.3f} at lambda = 10, and across the "
                 f"ten scales the AllD label share tracks dyad cooperation at Spearman "
                 f"rho = {rho.statistic:+.2f}.",
        figure="07_game_theory/G06_absorption_lockin.png, G09_lambda_strategy_mix.png, "
               "G07_transition_markov.png",
        strength="moderate",
        robustness="the CC-over-DD ordering holds in five of six models; Qwen3 is the "
                   "exception, with DD lock-in 0.231 against CC lock-in 0.074",
        interpretation="The lambda contrast is experimental: lambda is assigned by "
                       "design and matched across cells by common random numbers, so "
                       "the association between lambda and lock-in is a real "
                       "manipulation effect. The mediation through the AllD share, by "
                       "contrast, is an association over ten aggregated points and is "
                       "not a tested causal pathway.",
        caveat="Ten rounds gives absorption little time to be observed, so both "
               "absorption shares are lower bounds on what a longer game would show. "
               "The lambda-to-AllD-share link rests on 10 aggregate points.",
        claim="In self-play iterated prisoner's dilemma, frontier LLMs lock into "
              "mutual cooperation three times more often than into the "
              "mutual-defection equilibrium, and the violation of payoff-scale "
              "invariance operates by shifting dyads between those two attractors.")

    return {"composition": comp, "residual": tax, "fingerprint": fp, "endgame": end,
            "absorption": absorb, "performance": perf, "lambda_mix": lam,
            "reciprocity": rec, "transitions": Tn}
