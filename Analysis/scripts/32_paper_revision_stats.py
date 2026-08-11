"""Statistics the main text needs and the FR suite does not already write.

Four quantities were reported in the manuscript in a form that a re-analysis
would contradict, and all four are fixed here rather than in prose:

  1. **The variance decomposition was main-effects only.**  `T_FR17` enters
     four factors and calls everything it leaves over "within-cell", but a
     main-effects fit leaves every interaction among those same four factors in
     the residual.  The saturated cell model separates them, and the separation
     matters because the manuscript's own headline effects *are* interactions.

  2. **The payoff-scale effect was summarised by a slope.**  Cooperation is a
     step in $\\lambda$, not a line: almost the whole movement happens between
     $\\times 0.1$ and $\\times 1$ and nothing happens between $\\times 1$ and
     $\\times 10$.  A slope per decade fitted through that step reports a
     number no model produced, and for two models the step is not even
     monotone.  Two orthogonal contrasts say what the slope was meant to.

  3. **The persona effect was pooled over $\\lambda$.**  It is the single
     largest structured effect in the corpus and it changes sign: strongly
     inverted at $\\lambda = 0.1$, correctly signed at $\\lambda \\ge 1$ for two
     models.  Pooling averages the two into a number that describes neither.

  4. **GRIM was in the distance vocabulary but not the exact-match one.**
     Three of four models are nearest to GRIM, so a reader is owed the
     coverage the rule would have added had it been allowed to match.

Everything is bootstrapped by resampling whole dyads, matching
`\\S statistical analysis`: the two agent-rows of a game share a history, so a
row-level resample would understate every interval by roughly $\\sqrt 2$.

Writes `tables/T_PAPER_*.csv`; `31_fig_paper_main.py` reads them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pdlib.natstyle import DATADIR, FRONTIER, SCALE_ORDER, TABDIR

N_BOOT = 2000
SEED = 20260811
FACTORS = ["model", "scale_nominal", "language", "dyad"]
CELL = FACTORS + ["agent"]          # the ten replicates of one exact prompt


# --------------------------------------------------------------------------
# bootstrap over dyads
# --------------------------------------------------------------------------
def dyad_boot(df, stat, n_boot=N_BOOT, seed=SEED):
    """95% CI for `stat(frame)`, resampling whole dyads with replacement.

    `df` must carry `game_uid`; both agent-rows of a resampled dyad travel
    together, which is the whole point.
    """
    rng = np.random.default_rng(seed)
    uids = df.game_uid.unique()
    idx = df.groupby("game_uid").indices          # uid -> positional rows
    vals = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(uids, size=len(uids), replace=True)
        rows = np.concatenate([idx[u] for u in draw])
        vals[b] = stat(df.iloc[rows])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def persona_gap(d):
    """cooperative-persona minus selfish-persona cooperation."""
    c = d.loc[d.own == "C", "coop_rate"]
    s = d.loc[d.own == "S", "coop_rate"]
    if len(c) == 0 or len(s) == 0:
        return np.nan
    return float(c.mean() - s.mean())


def mean_gap(d, a, b, col="scale_nominal", val="coop_rate"):
    """mean(val | col == b) - mean(val | col == a)."""
    xa = d.loc[d[col] == a, val]
    xb = d.loc[d[col] == b, val]
    if len(xa) == 0 or len(xb) == 0:
        return np.nan
    return float(xb.mean() - xa.mean())


# --------------------------------------------------------------------------
# 1. variance: main effects, interactions, position, replicate noise
# --------------------------------------------------------------------------
def design_matrix(g, groups):
    X = np.ones((len(g), 1))
    for cols in groups:
        key = g[cols[0]].astype(str)
        for c in cols[1:]:
            key = key + "|" + g[c].astype(str)
        X = np.hstack([X, pd.get_dummies(key, drop_first=True,
                                         dtype=float).to_numpy()])
    return X


def r2_of(g, groups):
    """R-squared and the *rank* of the design.

    Stacking a coarse grouping and a finer one that refines it makes the
    columns collinear, which lstsq resolves but `shape[1]` does not: the
    adjusted R-squared needs the number of parameters actually fitted.
    """
    y = g.coop_rate.to_numpy()
    X = design_matrix(g, groups)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - float(((y - X @ beta) ** 2).sum()) / ss_tot, \
        int(np.linalg.matrix_rank(X))


def variance_full(g):
    y = g.coop_rate.to_numpy()
    n = len(y)
    v_tot = float(((y - y.mean()) ** 2).mean())

    # Type I main effects, in the order the manuscript declares
    shares, prev = [], 0.0
    for col in FACTORS:
        r2, _ = r2_of(g, [[c] for c in FACTORS[:FACTORS.index(col) + 1]])
        shares.append(r2 - prev)
        prev = r2
    r2_main = prev

    r2_cells, k_cells = r2_of(g, [FACTORS])          # 240 saturated cells
    r2_full, k_full = r2_of(g, [FACTORS, CELL])      # 480, position included
    within = 1 - r2_full

    # is the position split real, or 240 extra parameters fitting noise?
    rng = np.random.default_rng(SEED)
    null = np.empty(200)
    h = g.copy()
    for b in range(len(null)):
        h["agent"] = (h.groupby(FACTORS, sort=False)["agent"]
                       .transform(lambda s: rng.permutation(s.to_numpy())))
        null[b] = r2_of(h, [FACTORS, CELL])[0] - r2_cells
    pos = r2_full - r2_cells

    cell_mu = g.groupby(CELL, sort=False).coop_rate.transform("mean")
    floor = float((cell_mu * (1 - cell_mu) / int(g.n_rounds.iloc[0])).mean())

    rows = [{"component": f"{c} (main effect)", "share": s}
            for c, s in zip(["model", "payoff scale", "language",
                             "persona pairing"], shares)]
    rows += [
        {"component": "interactions among the four", "share": r2_cells - r2_main},
        {"component": "position within the dyad", "share": pos},
        {"component": "replicate (identical prompt)", "share": within},
    ]
    out = pd.DataFrame(rows)
    meta = pd.DataFrame([{
        "n": n, "total_variance": v_tot,
        "r2_main_effects": r2_main, "r2_cells_240": r2_cells,
        "r2_full_480": r2_full, "k_full": k_full,
        "r2_full_adjusted": 1 - (1 - r2_full) * (n - 1) / (n - k_full),
        "position_share": pos, "position_null_mean": float(null.mean()),
        "position_null_q95": float(np.quantile(null, 0.95)),
        "within_share": within,
        "binomial_floor_variance": floor,
        "binomial_floor_share_of_total": floor / v_tot,
        "binomial_floor_share_of_within": floor / (within * v_tot),
    }])
    return out, meta


# --------------------------------------------------------------------------
# 2. payoff scale as two contrasts rather than one slope
# --------------------------------------------------------------------------
def scale_contrasts(g):
    rows = []
    for mdl in FRONTIER:
        d = g[g.model == mdl]
        rec = {"model": mdl}
        for tag, (a, b) in (("step1", (0.1, 1.0)), ("step2", (1.0, 10.0))):
            est = mean_gap(d, a, b)
            lo, hi = dyad_boot(d[d.scale_nominal.isin([a, b])],
                               lambda x, a=a, b=b: mean_gap(x, a, b))
            rec |= {f"{tag}": est, f"{tag}_lo": lo, f"{tag}_hi": hi}
        for s in SCALE_ORDER:
            rec[f"coop_{s:g}"] = float(d.loc[d.scale_nominal == s,
                                             "coop_rate"].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. the persona effect, resolved by payoff scale
# --------------------------------------------------------------------------
def persona_by_scale(g):
    rows = []
    for mdl in list(FRONTIER) + ["pooled"]:
        d = g if mdl == "pooled" else g[g.model == mdl]
        for s in SCALE_ORDER:
            ds = d[d.scale_nominal == s]
            lo, hi = dyad_boot(ds, persona_gap)
            rows.append({"model": mdl, "scale": s, "effect": persona_gap(ds),
                         "lo": lo, "hi": hi,
                         "coop_persona": float(ds.loc[ds.own == "C",
                                                      "coop_rate"].mean()),
                         "selfish_persona": float(ds.loc[ds.own == "S",
                                                         "coop_rate"].mean())})
        lo, hi = dyad_boot(d, persona_gap)
        rows.append({"model": mdl, "scale": np.nan, "effect": persona_gap(d),
                     "lo": lo, "hi": hi,
                     "coop_persona": float(d.loc[d.own == "C",
                                                 "coop_rate"].mean()),
                     "selfish_persona": float(d.loc[d.own == "S",
                                                    "coop_rate"].mean())})
    return pd.DataFrame(rows)


def persona_gap_by_round(r):
    """Round-by-round persona gap, split by payoff scale.

    The manuscript said the opening gap is "eroded by the interaction" and
    closes to zero.  It does that only at $\\lambda \\ge 1$, where it does not
    close but crosses; at $\\lambda = 0.1$ it never closes at all.
    """
    r = r.copy()
    r["own"] = r.dyad.str[0]
    t = (r.pivot_table(index=["scale_nominal", "round"], columns="own",
                       values="coop")
          .rename(columns={"C": "coop_persona", "S": "selfish_persona"}))
    t["gap"] = t.coop_persona - t.selfish_persona
    return t.reset_index()


# --------------------------------------------------------------------------
# 4. what GRIM would have matched, had it been allowed to
# --------------------------------------------------------------------------
def grim_sensitivity(r):
    """Exact coverage of the four-rule vocabulary, with and without GRIM.

    GRIM is memory-one -- (1,0,0,0) opening C -- so it belongs to the 32-rule
    class of \\S residual already; the only question is what the *canonical*
    read-out drops by leaving it out of the four.
    """
    rows = []
    for (mdl, uid, ag), d in r.sort_values(["game_uid", "agent", "round"]) \
                              .groupby(["model", "game_uid", "agent"],
                                       sort=False):
        a = d.coop.to_numpy().astype(bool)
        o = d.opp_coop.to_numpy().astype(bool)
        n = len(a)
        tft = a[0] and all(a[i] == o[i - 1] for i in range(1, n))
        wsls = a[0] and all(a[i] == (a[i - 1] == o[i - 1]) for i in range(1, n))
        trig, grim = False, True
        for i in range(n):
            if a[i] != (not trig):
                grim = False
                break
            if not o[i]:
                trig = True
        rows.append((mdl, bool(a.all()), bool((~a).all()), tft, wsls, grim))
    df = pd.DataFrame(rows, columns=["model", "AllC", "AllD", "TFT", "WSLS",
                                     "GRIM"])
    four = df[["AllC", "AllD", "TFT", "WSLS"]].sum(axis=1)
    five = df[["AllC", "AllD", "TFT", "WSLS", "GRIM"]].sum(axis=1)
    out = pd.DataFrame({"model": df.model, "single4": four == 1,
                        "set4": four >= 1, "single5": five == 1,
                        "set5": five >= 1, "grim_alone": df.GRIM})
    agg = out.groupby("model").mean(numeric_only=True).reindex(FRONTIER)
    agg["set_delta"] = agg.set5 - agg.set4
    return agg.reset_index()


# --------------------------------------------------------------------------
def main():
    games = pd.read_parquet(DATADIR / "frontier_games.parquet")
    rounds = pd.read_parquet(DATADIR / "frontier_rounds.parquet")
    games = games.copy()
    games["own"] = games.dyad.str[0]

    var, meta = variance_full(games)
    var.to_csv(TABDIR / "T_PAPER_variance_full.csv", index=False)
    meta.to_csv(TABDIR / "T_PAPER_variance_meta.csv", index=False)

    scale_contrasts(games).to_csv(TABDIR / "T_PAPER_scale_contrasts.csv",
                                  index=False)
    persona_by_scale(games).to_csv(TABDIR / "T_PAPER_persona_by_scale.csv",
                                   index=False)
    persona_gap_by_round(rounds).to_csv(TABDIR / "T_PAPER_persona_by_round.csv",
                                        index=False)
    grim_sensitivity(rounds).to_csv(TABDIR / "T_PAPER_grim_sensitivity.csv",
                                    index=False)

    print(var.to_string(index=False))
    print()
    print(meta.T.to_string())


if __name__ == "__main__":
    main()
