"""Statistics for the payoff-scaling study.

Design note that governs every interval below: the two agents of a dyad play
each other, so their decisions are not independent, and the ten rounds inside an
agent-game are strongly autocorrelated. Every confidence interval therefore
resamples *whole dyads*, and every regression uses standard errors clustered on
the dyad. A round-level interval computed as if the 200,000 decisions were
independent would be far too narrow.

Because a positive rescaling is inert under the theory, the null here is a
theorem rather than a convention, and the honest test is a permutation test that
destroys the association between the payoff scale and behaviour while keeping
everything else - model, language, persona, dyad structure - exactly as
observed.

Writes tables T02..T09 to tables/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figstyle import MODEL_ORDER  # noqa: E402  the five models the main text reports

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TAB = HERE / "tables"
TAB.mkdir(exist_ok=True)

N_BOOT = 1500
N_PERM = 2000
SEED = 20260905
SUBUNIT = [0.01, 0.1]        # every payoff printed is at most 1 at these scales


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def cell_matrices(df, cell_keys, value="coop_rate"):
    """Per-dyad sums and counts of `value` within each cell of `cell_keys`.

    Reducing the bootstrap to two small (n_dyads x n_cells) matrices is what
    makes it affordable: a draw is then an integer index plus two column sums,
    instead of rebuilding a 20,000-row frame 1,500 times over.
    """
    u, uidx = pd.factorize(df["game_uid"], sort=False)
    key = df[cell_keys[0]].astype(str) if len(cell_keys) == 1 else \
        df[cell_keys].astype(str).agg("|".join, axis=1)
    c, cidx = pd.factorize(key, sort=True)
    S = np.zeros((len(uidx), len(cidx)))
    N = np.zeros_like(S)
    np.add.at(S, (u, c), df[value].to_numpy(dtype=float))
    np.add.at(N, (u, c), 1.0)
    return S, N, list(cidx)


def boot_matrix(S, N, n_boot=N_BOOT, seed=SEED):
    """(n_boot, n_cells) array of cell means under dyad resampling."""
    rng = np.random.default_rng(seed)
    nd = S.shape[0]
    out = np.empty((n_boot, S.shape[1]))
    for i in range(n_boot):
        t = rng.integers(0, nd, nd)
        s, n = S[t].sum(0), N[t].sum(0)
        out[i] = np.divide(s, n, out=np.full_like(s, np.nan), where=n > 0)
    return out


def ci(vals):
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _spread(m):
    return float(np.nanmax(m) - np.nanmin(m))


def _range_over_scales(d):
    m = d.groupby("scale_nominal")["coop_rate"].mean()
    return float(m.max() - m.min())


def unordered_pairing(s):
    """CvS and SvC are one dyad seen from its two sides, so they share a stratum."""
    return s.map({"CvC": "CC", "SvS": "SS", "CvS": "CS", "SvC": "CS"})


def permutation_p(d, observed, strata, n_perm=N_PERM, seed=SEED):
    """P(range >= observed) when the scale label is shuffled within strata.

    Shuffling *within* language and persona keeps every other feature of the
    design intact, so the only thing destroyed is the association between the
    payoff scale and behaviour, which is exactly the null the theorem asserts.

    The unit shuffled is the DYAD, not the agent-game. The payoff scale was
    assigned to a whole dyad, so under the null both agent-games of a dyad carry
    the same label, exactly as they do in the data. Shuffling agent-games
    independently would break that pairing and draw from a null narrower than
    the true one, making the p-value anti-conservative. It is the same
    dependence that makes every interval here resample whole dyads and every
    regression cluster on them.
    """
    rng = np.random.default_rng(seed)
    key = d[strata].astype(str).agg("|".join, axis=1).to_numpy()
    code = pd.factorize(d["scale_nominal"], sort=True)[0]
    ncell = int(code.max()) + 1
    v = d["coop_rate"].to_numpy(dtype=float)
    uid = d["game_uid"].to_numpy()

    # One label per dyad per stratum; scale is constant within a dyad, which
    # s00_build.py enforces, so taking it per dyad loses nothing.
    blocks = []
    for k in np.unique(key):
        idx = np.where(key == k)[0]
        _, inv = np.unique(uid[idx], return_inverse=True)
        lab = np.zeros(inv.max() + 1, dtype=int)
        lab[inv] = code[idx]
        blocks.append((idx, inv, lab))

    hits = 0
    for _ in range(n_perm):
        newcode = np.empty_like(code)
        for idx, inv, lab in blocks:
            newcode[idx] = rng.permutation(lab)[inv]
        denom = np.bincount(newcode, minlength=ncell)
        m = np.bincount(newcode, weights=v, minlength=ncell) / denom
        if float(m.max() - m.min()) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# --------------------------------------------------------------------------
# T02 / T03  cooperation by scale and model, and the per-model effect size
# --------------------------------------------------------------------------
def _scale_cells(d):
    """Cells are the ten payoff scales, ordered numerically rather than as text."""
    S, N, keys = cell_matrices(d, ["scale_nominal"])
    order = np.argsort([float(k) for k in keys])
    return S[:, order], N[:, order], [float(keys[i]) for i in order]


def t02_scale_by_model(g):
    rows = []
    for m, d in g.groupby("model"):
        S, N, keys = _scale_cells(d)
        B = boot_matrix(S, N)
        obs = d.groupby("scale_nominal")["coop_rate"].mean()
        for j, s in enumerate(keys):
            lo, hi = ci(B[:, j])
            rows.append({"model": m, "scale": s,
                         "n_dyads": d[d.scale_nominal == s].game_uid.nunique(),
                         "coop": float(obs.loc[s]), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def t03_effect_size(g):
    rows = []
    for m, d in g.groupby("model"):
        S, N, keys = _scale_cells(d)
        B = boot_matrix(S, N)
        lo, hi = ci(np.nanmax(B, axis=1) - np.nanmin(B, axis=1))
        obs = _range_over_scales(d)
        p = permutation_p(d.assign(pairing=unordered_pairing(d["dyad"])),
                          obs, ["language", "pairing"])
        mm = d.groupby("scale_nominal")["coop_rate"].mean()
        rows.append({"model": m, "range": obs, "lo": lo, "hi": hi, "p_perm": p,
                     "argmin_scale": mm.idxmin(), "argmax_scale": mm.idxmax(),
                     "coop_min": mm.min(), "coop_max": mm.max()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# T04  the pooling result: averaging across models cancels the effect
# --------------------------------------------------------------------------
def t04_pooling(g):
    per_model = g.groupby(["model", "scale_nominal"])["coop_rate"].mean().unstack()
    pooled = per_model.mean(axis=0)                 # equal weight per model
    within = per_model.max(axis=1) - per_model.min(axis=1)
    corr = per_model.T.corr()
    iu = np.triu_indices_from(corr.to_numpy(), k=1)
    pair = corr.to_numpy()[iu]
    tab = pd.DataFrame([{
        "pooled_range": float(pooled.max() - pooled.min()),
        "mean_within_model_range": float(within.mean()),
        "max_within_model_range": float(within.max()),
        "attenuation_factor": float(within.mean() / (pooled.max() - pooled.min())),
        "mean_pairwise_shape_corr": float(np.mean(pair)),
        "min_pairwise_shape_corr": float(np.min(pair)),
        "max_pairwise_shape_corr": float(np.max(pair)),
        "n_negative_pairs": int((pair < 0).sum()),
        "n_pairs": len(pair),
    }])
    return tab, corr


# --------------------------------------------------------------------------
# T05 / T06  language
# --------------------------------------------------------------------------
def t05_language(g):
    rows = []
    for (m, l), d in g.groupby(["model", "language"]):
        mm = d.groupby("scale_nominal")["coop_rate"].mean()
        rows.append({"model": m, "language": l, "coop": d.coop_rate.mean(),
                     "range_over_scales": float(mm.max() - mm.min()),
                     "argmax_scale": mm.idxmax(), "argmin_scale": mm.idxmin()})
    return pd.DataFrame(rows)


def t06_variance_decomposition(g):
    """Share of cell-mean variance carried by each factor and interaction.

    Computed on the 250 cell means (10 scales x 5 models x 5 languages), the
    granularity at which the design is exactly balanced.
    """
    cells = (g.groupby(["model", "language", "scale_nominal"])["coop_rate"]
             .mean().reset_index())
    mod = smf.ols("coop_rate ~ C(model) + C(language) + C(scale_nominal)"
                  " + C(model):C(language) + C(model):C(scale_nominal)"
                  " + C(language):C(scale_nominal)", data=cells).fit()
    aov = sm.stats.anova_lm(mod, typ=2).reset_index().rename(columns={"index": "term"})
    aov["pct_variance"] = 100 * aov["sum_sq"] / aov["sum_sq"].sum()
    return aov


# --------------------------------------------------------------------------
# T07  persona gating
# --------------------------------------------------------------------------
def t07_persona(g):
    rows = []
    for (m, s), d in g.groupby(["model", "scale_nominal"]):
        c = d[d.personality == "cooperative"].coop_rate.mean()
        sf = d[d.personality == "selfish"].coop_rate.mean()
        rows.append({"model": m, "scale": s, "coop_persona": c,
                     "selfish_persona": sf, "persona_effect": c - sf})
    per = pd.DataFrame(rows)

    # The contrast of interest is a difference of differences: how much larger
    # the persona effect is above the sub-unit boundary than below it.  Four
    # cells suffice, so it rides on the same fast bootstrap.
    out = []
    for m, d in per.groupby("model"):
        sub = d[d.scale.isin(SUBUNIT)].persona_effect.mean()
        sup = d[~d.scale.isin(SUBUNIT)].persona_effect.mean()

        raw = g[g.model == m].copy()
        raw["regime"] = np.where(raw.scale_nominal.isin(SUBUNIT), "sub", "sup")
        S, N, keys = cell_matrices(raw, ["regime", "personality"])
        idx = {k: i for i, k in enumerate(keys)}
        B = boot_matrix(S, N)
        sub_draws = B[:, idx["sub|cooperative"]] - B[:, idx["sub|selfish"]]
        sup_draws = B[:, idx["sup|cooperative"]] - B[:, idx["sup|selfish"]]
        sub_lo, sub_hi = ci(sub_draws)
        sup_lo, sup_hi = ci(sup_draws)
        contrast = sup_draws - sub_draws
        lo, hi = ci(contrast)

        # A sign flip is a claim about two means, so it needs a test on each of
        # them and not a comparison of two point estimates. We call it a flip
        # only when both regime intervals exclude zero and they fall on opposite
        # sides of it. Where one interval straddles zero the honest reading is
        # that the effect is abolished, not that it is reversed.
        flip = bool((sub_hi < 0 < sup_lo) or (sup_hi < 0 < sub_lo))
        out.append({"model": m, "persona_effect_subunit": sub,
                    "persona_effect_suprunit": sup, "shift": sup - sub,
                    "shift_lo": lo, "shift_hi": hi,
                    "sub_lo": sub_lo, "sub_hi": sub_hi,
                    "sup_lo": sup_lo, "sup_hi": sup_hi,
                    "sign_flip": flip})
    return per, pd.DataFrame(out)


# --------------------------------------------------------------------------
# T08  where in the game the effect enters
# --------------------------------------------------------------------------
def t08_first_move(r):
    rows = []
    for m, d in r.groupby("model"):
        first = d[d["round"] == 1].groupby("scale_nominal")["coop"].mean()
        later = d[d["round"] > 1].groupby("scale_nominal")["coop"].mean()
        r1 = float(first.max() - first.min())
        rl = float(later.max() - later.min())
        rows.append({"model": m, "range_round1": r1, "range_later": rl,
                     "ratio": r1 / rl if rl else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# T09  regression: is the scale factor jointly significant, per model?
# --------------------------------------------------------------------------
def t09_regression(r):
    order = [str(x) for x in [1.0, 0.01, 0.1, 0.25, 0.5, 2.0, 5.0, 10.0, 100.0, 1000.0]]
    rows = []
    for m, d in r.groupby("model"):
        d = d.copy()
        d["scale_f"] = pd.Categorical(d.scale_nominal.astype(str), categories=order)
        fit = smf.glm("coop ~ scale_f + C(language) + C(personality)"
                      " + C(opp_personality)", data=d,
                      family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d["game_uid"]})
        w = fit.wald_test_terms(skip_single=False, scalar=True).table
        terms = [t for t in fit.params.index if t.startswith("scale_f")]
        rows.append({
            "model": m, "n_obs": int(fit.nobs),
            "n_clusters": int(d.game_uid.nunique()),
            "wald_chi2_scale": float(w.loc["scale_f", "statistic"]),
            "df": int(w.loc["scale_f", "df_constraint"]),
            "p_scale": float(w.loc["scale_f", "pvalue"]),
            "max_abs_logodds_vs_lambda1": float(fit.params[terms].abs().max()),
        })
    return pd.DataFrame(rows)


def main():
    g = pd.read_parquet(DATA / "games.parquet")
    r = pd.read_parquet(DATA / "rounds.parquet")

    # The corpus holds six models; the main text reports five and the
    # supplement reports the sixth (see figstyle.MODEL_ORDER for which and
    # why). Per-model tables below are computed for ALL six, and s06_tables.py
    # selects the five it prints. The two POOLED quantities, the pooling
    # attenuation and the variance decomposition, are statements about a set of
    # models rather than about one, so they are computed on the five the main
    # text reports, and again on all six for the supplement.
    g_main = g[g.model.isin(MODEL_ORDER)]
    r_main = r[r.model.isin(MODEL_ORDER)] if "model" in r.columns else r
    print(f"corpus {g.model.nunique()} models; main text {g_main.model.nunique()}")

    print("T02 cooperation by scale x model")
    t02_scale_by_model(g).to_csv(TAB / "T02_scale_by_model.csv", index=False)

    print("T03 effect size + permutation test")
    t03 = t03_effect_size(g)
    t03.to_csv(TAB / "T03_effect_size.csv", index=False)
    print(t03.round(4).to_string(index=False))

    print("\nT04 pooling (main-text models)")
    t04, corr = t04_pooling(g_main)
    t04.to_csv(TAB / "T04_pooling.csv", index=False)
    corr.to_csv(TAB / "T04_shape_correlations.csv")
    print(t04.round(4).to_string(index=False))

    print("T04 pooling (all six, for the supplement)")
    t04a, corra = t04_pooling(g)
    t04a.to_csv(TAB / "T04_pooling_all.csv", index=False)
    corra.to_csv(TAB / "T04_shape_correlations_all.csv")
    print(t04a.round(4).to_string(index=False))

    print("\nT05 language")
    t05_language(g).to_csv(TAB / "T05_language.csv", index=False)

    print("T06 variance decomposition (main-text models)")
    t06 = t06_variance_decomposition(g_main)
    t06.to_csv(TAB / "T06_variance.csv", index=False)
    print(t06.round(3).to_string(index=False))

    print("T06 variance decomposition (all six, for the supplement)")
    t06_variance_decomposition(g).to_csv(TAB / "T06_variance_all.csv", index=False)

    print("\nT07 persona")
    per, t07 = t07_persona(g)
    per.to_csv(TAB / "T07_persona_by_scale.csv", index=False)
    t07.to_csv(TAB / "T07_persona_gating.csv", index=False)
    print(t07.round(4).to_string(index=False))

    print("\nT08 first move")
    t08 = t08_first_move(r)
    t08.to_csv(TAB / "T08_first_move.csv", index=False)
    print(t08.round(4).to_string(index=False))

    print("\nT09 regression")
    t09 = t09_regression(r)
    t09.to_csv(TAB / "T09_regression.csv", index=False)
    print(t09.round(4).to_string(index=False))
    print(f"\nwrote tables to {TAB}")


if __name__ == "__main__":
    main()
