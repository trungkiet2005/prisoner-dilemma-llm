"""Inference that treats the matched ten-scale ladder as the unit.

Every game is replayed at all ten payoff scales with the same language, the same
persona pairing, the same replicate index and the same requested sampling seeds.
A *ladder* is that set of ten dyads, identified by (model, language, game_id);
each model has 200 of them. The primary analyses in s03_stats.py resample or
cluster on the dyad, which treats the ten dyads of a ladder as independent. This
script repeats the headline inferences with the ladder as the resampling and
clustering unit, and adds two tests that are exact within-ladder comparisons:
a fixed-effects regression with one intercept per ladder, and a Friedman test.

No model endpoint is called. Writes
  tables/T24_ladder_inference.csv      per model: range, scale tests, persona shift
  tables/T25_ladder_conditioning.csv   per model: memory-one contrasts
  tables/T26_ladder_pooled.csv         five-model pooled scale test and leave-one-out
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MODEL_ORDER, MODEL_ORDER_ALL  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TAB = HERE / "tables"
PAPER_DIR = Path(os.environ.get("PD_PAPER_DIR", HERE.parents[1] / "papers" / "interface-focus"))
N_BOOT = 1500
SEED = 20260908
SUBUNIT = [0.01, 0.1]
SCALE_ORDER = [str(x) for x in [1.0, 0.01, 0.1, 0.25, 0.5, 2.0, 5.0, 10.0, 100.0, 1000.0]]


def add_ladder(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ladder"] = df.model + "|" + df.language + "|" + df.game_id
    return df


def unit_matrices(df, cell_keys, unit="ladder", value="coop_rate"):
    """Per-unit sums and counts of `value` in each cell, as in s03_stats."""
    u, uidx = pd.factorize(df[unit], sort=False)
    key = df[cell_keys[0]].astype(str) if len(cell_keys) == 1 else \
        df[cell_keys].astype(str).agg("|".join, axis=1)
    c, cidx = pd.factorize(key, sort=True)
    S = np.zeros((len(uidx), len(cidx)))
    N = np.zeros_like(S)
    np.add.at(S, (u, c), df[value].to_numpy(dtype=float))
    np.add.at(N, (u, c), 1.0)
    return S, N, list(cidx)


def boot_means(S, N, rng, n_boot=N_BOOT):
    nu = S.shape[0]
    out = np.empty((n_boot, S.shape[1]))
    for i in range(n_boot):
        t = rng.integers(0, nu, nu)
        s, n = S[t].sum(0), N[t].sum(0)
        out[i] = np.divide(s, n, out=np.full_like(s, np.nan), where=n > 0)
    return out


def ci(v):
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def wald(fit, prefix):
    names = [n for n in fit.params.index if n.startswith(prefix)]
    R = np.zeros((len(names), len(fit.params)))
    for i, n in enumerate(names):
        R[i, list(fit.params.index).index(n)] = 1
    w = fit.wald_test(R, scalar=True)
    return float(w.statistic), len(names), float(w.pvalue)


def per_model(g, r, rng):
    rows = []
    for m in MODEL_ORDER_ALL:
        d = g[g.model == m]
        # range over the ten scales, with ladders resampled whole
        S, N, keys = unit_matrices(d, ["scale_nominal"])
        B = boot_means(S, N, rng)
        means = d.groupby("scale_nominal").coop_rate.mean()
        obs = float(means.max() - means.min())
        lo, hi = ci(np.nanmax(B, 1) - np.nanmin(B, 1))

        # dyad-level outcome: the two agent-games of a dyad are averaged first
        dy = d.groupby(["ladder", "scale_nominal", "game_uid"], as_index=False).coop_rate.mean()
        dy["scale_f"] = pd.Categorical(dy.scale_nominal.astype(str), categories=SCALE_ORDER)
        fe = smf.ols("coop_rate ~ scale_f + C(ladder)", data=dy).fit(
            cov_type="cluster", cov_kwds={"groups": dy.ladder})
        fe_chi2, fe_df, fe_p = wald(fe, "scale_f")

        wide = dy.pivot(index="ladder", columns="scale_nominal", values="coop_rate")
        fr = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])

        # the per-model logistic model of T09, clustered on the ladder
        rr = r[r.model == m].copy()
        rr["scale_f"] = pd.Categorical(rr.scale_nominal.astype(str), categories=SCALE_ORDER)
        lg = smf.glm("coop ~ scale_f + C(language) + C(personality) + C(opp_personality)",
                     data=rr, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": rr.ladder})
        lg_chi2, lg_df, lg_p = wald(lg, "scale_f")

        # persona shift (difference of differences), ladders resampled whole
        dd = d.assign(regime=np.where(d.scale_nominal.isin(SUBUNIT), "sub", "sup"))
        S, N, keys = unit_matrices(dd, ["regime", "personality"])
        ix = {k: i for i, k in enumerate(keys)}
        B = boot_means(S, N, rng)
        sub = B[:, ix["sub|cooperative"]] - B[:, ix["sub|selfish"]]
        sup = B[:, ix["sup|cooperative"]] - B[:, ix["sup|selfish"]]
        cm = dd.groupby(["regime", "personality"]).coop_rate.mean()
        sub_obs = cm["sub", "cooperative"] - cm["sub", "selfish"]
        sup_obs = cm["sup", "cooperative"] - cm["sup", "selfish"]
        sub_lo, sub_hi = ci(sub)
        sup_lo, sup_hi = ci(sup)
        sh_lo, sh_hi = ci(sup - sub)

        rows.append({
            "model": m, "n_ladders": d.ladder.nunique(),
            "range": obs, "range_lo": lo, "range_hi": hi,
            "fe_chi2": fe_chi2, "fe_df": fe_df, "fe_p": fe_p,
            "friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue),
            "logit_chi2": lg_chi2, "logit_df": lg_df, "logit_p": lg_p,
            "persona_sub": sub_obs, "sub_lo": sub_lo, "sub_hi": sub_hi,
            "persona_sup": sup_obs, "sup_lo": sup_lo, "sup_hi": sup_hi,
            "shift": sup_obs - sub_obs, "shift_lo": sh_lo, "shift_hi": sh_hi,
            "sign_flip": bool((sub_hi < 0 < sup_lo) or (sup_hi < 0 < sub_lo)),
        })
    return pd.DataFrame(rows)


def conditioning(r, rng):
    """Memory-one contrasts of s12_conditioning_ci.py with ladders resampled."""
    r = r.dropna(subset=["prev_action", "prev_opp_action"]).copy()
    r["ctx"] = r.prev_action + r.prev_opp_action
    rows = []
    for m in MODEL_ORDER_ALL:
        d = r[r.model == m]
        tab = d.pivot_table(index="ladder", columns="ctx", values="coop",
                            aggfunc=["sum", "count"], fill_value=0)
        sums = tab["sum"].reindex(columns=["CC", "CD", "DC", "DD"], fill_value=0).to_numpy(float)
        cnts = tab["count"].reindex(columns=["CC", "CD", "DC", "DD"], fill_value=0).to_numpy(float)

        def contrasts(s, n):
            p = s / n
            rec = (p[..., 0] + p[..., 2] - p[..., 1] - p[..., 3]) / 2
            per = (p[..., 0] + p[..., 1] - p[..., 2] - p[..., 3]) / 2
            return rec, per

        rec, per = contrasts(sums.sum(0), cnts.sum(0))
        nu = len(sums)
        idx = rng.integers(0, nu, (N_BOOT, nu))
        bs = sums[idx].sum(1)
        bn = cnts[idx].sum(1)
        brec, bper = contrasts(bs, bn)
        for name, est, draws in [("reciprocity", rec, brec), ("persistence", per, bper),
                                 ("persistence_minus_reciprocity", per - rec, bper - brec)]:
            lo, hi = ci(draws)
            rows.append({"model": m, "contrast": name, "estimate": float(est),
                         "lo": lo, "hi": hi, "n_ladders": nu, "n_boot": N_BOOT})
    return pd.DataFrame(rows)


def pooled(g):
    """The leave-one-model-out check of T23, clustered on the ladder."""
    g5 = g[g.model.isin(MODEL_ORDER)]
    rows = []
    for removed in [None] + MODEL_ORDER:
        d = g5 if removed is None else g5[g5.model != removed]
        means = d.groupby("scale_nominal").coop_rate.mean()
        fit = smf.ols("coop_rate ~ C(scale_nominal)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.ladder})
        chi2, df, p = wald(fit, "C(scale_nominal)")
        rows.append({"removed": removed or "none", "pooled_range": float(means.max() - means.min()),
                     "chi2": chi2, "df": df, "p": p, "n_ladders": d.ladder.nunique()})
    return pd.DataFrame(rows)


PRINT = {"Claude-Haiku-4.5": "Claude Haiku 4.5", "GPT-5.4-Nano": "GPT-5.4 Nano",
         "Gemini-3.5-Flash-Lite": "Gemini 3.5 Flash-Lite",
         "Qwen3-235B-A22B": "Qwen3 235B-A22B", "Grok-4.20-Non-Reasoning": "Grok 4.20",
         "Gemini-3.1-Flash-Lite-Preview": "Gemini 3.1 Flash-Lite"}


def latex_table(t24, t25):
    """The appendix table, in the house format of conditioning_ci_auto.tex."""
    def est(v, lo, hi):
        return f"${v:.3f}\\,[{lo:.3f},\\,{hi:.3f}]$"

    gap = t25[t25.contrast == "persistence_minus_reciprocity"].set_index("model")
    rows = []
    for _, x in t24.iterrows():
        c = gap.loc[x.model]
        rows.append(" & ".join([
            PRINT[x.model], est(x.range, x.range_lo, x.range_hi),
            f"{x.fe_chi2:.1f}", f"{x.friedman_chi2:.1f}", f"{x.logit_chi2:.1f}",
            est(x["shift"], x.shift_lo, x.shift_hi),
            est(c.estimate, c.lo, c.hi)]) + " \\\\")
    pmax = float(t24[["fe_p", "friedman_p", "logit_p"]].to_numpy().max())
    # round the bound up to one significant digit, written as a power of ten
    exp = int(np.floor(np.log10(pmax)))
    bound = f"{int(np.ceil(pmax / 10 ** exp))}\\times10^{{{exp}}}"
    caption = (
        "\\textbf{Headline inferences with the matched ladder as the unit.} A "
        "ladder is the ten dyads, one per payoff scale, that share a model, a "
        "language, a persona pairing, a replicate index and the requested "
        "sampling seeds; each model has 200. Intervals are 95\\% percentile "
        f"intervals from {N_BOOT:,} whole-ladder bootstrap draws. The three "
        "$\\chi^2_9$ statistics test the nine payoff-scale contrasts: a linear "
        "model of dyad cooperation with one fixed effect per ladder, a Friedman "
        "test on the 200 ladders by ten scales, and the round-level logistic "
        "model of table~\\ref{tab:effect} with standard errors clustered on the "
        f"ladder. Every one of the {3 * len(t24)} tests has $p < {bound}$. "
        "Persona shift is the persona effect above the sub-unit boundary "
        "minus that below it; the last column is the persistence contrast "
        "minus the reciprocity contrast.")
    return "\n".join([
        "% Generated by Analysis/scaling/s14_matched_ladder_inference.py - do not edit.",
        # a manuscript may colour the table (e.g. to mark a revision) by
        # defining \revisioncolor; elsewhere it is a no-op
        "\\begin{table}[htbp]", "\\providecommand{\\revisioncolor}{}\\revisioncolor",
        "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}", f"\\caption{{{caption}}}",
        "\\label{tab:D-ladder}", "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
        "Model & range & FE $\\chi^2_9$ & Friedman $\\chi^2_9$ & logit $\\chi^2_9$ "
        "& persona shift & persist.\\,$-$\\,recip. \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""])


def main():
    g = add_ladder(pd.read_parquet(DATA / "games.parquet"))
    r = add_ladder(pd.read_parquet(DATA / "rounds.parquet"))
    n = g.groupby("ladder").scale_nominal.nunique()
    if not (n == 10).all() or len(n) != 200 * len(MODEL_ORDER_ALL):
        raise ValueError("every ladder must hold all ten scales")
    rng = np.random.default_rng(SEED)
    t24 = per_model(g, r, rng)
    t25 = conditioning(r, rng)
    t26 = pooled(g)
    t24.to_csv(TAB / "T24_ladder_inference.csv", index=False)
    t25.to_csv(TAB / "T25_ladder_conditioning.csv", index=False)
    t26.to_csv(TAB / "T26_ladder_pooled.csv", index=False)
    (PAPER_DIR / "ladder_tables_auto.tex").write_text(latex_table(t24, t25), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(t24.round(4).to_string(index=False))
        print(t25.round(4).to_string(index=False))
        print(t26.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
