"""Check every number typed into the manuscript prose against the data.

The tables are machine-generated, but the Results and Discussion prose quotes
figures by hand, and a wrong one there is the failure mode this project has
actually suffered. Each check below names the claim, recomputes it from the
parquet or the tables, and compares against what the paper says.

Scope. The corpus holds six models. `paper_scaling/main.tex` reports five of
them and `paper_scaling/supplementary.tex` reports all six, so the ledger
covers both documents: a check whose value is printed only in the electronic
supplementary material is named with a leading "ESM ", and a check whose value
reaches the reader through a generated table rather than through prose is named
with a leading "T1 ", "T2 " or "T3 ". Everything else is main-text prose.

Run after any change to the corpus or the pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TAB = HERE / "tables"
SUBUNIT = [0.01, 0.1]

# The five models the main text reports.  The sixth, Gemini 3.1 Flash-Lite, is
# reported in the electronic supplementary material; see figstyle.MODEL_ORDER.
MAIN = ["Claude-Haiku-4.5", "GPT-5.4-Nano", "Gemini-3.5-Flash-Lite",
        "Qwen3-235B-A22B", "Grok-4.20-Non-Reasoning"]
SUPP = "Gemini-3.1-Flash-Lite-Preview"

g = pd.read_parquet(DATA / "games.parquet")
r = pd.read_parquet(DATA / "rounds.parquet")
d = pd.read_parquet(DATA / "readout.parquet")

checks: list[tuple[str, float, float, float]] = []


def chk(name, claimed, actual, tol=0.0006):
    checks.append((name, claimed, float(actual), tol))


# ---- design -----------------------------------------------------------------
chk("dyads", 12000, g.game_uid.nunique(), 0)
chk("agent-games", 24000, len(g), 0)
chk("decisions", 240000, len(r), 0)
chk("cells", 300, g.groupby(["model", "language", "scale_nominal"]).ngroups, 0)
chk("overall cooperation", 0.5895, g.coop_rate.mean(), 0.0006)
chk("main-text decisions", 200000, len(r[r.model.isin(MAIN)]), 0)
chk("main-text dyads", 10000, g[g.model.isin(MAIN)].game_uid.nunique(), 0)
chk("main-text agent-games", 20000, len(g[g.model.isin(MAIN)]), 0)
chk("main-text cells", 250,
    g[g.model.isin(MAIN)].groupby(
        ["model", "language", "scale_nominal"]).ngroups, 0)

# ---- language levels and sensitivities quoted in 3.2 -------------------------
lang = g.pivot_table(index="model", columns="language", values="coop_rate")
chk("Claude cn", 0.574, lang.loc["Claude-Haiku-4.5", "cn"])
chk("Claude vn", 0.234, lang.loc["Claude-Haiku-4.5", "vn"])
chk("GPT en", 0.798, lang.loc["GPT-5.4-Nano", "en"])
chk("GPT ar", 0.458, lang.loc["GPT-5.4-Nano", "ar"])

rng = (g.groupby(["model", "language", "scale_nominal"]).coop_rate.mean()
       .groupby(["model", "language"]).agg(lambda s: s.max() - s.min()))
rng_main = rng.loc[MAIN]
chk("Claude range en", 0.136, rng.loc[("Claude-Haiku-4.5", "en")])
chk("Claude range fr", 0.314, rng.loc[("Claude-Haiku-4.5", "fr")])
chk("Gemini3.5 range cn", 0.162, rng.loc[("Gemini-3.5-Flash-Lite", "cn")])
chk("Gemini3.5 range en", 0.412, rng.loc[("Gemini-3.5-Flash-Lite", "en")])
chk("min lang range, main five", 0.083, rng_main.min())
chk("max lang range", 0.855, rng_main.max())
chk("min lang range is Qwen fr", 0.083, rng.loc[("Qwen3-235B-A22B", "fr")])
chk("max lang range is Grok en", 0.855,
    rng.loc[("Grok-4.20-Non-Reasoning", "en")])
chk("ESM min lang range, all six", 0.080, rng.min())
chk("ESM min lang range is G3.1 ar", 0.080, rng.loc[(SUPP, "ar")])

# ---- Qwen3 persona inversion quoted in 3.3 -----------------------------------
q = g[g.model == "Qwen3-235B-A22B"].groupby("personality").coop_rate.mean()
chk("Qwen3 coop persona", 0.072, q.loc["cooperative"])
chk("Qwen3 selfish persona", 0.798, q.loc["selfish"])

# ---- strategy numbers quoted in 3.4 and the supplement -----------------------
prov = d.provenance.value_counts(normalize=True) * 100
chk("deduced overall %", 23.0, prov["deduced"], 0.06)
chk("ambiguous overall %", 17.5, prov["ambiguous"], 0.06)
chk("unmatched overall %", 59.6, prov["unmatched"], 0.06)
chk("ESM LSTM-dependent %", 77.0, 100 - prov["deduced"], 0.06)

t13 = pd.read_csv(TAB / "T13_pooled_strategy.csv").set_index("scale_nominal")
chk("deduced at 0.01", 38.6, t13.loc[0.01, "deduced"], 0.06)
chk("deduced at 1000", 17.3, t13.loc[1000.0, "deduced"], 0.06)
chk("unmatched at 0.01", 51.6, t13.loc[0.01, "unmatched"], 0.06)
chk("unmatched at 1000", 64.0, t13.loc[1000.0, "unmatched"], 0.06)
chk("AllC at 0.01", 38.1, t13.loc[0.01, "AllC"], 0.06)
chk("AllC at 1000", 45.8, t13.loc[1000.0, "AllC"], 0.06)
chk("TFT at 0.01", 9.1, t13.loc[0.01, "TFT"], 0.06)
chk("TFT at 1000", 12.7, t13.loc[1000.0, "TFT"], 0.06)
chk("AllD at 0.01", 43.6, t13.loc[0.01, "AllD"], 0.06)
chk("AllD at 1000", 28.3, t13.loc[1000.0, "AllD"], 0.06)

# provenance per label: the numbers that bound how the compositional claim in
# 3.4 and the supplement may be read
pl = pd.crosstab(d.label, d.provenance, normalize="index") * 100
chk("ESM AllC deduced %", 23.6, pl.loc["AllC", "deduced"], 0.06)
chk("ESM AllC ambiguous %", 36.9, pl.loc["AllC", "ambiguous"], 0.06)
chk("ESM AllD deduced %", 35.6, pl.loc["AllD", "deduced"], 0.06)
chk("TFT unmatched %", 94.6, pl.loc["TFT", "unmatched"], 0.06)
chk("WSLS unmatched %", 97.1, pl.loc["WSLS", "unmatched"], 0.06)

mix = pd.read_csv(TAB / "T12_label_mix.csv")


def m(model, label, scale):
    row = mix[(mix.model == model) & (mix.scale == scale)]
    return float(row[label].iloc[0])


chk("GPT AllC 0.01", 29.8, m("GPT-5.4-Nano", "AllC", 0.01), 0.06)
chk("GPT AllC 1000", 50.7, m("GPT-5.4-Nano", "AllC", 1000.0), 0.06)
chk("GPT AllD 0.01", 46.0, m("GPT-5.4-Nano", "AllD", 0.01), 0.06)
chk("GPT AllD 1000", 17.5, m("GPT-5.4-Nano", "AllD", 1000.0), 0.06)
chk("G3.5 AllC 0.01", 73.8, m("Gemini-3.5-Flash-Lite", "AllC", 0.01), 0.06)
chk("G3.5 AllC 1000", 51.5, m("Gemini-3.5-Flash-Lite", "AllC", 1000.0), 0.06)
chk("G3.5 TFT 0.01", 6.8, m("Gemini-3.5-Flash-Lite", "TFT", 0.01), 0.06)
chk("G3.5 TFT 1000", 14.2, m("Gemini-3.5-Flash-Lite", "TFT", 1000.0), 0.06)

t11 = pd.read_csv(TAB / "T11_rule_distance.csv").set_index("model")
chk("G3.5 dist min", 0.48, t11.loc["Gemini-3.5-Flash-Lite", "dist_at_min_scale"], 0.006)
chk("G3.5 dist max", 1.36, t11.loc["Gemini-3.5-Flash-Lite", "dist_at_max_scale"], 0.006)
chk("Qwen dist min", 0.27, t11.loc["Qwen3-235B-A22B", "dist_at_min_scale"], 0.006)
chk("Qwen dist max", 1.14, t11.loc["Qwen3-235B-A22B", "dist_at_max_scale"], 0.006)
chk("Grok dist min", 0.92, t11.loc["Grok-4.20-Non-Reasoning", "dist_at_min_scale"], 0.006)
chk("Grok dist max", 0.37, t11.loc["Grok-4.20-Non-Reasoning", "dist_at_max_scale"], 0.006)
chk("Grok dist beta", -0.118, t11.loc["Grok-4.20-Non-Reasoning", "beta_logscale"])
chk("ESM G3.1 dist beta", -0.019, t11.loc[SUPP, "beta_logscale"])
chk("ESM G3.1 dist p", 0.26, t11.loc[SUPP, "p"], 0.006)

# ---- first move quoted in 3.5 ------------------------------------------------
t08 = pd.read_csv(TAB / "T08_first_move.csv").set_index("model")
chk("ratio Claude", 2.0, t08.loc["Claude-Haiku-4.5", "ratio"], 0.05)
chk("ratio Gemini3.5", 1.9, t08.loc["Gemini-3.5-Flash-Lite", "ratio"], 0.05)
chk("ratio GPT", 1.3, t08.loc["GPT-5.4-Nano", "ratio"], 0.05)
chk("ratio Grok", 1.2, t08.loc["Grok-4.20-Non-Reasoning", "ratio"], 0.05)
chk("ESM ratio Gemini3.1", 3.68, t08.loc[SUPP, "ratio"], 0.05)

# ---- pooling quoted in 3.1, the Introduction and the Discussion --------------
t04 = pd.read_csv(TAB / "T04_pooling.csv").iloc[0]
chk("pooled range", 0.154, t04.pooled_range, 0.0006)
chk("mean within-model range", 0.249, t04.mean_within_model_range, 0.0006)
chk("attenuation", 1.62, t04.attenuation_factor, 0.006)
chk("mean shape corr", -0.10, t04.mean_pairwise_shape_corr, 0.006)
chk("min shape corr", -0.72, t04.min_pairwise_shape_corr, 0.006)
chk("max shape corr", 0.85, t04.max_pairwise_shape_corr, 0.006)
chk("negative pairs", 6, t04.n_negative_pairs, 0)

t04a = pd.read_csv(TAB / "T04_pooling_all.csv").iloc[0]
chk("ESM pooled range, all six", 0.125, t04a.pooled_range, 0.0006)
chk("ESM mean within range, all six", 0.219, t04a.mean_within_model_range, 0.0006)
chk("ESM attenuation, all six", 1.76, t04a.attenuation_factor, 0.006)
chk("ESM mean shape corr, all six", -0.08, t04a.mean_pairwise_shape_corr, 0.006)
chk("ESM min shape corr, all six", -0.72, t04a.min_pairwise_shape_corr, 0.006)
chk("ESM max shape corr, all six", 0.85, t04a.max_pairwise_shape_corr, 0.006)
chk("ESM negative pairs, all six", 9, t04a.n_negative_pairs, 0)
chk("ESM pairs, all six", 15, t04a.n_pairs, 0)

# ---- effect sizes quoted in the abstract and 3.1 -----------------------------
t03 = pd.read_csv(TAB / "T03_effect_size.csv").set_index("model")
chk("Claude range", 0.081, t03.loc["Claude-Haiku-4.5", "range"])
chk("Claude range lo", 0.062, t03.loc["Claude-Haiku-4.5", "lo"])
chk("Claude range hi", 0.135, t03.loc["Claude-Haiku-4.5", "hi"])
chk("Grok range", 0.636, t03.loc["Grok-4.20-Non-Reasoning", "range"])
chk("Grok range lo", 0.588, t03.loc["Grok-4.20-Non-Reasoning", "lo"])
chk("Grok range hi", 0.686, t03.loc["Grok-4.20-Non-Reasoning", "hi"])
chk("Grok coop min", 0.274, t03.loc["Grok-4.20-Non-Reasoning", "coop_min"])
chk("Grok coop max", 0.910, t03.loc["Grok-4.20-Non-Reasoning", "coop_max"])
chk("GPT range", 0.243, t03.loc["GPT-5.4-Nano", "range"])
chk("GPT range lo", 0.200, t03.loc["GPT-5.4-Nano", "lo"])
chk("GPT range hi", 0.290, t03.loc["GPT-5.4-Nano", "hi"])
chk("GPT coop min", 0.438, t03.loc["GPT-5.4-Nano", "coop_min"])
chk("G3.5 range", 0.190, t03.loc["Gemini-3.5-Flash-Lite", "range"])
chk("Claude p_perm", 0.0025, t03.loc["Claude-Haiku-4.5", "p_perm"], 0.00006)
chk("Grok p_perm", 0.0005, t03.loc["Grok-4.20-Non-Reasoning", "p_perm"], 0.00006)
chk("ESM G3.1 range", 0.072, t03.loc[SUPP, "range"])
chk("ESM G3.1 range lo", 0.060, t03.loc[SUPP, "lo"])
chk("ESM G3.1 range hi", 0.133, t03.loc[SUPP, "hi"])
chk("ESM G3.1 p_perm", 0.0005, t03.loc[SUPP, "p_perm"], 0.00006)

# ---- clustered Wald test quoted in 3.1 ---------------------------------------
t09 = pd.read_csv(TAB / "T09_regression.csv").set_index("model")
chk("Wald min, Claude", 32.3, t09.loc["Claude-Haiku-4.5", "wald_chi2_scale"], 0.06)
chk("Wald max, Grok", 1227.3, t09.loc["Grok-4.20-Non-Reasoning", "wald_chi2_scale"], 0.06)
chk("log-odds min, Claude", 0.36,
    t09.loc["Claude-Haiku-4.5", "max_abs_logodds_vs_lambda1"], 0.006)
chk("log-odds max, Grok", 3.74,
    t09.loc["Grok-4.20-Non-Reasoning", "max_abs_logodds_vs_lambda1"], 0.006)
chk("n decisions per model", 40000, t09.loc["Grok-4.20-Non-Reasoning", "n_obs"], 0)
chk("n clusters per model", 2000, t09.loc["Grok-4.20-Non-Reasoning", "n_clusters"], 0)
chk("ESM G3.1 Wald", 83.1, t09.loc[SUPP, "wald_chi2_scale"], 0.06)

# ---- variance decomposition, five models and all six -------------------------
t06 = pd.read_csv(TAB / "T06_variance.csv").set_index("term")
chk("model %", 45.4, t06.loc["C(model)", "pct_variance"], 0.06)
chk("model x language %", 17.0, t06.loc["C(model):C(language)", "pct_variance"], 0.06)
chk("scale main %", 5.8, t06.loc["C(scale_nominal)", "pct_variance"], 0.06)
chk("scale main F", 10.64, t06.loc["C(scale_nominal)", "F"], 0.006)
chk("model x scale %", 17.9, t06.loc["C(model):C(scale_nominal)", "pct_variance"], 0.06)
chk("language x scale %", 3.6, t06.loc["C(language):C(scale_nominal)", "pct_variance"], 0.06)
chk("language x scale p", 0.018, t06.loc["C(language):C(scale_nominal)", "PR(>F)"], 0.0006)
chk("residual df", 144, t06.loc["Residual", "df"], 0)

t06a = pd.read_csv(TAB / "T06_variance_all.csv").set_index("term")
chk("ESM model %, all six", 44.8, t06a.loc["C(model)", "pct_variance"], 0.06)
chk("ESM scale main %, all six", 4.3, t06a.loc["C(scale_nominal)", "pct_variance"], 0.06)
chk("ESM model x scale %, all six", 19.1,
    t06a.loc["C(model):C(scale_nominal)", "pct_variance"], 0.06)
chk("ESM model x language %, all six", 18.0,
    t06a.loc["C(model):C(language)", "pct_variance"], 0.06)
chk("ESM language x scale %, all six", 3.1,
    t06a.loc["C(language):C(scale_nominal)", "pct_variance"], 0.06)
chk("ESM residual df, all six", 180, t06a.loc["Residual", "df"], 0)

# The claim in 3.1, S4 and the Discussion that the significant main effect is
# carried by one model: refit the same decomposition without it and check that
# the payoff scale returns the null the earlier five-model panel reported.
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    cell = (g[g.model != "Grok-4.20-Non-Reasoning"]
            .groupby(["model", "language", "scale_nominal"], as_index=False)
            .coop_rate.mean())
    fit = smf.ols("coop_rate ~ C(model)*C(language) + C(model)*C(scale_nominal)"
                  " + C(language)*C(scale_nominal)", data=cell).fit()
    av = sm.stats.anova_lm(fit, typ=2)
    av["pct"] = av.sum_sq / av.sum_sq.sum() * 100
    chk("ESM no-Grok scale main F", 1.10, av.loc["C(scale_nominal)", "F"], 0.006)
    chk("ESM no-Grok scale main p", 0.36, av.loc["C(scale_nominal)", "PR(>F)"], 0.006)
    chk("ESM no-Grok scale main %", 0.7, av.loc["C(scale_nominal)", "pct"], 0.06)
except ImportError:                                    # pragma: no cover
    print("statsmodels missing: the no-Grok panel check was skipped")

# ---- persona -----------------------------------------------------------------
t07 = pd.read_csv(TAB / "T07_persona_gating.csv").set_index("model")
for mod, sub, sup, sh, lo, hi in [
        ("Gemini-3.5-Flash-Lite", -0.456, 0.224, 0.680, 0.625, 0.736),
        ("Claude-Haiku-4.5", -0.222, 0.207, 0.429, 0.389, 0.471),
        ("GPT-5.4-Nano", -0.135, 0.018, 0.153, 0.105, 0.200),
        ("Grok-4.20-Non-Reasoning", -0.441, -0.303, 0.138, 0.090, 0.185)]:
    chk(f"T2 {mod} persona sub", sub, t07.loc[mod, "persona_effect_subunit"])
    chk(f"T2 {mod} persona sup", sup, t07.loc[mod, "persona_effect_suprunit"])
    chk(f"{mod} persona shift", sh, t07.loc[mod, "shift"])
    chk(f"{mod} shift lo", lo, t07.loc[mod, "shift_lo"])
    chk(f"{mod} shift hi", hi, t07.loc[mod, "shift_hi"])
chk("T2 Qwen persona sub", -0.911, t07.loc["Qwen3-235B-A22B", "persona_effect_subunit"])
chk("T2 Qwen persona sup", -0.679, t07.loc["Qwen3-235B-A22B", "persona_effect_suprunit"])
chk("T2 Qwen shift", 0.233, t07.loc["Qwen3-235B-A22B", "shift"])
chk("ESM G3.1 persona sub", 0.577, t07.loc[SUPP, "persona_effect_subunit"])
chk("ESM G3.1 persona sup", 0.680, t07.loc[SUPP, "persona_effect_suprunit"])
chk("ESM G3.1 shift", 0.103, t07.loc[SUPP, "shift"])
chk("ESM G3.1 shift lo", 0.075, t07.loc[SUPP, "shift_lo"])
chk("ESM G3.1 shift hi", 0.131, t07.loc[SUPP, "shift_hi"])

# ---- classifier --------------------------------------------------------------
t01 = pd.read_csv(TAB / "T01_classifier.csv").set_index("split")
chk("clf held-out", 0.985, t01.loc["held-out (NoNoise + Noise005)", "accuracy"], 0.0006)
chk("clf OOD 10%", 0.936, t01.loc["OOD (Noise01)", "accuracy"], 0.0006)
chk("clf OOD 20%", 0.811, t01.loc["OOD (Noise02)", "accuracy"], 0.0006)

# ---- trend coefficients quoted in 3.4 ----------------------------------------
t10 = pd.read_csv(TAB / "T10_provenance_trend.csv").set_index("model")
chk("Claude beta", -0.545, t10.loc["Claude-Haiku-4.5", "beta_logscale"])
chk("Grok beta", -0.301, t10.loc["Grok-4.20-Non-Reasoning", "beta_logscale"])
chk("Qwen beta", -0.267, t10.loc["Qwen3-235B-A22B", "beta_logscale"])
chk("G3.5 beta", -0.242, t10.loc["Gemini-3.5-Flash-Lite", "beta_logscale"])
chk("G3.1 beta", 0.125, t10.loc[SUPP, "beta_logscale"])
chk("GPT beta p", 0.44, t10.loc["GPT-5.4-Nano", "p"], 0.006)
chk("ESM G3.1 deduced sub-unit", 17.8, t10.loc[SUPP, "deduced_subunit"], 0.06)
chk("ESM G3.1 deduced supra-unit", 26.3, t10.loc[SUPP, "deduced_suprunit"], 0.06)


def main():
    bad = []
    for name, claimed, actual, tol in checks:
        if abs(claimed - actual) > tol:
            bad.append((name, claimed, actual, tol))
    print(f"{len(checks)} numbers checked against the data")
    if bad:
        print(f"\n{len(bad)} MISMATCH:")
        for name, c, a, t in bad:
            print(f"  {name:38s} paper says {c:<10g} data says {a:<12.6g} (tol {t})")
        raise SystemExit(1)
    print("all match")


if __name__ == "__main__":
    main()
