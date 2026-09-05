"""Check every number typed into the manuscript prose against the data.

The tables are machine-generated, but the Results and Discussion prose quotes
figures by hand, and a wrong one there is the failure mode this project has
actually suffered. Each check below names the claim, recomputes it from the
parquet or the tables, and compares against what the paper says.

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

g = pd.read_parquet(DATA / "games.parquet")
r = pd.read_parquet(DATA / "rounds.parquet")
d = pd.read_parquet(DATA / "readout.parquet")

checks: list[tuple[str, float, float, float]] = []


def chk(name, claimed, actual, tol=0.0006):
    checks.append((name, claimed, float(actual), tol))


# ---- design -----------------------------------------------------------------
chk("dyads", 10000, g.game_uid.nunique(), 0)
chk("agent-games", 20000, len(g), 0)
chk("decisions", 200000, len(r), 0)
chk("cells", 250, g.groupby(["model", "language", "scale_nominal"]).ngroups, 0)
chk("overall cooperation", 0.565, g.coop_rate.mean(), 0.0006)

# ---- language levels quoted in 3.4 -------------------------------------------
lang = g.pivot_table(index="model", columns="language", values="coop_rate")
chk("Claude cn", 0.574, lang.loc["Claude-Haiku-4.5", "cn"])
chk("Claude vn", 0.234, lang.loc["Claude-Haiku-4.5", "vn"])
chk("GPT en", 0.798, lang.loc["GPT-5.4-Nano", "en"])
chk("GPT ar", 0.458, lang.loc["GPT-5.4-Nano", "ar"])

rng = (g.groupby(["model", "language", "scale_nominal"]).coop_rate.mean()
       .groupby(["model", "language"]).agg(lambda s: s.max() - s.min()))
chk("Claude range en", 0.136, rng.loc[("Claude-Haiku-4.5", "en")])
chk("Claude range fr", 0.314, rng.loc[("Claude-Haiku-4.5", "fr")])
chk("Gemini3.5 range cn", 0.162, rng.loc[("Gemini-3.5-Flash-Lite", "cn")])
chk("Gemini3.5 range en", 0.412, rng.loc[("Gemini-3.5-Flash-Lite", "en")])
chk("min lang range", 0.080, rng.min())
chk("max lang range", 0.449, rng.max())

# ---- Qwen3 persona inversion quoted in 3.5 -----------------------------------
q = g[g.model == "Qwen3-235B-A22B"].groupby("personality").coop_rate.mean()
chk("Qwen3 coop persona", 0.072, q.loc["cooperative"])
chk("Qwen3 selfish persona", 0.798, q.loc["selfish"])

# ---- strategy numbers quoted in 3.6 and the Discussion -----------------------
prov = d.provenance.value_counts(normalize=True) * 100
chk("deduced overall %", 23.0, prov["deduced"], 0.06)
chk("ambiguous overall %", 13.2, prov["ambiguous"], 0.06)
chk("unmatched overall %", 63.7, prov["unmatched"], 0.06)

t13 = pd.read_csv(TAB / "T13_pooled_strategy.csv").set_index("scale_nominal")
chk("deduced at 0.01", 31.4, t13.loc[0.01, "deduced"], 0.06)
chk("deduced at 1000", 19.5, t13.loc[1000.0, "deduced"], 0.06)
chk("unmatched at 0.01", 54.4, t13.loc[0.01, "unmatched"], 0.06)
chk("unmatched at 1000", 69.2, t13.loc[1000.0, "unmatched"], 0.06)
chk("AllC at 0.01", 45.6, t13.loc[0.01, "AllC"], 0.06)
chk("AllC at 1000", 40.4, t13.loc[1000.0, "AllC"], 0.06)
chk("TFT at 0.01", 9.2, t13.loc[0.01, "TFT"], 0.06)
chk("TFT at 1000", 13.0, t13.loc[1000.0, "TFT"], 0.06)

# provenance per label: the numbers that bound how the compositional claim in
# 2.6 and the Discussion may be read
pl = pd.crosstab(d.label, d.provenance, normalize="index") * 100
chk("AllC deduced %", 26.0, pl.loc["AllC", "deduced"], 0.06)
chk("AllC ambiguous %", 29.6, pl.loc["AllC", "ambiguous"], 0.06)
chk("AllD deduced %", 32.7, pl.loc["AllD", "deduced"], 0.06)
chk("TFT unmatched %", 95.4, pl.loc["TFT", "unmatched"], 0.06)
chk("WSLS unmatched %", 97.3, pl.loc["WSLS", "unmatched"], 0.06)

mix = pd.read_csv(TAB / "T12_label_mix.csv")


def m(model, label, scale):
    row = mix[(mix.model == model) & (mix.scale == scale)]
    return float(row[label].iloc[0])


chk("GPT AllC 0.01", 29.8, m("GPT-5.4-Nano", "AllC", 0.01), 0.06)
chk("GPT AllC 1000", 50.8, m("GPT-5.4-Nano", "AllC", 1000.0), 0.06)
chk("GPT AllD 0.01", 46.0, m("GPT-5.4-Nano", "AllD", 0.01), 0.06)
chk("GPT AllD 1000", 17.5, m("GPT-5.4-Nano", "AllD", 1000.0), 0.06)
chk("G3.5 AllC 0.01", 73.8, m("Gemini-3.5-Flash-Lite", "AllC", 0.01), 0.06)
chk("G3.5 AllC 1000", 51.5, m("Gemini-3.5-Flash-Lite", "AllC", 1000.0), 0.06)
chk("G3.5 TFT 0.01", 6.8, m("Gemini-3.5-Flash-Lite", "TFT", 0.01), 0.06)
chk("G3.5 TFT 1000", 14.3, m("Gemini-3.5-Flash-Lite", "TFT", 1000.0), 0.06)

t11 = pd.read_csv(TAB / "T11_rule_distance.csv").set_index("model")
chk("G3.5 dist min", 0.48, t11.loc["Gemini-3.5-Flash-Lite", "dist_at_min_scale"], 0.006)
chk("G3.5 dist max", 1.37, t11.loc["Gemini-3.5-Flash-Lite", "dist_at_max_scale"], 0.006)
chk("Qwen dist min", 0.27, t11.loc["Qwen3-235B-A22B", "dist_at_min_scale"], 0.006)
chk("Qwen dist max", 1.14, t11.loc["Qwen3-235B-A22B", "dist_at_max_scale"], 0.006)

# ---- first move quoted in 3.7 ------------------------------------------------
t08 = pd.read_csv(TAB / "T08_first_move.csv").set_index("model")
chk("ratio Gemini3.1", 3.7, t08.loc["Gemini-3.1-Flash-Lite-Preview", "ratio"], 0.05)
chk("ratio Claude", 2.0, t08.loc["Claude-Haiku-4.5", "ratio"], 0.05)
chk("ratio Gemini3.5", 1.9, t08.loc["Gemini-3.5-Flash-Lite", "ratio"], 0.05)
chk("ratio GPT", 1.3, t08.loc["GPT-5.4-Nano", "ratio"], 0.05)

# ---- pooling quoted in 3.3 and the abstract ----------------------------------
t04 = pd.read_csv(TAB / "T04_pooling.csv").iloc[0]
chk("pooled range", 0.048, t04.pooled_range, 0.0006)
chk("mean within-model range", 0.136, t04.mean_within_model_range, 0.0006)
chk("attenuation", 2.8, t04.attenuation_factor, 0.05)
chk("mean shape corr", -0.10, t04.mean_pairwise_shape_corr, 0.006)
chk("min shape corr", -0.72, t04.min_pairwise_shape_corr, 0.006)
chk("max shape corr", 0.73, t04.max_pairwise_shape_corr, 0.006)
chk("negative pairs", 6, t04.n_negative_pairs, 0)

# ---- effect sizes quoted in the abstract and 3.2 -----------------------------
t03 = pd.read_csv(TAB / "T03_effect_size.csv").set_index("model")
chk("GPT range", 0.243, t03.loc["GPT-5.4-Nano", "range"])
chk("GPT range lo", 0.200, t03.loc["GPT-5.4-Nano", "lo"])
chk("GPT range hi", 0.290, t03.loc["GPT-5.4-Nano", "hi"])
chk("G3.1 range", 0.072, t03.loc["Gemini-3.1-Flash-Lite-Preview", "range"])
chk("G3.5 range", 0.191, t03.loc["Gemini-3.5-Flash-Lite", "range"])
chk("GPT coop min", 0.438, t03.loc["GPT-5.4-Nano", "coop_min"])

# ---- variance decomposition --------------------------------------------------
t06 = pd.read_csv(TAB / "T06_variance.csv").set_index("term")
chk("model %", 56.5, t06.loc["C(model)", "pct_variance"], 0.06)
chk("model x language %", 19.7, t06.loc["C(model):C(language)", "pct_variance"], 0.06)
chk("scale main %", 0.7, t06.loc["C(scale_nominal)", "pct_variance"], 0.06)
chk("scale main p", 0.36, t06.loc["C(scale_nominal)", "PR(>F)"], 0.006)
chk("scale main F", 1.10, t06.loc["C(scale_nominal)", "F"], 0.006)
chk("model x scale %", 6.5, t06.loc["C(model):C(scale_nominal)", "pct_variance"], 0.06)
chk("language x scale %", 3.8, t06.loc["C(language):C(scale_nominal)", "pct_variance"], 0.06)
chk("language x scale p", 0.041, t06.loc["C(language):C(scale_nominal)", "PR(>F)"], 0.0006)

# ---- persona -----------------------------------------------------------------
t07 = pd.read_csv(TAB / "T07_persona_gating.csv").set_index("model")
for mod, sub, sup, sh, lo, hi in [
        ("Gemini-3.5-Flash-Lite", -0.456, 0.224, 0.680, 0.625, 0.736),
        ("Claude-Haiku-4.5", -0.222, 0.207, 0.429, 0.389, 0.471),
        ("GPT-5.4-Nano", -0.135, 0.018, 0.153, 0.105, 0.200)]:
    chk(f"{mod} persona sub", sub, t07.loc[mod, "persona_effect_subunit"])
    chk(f"{mod} persona sup", sup, t07.loc[mod, "persona_effect_suprunit"])
    chk(f"{mod} persona shift", sh, t07.loc[mod, "shift"])
    chk(f"{mod} shift lo", lo, t07.loc[mod, "shift_lo"])
    chk(f"{mod} shift hi", hi, t07.loc[mod, "shift_hi"])
chk("Qwen persona sub", -0.912, t07.loc["Qwen3-235B-A22B", "persona_effect_subunit"])
chk("Qwen persona sup", -0.679, t07.loc["Qwen3-235B-A22B", "persona_effect_suprunit"])
chk("G3.1 shift", 0.103, t07.loc["Gemini-3.1-Flash-Lite-Preview", "shift"])
chk("Qwen shift", 0.233, t07.loc["Qwen3-235B-A22B", "shift"])

# ---- classifier --------------------------------------------------------------
t01 = pd.read_csv(TAB / "T01_classifier.csv").set_index("split")
chk("clf held-out", 0.985, t01.loc["held-out (NoNoise + Noise005)", "accuracy"], 0.0006)
chk("clf OOD 10%", 0.936, t01.loc["OOD (Noise01)", "accuracy"], 0.0006)
chk("clf OOD 20%", 0.811, t01.loc["OOD (Noise02)", "accuracy"], 0.0006)

# ---- trend coefficients quoted in 3.6 ----------------------------------------
t10 = pd.read_csv(TAB / "T10_provenance_trend.csv").set_index("model")
chk("Claude beta", -0.545, t10.loc["Claude-Haiku-4.5", "beta_logscale"])
chk("Qwen beta", -0.267, t10.loc["Qwen3-235B-A22B", "beta_logscale"])
chk("G3.5 beta", -0.242, t10.loc["Gemini-3.5-Flash-Lite", "beta_logscale"])
chk("G3.1 beta", 0.125, t10.loc["Gemini-3.1-Flash-Lite-Preview", "beta_logscale"])
chk("GPT beta p", 0.44, t10.loc["GPT-5.4-Nano", "p"], 0.006)


def main():
    bad = []
    for name, claimed, actual, tol in checks:
        if abs(claimed - actual) > tol:
            bad.append((name, claimed, actual, tol))
    print(f"{len(checks)} numbers checked against the data")
    if bad:
        print(f"\n{len(bad)} MISMATCH:")
        for name, c, a, t in bad:
            print(f"  {name:34s} paper says {c:<10g} data says {a:<12.6g} (tol {t})")
        raise SystemExit(1)
    print("all match")


if __name__ == "__main__":
    main()
