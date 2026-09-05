"""Does the payoff scale change *which strategy* the model plays?

Three questions, in increasing order of how much they depend on the classifier:

1. Rule-governedness. What share of agent-games are matched exactly by one of
   the four canonical rules? This uses no learning at all, so it is the claim
   least exposed to classifier error.
2. Distance to the nearest canonical rule, in rounds that violate it. A
   continuous version of the same question, and it needs no labels either.
3. Composition. Among the four labels, how does the mix move with the scale?
   This is the only part that leans on the LSTM, and it leans on it only for the
   agent-games where several rules fit or none does.

Trend tests use log10(scale) as a continuous predictor with standard errors
clustered on the dyad, because a rescaling is multiplicative and the design
samples it geometrically.

Writes T10..T13 to tables/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TAB = HERE / "tables"
TAB.mkdir(exist_ok=True)

LABELS = ["AllC", "TFT", "WSLS", "AllD"]
SUBUNIT = [0.01, 0.1]


def _cluster_logit(d, outcome, extra=""):
    """Logistic trend on log10(scale), clustered on the dyad."""
    fit = smf.glm(f"{outcome} ~ logscale{extra}", data=d,
                  family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": d["game_uid"]})
    return (float(fit.params["logscale"]), float(fit.bse["logscale"]),
            float(fit.pvalues["logscale"]))


def t10_provenance(d):
    """Share deduced / ambiguous / unmatched, by scale and model, plus trend."""
    rows = []
    for m, s in d.groupby("model"):
        tab = (s.pivot_table(index="scale_nominal", columns="provenance",
                             values="game_uid", aggfunc="count")
               .reindex(columns=["deduced", "ambiguous", "unmatched"]).fillna(0))
        tab = 100 * tab.div(tab.sum(axis=1), axis=0)
        for sc, r in tab.iterrows():
            rows.append({"model": m, "scale": sc, **r.to_dict()})
    per = pd.DataFrame(rows)

    trend = []
    for m, s in d.groupby("model"):
        s = s.assign(logscale=np.log10(s.scale_nominal),
                     is_deduced=(s.provenance == "deduced").astype(int))
        b, se, p = _cluster_logit(s, "is_deduced")
        lo = s[s.scale_nominal.isin(SUBUNIT)].is_deduced.mean()
        hi = s[~s.scale_nominal.isin(SUBUNIT)].is_deduced.mean()
        trend.append({"model": m, "beta_logscale": b, "se": se, "p": p,
                      "deduced_subunit": 100 * lo, "deduced_suprunit": 100 * hi})
    return per, pd.DataFrame(trend)


def t11_distance(d):
    """Mean rounds violating the nearest canonical rule, and its trend."""
    rows = []
    for m, s in d.groupby("model"):
        s = s.assign(logscale=np.log10(s.scale_nominal))
        fit = smf.ols("min_deviations ~ logscale", data=s).fit(
            cov_type="cluster", cov_kwds={"groups": s["game_uid"]})
        mm = s.groupby("scale_nominal").min_deviations.mean()
        rows.append({"model": m, "beta_logscale": float(fit.params["logscale"]),
                     "se": float(fit.bse["logscale"]),
                     "p": float(fit.pvalues["logscale"]),
                     "dist_at_min_scale": float(mm.loc[mm.index.min()]),
                     "dist_at_max_scale": float(mm.loc[mm.index.max()]),
                     "mean_dist": float(s.min_deviations.mean())})
    return pd.DataFrame(rows)


def t12_composition(d):
    """Label mix by scale and model, and a per-label trend on log10(scale)."""
    rows = []
    for m, s in d.groupby("model"):
        tab = (s.pivot_table(index="scale_nominal", columns="label",
                             values="game_uid", aggfunc="count")
               .reindex(columns=LABELS).fillna(0))
        tab = 100 * tab.div(tab.sum(axis=1), axis=0)
        for sc, r in tab.iterrows():
            rows.append({"model": m, "scale": sc, **r.to_dict()})
    per = pd.DataFrame(rows)

    trend = []
    for m, s in d.groupby("model"):
        s = s.assign(logscale=np.log10(s.scale_nominal))
        for lab in LABELS:
            s = s.assign(y=(s.label == lab).astype(int))
            b, se, p = _cluster_logit(s, "y")
            mm = s.groupby("scale_nominal").y.mean()
            trend.append({"model": m, "label": lab, "beta_logscale": b,
                          "se": se, "p": p,
                          "pct_at_min_scale": 100 * float(mm.loc[mm.index.min()]),
                          "pct_at_max_scale": 100 * float(mm.loc[mm.index.max()])})
    return per, pd.DataFrame(trend)


def t13_pooled(d):
    """The same three quantities pooled over models, for the headline sentence."""
    d = d.assign(logscale=np.log10(d.scale_nominal),
                 is_deduced=(d.provenance == "deduced").astype(int))
    prov = (d.pivot_table(index="scale_nominal", columns="provenance",
                          values="game_uid", aggfunc="count")
            .reindex(columns=["deduced", "ambiguous", "unmatched"]).fillna(0))
    prov = 100 * prov.div(prov.sum(axis=1), axis=0)
    mix = (d.pivot_table(index="scale_nominal", columns="label",
                         values="game_uid", aggfunc="count")
           .reindex(columns=LABELS).fillna(0))
    mix = 100 * mix.div(mix.sum(axis=1), axis=0)
    out = prov.join(mix, rsuffix="_pct")
    out["mean_min_deviations"] = d.groupby("scale_nominal").min_deviations.mean()
    return out.reset_index()


def main():
    d = pd.read_parquet(DATA / "readout.parquet")

    print("T10 provenance")
    per, trend = t10_provenance(d)
    per.to_csv(TAB / "T10_provenance_by_scale.csv", index=False)
    trend.to_csv(TAB / "T10_provenance_trend.csv", index=False)
    print(trend.round(4).to_string(index=False))

    print("\nT11 distance to nearest canonical rule")
    t11 = t11_distance(d)
    t11.to_csv(TAB / "T11_rule_distance.csv", index=False)
    print(t11.round(4).to_string(index=False))

    print("\nT12 composition")
    per12, trend12 = t12_composition(d)
    per12.to_csv(TAB / "T12_label_mix.csv", index=False)
    trend12.to_csv(TAB / "T12_label_trend.csv", index=False)
    print(trend12.round(4).to_string(index=False))

    print("\nT13 pooled")
    t13 = t13_pooled(d)
    t13.to_csv(TAB / "T13_pooled_strategy.csv", index=False)
    print(t13.round(2).to_string(index=False))
    print(f"\nwrote tables to {TAB}")


if __name__ == "__main__":
    main()
