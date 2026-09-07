"""Non-API robustness analyses requested by the reviewer.

Outputs threshold sensitivity for the conditioning contrasts, a formal
scale-by-language interaction test, and leave-one-model-out pooled checks.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TAB = HERE / "tables"
MODELS = ["Claude-Haiku-4.5", "GPT-5.4-Nano", "Gemini-3.1-Flash-Lite-Preview",
          "Gemini-3.5-Flash-Lite", "Grok-4.20-Non-Reasoning", "Qwen3-235B-A22B"]

def cell_contrasts(d, threshold=30, smooth=False):
    d = d.dropna(subset=["prev_action", "prev_opp_action"]).copy()
    d["ctx"] = d.prev_action + d.prev_opp_action
    g = d.groupby(["model", "scale_nominal", "language", "ctx"], observed=True).coop.agg(["sum", "count"])
    rows = []
    for key, x in g.groupby(level=[0,1,2]):
        x = x.droplevel([0,1,2]).reindex(["CC","CD","DC","DD"])
        if not smooth and (x["count"] < threshold).any():
            continue
        s, n = x["sum"].to_numpy(float), x["count"].to_numpy(float)
        if smooth: s, n = s + 0.5, n + 1.0
        p = s / n
        rows.append(dict(model=key[0], scale=key[1], language=key[2],
                         reciprocity=(p[0]+p[2]-p[1]-p[3])/2,
                         persistence=(p[0]+p[1]-p[2]-p[3])/2,
                         matching=(p[0]+p[3]-p[1]-p[2])/2))
    return pd.DataFrame(rows)

def threshold_table(r):
    rows=[]
    for t in [10,20,30,40,50]:
        c=cell_contrasts(r, threshold=t)
        for m,x in c.groupby("model"):
            diff=x.persistence-x.reciprocity
            rows.append(dict(model=m, min_context_count=t, n_cells=len(x),
                             mean_difference=diff.mean(), median_difference=diff.median(),
                             share_persistence_greater=float((diff>0).mean())))
    c=cell_contrasts(r, smooth=True)
    for m,x in c.groupby("model"):
        diff=x.persistence-x.reciprocity
        rows.append(dict(model=m, min_context_count="Jeffreys", n_cells=len(x),
                         mean_difference=diff.mean(), median_difference=diff.median(),
                         share_persistence_greater=float((diff>0).mean())))
    return pd.DataFrame(rows)

def language_interactions(r):
    rows=[]
    for m in MODELS:
        d=r[r.model==m].copy()
        d["cluster"] = d.language.astype(str)+"|"+d.game_id.astype(str)
        fit=smf.glm("coop ~ C(scale_nominal)*C(language) + C(personality) + C(opp_personality)",
                    data=d, family=sm.families.Binomial()).fit(
                    cov_type="cluster", cov_kwds={"groups":d.cluster})
        names=[n for n in fit.params.index if ":C(language)" in n]
        R=np.zeros((len(names),len(fit.params)))
        for i,n in enumerate(names): R[i,list(fit.params.index).index(n)]=1
        wt=fit.wald_test(R, scalar=True)
        scales=[n for n in fit.params.index if n.startswith("C(scale_nominal)") and ":" not in n]
        Rs=np.zeros((len(scales),len(fit.params)))
        for i,n in enumerate(scales): Rs[i,list(fit.params.index).index(n)]=1
        ws=fit.wald_test(Rs, scalar=True)
        rows.append(dict(model=m, chi2=float(wt.statistic), df=len(names),
                         p=float(wt.pvalue), p_scale_block=float(ws.pvalue),
                         chi2_scale_block=float(ws.statistic), df_scale=len(scales),
                         n_obs=len(d), n_clusters=d.cluster.nunique()))
    out=pd.DataFrame(rows); out["p_holm"]=multipletests(out.p.to_numpy(), method="holm")[1]
    return out

def loo(g):
    rows=[]
    for removed in MODELS:
        d=g[g.model!=removed]
        means=d.groupby("scale_nominal").coop_rate.mean()
        fit=smf.ols("coop_rate ~ C(scale_nominal)", data=d).fit(cov_type="cluster", cov_kwds={"groups":d.game_uid})
        terms=[n for n in fit.params.index if n.startswith("C(scale_nominal)")]
        R=np.zeros((len(terms),len(fit.params)))
        for i,n in enumerate(terms): R[i,list(fit.params.index).index(n)]=1
        wt=fit.wald_test(R, scalar=True)
        rows.append(dict(removed=removed, pooled_range=float(means.max()-means.min()),
                         scale_chi2=float(wt.statistic), scale_df=len(terms),
                         scale_p=float(wt.pvalue), n_dyads=d.game_uid.nunique()))
    return pd.DataFrame(rows)

def main():
    r=pd.read_parquet(DATA/"rounds.parquet")
    g=pd.read_parquet(DATA/"games.parquet")
    threshold_table(r).to_csv(TAB/"T21_conditioning_threshold_sensitivity.csv",index=False)
    language_interactions(r).to_csv(TAB/"T22_scale_language_interaction.csv",index=False)
    loo(g).to_csv(TAB/"T23_leave_one_model_out.csv",index=False)
    print("wrote T21, T22, T23")

if __name__ == "__main__": main()
