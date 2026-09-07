"""Whole-dyad bootstrap intervals for the memory-one contrasts.

This is a descriptive sensitivity analysis. It uses the released round table,
keeps both agents of each dyad together, and does not call a model endpoint.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "rounds.parquet"
OUT = HERE / "tables" / "T20_conditioning_ci.csv"
N_BOOT = 4000
SEED = 20260907

def one(v):
    out = {}
    for own in "CD":
        for opp in "CD":
            x = v[(v.prev_action == own) & (v.prev_opp_action == opp)].coop
            out[f"p{own}{opp}"] = x.mean() if len(x) else np.nan
    p = out
    out["reciprocity"] = (p["pCC"] + p["pDC"] - p["pCD"] - p["pDD"]) / 2
    out["persistence"] = (p["pCC"] + p["pCD"] - p["pDC"] - p["pDD"]) / 2
    out["matching"] = (p["pCC"] + p["pDD"] - p["pCD"] - p["pDC"]) / 2
    return out

def main():
    r = pd.read_parquet(DATA)
    r = r[r.model.isin(["Claude-Haiku-4.5", "GPT-5.4-Nano",
                        "Gemini-3.1-Flash-Lite-Preview", "Gemini-3.5-Flash-Lite",
                        "Grok-4.20-Non-Reasoning", "Qwen3-235B-A22B"])]
    r = r.dropna(subset=["prev_action", "prev_opp_action"])
    rng = np.random.default_rng(SEED)
    rows = []
    for model, d in r.groupby("model"):
        uids = d.game_uid.drop_duplicates().to_numpy()
        # Store context-level successes and trials once, then bootstrap sums.
        d = d.copy()
        d["ctx"] = d.prev_action + d.prev_opp_action
        tab = d.pivot_table(index="game_uid", columns="ctx", values="coop",
                            aggfunc=["sum", "count"], fill_value=0)
        for stat in ("sum", "count"):
            for ctx in ["CC", "CD", "DC", "DD"]:
                if (stat, ctx) not in tab:
                    tab[(stat, ctx)] = 0
        tab = tab.sort_index()
        sums = tab["sum"][["CC", "CD", "DC", "DD"]].to_numpy(float)
        counts = tab["count"][["CC", "CD", "DC", "DD"]].to_numpy(float)
        def from_arrays(ss, nn):
            p = np.divide(ss, nn, out=np.full(4, np.nan), where=nn > 0)
            return {"reciprocity": (p[0]+p[2]-p[1]-p[3])/2,
                    "persistence": (p[0]+p[1]-p[2]-p[3])/2,
                    "matching": (p[0]+p[3]-p[1]-p[2])/2}
        obs = from_arrays(sums.sum(0), counts.sum(0))
        draws = {k: [] for k in ["reciprocity", "persistence", "matching"]}
        for _ in range(N_BOOT):
            ix = rng.integers(0, len(uids), len(uids))
            b = from_arrays(sums[ix].sum(0), counts[ix].sum(0))
            for k in draws:
                draws[k].append(b[k])
        for k, vals in draws.items():
            rows.append({"model": model, "contrast": k, "estimate": obs[k],
                         "lo": np.nanpercentile(vals, 2.5),
                         "hi": np.nanpercentile(vals, 97.5),
                         "n_dyads": len(uids), "n_boot": N_BOOT})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(OUT)

if __name__ == "__main__":
    main()
