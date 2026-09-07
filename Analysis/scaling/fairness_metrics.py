"""Scale-normalised within-dyad fairness metrics for the supplement."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute(games: pd.DataFrame) -> pd.DataFrame:
    """Return one row per dyad using the released ten-round payoff support."""
    g = games.copy()
    p = g.pivot(index="game_uid", columns="agent", values="payoff_base").dropna()
    n_rounds = g.groupby("game_uid").n_rounds.first().reindex(p.index)
    span = 10.0 * n_rounds
    utility = 1.0 - p.div(span, axis=0)
    gap = (utility.max(axis=1) - utility.min(axis=1)).rename("utility_gap")
    total = utility.sum(axis=1)
    fairness = (1.0 - (utility.max(axis=1) - utility.min(axis=1)) / total)
    meta = g.groupby("game_uid").agg(
        model=("model", "first"), scale=("scale_nominal", "first"),
        p_exploit=("dc_rate", "first"), p_return=("cd_rate", "first"),
    ).reindex(p.index)
    out = meta.assign(
        utility_gap=gap,
        fairness=fairness,
        gini=1.0 - fairness,
        welfare=utility.mean(axis=1),
    )
    out["p_exploit"] = out["p_exploit"] + out["p_return"]
    return out.reset_index()


def headline_metrics(frame: pd.DataFrame) -> dict[str, float]:
    x = np.log10(frame["scale"].to_numpy())
    fit = stats.linregress(x, frame["utility_gap"].to_numpy())
    return {
        "zero_gap": float((frame.utility_gap == 0).mean()),
        "gap_exploit_r": float(frame.utility_gap.corr(frame.p_exploit)),
        "slope": float(fit.slope),
        "p_slope": float(fit.pvalue),
        "fairness_mean": float(frame.fairness.mean()),
        "fairness_min": float(frame.fairness.min()),
    }
