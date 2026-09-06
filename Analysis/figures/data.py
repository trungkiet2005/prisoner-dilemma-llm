"""Loaders and derived quantities shared by the figure scripts.

Everything here reads `Analysis/scaling/data/*.parquet`, written by
`s00_build.py`, and `Analysis/scaling/tables/*.csv`, written by the s03..s09
statistics scripts.  Nothing here recomputes a number the manuscript quotes;
`s08_verify_paper.py` remains the single authority on those.

The one derived object that is new is the memory-one fingerprint.  A memory-one
strategy in the evolutionary literature is a five-vector: the probability of
opening with cooperation, and the probability of cooperating after each of the
four possible previous outcomes.  Writing the corpus in that vector is what
lets an empirical agent and a textbook rule be plotted in the same space, and
it is the space the evolutionary baseline in `s09_egt.py` already lives in.

Context labels are `own` then `opponent`: `pCD` is the probability of
cooperating after a round in which this agent cooperated and its opponent
defected.  Tit-for-tat therefore has pCD = 0 and pDC = 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import style as S


def rounds() -> pd.DataFrame:
    r = pd.read_parquet(S.DATA / "rounds.parquet")
    r["ctx"] = r.prev_action.fillna("") + r.prev_opp_action.fillna("")
    return r


def games() -> pd.DataFrame:
    return pd.read_parquet(S.DATA / "games.parquet")


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(S.TABLES / name)


# --- memory-one fingerprints -------------------------------------------------

def _mem1(g: pd.DataFrame, min_n: int) -> pd.Series:
    """The five-vector, plus the support behind each entry.

    A context observed fewer than `min_n` times returns NaN rather than a
    proportion computed from three rounds; the figures then show the gap
    instead of drawing a point that is mostly noise.
    """
    out = {}
    first = g.loc[g["round"] == 1, "coop"]
    out["p0"] = first.mean() if len(first) >= min_n else np.nan
    out["n0"] = len(first)
    for c in ("CC", "CD", "DC", "DD"):
        s = g.loc[g.ctx == c, "coop"]
        out["p" + c] = s.mean() if len(s) >= min_n else np.nan
        out["n" + c] = len(s)
    return pd.Series(out)


def fingerprints(r: pd.DataFrame, by, *, min_n: int = 30) -> pd.DataFrame:
    """Memory-one vector for every cell of `by`."""
    return (r.groupby(list(by), observed=True)
             .apply(_mem1, min_n=min_n, include_groups=False)
             .reset_index())


def canonical_frame() -> pd.DataFrame:
    """The four textbook rules as rows in the same five columns."""
    return pd.DataFrame(
        [dict(zip(S.MEM1, S.STRAT_VEC[k]), rule=k) for k in S.STRAT_ORDER]
    )


def contrasts(fp: pd.DataFrame) -> pd.DataFrame:
    """The four contexts are a 2x2 design, so read them as one.

    The predecessor of a round is a pair (own last action, opponent's last
    action), which is a two-by-two factorial.  The four cell probabilities
    therefore decompose without remainder into a level and three contrasts,
    and each contrast is a named behaviour rather than an unnamed axis:

      level        mean cooperation over the four cells
      reciprocity  p(C | opponent cooperated) - p(C | opponent defected)
                   the opponent main effect; this is what tit-for-tat is
      persistence  p(C | I cooperated) - p(C | I defected)
                   the own main effect; conditioning on oneself, not on
                   the other player
      matching     (pCC + pDD)/2 - (pCD + pDC)/2
                   the interaction; this is what win-stay lose-shift is

    The four textbook rules sit at known coordinates, which is what makes the
    space readable: AllC and AllD are both (0, 0, 0) and differ only in level,
    tit-for-tat is reciprocity 1, win-stay lose-shift is matching 1.

    Note for anyone tempted by the simpler contrast pCD - pDC: it equals
    persistence minus reciprocity exactly, so it confounds the two main
    effects and must not be read as a reciprocity measure.
    """
    return pd.DataFrame({
        "level": fp[["pCC", "pCD", "pDC", "pDD"]].mean(axis=1),
        "reciprocity": (fp.pCC + fp.pDC) / 2 - (fp.pCD + fp.pDD) / 2,
        "persistence": (fp.pCC + fp.pCD) / 2 - (fp.pDC + fp.pDD) / 2,
        "matching": (fp.pCC + fp.pDD) / 2 - (fp.pCD + fp.pDC) / 2,
    }, index=fp.index)


CONTRASTS = ["level", "reciprocity", "persistence", "matching"]
CONTRAST_LABEL = {
    "level": "cooperation level",
    "reciprocity": "reciprocity\nanswer the opponent",
    "persistence": "persistence\nanswer yourself",
    "matching": "matching\nrepeat when you agreed",
}


def coop_matrix(g: pd.DataFrame, models=None, scales=None) -> pd.DataFrame:
    """Cooperation rate as a model-by-scale matrix, the shape a heatmap wants."""
    models = models or S.MODEL_ORDER
    scales = scales or S.SCALES
    m = (g[g.model.isin(models)]
         .groupby(["model", "scale_nominal"], observed=True).coop_rate.mean()
         .unstack("scale_nominal"))
    return m.reindex(index=models, columns=scales)


def bootstrap_ci(x, *, n=2000, alpha=0.05, seed=20260906):
    """Percentile bootstrap of the mean, clustered on nothing in particular.

    Used only for the visual interval on descriptive panels; every inferential
    number in the manuscript comes from the tables, not from here.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    draws = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))
