"""Behavioural metrics computed from the tidy round table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ingest import cluster_bootstrap_ci


# --------------------------------------------------------------------------
# conditional / reciprocity metrics
# --------------------------------------------------------------------------
def reciprocity(df: pd.DataFrame) -> pd.Series:
    """P(C | opponent cooperated last round) and its complement.

    nice        P(C) on round 1                 -- opening disposition
    p_c_after_c P(C_t | opp C_{t-1})            -- reward for cooperation
    p_c_after_d P(C_t | opp D_{t-1})            -- forgiveness after betrayal
    reciprocity p_c_after_c - p_c_after_d       -- 1 for TFT, 0 for AllC/AllD
    retaliation 1 - p_c_after_d                 -- how hard it punishes
    """
    d = df.dropna(subset=["prev_opp_action"])
    after_c = d.loc[d.prev_opp_action == "C", "coop"]
    after_d = d.loc[d.prev_opp_action == "D", "coop"]
    pc = after_c.mean() if len(after_c) else np.nan
    pd_ = after_d.mean() if len(after_d) else np.nan
    return pd.Series({
        "nice": df.loc[df["round"] == 1, "coop"].mean(),
        "p_c_after_c": pc,
        "p_c_after_d": pd_,
        "reciprocity": pc - pd_,
        "retaliation": 1 - pd_,
        "forgiveness": pd_,
        "n_after_c": len(after_c),
        "n_after_d": len(after_d),
    })


def state_transitions(df: pd.DataFrame) -> pd.Series:
    """P(cooperate | previous joint outcome).

    These four numbers are the behavioural fingerprint of a memory-one
    strategy: (1,1,1,1) is AllC, (0,0,0,0) is AllD, (1,0,1,0) is TFT and
    (1,0,0,1) is WSLS -- in the order R, S, T, P.
    """
    out = {}
    for st in ["R", "S", "T", "P"]:
        sub = df.loc[df.prev_letter == st, "coop"]
        out[f"pC_{st}"] = sub.mean() if len(sub) else np.nan
        out[f"n_{st}"] = len(sub)
    return pd.Series(out)


MEMORY1_REFERENCE = {
    "AllC": (1, 1, 1, 1),
    "AllD": (0, 0, 0, 0),
    "TFT": (1, 0, 1, 0),
    "WSLS": (1, 0, 0, 1),
    "GRIM": (1, 0, 0, 0),
}


# --------------------------------------------------------------------------
# scale sensitivity
# --------------------------------------------------------------------------
def scale_slope(games: pd.DataFrame, value: str = "coop_rate",
                n_boot: int = 2000, seed: int = 0) -> pd.Series:
    """OLS slope of `value` on log10(payoff scale), with a game-level bootstrap.

    Under the standard game-theoretic reading a PD is invariant to a positive
    rescaling of every payoff: the ordering T > R > P > S and the ratio
    structure are untouched, so a rational agent's policy must not move.  A
    slope reliably different from zero is therefore direct evidence that the
    model responds to the *magnitude* of the numbers rather than to the
    incentive structure they encode.
    """
    x = np.log10(games["scale_nominal"].to_numpy())
    y = games[value].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(np.unique(x)) < 2:
        return pd.Series({"slope": np.nan, "lo": np.nan, "hi": np.nan, "r2": np.nan})
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        j = rng.integers(0, n, n)
        try:
            slopes[i] = np.polyfit(x[j], y[j], 1)[0]
        except Exception:
            slopes[i] = np.nan
    return pd.Series({
        "slope": b, "intercept": a,
        "lo": np.nanpercentile(slopes, 2.5), "hi": np.nanpercentile(slopes, 97.5),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    })


# --------------------------------------------------------------------------
# fairness / bias
# --------------------------------------------------------------------------
def fairness_gap(games: pd.DataFrame, by: str = "language",
                 value: str = "coop_rate") -> float:
    """max - min of the group means: FAIRGAME's disparity statistic."""
    m = games.groupby(by)[value].mean()
    return float(m.max() - m.min()) if len(m) > 1 else np.nan


def permutation_gap_p(games: pd.DataFrame, by: str = "language",
                      value: str = "coop_rate", n_perm: int = 5000,
                      seed: int = 0) -> tuple[float, float]:
    """Observed disparity and its permutation p-value.

    Labels are shuffled at the *game* level, so the null keeps the number of
    games per group fixed and only destroys the association with the group.
    """
    obs = fairness_gap(games, by, value)
    rng = np.random.default_rng(seed)
    lab = games[by].to_numpy().copy()
    val = games[value].to_numpy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(lab)
        s = pd.Series(val).groupby(lab).mean()
        null[i] = s.max() - s.min()
    return obs, float((null >= obs).mean())


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


# --------------------------------------------------------------------------
# helpers for plotting
# --------------------------------------------------------------------------
def mean_with_ci(games: pd.DataFrame, value: str, n_boot: int = 2000, seed: int = 0):
    m = games[value].mean()
    lo, hi = cluster_bootstrap_ci(games, value, cluster="game_uid",
                                  n_boot=n_boot, seed=seed)
    return m, lo, hi


def grouped_ci(games: pd.DataFrame, groupby: list[str], value: str,
               n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for key, sub in games.groupby(groupby, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        m, lo, hi = mean_with_ci(sub, value, n_boot=n_boot, seed=seed)
        rows.append(dict(zip(groupby, key)) | {"mean": m, "lo": lo, "hi": hi,
                                               "n": len(sub)})
    return pd.DataFrame(rows)
