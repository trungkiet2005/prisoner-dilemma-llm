"""Tools for mining the trajectories that no canonical rule explains.

Two thirds of LLM play falls outside the AllC / AllD / TFT / WSLS vocabulary.
This module widens the hypothesis class in controlled steps so the question
"what *is* that play?" gets an answer with a null baseline attached:

  1. the 4 canonical rules
  2. all 32 deterministic memory-one rules  (opening x action for R,S,T,P)
  3. any of those with a single mid-game regime switch
  4. whatever is left

Every coverage number is paired with a permutation null in which the focal
player's own action sequence is shuffled, holding both its cooperation rate and
the opponent's actual sequence fixed.  That null matters: a two-segment fit over
ten rounds has enough freedom to explain 31% of shuffled play by chance.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .seqcode import PAD, STOI, VOCAB

STATES = ["R", "S", "T", "P"]
_OUTCOME = {("C", "C"): "R", ("C", "D"): "S", ("D", "C"): "T", ("D", "D"): "P"}

# every deterministic memory-one rule: an opening move plus an action per state
FULL_RULES = {op + "".join(c): {"E": op, **dict(zip(STATES, c))}
              for op in "CD" for c in itertools.product("CD", repeat=4)}
# the same without the opening, for the tail segment of a switched trajectory
CORE_RULES = {"".join(c): dict(zip(STATES, c))
              for c in itertools.product("CD", repeat=4)}

FULL_NAMES = list(FULL_RULES)
CORE_NAMES = list(CORE_RULES)

NAMED = {"CCCCC": "AllC", "DDDDD": "AllD", "CCDCD": "TFT", "CCDDC": "WSLS",
         "CCDDD": "GRIM", "DCDCD": "Suspicious TFT", "CCCCD": "Prober",
         "DCCCC": "Suspicious AllC", "CDDDD": "Contrarian open"}
CANONICAL_4 = ["CCCCC", "DDDDD", "CCDCD", "CCDDC"]


def _token_table(rules, ignore_opening: bool) -> np.ndarray:
    t = np.zeros((len(rules), len(VOCAB)), dtype=bool)
    for k, name in enumerate(rules):
        for vid, tok in enumerate(VOCAB):
            if vid == PAD:
                t[k, vid] = True
            elif tok[0] == "E" and ignore_opening:
                t[k, vid] = True
            else:
                t[k, vid] = rules[name][tok[0]] == tok[1]
    return t


FULL_OK = _token_table(FULL_RULES, ignore_opening=False)
CORE_OK = _token_table(CORE_RULES, ignore_opening=True)


def encode(own: list[list[str]], opp: list[list[str]], max_len: int):
    """(own, opponent) action sequences -> padded corpus-token id matrix."""
    X = np.full((len(own), max_len), PAD, dtype=np.int64)
    lens = np.zeros(len(own), dtype=np.int64)
    for i, (o, q) in enumerate(zip(own, opp)):
        prev, ids = "E", []
        for t in range(min(len(o), max_len)):
            ids.append(STOI[prev + o[t]])
            prev = _OUTCOME[(o[t], q[t])]
        X[i, :len(ids)] = ids
        lens[i] = len(ids)
    return X, lens


def hypothesis_coverage(X: np.ndarray, max_len: int):
    """Which of the nested hypothesis classes explains each trajectory."""
    one = FULL_OK[:, X].all(axis=2).T                      # (N, 32)
    canon = one[:, [FULL_NAMES.index(k) for k in CANONICAL_4]].any(axis=1)
    mem1 = one.any(axis=1)

    pre = np.cumprod(FULL_OK[:, X], axis=2)                # prefix consistency
    suf = np.cumprod(CORE_OK[:, X][:, :, ::-1], axis=2)[:, :, ::-1]
    ap, asf = pre.any(axis=0), suf.any(axis=0)
    two = np.zeros(len(X), dtype=bool)
    cut = np.full(len(X), -1, dtype=int)
    for t in range(max_len - 1):
        hit = ap[:, t] & asf[:, t + 1] & (cut < 0)
        cut[hit] = t + 1
        two |= ap[:, t] & asf[:, t + 1]
    return {"canonical4": canon, "memory1_32": mem1, "two_regime": two,
            "switch_round": cut, "per_rule": one}


def shuffle_null(own, opp, max_len, seed=0):
    """Permutation null: shuffle the focal player's own actions in place.

    Cooperation rate and the opponent's realised sequence are preserved, only
    the temporal contingency between them is destroyed.
    """
    rng = np.random.default_rng(seed)
    own_s = [list(rng.permutation(o)) for o in own]
    Xs, _ = encode(own_s, opp, max_len)
    return hypothesis_coverage(Xs, max_len)


# --------------------------------------------------------------------------
def conditional_profile(df: pd.DataFrame, smooth: float = 1.0) -> pd.Series:
    """Laplace-smoothed P(cooperate | previous joint outcome), per trajectory.

    Smoothing matters: over ten rounds many states are never visited, and an
    unobserved state shrinks to 0.5 -- "no information" -- rather than dropping
    the trajectory or inventing a 0/1.
    """
    out = {"first_move": float(df.coop.iloc[0]),
           "coop_rate": float(df.coop.mean())}
    for st in STATES:
        s = df.loc[df.prev_letter == st, "coop"]
        out[f"pC_{st}"] = (s.sum() + smooth) / (len(s) + 2 * smooth)
        out[f"n_{st}"] = len(s)
    lag = df.dropna(subset=["prev_action"])
    out["self_persist"] = float((lag.action == lag.prev_action).mean()) if len(lag) else np.nan
    out["opp_match"] = float((lag.action == lag.prev_opp_action).mean()) if len(lag) else np.nan
    return pd.Series(out)


def bic_memory_depth(sub: pd.DataFrame) -> dict:
    """BIC of nested predictors of the next move, on rounds 3 and later.

    All four models are scored on identical rows, so the comparison is about
    how much history the behaviour actually uses, not about sample size.
    """
    y = sub.coop.to_numpy()
    n = len(y)
    if n < 50:
        return {}
    prev1 = sub.prev_letter.to_numpy()
    prev2 = sub.prev2.to_numpy()
    late = (sub.round_frac.to_numpy() > 0.5)

    def score(labels):
        ll, k = 0.0, 0
        for v in pd.unique(labels):
            yy = y[labels == v]
            if not len(yy):
                continue
            p = np.clip(yy.mean(), 1e-9, 1 - 1e-9)
            ll += float((yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum())
            k += 1
        return -2 * ll + k * np.log(n)

    b0 = score(np.zeros(n))
    return {
        "n": n,
        "BIC_memoryless": b0,
        "BIC_memory1": score(prev1),
        "BIC_memory2": score(np.char.add(prev2.astype(str), prev1.astype(str))),
        "BIC_memory1_time": score(np.char.add(prev1.astype(str), late.astype(str))),
        "base": b0,
    }
