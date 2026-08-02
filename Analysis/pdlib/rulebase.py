"""Exact rule matching for the four canonical strategies.

Every rule here is memory-one, so the action it prescribes depends only on the
previous round's joint outcome -- which is exactly the first letter of each
corpus token.  That makes "is this trajectory *exactly* what strategy k would
have played?" a per-token lookup, and it is checked without any learning.

    prev  own/opp last round      AllC  AllD  TFT   WSLS
    E     (first move)             C     D     C     C
    R     both cooperated          C     D     C     C     stay, won
    S     I cooperated, was hit    C     D     D     D     shift, lost
    T     I defected, they didn't  C     D     C     D     TFT copies C, WSLS stays D
    P     both defected            C     D     D     C     TFT copies D, WSLS shifts

Because several rules can prescribe the same actions on the history that
actually occurred, the matcher returns a *set*.  A player who cooperated for
thirty rounds against a cooperator is exactly consistent with AllC, TFT and
WSLS at once, and no method -- learned or not -- can separate them; saying so
is the correct answer, not a failure.
"""
from __future__ import annotations

import numpy as np

from .seqcode import PAD, STRATEGIES, VOCAB

# prescribed action for each (strategy, previous-outcome letter)
RULES: dict[str, dict[str, str]] = {
    "AllC": {"E": "C", "R": "C", "S": "C", "T": "C", "P": "C"},
    "AllD": {"E": "D", "R": "D", "S": "D", "T": "D", "P": "D"},
    "TFT":  {"E": "C", "R": "C", "S": "D", "T": "C", "P": "D"},
    "WSLS": {"E": "C", "R": "C", "S": "D", "T": "D", "P": "C"},
}

N_STRAT = len(STRATEGIES)


def _token_ok_table() -> np.ndarray:
    """(n_strategies, vocab) boolean: is this token allowed under the rule?

    Padding is always allowed so that short sequences are handled by the same
    vectorised AND-reduction.
    """
    ok = np.zeros((N_STRAT, len(VOCAB)), dtype=bool)
    for k, s in enumerate(STRATEGIES):
        for vid, tok in enumerate(VOCAB):
            if vid == PAD:
                ok[k, vid] = True
            else:
                prev, act = tok[0], tok[1]
                ok[k, vid] = RULES[s][prev] == act
    return ok


TOKEN_OK = _token_ok_table()


def consistent_mask(X: np.ndarray) -> np.ndarray:
    """(N, n_strategies) boolean: does each trajectory exactly follow the rule?"""
    per_token = TOKEN_OK[:, X]            # (n_strat, N, L)
    return per_token.all(axis=2).T        # (N, n_strat)


def deviation_counts(X: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """(N, n_strategies) int: how many rounds violate each rule.

    Zero means an exact match; the minimum over strategies is a natural
    "distance to the nearest canonical rule" that needs no model.
    """
    bad = (~TOKEN_OK)[:, X]               # (n_strat, N, L); padding is never bad
    return bad.sum(axis=2).T


def label_sets(X: np.ndarray) -> list[frozenset[str]]:
    m = consistent_mask(X)
    return [frozenset(np.array(STRATEGIES)[row]) for row in m]


def set_name(s: frozenset[str]) -> str:
    """Canonical printable name, in the project's strategy order."""
    if not s:
        return "none"
    order = {k: i for i, k in enumerate(["AllC", "TFT", "WSLS", "AllD"])}
    return "+".join(sorted(s, key=lambda x: order[x]))


def hybrid_predict(X: np.ndarray, proba: np.ndarray):
    """Rules decide the candidate set; the LSTM only ranks inside it.

    Returns
    -------
    sets    list of frozensets -- the multi-label answer (empty if no rule fits)
    single  int array -- one label per trajectory, for metrics that need one:
            the LSTM argmax *restricted* to the rule set when a rule fits, the
            unrestricted LSTM argmax otherwise
    source  str array -- 'rule-unique', 'rule-set', or 'lstm'
    """
    m = consistent_mask(X)
    n_fit = m.sum(axis=1)

    masked = np.where(m, proba, -np.inf)
    single = np.where(n_fit > 0, masked.argmax(axis=1), proba.argmax(axis=1))

    source = np.where(n_fit == 1, "rule-unique",
                      np.where(n_fit > 1, "rule-set", "lstm"))
    sets = [frozenset(np.array(STRATEGIES)[row]) for row in m]
    return sets, single.astype(int), source
