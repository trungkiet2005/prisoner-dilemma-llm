"""Mining the play the classifier itself declines to name.

`scripts/11` mines the whole residual -- every trajectory no canonical rule
explains exactly.  That set is defined by the *rules* failing.  This module
carves out a stricter subset, defined by the *network* failing: trajectories
where no rule fits **and** the LSTM's top posterior stays under a confidence
floor.  Those are the games about which the pipeline has nothing honest to say,
and the question here is what they are made of.

Three hypotheses are separable with the data at hand:

  (i)   noise          -- near-uniform posterior, unstable across replicates,
                          no rule fits better than a shuffled sequence does;
  (ii)  interpolation  -- a two-way tie between canonical rules, sitting on an
                          edge of the reactive-strategy square;
  (iii) a missing rule -- a dense mode away from every canonical corner that a
                          wider vocabulary names exactly, above its own null.

Every coverage number is therefore paired with a permutation null in which the
focal player's own actions are shuffled, holding its cooperation rate and the
opponent's realised sequence fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The confidence floor.  0.90 is not tuned: it is the valley of a strongly
# bimodal posterior-confidence distribution (68% of the residual sits above
# 0.95, and the mass between 0.5 and 0.9 is a thin shoulder), so the cut falls
# in a gap rather than through a crowd.  fig_anatomy panel (b) sweeps it.
THRESHOLD = 0.90

STRATEGIES4 = ["AllC", "TFT", "WSLS", "AllD"]
PCOLS = ["p_AllC", "p_TFT", "p_WSLS", "p_AllD"]

BUCKETS = ["exact", "ambiguous", "confident", "unclassified"]
BUCKET_LABEL = {
    "exact": "exact rule",
    "ambiguous": "several rules fit",
    "confident": f"LSTM ≥ {THRESHOLD:.2f}",
    "unclassified": f"LSTM < {THRESHOLD:.2f}",
}


def add_buckets(arche: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Four-way read-out: rule-exact, rule-ambiguous, confident LSTM, abstain."""
    a = arche.copy()
    a["bucket"] = np.where(
        a.assignment == "exact", "exact",
        np.where(a.assignment == "ambiguous", "ambiguous",
                 np.where(a.confidence >= threshold, "confident", "unclassified")))
    return a


def posterior_geometry(arche: pd.DataFrame) -> pd.DataFrame:
    """Rank the four posteriors and record the shape of the disagreement.

    `top2_mass` separates hypothesis (i) from (ii): a network that is genuinely
    lost spreads mass over all four labels, while one caught between two rules
    puts almost everything on two of them.
    """
    a = arche.copy()
    P = a[PCOLS].to_numpy()
    names = np.array(STRATEGIES4)
    order = np.argsort(-P, axis=1)
    srt = np.take_along_axis(P, order, axis=1)
    a["top1"] = names[order[:, 0]]
    a["top2"] = names[order[:, 1]]
    a["p1"], a["p2"] = srt[:, 0], srt[:, 1]
    a["margin"] = srt[:, 0] - srt[:, 1]
    a["top2_mass"] = srt[:, 0] + srt[:, 1]
    a["pair"] = ["+".join(sorted(p, key=STRATEGIES4.index))
                 for p in zip(a.top1, a.top2)]
    return a


# --------------------------------------------------------------------------
# reactive-strategy coordinates
# --------------------------------------------------------------------------
def reactive_coordinates(rounds: pd.DataFrame) -> pd.DataFrame:
    """Nowak's reactive square: p = P(C | opponent C), q = P(C | opponent D).

    The canonical rules are corners of this square -- AllC (1,1), TFT (1,0),
    AllD (0,0) -- so a trajectory's distance from the nearest corner is a
    model-free statement about how far the behaviour is from the vocabulary.
    WSLS is deliberately *not* a corner here: it needs the player's own last
    action, which is exactly the point -- anything sitting in the interior is
    either stochastic, deeper than memory-one, or both.
    """
    r = rounds.loc[rounds.prev_letter != "E", ["game_uid", "agent",
                                               "prev_opp_action", "coop"]].copy()
    r["opp_c"] = r.prev_opp_action == "C"
    agg = r.groupby(["game_uid", "agent", "opp_c"]).coop.agg(["mean", "size"])
    mean = agg["mean"].unstack("opp_c")
    size = agg["size"].unstack("opp_c").fillna(0)
    out = pd.DataFrame({
        "p_CgivenC": mean.get(True), "q_CgivenD": mean.get(False),
        "n_after_C": size.get(True), "n_after_D": size.get(False),
    })
    return out.reset_index()


CORNERS = {"AllC": (1.0, 1.0), "TFT": (1.0, 0.0), "AllD": (0.0, 0.0)}


def corner_distance(react: pd.DataFrame) -> pd.DataFrame:
    pts = react[["p_CgivenC", "q_CgivenD"]].to_numpy()
    d = np.stack([np.hypot(pts[:, 0] - a, pts[:, 1] - b)
                  for a, b in CORNERS.values()], axis=1)
    out = react.copy()
    out["d_nearest_corner"] = d.min(axis=1)
    out["nearest_corner"] = np.array(list(CORNERS))[d.argmin(axis=1)]
    return out


# --------------------------------------------------------------------------
# an extended strategy vocabulary
# --------------------------------------------------------------------------
# Each entry maps (own_history, opponent_history, t) -> action.  The library is
# chosen to span the memory classes a memory-one rule cannot express: triggers
# that never forgive, counters over the whole history, two-round memory, and
# clocks that switch on the round index rather than on the opponent.
def _allc(own, opp, t):  return "C"
def _alld(own, opp, t):  return "D"
def _tft(own, opp, t):   return "C" if t == 0 else opp[t - 1]
def _stft(own, opp, t):  return "D" if t == 0 else opp[t - 1]


def _wsls(own, opp, t):
    if t == 0:
        return "C"
    return own[t - 1] if opp[t - 1] == "C" else ("D" if own[t - 1] == "C" else "C")


def _grim(own, opp, t):
    return "D" if "D" in opp[:t] else "C"


def _tf2t(own, opp, t):
    return "D" if t >= 2 and opp[t - 1] == "D" and opp[t - 2] == "D" else "C"


def _ttft(own, opp, t):
    return "D" if "D" in opp[max(0, t - 2):t] else "C"


def _alternator(own, opp, t):      return "C" if t % 2 == 0 else "D"
def _anti_alternator(own, opp, t): return "D" if t % 2 == 0 else "C"


def _soft_majority(own, opp, t):
    if t == 0:
        return "C"
    return "C" if opp[:t].count("C") >= opp[:t].count("D") else "D"


def _hard_majority(own, opp, t):
    if t == 0:
        return "D"
    return "C" if opp[:t].count("C") > opp[:t].count("D") else "D"


def _gradual(own, opp, t):
    """Beaufils: answer the n-th defection with n retaliations, then calm down."""
    if t == 0 or "D" not in opp[:t]:
        return "C"
    last = max(i for i in range(t) if opp[i] == "D")
    n_at = opp[:last + 1].count("D")
    return "D" if (t - last - 1) < n_at else "C"


def _prober(own, opp, t):
    if t < 3:
        return ["C", "D", "C"][t]
    return "D" if (opp[1] == "C" and opp[2] == "C") else opp[t - 1]


def _contrite_tft(own, opp, t):
    """TFT that stops a mutual-defection spiral it started itself."""
    if t == 0:
        return "C"
    if own[t - 1] == "D" and opp[t - 1] == "D":
        return "C"
    return opp[t - 1]


BASE_LIBRARY = {
    "AllC": _allc, "AllD": _alld, "TFT": _tft, "WSLS": _wsls,
    "SuspiciousTFT": _stft, "GRIM": _grim, "TF2T": _tf2t,
    "TwoTitsForTat": _ttft, "Alternator": _alternator,
    "AntiAlternator": _anti_alternator, "SoftMajority": _soft_majority,
    "HardMajority": _hard_majority, "Gradual": _gradual, "Prober": _prober,
    "ContriteTFT": _contrite_tft,
}
CANONICAL4 = ["AllC", "AllD", "TFT", "WSLS"]

# Clock rules are spliced from two opponent-only strategies, so a trajectory's
# deviation count for "A->B@k" is just a prefix sum of A's mismatches plus a
# suffix sum of B's.  That identity is what keeps the whole sweep cheap; it
# would not hold for WSLS, which reads its own last action.
CLOCK_PAIRS = [("TFT", "AllD"), ("AllC", "AllD"), ("AllD", "AllC"),
               ("AllD", "TFT"), ("TFT", "AllC"), ("AllC", "TFT")]


def replay(fn, opp: list[str]) -> list[str]:
    own: list[str] = []
    for t in range(len(opp)):
        own.append(fn(own, opp, t))
    return own


def _mismatch_matrix(own: list[str], opp: list[str]) -> dict[str, np.ndarray]:
    truth = np.array(own)
    return {name: (np.array(replay(fn, opp)) != truth)
            for name, fn in BASE_LIBRARY.items()}


def deviation_profile(own: list[str], opp: list[str]) -> tuple[dict, str, int]:
    """Deviations of every candidate rule, plus the best name and switch point."""
    mm = _mismatch_matrix(own, opp)
    dev = {k: int(v.sum()) for k, v in mm.items()}
    best_name, best_k = min(dev, key=dev.get), 0
    best = dev[best_name]

    n = len(own)
    for a, b in CLOCK_PAIRS:
        ca = np.concatenate([[0], np.cumsum(mm[a])])
        cb = np.concatenate([[0], np.cumsum(mm[b])])
        tot_b = cb[n]
        for k in range(1, n):
            d = int(ca[k] + (tot_b - cb[k]))
            dev[f"{a}->{b}@{k}"] = d
            if d < best:
                best, best_name, best_k = d, f"{a}->{b}@{k}", k
    return dev, best_name, best_k


def library_table(rounds: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Nearest rule under the canonical four and under the wide library.

    The shuffled column is the same search run against a permutation of the
    player's own actions; without it the wide library's coverage is
    uninterpretable, because 130 candidate rules will fit *something*.
    """
    rng = np.random.default_rng(seed)
    r = rounds.sort_values(["game_uid", "agent", "round"])
    g = r.groupby(["game_uid", "agent"], sort=False)
    own_s, opp_s = g.action.apply(list), g.opp_action.apply(list)

    rows = []
    for (uid, agent), own, opp in zip(own_s.index, own_s, opp_s):
        dev, best, k = deviation_profile(own, opp)
        shuffled, _, _ = deviation_profile(list(rng.permutation(own)), opp)
        rows.append({
            "game_uid": uid, "agent": agent, "n_rounds": len(own),
            "dev_canonical": min(dev[c] for c in CANONICAL4),
            "dev_extended": min(dev.values()),
            "dev_extended_null": min(shuffled.values()),
            "best_rule": best, "switch_round": k,
        })
    out = pd.DataFrame(rows)
    out["best_family"] = out.best_rule.str.replace(r"@\d+$", "", regex=True)
    out["is_canonical"] = out.best_family.isin(CANONICAL4)
    return out


# --------------------------------------------------------------------------
# sequence-level behaviour
# --------------------------------------------------------------------------
def sequence_stats(rounds: pd.DataFrame) -> pd.DataFrame:
    """Switching, run structure and the iid expectation for the same coop rate.

    `alternation_excess` is the observed switch rate minus 2p(1-p), the rate a
    coin with the same cooperation rate would produce.  Positive means the
    player alternates more than chance -- the signature of a clock-like rule --
    and negative means it perseverates in runs.
    """
    r = rounds.sort_values(["game_uid", "agent", "round"])
    g = r.groupby(["game_uid", "agent"], sort=False)
    seq = g.action.apply(list)

    rows = []
    for (uid, agent), acts in zip(seq.index, seq):
        a = np.array(acts)
        n = len(a)
        sw = int((a[1:] != a[:-1]).sum())
        p = float((a == "C").mean())
        brk = np.flatnonzero(a[1:] != a[:-1]) + 1
        runs = np.diff(np.concatenate([[0], brk, [n]]))
        rows.append({"game_uid": uid, "agent": agent, "n_rounds": n,
                     "switch_rate": sw / (n - 1), "exp_switch_iid": 2 * p * (1 - p),
                     "max_run": int(runs.max()), "n_runs": len(runs),
                     "coop": p})
    out = pd.DataFrame(rows)
    out["alternation_excess"] = out.switch_rate - out.exp_switch_iid
    return out


def split_half_reactive(rounds: pd.DataFrame) -> pd.DataFrame:
    """Reactive coordinates estimated separately on each half of a game.

    If the behaviour is a strategy, the two halves agree; if it is drift or
    noise, they do not.  Only games long enough to see both conditioning
    states twice in each half are returned.
    """
    r = rounds[rounds.prev_letter != "E"].sort_values(
        ["game_uid", "agent", "round"]).copy()
    mid = r.groupby(["game_uid", "agent"])["round"].transform("max") / 2
    r["half"] = np.where(r["round"] <= mid, "first", "second")
    r["opp_c"] = r.prev_opp_action == "C"

    agg = (r.groupby(["game_uid", "agent", "half", "opp_c"]).coop
           .agg(["mean", "size"]).reset_index())
    wide = agg.pivot_table(index=["game_uid", "agent"], columns=["half", "opp_c"],
                           values=["mean", "size"])
    wide.columns = [f"{a}_{b}_{'C' if c else 'D'}" for a, b, c in wide.columns]
    need = ["size_first_C", "size_first_D", "size_second_C", "size_second_D"]
    wide = wide.dropna(subset=need)
    ok = (wide[need] >= 2).all(axis=1)
    return wide[ok].reset_index()
