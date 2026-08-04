"""Parse the raw FAIRGAME prisoner's-dilemma logs into tidy tables.

Two long tables come out of `build_master()`:

* `rounds` -- one row per (game, round, focal agent).  Both agents of a game
  appear, so every dyad contributes two rows per round and agent-level
  statistics are symmetric by construction.
* `games`  -- one row per (game, focal agent), with the per-game aggregates.

Coding conventions
------------------
**OptionA == Defect (D), OptionB == Cooperate (C).**

The FAIRGAME prisoner's-dilemma templates state the numbers as *penalties*
(years of imprisonment) and instruct the agent to **minimise** them, so a
smaller number is a better outcome.  Under that framing the recorded cells are

                      opponent C      opponent D
        focal C          R = 2           S = 10
        focal D          T = 0           P = 6

with P = 6 for the frontier runs and P = 8 for the small open-weight runs.
Reading these as penalties gives T < R < P < S, which is the penalty-space
image of the usual T > R > P > S: mutual cooperation (B,B) carries the lowest
symmetric penalty, defecting on a cooperator carries none at all, and A is
therefore the dominant -- that is, the defecting -- action.

This polarity is not cosmetic.  Mapping OptionA to Cooperate inverts every
cooperation rate in the corpus (x -> 1 - x) and reverses the sign of the
payoff-scale and endgame effects; it was the convention used here until it was
caught by reproducing Fig. 2a of arXiv:2601.19082 from these tables, which
matches to a mean absolute error of 0.002 under the mapping above and 0.171
under the inverted one.  `payoff_matrix()` returns these raw penalties
together with the greed/fear indices, which are computed on the negated
(utility) scale so they remain comparable with the standard PD literature.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .style import DATASET

# --------------------------------------------------------------------------
# label dictionaries
# --------------------------------------------------------------------------
ACTION_MAP = {"OptionA": "D", "OptionB": "C"}

PERSONALITY_MAP = {
    # english
    "cooperative": "cooperative", "COOPERATIVE": "cooperative",
    "selfish": "selfish", "SELFISH": "selfish",
    # french
    "coopératif": "cooperative", "égoïste": "selfish",
    # vietnamese
    "một người hợp tác": "cooperative", "một người ích kỷ": "selfish",
    # chinese
    "合作型的": "cooperative", "自私型的": "selfish",
    # arabic
    "متعاون": "cooperative", "أناني": "selfish",
}

MODEL_MAP = {
    "claude": "Claude-3.5-Haiku",
    "gpt": "GPT-4o",
    "mistra": "Mistral-Large",
    "gemini-3.5-flash-lite": "Gemini-3.5-Flash-Lite",
    "gemma_3_12b": "Gemma-3-12B",
    "llama-3-1-8b": "Llama-3.1-8B",
    "qwen3_8b": "Qwen3-8B",
}

FAMILY_DIR = {
    "data_fairgame_frontier_llm": "frontier",
    "data_fairgame_small_llm": "small",
}

# base (scale = 1) PENALTY matrices, keyed by family: these are the numbers
# the agent is actually shown and told to minimise, so T < R < P < S.  They
# are also what `build_master` divides the recorded scores by to recover the
# realised payoff scale, so they must stay in raw recorded units.
_BASE_MATRIX = {
    "frontier": {"T": 0.0, "R": 2.0, "P": 6.0, "S": 10.0},
    "small": {"T": 0.0, "R": 2.0, "P": 8.0, "S": 10.0},
}


def payoff_matrix(family: str) -> dict:
    """T/R/P/S as raw penalties, plus the two classic PD indices.

    The indices are defined on the *utility* scale u = -penalty, where the
    familiar ordering T > R > P > S holds, so they can be read against the
    standard literature:

        greed = (T - R) / (T - S)   incentive to defect against a cooperator
        fear  = (P - S) / (T - S)   incentive to defect against a defector

    Substituting u = -penalty turns those into the penalty-space expressions
    below, which is why the signs look flipped relative to a reward matrix.
    """
    m = dict(_BASE_MATRIX[family])
    span = m["S"] - m["T"]                       # = T_u - S_u
    m["greed"] = (m["R"] - m["T"]) / span
    m["fear"] = (m["S"] - m["P"]) / span
    # k-index of Rapoport: how far R sits between P and T
    m["k"] = (m["P"] - m["R"]) / span
    # utilities, for anything that needs a higher-is-better scale
    m["u"] = {k: -m[k] for k in ("T", "R", "P", "S")}
    return m


def outcome_code(own: str, opp: str) -> str:
    return own + opp  # 'CC' | 'CD' | 'DC' | 'DD'


# outcome letter of the *focal* player's payoff, as used by the noise corpus
OUTCOME_LETTER = {"CC": "R", "CD": "S", "DC": "T", "DD": "P"}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def _lit(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def _iter_files(families: tuple[str, ...] | None = None):
    for famdir, family in FAMILY_DIR.items():
        if families is not None and family not in families:
            continue
        root = DATASET / famdir
        if not root.exists():
            continue
        for scale_dir in sorted(root.iterdir(), key=lambda p: float(p.name)):
            for model_dir in sorted(scale_dir.iterdir()):
                for csv in sorted(model_dir.glob("*.csv")):
                    yield family, float(scale_dir.name), MODEL_MAP[model_dir.name], csv


def build_master(families: tuple[str, ...] | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the logs. `families` restricts to a subset of FAMILY_DIR values."""
    round_rows: list[dict] = []

    for family, scale_nominal, model, csv in _iter_files(families):
        lang = re.match(r"^x[\d.]+_([a-z]{2})_", csv.name).group(1)
        df = pd.read_csv(csv)
        base = _BASE_MATRIX[family]

        # `game_id` is NOT a unique key: the frontier logs repeat each
        # personality condition ten times under the same id, so the row
        # position inside the file is what identifies a replicate.
        rep_counter: dict[str, int] = {}

        for row_idx, (_, r) in enumerate(df.iterrows()):
            rep = rep_counter.get(r.game_id, 0)
            rep_counter[r.game_id] = rep + 1
            a = [_lit(r.agent1_strategies), _lit(r.agent2_strategies)]
            s = [_lit(r.agent1_scores), _lit(r.agent2_scores)]
            pers = [PERSONALITY_MAP[r.agent1_personality],
                    PERSONALITY_MAP[r.agent2_personality]]

            n = min(len(a[0]), len(a[1]))
            acts = [[ACTION_MAP[x] for x in a[0][:n]], [ACTION_MAP[x] for x in a[1][:n]]]

            # A few logs mix payoff scales inside one file, so the folder name
            # is not authoritative.  Every round's outcome pins down which of
            # T/R/P/S was paid, so the realised scale is recovered exactly as
            # the median of payoff / base-value over the non-zero rounds.
            ratios = []
            for i in (0, 1):
                j = 1 - i
                for t in range(n):
                    ref = base[OUTCOME_LETTER[outcome_code(acts[i][t], acts[j][t])]]
                    if ref > 0:
                        ratios.append(float(s[i][t]) / ref)
            scale_eff = float(np.median(ratios)) if ratios else scale_nominal

            for i in (0, 1):
                j = 1 - i
                for t in range(n):
                    own, opp = acts[i][t], acts[j][t]
                    oc = outcome_code(own, opp)
                    round_rows.append({
                        "family": family,
                        "model": model,
                        "language": lang,
                        "scale_nominal": scale_nominal,
                        "scale_eff": scale_eff,
                        "game_uid": f"{csv.stem}#{row_idx:03d}",
                        "game_id": r.game_id,
                        "rep": rep,
                        "agent": i + 1,
                        "personality": pers[i],
                        "opp_personality": pers[j],
                        "dyad": f"{pers[i][0].upper()}v{pers[j][0].upper()}",
                        "rounds_known": bool(r.n_rounds_is_known),
                        "max_rounds": int(r.max_rounds),
                        "round": t + 1,
                        "round_frac": (t + 1) / n,
                        "action": own,
                        "opp_action": opp,
                        "coop": int(own == "C"),
                        "opp_coop": int(opp == "C"),
                        "outcome": oc,
                        "payoff_raw": float(s[i][t]),
                        "payoff_base": float(s[i][t]) / scale_eff,
                    })

    rounds = pd.DataFrame(round_rows)

    # ---- lagged features (own / opponent previous move, previous outcome) ---
    rounds = rounds.sort_values(["game_uid", "agent", "round"]).reset_index(drop=True)
    g = rounds.groupby(["game_uid", "agent"], sort=False)
    rounds["prev_action"] = g["action"].shift(1)
    rounds["prev_opp_action"] = g["opp_action"].shift(1)
    rounds["prev_outcome"] = g["outcome"].shift(1)
    rounds["prev_letter"] = rounds["prev_outcome"].map(OUTCOME_LETTER).fillna("E")
    rounds["token"] = rounds["prev_letter"] + rounds["action"]

    # normalised outcome in [0, 1] on the S..T span, oriented so that higher is
    # better: the worst penalty S maps to 0 and the best penalty T maps to 1.
    span = rounds["family"].map(lambda f: _BASE_MATRIX[f]["T"] - _BASE_MATRIX[f]["S"])
    smin = rounds["family"].map(lambda f: _BASE_MATRIX[f]["S"])
    rounds["payoff_norm"] = (rounds["payoff_base"] - smin) / span
    # Efficiency against the mutual-cooperation benchmark, again higher-is-
    # better: 1.0 is the penalty a dyad pays when it cooperates every round,
    # 0.5 is mutual defection, and a round of successful exploitation exceeds
    # 1.0.  Dividing the raw penalty by R would invert this, because under the
    # penalty framing a *smaller* number is the better outcome.
    rben = rounds["family"].map(lambda f: _BASE_MATRIX[f]["R"])
    swst = rounds["family"].map(lambda f: _BASE_MATRIX[f]["S"])
    rounds["efficiency"] = (swst - rounds["payoff_base"]) / (swst - rben)

    # ---- per-game aggregates ------------------------------------------------
    keys = ["family", "model", "language", "scale_nominal", "game_uid", "game_id",
            "rep", "agent", "personality", "opp_personality", "dyad", "rounds_known",
            "max_rounds"]
    games = (rounds.groupby(keys, sort=False)
             .agg(n_rounds=("round", "size"),
                  coop_rate=("coop", "mean"),
                  opp_coop_rate=("opp_coop", "mean"),
                  first_move_coop=("coop", "first"),
                  last_move_coop=("coop", "last"),
                  payoff_base=("payoff_base", "sum"),
                  payoff_per_round=("payoff_base", "mean"),
                  efficiency=("efficiency", "mean"),
                  cc_rate=("outcome", lambda x: (x == "CC").mean()),
                  dd_rate=("outcome", lambda x: (x == "DD").mean()),
                  cd_rate=("outcome", lambda x: (x == "CD").mean()),
                  dc_rate=("outcome", lambda x: (x == "DC").mean()))
             .reset_index())

    # longest defection run, and whether the dyad ever locks into DD
    def _longest_run(seq, val="D"):
        best = cur = 0
        for x in seq:
            cur = cur + 1 if x == val else 0
            best = max(best, cur)
        return best

    runs = (rounds.groupby("game_uid", sort=False)
            .apply(lambda d: pd.Series({
                "longest_D_run": _longest_run(d[d.agent == 1]["action"].tolist()),
                "dd_absorbed": _dd_absorbing(d[d.agent == 1]["outcome"].tolist()),
            }), include_groups=False)
            .reset_index())
    games = games.merge(runs, on="game_uid", how="left")

    return rounds, games


def _dd_absorbing(outcomes: list[str]) -> int:
    """1 if the dyad reaches DD and never leaves it before the game ends."""
    if "DD" not in outcomes:
        return 0
    first = outcomes.index("DD")
    return int(all(o == "DD" for o in outcomes[first:]))


# --------------------------------------------------------------------------
# convenience: Wilson interval for proportion bars
# --------------------------------------------------------------------------
def wilson(p: np.ndarray | float, n: np.ndarray | float, z: float = 1.96):
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    n = np.where(n == 0, np.nan, n)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def cluster_bootstrap_ci(df: pd.DataFrame, value: str, cluster: str = "game_uid",
                         n_boot: int = 2000, seed: int = 0, agg="mean"):
    """Bootstrap CI that resamples whole games, not rounds.

    Rounds inside a game are strongly autocorrelated, so a naive round-level
    interval is far too narrow; resampling at the game level is the honest
    version and is what every interval in the paper uses.
    """
    rng = np.random.default_rng(seed)
    groups = df.groupby(cluster, sort=False)[value].agg(agg).to_numpy()
    if len(groups) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    boots = groups[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
