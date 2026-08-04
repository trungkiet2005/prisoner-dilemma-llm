"""Apply the strategy read-out to frontier play, and write the archetype table.

Read-out order, which is the whole point of the design:

  1. **Exact rule matching first.**  All four canonical strategies are
     memory-one, so "is this trajectory exactly what AllC / AllD / TFT / WSLS
     would have played?" is a per-token lookup with no learning involved.
  2. **Several rules fitting is an answer, not a failure.**  A player that
     cooperates for ten rounds against a cooperator *is* AllC and TFT and WSLS
     at once; reporting the set is correct, and an argmax tie-break would
     manufacture a result.
  3. **The LSTM only speaks when no rule fits**, and then its label means
     "nearest rule", not "is".

The network is `models/strategy_lstm_h10.pt`, trained by `05_train_classifier.py`
on the 10-round synthetic corpus (noise-free + 5% execution noise).  The
frontier games are 10 rounds, so the horizon matches exactly; nothing is
retrained here.

Writes `data/frontier_archetypes.parquet` and `data/frontier_prefixes.npz`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from pdlib.lstm import StrategyLSTM, predict, predict_prefixes
from pdlib.natstyle import DATADIR, FRONTIER, MODELDIR, TABDIR
from pdlib.rulebase import consistent_mask, deviation_counts, set_name
from pdlib.seqcode import PAD, STOI, STRATEGIES

MAX_LEN = 10
TAG = "h10"


def load_readout(tag: str = TAG):
    ck = torch.load(MODELDIR / f"strategy_lstm_{tag}.pt", weights_only=False)
    m = StrategyLSTM()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def build_sequences(rounds: pd.DataFrame, max_len: int = MAX_LEN):
    """One padded token sequence per (game, focal agent)."""
    r = rounds.sort_values(["game_uid", "agent", "round"])
    keys, seqs = [], []
    for (uid, ag), d in r.groupby(["game_uid", "agent"], sort=False):
        keys.append((uid, ag))
        seqs.append([STOI[t] for t in d.token.tolist()[:max_len]])
    X = np.full((len(seqs), max_len), PAD, dtype=np.int64)
    L = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s
        L[i] = len(s)
    return keys, X, L


def main():
    rounds = pd.read_parquet(DATADIR / "frontier_rounds.parquet")
    games = pd.read_parquet(DATADIR / "frontier_games.parquet")

    model = load_readout()
    keys, X, L = build_sequences(rounds)
    proba = predict(model, X, L)
    pref = predict_prefixes(model, X)

    mask = consistent_mask(X)
    n_fit = mask.sum(axis=1)
    sets = [frozenset(np.array(STRATEGIES)[row]) for row in mask]
    dev = deviation_counts(X, L)

    arche, kind = [], []
    for i in range(len(X)):
        if n_fit[i] == 1:
            arche.append(next(iter(sets[i])))
            kind.append("exact")
        elif n_fit[i] > 1:
            arche.append("Ambiguous")
            kind.append("ambiguous")
        else:
            arche.append(STRATEGIES[int(proba[i].argmax())])
            kind.append("approx")

    ent = -(proba * np.log(np.clip(proba, 1e-12, 1))).sum(1) / np.log(len(STRATEGIES))

    df = pd.DataFrame(keys, columns=["game_uid", "agent"])
    df["archetype"] = arche
    df["assignment"] = kind
    df["rule_set"] = [set_name(s) for s in sets]
    df["n_rule_fits"] = n_fit
    df["min_deviations"] = dev.min(axis=1)
    df["nearest_rule"] = [STRATEGIES[int(j)] for j in dev.argmin(axis=1)]
    df["lstm_archetype"] = [STRATEGIES[i] for i in proba.argmax(1)]
    df["confidence"] = proba.max(1)
    df["entropy"] = ent
    for i, s in enumerate(STRATEGIES):
        df[f"p_{s}"] = proba[:, i]

    df = df.merge(games[["game_uid", "agent", "model", "language",
                         "scale_nominal", "personality", "opp_personality",
                         "dyad", "coop_rate", "efficiency", "payoff_per_round",
                         "cc_rate", "dd_rate"]],
                  on=["game_uid", "agent"], how="left")
    if df.model.isna().any():
        raise SystemExit("archetype rows failed to join back to games.parquet")

    df.to_parquet(DATADIR / "frontier_archetypes.parquet", index=False)
    np.savez_compressed(DATADIR / "frontier_prefixes.npz", pref=pref, lens=L,
                        X=X, keys=np.array([f"{a}|{b}" for a, b in keys]))

    mix = (df.groupby(["model", "assignment"]).size()
           .unstack("assignment").reindex(FRONTIER).fillna(0))
    mix = mix.div(mix.sum(axis=1), axis=0)
    mix["mean_deviations"] = df.groupby("model").min_deviations.mean().reindex(FRONTIER)
    mix.to_csv(TABDIR / "T_FR27_assignment_mix.csv")

    shares = (df.groupby(["model", "archetype"]).size()
              .unstack("archetype").reindex(FRONTIER).fillna(0))
    shares = shares.div(shares.sum(axis=1), axis=0)
    shares.to_csv(TABDIR / "T_FR28_archetype_shares.csv")

    print(f"{len(df):,} agent-games classified   "
          f"({df.assignment.value_counts().to_dict()})")
    print()
    print("assignment mix (share of agent-games)")
    print(mix.round(3).to_string())
    print()
    print("archetype shares")
    print(shares.round(3).to_string())


if __name__ == "__main__":
    main()
