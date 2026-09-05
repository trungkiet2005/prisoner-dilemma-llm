"""Hybrid rule-base + LSTM strategy read-out on every agent-game.

The two branches do different jobs and the order matters.  The rule base asks a
question with an exact answer - "is this trajectory *precisely* what AllC / AllD
/ TFT / WSLS would have played, given the history that actually occurred?" - and
returns a *set*, because over ten rounds several canonical rules routinely
prescribe the same moves.  A player who cooperated throughout against a
cooperator is exactly consistent with AllC, TFT and WSLS at once; reporting all
three is the correct answer, not a failure of the method.  The LSTM never
overrides that set.  It only ranks inside it, and it decides alone only when no
canonical rule fits at all.

Reporting therefore separates three provenances, and the paper never merges them:

    deduced       exactly one canonical rule fits - no learning involved
    ambiguous     several fit and the LSTM picks among them
    unmatched     none fits; the LSTM's label is an attribution, not a deduction

Outputs:
    data/readout.parquet   one row per agent-game, with label / set / provenance
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdlib.lstm import StrategyLSTM, predict                      # noqa: E402
from pdlib.rulebase import (consistent_mask, deviation_counts,    # noqa: E402
                            hybrid_predict, set_name)
from pdlib.seqcode import PAD, STRATEGIES, encode_pair, tokens_to_ids   # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MAX_LEN = 10


def build_sequences(rounds: pd.DataFrame):
    """One token sequence per (dyad, focal agent), in a fixed row order."""
    rounds = rounds.sort_values(["game_uid", "agent", "round"])
    keys, seqs = [], []
    for (uid, agent), d in rounds.groupby(["game_uid", "agent"], sort=False):
        own = d["action"].tolist()
        opp = d["opp_action"].tolist()
        seqs.append(tokens_to_ids(encode_pair(own, opp)))
        keys.append((uid, agent))

    X = np.full((len(seqs), MAX_LEN), PAD, dtype=np.int64)
    lens = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:MAX_LEN]
        X[i, :len(s)] = s
        lens[i] = len(s)
    return pd.DataFrame(keys, columns=["game_uid", "agent"]), X, lens


def main() -> None:
    rounds = pd.read_parquet(DATA / "rounds.parquet")
    games = pd.read_parquet(DATA / "games.parquet")

    keys, X, lens = build_sequences(rounds)
    print(f"encoded {len(X):,} agent-games")

    ckpt = torch.load(HERE / "models" / "strategy_lstm.pt", map_location="cpu")
    model = StrategyLSTM()
    model.load_state_dict(ckpt["state_dict"])
    proba = predict(model, X, lens)

    sets, single, source = hybrid_predict(X, proba)
    mask = consistent_mask(X)
    dev = deviation_counts(X, lens)

    out = keys.copy()
    out["label"] = [STRATEGIES[i] for i in single]
    out["rule_set"] = [set_name(s) for s in sets]
    out["n_rules_fit"] = mask.sum(axis=1)
    out["provenance"] = np.where(out.n_rules_fit == 1, "deduced",
                                 np.where(out.n_rules_fit > 1, "ambiguous", "unmatched"))
    out["min_deviations"] = dev.min(axis=1)
    out["lstm_conf"] = proba.max(axis=1)
    for k, s in enumerate(STRATEGIES):
        out[f"p_{s}"] = proba[:, k]

    merged = games.merge(out, on=["game_uid", "agent"], how="left", validate="1:1")
    assert merged["label"].notna().all(), "read-out did not cover every agent-game"
    merged.to_parquet(DATA / "readout.parquet", index=False)

    print("\nprovenance overall:")
    print((merged.provenance.value_counts(normalize=True) * 100).round(1).to_string())
    print("\nlabel mix overall (%):")
    print((merged.label.value_counts(normalize=True) * 100).round(1).to_string())
    print(f"\nwrote {DATA/'readout.parquet'}")


if __name__ == "__main__":
    main()
