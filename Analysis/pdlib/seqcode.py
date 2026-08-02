"""Shared token alphabet between the synthetic strategy corpus and LLM play.

The noise corpus encodes a player's trajectory as a sequence of two-letter
tokens `<previous-outcome-letter><current-action>`:

    E  no history yet (round 1)          C  the player cooperated this round
    R  both cooperated last round        D  the player defected this round
    S  player was the sucker last round
    T  player exploited last round
    P  mutual defection last round

so `WSLS: EC SD PC RC SD` reads "opened C; got suckered so shifted to D; got
punished so shifted back to C; was rewarded so stayed C; got suckered again".

`encode_pair()` produces exactly the same alphabet from a raw (own, opponent)
action pair sequence, which is what makes a classifier trained on the corpus
directly applicable to LLM transcripts.
"""
from __future__ import annotations

import numpy as np

OUTCOME_LETTERS = ["E", "R", "S", "T", "P"]
ACTIONS = ["C", "D"]

VOCAB = ["<pad>"] + [o + a for o in OUTCOME_LETTERS for a in ACTIONS]
STOI = {t: i for i, t in enumerate(VOCAB)}
ITOS = {i: t for t, i in STOI.items()}
PAD = 0
VOCAB_SIZE = len(VOCAB)

STRATEGIES = ["AllC", "AllD", "TFT", "WSLS"]
LTOI = {s: i for i, s in enumerate(STRATEGIES)}

_OUTCOME_FROM_PAIR = {("C", "C"): "R", ("C", "D"): "S",
                      ("D", "C"): "T", ("D", "D"): "P"}


def encode_pair(own: list[str], opp: list[str]) -> list[str]:
    """(own actions, opponent actions) -> list of corpus tokens."""
    toks, prev = [], "E"
    for t, a in enumerate(own):
        toks.append(prev + a)
        prev = _OUTCOME_FROM_PAIR[(own[t], opp[t])]
    return toks


def tokens_to_ids(tokens: list[str]) -> list[int]:
    return [STOI[t] for t in tokens]


def read_corpus(path, max_len: int | None = None):
    """Read one `<label>: tok tok ...` file -> (padded id matrix, lengths, y)."""
    seqs, labels = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or ":" not in line:
                continue
            lab, rest = line.split(":", 1)
            lab = lab.strip()
            if lab not in LTOI:
                continue
            toks = rest.split()
            if not toks:
                continue
            seqs.append([STOI[t] for t in toks])
            labels.append(LTOI[lab])

    L = max_len or max(len(s) for s in seqs)
    X = np.full((len(seqs), L), PAD, dtype=np.int64)
    lens = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:L]
        X[i, :len(s)] = s
        lens[i] = len(s)
    return X, lens, np.asarray(labels, dtype=np.int64)
