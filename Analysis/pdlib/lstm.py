"""A small sequence classifier that reads a player's trajectory and names the
canonical strategy it is following (AllC / AllD / TFT / WSLS).

The corpus is synthetic and noisy on purpose: training on the noise-free and
the 5%-execution-noise variants together forces the model to identify a rule
from *imperfect* behaviour, which is the regime LLM transcripts live in.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .seqcode import PAD, STRATEGIES, VOCAB_SIZE


class StrategyLSTM(nn.Module):
    def __init__(self, emb: int = 24, hidden: int = 96, layers: int = 1,
                 n_class: int = len(STRATEGIES), dropout: float = 0.15):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True,
                            bidirectional=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_class)
        self.hidden_size = hidden

    def encode(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """Hidden state at the last *valid* step of every sequence."""
        h, _ = self.lstm(self.emb(x))
        idx = (lens - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(2))
        return h.gather(1, idx).squeeze(1)

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.encode(x, lens)))

    def forward_all_steps(self, x: torch.Tensor) -> torch.Tensor:
        """Logits after every prefix length -- used for the identifiability
        curve and for the sliding-window read-out on LLM transcripts."""
        h, _ = self.lstm(self.emb(x))
        return self.head(h)


def make_loader(X, lens, y, batch: int = 512, shuffle: bool = True):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X),
                                        torch.from_numpy(lens),
                                        torch.from_numpy(y))
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle,
                                       num_workers=0)


@torch.no_grad()
def predict(model, X, lens, batch: int = 2048):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch])
        lb = torch.from_numpy(lens[i:i + batch])
        out.append(torch.softmax(model(xb, lb), dim=1).numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_prefixes(model, X, batch: int = 2048):
    """(N, L, C) posterior after each prefix length."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch])
        out.append(torch.softmax(model.forward_all_steps(xb), dim=2).numpy())
    return np.concatenate(out)


@torch.no_grad()
def encode_hidden(model, X, lens, batch: int = 2048):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch])
        lb = torch.from_numpy(lens[i:i + batch])
        out.append(model.encode(xb, lb).numpy())
    return np.concatenate(out)
