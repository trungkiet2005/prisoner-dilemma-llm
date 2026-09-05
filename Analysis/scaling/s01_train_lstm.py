"""Train the LSTM branch of the hybrid strategy read-out.

The training corpus is synthetic and has nothing to do with any language model:
it is `Dataset/noise_dataset`, canonical AllC / AllD / TFT / WSLS trajectories
emitted at four execution-noise levels.  Training on the noise-free and the
5%-noise variants together forces the network to recognise a rule from
*imperfect* play, which is the regime LLM transcripts live in; the 10% and 20%
variants are never trained on and are kept as an out-of-distribution check.

Sequence length is capped at 10 to match the game length in the corpus exactly,
so nothing is padded away and nothing is truncated.

Outputs:
    models/strategy_lstm.pt      weights + the config needed to rebuild
    tables/T01_classifier.csv    accuracy in-distribution and OOD
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdlib.lstm import StrategyLSTM, make_loader, predict     # noqa: E402
from pdlib.seqcode import STRATEGIES, read_corpus             # noqa: E402
from pdlib.style import DATASET                               # noqa: E402

HERE = Path(__file__).resolve().parent
MODELDIR = HERE / "models"; MODELDIR.mkdir(exist_ok=True)
TABDIR = HERE / "tables"; TABDIR.mkdir(exist_ok=True)

MAX_LEN = 10
SEED = 1234
EPOCHS = 12
CORPUS = "4stratsAllCAllDTFTWSLS"
TRAIN_LEVELS = {"NoNoise": "nonoise", "Noise005": "noise005"}
OOD_LEVELS = {"Noise01": "noise01", "Noise02": "noise02"}


def _load(level: str, suffix: str):
    path = DATASET / "noise_dataset" / level / f"{CORPUS}_{suffix}.txt"
    return read_corpus(path, max_len=MAX_LEN)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    Xs, Ls, Ys = [], [], []
    for level, suffix in TRAIN_LEVELS.items():
        X, lens, y = _load(level, suffix)
        Xs.append(X); Ls.append(lens); Ys.append(y)
        print(f"  {level}: {len(X):,} sequences")
    X = np.concatenate(Xs); lens = np.concatenate(Ls); y = np.concatenate(Ys)

    # 80/20 split, stratification is unnecessary because the corpus is exactly
    # balanced (40,320 per strategy per level).
    perm = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = perm[:cut], perm[cut:]

    model = StrategyLSTM()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss()
    loader = make_loader(X[tr], lens[tr], y[tr], batch=512, shuffle=True)

    for ep in range(EPOCHS):
        model.train()
        tot = n = 0
        for xb, lb, yb in loader:
            opt.zero_grad()
            loss = lossf(model(xb, lb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(yb); n += len(yb)
        sched.step()
        acc = (predict(model, X[te], lens[te]).argmax(1) == y[te]).mean()
        print(f"  epoch {ep+1:2d}  loss {tot/n:.4f}  held-out acc {acc:.4f}")

    rows = [{"split": "held-out (NoNoise + Noise005)", "n": len(te),
             "accuracy": float((predict(model, X[te], lens[te]).argmax(1) == y[te]).mean())}]
    for level, suffix in OOD_LEVELS.items():
        Xo, lo, yo = _load(level, suffix)
        rows.append({"split": f"OOD ({level})", "n": len(Xo),
                     "accuracy": float((predict(model, Xo, lo).argmax(1) == yo).mean())})

    # per-strategy recall on the held-out split, which is what the read-out
    # actually leans on when the rule set is ambiguous
    p = predict(model, X[te], lens[te]).argmax(1)
    for k, s in enumerate(STRATEGIES):
        m = y[te] == k
        rows.append({"split": f"held-out recall: {s}", "n": int(m.sum()),
                     "accuracy": float((p[m] == k).mean())})

    tab = pd.DataFrame(rows)
    tab.to_csv(TABDIR / "T01_classifier.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "max_len": MAX_LEN,
                "strategies": STRATEGIES, "seed": SEED},
               MODELDIR / "strategy_lstm.pt")
    print(tab.to_string(index=False))
    print(f"wrote {MODELDIR/'strategy_lstm.pt'}")


if __name__ == "__main__":
    main()
