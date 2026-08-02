"""Train the strategy classifiers used everywhere downstream.

Two models are fit:

  horizon-10  trained on the 10-round corpus (variable prefix lengths 5-10),
              applied later to the frontier transcripts;
  horizon-30  trained on the 30-round corpus, applied to the open-weight
              transcripts.

Both are trained on the union of the noise-free and the 5% execution-noise
splits.  Held-out noise levels (10% and 20%, available for the 10-round
corpus) are never seen in training and are used purely as a robustness test.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pdlib.lstm import StrategyLSTM, make_loader, predict
from pdlib.seqcode import LTOI, STRATEGIES, read_corpus
from pdlib.style import DATADIR, DATASET, MODELDIR, TABDIR

SEED = 17
CORPUS = {
    "h10": {
        "root": DATASET / "noise_dataset",
        "max_len": 10,
        "train_splits": [("NoNoise", "4stratsAllCAllDTFTWSLS_nonoise.txt"),
                         ("Noise005", "4stratsAllCAllDTFTWSLS_noise005.txt")],
        "ood_splits": [("Noise01", "4stratsAllCAllDTFTWSLS_noise01.txt"),
                       ("Noise02", "4stratsAllCAllDTFTWSLS_noise02.txt")],
        "unseen_strategy": ("Noise005", "5stratsAllCAllDTFTWSLSGTFT_noise005.txt"),
        "dup_cap": 40,
    },
    "h30": {
        "root": DATASET / "noise_dataset_30round",
        "max_len": 30,
        "train_splits": [("NoNoise", "4stratsAllCAllDTFTWSLS_nonoise.txt"),
                         ("Noise005", "4stratsAllCAllDTFTWSLS_noise005.txt")],
        "ood_splits": [],
        "unseen_strategy": ("Noise005", "5stratsAllCAllDTFTWSLSGTFT_noise005.txt"),
        "dup_cap": 40,
    },
}


def load_split(root, parts, max_len, dup_cap: int | None = None):
    """Read the corpus files, optionally capping exact duplicates.

    The 30-round noise-free file is degenerate: 161,280 lines collapse to
    eight distinct trajectories, whose labels are genuinely ambiguous, so its
    Bayes ceiling is only 0.63.  Left uncapped that block of pure label noise
    dominates the loss.  Capping how many copies of any identical
    (sequence, label) pair survive keeps the informative patterns and their
    relative frequencies while removing the degenerate mass.
    """
    Xs, Ls, Ys, Src = [], [], [], []
    for sub, fname in parts:
        X, L, Y = read_corpus(root / sub / fname, max_len=max_len)
        if dup_cap is not None:
            keep = _cap_duplicates(X, Y, dup_cap)
            print(f"    {sub}/{fname}: {len(Y):,} -> {keep.sum():,} after "
                  f"capping duplicates at {dup_cap}")
            X, L, Y = X[keep], L[keep], Y[keep]
        Xs.append(X)
        Ls.append(L)
        Ys.append(Y)
        Src.append(np.full(len(Y), sub))
    return (np.concatenate(Xs), np.concatenate(Ls), np.concatenate(Ys),
            np.concatenate(Src))


def _cap_duplicates(X, Y, cap: int):
    import collections
    seen = collections.Counter()
    keep = np.zeros(len(Y), dtype=bool)
    for i in range(len(Y)):
        key = (X[i].tobytes(), int(Y[i]))
        if seen[key] < cap:
            seen[key] += 1
            keep[i] = True
    return keep


def stratified_split(y, seed=SEED, fracs=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
        idx_tr.append(idx[:a])
        idx_va.append(idx[a:b])
        idx_te.append(idx[b:])
    return (np.concatenate(idx_tr), np.concatenate(idx_va),
            np.concatenate(idx_te))


def train_one(tag: str, cfg: dict, epochs: int = 14, lr: float = 3e-3):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, L, Y, S = load_split(cfg["root"], cfg["train_splits"], cfg["max_len"],
                            dup_cap=cfg.get("dup_cap"))
    tr, va, te = stratified_split(Y)
    print(f"[{tag}] corpus {len(Y):,}   train {len(tr):,}  val {len(va):,}  "
          f"test {len(te):,}   classes {np.bincount(Y)}")

    model = StrategyLSTM()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()
    loader = make_loader(X[tr], L[tr], Y[tr], batch=512, shuffle=True)

    hist = []
    best_va, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, lb, yb in loader:
            opt.zero_grad()
            out = model(xb, lb)
            loss = lossf(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            loss_sum += float(loss.detach()) * len(yb)
            correct += int((out.argmax(1) == yb).sum())
            tot += len(yb)
        sched.step()

        pv = predict(model, X[va], L[va])
        va_acc = float((pv.argmax(1) == Y[va]).mean())
        va_loss = float(-np.log(np.clip(pv[np.arange(len(va)), Y[va]], 1e-9, 1)).mean())
        hist.append({"epoch": ep, "train_loss": loss_sum / tot,
                     "train_acc": correct / tot, "val_loss": va_loss,
                     "val_acc": va_acc, "lr": sched.get_last_lr()[0],
                     "sec": time.time() - t0})
        print(f"  ep{ep:>2}  train {correct / tot:.4f}  val {va_acc:.4f}  "
              f"({time.time() - t0:.1f}s)")
        if va_acc > best_va:
            best_va, best_state = va_acc, {k: v.clone() for k, v in
                                           model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "tag": tag,
                "max_len": cfg["max_len"], "classes": STRATEGIES},
               MODELDIR / f"strategy_lstm_{tag}.pt")
    pd.DataFrame(hist).to_csv(TABDIR / f"T09_train_history_{tag}.csv", index=False)

    # ---- held-out test set -------------------------------------------------
    pt = predict(model, X[te], L[te])
    np.savez_compressed(DATADIR / f"clf_test_{tag}.npz",
                        proba=pt, y=Y[te], lens=L[te], src=S[te], X=X[te])
    print(f"[{tag}] test accuracy = {(pt.argmax(1) == Y[te]).mean():.4f}")

    # ---- unseen noise levels ------------------------------------------------
    ood = []
    for sub, fname in cfg["ood_splits"]:
        Xo, Lo, Yo = read_corpus(cfg["root"] / sub / fname, max_len=cfg["max_len"])
        po = predict(model, Xo, Lo)
        ood.append({"split": sub, "n": len(Yo),
                    "acc": float((po.argmax(1) == Yo).mean())})
        np.savez_compressed(DATADIR / f"clf_ood_{tag}_{sub}.npz",
                            proba=po, y=Yo, lens=Lo)
    if ood:
        pd.DataFrame(ood).to_csv(TABDIR / f"T10_ood_noise_{tag}.csv", index=False)
        print(f"[{tag}] OOD:", ood)

    # ---- an unseen strategy (GTFT) -----------------------------------------
    sub, fname = cfg["unseen_strategy"]
    Xg, Lg, Yg = _read_only_label(cfg["root"] / sub / fname, "GTFT", cfg["max_len"])
    if len(Xg):
        pg = predict(model, Xg, Lg)
        np.savez_compressed(DATADIR / f"clf_unseen_{tag}.npz", proba=pg, lens=Lg)
        print(f"[{tag}] GTFT ({len(Xg):,} seqs) is absorbed as:",
              dict(zip(STRATEGIES, np.bincount(pg.argmax(1), minlength=4)
                       / len(pg))))
    return model


def _read_only_label(path, label, max_len):
    from pdlib.seqcode import PAD, STOI
    seqs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or ":" not in line:
                continue
            lab, rest = line.split(":", 1)
            if lab.strip() != label:
                continue
            seqs.append([STOI[t] for t in rest.split()])
    if not seqs:
        return np.zeros((0, max_len), np.int64), np.zeros(0, np.int64), None
    X = np.full((len(seqs), max_len), PAD, dtype=np.int64)
    lens = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:max_len]
        X[i, :len(s)] = s
        lens[i] = len(s)
    return X, lens, None


def main():
    for tag, cfg in CORPUS.items():
        train_one(tag, cfg)
        print()


if __name__ == "__main__":
    main()
