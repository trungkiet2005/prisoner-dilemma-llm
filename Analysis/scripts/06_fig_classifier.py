"""F10 -- the synthetic strategy corpus, F11 -- classifier evaluation,
F12 -- how many rounds it takes to identify a strategy, F13 -- latent space."""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

from pdlib.lstm import (StrategyLSTM, encode_hidden, predict, predict_prefixes)
from pdlib.seqcode import (ITOS, LTOI, STRATEGIES, VOCAB, read_corpus)
from pdlib.style import (CMAP_DIV, CMAP_SEQ, C_COOP, C_DEFECT, DATADIR,
                         DATASET, GRID, INK, INK2, MODELDIR, MUTED, STRATEGY,
                         STRATEGY_ORDER, SURFACE, TABDIR, panel_tag, savefig,
                         use_paper_style)

use_paper_style()

CORPORA = {
    "h10": (DATASET / "noise_dataset", 10,
            [("NoNoise", "nonoise"), ("Noise005", "noise005"),
             ("Noise01", "noise01"), ("Noise02", "noise02")]),
    "h30": (DATASET / "noise_dataset_30round", 30,
            [("NoNoise", "nonoise"), ("Noise005", "noise005")]),
}
NOISE_PCT = {"NoNoise": 0.0, "Noise005": 5.0, "Noise01": 10.0, "Noise02": 20.0}


def load_model(tag):
    ck = torch.load(MODELDIR / f"strategy_lstm_{tag}.pt", weights_only=False)
    m = StrategyLSTM()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck["max_len"]


# --------------------------------------------------------------------------
def fig_corpus():
    fig = plt.figure(figsize=(11.2, 7.0))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42)

    # (a) token distribution per strategy (10-round, 5% noise) ---------------
    ax = fig.add_subplot(gs[0, :2])
    path = DATASET / "noise_dataset" / "Noise005" / "4stratsAllCAllDTFTWSLS_noise005.txt"
    cnt = {s: collections.Counter() for s in STRATEGY_ORDER}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            lab, rest = line.split(":", 1)
            lab = lab.strip()
            if lab in cnt:
                cnt[lab].update(rest.split())
    toks = [t for t in VOCAB if t != "<pad>"]
    mat = np.array([[cnt[s][t] for t in toks] for s in STRATEGY_ORDER], float)
    mat = mat / mat.sum(axis=1, keepdims=True)
    im = ax.imshow(mat, cmap=CMAP_SEQ, aspect="auto", vmin=0, vmax=mat.max())
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0.004:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if mat[i, j] > 0.55 * mat.max() else INK)
    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels(toks)
    ax.set_yticks(range(len(STRATEGY_ORDER)))
    ax.set_yticklabels(STRATEGY_ORDER)
    ax.grid(False)
    ax.set_xlabel("token  =  ⟨outcome of previous round⟩⟨action this round⟩")
    ax.set_title("Token signature of each strategy")
    panel_tag(ax, "a", dx=-0.11)

    # (b) sequence-length distribution --------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    X, L, Y = read_corpus(DATASET / "noise_dataset" / "NoNoise" /
                          "4stratsAllCAllDTFTWSLS_nonoise.txt", max_len=10)
    vals, counts = np.unique(L, return_counts=True)
    ax.bar(vals, counts / counts.sum(), color=C_COOP, edgecolor=SURFACE,
           linewidth=1.0, width=0.75)
    ax.set_xlabel("observed rounds per trajectory")
    ax.set_ylabel("share")
    ax.set_title("10-round corpus is truncated")
    panel_tag(ax, "b", dx=-0.36)

    # (c) how much distinct information each split carries ------------------
    ax = fig.add_subplot(gs[1, :2])
    rows = []
    for tag, (root, ml, splits) in CORPORA.items():
        for sub, sfx in splits:
            f = root / sub / f"4stratsAllCAllDTFTWSLS_{sfx}.txt"
            d = collections.defaultdict(collections.Counter)
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    lab, rest = line.split(":", 1)
                    d[rest.strip()][lab.strip()] += 1
            n = sum(sum(c.values()) for c in d.values())
            rows.append({"corpus": tag, "split": sub, "noise": NOISE_PCT[sub],
                         "n_lines": n, "n_unique": len(d),
                         "bayes": sum(max(c.values()) for c in d.values()) / n})
    info = pd.DataFrame(rows)
    info.to_csv(TABDIR / "T11_corpus_information.csv", index=False)

    x = np.arange(len(info))
    ax.bar(x, info.n_unique, color=[C_COOP if c == "h10" else C_DEFECT
                                    for c in info.corpus],
           edgecolor=SURFACE, linewidth=1.0, width=0.66)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.corpus}\n{r.noise:g}% noise" for r in info.itertuples()],
                       fontsize=7)
    ax.set_ylabel("distinct trajectories  (log)")
    for xi, r in zip(x, info.itertuples()):
        ax.text(xi, r.n_unique * 1.25, f"{r.n_unique:,}", ha="center",
                fontsize=6.8, color=INK2)
    ax.set_ylim(4, 5e5)
    ax.legend(handles=[Patch(facecolor=C_COOP, label="10-round corpus"),
                       Patch(facecolor=C_DEFECT, label="30-round corpus")],
              ncol=2, loc="upper left", fontsize=7)
    ax.set_title("Distinct trajectories per split (161,280 lines each)")
    panel_tag(ax, "c", dx=-0.11)

    # (d) Bayes ceiling -----------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(x, info.bayes, color=[C_COOP if c == "h10" else C_DEFECT
                                 for c in info.corpus],
           edgecolor=SURFACE, linewidth=1.0, width=0.66)
    for xi, r in zip(x, info.itertuples()):
        ax.text(xi, r.bayes + 0.015, f"{r.bayes:.2f}", ha="center", fontsize=6.5,
                color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.noise:g}%" for r in info.itertuples()], fontsize=7)
    ax.set_xlabel("execution noise")
    ax.set_ylabel("Bayes-optimal accuracy")
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_title("Label ambiguity ceiling")
    panel_tag(ax, "d", dx=-0.40)

    fig.suptitle("The synthetic strategy corpus used to train the read-out model",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F10_corpus_overview")
    return info


# --------------------------------------------------------------------------
def fig_eval(info):
    fig = plt.figure(figsize=(11.2, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.68, wspace=0.48)

    # (a) learning curves ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for tag, col in (("h10", C_COOP), ("h30", C_DEFECT)):
        h = pd.read_csv(TABDIR / f"T09_train_history_{tag}.csv")
        ax.plot(h.epoch, h.train_acc, "-", color=col, label=f"{tag} train")
        ax.plot(h.epoch, h.val_acc, linestyle=(0, (3, 2)), color=col, marker="o", ms=3.5,
                mec=SURFACE, mew=0.8, label=f"{tag} validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.9, 1.0)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("Learning curves")
    panel_tag(ax, "a", dx=-0.30)

    # (b)(c) confusion matrices ----------------------------------------------
    for k, tag in enumerate(("h10", "h30")):
        ax = fig.add_subplot(gs[0, 1 + k])
        d = np.load(DATADIR / f"clf_test_{tag}.npz", allow_pickle=True)
        yhat = d["proba"].argmax(1)
        y = d["y"]
        cm = np.zeros((4, 4))
        for a, b in zip(y, yhat):
            cm[a, b] += 1
        cmn = cm / cm.sum(axis=1, keepdims=True)
        order = [LTOI[s] for s in STRATEGY_ORDER]
        cmn = cmn[np.ix_(order, order)]
        im = ax.imshow(cmn, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="equal")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cmn[i, j]:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if cmn[i, j] > 0.55 else INK)
        ax.set_xticks(range(4))
        ax.set_xticklabels(STRATEGY_ORDER, rotation=25, ha="right")
        ax.set_yticks(range(4))
        ax.set_yticklabels(STRATEGY_ORDER)
        ax.set_xlabel("predicted")
        if k == 0:
            ax.set_ylabel("true strategy")
        ax.grid(False)
        acc = (yhat == y).mean()
        ax.set_title(f"{'10' if tag == 'h10' else '30'}-round model · acc {acc:.3f}")
        panel_tag(ax, "bc"[k], dx=-0.34)

    # (d) per-class F1 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    w = 0.36
    for k, tag in enumerate(("h10", "h30")):
        d = np.load(DATADIR / f"clf_test_{tag}.npz", allow_pickle=True)
        yhat, y = d["proba"].argmax(1), d["y"]
        f1 = []
        for s in STRATEGY_ORDER:
            c = LTOI[s]
            tp = ((yhat == c) & (y == c)).sum()
            fp = ((yhat == c) & (y != c)).sum()
            fn = ((yhat != c) & (y == c)).sum()
            f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        xx = np.arange(4) + (k - 0.5) * w
        b = ax.bar(xx, f1, width=w * 0.9,
                   color=[STRATEGY[s] for s in STRATEGY_ORDER],
                   alpha=1.0 if k == 0 else 0.55, edgecolor=SURFACE, linewidth=1.0)
        ax.bar_label(b, fmt="%.3f", fontsize=5.6, padding=2, color=INK2,
                     rotation=90)
    ax.set_xticks(range(4))
    ax.set_xticklabels(STRATEGY_ORDER)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1")
    ax.text(0.0, 1.15, "solid = 10-round model, faded = 30-round",
            fontsize=6.3, color=MUTED)
    panel_tag(ax, "d", dx=-0.30)

    # (e) identifiability: accuracy vs number of observed rounds -------------
    ax = fig.add_subplot(gs[1, 1])
    ident_rows = []
    for tag, col, ml in (("h10", C_COOP, 10), ("h30", C_DEFECT, 30)):
        model, _ = load_model(tag)
        d = np.load(DATADIR / f"clf_test_{tag}.npz", allow_pickle=True)
        X, y, lens = d["X"], d["y"], d["lens"]
        sel = np.where(lens == ml)[0][:20000]
        pref = predict_prefixes(model, X[sel])
        acc = (pref.argmax(2) == y[sel][:, None]).mean(axis=0)
        ax.plot(np.arange(1, ml + 1), acc, "-o", color=col, ms=3.4, mec=SURFACE,
                mew=0.8, label=f"{ml}-round model")
        for s in STRATEGY_ORDER:
            c = LTOI[s]
            m = y[sel] == c
            ident_rows += [{"model": tag, "strategy": s, "round": r + 1,
                            "acc": float((pref[m, r].argmax(1) == c).mean())}
                           for r in range(ml)]
    ax.axhline(0.25, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(0.6, 0.27, "chance", fontsize=6.5, color=MUTED)
    ax.axhline(0.95, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.set_xlabel("rounds observed")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.2, 1.02)
    ax.legend(fontsize=6.8, loc="lower right")
    ax.set_title("Rounds needed to identify")
    panel_tag(ax, "e", dx=-0.30)
    ident = pd.DataFrame(ident_rows)
    ident.to_csv(TABDIR / "T12_identifiability.csv", index=False)

    # (f) robustness to unseen noise -----------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    pts = [(0.0, None), (5.0, None)]
    d = np.load(DATADIR / "clf_test_h10.npz", allow_pickle=True)
    yhat, y, src = d["proba"].argmax(1), d["y"], d["src"]
    seen = [(NOISE_PCT[s], float((yhat[src == s] == y[src == s]).mean()))
            for s in ["NoNoise", "Noise005"]]
    unseen = []
    for s in ["Noise01", "Noise02"]:
        o = np.load(DATADIR / f"clf_ood_h10_{s}.npz")
        unseen.append((NOISE_PCT[s], float((o["proba"].argmax(1) == o["y"]).mean())))
    allpts = seen + unseen
    bayes = info[info.corpus == "h10"].set_index("noise").bayes
    ax.plot([p[0] for p in allpts], [bayes.loc[p[0]] for p in allpts], "-",
            color=MUTED, lw=1.4, ls=(0, (4, 3)), label="Bayes ceiling")
    ax.plot([p[0] for p in seen], [p[1] for p in seen], "-o", color=C_COOP,
            ms=7, mec=SURFACE, mew=1.3, label="noise levels seen in training")
    ax.plot([p[0] for p in unseen], [p[1] for p in unseen], "--s", color=C_DEFECT,
            ms=7, mec=SURFACE, mew=1.3, label="never seen in training")
    ax.plot([seen[-1][0], unseen[0][0]], [seen[-1][1], unseen[0][1]], "--",
            color=C_DEFECT, lw=1.6)
    for xx, yy in allpts:
        ax.annotate(f"{yy:.3f}", (xx, yy), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=6.6, color=INK2)
    ax.set_xlabel("execution noise in the test data (%)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.72, 1.03)
    ax.legend(fontsize=6.3, loc="lower left")
    ax.set_title("Unseen noise levels")
    panel_tag(ax, "f", dx=-0.30)

    fig.suptitle("Reading a strategy off a trajectory: classifier behaviour",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F11_classifier_eval")
    return ident


# --------------------------------------------------------------------------
def fig_identifiability_detail(ident):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    for ax, tag, ml in zip(axes, ("h10", "h30"), (10, 30)):
        sub = ident[ident.model == tag]
        for s in STRATEGY_ORDER:
            d = sub[sub.strategy == s]
            ax.plot(d["round"], d.acc, "-o", color=STRATEGY[s], ms=3.4,
                    mec=SURFACE, mew=0.8, label=s)
        ax.axhline(0.25, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.set_xlabel("rounds observed")
        ax.set_ylabel("recall")
        ax.set_ylim(0, 1.03)
        ax.set_title(f"{ml}-round model")
        ax.legend(ncol=4, fontsize=6.8, loc="lower right")
    panel_tag(axes[0], "a", dx=-0.15)
    panel_tag(axes[1], "b", dx=-0.15)
    fig.suptitle("AllC and AllD are legible immediately; TFT and WSLS need "
                 "a provocation to separate", x=0.02, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig(fig, "F12_identifiability")


# --------------------------------------------------------------------------
def fig_latent():
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9))

    model, _ = load_model("h10")
    d = np.load(DATADIR / "clf_test_h10.npz", allow_pickle=True)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(d["y"]), 6000, replace=False)
    H = encode_hidden(model, d["X"][sel], d["lens"][sel])
    y = d["y"][sel]

    Hc = H - H.mean(0)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    Z = Hc @ Vt[:2].T
    var = (S ** 2 / (S ** 2).sum())[:2]

    ax = axes[0]
    for s in STRATEGY_ORDER:
        m = y == LTOI[s]
        ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.35, color=STRATEGY[s],
                   linewidths=0, label=s)
    ax.set_xlabel(f"PC1  ({var[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2  ({var[1]:.0%})")
    ax.set_title("Hidden state separates the rules")
    ax.legend(ncol=2, fontsize=6.8, markerscale=3.0, loc="best")
    ax.grid(True, axis="both")
    panel_tag(ax, "a", dx=-0.22)

    # confidence distribution
    ax = axes[1]
    conf = d["proba"].max(1)
    for s in STRATEGY_ORDER:
        m = d["y"] == LTOI[s]
        ax.hist(conf[m], bins=40, range=(0.25, 1.0), histtype="step", lw=1.8,
                color=STRATEGY[s], label=s, density=True)
    ax.set_xlabel("posterior probability of the predicted class")
    ax.set_ylabel("density")
    ax.set_title("Confidence is bimodal")
    ax.legend(ncol=2, fontsize=6.8)
    panel_tag(ax, "b", dx=-0.22)

    # what happens to an unseen strategy
    ax = axes[2]
    g = np.load(DATADIR / "clf_unseen_h10.npz")
    share = np.bincount(g["proba"].argmax(1), minlength=4) / len(g["proba"])
    order = [LTOI[s] for s in STRATEGY_ORDER]
    ax.bar(range(4), share[order], color=[STRATEGY[s] for s in STRATEGY_ORDER],
           edgecolor=SURFACE, linewidth=1.0, width=0.66)
    for i, v in enumerate(share[order]):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(STRATEGY_ORDER)
    ax.set_ylabel("share of GTFT trajectories")
    ax.set_ylim(0, 1.05)
    ax.set_title("Unseen rule GTFT folds into TFT")
    panel_tag(ax, "c", dx=-0.22)

    fig.suptitle("What the read-out model has learned", x=0.02, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig(fig, "F13_latent_space")


def main():
    info = fig_corpus()
    ident = fig_eval(info)
    fig_identifiability_detail(ident)
    fig_latent()
    print(info.to_string(index=False))


if __name__ == "__main__":
    main()
