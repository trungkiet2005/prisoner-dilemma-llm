"""FR29: what the read-out model has learned, and what it does with a rule it
has never seen.

FR18 asks whether the instrument is accurate.  This figure asks what it is
accurate *about*: whether the hidden state carries a separable representation
of the four rules, how confident it is when it is right, and -- the part that
matters most for reading LLM play -- where a strategy outside its vocabulary
gets filed.

Generous tit-for-tat is the test case.  GTFT is TFT that forgives a defection
with some probability, so it is a genuine fifth strategy the network was never
trained on.  If an unseen rule folded into an arbitrary label, every archetype
in FR19 would be suspect; if it folds into its nearest true neighbour, the
labels degrade gracefully and "nearest canonical rule" means what it says.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdlib.lstm import encode_hidden
from pdlib.natstyle import (DATADIR, INK, INK2, MUTED, PAGE, RULE, SPINE,
                            TABDIR, W2, bars, caption, figure, finalize, hgrid,
                            save, use_journal_style)
from pdlib.seqcode import LTOI, STRATEGIES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

use_journal_style()

SEED = 0
N_POINTS = 6000

# same five-rule palette as FR19, minus the "several fit" grey
RULE_ORDER = ["AllC", "TFT", "WSLS", "AllD"]
RULE_C = {"AllC": "#0072b2", "TFT": "#56b4e9", "WSLS": "#009e73",
          "AllD": "#d55e00"}
RULE_M = {"AllC": "o", "TFT": "s", "WSLS": "^", "AllD": "D"}


def fig_latent():
    readout = import_module("26_frontier_readout")
    model = readout.load_readout()
    d = np.load(DATADIR / "clf_test_h10.npz", allow_pickle=True)

    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(d["y"]), min(N_POINTS, len(d["y"])), replace=False)
    H = encode_hidden(model, d["X"][sel], d["lens"][sel])
    y = d["y"][sel]

    # PCA by SVD on the centred hidden states
    Hc = H - H.mean(0)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    Z = Hc @ Vt[:2].T
    var = (S ** 2 / (S ** 2).sum())[:2]

    fig = figure(W2, 2.55)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.05, 0.95])

    # (a) the hidden state ----------------------------------------------------
    ax = pa = fig.add_subplot(gs[0, 0])
    hgrid(ax, axis="both")
    for s in RULE_ORDER:
        m = y == LTOI[s]
        ax.plot(Z[m, 0], Z[m, 1], "o", ms=1.5, mfc=RULE_C[s], mec="none",
                alpha=0.35, ls="none", zorder=3, label=s)
    ax.set_xlabel(f"PC1  ({var[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2  ({var[1]:.0%})")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10),
              handlelength=0.6, columnspacing=0.7, markerscale=3.2)
    ax.set_title("The hidden state separates the rules", pad=16)

    # (b) confidence, per true class -----------------------------------------
    ax = pb = fig.add_subplot(gs[0, 1])
    hgrid(ax)
    conf = d["proba"].max(1)
    rows = []
    for s in RULE_ORDER:
        m = d["y"] == LTOI[s]
        ax.hist(conf[m], bins=np.linspace(0.25, 1.0, 41), histtype="step",
                lw=1.1, color=RULE_C[s], density=True, zorder=4, label=s)
        rows.append({"strategy": s, "n": int(m.sum()),
                     "median_confidence": float(np.median(conf[m])),
                     "share_above_0.95": float((conf[m] >= 0.95).mean())})
    confd = pd.DataFrame(rows)
    confd.to_csv(TABDIR / "T_FR47_confidence_by_class.csv", index=False)
    # Log density, because the mass at 1.0 is ~50x anything else: on a linear
    # axis the spike is the only visible feature and the tail -- which is the
    # part that matters -- disappears into the baseline.
    ax.set_yscale("log")
    ax.set_xlabel("posterior of the predicted class")
    ax.set_ylabel("density (log scale)")
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(1e-3, 2e2)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10),
              handlelength=1.0, columnspacing=0.7)
    ax.set_title("Almost always certain, on synthetic play", pad=16)

    # (c) an unseen rule ------------------------------------------------------
    ax = pc = fig.add_subplot(gs[0, 2])
    hgrid(ax)
    g = np.load(DATADIR / "clf_unseen_h10.npz", allow_pickle=True)
    pred = g["proba"].argmax(1)
    share = np.array([(pred == LTOI[s]).mean() for s in RULE_ORDER])
    pd.DataFrame({"strategy": RULE_ORDER, "share": share,
                  "n": len(pred)}).to_csv(
        TABDIR / "T_FR48_unseen_gtft.csv", index=False)

    x = np.arange(len(RULE_ORDER))
    bars(ax, x, share, [RULE_C[s] for s in RULE_ORDER], width=0.62)
    for xi, v in zip(x, share):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                fontsize=6.0, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(RULE_ORDER, fontsize=6.2)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("share of GTFT trajectories")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_title("Unseen rule GTFT folds into TFT", pad=16)

    finalize(fig, [pa, pb, pc], ["a", "b", "c"])
    caption(fig, f"Synthetic play only. Panel a is a PCA of the LSTM's final "
                 f"hidden state on {len(sel):,} held-out trajectories, coloured "
                 f"by the strategy that actually generated them -- the network "
                 f"was never asked to make them separable, so the four clusters "
                 f"are a property of the representation. In b the median "
                 f"posterior is 1.00 for all four classes and "
                 f"{confd['share_above_0.95'].min():.0%}-"
                 f"{confd['share_above_0.95'].max():.0%} of trajectories sit "
                 f"above 0.95; that certainty is earned on synthetic play and "
                 f"does not transfer to LLM transcripts, where a fifth of games "
                 f"never reach 0.90 (FR21b). Panel c feeds it "
                 f"generous tit-for-tat, a fifth rule absent from training: "
                 f"{share[RULE_ORDER.index('TFT')]:.0%} is filed as TFT, its "
                 f"nearest true neighbour, and none as AllD. An unseen rule "
                 f"therefore degrades towards its neighbour rather than "
                 f"scattering, which is what makes 'nearest canonical rule' a "
                 f"usable label in FR19.")
    save(fig, "FR29_latent_space")
    return confd, share


def main():
    confd, share = fig_latent()
    print(confd.round(3).to_string(index=False))
    print()
    print(pd.Series(share, index=RULE_ORDER, name="GTFT filed as").round(3)
          .to_string())


if __name__ == "__main__":
    main()
