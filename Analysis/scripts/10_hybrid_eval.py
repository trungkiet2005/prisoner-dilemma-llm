"""Evaluate the hybrid read-out: exact rules first, LSTM only where rules fail.

Three answers are compared on every split of both corpora:

  LSTM        the learned classifier alone, forced to name one strategy
  rules       exact rule matching, allowed to return a *set* of strategies
  hybrid      rules decide the candidate set; the LSTM ranks inside it, and
              takes over entirely when no rule fits

The single-label Bayes ceiling that caps the LSTM is a consequence of forcing
one answer where several are equally true.  The set-valued rule output is not
bound by it, which is the point of the comparison.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

from pdlib.lstm import StrategyLSTM, predict
from pdlib.rulebase import (STRATEGIES, consistent_mask, deviation_counts,
                            hybrid_predict, label_sets, set_name)
from pdlib.seqcode import LTOI, read_corpus
from pdlib.style import (CMAP_SEQ, C_COOP, C_DEFECT, DATADIR, DATASET, INK,
                         INK2, MODELDIR, MUTED, STRATEGY, STRATEGY_ORDER,
                         SURFACE, TABDIR, panel_tag, savefig, use_paper_style)

use_paper_style()

SPLITS = [
    ("h10", 10, DATASET / "noise_dataset", "NoNoise", "nonoise", 0.0, True),
    ("h10", 10, DATASET / "noise_dataset", "Noise005", "noise005", 5.0, True),
    ("h10", 10, DATASET / "noise_dataset", "Noise01", "noise01", 10.0, False),
    ("h10", 10, DATASET / "noise_dataset", "Noise02", "noise02", 20.0, False),
    ("h30", 30, DATASET / "noise_dataset_30round", "NoNoise", "nonoise", 0.0, True),
    ("h30", 30, DATASET / "noise_dataset_30round", "Noise005", "noise005", 5.0, True),
]

RULE_COLOR = "#4a3aa7"
LSTM_COLOR = "#eb6834"
HYBRID_COLOR = "#1baf7a"


def load_model(tag):
    ck = torch.load(MODELDIR / f"strategy_lstm_{tag}.pt", weights_only=False)
    m = StrategyLSTM()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def bayes_ceiling(X, y):
    """Best possible single-label accuracy: predict the majority label of every
    group of identical trajectories."""
    groups = collections.defaultdict(collections.Counter)
    for i in range(len(y)):
        groups[X[i].tobytes()][int(y[i])] += 1
    return sum(max(c.values()) for c in groups.values()) / len(y)


def evaluate():
    rows, setrows = [], []
    models = {tag: load_model(tag) for tag in ("h10", "h30")}

    for tag, ml, root, sub, sfx, noise, seen in SPLITS:
        X, L, y = read_corpus(root / sub / f"4stratsAllCAllDTFTWSLS_{sfx}.txt",
                              max_len=ml)
        proba = predict(models[tag], X, L)
        mask = consistent_mask(X)
        sets, single, source = hybrid_predict(X, proba)

        n_fit = mask.sum(axis=1)
        true_in_set = mask[np.arange(len(y)), y]

        rows.append({
            "corpus": tag, "split": sub, "noise": noise,
            "seen_in_training": seen, "n": len(y),
            "bayes_single_label": bayes_ceiling(X, y),
            "lstm_acc": float((proba.argmax(1) == y).mean()),
            "hybrid_acc": float((single == y).mean()),
            "rule_coverage": float((n_fit > 0).mean()),
            "rule_unique": float((n_fit == 1).mean()),
            "rule_multi": float((n_fit > 1).mean()),
            "rule_none": float((n_fit == 0).mean()),
            "mean_set_size": float(n_fit[n_fit > 0].mean()) if (n_fit > 0).any() else np.nan,
            # set-valued correctness, only where a rule fired
            "rule_hit_rate": float(true_in_set[n_fit > 0].mean()) if (n_fit > 0).any() else np.nan,
            # set-valued correctness over the whole split (empty set counts as a miss)
            "rule_hit_rate_all": float(true_in_set.mean()),
            # correctness of the *hybrid answer*, which is a set where a rule
            # fires and the LSTM's single label otherwise -- the fair thing to
            # put beside the single-label ceiling
            "hybrid_hit_rate": float(np.where(n_fit > 0, true_in_set,
                                              proba.argmax(1) == y).mean()),
            "mean_answer_size": float(np.where(n_fit > 0, n_fit, 1).mean()),
            # accuracy of each branch of the hybrid
            "acc_where_rule_unique": float((single == y)[n_fit == 1].mean()) if (n_fit == 1).any() else np.nan,
            "acc_where_rule_multi": float((single == y)[n_fit > 1].mean()) if (n_fit > 1).any() else np.nan,
            "acc_where_lstm": float((single == y)[n_fit == 0].mean()) if (n_fit == 0).any() else np.nan,
            "lstm_acc_where_rule_none": float((proba.argmax(1) == y)[n_fit == 0].mean()) if (n_fit == 0).any() else np.nan,
        })

        cnt = collections.Counter(set_name(s) for s in sets)
        for name, c in cnt.most_common():
            setrows.append({"corpus": tag, "split": sub, "label_set": name,
                            "n": c, "share": c / len(sets)})

    res = pd.DataFrame(rows)
    sets_df = pd.DataFrame(setrows)
    res.to_csv(TABDIR / "T17_hybrid_evaluation.csv", index=False)
    sets_df.to_csv(TABDIR / "T18_label_set_distribution.csv", index=False)
    return res, sets_df


def fig_hybrid(res, sets_df):
    fig = plt.figure(figsize=(11.2, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.78, wspace=0.46)

    lab = [f"{r.corpus}\n{r.noise:g}%" for r in res.itertuples()]
    x = np.arange(len(res))

    # (a) single-label accuracy vs the ceiling -------------------------------
    ax = fig.add_subplot(gs[0, :2])
    w = 0.34
    ax.bar(x - w / 2, res.lstm_acc, width=w * 0.94, color=LSTM_COLOR,
           edgecolor=SURFACE, linewidth=1.0, label="LSTM alone")
    ax.bar(x + w / 2, res.hybrid_acc, width=w * 0.94, color=HYBRID_COLOR,
           edgecolor=SURFACE, linewidth=1.0, label="hybrid (rules ∩ LSTM)")
    for xi, r in zip(x, res.itertuples()):
        ax.plot([xi - w, xi + w], [r.bayes_single_label] * 2, color=INK, lw=1.6,
                zorder=5)
        ax.text(xi, r.hybrid_acc + 0.015, f"{r.hybrid_acc:.3f}", ha="center",
                fontsize=6.4, color=INK2)
    ax.plot([], [], color=INK, lw=1.6, label="single-label Bayes ceiling")
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylim(0.5, 1.06)
    ax.set_ylabel("single-label accuracy")
    ax.set_xlabel("corpus and execution noise  (10% and 20% never seen in training)")
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Forcing one answer: both methods sit at the ceiling")
    panel_tag(ax, "a", dx=-0.085)

    # (b) set-valued correctness escapes the ceiling -------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(x, res.hybrid_hit_rate, color=RULE_COLOR, edgecolor=SURFACE,
           linewidth=1.0, width=0.66, label="true label ∈ hybrid answer")
    ax.plot(x, res.bayes_single_label, "o", color=INK, ms=6, mec=SURFACE,
            mew=1.2, label="single-label ceiling")
    for xi, r in zip(x, res.itertuples()):
        ax.text(xi, r.hybrid_hit_rate + 0.025, f"{r.hybrid_hit_rate:.3f}",
                ha="center", fontsize=6.0, color=INK2)
        ax.text(xi, 0.03, "|answer|\n" + f"{r.mean_answer_size:.2f}",
                ha="center", fontsize=5.6, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=6.5)
    ax.set_ylim(0, 1.24)
    ax.set_ylabel("share of trajectories")
    ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    ax.set_title("Allowing a set escapes the ceiling")
    panel_tag(ax, "b", dx=-0.30)

    # (c) who answers -- rules or the network --------------------------------
    ax = fig.add_subplot(gs[1, 0])
    bottom = np.zeros(len(res))
    for key, col, name in (("rule_unique", RULE_COLOR, "rule, one strategy"),
                           ("rule_multi", "#9085e9", "rule, several"),
                           ("rule_none", MUTED, "no rule fits → LSTM")):
        v = res[key].to_numpy()
        ax.bar(x, v, bottom=bottom, color=col, edgecolor=SURFACE, linewidth=1.1,
               width=0.66, label=name)
        for xi, (b, q) in enumerate(zip(bottom, v)):
            if q > 0.11:
                ax.text(xi, b + q / 2, f"{q:.2f}", ha="center", va="center",
                        fontsize=6.0, color="white", fontweight="semibold")
        bottom = bottom + v
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of trajectories")
    ax.grid(False)
    ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Division of labour")
    panel_tag(ax, "c", dx=-0.28)

    # (d) accuracy of each branch --------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    w = 0.27
    for k, (key, col, name) in enumerate((
            ("acc_where_rule_unique", RULE_COLOR, "rule unique"),
            ("acc_where_rule_multi", "#9085e9", "rule set"),
            ("acc_where_lstm", MUTED, "LSTM fallback"))):
        ax.bar(x + (k - 1) * w, res[key], width=w * 0.9, color=col,
               edgecolor=SURFACE, linewidth=0.9, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("single-label accuracy")
    ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3)
    ax.set_title("Accuracy of each branch")
    panel_tag(ax, "d", dx=-0.28)

    # (e) which label sets occur ---------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    key = sets_df[sets_df.split.isin(["NoNoise", "Noise005"])]
    piv = (key.groupby("label_set").n.sum().sort_values(ascending=False)
           .head(8))
    piv = piv / key.n.sum()
    y = np.arange(len(piv))[::-1]
    cols = []
    for name in piv.index:
        if name == "none":
            cols.append(MUTED)
        elif "+" in name:
            cols.append("#9085e9")
        else:
            cols.append(STRATEGY[name])
    ax.barh(y, piv.to_numpy(), color=cols, edgecolor=SURFACE, linewidth=1.0,
            height=0.66)
    for yi, (name, v) in zip(y, piv.items()):
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xlim(0, float(piv.max()) * 1.32)
    ax.set_xlabel("share of trajectories")
    ax.set_title("Label sets returned")
    ax.grid(True, axis="x")
    panel_tag(ax, "e", dx=-0.52)

    fig.suptitle("Rules first, network second: what the hybrid read-out buys",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F25_hybrid_evaluation")


def fig_rule_survival():
    """How long an LLM's play keeps being *exactly* consistent with each rule.

    A trajectory starts consistent with every rule that opens the same way and
    drops out the first round it deviates.  The curves are therefore monotone
    and need no model at all -- they are a direct, falsifiable statement about
    how canonical the play is.
    """
    from pdlib.rulebase import TOKEN_OK

    rounds = pd.read_parquet(DATADIR / "rounds.parquet")
    arche = pd.read_parquet(DATADIR / "llm_archetypes.parquet")
    key2model = arche.set_index(["game_uid", "agent"]).model.to_dict()

    from pdlib.seqcode import PAD, STOI
    from pdlib.style import MODEL, MODEL_ORDER

    fig = plt.figure(figsize=(11.2, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.40)

    surv = {}
    for k, (fam, ml) in enumerate((("frontier", 10), ("small", 30))):
        sub = rounds[rounds.family == fam].sort_values(
            ["game_uid", "agent", "round"])
        g = sub.groupby(["game_uid", "agent"], sort=False).token.apply(list)
        X = np.full((len(g), ml), PAD, dtype=np.int64)
        for i, toks in enumerate(g):
            t = [STOI[x] for x in toks][:ml]
            X[i, :len(t)] = t
        ok = TOKEN_OK[:, X]                       # (n_strat, N, L)
        alive = np.cumprod(ok, axis=2)            # still consistent after t
        surv[fam] = (np.array([key2model[kk] for kk in g.index]), alive)

        ax = fig.add_subplot(gs[0, k])
        for si, s in enumerate(STRATEGIES):
            ax.plot(np.arange(1, ml + 1), alive[si].mean(axis=0), "-o",
                    color=STRATEGY[s], ms=3.2, mec=SURFACE, mew=0.8, label=s)
        ax.plot(np.arange(1, ml + 1), alive.any(axis=0).mean(axis=0),
                color=INK, lw=1.4, ls=(0, (4, 3)), label="any rule")
        ax.set_xlabel("rounds observed")
        ax.set_ylabel("share still exactly consistent")
        ax.set_ylim(0, 1)
        ax.set_xlim(1, ml)
        ax.set_title(f"{'Frontier' if fam == 'frontier' else 'Open-weight'} play")
        if k == 0:
            ax.legend(ncol=5, fontsize=6.5, loc="upper center",
                      bbox_to_anchor=(0.5, 1.20))
        panel_tag(ax, "ab"[k], dx=-0.26)

    # (c) per model, survival of "any rule" ----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    for fam, ml, ls in (("frontier", 10, (0, (3, 2))), ("small", 30, "-")):
        mdls, alive = surv[fam]
        anyrule = alive.any(axis=0)
        for mdl in MODEL_ORDER:
            m = mdls == mdl
            if not m.any():
                continue
            ax.plot(np.arange(1, ml + 1), anyrule[m].mean(axis=0),
                    linestyle=ls, color=MODEL[mdl], lw=1.8,
                    label=mdl if fam == "small" or mdl in ("Claude-3.5-Haiku",
                                                           "GPT-4o",
                                                           "Mistral-Large")
                    else None)
    ax.set_xlabel("rounds observed")
    ax.set_ylabel("share consistent with any rule")
    ax.set_ylim(0, 1)
    ax.set_title("Consistency with any rule")
    ax.text(0.0, 1.02, "dashed = frontier, solid = open-weight",
            transform=ax.transAxes, fontsize=6.4, color=MUTED, va="bottom")
    ax.legend(ncol=2, fontsize=5.8, loc="upper right")
    panel_tag(ax, "c", dx=-0.30, dy=1.12)

    # (d) distance to the nearest rule ---------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    parts = [arche.loc[arche.model == m, "min_deviations"].to_numpy()
             for m in MODEL_ORDER]
    vp = ax.violinplot(parts, showextrema=False, widths=0.85)
    for b, mdl in zip(vp["bodies"], MODEL_ORDER):
        b.set_facecolor(MODEL[mdl])
        b.set_alpha(0.8)
        b.set_edgecolor(SURFACE)
    for i, p in enumerate(parts):
        ax.plot(i + 1, np.median(p), "o", ms=5, color=INK, mec=SURFACE, mew=1.0)
        ax.text(i + 1, np.percentile(p, 97) + 0.4, f"mean {p.mean():.1f}",
                ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
    ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylabel("rounds that violate the nearest rule")
    ax.set_title("How far the play is from *any* canonical rule")
    panel_tag(ax, "d", dx=-0.075)

    # (e) assignment mix per model -------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    mix = (arche.groupby("model").assignment.value_counts(normalize=True)
           .unstack().reindex(index=MODEL_ORDER,
                              columns=["exact", "ambiguous", "approx"])
           .fillna(0))
    bottom = np.zeros(len(mix))
    for col, c, name in (("exact", "#4a3aa7", "exact rule"),
                         ("ambiguous", "#9085e9", "several rules"),
                         ("approx", MUTED, "no rule → LSTM")):
        v = mix[col].to_numpy()
        ax.bar(np.arange(len(mix)), v, bottom=bottom, color=c,
               edgecolor=SURFACE, linewidth=1.1, width=0.66, label=name)
        for xi, (b, q) in enumerate(zip(bottom, v)):
            if q > 0.10:
                ax.text(xi, b + q / 2, f"{q:.2f}", ha="center", va="center",
                        fontsize=6.2, color="white", fontweight="semibold")
        bottom = bottom + v
    ax.set_xticks(range(len(mix)))
    ax.set_xticklabels([m.split("-")[0] for m in mix.index], rotation=25,
                       ha="right")
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.26),
              ncol=1)
    ax.set_title("Who answers, per model")
    panel_tag(ax, "e", dx=-0.32)

    mix.round(3).to_csv(TABDIR / "T19_llm_assignment_mix.csv")
    fig.suptitle("How canonical is LLM play, without asking a model?", x=0.02,
                 ha="left", fontweight="bold", color=INK)
    savefig(fig, "F26_rule_survival")
    return mix


def main():
    res, sets_df = evaluate()
    cols = ["corpus", "split", "noise", "seen_in_training", "n",
            "bayes_single_label", "lstm_acc", "hybrid_acc", "hybrid_hit_rate",
            "mean_answer_size", "rule_coverage", "rule_unique", "rule_multi",
            "rule_none"]
    print(res[cols].round(4).to_string(index=False))
    print()
    print(res[["corpus", "split", "acc_where_rule_unique", "acc_where_rule_multi",
               "acc_where_lstm", "lstm_acc_where_rule_none"]].round(4)
          .to_string(index=False))
    fig_hybrid(res, sets_df)
    mix = fig_rule_survival()
    print()
    print(mix.round(3).to_string())


if __name__ == "__main__":
    main()
