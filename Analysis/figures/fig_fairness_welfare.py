"""Figure B.4: scale-normalised outcome fairness and dyad welfare."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scaling"))
from fairness_metrics import compute, headline_metrics  # noqa: E402
import style as S  # noqa: E402


def main() -> None:
    games = pd.read_parquet(
        Path(__file__).resolve().parents[2] / "Analysis" / "scaling" / "data" / "games.parquet"
    )
    d = compute(games)
    h = headline_metrics(d)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(S.FULL, 3.15))

    for model in S.MODEL_ORDER_ALL:
        q = d[d.model == model]
        ax_a.scatter(q.welfare, q.fairness, s=4, alpha=.20,
                     color=S.MODEL_C.get(model, "#777777"), label=S.MODEL_LABEL.get(model, model))
    ax_a.set(xlabel="dyad welfare (normalised utility)", ylabel="fairness (1 - Gini)")
    ax_a.set_xlim(.37, .83); ax_a.set_ylim(-.05, 1.05)
    ax_a.set_title("a  equality and welfare")

    bins = np.linspace(d.welfare.min(), d.welfare.max(), 9)
    d["wbin"] = pd.cut(d.welfare, bins=bins, include_lowest=True)
    b = d.groupby("wbin", observed=True).agg(welfare=("welfare", "mean"), fairness=("fairness", "mean"))
    ax_b.plot(b.welfare, b.fairness, "o-", color=S.INK, lw=1.1, ms=3)
    ax_b.set(xlabel="welfare bin centre", ylabel="mean fairness")
    ax_b.set_ylim(.45, 1.03); ax_b.set_title("b  binned frontier")

    means = d.groupby(["model", "scale"], as_index=False).fairness.mean()
    for model in S.MODEL_ORDER_ALL:
        q = means[means.model == model]
        ax_c.plot(q.scale, q.fairness, marker="o", ms=2.2, lw=.8,
                  color=S.MODEL_C.get(model, "#777777"), label=S.MODEL_SHORT.get(model, model))
    ax_c.set_xscale("log"); ax_c.set(xlabel="payoff scale", ylabel="mean fairness")
    ax_c.set_ylim(.45, 1.03); ax_c.set_title("c  scale-normalised fairness")

    fig.suptitle(f"zero-gap share {h['zero_gap']:.1%}; gap--exploitation r = {h['gap_exploit_r']:.3f}; "
                 f"scale slope = {h['slope']:.4f} (p = {h['p_slope']:.2f})", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, .93))
    S.save(fig, "fa7_fairness_welfare")


if __name__ == "__main__":
    main()
