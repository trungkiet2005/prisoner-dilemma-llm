"""Parse failures, retries and fallbacks by model, language and payoff scale.

The per-game CSVs in Dataset/ do not carry these counts, but the per-game JSON
checkpoints that the Kaggle Benchmarks task writes do, one record per decision:
`parse_failed` counts the attempts the matcher could not resolve (the first
attempt plus up to two retries) and `fell_back` marks a decision recorded as
Option A because all three failed. This script tabulates every checkpoint of a
main-grid run it can find under results/kbench, so the table grows as further
runs are downloaded with kaggle/benchmarks/run_and_watch.py.

Repeat runs (model tag ending in -rep2) and smoke runs are excluded: only runs
whose games were promoted into Dataset/data_fairgame_frontier_llm count.

Writes tables/T27_parse_fallback.csv and tables/T27_parse_fallback_summary.csv.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KBENCH = REPO / "results" / "kbench"
TAB = HERE / "tables"


def main_grid_runs():
    for d in sorted(KBENCH.glob("exp*/run*/*/checkpoints")):
        tag = d.parent.name
        # repeat runs, and the gemini-3-flash-preview smoke runs, are not in Dataset/
        if tag.endswith("-rep2") or tag.startswith("gemini-3-flash-preview"):
            continue
        yield tag, d


def main():
    acc = defaultdict(lambda: {"decisions": 0, "failed_attempts": 0,
                               "decisions_with_retry": 0, "fallbacks": 0})
    for tag, d in main_grid_runs():
        for f in d.glob("*.json"):
            rec = json.loads(f.read_text(encoding="utf-8"))
            for t in rec["turns"]:
                k = (tag, t["language"], float(t["scale"]))
                acc[k]["decisions"] += 1
                acc[k]["failed_attempts"] += int(t["parse_failed"])
                acc[k]["decisions_with_retry"] += int(t["parse_failed"] > 0)
                acc[k]["fallbacks"] += int(t["fell_back"])
    rows = [{"model": m, "language": l, "scale": s, **v} for (m, l, s), v in acc.items()]
    t = pd.DataFrame(rows).sort_values(["model", "language", "scale"])
    t["retry_rate"] = t.decisions_with_retry / t.decisions
    t["fallback_rate"] = t.fallbacks / t.decisions
    t.to_csv(TAB / "T27_parse_fallback.csv", index=False)
    s = (t.groupby("model")[["decisions", "failed_attempts", "decisions_with_retry", "fallbacks"]]
         .sum().assign(n_cells=t.groupby("model").size()))
    s["max_cell_fallback_rate"] = t.groupby("model").fallback_rate.max()
    s.to_csv(TAB / "T27_parse_fallback_summary.csv")
    print(s.to_string())


if __name__ == "__main__":
    main()
