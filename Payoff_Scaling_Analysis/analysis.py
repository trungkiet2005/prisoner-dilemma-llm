#!/usr/bin/env python
"""Reproducible end-to-end analysis of the frontier prisoner's-dilemma corpus.

Run from this directory:

    python analysis.py                 # full run, reusing the parsed cache
    python analysis.py --rebuild       # re-parse all 300 CSV files from scratch
    python analysis.py --only 04 07    # run selected sections only
    python analysis.py --report-only   # regenerate findings.md and analysis_report.md

What it does
------------
Parses `Dataset/data_fairgame_frontier_llm` once into three tidy tables (rounds,
agent-games, dyads), caches them as parquet, then runs ten analysis sections. Each
section writes its figures into its own numbered directory, its tables into
`tables/`, and appends to three shared registries:

    figure_inventory.csv     every figure, with the question it answers
    tables/statistical_tests.csv   every test, with FDR-adjusted p-values
    cache/findings.json      every recorded scientific finding

The registries are flushed serially here, after each section, which is why the
sections must not flush themselves.

Determinism
-----------
Every bootstrap and permutation takes an explicit seed, so a rerun reproduces the
numbers exactly. The only nondeterminism is UMAP, which is seeded but can differ
across library versions; nothing in the findings depends on it.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mining import core as C  # noqa: E402

SECTIONS = [
    ("01", "mining.m01_audit", "structural audit and data quality"),
    ("02", "mining.m02_distributions", "univariate distribution mining"),
    ("03", "mining.m03_payoff", "payoff-centric analysis"),
    ("04", "mining.m04_scaling", "payoff-scale invariance and scaling"),
    ("05", "mining.m05_fairness", "fairness and inequality"),
    ("06", "mining.m06_llm_comparison", "systematic model comparison"),
    ("07", "mining.m07_game_theory", "strategy and temporal dynamics"),
    ("08", "mining.m08_frontier", "Pareto and frontier analysis"),
    ("09", "mining.m09_robustness", "robustness and sensitivity"),
    ("10", "mining.m10_advanced", "correlation, interactions, discovery"),
    ("11", "mining.m11_structure", "dimensionality reduction and regimes"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuild", action="store_true",
                    help="re-parse the raw CSV corpus instead of using the cache")
    ap.add_argument("--only", nargs="+", metavar="NN",
                    help="run only these section numbers")
    ap.add_argument("--report-only", action="store_true",
                    help="skip the analysis and only rebuild the two markdown reports")
    ap.add_argument("--fresh-registries", action="store_true",
                    help="clear the figure/test/finding registries before running")
    args = ap.parse_args()

    if args.fresh_registries:
        for p in [C.OUT / "figure_inventory.csv", C.TABLES / "statistical_tests.csv",
                  C.CACHE / "findings.json"]:
            p.unlink(missing_ok=True)
        print("registries cleared")

    if not args.report_only:
        t0 = time.time()
        rounds, ag, dy = C.load(rebuild=args.rebuild)
        print(f"loaded in {time.time()-t0:.1f}s: "
              f"{len(rounds):,} rounds / {len(ag):,} agent-games / {len(dy):,} dyads")

        failed = []
        for num, mod, desc in SECTIONS:
            if args.only and num not in args.only:
                continue
            try:
                m = importlib.import_module(mod)
            except ModuleNotFoundError:
                print(f"[skip] {num} {desc}: {mod} not present")
                continue
            t = time.time()
            try:
                m.run(rounds, ag, dy)
            except Exception as exc:                      # noqa: BLE001
                failed.append((num, desc, repr(exc)))
                print(f"[FAIL] {num} {desc}: {exc!r}")
            finally:
                # flushed here, serially, so concurrent sections cannot race
                C.flush_inventory()
                C.flush_tests()
                C.flush_findings()
            print(f"  ({time.time()-t:.1f}s)")

        if failed:
            print("\nsections that failed:")
            for num, desc, exc in failed:
                print(f"  {num} {desc}: {exc}")

    from mining import report
    report.build()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
