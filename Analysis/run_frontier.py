"""Regenerate the frontier suite and the manuscript figures from Dataset/.

    python Analysis/run_frontier.py             # everything
    python Analysis/run_frontier.py --no-build  # reuse the parsed parquet

Two deliverables come out of this one pipeline.  Steps 20-30 write the
supplementary suite FR1-FR29 to `Analysis/figures/frontier/` together with
`Analysis/tables/T_FR*`.  Steps 33-34 then write `T_S*` and the six main-text
figures straight into `paper/figures/`, reading their panel numbers from those
tables so a main-text panel cannot drift from the table behind it.

Nothing is trained here: everything downstream of step 26 reads the strategy
classifier `models/strategy_lstm_h10.pt`, which `05_train_classifier.py`
produces, and the frontier games are 10 rounds so that checkpoint's horizon
matches exactly.

This is separate from `run_all.py` on purpose: that pipeline produces the
legacy F01-F34 figures over the three-model frontier arm plus the open-weight
arm, and its numbers should not shift because a fourth frontier model was
added.  Nothing here writes to the files `run_all.py` reads.
"""
import argparse
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NEEDS_CHECKPOINT = HERE / "models" / "strategy_lstm_h10.pt"

STEPS = [
    ("20_frontier_build.py", "parse the frontier arm into tidy tables", True),
    ("21_fig_frontier_core.py", "FR1-FR4 design, payoff scale, outcomes, language", False),
    ("22_fig_frontier_dynamics.py", "FR5-FR7 persona, round dynamics, reciprocity", False),
    ("23_fig_frontier_synthesis.py", "FR8 signature and variance decomposition", False),
    ("24_fig_frontier_profiles.py", "FR9-FR12 one model card per frontier LLM", False),
    ("25_fig_frontier_behaviour.py", "FR13-FR17 payoffs, bimodality, openings, audit", False),
    ("26_frontier_readout.py", "apply the strategy read-out to frontier play", True),
    ("27_fig_frontier_readout.py", "FR18-FR22 calibration, archetypes, rule distance", False),
    ("28_fig_frontier_residual.py", "FR23-FR26 the residual and the abstention set", False),
    ("29_fig_frontier_payoff_rules.py", "FR27-FR28 payoff space, distance to each rule", False),
    ("30_fig_frontier_latent.py", "FR29 what the read-out model has learned", False),
    ("33_strategy_stats.py", "T_S01-T_S22 statistics for the manuscript", False),
    ("34_fig_paper_strategy.py", "the six main-text figures -> paper/figures/", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true",
                    help="skip parsing and reuse Analysis/data/frontier_*.parquet")
    args = ap.parse_args()

    if not NEEDS_CHECKPOINT.exists():
        raise SystemExit(
            f"missing {NEEDS_CHECKPOINT.relative_to(HERE.parent)} -- FR18-FR26 "
            f"read the strategy classifier.\nRun "
            f"`python Analysis/scripts/05_train_classifier.py` first "
            f"(about 8 minutes on CPU).")

    t_all = time.time()
    for script, desc, is_build in STEPS:
        if is_build and args.no_build:
            print(f"--- skipping {script} ({desc})")
            continue
        print(f"\n=== {script}  |  {desc}")
        t0 = time.time()
        runpy.run_path(str(HERE / "scripts" / script), run_name="__main__")
        print(f"    done in {time.time() - t0:.1f}s")
    print(f"\nfrontier suite finished in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
