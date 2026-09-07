"""Rebuild the current Interface Focus artifact from the released corpus.

This single entry point uses the current figure modules in ``Analysis/figures``
rather than the legacy ``s05`` drivers. It performs no API calls.
"""
from __future__ import annotations

import subprocess
import sys
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def run(script: Path) -> None:
    print(f"\n>>> {script.relative_to(REPO)}", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-egt", action="store_true",
        help="reuse committed EGT outputs when egttools is unavailable",
    )
    args = parser.parse_args()
    scaling = HERE
    figures = REPO / "Analysis" / "figures"
    for name in ("s00_build.py", "s01_train_lstm.py", "s02_readout.py",
                 "s03_stats.py", "s04_strategy_stats.py"):
        run(scaling / name)
    for name in ("s03_matched_block_robustness.py", "s12_conditioning_ci.py",
                 "s13_review_robustness.py"):
        run(scaling / name)
    if not args.skip_egt:
        for name in ("s09_egt.py", "s10_egt_robustness.py"):
            run(scaling / name)
    else:
        print("\n>>> reusing committed EGT outputs (--skip-egt)", flush=True)
    figure_names = ("fig_overview.py", "fig_landscape.py", "fig_language.py",
                 "fig_persona.py", "fig_strategy_mix.py", "fig_firstmove.py",
                 "fig_egt_vs_llm.py", "fig_strategy_space.py",
                 "fig_robustness.py", "fig_fairness_welfare.py")
    if not args.skip_egt:
        figure_names += ("fig_invasion.py", "fig_simplex.py")
    for name in figure_names:
        run(figures / name)
    for name in ("s06_tables.py", "s07_supplementary.py"):
        run(scaling / name)
    run(scaling / "s05b_appendix_figures.py")
    run(scaling / "s08_verify_paper.py")
    print("\nInterface Focus rebuild completed. Build appendix.tex, then main.tex.")


if __name__ == "__main__":
    main()
