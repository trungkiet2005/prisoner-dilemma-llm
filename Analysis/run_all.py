"""Regenerate every table and figure from the raw Dataset/ folder.

    python Analysis/run_all.py            # everything, including LSTM training
    python Analysis/run_all.py --no-train # reuse the saved checkpoints

Training the two classifiers takes about 8 minutes on CPU; everything else is
under two minutes.
"""
import argparse
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STEPS = [
    ("00_build_master.py", "parse the raw logs into tidy tables", False),
    ("01_fig_design.py", "F01 experimental design", False),
    ("02_fig_cooperation.py", "F02-F03 cooperation and payoff scale", False),
    ("03_fig_fairness.py", "F04-F05 language and persona", False),
    ("04_fig_dynamics.py", "F06-F09 dynamics, reciprocity, payoffs", False),
    ("05_train_classifier.py", "train the two strategy classifiers", True),
    ("06_fig_classifier.py", "F10-F13 corpus and classifier evaluation", False),
    ("07_fig_llm_strategy.py", "F14-F16 archetypes of LLM play", False),
    ("08_fig_synthesis.py", "F17-F20 signatures and invariances", False),
    ("09_fig_supplementary.py", "F21-F24 supplementary analyses", False),
    ("10_hybrid_eval.py", "F25-F26 hybrid rule+LSTM read-out", False),
    ("11_fig_residual.py", "F27-F29 mining the unexplained residual", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true",
                    help="skip LSTM training and reuse Analysis/models/*.pt")
    args = ap.parse_args()

    t_all = time.time()
    for script, desc, is_train in STEPS:
        if is_train and args.no_train:
            print(f"--- skipping {script} ({desc})")
            continue
        print(f"\n=== {script}  ·  {desc}")
        t0 = time.time()
        runpy.run_path(str(HERE / "scripts" / script), run_name="__main__")
        print(f"    done in {time.time() - t0:.1f}s")
    print(f"\nall steps finished in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
