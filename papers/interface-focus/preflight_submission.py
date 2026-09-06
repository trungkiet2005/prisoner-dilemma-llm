#!/usr/bin/env python3
"""Pre-submission checks for the Interface Focus manuscript.

Run from papers/interface-focus:
    python preflight_submission.py

This script deliberately fails while author-owned placeholders remain. It uses
only the Python standard library and does not rebuild statistics or figures.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAIN = HERE / "main.tex"
APP = HERE / "appendix.tex"
FIGDIR = HERE / "figures"
STYLE = ROOT / "Analysis" / "figures" / "style.py"

EXPECTED_MAIN = [
    "figures/f_overview.pdf",
    "figures/f_landscape.pdf",
    "figures/f_language.pdf",
    "figures/f_persona.pdf",
    "figures/f_strategy_mix.pdf",
    "figures/f_firstmove.pdf",
    "figures/f_simplex.pdf",
    "figures/f_invasion.pdf",
    "figures/f_egt_vs_llm.pdf",
    "figures/f_robustness.pdf",
    "figures/f_strategy_space.pdf",
]
EXPECTED_APP = [
    "figures/fa4_classifier.pdf",
    "figures/fa3_supp_model.pdf",
    "figures/fa2_egt_vs_llm.pdf",
    "figures/fa1_egt_grid.pdf",
]
LEGACY_MAIN = [f"figures/f{i}_{name}.pdf" for i, name in [
    (1, "scaling"), (2, "inference"), (3, "language"), (4, "heterogeneity"),
    (5, "persona"), (6, "ruleforce"), (7, "strategy_mix"), (8, "firstmove"),
]]

errors: list[str] = []
warnings: list[str] = []

main = MAIN.read_text(encoding="utf-8")
app = APP.read_text(encoding="utf-8")

def included(tex: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)

def labels(tex: str) -> set[str]:
    return set(re.findall(r"\\label\{([^}]+)\}", tex))

main_figs = included(main)
app_figs = included(app)

if main_figs != EXPECTED_MAIN:
    errors.append(
        "main figure inventory differs from the expected 11-figure submission suite:\n"
        f"  found: {main_figs}"
    )
if app_figs != EXPECTED_APP:
    errors.append(f"appendix figure inventory differs from expected fa1-fa4 set: {app_figs}")

for stale in LEGACY_MAIN:
    if stale in main:
        errors.append(f"legacy main-text figure still referenced: {stale}")

for rel in sorted(set(main_figs + app_figs)):
    if not (HERE / rel).is_file():
        errors.append(f"missing included figure file: {rel}")

# Local figure references must resolve in main; appendix:* references are
# external-document labels and are intentionally excluded here.
defined = labels(main)
for ref in re.findall(r"\\(?:auto|page)?ref\{(fig:[^}]+)\}", main):
    if ref not in defined:
        errors.append(f"undefined main-text figure reference: {ref}")

# Submission blockers intentionally remain explicit rather than being guessed.
blockers = [
    "[DATA LICENCE]",
    "[CODE LICENCE]",
    ">>> FILL IN <<<",
]
for token in blockers:
    if token in main or token in app:
        errors.append(f"unresolved author-owned submission placeholder: {token}")

# The abstract should stay within the journal's compact abstract convention.
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, flags=re.S)
if not m:
    errors.append("abstract not found")
else:
    abstract = re.sub(r"%.*?$", " ", m.group(1), flags=re.M)
    abstract = re.sub(r"\\[A-Za-z]+(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", abstract)
    abstract = re.sub(r"\\[A-Za-z]+|[{}$~]", " ", abstract)
    n_words = len(re.findall(r"\b[\w'-]+\b", abstract))
    if n_words > 200:
        errors.append(f"abstract is approximately {n_words} words (>200)")
    else:
        print(f"OK  abstract ~{n_words} words")

# Figure typography: warn, do not fail. The Royal Society production guidance
# recommends avoiding very small lettering; binary files must be checked after
# regeneration because source point sizes can be rescaled at save/placement.
if STYLE.is_file():
    style = STYLE.read_text(encoding="utf-8")
    for key in ("FS_NOTE", "FS_TICK"):
        mm = re.search(rf"^{key}\s*=\s*([0-9.]+)", style, flags=re.M)
        if mm and float(mm.group(1)) < 7.5:
            warnings.append(
                f"{key}={mm.group(1)} pt in Analysis/figures/style.py; "
                "inspect final-size figure lettering before upload"
            )

if warnings:
    print("\nWARNINGS")
    for item in warnings:
        print(f"  - {item}")

if errors:
    print("\nBLOCKERS")
    for item in errors:
        print(f"  - {item}")
    sys.exit(1)

print("\nPASS  manuscript source passes submission preflight")
