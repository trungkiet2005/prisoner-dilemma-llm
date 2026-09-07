# Interface Focus submission, RSFS-2026-0050

Single-column manuscript for the *Interface Focus* theme issue "Machine behaviour in
the age of large language models: social, cognitive and evolutionary perspectives",
class `rsproca_new`. Title: *Frontier language models violate payoff-scale
invariance and show stronger persistence than reciprocity in the prisoner's
dilemma across languages*.

## What this directory now contains

This is a **major revision** (2026-09-06). Every result is computed on
`../../Dataset/data_fairgame_frontier_llm`, the six-model by ten-payoff-scale corpus:
12,000 dyads, 24,000 agent-games, 240,000 decisions, of which the five models reported
in the main text carry 10,000 dyads and 200,000 decisions.

The previous version of this article reported an earlier and smaller collection
(GPT-4o, Claude 3.5 Haiku and Mistral Large; three payoff scales; 1,800 games)
together with an open-weight replication and a baseline FAIRGAME corpus. All of that
is superseded and has been removed; it remains in git history. That earlier collection
is now cited as prior work by the same authors, under `selfpreprint2026scaling`.

The evolutionary game theory baseline has been **recomputed** on the new ten-value
payoff grid rather than carried across, and the whole EGT-versus-corpus comparison now
lives in the appendix, with a short pointer in main-text §3.8.

## Files

| File | Role |
|---|---|
| `main.tex` / `main.pdf` | Main manuscript, 17 pp., 8 figures, 3 tables |
| `appendix.tex` / `appendix.pdf` | Standalone electronic supplementary material, 17 pp. |
| `tables_auto.tex` | The three main-text tables, **generated**, do not edit |
| `supp_tables_auto.tex` | The thirteen supplementary tables, **generated**, do not edit |
| `egt_tables_auto.tex` | The three evolutionary-baseline tables, **generated**, do not edit |
| `mybib.bib` | Bibliography (biblatex/biber, style fixed by the class), 155 entries |
| `rsproca_new.cls` | Journal class |
| `TemplateFigs/` | Journal logos **required by the class** (`\maketitle` first page), do not delete |
| `figures/` | The twelve figures, `f1`-`f8` main text and `fa1`-`fa4` appendix, **generated** |

Nothing in `figures/` and none of the three `*_auto.tex` files is written by hand.
They are produced by the analysis pipeline described below, so a number cannot drift
between the data and the manuscript.

## Regenerating the figures and tables

The pipeline lives in `../../Analysis/scaling/`. It writes into whichever
manuscript directory `PD_PAPER_DIR` and `PD_FIGDIR` name, and with neither set it
resolves to **this** directory, so no export is needed:

```bash
cd ../../Analysis/scaling

python s05_figures.py            # f1..f8, the main-text figures
python s05b_appendix_figures.py  # fa1..fa4, the appendix figures
python s06_tables.py             # tables_auto.tex
python s07_supplementary.py      # supp_tables_auto.tex
python s09_egt.py                # the evolutionary grid + egt_tables_auto.tex
```

`s09_egt.py` recomputes the evolutionary baseline from scratch in 200-digit arithmetic
and takes about seven seconds. It is analytical and calls no model.

To rebuild the statistics the manuscript quotes, run `s00_build`, `s01_train_lstm`,
`s02_readout`, `s03_stats` and `s04_strategy_stats` first, then `s08_verify_paper`,
which recomputes every quoted quantity from the data and fails if any disagrees.

## Build order

The two documents reference each other through `xr`, so build the appendix first and
then run each once more if a cross-referenced number has moved:

```
pdflatex appendix && biber appendix && pdflatex appendix && pdflatex appendix
pdflatex main     && biber main     && pdflatex main     && pdflatex main
```

Delete `main.bbl` and `appendix.bbl` after editing `mybib.bib`, or a stale `.bbl` will
silently produce a stale reference list.

The `\RequirePackage[2025-06-01]{latexrelease}` line at the top of both sources works
around this machine's MiKTeX (2026 kernel with a biblatex that will not load under it).
It is guarded so that it stays a no-op on Overleaf and TeX Live, where the
unconditional form patches nothing kernel-side and yet still rolls `graphics` and
`amsmath` back to their 2019 releases. Do not simplify it to a bare
`\RequirePackage`.

### Building on Overleaf

Overleaf always compiles with `-jobname=output`, so it can never produce an
`appendix.aux`, and `main.tex` reads that file through `xr` to resolve its references
into the appendix. Uploading the sources alone therefore renders every such reference
as `??`. Build the appendix locally, then upload the resulting `appendix.aux` alongside
the sources, and re-upload it whenever an appendix section, figure or table number
moves. LaTeX gives no warning when those exported numbers go stale; it simply prints
the old ones.

## Before submitting

Before submission, the submitting author must enter an ORCID in ScholarOne.
The manuscript records the public repository, CC-BY-4.0 transcript licence,
MIT code licence, no external funding, CRediT roles, and the repository-bounded
collection window. The repository does not retain per-request provider timestamps.

House style: **no em dashes or en dashes in prose**, plain hyphens only. En dashes are
permitted in numeric ranges (`50--86`), paired proper names (`Neumann--Morgenstern`)
and section ranges. Check before committing:

```bash
grep -n -- "---" main.tex appendix.tex | grep -v ":%"     # must be empty
```
