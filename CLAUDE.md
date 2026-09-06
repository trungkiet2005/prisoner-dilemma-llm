# prisoner-dilemma-llm

Cross-lingual, cross-payoff study of how LLM agents play the iterated prisoner's
dilemma. Built on FAIRGAME.

## What the current paper claims

Multiplying every payoff by a positive constant is inert under the theory: it
changes no preference, best reply, equilibrium or orbit of the replicator
dynamics. It nonetheless moves cooperation by up to 0.636, reverses the sign of
the persona instruction in two models of five, and changes which canonical
strategy the transcript matches. Because the effect has a different shape in each
model, a pooled analysis reports a movement no model in the pool makes: the
scale-by-model interaction carries 17.9% of the cell variance against 5.8% for
the main effect.

Corpus: `Dataset/data_fairgame_frontier_llm`, six models by ten payoff scales,
12,000 dyads, 24,000 agent-games, 240,000 decisions. The main text reports five
models and the supplement all six.

## Where the truth lives

| question | authoritative file |
|---|---|
| what the paper claims, and its build | `papers/interface-focus/README.md` |
| every quoted number | `Analysis/scaling/s08_verify_paper.py`, which recomputes them and fails on disagreement |
| the collection plan and its experiment numbers | `legacy/paper_scaling/RUN_PLAN.md` |
| how a Kaggle run becomes data | `results/README.md` |
| the open-weight arm's payoff caveat | `Analysis/README.md` |
| the figure palette, geometry and helpers | `Analysis/figures/style.py` |
| the memory-one read-out and its contrasts | `Analysis/figures/data.py` |

## The environment

The analysis runs on the conda env `egt` (Python 3.12):
`/d/Anaconda/envs/egt/python.exe`. egttools has no wheel for the system Python
3.14 and building it needs a C++ toolchain this machine does not have.

## Two figure generations

`Analysis/scaling/s05_figures.py` draws the original set, `f1_`..`f8_`, which
`main.tex` currently includes. `Analysis/figures/` draws the replacement set,
`f_`-prefixed, one script per figure, all importing `style.py`. Both write into
`papers/interface-focus/figures/`; the names do not collide, so the two
generations coexist until the manuscript has moved over.

The replacement set exists for two reasons. The old figures are drawn 183 mm
wide against a 170 mm column, so LaTeX scales all of them to 93% and prints
7 pt labels at 6.5 pt; and several of them are five overlapping curves where a
heatmap or a small multiple says the same thing at a glance.

Figures and the three `*_auto.tex` table files are **generated**. Never edit them
by hand; edit the script and rerun.

## Layout

```
FAIRGAME/    simulation framework (vendored upstream, keeps its own licence)
kaggle/      execution: open-weight models on GPU, Gemini via Kaggle Benchmarks
Dataset/     collected corpora, one CSV per (payoff scale, model, language)
results/     raw Benchmarks output, before promotion into Dataset/
Analysis/    scaling/ = current pipeline; scripts/ + pdlib/ = the earlier study
papers/      interface-focus (current) and deduce-before-you-label (separate)
legacy/      paper_scaling, the superseded draft of the current paper
private/     gitignored: editorial correspondence, not for the public tree
```

`Analysis/scaling/` writes into `papers/interface-focus` by default.
`PD_PAPER_DIR` and `PD_FIGDIR` redirect it; pointing `PD_FIGDIR` at
`legacy/paper_scaling/figures` is what enables the old `fig1_`..`fig6_` aliases.

## Two manuscripts, one repository

`papers/interface-focus` and `papers/deduce-before-you-label` are different
papers on the same corpus: the first is about the payoff scale, the second about
strategy attribution. They share `Dataset/` and nothing else. Do not carry a
number, a figure or a framing from one into the other.

## Traps

1. **Never copy a Kaggle run into `Dataset/` by hand.** Use
   `kaggle/benchmarks/collect_run.py`. It has failed silently twice: Kaggle
   mounts the previous run's output for RESUME, so the download contains λ
   directories this run never produced; and the `-rep2` run tag sticks to
   directory names, file names *and* the `agent1_llm` / `agent2_llm` columns, so
   a raw copy either raises a `KeyError` in `ingest.py` or, worse, counts one
   model as two.
2. **The open-weight and frontier corpora are not comparable.** They were
   collected under `mild` (`R = 8`) and `conventional` (`R = 6`) respectively.
3. **A stale `.bbl` produces a stale reference list without warning.** Delete
   `main.bbl` and `appendix.bbl` after editing `mybib.bib`.
4. **Overleaf compiles with `-jobname=output`** and so can never produce
   `appendix.aux`, which `main.tex` reads through `xr`. Build the appendix
   locally and upload the resulting `appendix.aux` alongside the sources, or
   every cross-reference into the appendix renders as `??`.
5. **The `latexrelease` line at the top of both sources is load-bearing.** It
   works around this machine's MiKTeX and is guarded to be a no-op elsewhere. Do
   not simplify it to a bare `\RequirePackage`.
6. **Strategy labels are not identifications.** TFT and WSLS shares are 94.6% and
   97.1% LSTM attributions on trajectories matching no canonical rule. Say
   "resembles", not "plays".
7. **egttools underflows in float64 at large `beta * lambda`.** At
   `epsilon = 0.05` and `lambda >= 100` the fixation probabilities reach exactly
   zero, the embedded chain gains a second absorbing state, and the eigen solver
   returns whichever basis vector it lands on: it reports `WSLS = 1.0` where the
   answer is `AllD = 1.0`. `s09_egt.py` solves those cells at 200 digits and
   records `n_absorbing_float64` per cell. Never draw a float64 stationary
   distribution without checking it against `T14_egt_stationary.csv`;
   `fig_invasion.py` shows the guard.
8. **`pCD - pDC` is not a reciprocity measure.** It equals persistence minus
   reciprocity exactly, so it confounds the two main effects of the two-by-two
   predecessor design. Use `data.contrasts`.

## House style

No em dashes or en dashes in prose, plain hyphens only. En dashes are allowed in
numeric ranges, paired proper names and section ranges. Before committing:

```bash
grep -n -- "---" papers/interface-focus/*.tex | grep -v ":%"   # must be empty
```

## Before submitting

Search both sources for `>>> FILL IN <<<`.
