# Payoff scale reshapes how language models play the prisoner's dilemma

Major revision, 2026-09-05. The manuscript, its analysis pipeline, its figures and
its tables were all rebuilt from scratch against the completed frontier corpus. No
number, figure or table survives from the earlier "Units matter" draft.

## What the paper claims

One contribution, of the **insight** kind: multiplying every payoff by a positive
constant is inert under the theory - it changes no preference, best reply,
equilibrium or replicator trajectory - and yet it moves cooperation by up to 0.243,
reverses the sign of the persona instruction in three models of five, and changes
which canonical strategy the transcript matches. Because the effect has a different
shape in each model, pooling across models reports it as almost absent.

Section by section:

| Section | Claim | Evidence |
|---|---|---|
| 2.1 | The design, and its completeness guard | - |
| 2.2 | Cooperation moves in all five models | Table 1, Fig. 1b; permutation test, clustered Wald test |
| 2.3 | Pooling reports the effect as almost absent | Fig. 1c; variance decomposition, shape correlations |
| 2.4 | The prompt language sets the size of the effect | Fig. 2; language x scale interaction |
| 2.5 | The scale gates the persona instruction | Table 2, Fig. 3; sub-unit vs supra-unit contrast |
| 2.6 | The strategy played changes, not just the rate | Table 3, Figs. 4 and 5; hybrid rule-base + LSTM read-out |
| 2.7 | The effect is present on the opening move | Fig. 6 |

Figure 5 gives one panel per canonical strategy (AllC, TFT, WSLS, AllD), share
against the ten payoff scales, five model lines, on a shared vertical axis
because the four shares sum to 100% within each model and scale.

A caution that governs how Figure 5 may be read, and that the caption, §2.6 and
the Discussion all state: the AllC and AllD labels carry deduced shares of 26.0%
and 32.7%, but TFT and WSLS are 95.4% and 97.3% LSTM attributions on
trajectories matching no canonical rule. A move in the TFT panel is a
resemblance, not an identification. Table S7 carries the full breakdown.

## Scope of the data

Fixed by decision on 2026-09-05, and deliberately narrow:

- **`Dataset/data_fairgame_frontier_llm`** - the only corpus of language-model
  transcripts used. 10 payoff scales x 5 models x 5 languages x 4 persona pairings
  x 10 replicates x 10 rounds x 2 agents = 10,000 dyads, 20,000 agent-games,
  200,000 decisions, with every cell at exactly 200 dyads.
- **`Dataset/noise_dataset`** - synthetic AllC/AllD/TFT/WSLS trajectories at four
  execution-noise levels. This trains the LSTM branch of the strategy read-out and
  contains no language-model output at all.

Everything else in `Dataset/` is **out of scope** and must not be reintroduced:
`data_stag_hunt_frontier`, `data_replicate_frontier`, `data_fairgame_e2_notation`,
`data_fairgame_small_llm`, `archive_data_fairgame_frontier_llm`.

Two consequences follow, and both are stated in the paper rather than hidden.
Dropping the replicate sweep removes the measured test-retest floor, so the
reference for "large" is the dyad bootstrap and the permutation null instead; the
Discussion says so and names a re-run sweep as the obvious next measurement.
Dropping the notation sweep removes the notation-against-magnitude comparison
entirely, so the paper makes no claim about decimal rendering beyond the one the
main grid supports on its own: that the boundary gating the persona instruction is
one of magnitude, since lambda = 0.25 is still printed with a decimal point and
already behaves like the large scales.

## Building

```bash
cd paper_scaling
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplementary && pdflatex supplementary
```

Both currently build with **0 errors, 0 undefined references and 0 overfull boxes**;
main is 13 pages, the supplement 8.

`tables_auto.tex` and `supp_tables_auto.tex` are machine-generated. Do not edit them,
and do not retype a number from them into the prose - regenerate instead.

## Regenerating everything

The pipeline lives in `Analysis/scaling/` and runs in order. Nothing in it calls a
model; it reads transcripts that are already on disk.

```bash
python Analysis/scaling/s00_build.py           # transcripts  -> rounds/games.parquet
python Analysis/scaling/s01_train_lstm.py      # noise corpus -> models/strategy_lstm.pt  (~4 min)
python Analysis/scaling/s02_readout.py         # hybrid read-out -> readout.parquet
python Analysis/scaling/s03_stats.py           # T02..T09      (~3 min, bootstrap + permutation)
python Analysis/scaling/s04_strategy_stats.py  # T10..T13
python Analysis/scaling/s05_figures.py         # Figs 1-5
python Analysis/scaling/s06_tables.py          # -> paper_scaling/tables_auto.tex
python Analysis/scaling/s07_supplementary.py   # -> paper_scaling/supp_tables_auto.tex
python Analysis/scaling/s08_verify_paper.py    # 92 prose numbers vs the data
```

`s08_verify_paper.py` is the guard on the one thing the generated tables cannot
protect: the figures typed by hand into the Results and Discussion prose. It
recomputes all 92 of them and exits non-zero on any mismatch. Run it after any
change to the corpus, the pipeline, or a quoted number.

Seeds are fixed: 1234 for classifier training, 20260905 for the bootstrap and the
permutation test.

`s00_build.py` refuses to write anything if the grid is incomplete, if any game is
not ten rounds long, or if the payoff scale recovered from the recorded payoffs
disagrees with the directory it came from. That guard is deliberate: a silently
truncated cell is a failure this corpus has actually suffered.

## What was reused, and why

The pipeline is new, but four modules under `Analysis/pdlib/` were kept because they
encode correctness that is expensive to rediscover:

- `ingest.py` - CSV parsing and, critically, `ACTION_MAP = {OptionA: D, OptionB: C}`.
  The prompt states payoffs as penalties to be minimised, so OptionA is the dominant
  and therefore defecting action. This polarity was inverted once before; getting it
  wrong maps every cooperation rate to its complement.
- `seqcode.py` - the token alphabet shared between the synthetic corpus and real
  transcripts, which is what makes the classifier transferable.
- `rulebase.py` - exact memory-one rule matching.
- `lstm.py` - the classifier architecture.

`MODEL_MAP` in `ingest.py` gained `claude-haiku-4-5-20251001` and
`qwen3-235b-a22b-instruct-2507`; it is indexed directly, so any new model directory
must be registered there or the ingest raises `KeyError` rather than skipping it.

The old analysis scripts (`Analysis/scripts/00`-`56`) are not used by this paper.

## Figures

`s05_figures.py` writes the six figures straight into `paper_scaling/figures/`, as
both PDF (what LaTeX uses) and PNG (for quick viewing). `main.tex` picks them up
with `\graphicspath{{figures/}}`, so this directory is self-contained and there is
no copy step to forget.

The previous draft's five figures (`fig1_design`, `fig2_response`, `fig3_notation`,
`fig4_where`, `fig5_gating`) and its `supp_tables.tex` have been deleted. Git
history is the archive; they are in the commit before this revision if ever needed.

## House style

`CLAUDE.md` forbids em dashes in the manuscript. Verified clean:

```bash
grep -n -- "---" paper_scaling/main.tex | grep -v "^[0-9]*:%"   # empty
python -c "import re;print([l for l in open('paper_scaling/main.tex',encoding='utf-8') if re.search('[—–]',l)])"   # []
```

The only `--` in the manuscript is `Neumann--Morgenstern`, which the rule exempts.
