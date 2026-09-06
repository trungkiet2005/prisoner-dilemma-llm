# Payoff scale reshapes how language models play the prisoner's dilemma

Major revision, 2026-09-05. The manuscript, its analysis pipeline, its figures and
its tables were all rebuilt from scratch against the completed frontier corpus. No
number, figure or table survives from the earlier "Units matter" draft.

Sixth model added, 2026-09-06. Grok 4.20 (`grok-4.20-0309-non-reasoning`) was
collected on the same complete grid and ingested, taking the corpus to 6 models,
12,000 dyads, 24,000 agent-games, 240,000 decisions and 300
model-by-language-by-scale cells, with overall cooperation 0.5895. The main text
reports five models and the electronic supplementary material reports all six;
see "The five/six split" below. Every figure, every main table and every
supplementary table was regenerated, and the whole prose ledger was rebuilt.

Restructured for *Interface Focus*, 2026-09-06. The section order is now
Introduction, Material and methods, Results, Discussion, Conclusion, back matter:
Methods moved ahead of Results and was expanded, the Introduction gained an explicit
research-gap paragraph and three numbered contributions, the Abstract was cut from
about twenty numerals to three, the six Results subsections were merged into four,
the Discussion was stripped of statistics, and a Conclusion plus Royal Society back
matter were added. Operational collection detail moved from Methods into the
supplement rather than being deleted. No claim and no number changed, except five
prose figures corrected to agree with the machine-generated tables (see below).

## What the paper claims

One contribution, of the **insight** kind: multiplying every payoff by a positive
constant is inert under the theory - it changes no preference, best reply,
equilibrium or orbit of the replicator dynamics - and yet it moves cooperation by
up to 0.636,
reverses the sign of the persona instruction in two models of five, and changes
which canonical strategy the transcript matches. Because the effect has a different
shape in each model, a pooled analysis reports a movement no model in the pool
makes: the scale-by-model interaction carries 17.9% of the cell variance against
5.8% for the main effect, and averaging the five curves gives a range of 0.154
where the mean within-model range is 0.249. The main effect is significant here
only because Grok 4.20 departs far enough to survive the averaging; drop it and
the same design returns the null, so whether a pooled design finds an average
effect is a fact about the panel.

Section by section:

| Section | Claim | Evidence |
|---|---|---|
| 2 | The design, its completeness guard, the protocol and every statistic | - |
| 3.1 | Cooperation moves in all five models, and pooling reports a movement none of them makes | Table 1, Fig. 1b,c; permutation test, clustered Wald test, variance decomposition, shape correlations |
| 3.2 | The language sets the size of the effect; the scale gates the persona | Fig. 2, Fig. 3, Table 2; language x scale interaction, sub-unit vs supra-unit contrast |
| 3.3 | The strategy read-out changes, not just the rate (secondary analysis); four of five models one way, the supplementary sixth the other | Table 3, Figs. 4 and 5; hybrid rule-base + LSTM read-out |
| 3.4 | The effect is present on the opening move | Fig. 6 |

Figure 5 gives one panel per canonical strategy (AllC, TFT, WSLS, AllD), share
against the ten payoff scales, five model lines, on a shared vertical axis
because the four shares sum to 100% within each model and scale.

A caution that governs how Figure 5 may be read, and that the caption, §2.6, §3.3
and the Discussion all state: the AllC and AllD labels carry deduced shares of 23.6%
and 35.6%, but TFT and WSLS are 94.6% and 97.1% LSTM attributions on
trajectories matching no canonical rule. A move in the TFT panel is a
resemblance, not an identification. Table S8 carries the full breakdown.

## Corrections made during the 2026-09-06 restructure

Two classes of defect were found while moving the text around, and both are fixed.

**The cell arithmetic in Methods was wrong.** The old draft said "each of the 250
model-by-language-by-scale cells contains exactly 200 dyads", which multiplies out to
50,000 rather than the 10,000 the corpus holds. Checked against `games.parquet`: a
model-by-language-by-scale cell holds **40** dyads (four persona pairings x ten
replicates), and 200 is the size of a **model-by-scale** cell once the five languages
are pooled. Methods now says both.

**Five prose figures disagreed with the tables printed beside them.** In each case
the underlying value is an exact tie and the prose rounded up where the generated
table's format string rounds down. The tables are authoritative, so the prose and the
`s08_verify_paper.py` ledger were both moved onto the table's value.

| quantity | prose said | tables say | source value |
|---|---|---|---|
| Gemini 3.5 Flash-Lite cooperation range | 0.191 | 0.190 | 0.1904999999999999 |
| Qwen3 persona effect, sub-unit | -0.912 | -0.911 | -0.9115 |
| Gemini 3.5 rounds violating nearest rule, max scale | 1.37 | 1.36 | 1.365 |
| GPT-5.4 Nano AllC share at lambda = 1000 | 50.8% | 50.7% | 50.74999999999999 |
| Gemini 3.5 TFT share at lambda = 1000 | 14.3% | 14.2% | 14.249999999999998 |

The same rule was applied again when Grok 4.20 arrived: its cooperation minimum
and maximum are 0.27449999999999997 and 0.9105, which `Table 1` prints as 0.274
and 0.910, so the prose in 3.1 and the ledger use 0.274 and 0.910 rather than the
half-up 0.275 and 0.911. The range, 0.636, is computed before rounding and is the
same either way.

Two disclosures were also added to the manuscript rather than left implicit. The
Arabic and Chinese prompt templates state the payoff cells as penalties while phrasing
the goal sentence as maximising a reward; the templates are reproduced unaltered from
the published protocol, so the paper now says in Methods, Results and the Discussion
that a *level* difference between languages is a difference between two stimuli, and
why the *scale* result is unaffected (the wording is identical at every lambda). And
the claim that play becomes less rule-governed is now qualified everywhere it appears:
it holds in four models of five, is absent in the fifth, and reverses significantly in
the sixth model, the one the supplement carries, so stating it pooled was the very
error the paper argues against.

## The five/six split

The corpus holds six models. `figstyle.MODEL_ORDER` is the five the main text
reports - Claude Haiku 4.5, GPT-5.4 Nano, Gemini 3.5 Flash-Lite, Qwen3
235B-A22B, Grok 4.20 - `figstyle.SUPPLEMENT_ONLY` is Gemini 3.1 Flash-Lite, and
`figstyle.MODEL_ORDER_ALL` is all six. The six figures and the three main tables
are drawn on the five; every supplementary table carries all six; the two pooled
quantities and the pooled strategy table are computed both ways, with the all-six
versions in `T04_pooling_all.csv`, `T04_shape_correlations_all.csv`,
`T06_variance_all.csv` and `T13_pooled_strategy_all.csv`.

The reason for the split is the endpoint, not the result:
`gemini-3.1-flash-lite-preview` names a preview endpoint the provider does not
pin to a fixed version, so what was served under that name cannot be recovered
and re-addressed. Because that model is also the only one in the corpus whose
deduced-share trend runs positive, the manuscript discloses it three times - in
Methods where the models are named, in Results §3.4 where the trend is reported,
and in the limitations - and §S8 of the supplement gives it in full together with
the all-six version of every pooled quantity. Nothing about that model is left
out of the package; it is moved, and the move is announced.

## Scope of the data

Fixed by decision on 2026-09-05, and deliberately narrow:

- **`Dataset/data_fairgame_frontier_llm`** - the only corpus of language-model
  transcripts used. 10 payoff scales x 6 models x 5 languages x 4 persona pairings
  x 10 replicates x 10 rounds x 2 agents = 12,000 dyads, 24,000 agent-games,
  240,000 decisions, with every one of the 300 model-by-language-by-scale cells at
  exactly 40 dyads and every model-by-scale cell at exactly 200. The five models
  the main text reports carry 250 of those cells, 10,000 dyads, 20,000
  agent-games and 200,000 decisions.
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
main is 21 pages, the supplement 12.

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
python Analysis/scaling/s05_figures.py         # Figs 1-6
python Analysis/scaling/s06_tables.py          # -> paper_scaling/tables_auto.tex
python Analysis/scaling/s07_supplementary.py   # -> paper_scaling/supp_tables_auto.tex
python Analysis/scaling/s08_verify_paper.py    # 163 quoted numbers vs the data
```

`s08_verify_paper.py` is the guard on the one thing the generated tables cannot
protect: the figures typed by hand into the Results and Discussion prose. Note its
limit - it never opens `main.tex`, so it proves the script's ledger agrees with the
data, not that the manuscript still prints those values. Deleting a sentence that
carries a number will not trip it. It
recomputes all 163 of them and exits non-zero on any mismatch. Its scope is now
both documents: a ledger entry named with a leading `ESM ` is printed only in
the supplement, and one named `T1 `/`T2 `/`T3 ` reaches the reader through a
generated table rather than through prose. Run it after any
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

`MODEL_MAP` in `ingest.py` gained `claude-haiku-4-5-20251001`,
`qwen3-235b-a22b-instruct-2507` and `grok-4.20-0309-non-reasoning`; it is indexed
directly, so any new model directory
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

The only `--` in `main.tex` is `Neumann--Morgenstern`, which the rule exempts.
The check above greps `main.tex` only, so it does not see the reference list:
`references.bib` still contains one `---`, inside the published BioSystems title
"Coevolutionary games---a mini review", and that one stays because it is the
title as printed. Run the same two checks over `references.bib` and
`supplementary.tex` before submission.
