# Manuscript: "Units matter: cooperation in large language model agents tracks how the payoff matrix is written, not what it means"

The **payoff-scaling paper**, and the second of two manuscripts drawn from this
corpus. The other is [`../paper/`](../paper/README.md), whose spine is strategy
attribution. **Only one of the two should be submitted**: they share a corpus,
and §3.8 here overlaps figure 4a there.

Successor to **arXiv:2601.19082** ("Payoff scaling shapes cooperation in LLM
agents across languages"). That preprint established that the cooperation rate
of the open-weight arm moves with the payoff scale and with the language; this
manuscript asks what the movement is a *function* of, and adds the frontier
arm, the notation-against-magnitude test, the opening-move localisation, the
persona result, the coordination invariant, the strategy read-out and the
reporting standard. The
polarity correction recorded in `Analysis/pdlib/ingest.py` was validated
against fig. 2a of that preprint (MAE 0.002), so the numbers here are on the
same convention as the published version.

## Build

```bash
python Analysis/scripts/40_scaling_stats.py    # ~3 min -> tables/T_PS*.csv
python Analysis/scripts/41_fig_scaling.py      # ~20 s  -> paper_scaling/figures/
python Analysis/scripts/42_supp_tables.py      # ~1 s   -> paper_scaling/supp_tables.tex
cd paper_scaling
pdflatex main          && bibtex main          && pdflatex main          && pdflatex main
pdflatex supplementary && bibtex supplementary && pdflatex supplementary && pdflatex supplementary
```

Both compile clean: 0 errors, 0 undefined references, 0 overfull boxes.
Main text 23 pages, abstract 204 words, body ~8,400 words excluding floats;
electronic supplementary material 7 pages, 10 tables.

## Journal target: J. R. Soc. Interface

Formatted to Royal Society house style on a plain `article` class, because
their `.cls` is not needed until production and Interface accepts any legible
PDF at submission. What has been conformed:

- abstract 204 words, unstructured, single paragraph (their guidance is ~200);
- 6 keywords (their range is 3-6);
- `Material and methods` as the section heading, British spelling throughout,
  numbered sections;
- end statements present and **in the Royal Society order**: Ethics, Data
  accessibility, Declaration of AI use, Authors' contributions, Conflict of
  interest declaration, Funding, Acknowledgements;
- supplementary content referred to as **electronic supplementary material**
  and cited inline in their style (`electronic supplementary material,
  table S4`), numbered S1-S10;
- numbered references, `unsrtnat`.

Not done, deliberately: line numbers and double spacing are not applied, since
Interface does not require them at submission and they make the working PDF
harder to read. Add `\usepackage{lineno}` plus `\linenumbers` if a reviewer
asks for them.

## The spine

One manipulation, whose null is a theorem. Multiplying every payoff by
$\lambda > 0$ is the multiplicative half of the positive affine transformation
that a von Neumann--Morgenstern utility is defined up to, so it leaves the
outcome ordering, the dominant action, the equilibrium set, the greed/fear
indices and the replicator flow exactly where they were. The predicted effect
on any agent that is playing the game is **exactly zero**.

The corpus is both arms, and the open-weight arm is what carries the sweep:

| arm | models | scales | rounds | dyads | agent-games | decisions |
|---|---|---|---|---|---|---|
| open-weight | Gemma 3 12B, Llama 3.1 8B, Qwen3 8B | 6 (0.01 to 1000) | 30 | 3,600 | 7,200 | 216,000 |
| frontier | Claude 3.5 Haiku, Gemini 3.5 Flash-Lite, GPT-4o, Mistral Large | 3 (0.1 to 10) | 10 | 2,400 | 4,800 | 48,000 |

Both are balanced at exactly 80 agent-games per model x language x scale cell.
The two arms use **different base matrices** (frontier `P = 6`, open-weight
`P = 8`), different horizons and different decoding settings, so nothing is
pooled or compared across them; every result is measured within an arm and
within a model.

## Headline results, in the order the paper makes them

1. **The invariance fails.** Six of seven models move: swings of 0.316, 0.267,
   0.198 (open-weight) and 0.240, 0.234, 0.080 (frontier) against dyad-level
   permutation nulls of 0.047 to 0.121. Mistral Large is the honest exception
   at *P* = 0.38 on cooperation rate, though its strategy mix still moves
   (*P* = 0.006).
2. **The response is not monotone.** Over six scales it rises to a maximum at
   λ = 1 or 10 and falls again by λ = 100. A three-scale window at {0.1, 1, 10},
   which is what most published designs sample, lies entirely on the rising
   limb.
3. **Notation beats magnitude.** ΔBIC from the best specification, open-weight
   arm: no λ term 374, linear in log λ 247, quadratic 110, cubic 118,
   fractional flag 56, fractional + glyph count 16, **notation regime 0**,
   saturated in λ 20. A two-parameter description of *how the numbers are
   printed* matches what a free parameter per scale achieves. Five of the
   seven models prefer a notation description individually; **Gemma 3 12B
   prefers a cubic in log λ** (saturated 3.3 behind, notation regime 24.3
   behind), which is the one case in the corpus where the notation account
   loses to a polynomial, and Mistral Large prefers no λ term at all.
4. **λ decides which model wins.** In the open-weight arm the smallest
   within-model swing (0.198) exceeds the median between-model spread at a
   fixed scale (0.114). 8 of 45 model-pair comparisons change sign; in the
   frontier arm Mistral Large beats Claude 3.5 Haiku at λ = 0.1 and loses to it
   at λ = 1.
5. **It is set at the opening move.** The regime gap on round one alone is
   −0.129 against −0.199 over the whole game (open-weight) and −0.199 against
   −0.142 (frontier). Round one has no history, so this is a prior the notation
   induces.
6. **The level moves; the coordination does not.** Miscoordination (CD+DC)
   swings 0.039 across six scales (*P* = 0.14) and 0.015 across three
   (*P* = 0.56), while mutual cooperation over the same rows swings 0.198 and
   0.139.
7. **λ shifts instruction-following, but does not set its sign.** The
   persona effect moves toward compliance across the fractional-to-unit
   boundary in **all seven** models (+0.244, +0.052, +0.192 open-weight;
   +0.256, +0.691, +0.707, +0.396 frontier, every interval excluding zero)
   and back away from it in all three whose sweep reaches the large regime.
   But **five of the seven never change sign**: Gemma and Llama are compliant
   at every scale, Qwen3, Claude and Mistral inverted at every scale, and only
   GPT-4o and Gemini cross zero. The pooled curve (−0.095 → +0.113 → −0.003)
   crosses zero although almost no model does, so **do not report it alone**;
   model x regime x persona interaction *F*(3) = 50.8, *P* = 2.7e-32
   (frontier) and *F*(4) = 7.6, *P* = 4.2e-06 (open-weight).
8. **The strategy label moves, selectively.** Max TV per model 0.125 to 0.388,
   all seven above their dyad nulls; over the four *named* rules alone 0.127
   to 0.388, still all seven. AllD and AllC trace the cooperation curve and
   its mirror (swings 0.223 and 0.161 open-weight); TFT moves 0.049 and WSLS
   0.020, and WSLS does not clear its null. Reciprocity is what the rescaling
   leaves alone - the same dissociation result 6 records at the round level.
   Read-out provenance: 22.1% provable rule, 11.8% ambiguous, 66.1% LSTM
   (open-weight); 25.5%, 8.3%, 66.2% (frontier); and none of those shares is
   constant across the sweep.
9. **Framing does not explain it.** The maximise-framed and minimise-framed
   languages give the same non-monotone shape at different levels, so the
   reward-reading account of the scale effect is not sufficient.

## Files

| Path | What it is |
|---|---|
| `main.tex` | The manuscript. `article` class, single column, numbered references, Royal Society house style. |
| `supplementary.tex` | Electronic supplementary material: S1 validates the read-out, then tables S1-S10. |
| `supp_tables.tex` | **Generated** by `42_supp_tables.py` from the CSVs; do not edit by hand. |
| `references.bib` | 38 entries (37 shared with `../paper/`, plus the authors' own preprint), 31 cited. |
| `figures/fig1_design.pdf` | Fig. 1 - the stage game, what the rescaling prints, the corpus. |
| `figures/fig2_response.pdf` | Fig. 2 - the response curve and the invariance test. **The headline figure.** |
| `figures/fig3_notation.pdf` | Fig. 3 - notation regimes, the ΔBIC ladder, the ranking crossings. |
| `figures/fig4_where.pdf` | Fig. 4 - where the effect enters, and what it leaves alone. |
| `figures/fig5_gating.pdf` | Fig. 5 - persona, language and the strategy label. |

All five are regenerated by `Analysis/scripts/41_fig_scaling.py`, which reads
`Analysis/tables/T_PS*.csv` and computes nothing of its own, so a panel cannot
drift from the table behind it. Captions live in the LaTeX float.

## Tables written by `40_scaling_stats.py`

`T_PS00` payoff-polarity check - `T_PS01` corpus - `T_PS02` notation and the invariants - `T_PS03` response
curve with dyad bootstrap - `T_PS04` sensitivity, `T_PS04b` rank reversals, `T_PS04c` one-decade contrasts -
`T_PS05` the BIC ladder, `T_PS05b` regime means - `T_PS06` round blocks,
`T_PS06b` opening move, `T_PS06c` the share the opening carries - `T_PS07`
persona by scale, `T_PS07b` the pooled interaction test, `T_PS07c` persona by
model x regime, `T_PS07d` the shift across each regime boundary as a difference
in differences, `T_PS07e` the model x regime x persona three-way test -
`T_PS08` language cells,
`T_PS08b` tests, `T_PS08c` framing curves - `T_PS09` what moves and what does
not, `T_PS09b` role asymmetry - `T_PS10` strategy mix, `T_PS10b` its
permutation tests over both vocabularies, `T_PS10c` mix by model, `T_PS10d`
where each label came from (rule / ambiguous / LSTM above and below the 0.90
floor), `T_PS10e` one share curve per strategy with dyad bootstrap, `T_PS10f`
per-strategy swings against the permutation null.

## Before submission

1. **Fill the placeholders**: author names and affiliations, ORCID, funding,
   `[REPOSITORY URL]` in Data accessibility, the initials in Authors'
   contributions, and the `[AUTHORS TO COMPLETE]` slot in Declaration of AI
   use - that last one is a judgement only you can make, and Royal Society
   now requires the statement.
2. **Fix the self-citation.** `selfpreprint2026scaling` in `references.bib`
   carries placeholder authors; check the paragraph at the end of §1 against
   what the preprint actually claims, and decide whether a blinded submission
   needs it anonymised.
3. **Decide between this manuscript and `../paper/`.** They share a corpus and
   should not both be submitted. Note the read-out overlap is now explicit:
   §2.4 here describes the same hybrid rule+LSTM instrument that `../paper/`
   audits, and cites it as an instrument with stated coverage rather than as
   ground truth. A reviewer who sees both must not find them inconsistent.
4. **Deposit the data**, as for the other manuscript.
5. **Trim the abstract if the venue caps it at 250.** It is 299 words, up from
   260, because the persona and strategy results now state their own scope.

## The one experiment that would finish this paper

**Notation held fixed, magnitude varied, and the reverse.** In the present
design `λ < 1` and "the cells print with a decimal point" are the same event,
so a notation account and a small-stakes account are not separated. Two extra
conditions settle it:

* print the λ = 1 matrix as `0.0 / 2.0 / 8.0 / 10.0` - same magnitude, decimal
  notation;
* print the λ = 0.1 matrix in a unit where the cells are integers - same
  magnitude, integer notation.

This is a template change and one sweep, and §4.2 already states the
prediction. Running it would move the paper from "the response is a step
function over how the matrix is written" to a mechanism.

Second in value: **the additive half of the invariance** (`u -> a + u`), which
moves the reference point without touching the digit count, and **reasoning
models**, which is the extension shared with the other manuscript.
