# Analysis — FAIRGAME prisoner's dilemma with LLM agents

Everything here is generated from `Dataset/` by plain Python scripts. No web
rendering: every figure is written as **both `.png` (400 dpi)** and **`.pdf`
(vector, Type-42 fonts)** into `figures/`, and every number that appears in a
figure is also dumped to `tables/` as CSV.

```bash
python Analysis/run_all.py              # rebuild everything (~10 min, LSTM training included)
python Analysis/run_all.py --no-train   # reuse Analysis/models/*.pt (~2 min)
```

## Layout

| path | contents |
|---|---|
| `pdlib/style.py` | paper style + the colour system (validated, see below) |
| `pdlib/ingest.py` | raw CSV → tidy `rounds` / `games` tables |
| `pdlib/metrics.py` | reciprocity, memory-one fingerprints, fairness gaps, bootstrap CIs |
| `pdlib/seqcode.py` | the token alphabet shared by the synthetic corpus and LLM play |
| `pdlib/rulebase.py` | exact memory-one rule matching (no learning) |
| `pdlib/residual.py` | the widened hypothesis classes and their permutation null |
| `pdlib/lstm.py` | the learned fallback read-out |
| `scripts/00…11` | one script per figure group |
| `data/` | derived parquet/npz (regenerated, safe to delete) |
| `models/` | trained classifier checkpoints |
| `figures/`, `tables/` | the deliverables |

## Data as parsed

* **10,800 agent-games / 252,000 agent-rounds.** 6 models × 5 languages ×
  {3 or 6} payoff scales × 4 persona dyads × 10 replicates.
* `OptionA` = cooperate, `OptionB` = defect.
* Payoffs in base units: `T=10, P=2, S=0` for both families, but `R=6` for the
  frontier runs and `R=8` for the open-weight runs — so the two families sit at
  **different points of the PD greed/fear plane** (F01c) and their cooperation
  levels are not directly comparable. Every payoff figure normalises within
  family.
* Frontier runs are 10 rounds with the horizon **hidden**; open-weight runs are
  30 rounds with the horizon **announced**. That is a design difference, not a
  result — F06 keeps them in separate panels and adds a rescaled common clock.
* `game_id` is **not** a unique key in the frontier logs (each personality
  condition repeats under the same id); the row position inside the file
  identifies the replicate.
* One integrity flag: `x10_en_claude.csv` contains games whose realised payoffs
  are at ×1, not ×10 (`tables/T03_scale_anomalies.csv`). The parser recovers the
  realised scale per game from the outcomes, so nothing is mis-normalised.

## Figures

### Part A — behaviour

| fig | what it shows |
|---|---|
| **F01** | design: both payoff matrices, greed/fear plane, the model × scale grid, dyad inventory |
| **F02** | cooperation rate by model and payoff scale (game-clustered bootstrap CIs), plus outcome composition |
| **F03** | **payoff-scale invariance is violated** — cooperation is non-monotone in stake size, with a trough at ×1–×10 |
| **F04** | **language disparity** — Qwen3 swings 0.57 in cooperation across languages, GPT-4o only 0.04 (n.s.) |
| **F05** | persona conditioning; Gemma and Llama show an *inverted* own-persona effect |
| **F06** | round dynamics, endgame unravelling, DD as an absorbing state |
| **F07** | joint-outcome composition over the course of a game, per model |
| **F08** | reciprocity plane, memory-one fingerprints vs AllC/AllD/TFT/WSLS/GRIM |
| **F09** | payoff space, within-dyad inequality, who profits in mixed dyads |

### Part B — the strategy read-out model

| fig | what it shows |
|---|---|
| **F10** | the synthetic corpus: token signatures, sequence lengths, **distinct trajectories and the Bayes ceiling per split** |
| **F11** | learning curves, confusion matrices, per-class F1, identifiability, generalisation to unseen noise |
| **F12** | recall vs rounds observed, per strategy |
| **F13** | hidden-state PCA, confidence distribution, and where an unseen rule (GTFT) lands |

### Part C — reading LLM play

| fig | what it shows |
|---|---|
| **F14** | archetype mix per model and per payoff scale — solid = provable exact rule, hatched = LSTM nearest rule, grey = several rules fit |
| **F15** | archetype by language, persona dyad, payoff, and who-meets-whom |
| **F16** | how the posterior over the archetype crystallises round by round |
| **F17** | behavioural-signature radars (frontier vs open-weight) |
| **F18** | **perseveration vs imitation** — do models react to the opponent or to themselves? |
| **F19** | the model card: every metric, one page |
| **F20** | departures from three invariances the PD should respect |
| **F21** | variance decomposition — how much each design factor explains |
| **F22** | game-level cooperation is bimodal |
| **F23** | language × scale small multiples |
| **F24** | openings, and whether a cooperative opening pays |
| **F25** | hybrid evaluation: single-label ceiling, set-valued escape, division of labour |
| **F26** | rule-survival curves on LLM play, and distance to the nearest canonical rule |
| **F27** | the residual: how far a widening rule vocabulary gets, against a shuffled null |
| **F28** | memory depth of the residual, and its position in memory-one profile space |
| **F29** | mid-game regime switching |

## The read-out: exact rules first, LSTM second

Input is a player's own trajectory encoded as
`⟨previous-round outcome⟩⟨action this round⟩` tokens — exactly the encoding of
the noise corpus, which is what makes a model trained on synthetic strategies
directly applicable to an LLM transcript.

All four canonical rules are **memory-one**, so the action each prescribes
depends only on that previous-outcome letter. `pdlib/rulebase.py` therefore
checks *exactly* whether a trajectory is what strategy k would have played — no
learning involved — and returns the **set** of rules that fit:

| prev | meaning | AllC | AllD | TFT | WSLS |
|---|---|---|---|---|---|
| E | first move | C | D | C | C |
| R | both cooperated | C | D | C | C |
| S | I cooperated, was hit | C | D | D | D |
| T | I defected, they didn't | C | D | C | D |
| P | both defected | C | D | D | C |

The pipeline is: **one rule fits → that label; several fit → report the set;
none fits → LSTM gives the nearest rule.** The LSTM is trained on the union of
the **noise-free** and **5% execution-noise** splits of the 4-strategy files.

### Evaluation (`tables/T17_hybrid_evaluation.csv`, F25)

| corpus | noise | Bayes (single) | LSTM | hybrid single | **hybrid set** | mean answer size |
|---|---|---|---|---|---|---|
| 10-round | 0% | 0.9957 | 0.9954 | 0.9954 | **1.0000** | 1.01 |
| 10-round | 5% | 0.9866 | 0.9753 | 0.9753 | **0.9784** | 1.01 |
| 10-round | 10% *(unseen)* | 0.9727 | 0.9366 | 0.9366 | **0.9386** | 1.01 |
| 10-round | 20% *(unseen)* | 0.9397 | 0.8104 | 0.8104 | **0.8112** | 1.00 |
| 30-round | 0% | **0.6250** | 0.6250 | 0.6250 | **1.0000** | 2.13 |
| 30-round | 5% | 0.9659 | 0.9533 | 0.9533 | **0.9718** | 1.06 |

Where a rule fires uniquely it is right **95–100%** of the time, and the LSTM
covers the noisy remainder at 0.79–0.97.

Two things to state plainly in the paper:

1. **The hybrid does not beat the LSTM at single-label accuracy** — the two are
   identical to four decimals. What the rules buy is (a) provability, (b) the
   set-valued escape below, and (c) an honest "no canonical rule explains this".
2. **The 0.625 ceiling is an artefact of forcing one label.** The 30-round
   noise-free file collapses to **eight distinct trajectories** across all
   161,280 lines; 90,720 of them are the single sequence `EC RC RC … RC`,
   carrying the labels AllC, TFT and WSLS in equal thirds. A player who
   cooperates for thirty rounds against a cooperator *is* all three, so no
   method can separate them. Allowing a set answer, the hybrid is **100%
   correct on that file with an average answer of 2.13 labels**. Exact-rule
   coverage also drops as 0.95^n with n rounds of 5% noise — 63% at ten rounds,
   22% at thirty — which is exactly where the LSTM earns its keep.

The LSTM's own training still caps identical `(sequence, label)` pairs at 40
copies so the degenerate block does not dominate the gradient. On the *raw*
uncapped corpus that model scores 0.789 (ceiling 0.796), i.e. capping changes
which test set is being scored, not how well the network generalises.

## Colour system

The categorical palette is a validated instance (six checks: lightness band,
chroma floor, CVD separation, normal-vision floor, contrast). All groups used
here pass on the light chart surface `#fcfcfb`:

* **actions** — C `#2a78d6`, D `#e34948` (all-pairs CVD ΔE 21.6)
* **joint outcomes** — CC `#2a78d6`, CD `#1baf7a`, DC `#eda100`, DD `#e34948`
* **strategies** — AllC `#2a78d6`, TFT `#1baf7a`, WSLS `#4a3aa7`, AllD `#e34948`
* **models** — frontier trio and open-weight trio each validate all-pairs; the
  six together validate on the adjacent pairlist, which is why the one scatter
  with all models (F09a) is collapsed to two series.

Sequential encodings use a single blue ramp; diverging encodings use blue↔red
through a neutral grey midpoint. Never a rainbow, never a dual axis.

## Caveats to state in the paper

1. `R` differs between the two families (6 vs 8), so cross-family cooperation
   comparisons confound model with payoff geometry.
2. Horizon knowledge differs (hidden vs announced), which is exactly the
   condition that drives backward-induction unravelling.
3. **Roughly 88% of the variance in game-level cooperation is residual** — i.e.
   run-to-run, within an identical condition (F21). Any single-run claim about
   an LLM's "strategy" is unreliable; the design's 10 replicates per cell are
   the minimum, not a luxury.
4. Archetype labels are *nearest canonical rule*, not identity — and how often
   they are provable varies enormously (F26e, `tables/T19`):

   | model | exact rule | several rules | LSTM nearest | mean deviations from nearest rule |
   |---|---|---|---|---|
   | Gemma-3-12B | **0.56** | 0.10 | 0.34 | 0.8 |
   | Mistral-Large | 0.39 | 0.13 | 0.48 | 1.0 |
   | GPT-4o | 0.17 | 0.10 | 0.72 | 1.7 |
   | Qwen3-8B | 0.17 | 0.17 | 0.66 | 3.8 |
   | Claude-3.5-Haiku | 0.12 | 0.01 | 0.86 | 2.0 |
   | Llama-3.1-8B | **0.002** | 0.000 | **0.998** | **9.4** |

   Llama-3.1-8B's play violates its nearest rule on 9.4 rounds out of 30 — its
   "AllC" label is a weak nearest-neighbour statement, not a strategy. Gemma is
   the opposite: 38% of its games are *exactly* AllD.

5. **The earlier draft over-reported WSLS.** Forcing an argmax assigned the
   ambiguous `EC RC RC …` pattern to WSLS on a 0.375-vs-0.345 posterior margin,
   which inflated Qwen3's WSLS share to 0.218 and Gemma's to 0.119. Exact WSLS
   matches are 0.3% (frontier) and 0.0% (open-weight): **no model in this corpus
   plays Win-Stay-Lose-Shift**. Those games are now labelled *Ambiguous*.

## Mining the residual: what the 67% actually is

Two thirds of LLM play matches no canonical rule exactly, so `scripts/11`
widens the hypothesis class in steps and attaches a permutation null to each
(the focal player's own actions are shuffled, holding its cooperation rate and
the opponent's real sequence fixed).

| hypothesis class | frontier obs / null | open-weight obs / null |
|---|---|---|
| the 4 canonical rules | 0.00 / 0.00 | 0.00 / 0.00 |
| all 32 deterministic memory-one rules | 0.18 / 0.05 | 0.07 / 0.01 |
| …plus one mid-game regime switch | 0.59 / 0.31 | 0.18 / 0.02 |
| **still unexplained** | **0.41** | **0.82** |

Read the 30-round column: it is the stronger test, because a two-segment fit
over ten rounds has enough freedom to explain 31% of *shuffled* play.

Four things fall out:

1. **The canonical vocabulary is not merely incomplete, it is the wrong
   shape.** Widening from 4 to all 32 deterministic memory-one rules recovers
   only 7% of the 30-round residual. What is missing is not more rules.

2. **The residual is genuinely history-dependent, and deeper than memory-one.**
   On rounds 3+, memory-two beats memory-one beats memoryless by BIC for every
   model (T22). Per-round ΔBIC against the memoryless baseline ranges from
   −1.04 (Gemma-3-12B) to **−0.05 (Llama-3.1-8B)** — Llama's play is close to a
   biased coin, which is the same story its 9.4 deviations tell.

3. **There are no discrete non-canonical archetypes.** Silhouette declines
   monotonically from k=2 (T23), and in memory-one profile space the residual
   is one dense cloud sitting *between* the canonical corners rather than near
   any of them (F28c). Reporting "LLMs play strategy X" is the wrong frame;
   they occupy a continuum.

4. **Regime switching is the one real structure.** 41% of the frontier residual
   and 11% of the open-weight residual is two canonical rules with a single
   switch, over null rates of 31% and 2%. Switches cluster in the first third
   of the game, and the dominant transitions are AllC→AllD (0.080) and
   AllD→AllC (0.046). Mistral-Large switches in 47% of its residual games,
   GPT-4o 43%, Gemma 41%; **Llama-3.1-8B never does (0.00)** — consistent with
   its play carrying almost no history.

Deviations also land in a state-dependent way (T21): Gemma violates its nearest
rule on only 3% of R rounds but 16% of S rounds, while Llama violates at
~0.30 in every state, i.e. uniformly, which is what noise rather than a rule
looks like.
