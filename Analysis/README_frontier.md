# Frontier figure suite (FR1–FR26)

A journal-grade figure set covering **only the four frontier LLMs** in
`Dataset/data_fairgame_frontier_llm/`. Separate from the legacy `F01–F34` set
in [README.md](README.md), which also carries the open-weight arm.

```bash
python Analysis/run_frontier.py             # ~90 s, everything
python Analysis/run_frontier.py --no-build  # reuse the parsed parquet
```

FR18–FR26 read `models/strategy_lstm_h10.pt`, which
`scripts/05_train_classifier.py` produces. Nothing is trained here; the
frontier games are 10 rounds, so that checkpoint's horizon matches exactly.
`run_frontier.py` refuses to start if the checkpoint is missing.

Output: `figures/frontier/FR*.pdf` (vector, Type-42 fonts, for the
typesetter) and `FR*.png` (600 dpi, for drafts), plus `tables/T_FR*.csv` —
every number that appears in a figure is also written as CSV.

## House rules (enforced by `pdlib/natstyle.py`)

| rule | value |
|---|---|
| canvas | exact journal widths: 89 mm single, 120 mm intermediate, **183 mm double** — never rescaled by the typesetter |
| panels per figure | **at most 3**; ten of the twelve figures use 2 |
| type | Arial 7 pt labels, 6 pt ticks, **8 pt bold lowercase** panel letters, nothing under 5 pt |
| colour | Okabe–Ito, colour-blind safe, doubled by marker shape so the encoding survives greyscale |
| ink | white page, 0.5 pt spines, no top/right spine, no legend frame, no gradient behind data |
| files | vector PDF + 600 dpi PNG, `CreationDate` suppressed so reruns are byte-reproducible |

Two mechanics are worth knowing before editing the scripts.

**Panel letters cannot collide.** `finalize()` solves the constrained layout,
*freezes* it (`set_layout_engine("none")`), and only then stamps the letters —
anchored to each axes' **tight** bounding box, which already contains the tick
labels, axis label, title and any in-axes legend. Without the freeze,
`savefig(bbox_inches="tight")` re-solves the layout at save time and the
letters drift off their panels.

**Shared legends are pinned in figure coordinates.** `shared_model_legend()`
is called *after* `finalize`. `loc="outside lower center"` is the obvious tool
and it is the wrong one here: it is resolved by the layout engine that
`finalize` just switched off, so the legend is re-anchored at save time and
lands on the caption.

## The figures

**Part A — behaviour** (`21`, `22`, `23`, `25`)

| figure | panels | what it shows |
|---|---|---|
| `FR1_design` | 2 | stage game (T/R/P/S, greed, fear) · fully crossed design ladder |
| `FR2_payoff_scale` | 2 | cooperation vs λ with bootstrap bands · slope per decade of λ, 95 % CI |
| `FR3_outcomes` | 2 | CC / anti-coordination / DD composition · cooperation–efficiency plane |
| `FR4_language` | 2 | model × language cooperation heatmap · disparity against a permutation null |
| `FR5_persona` | 2 | cooperation by persona pairing · steerability by own vs opponent persona |
| `FR6_round_dynamics` | 2 | cooperation by round · opening vs final round |
| `FR7_reciprocity` | 3 | memory-one fingerprint · niceness–reciprocity plane · distance to nearest archetype |
| `FR8_synthesis` | 2 | z-scored behavioural signature · variance decomposition |
| `FR13_payoffs` | 2 | per-round payoff distribution · who profits in a mismatched pairing |
| `FR14_bimodality` | 2 | the discrete distribution of within-game cooperation · absorbing corners |
| `FR15_openings` | 2 | opening move by persona · what a cooperative opening buys |
| `FR16_reactivity` | 2 | perseveration–imitation plane · which history channel drives the next move |
| `FR17_invariance` | 3 | **audit**: role symmetry · replicate spread vs the independent-round floor · split-half reliability |

**Part B — per-model cards** (`24`)

| figure | panels | what it shows |
|---|---|---|
| `FR9`–`FR12_profile_*` | 3 | **one card per model**: language × λ surface · joint state by round · trajectory by persona |

**Part C — the strategy read-out** (`26`, `27`)

| figure | panels | what it shows |
|---|---|---|
| `FR18_calibration` | 3 | identifiability vs Bayes ceiling · confusion · degradation off-distribution |
| `FR19_archetypes` | 2 | nearest canonical archetype per model · how the label was reached |
| `FR20_archetype_conditions` | 2 | archetype mix by payoff scale · by persona pairing |
| `FR21_crystallisation` | 2 | posterior on the eventual label by round · when it crosses 0.90 |
| `FR22_rule_distance` | 3 | rule survival on LLM play · distance to nearest rule · provable coverage |

**Part D — the residual** (`28`)

| figure | panels | what it shows |
|---|---|---|
| `FR23_vocabulary` | 3 | coverage ladder vs a shuffled null · rules-per-game under-identification · excess over chance |
| `FR24_memory_depth` | 2 | ΔBIC across memory depths · where a two-regime fit puts the switch |
| `FR25_abstention` | 3 | four-way read-out composition · the confidence floor and its cost · abstention as a two-way tie |
| `FR26_geometry` | 3 | Nowak's reactive square by bucket · distance to the nearest corner · replicate coherence |

## Corpus

4 models × 3 payoff scales × 5 languages × 4 persona pairings × 10 replicates
= **2,400 dyads / 48,000 binary decisions**, fully balanced (40 dyads in every
model × λ × language cell), 10 rounds per game.

`OptionA` = cooperate, `OptionB` = defect. Payoffs in base units
`T = 10, R = 6, P = 2, S = 0`, multiplied by λ ∈ {0.1, 1, 10}.

## Caveats the figures carry

1. **Horizon is confounded with model.** The Gemini arm was run with
   `n_rounds_is_known = True` (`PD_ROUNDS_KNOWN=1` in
   `kaggle/benchmarks/pd_task.py`); Claude, GPT and Mistral were run with the
   horizon hidden. A known finite horizon is exactly the condition under which
   backward induction predicts unravelling, so the Gemini column is
   "model × horizon", not "model". `FR1b` marks it as a confound row and
   `T_FR01` records it per arm. `20_frontier_build.py` re-derives the flag from
   the logs and refuses to run if it disagrees with `natstyle.HORIZON`.
   FR6 is the figure where this matters most — and there the arm that *knew*
   the horizon is one of the three that warm up rather than unravel.

2. **80 rounds (4 dyads, 0.17 % of the corpus) carry ×1 payoffs inside the
   Claude / English / ×10 folder.** Written to `T_FR04_scale_anomalies.csv`.
   The realised scale is recovered per game from the payoffs themselves, not
   assumed from the folder name, so the affected dyads are labelled correctly
   in `scale_eff`; `scale_nominal` (used for grouping) still follows the folder.

3. **CD and DC shares are equal by construction.** Both agents of every dyad
   are rows in the corpus, so one agent's CD is the other's DC. Composition
   stacks merge them into a single anti-coordination band; the four-way split
   is kept only where the asymmetry between the two roles is the point.

4. **Intervals resample dyads, not rounds.** Rounds inside a game are strongly
   autocorrelated and the two agent-rows of a dyad share a history, so a
   round-level or row-level interval would be roughly √2 too narrow.

5. **The variance decomposition is sequential (Type I).** Factors are entered
   model → scale → language → persona, which credits shared variance to
   whichever is entered first. Model identity is entered first, so its share is
   an upper bound.

6. **An archetype label is the nearest canonical rule, not an identity.** Only
   12–41 % of agent-games are a *provable* exact match (FR19b); the rest are a
   nearest-neighbour statement from the LSTM, and play departs from its own
   nearest rule on 1.0–2.0 rounds out of ten. Any sentence of the form "model X
   plays TFT" has to carry that qualifier.

7. **Ten rounds under-identify the wider vocabulary.** A game rarely visits all
   four conditioning states, so rules differing only in an unvisited state are
   indistinguishable on it. Of the 4,800 agent-games, 2,192 match at least one
   of the 32 deterministic memory-one rules but only **89 match exactly one**
   (FR23b). Coverage numbers for the wider class are therefore statements about
   a *set* of rules.

8. **The 0.90 confidence floor is a convention, not a valley.** It was chosen on
   the 30-round corpus, where the posterior is bimodal. Over ten rounds it is
   not: moving the floor from 0.80 to 0.95 moves the abstained share from 0.16
   to 0.28 (`T_FR40b_floor_sensitivity.csv`, shown in FR25b).

9. **FR18 is measured on synthetic play, not on LLM play.** It is the
   instrument's calibration, where the generating strategy is known. It says
   nothing about how well an archetype label describes an LLM — that is FR19b
   and FR22.
