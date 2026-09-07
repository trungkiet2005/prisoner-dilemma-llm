# Interface Focus Revision Guide — Version 2
## Target paper title
**Payoff scaling shapes cooperation in LLM agents across languages**

## Primary editorial decision
The **main paper will use only the five-model panel**:

- Claude Haiku 4.5
- GPT-5.4 Nano
- Gemini 3.5 Flash-Lite
- Qwen3 235B-A22B
- Grok 4.20

Main-paper corpus counts must therefore be stated consistently as:

- **10,000 dyads**
- **20,000 agent-games**
- **200,000 decisions**

The sixth model, Gemini 3.1 Flash-Lite Preview, is **supplementary only** and must not be mixed into main-text figures, tables, headline statistics, or corpus counts.

---

# 1. Submission constraints from the invitation

The journal says the paper does not have a strict length limit, but asks authors to aim for approximately:

- **<= 8,000 words**
- **~250 words of tables**
- **~3 medium figures**
- **abstract/summary <= 200 words**

Therefore, the main revision objective is:

> Reduce the main manuscript to a focused five-model paper with approximately 4 main figures and 1 main table, while moving secondary analyses into the supplement.

The current abstract is already below 200 words, so abstract length is not the primary problem.

---

# 2. New central story

The revised paper should tell one clean story:

> Multiplying all payoffs by a positive scalar leaves the prisoner’s dilemma strategically unchanged under classical theory, yet cooperation in five frontier LLM agents changes substantially with the numerical payoff scale. The direction and magnitude of that response differ strongly across models and across prompt languages, and part of the effect is already present on the opening move before any strategic history exists. A finite-population evolutionary reference exhibits a qualitatively different scale response. Therefore, measured cooperation in LLM agents is not an intrinsic scalar property of a model: it depends on how an otherwise strategically equivalent payoff structure is numerically represented and linguistically framed.

This is the main-paper narrative.

Everything else should either:

1. directly establish this claim;
2. qualify this claim;
3. explain its measurement implications;

or be moved to the supplement.

---

# 3. Consequence of the new title

The new title is:

> **Payoff scaling shapes cooperation in LLM agents across languages**

This is substantially better for a focused submission because it no longer makes:

- persistence vs reciprocity;
- persona effects;
- strategy attribution;
- fairness;
- evolutionary dynamics

title-level claims.

Therefore:

## Move current §3.8 entirely to the supplement
The persistence-versus-reciprocity analysis no longer needs to occupy the main text.

Move:
- persistence;
- reciprocity;
- matching contrast;
- transition decomposition;
- PCA;
- current Figure 8;
- bootstrap uncertainty;
- the `pCD - pDC` technical identity;

to a supplementary section such as:

> **B.10 Conditioning on own and partner actions**

This is now a clean secondary analysis rather than a main-paper requirement.

---

# 4. Five-model-only policy for the main manuscript

## Main text should say

> We evaluate five frontier language-model endpoints in self-play across ten payoff scales and five prompt languages.

Use these counts everywhere in the main manuscript:

- 5 models
- 10 payoff scales
- 5 prompt languages
- 4 ordered persona pairings
- 10 replicates
- 10,000 dyads
- 20,000 agent-games
- 200,000 decisions

## Main text must not say

- "six models" when describing the primary analysis;
- "12,000 dyads";
- "24,000 agent-games";
- "240,000 decisions";

unless explicitly referring to supplementary data.

## Recommended transparency sentence in Methods

Use one concise statement such as:

> A sixth provisional preview endpoint was collected on the same grid but is reported only in the electronic supplementary material because its endpoint identifier was provisional and not reliably re-addressable; including it does not change the principal payoff-scaling conclusions.

This is enough.

Do not spend a full main-text paragraph explaining the sixth model.

---

# 5. Recommended final main-paper structure

# Abstract
Target: **160-190 words**

# 1. Introduction
Target: **1,000-1,150 words**

# 2. Material and methods
Target: **1,500-1,750 words**

Recommended subsections:

## 2.1 Experimental design
## 2.2 Payoff scaling and the invariance prediction
## 2.3 Models, languages, personas, and protocol
## 2.4 Statistical analysis

No standalone long strategy-readout subsection.

# 3. Results
Target: **2,500-2,800 words**

## 3.1 Payoff scaling changes cooperation across five LLM agents
Merge current §3.1 + essential current §3.3.

## 3.2 Scale sensitivity depends on prompt language
Use current §3.2 as the core.

## 3.3 Scale sensitivity is already present on the opening move
Condense current §3.6.

## 3.4 Comparison with an evolutionary reference
Condense current §3.7.

# 4. Discussion
Target: **1,200-1,400 words**

## 4.1 Payoff scaling as a measurement problem
## 4.2 Implications for multilingual LLM-agent evaluation
## 4.3 Limitations and future work

# 5. Conclusion
Target: **140-180 words**

## Preferred total
Aim for approximately:

**7,000-7,600 narrative words**

Do not aim exactly at 8,000.

---

# 6. Section-by-section revision map

## 6.1 Introduction

### Keep
- LLMs as strategic/autonomous agents.
- Cooperation as a machine-behaviour measurement problem.
- FAIRGAME as the closest protocol ancestor.
- Positive payoff scaling as a theoretically neutral transformation.
- Gap: current LLM cooperation studies usually evaluate one or a few numerical payoff matrices.
- Research question: does cooperation remain invariant under positive payoff scaling?
- Why multilingual evaluation matters.
- Short contribution list.

### Compress strongly
- broad review of retaliation/forgiveness/persona work;
- long survey of formatting sensitivity;
- repeated explanations of von Neumann-Morgenstern invariance;
- long explanation of machine behaviour as a field;
- lengthy discussion of previous related pilot work.

### Remove from main introduction
Detailed motivation for:
- learned strategy classification;
- canonical rule identification;
- persistence/reciprocity decomposition.

These are now supplementary.

### Target
Reduce current Introduction by approximately **40%**.

---

# 7. Methods revision

## Current §2.1 Framework

### Action
**Compress.**

Main text needs only:
- FAIRGAME provenance;
- self-play;
- ten rounds;
- five languages;
- ten scales;
- four persona pairings;
- ten replicates;
- 10,000 dyads / 20,000 agent-games / 200,000 decisions;
- completeness check in one sentence.

Move implementation details to Supplement A.1.

---

## Current §2.2 Payoff-scaled prisoner's dilemma and invariance

### Action
**Keep as core.**

This section is central to the paper.

Keep:
- base payoff matrix;
- penalty framing;
- `Option A = defect`;
- `Option B = cooperate`;
- ten λ values;
- positive scaling transformation;
- exact zero-effect theoretical prediction.

Move:
- full ten-scale payoff table -> A.3;
- repeated worked examples;
- redundant algebra.

### Important
Never alter the action coding:

- **Option A = D**
- **Option B = C**

This is confirmed by the data card and is load-bearing for every cooperation result.

---

## Current §2.3 Models, prompt languages, personas and protocol

### Action
**Keep but shorten substantially.**

Main text should contain:
- the five primary models;
- five languages;
- self-play only;
- temperature 1.0;
- four persona pairings;
- ten-round known horizon;
- no communication;
- one sentence on the sixth preview model being supplementary;
- one sentence on the Arabic/Chinese prompt artefact.

Move to supplement:
- exact endpoint strings;
- endpoint reproducibility discussion;
- seed semantics;
- collection dates;
- parsing fallback;
- full prompt-template details.

---

## Current §2.4 Strategy read-out

### Action
**Remove as a standalone main subsection.**

Move almost all of it to:
- A.5 rule base;
- A.6 classifier;
- B.5 full strategy results.

If the evolutionary reference in §3.4 still uses AllD/strategy labels, retain only a short paragraph in Methods:

> For the evolutionary comparison, agent trajectories were also read against four canonical memory-one rules using a provenance-aware rule-based/learned procedure; full definitions, classifier validation, and attribution provenance are given in the electronic supplementary material.

Do not retain:
- token alphabet details;
- LSTM architecture;
- ambiguity cases;
- synthetic corpus construction;
- per-class accuracy discussion.

---

## Current §2.5 Statistical analysis

### Action
**Keep inferential skeleton only.**

Main text should explain:
- dyads are the resampling unit;
- whole-dyad bootstrap;
- permutation test for scale sensitivity;
- clustered per-model logistic regressions;
- variance decomposition;
- prompt-language analysis.

Move implementation detail to A.7:
- exact resampling stream;
- matched-block variants;
- secondary regressions;
- robustness permutations;
- technical details of randomisation.

---

# 8. Results restructuring

# New §3.1 — Payoff scaling changes cooperation across five LLM agents

Merge:
- current §3.1;
- the essential part of current §3.3.

### Keep
- all five models violate scale invariance;
- per-model cooperation curves;
- cooperation range;
- strong heterogeneity;
- pooling attenuates the effect;
- model x scale interaction is much larger than a simple shared-scale story.

### Keep Figure 2
Current Figure 2 is one of the strongest figures in the paper.

Recommended panels:
- model x scale cooperation matrix;
- per-model curves;
- within-model range vs pooled range.

### Keep Table 1
This is the main inferential table.

### Move out
Move the fairness result from current §3.1 to Supplement B.9.

Move detailed policy-geometry/boundary-mass analysis from §3.3 to B.6.

---

# New §3.2 — Scale sensitivity depends on prompt language

Base this section on current §3.2.

### Keep
- scale response differs across prompt languages;
- the shape differs after removing language-specific mean cooperation level;
- effects remain model-dependent;
- scale sensitivity is a model × prompt-language property.

### Required caveat
Arabic and Chinese have an inherited wording inconsistency.

Therefore use:

- "prompt language";
- "prompt template";
- "prompt stimulus";
- "scale-by-prompt interaction".

Avoid causal language such as:
- "language causes...";
- "culture causes...";
- "Chinese prompts make models...".

### Persona result
Move current §3.4 **entirely to Supplement B.4** unless one sentence is useful in the Discussion.

The new title does not require persona to remain a main result.

This saves substantial text and one figure.

---

# New §3.3 — Scale sensitivity is already present on the opening move

Base this on current §3.6.

### Keep
- round 1 contains no strategic history;
- four of five models show an opening-move scale range comparable to or larger than later-round sensitivity;
- Qwen is the exception;
- therefore part of the scale association precedes strategic feedback.

### Do not claim
- mechanism;
- numerical threshold detection;
- backward induction;
- that later interaction is irrelevant.

### Move to B.8
- full round-by-round curves;
- boundary-atom analysis;
- opening vs rounds 2-10 ratios;
- full secondary statistics.

---

# New §3.4 — Comparison with an evolutionary reference

Condense current §3.7 to approximately **300-400 words**.

### Keep
- deterministic replicator dynamics are scale-neutral geometrically;
- finite-population Fermi imitation is scale-sensitive because scaling affects effective selection strength;
- the empirical five-model corpus exhibits a qualitatively different scale response;
- the comparison is a reference, not a mechanistic model of LLM cognition.

### Move to Supplement C
- derivations;
- fixation probabilities;
- numerical solver;
- precision audit;
- execution-noise details;
- 80-parameter robustness grid;
- full strategy composition.

### Important
If main Figure 4 uses canonical strategy labels, include one concise Methods pointer to Supplement A.5/A.6.

---

# 9. Analyses to move fully to supplement

## Current §3.4 Persona
Move to **B.4**.

Move:
- full persona curves;
- sub-unit inversion;
- Table 2;
- current Figure 4;
- detailed sign-flip discussion.

The main Discussion may mention persona as one secondary example:
> Supplementary analyses show that payoff magnitude can also modulate the behavioural effect of persona instructions.

No more is necessary in main.

---

## Current §3.5 Strategy composition
Move to **B.5**.

Move:
- full section;
- current Figure 5;
- current Table 3;
- TFT/WSLS provenance analysis;
- exact-rule/rule-distance details.

This is valuable science but not necessary for the new title.

---

## Current §3.8 Persistence versus reciprocity
Move fully to **B.10 / D.7**.

Move:
- current Figure 8;
- transition geometry;
- PCA;
- reciprocity;
- persistence;
- matching;
- uncertainty;
- technical contrast identities.

Because the new title no longer mentions this result, there is no reason to spend main-text space on it.

---

## Fairness / welfare
Keep entirely in **B.9**.

Do not retain the fairness paragraph in main §3.1.

---

# 10. Recommended main figure plan

The preferred main-paper figure count is now **4**.

## Figure 1 — Experimental overview
**Keep current Figure 1 unchanged.**

Only verify that its displayed corpus is consistently the five-model panel:

- 5 models;
- 10,000 dyads;
- 20,000 agent-games;
- 200,000 decisions.

Do not add the sixth preview model to this figure.

---

## Figure 2 — Core payoff-scaling result
Use/refine current Figure 2.

Panels:
A. model × payoff-scale cooperation matrix;
B. five per-model response curves;
C. within-model range versus pooled range.

This supports:
- scale sensitivity;
- heterogeneity;
- pooling/cancellation.

---

## Figure 3 — Language dependence and opening move
Build a new composite from current Figures 3 and 6.

Recommended panels:

A. prompt-language × payoff-scale cooperation heatmap;
B. centered scale-response curves by language;
C. opening-move cooperation by payoff scale;
D. opening-move scale range versus later-round range, or opening vs all-round cooperation.

This figure directly matches the new title:
> payoff scaling + cooperation + across languages.

Do **not** include persona unless the composition remains visually simple.

Preferred version: no persona in main Figure 3.

---

## Figure 4 — Evolutionary reference
Simplify current Figure 7.

Recommended:
A. evolutionary reference response across λ;
B. empirical five-model reference response;
C. direct comparison.

Avoid carrying a visually dense four-strategy decomposition if a simpler directional comparison communicates the point.

---

# 11. Figures to move to supplement

Move:

- current Figure 4 persona -> B.4;
- current Figure 5 strategy attribution -> B.5;
- detailed panels from current Figure 6 -> B.8;
- current Figure 8 persistence/PCA -> B.10;
- full evolutionary strategy figures -> C as needed;
- fairness Figure B.4 remains supplementary.

### Final main figure count
**4 figures**

This is much closer to the journal's "approximately 3 medium figures" guidance than the current 8.

---

# 12. Table plan

## Keep
### Table 1
Core per-model payoff-scale statistics.

This table directly supports the primary claim.

## Move
### Current Table 2
Persona -> B.4.

### Current Table 3
Rule agreement/distance -> B.5.

### Final main table count
**1 table**

---

# 13. Revised supplement organization

The supplement can absorb the moved material cleanly.

# A. Supplement to the methods
- A.1 corpus scope and completeness
- A.2 endpoints, seeds, parsing, fallback
- A.3 all ten payoff matrices
- A.4 prompt templates and framing artefact
- A.5 canonical rule base
- A.6 classifier training and validation
- A.7 resampling and testing

# B. Supplement to the results
- B.1 cooperation by model and payoff scale
- B.2 cooperation by model, language, and payoff scale
- B.3 variance decomposition
- B.4 persona effects
- B.5 strategy read-out
- B.6 why model curves cancel under pooling
- B.7 sixth provisional model
- B.8 opening-move detail
- B.9 fairness/welfare
- **B.10 persistence, reciprocity, matching, and PCA**

# C. Evolutionary baseline
Keep full theoretical/numerical details.

# D. Robustness
Keep:
- sixth-model robustness;
- matched scale ladders;
- influence checks;
- classifier edge cases;
- conditioning uncertainty;
- parsing/fallback sensitivity.

# E. Reproduction
Keep pipeline, software, package versions, and seeds.

---

# 14. How to handle the sixth model

The main paper should not mix six-model numbers with five-model numbers.

## Main text
Use only:
- 5 models;
- 10,000 dyads;
- 20,000 agent-games;
- 200,000 decisions.

## Supplement
Report Gemini 3.1 Flash-Lite Preview fully in B.7.

## Recommended main-text wording
One sentence is enough:

> A sixth provisional preview endpoint was collected on the same design and is reported separately in the electronic supplementary material; including it does not alter the principal payoff-scaling conclusions.

This preserves transparency without diluting the five-model narrative.

---

# 15. Recommended revised abstract

Target: **160-180 words**.

Suggested logic:

1. LLM agents increasingly make decisions in strategic settings.
2. Multiplying all payoffs in a game by a positive constant is theoretically strategy-preserving.
3. Test five frontier LLMs over:
   - ten payoff scales;
   - five prompt languages;
   - 10,000 self-play dyads.
4. Cooperation changes substantially with payoff scale in every model.
5. Direction and magnitude differ strongly across models.
6. Scale sensitivity also depends on prompt language.
7. Much of the effect is already visible on the opening move.
8. Practical implication: cooperation rates should not be reported as context-free model traits.

Do not include:
- persistence/reciprocity;
- strategy classifier;
- fairness;
- persona;
unless there is space after the core story is complete.

---

# 16. Possible abstract draft

**Do not treat this as frozen wording; regenerate after the final Results are fixed.**

> Large language models increasingly act as agents in strategic environments, yet their measured cooperative behaviour may depend on arbitrary features of how a game is represented. We test a theoretically neutral transformation: multiplying every payoff in an iterated prisoner’s dilemma by the same positive constant, which preserves preference rankings, best replies and equilibria. Across five frontier language models, ten payoff scales, five prompt languages and 10,000 self-play dyads, cooperation changes substantially under this rescaling. The magnitude and direction of the response differ strongly across models, so pooling can obscure large model-specific effects. Scale sensitivity also varies across prompt languages, showing that the numerical payoff representation and linguistic stimulus jointly shape measured cooperation. For four of five models, substantial scale sensitivity is already present on the opening move, before any strategic history is observed. These results show that cooperation rates in LLM agents are not context-free model traits and should be reported together with the payoff scale and prompt conditions under which they were measured.

Approximate target length: comfortably below 200 words.

---

# 17. Discussion restructuring

## New §4.1 Payoff scaling as a measurement problem
Merge current Discussion sections that repeat the theorem-backed null.

Focus on:
- invariance failure;
- model-specific response;
- why a single cooperation number is fragile.

---

## New §4.2 Implications for multilingual LLM-agent evaluation
Discuss:
- prompt-language dependence;
- reporting standards;
- benchmarking;
- governance implications;
- need to specify numerical representation.

Avoid claiming a cultural mechanism.

---

## New §4.3 Limitations and future work
Consolidate all caveats:

- self-play only;
- five primary model endpoints;
- one game family;
- known 10-round horizon;
- no communication;
- Arabic/Chinese prompt inconsistency;
- no causal mechanism identified;
- numerical magnitude is not isolated from every formatting property;
- sixth preview endpoint supplementary;
- stochastic provider behaviour/seeds not fully reconstructable.

Then propose:
- cross-model dyads;
- multiple game families;
- notation controls;
- unknown horizons;
- explicit numerical-format manipulations;
- cross-provider replication.

---

# 18. Terminology rules for the revision

Prefer:

- payoff scaling;
- positive rescaling;
- payoff magnitude;
- prompt language;
- prompt template;
- prompt stimulus;
- scale sensitivity;
- cooperation rate.

Avoid making "stakes" the formal term.

Use "stakes" only in intuitive prose.

The paper title says **Payoff scaling**, so use that phrase consistently.

---

# 19. Scientific consistency checks

## Action coding
Must remain:

- Option A = defect
- Option B = cooperate

## Primary corpus
Main paper:

- 5 models
- 10,000 dyads
- 20,000 agent-games
- 200,000 decisions

## Sixth model
Supplement only.

## Self-play
All main claims must be framed as same-model self-play.

Do not generalize to cross-model interaction.

## Horizon
Ten rounds, horizon disclosed.

Do not claim identification of backward induction.

## Language caveat
AR/CN contain inherited wording inconsistency.

Therefore cross-language differences are not clean causal language effects.

## Strategy labels
Supplement only.
Use "resemblance" rather than internal-policy identification.

## Mechanism
Do not claim payoff scaling reveals a specific cognitive mechanism.

Use:
- associated with;
- consistent with;
- suggests;
- does not identify mechanism.

---

# 20. Detailed move table

| ID | Current content | New location | Action |
|---|---|---|---|
| R01 | Long Introduction literature review | Main §1 | Cut 40% |
| R02 | §2.1 completeness detail | A.1 | Move |
| R03 | §2.2 full matrices | A.3 | Move |
| R04 | §2.3 endpoint details | A.2 | Move |
| R05 | §2.3 prompt artefact detail | A.4 | Move |
| R06 | §2.3 parsing/fallback detail | A.2 | Move |
| R07 | §2.4 strategy read-out | A.5/A.6/B.5 | Move nearly all |
| R08 | §2.5 resampling implementation | A.7 | Move |
| R09 | §3.1 core invariance | New §3.1 | Keep |
| R10 | §3.1 fairness | B.9 | Move |
| R11 | §3.2 language | New §3.2 | Keep/shorten |
| R12 | §3.3 pooling | New §3.1 + B.6 | Merge/move detail |
| R13 | §3.4 persona | B.4 | Move |
| R14 | Current Figure 4 | B.4 | Move |
| R15 | Current Table 2 | B.4 | Move |
| R16 | §3.5 strategy composition | B.5 | Move |
| R17 | Current Figure 5 | B.5 | Move |
| R18 | Current Table 3 | B.5 | Move |
| R19 | §3.6 opening move | New §3.3 | Keep concise |
| R20 | Detailed Fig.6 panels | B.8 | Move |
| R21 | §3.7 evolutionary baseline | New §3.4 | Keep concise |
| R22 | Evolutionary derivations/grid | C/D | Move |
| R23 | §3.8 persistence/reciprocity | B.10/D.7 | Move fully |
| R24 | Current Figure 8 | B.10 | Move |
| R25 | Discussion repetitive sections | New §4 | Merge |
| R26 | Conclusion | §5 | Cut to 140-180 words |

---

# 21. Figure redesign instructions for AI

## Figure 1
Do not redesign.

Check only:
- typography;
- five model labels;
- five-model corpus counts;
- export quality.

---

## Figure 2
Keep as the main payoff-scaling figure.

Do not overload it with:
- persona;
- strategy labels;
- fairness.

---

## New Figure 3
Redraw from current Figures 3 + 6.

Goal:
> Show that scale sensitivity depends on language and appears before interaction.

Recommended layout:
- top: language × scale heatmap;
- lower left: centered scale-response profiles;
- lower right: opening-move response.

Use the same model color mapping as Figure 2.

---

## New Figure 4
Redraw/simplify current evolutionary figure.

Goal:
> Show a clean reference contrast, not every theoretical detail.

Move dense technical panels to Supplement C.

---

# 22. Main-text figure/table budget

Recommended final state:

- Figure 1 — overview
- Figure 2 — core scaling
- Figure 3 — languages + opening
- Figure 4 — evolutionary reference
- Table 1 — primary statistics

Total:
- **4 main figures**
- **1 main table**

This is the preferred submission version.

---

# 23. AI execution workflow

## Phase 1 — inventory
Before editing:
1. Parse `.tex`.
2. Count narrative words by section.
3. List all figures/tables.
4. Record all current numeric headline claims.
5. Record all main/supplement cross-references.
6. Freeze pre-edit version.

---

## Phase 2 — structural moves
Perform in this order:

1. Move §3.8 + Figure 8 to B.10.
2. Move §3.5 + Figure 5 + Table 3 to B.5.
3. Move §3.4 persona + Figure 4 + Table 2 to B.4.
4. Move fairness content to B.9.
5. Merge core §3.3 into §3.1.
6. Split §3.6 into short main + B.8.
7. Compress §3.7; move technical detail to C/D.
8. Remove standalone strategy-readout Methods section.
9. Shorten §2.1, §2.3, §2.5.
10. Rewrite Introduction.
11. Rewrite Discussion.
12. Rewrite Abstract last.

Do not duplicate moved material.

---

## Phase 3 — corpus consistency pass
Search the entire main manuscript for:

- `six`
- `6 models`
- `12,000`
- `24,000`
- `240,000`
- `Gemini 3.1`

Any occurrence must be checked.

Default main-paper corpus must remain:

- five models;
- 10,000 dyads;
- 20,000 agent-games;
- 200,000 decisions.

Allowed exception:
one concise sentence pointing to the supplementary sixth-model analysis.

---

## Phase 4 — title consistency pass
The paper title is exactly:

> **Payoff scaling shapes cooperation in LLM agents across languages**

Check that:
- running title;
- PDF metadata;
- supplement title;
- cover letter;
- ScholarOne title;
- repository README;
- figure-generation metadata if applicable

all use the revised title.

---

## Phase 5 — word-count gate
Generate section-level word counts from source.

Preferred:
- 7,000-7,600 narrative words.

If above 7,800:
cut again.

Cut order:
1. Introduction;
2. Methods implementation detail;
3. repeated Result interpretation;
4. Discussion repetition;
5. long captions.

Do not cut:
- experiment definition;
- payoff matrix;
- theoretical zero-effect prediction;
- main statistical test;
- language caveat;
- self-play limitation.

---

# 24. Submission-ready checklist

## Main manuscript
- [ ] Title updated everywhere.
- [ ] Abstract <=200 words.
- [ ] Main narrative <=~8,000 words.
- [ ] Main corpus always 5 models / 10k / 20k / 200k.
- [ ] Figure count reduced to 4 preferred.
- [ ] One main table.
- [ ] All secondary analyses moved correctly.
- [ ] References compile.
- [ ] Figure/table numbering correct.

## Scientific checks
- [ ] Option A = D.
- [ ] Option B = C.
- [ ] Self-play stated.
- [ ] Ten-round known horizon stated.
- [ ] AR/CN caveat retained.
- [ ] Sixth model not mixed into main estimates.
- [ ] No unsupported causal mechanism claims.
- [ ] No strategy labels described as internal policies.

## Supplement
- [ ] Persona fully reported.
- [ ] Strategy analysis fully reported.
- [ ] Persistence/reciprocity fully reported.
- [ ] Fairness fully reported.
- [ ] Sixth model fully reported.
- [ ] Evolutionary technical derivations fully reported.
- [ ] Robustness checks retained.

## Submission package
- [ ] manuscript source;
- [ ] main PDF;
- [ ] supplement PDF;
- [ ] individual vector figures;
- [ ] data accessibility statement;
- [ ] ethics statement;
- [ ] AI-use statement;
- [ ] competing-interests statement;
- [ ] funding;
- [ ] reviewer-accessible data/code;
- [ ] ORCID linked.

---

# 25. Final recommended manuscript shape

The preferred submission should look like this:

## Main
1. Introduction
2. Material and methods
   - design
   - payoff scaling/invariance
   - five models/languages/protocol
   - statistics
3. Results
   - payoff scaling changes cooperation
   - language dependence
   - opening-move evidence
   - evolutionary reference
4. Discussion
5. Conclusion

## Main figures
- Fig. 1 overview
- Fig. 2 payoff scaling
- Fig. 3 language + opening move
- Fig. 4 evolutionary reference

## Main table
- Table 1 payoff-scale statistics

## Supplement
Everything else:
- sixth model;
- persona;
- strategy composition;
- rule classifier;
- persistence/reciprocity;
- PCA;
- fairness;
- detailed opening dynamics;
- evolutionary derivations;
- robustness;
- reproduction.

---

# 26. Final instruction to the revising AI

Do not try to preserve every current main-text result.

The revised paper should optimize for:

1. **one clear claim**;
2. **one consistent five-model analysis panel**;
3. **a short path from theoretical invariance to empirical violation**;
4. **cross-language evidence that directly supports the title**;
5. **opening-move evidence that strengthens interpretation**;
6. **a concise evolutionary reference that fits the issue theme**;
7. **full transparency through the supplement**.

The main paper should read as a focused research article, not as a catalogue of every analysis available in the project.
