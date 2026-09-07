# Response to reviewers

We thank the reviewers for their careful and constructive assessments. We have
revised the manuscript and supplementary material while keeping the frozen
corpus and collection protocol unchanged. The revision adds no new model
calls. The main changes are described below.

## Reviewer 1

### Model endpoint reproducibility

We now state the endpoint identifiers, collection window and provider
limitations explicitly in the methods and data card. The released corpus and
analysis code are versioned in the public repository. The endpoint names are
descriptive labels supplied by the serving interface, not claims about hidden
model architecture. We therefore limit model-specific conclusions to the
observed endpoints and state that replication across future endpoint versions
is an open question.

### Arabic and Chinese prompt wording

We now treat the five language conditions as distinct prompt stimuli rather
than equivalent translations. The Arabic and Chinese templates state payoffs
as penalties but phrase the objective as maximising a reward. The revised text
states that fixed wording preserves the within-language rescaling comparison,
but may interact with numerical magnitude. Cross-language differences are
therefore not attributed to language or culture alone. The formal
scale-by-language interaction is interpreted as a prompt-stimulus interaction.

### Persona semantics

The revision bounds the persona result to the exact prompt instructions and
reports the sign reversal descriptively. It does not claim that all endpoints
assign the same target to “selfish” or “cooperative”. The persona wording and
the absence of a hidden opponent persona are documented in the methods. A
notation-control analysis is also reported as a limited robustness check. A
cross-language comprehension manipulation would require a new collection and
is identified as future work.

### Decoding settings

The decoding configuration is now recorded in the data card and methods: the
requests set temperature to 1.0 and pass a requested seed where supported, but
provider-specific seed semantics are not assumed. A
temperature or top-p ablation would change the collection protocol and is not
needed to test the stated invariance null in the frozen corpus. We have made
this scope boundary explicit and identify decoding sensitivity as future work.

### Self-play and horizon

The manuscript now states that the design studies same-model dyads and a known
ten-round horizon. The opening-move analysis isolates the part of the effect
that appears before interaction, while the later-round and conditioning results
are not presented as horizon-invariant. Cross-model play and unknown-horizon
conditions remain future extensions.

### Strategy attribution

We retain the four canonical memory-one rules as a readable reference set, but
we now consistently call unmatched labels “resemblances” rather than direct
strategy identifications. Provenance, classifier validation, continuous rule
distance and the limitation to ten rounds are reported in the manuscript and
appendix. Extortionate, zero-determinant and longer-memory strategies are not
claimed to be excluded.

### Parsing and fallback

The completeness guard checks every cell and the data card documents the
fallback rule and its direction of bias. The manuscript now states that this
guard does not establish semantic equivalence across endpoint versions. A
cell-level fallback audit would require reopening the collection pipeline and
is outside the frozen analysis; the limitation is recorded rather than hidden.

### Reciprocity wording

We changed the title and result language from “cooperate without reciprocity”
to “stronger persistence than reciprocity”. The supplementary table reports
whole-dyad bootstrap intervals for each model, including positive reciprocity
for four models and negative estimates for Gemini 3.1 and Qwen. The conclusion
is a relative model-level comparison, not a claim that reciprocity is absent.

### Related work

The introduction now positions the study directly against FAIRGAME and the
authors’ earlier smaller payoff-scaling collection. The contribution is the
theorem-backed zero-effect test over a balanced multilingual, multi-model
scale grid, together with per-model heterogeneity and the pooling analysis.
We do not compare against an extension that uses the same payoff intervention
as the present manuscript.

### Requested additional experiments

Integer-unit, harmonised-template, temperature, cross-model, longer-horizon,
comprehension and alternative-game studies would be valuable follow-up work.
They would require new model calls and answer broader questions than the
invariance test addressed here. We have instead strengthened the existing
evidence with matched-ladder permutations, notation control, leave-one-model-
out checks, language-by-scale interaction tests, threshold sensitivity and
test-retest data already present in the release.

## Reviewer 2

### Statistical uncertainty and clustering

Primary intervals use whole-dyad bootstrap resampling. Primary per-model and
trend regressions cluster on the dyad. The supplementary scale-by-language
model uses documented language-by-game-ID blocks to preserve matched
ten-scale ladders. The methods and appendix now state this distinction
explicitly. Figure 3 and Figure 6 descriptive intervals were regenerated with
whole-dyad bootstrap resampling as well.

### Strategic polarity and comprehension

The methods report the three strategic-polarity correlations and the
verification ledger now recomputes them from the released game table. The
result is used only to establish the payoff ordering of the dilemma, not to
claim that an endpoint has human-like comprehension.

### Numeric formatting and notation

The main grid is intentionally retained because it is the protocol under
study. The supplementary notation-control slice changes decimal formatting at
fixed magnitudes and weakens the explanation that trailing zeros alone drive
the effect. We now state that it does not identify a complete magnitude versus
formatting mechanism.

### Evolutionary baseline

The baseline is explicitly a four-rule reference model, not a complete model
of every possible language-model policy. The appendix reports an 80-setting
sensitivity sweep and states that the empirical comparison is descriptive and
does not identify a mechanism. Persistence-biased extensions are a natural
future analysis.

### Repository and verification

The public README and data card now match the 20-page main manuscript and
23-page supplement, including the fallback, decoding and matched-seed caveats.
Generated tables and figures remain pipeline outputs.
The verification script checks 197 headline quantities, including the three
strategic-polarity correlations and the matched-block robustness results; the
final run reports that all values match the data.

We appreciate the reviewers’ suggestions. The revised manuscript now makes
the supported contribution sharper while stating the endpoint, prompt,
decoding, horizon and strategy-readout boundaries directly.
