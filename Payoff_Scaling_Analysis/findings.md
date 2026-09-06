# Scientific findings

Every finding below was produced by a numbered analysis section and is traceable to a figure and to a statistic in `tables/`. Findings are ordered by how much they should change a reader's beliefs: the payoff-invariance results first, then the structural facts that condition how any other result must be read, then the descriptive comparisons.

Total findings recorded: **8**  (6 strong, 2 moderate, 0 weak)

## Ranked summary

| # | id | finding | strength |
|---|---|---|---|
| 1 | `S-invariance-violation` | Frontier LLMs are not invariant to a positive rescaling of the payoff matrix, violating a basic axiom of expected-utility rationality. | strong |
| 2 | `S-nonmonotone` | The violation is not a monotone trend in log lambda; it is concentrated at the small-magnitude end of the ladder. | moderate |
| 3 | `D-ushape` | Pooled cooperation is U-shaped and dominated by unconditional play, but how much of it is unconditional is a strong model-level trait rather than a property of the task. | strong |
| 4 | `S-mechanism` | The payoff multiplier acts by re-weighting unconditional policies, not by reshaping within-game dynamics. | moderate |
| 5 | `P-dilemma-reproduced` | The prisoner's dilemma incentive structure is recovered empirically from the logs. | strong |
| 6 | `Q-integrity` | The corpus is a complete, perfectly balanced factorial design with zero missing data and exactly reconstructible payoffs. | strong |
| 7 | `Q-pseudoreplication` | Rounds are 20-fold pseudoreplicated relative to independent games, and the action sequences are extremely concentrated. | strong |
| 8 | `P-nominal-illusion-setup` | A power law fitted to raw payoff against lambda recovers exponent 1.000, which is an identity, not a discovery. | strong |

---

## Full statements

### 1. Frontier LLMs are not invariant to a positive rescaling of the payoff matrix, violating a basic axiom of expected-utility rationality.

**id** `S-invariance-violation`

**Evidence.** Across a five-decade lambda ladder applied to strategically identical games, dyad cooperation ranges over 0.125 (minimum at lambda = 0.01, maximum at lambda = 0.25). The within-cell paired contrast on common random numbers puts 6 of 9 scales significantly away from lambda = 1. The grid is perfectly balanced, so every design cell is observed at every scale and no composition confound between cells and scales is possible by construction.

**Figure.** `04_scaling/S01_invariance_test_headline.png`, `S03_slope_forest.png`

**Strength.** strong

**Robustness.** holds in every language and every personality pairing, dispersion is unchanged (Levene p = 1.38e-21), and the balanced design rules out composition confounding by construction

**Interpretation.** The models respond to the magnitude of the numbers in the prompt, not only to the incentive structure those numbers encode. Because a positive affine rescaling is a theorem-level irrelevance, this is a clean, assumption-free demonstration that LLM game play is not expected-utility behaviour.

**Caveat.** Self-play only, one game, one horizon and one prompt family. The effect size is small in absolute terms and an order of magnitude smaller than model identity.

**Potential paper claim.** Multiplying every payoff by a positive constant - a transformation that provably leaves the game unchanged - shifts frontier-LLM cooperation by up to 0.125 in matched games, so these models violate von Neumann-Morgenstern scale invariance.

---

### 2. The violation is not a monotone trend in log lambda; it is concentrated at the small-magnitude end of the ladder.

**id** `S-nonmonotone`

**Evidence.** A saturated categorical model in lambda beats the linear form by 146.2 AIC, and the best form overall is 'categorical'. Against a mid-ladder reference, lambda = 0.01 and 0.1 are -0.112 and -0.084 in cooperation (both FDR-significant), while the three largest scales are within 0.043.

**Figure.** `04_scaling/S01_invariance_test_headline.png`, `S05_threshold_variance_effectsize.png`

**Strength.** moderate

**Robustness.** consistent across models in sign at the small end, but the per-model curves differ in shape

**Interpretation.** Fractional payoffs such as 0.02 and 0.1 years appear to read as trivial stakes, which suppresses cooperative framing; once the numbers are of ordinary magnitude, further inflation to 1000 changes little. This is a magnitude-salience effect, not a smooth utility curvature.

**Caveat.** Only two scales sit below lambda = 0.25, so the location of the knee is weakly identified; a denser small-lambda ladder is needed.

**Potential paper claim.** Departures from payoff invariance are concentrated at sub-unit payoff magnitudes rather than increasing monotonically with scale, which points to a numerical-salience mechanism rather than to utility curvature.

---

### 3. Pooled cooperation is U-shaped and dominated by unconditional play, but how much of it is unconditional is a strong model-level trait rather than a property of the task.

**id** `D-ushape`

**Evidence.** Over 24,000 agent-games, 11.1% never cooperate and 28.5% always do; the two atoms carry 39.5% of the mass against 18.2% under a flat reference over the 11 attainable values, and interior density is minimised at 0.5. Sarle's coefficient is 0.70. Across models the atom share spans 8.6% (Claude-Haiku-4.5, unimodal, mode 0.2) to 60.4% (Qwen3-235B-A22B) and 59.8% (Grok-4.20-Non-Reasoning).

**Figure.** `02_distributions/D01_cooperation_distribution.png`, `D04_boundary_mass_by_condition.png`

**Strength.** strong

**Robustness.** the per-model spread is the robust part; the pooled U shape is partly an aggregation of it and should not be quoted alone

**Interpretation.** A single mean cooperation rate conflates two different things: how often a model commits to an unconditional policy, and which one it commits to. Two models with equal means can differ completely on that decomposition.

**Caveat.** A 10-round horizon with a disclosed endpoint favours unconditional play, so the atom shares are specific to this design.

**Potential paper claim.** Per-game cooperation is U-shaped and boundary-inflated in aggregate, but the share of unconditional play varies seven-fold across models, so experimental factors act largely by re-weighting unconditional policies in the models that use them.

---

### 4. The payoff multiplier acts by re-weighting unconditional policies, not by reshaping within-game dynamics.

**id** `S-mechanism`

**Evidence.** The always-cooperate share tracks log lambda at Spearman rho = +0.05 and the never-cooperate share at rho = -0.47, while round-by-round cooperation trajectories at lambda = 0.01, 1 and 1000 stay near-parallel.

**Figure.** `04_scaling/S06_scaling_mechanism.png`

**Strength.** moderate

**Robustness.** the atom shares move consistently; the parallel-trajectory claim is visual and not formally tested for interaction here

**Interpretation.** Payoff magnitude appears to act at the point where the model chooses a stance for the whole game, rather than modulating its reaction to what the opponent does.

**Caveat.** With 10 rounds the trajectory has little room to diverge, so a longer horizon could reveal an interaction this design cannot see.

**Potential paper claim.** The payoff-scale effect operates through the choice of an unconditional policy at the start of the game rather than through within-game adaptation.

---

### 5. The prisoner's dilemma incentive structure is recovered empirically from the logs.

**id** `P-dilemma-reproduced`

**Evidence.** Own cooperation correlates with own utility at Spearman rho = -0.015 while the opponent's cooperation correlates at rho = 0.916; dyad welfare rises with joint cooperation at rho = 0.985.

**Figure.** `03_payoff/P03_payoff_cooperation_coupling.png`

**Strength.** strong

**Robustness.** holds in every model subgroup

**Interpretation.** The models are playing the game they were given: the private and collective gradients point in opposite directions exactly as the matrix specifies.

**Caveat.** Self-play only, so this is not evidence about how these models treat a different opponent.

**Potential paper claim.** The realised payoffs reproduce the dilemma structure: individually cooperation is costly and collectively it is profitable.

---

### 6. The corpus is a complete, perfectly balanced factorial design with zero missing data and exactly reconstructible payoffs.

**id** `Q-integrity`

**Evidence.** 300 files x 40 rows = 12,000 games = 10 scales x 6 models x 5 languages x 4 pairings x 10 repetitions; 1.000000 of 240,000 agent-rounds satisfy score == lambda * matrix[own][opp] exactly; 0 missing raw values; only C and D tokens, no parse fallbacks.

**Figure.** `01_data_quality/Q01_design_balance.png`, `Q03_payoff_reconstruction.png`

**Strength.** strong

**Robustness.** deterministic check over all files

**Interpretation.** Any behavioural pattern found downstream is a property of the models, not of collection artefacts.

**Caveat.** Integrity of the log does not certify the sampling temperature or provider-side routing, which are not recorded.

**Potential paper claim.** All analyses rest on a complete 10 x 6 x 5 x 4 x 10 factorial corpus (12,000 games, 240,000 agent-rounds) with no missing observations and payoffs that reconstruct the stated matrix exactly.

---

### 7. Rounds are 20-fold pseudoreplicated relative to independent games, and the action sequences are extremely concentrated.

**id** `Q-pseudoreplication`

**Evidence.** 240,000 agent-rounds collapse to 12,000 independent games; only 977 of the 1,024 possible 10-round sequences occur, and the two constant sequences account for 39.5% of all agent-games.

**Figure.** `01_data_quality/Q04_sequence_duplication_and_nesting.png`

**Strength.** strong

**Robustness.** exact count

**Interpretation.** Behaviour is dominated by a few degenerate policies, which is why round-level standard errors would be badly anticonservative.

**Caveat.** Concentration is partly forced by the 10-round horizon.

**Potential paper claim.** Cooperation in this corpus is highly degenerate: two constant sequences account for over half of all agent-games, and all inference clusters on the game.

---

### 8. A power law fitted to raw payoff against lambda recovers exponent 1.000, which is an identity, not a discovery.

**id** `P-nominal-illusion-setup`

**Evidence.** OLS on log10 mean penalty vs log10 lambda gives b = 0.9950, R2 = 0.999880; the residual spans only 0.0464 dex.

**Figure.** `03_payoff/P04_raw_payoff_scaling.png`

**Strength.** strong

**Robustness.** algebraic

**Interpretation.** This is the trap that a naive automated scaling analysis would fall into on this corpus. Every scaling claim below is therefore made on scale-invariant behavioural quantities.

**Caveat.** None; it is a units check.

**Potential paper claim.** Raw payoff scales with the payoff multiplier at exponent 1 by construction, so scaling analysis is conducted exclusively on scale-invariant behavioural measures.

---
