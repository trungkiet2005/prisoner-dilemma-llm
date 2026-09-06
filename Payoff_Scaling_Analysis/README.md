# Payoff scaling analysis

Full data-mining analysis of `Dataset/data_fairgame_frontier_llm`, the six-model
frontier prisoner's-dilemma corpus with a ten-level payoff-scale ladder.

## Reproduce

```bash
cd Payoff_Scaling_Analysis
python analysis.py                 # full run, reusing the parsed cache
python analysis.py --rebuild       # re-parse all 300 raw CSV files first
python analysis.py --only 04 09    # selected sections only
python analysis.py --report-only   # regenerate the two markdown documents
```

The first run parses the corpus into three tidy tables and caches them as parquet
in `cache/`. Everything downstream reads the cache, so a full rerun takes minutes
rather than tens of minutes. Every bootstrap and permutation is explicitly seeded,
so numbers reproduce exactly.

## Read this first

| file | what it is |
|---|---|
| `analysis_report.md` | the full analytical report, 21 sections |
| `findings.md` | every recorded finding with evidence, robustness and caveats, ranked |
| `figure_inventory.csv` | every figure, the question it answers and its main finding |

## Layout

```
analysis.py              top-level reproducible runner
mining/                  the analysis package
  core.py                loader, derived variables, style, statistics helpers
  m01_audit.py           structural audit and data quality
  m02_distributions.py   univariate distribution mining
  m03_payoff.py          payoff-centric analysis
  m04_scaling.py         payoff-scale invariance and scaling
  m05_fairness.py        fairness and inequality
  m06_llm_comparison.py  systematic model comparison
  m07_game_theory.py     strategy and temporal dynamics
  m08_frontier.py        Pareto and frontier analysis
  m09_robustness.py      robustness and sensitivity
  m10_advanced.py        correlation, interactions, discovery
  m11_structure.py       dimensionality reduction, regimes, anomalies
  report.py              composes the two markdown documents
01_data_quality/ ... 10_advanced/    figures, .png and .pdf at 300 dpi
tables/                  every result table as CSV
cache/                   parsed parquet tables and the findings registry
```

## Three facts that decide whether an analysis of this corpus is right

1. **`OptionA` is DEFECT and `OptionB` is COOPERATE.** The prompt states payoffs as
   penalties and tells the agent to minimise, so the dominant action is the one with
   the lower number in both columns. Getting this backwards inverts every rate in the
   corpus. It is verified rather than assumed in `01_data_quality/Q03`.
2. **Rounds are not independent observations.** 240,000 agent-rounds are 24,000
   agent-games, 12,000 independent games and 1,200 design cells. Every interval here
   clusters on the game.
3. **Raw payoff scales with lambda at exponent 1 by construction.** That is the
   identity `payoff = lambda x base`, not a scaling law. All scaling analysis is done
   on scale-invariant behavioural quantities. See `03_payoff/P04`.

The corpus itself is documented in `Dataset/DATA_CARD_frontier_llm.md`.
