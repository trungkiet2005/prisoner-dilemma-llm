# Payoff scale and the iterated prisoner's dilemma with LLM agents

Two agents play a repeated prisoner's dilemma against each other. Neither can
talk to the other, each is told to be cooperative or selfish, and every payoff
in the matrix is multiplied by a constant. Under game theory that constant is
inert: it changes no preference, no best reply, no equilibrium. The models
behave as though it were not.

This repository holds the simulation framework, the collected corpora, the
analysis that turns them into figures and tables, and the manuscripts that
report the result. It is built on [FAIRGAME](FAIRGAME/README.md).

```
FAIRGAME/    the simulation framework: game engine, prompts, configs, connectors
kaggle/      how the runs are executed: 7 open-weight models on GPU, Gemini via Benchmarks
Dataset/     the collected corpora, one CSV per (payoff scale, model, language)
results/     raw Kaggle Benchmarks output, before it is promoted into Dataset/
Analysis/    ingest, statistics, strategy read-out, figures and tables
papers/      the manuscripts
legacy/      superseded drafts, kept because their numbers are cited elsewhere
reference/   the FAIRGAME paper
```

## The game

| | agent 2: OptionA | agent 2: OptionB |
|---|---|---|
| **agent 1: OptionA** | `w1`, `w1` | `w3`, `w2` |
| **agent 1: OptionB** | `w2`, `w3` | `w4`, `w4` |

Agents minimise a penalty, so `w1 = 6, w2 = 10, w3 = 0, w4 = 2` (the
*conventional* config) is the payoff matrix `T = 10, R = 6, P = 2, S = 0` that
`Analysis/` reads, with `OptionA` meaning cooperate. Every entry is then
multiplied by the payoff scale λ.

## The corpora

| arm | models | payoff scales λ | rounds | horizon |
|---|---|---|---|---|
| frontier | 6 models, incl. Claude Haiku 4.5, GPT-5.4 Nano, Gemini 3.5 Flash-Lite, Qwen3-235B, Grok 4.20 | ten values, 0.01 to 1000 | 30 | announced |
| open-weight | Qwen2.5 7/32/72B, Gemma-2 9/27B, Llama-3.1-8B, Llama-3.3-70B | 0.01, 0.1, 1, 10, 100, 1000 | 30 | announced |
| archived frontier | Claude-3.5-Haiku, GPT-4o, Mistral-Large | 0.1, 1, 10 | 10 | hidden |

The frontier corpus, `Dataset/data_fairgame_frontier_llm`, is the one the
current manuscript reports: 12,000 dyads, 24,000 agent-games, 240,000 decisions.

## Payoff correction: read before using `Dataset/data_fairgame_small_llm/`

The open-weight data was collected with the **`mild`** config (`w1 = 8`, so
`R = 8`) while the frontier data used **`conventional`** (`R = 6`). The two
families therefore sit at different points of the greed/fear plane and their
cooperation levels are **not** directly comparable. The caveat is recorded in
[`Analysis/README.md`](Analysis/README.md).

[`kaggle/`](kaggle/README.md) is the re-run that fixes this: both arms pinned to
`conventional` and extended to six payoff scales. When the new data replaces the
old, update [`Analysis/pdlib/ingest.py`](Analysis/pdlib/ingest.py) — add the new
`short_name`s to `MODEL_MAP`, and change `_BASE_MATRIX["small"]` from `R: 8.0`
to `R: 6.0`.

## Papers

| directory | manuscript | status |
|---|---|---|
| [`papers/interface-focus`](papers/interface-focus/README.md) | *Payoff scale reshapes how language models play the prisoner's dilemma* | **current**, Interface Focus RSFS-2026-0050 |
| [`papers/deduce-before-you-label`](papers/deduce-before-you-label/README.md) | *Deduce before you label: strategy attribution for LLM agents in the iterated prisoner's dilemma* | separate manuscript |
| [`legacy/paper_scaling`](legacy/README.md) | the pre-restructure draft of the Interface Focus paper | superseded |

Figures and tables in `papers/interface-focus` are generated, never typed. The
pipeline writes into that directory by default, so a number cannot drift between
the data and the manuscript.

## Reproducing

```bash
pip install -r requirements.txt

# rebuild the current manuscript's figures and tables from Dataset/
cd Analysis/scaling
python s00_build.py && python s01_train_lstm.py && python s02_readout.py
python s03_stats.py && python s04_strategy_stats.py
python s05_figures.py && python s05b_appendix_figures.py
python s06_tables.py && python s07_supplementary.py && python s09_egt.py
python s08_verify_paper.py     # recomputes every quoted number, fails on disagreement

# the older three-scale study
python Analysis/run_all.py
```

`s08_verify_paper.py` is the guard worth knowing about: it recomputes every
quantity the manuscript quotes and exits non-zero if any of them disagrees with
the data.

To send the pipeline at a different manuscript, set `PD_PAPER_DIR` and
`PD_FIGDIR`; unset, both resolve to `papers/interface-focus`.

## Re-running the collection

```bash
# parity test for the Kaggle Benchmarks arm, no model calls and no cost
PYTHONUTF8=1 python -m pytest kaggle/benchmarks/test_pd_task_parity.py -q

# cheap smoke of the Gemini arm
cd kaggle/benchmarks && kaggle b auth -y
PD_LAMBDAS=1 PD_LANGS=en PD_REPS=1 PD_ROUNDS=5 PYTHONUTF8=1 python pd_task.py
```

The open-weight models run in a Kaggle GPU notebook with Internet off, under
vLLM. See [`kaggle/README.md`](kaggle/README.md) for the input datasets, session
splitting and resume behaviour, and [`results/README.md`](results/README.md) for
how a finished run is promoted into `Dataset/`.

## Licence

`FAIRGAME/` keeps its upstream licence, see [`FAIRGAME/LICENSE`](FAIRGAME/LICENSE)
and [`FAIRGAME/NOTICE.md`](FAIRGAME/NOTICE.md).
