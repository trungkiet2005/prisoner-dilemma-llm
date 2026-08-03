# FAIRGAME — Iterated Prisoner's Dilemma with LLM agents

Cross-lingual, cross-payoff study of how LLM agents play a repeated Prisoner's
Dilemma, built on [FAIRGAME](FAIRGAME/README.md). Two model families
(frontier API models and open-weight models), five languages, and a payoff-scale
sweep, analysed into publication-ready figures and tables.

```
Prisoner_Dilemma_Game/
├── FAIRGAME/     # the simulation framework (game engine, prompts, configs, connectors)
├── kaggle/       # how the runs are executed: 7 open-source models + Gemini   ← start here to re-run
├── Dataset/      # collected runs, one CSV per (payoff scale, model, language)
├── Analysis/     # ingest → metrics → 26 figures + tables (`python Analysis/run_all.py`)
└── reference/    # the FAIRGAME paper
```

## The game

Two agents, iterated Prisoner's Dilemma, no communication. Each agent is given a
personality (`cooperative` / `selfish`) and asked to minimise its penalty:

| | agent 2: OptionA | agent 2: OptionB |
|---|---|---|
| **agent 1: OptionA** | `w1`, `w1` | `w3`, `w2` |
| **agent 1: OptionB** | `w2`, `w3` | `w4`, `w4` |

with `w1=6, w2=10, w3=0, w4=2` (the *conventional* config) scaled by a factor λ.
`Analysis/` reads this as `T=10, R=6, P=2, S=0` with `OptionA` = cooperate.

## The two arms

| arm | models | payoff scales λ | rounds | horizon |
|---|---|---|---|---|
| frontier | Claude-3.5-Haiku, GPT-4o, Mistral-Large | 0.1, 1, 10 | 10 | hidden |
| open-weight | 7 open-source models (Qwen2.5 7/32/72B, Gemma-2 9/27B, Llama-3.1-8B, Llama-3.3-70B) | 0.01, 0.1, 1, 10, 100, 1000 | 30 | announced |
| API benchmark | Gemini (via Kaggle Benchmarks) | same 6 as open-weight | 30 | announced |

## ⚠️ Payoff correction — read before using `Dataset/data_fairgame_small_llm/`

The open-weight data currently in `Dataset/` was collected with the **`mild`**
config (`w1 = 8`, i.e. `R = 8`), while the frontier data used **`conventional`**
(`w1 = 6`, `R = 6`). That puts the two families at different points of the PD
greed/fear plane, so their cooperation levels are **not** directly comparable —
see the caveat already recorded in [`Analysis/README.md`](Analysis/README.md).

[`kaggle/`](kaggle/README.md) is the re-run that fixes this: both the
open-source arm and the Gemini arm are pinned to `conventional` (`R = 6`) and
extended to six payoff scales. When the new data replaces the old, update
`Analysis/pdlib/ingest.py`:

* add the new model `short_name`s to `MODEL_MAP`;
* change `_BASE_MATRIX["small"]` from `R: 8.0` to `R: 6.0`.

## Re-running

```bash
# parity test for the Kaggle Benchmarks arm (no model calls, no cost)
PYTHONUTF8=1 python -m pytest kaggle/benchmarks/test_pd_task_parity.py -q

# cheap smoke of the Gemini arm
cd kaggle/benchmarks && kaggle b auth -y
PD_LAMBDAS=1 PD_LANGS=en PD_REPS=1 PD_ROUNDS=5 PYTHONUTF8=1 python pd_task.py

# rebuild every figure and table from Dataset/
python Analysis/run_all.py
```

The 7 open-source models run in a Kaggle GPU notebook (Internet OFF, vLLM) —
see [`kaggle/README.md`](kaggle/README.md) for the input datasets, session
splitting, and resume behaviour.

## Licence

`FAIRGAME/` keeps its upstream licence (see [`FAIRGAME/LICENSE`](FAIRGAME/LICENSE)
and [`FAIRGAME/NOTICE.md`](FAIRGAME/NOTICE.md)).
