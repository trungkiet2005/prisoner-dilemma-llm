---
name: write-kaggle-benchmarks
description: Use this skill when the user wants to create, push, run, monitor, download, or publish Kaggle benchmark tasks using the Kaggle CLI and the Kaggle Benchmarks Python SDK for this repository.
---

# Write Kaggle Benchmarks

Use this skill when the user wants to create, push, run, monitor, download, or publish Kaggle benchmark tasks using the Kaggle CLI and Kaggle Benchmarks Python SDK.

## Trigger

Use this skill for requests involving:
- creating a new Kaggle benchmark task
- pushing a task to Kaggle
- running a benchmark against one or more models
- checking run status or logs
- downloading results or source notebooks
- publishing a benchmark task
- debugging benchmark workflow failures
- working with Kaggle benchmark task files like `kaggle/benchmarks/pd_task.py`

## Core rules for this repo

This repository contains a Kaggle benchmark task for the Prisoner's Dilemma benchmark. Follow these rules whenever working with model execution:

- Always drive model calls through Kaggle server-side execution only.
- Do not call models from local machine for benchmark validation or smoke tests when the task is intended to run on Kaggle.
- Prefer `kaggle benchmarks tasks push` and `kaggle benchmarks tasks run` for real smoke checks and production runs.
- Use local parity checks and prompt rendering only for non-model validation logic.
- Keep separate `KAGGLE_CONFIG_DIR` per Kaggle account to avoid credential collisions.
- If a run fails because of quota, rate limits, or a stale proxy key, switch accounts and rerun the smoke check on the replacement account rather than guessing from local diagnostics.

## Common commands

```bash
# Push a task snapshot to Kaggle
cd kaggle/benchmarks
python -m kaggle benchmarks tasks push prisoner-dilemma-fairgame -f pd_task.py --wait

# Run a single model on the server
python -m kaggle benchmarks tasks run prisoner-dilemma-fairgame -m gpt-5.4-nano-2026-03-17 --wait

# Watch the foreground run and download outputs
python kaggle/benchmarks/run_and_watch.py -m google/gemini-3.5-flash-lite
python kaggle/benchmarks/run_and_watch.py --attach

# Download task outputs to a temporary directory
python -m kaggle benchmarks tasks download prisoner-dilemma-fairgame -o D:/tmp/kb

# Validate prompt parity without calling a model
PD_SKIP_RUN=1 python -m pytest kaggle/benchmarks/test_pd_task_parity.py -q
```

## Important repo-specific guardrails

### 1. Default values in `pd_task.py` are the real run configuration

The server executes the snapshot that was pushed. Local env vars do not override the server-side task defaults unless the task file itself is modified and pushed again.

Before a real run:
- confirm `LAMBDAS`, `LANGS`, `REPS`, `N_ROUNDS`, and `RUN_TAG` in `kaggle/benchmarks/pd_task.py`
- ensure the task file is pushed again after any config change
- avoid running a full sweep against stale default settings

### 2. Smoke checks must be real server-side checks

For this repo, `PD_SMOKE_MODELS` and `_SMOKE_ONLY` logic is used to avoid accidental full-sweep execution during a task push.

Use a real server-side smoke run to check:
- model availability for the account
- whether the model accepts the benchmark task schema
- whether the account still has quota headroom

### 3. Prompt encoding and parity checks matter

This benchmark includes multilingual prompt templates (ar, cn, vn, fr, en). Encoding issues can silently corrupt the prompt text sent to models.

Always run the parity test before trusting new output:

```bash
PD_SKIP_RUN=1 python -m pytest kaggle/benchmarks/test_pd_task_parity.py -q
```

### 4. Watch for benchmark-specific failure modes

Common issues in this repo include:
- stale default config values being pushed without awareness
- model slug mismatch between CLI and server-side `MODEL` naming
- proxy-local vs server-side differences
- `reasoning_effort` rejected by the proxy
- prompt / parse issues caused by low output token caps or thinking models
- run resume mixing in stale output shards from earlier failed runs

## Good workflow for this repo

1. Update the task defaults in `kaggle/benchmarks/pd_task.py`.
2. Run the parity test.
3. Set a dedicated `KAGGLE_CONFIG_DIR` for the account used for the run.
4. Push the task snapshot.
5. Run the chosen model via Kaggle server-side benchmark runner.
6. Watch progress and fetch logs.
7. Download outputs to a temp directory such as `D:/tmp/kb`.
8. Collect only valid shards into `Dataset/data_fairgame_frontier_llm` using the repository’s collection script, not raw `cp`.
9. Check row counts and completeness before accepting the data.

## Output validation checklist

Before treating benchmark outputs as valid, verify:
- `n_games` matches the expected count for the current lambda grid
- `fallback_rate` is near zero
- `parse_fail_rate` is near zero
- `complete` is true
- there are no missing `coop_by_scale` values
- each lambda has the expected number of data rows

## Example execution notes

```bash
export KAGGLE_CONFIG_DIR="D:/AI_PhD/GameTheory/kaggle_for_research/kaggle-api/chunaiu"
export KAGGLE_API_TOKEN="KGAT_xxxxxxxx"

cd kaggle/benchmarks
python -m kaggle benchmarks tasks push prisoner-dilemma-fairgame -f pd_task.py --wait
python -m kaggle benchmarks tasks run prisoner-dilemma-fairgame -m gpt-5.4-nano-2026-03-17 --wait
```

## When to stop and ask for confirmation

Ask the user before proceeding if any of these are true:
- the task will consume a large quota or multiple accounts
- the benchmark configuration changes lambda grid, model set, or weight format unexpectedly
- a model test is intended to replace a previously accepted dataset
- the user wants to publish a task or change task visibility

## Repository-specific file references

Relevant files in this repo:
- `kaggle/benchmarks/pd_task.py`
- `kaggle/benchmarks/run_and_watch.py`
- `kaggle/benchmarks/collect_run.py`
- `kaggle/benchmarks/test_pd_task_parity.py`
- `Dataset/data_fairgame_frontier_llm/`
- `CLAUDE.md`
