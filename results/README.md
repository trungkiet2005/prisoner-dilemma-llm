# Kaggle Benchmarks runs

Raw output of the Gemini arm, downloaded from the Kaggle Benchmarks task
`prisoner-dilemma-fairgame` by `kaggle/benchmarks/run_and_watch.py`. This is the
unprocessed record: the CSVs that `Dataset/` was built from, and one JSON
checkpoint per game holding the full transcript.

## Layout

```
results/kbench/exp<E>/run<ID>/<model_tag>/
├── <lambda>/<model_tag>/x<lambda>_<lang>_<model_tag>.csv   # one row per dyad
└── checkpoints/x<lambda>__lang-<l>__p<persona>__rep-<n>.json
```

`<E>` is the experiment number of `legacy/paper_scaling/RUN_PLAN.md`, `<ID>` the
Kaggle run id, and `<model_tag>` carries the `-rep2` suffix on a repeat run. The
runs held here:

| exp | run id | model tag | payoff scales λ |
|---|---|---|---|
| 1 | 971013 | `gemini-3-flash-preview` | 1 |
| 2 | 359521 | `gemini-3-flash-preview` | 1 |
| 2 | 971018 | `gemini-3-flash-preview` | 1 |
| 2 | 359522 | `gemini-3.5-flash-lite` | 0.1, 1, 10 |
| 2 | 971021 | `gemini-3.5-flash-lite` | 0.01, 0.25, 0.5, 2, 5, 100, 1000 |
| 4 | 1029369 | `gemini-3-flash-preview-rep2` | 1 |
| 4 | 1029391 | `gemini-3.5-flash-lite-rep2` | 0.1, 0.25, 0.5, 1 |

Each λ directory holds 200 dyads: five languages by forty games.

## Promoting a run into `Dataset/`

Never copy by hand. `collect_run.py` checks the row count of every cell and
strips the `-rep2` suffix out of directory names, file names and the
`agent1_llm` / `agent2_llm` columns alike, all of which have silently corrupted
an analysis before:

```bash
python kaggle/benchmarks/collect_run.py results/kbench/exp4/run1029391 \
    --dest Dataset/data_fairgame_frontier_llm --strip="-rep2" --dry-run
```

## Downloading a new run

`kaggle b t download` writes a deeper tree than the one stored here: it wraps
every run in `<task>/<exp>/<model>/<run id>/results/kbench/`. The four constant
levels are dropped on the way in. To normalise a fresh download into the same
shape:

```bash
python kaggle/benchmarks/run_and_watch.py --dest results/kbench   # then, per run:
mv <dest>/<task>/<E>/<model>/<ID>/results/kbench/<tag> results/kbench/exp<E>/run<ID>/<tag>
```

Everything that reads this tree (`collect_run.py`, `run_and_watch.py`) globs for
`x*.csv` recursively and takes the model and λ from the last two path
components, so the levels above them carry no meaning to the code.
