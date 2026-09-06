# Data card: `Dataset/data_fairgame_frontier_llm/`

Frontier-LLM prisoner's-dilemma corpus produced by the Kaggle Benchmarks task
`prisoner-dilemma-fairgame` (`kaggle/benchmarks/pd_task.py`), which reproduces the
FAIRGAME prompt templates byte for byte. This file is the self-contained reference
for anyone (human or model) writing analysis code against the corpus: paths, file
names, column semantics, action coding, verified invariants, and a runnable loader.

Everything below was verified by scanning all 300 CSV files on 2026-09-06. Numbers
quoted as "verified" are measured, not assumed.

---

## 1. At a glance

| property | value |
|---|---|
| root | `Dataset/data_fairgame_frontier_llm/` |
| game | iterated prisoner's dilemma, penalty framing, no communication |
| files | 300 CSV, UTF-8, comma-separated, 1 header row + 40 data rows each |
| on-disk size | ~6.2 MB |
| rows (games) | 12,000 |
| agent-games | 24,000 (each game contributes 2 agents) |
| agent-rounds | 240,000 (each agent-game is 10 rounds) |
| design | 10 payoff scales x 6 models x 5 languages x 4 personality pairings x 10 repetitions |
| rounds per game | 10, fixed, and the horizon is disclosed to the agents |
| missing data | none: every file has exactly 40 rows, no NaN, no truncated game |

The grid is **fully balanced and complete**: 10 x 6 x 5 x 4 x 10 = 12,000 games,
which is exactly the number of rows present.

---

## 2. Folder tree and path grammar

```
Dataset/data_fairgame_frontier_llm/
├── 0.01/                                  <- payoff scale lambda (directory name = the number)
│   ├── claude-haiku-4-5-20251001/         <- model slug (matches the agent1_llm / agent2_llm cell)
│   │   ├── x0.01_ar_claude-haiku-4-5-20251001.csv
│   │   ├── x0.01_cn_claude-haiku-4-5-20251001.csv
│   │   ├── x0.01_en_claude-haiku-4-5-20251001.csv
│   │   ├── x0.01_fr_claude-haiku-4-5-20251001.csv
│   │   └── x0.01_vn_claude-haiku-4-5-20251001.csv
│   ├── gemini-3.1-flash-lite-preview/     <- same 5 files, model slug substituted
│   ├── gemini-3.5-flash-lite/
│   ├── gpt-5.4-nano-2026-03-17/
│   ├── grok-4.20-0309-non-reasoning/
│   └── qwen3-235b-a22b-instruct-2507/
├── 0.1/     ... identical 6 model dirs x 5 language files
├── 0.25/    ...
├── 0.5/     ...
├── 1/       ...
├── 2/       ...
├── 5/       ...
├── 10/      ...
├── 100/     ...
└── 1000/    ...
```

**Path grammar (exact, no exceptions in the corpus):**

```
<root>/<lambda>/<model>/x<lambda>_<lang>_<model>.csv
```

* `<lambda>` in the directory name and in the file name are the **same literal string**
  (`0.01`, `0.1`, `0.25`, `0.5`, `1`, `2`, `5`, `10`, `100`, `1000`). Note there is no
  trailing `.0`: the scale 1 lives in `1/` and its files are `x1_en_*.csv`.
* `<lang>` is a 2-letter code from `{ar, cn, en, fr, vn}`.
* `<model>` is the slug listed in section 3 and is repeated verbatim inside the file in
  `agent1_llm` and `agent2_llm` (verified: 0 mismatches in 12,000 rows).

Glob patterns that work:

```python
root.glob("*/*/*.csv")                          # all 300 files
root.glob("1/*/x1_en_*.csv")                    # the 6 English files at lambda = 1
root.glob("*/gpt-5.4-nano-2026-03-17/*.csv")    # one model, all scales and languages
```

**Do not add non-numeric files or directories directly under the corpus root.**
`Analysis/pdlib/ingest.py` sorts the scale directories with
`key=lambda p: float(p.name)`, so a stray `README.md` or `notes/` at that level raises
`ValueError` and aborts the whole ingest. That is why this data card lives one level up,
in `Dataset/`.

---

## 3. The six models

Directory slug is the raw provider slug; the display name is the one
`Analysis/pdlib/ingest.py::MODEL_MAP` assigns.

| directory slug | display name | vendor | note |
|---|---|---|---|
| `claude-haiku-4-5-20251001` | Claude-Haiku-4.5 | Anthropic | |
| `gemini-3.1-flash-lite-preview` | Gemini-3.1-Flash-Lite-Preview | Google | preview build |
| `gemini-3.5-flash-lite` | Gemini-3.5-Flash-Lite | Google | first model collected |
| `gpt-5.4-nano-2026-03-17` | GPT-5.4-Nano | OpenAI | |
| `grok-4.20-0309-non-reasoning` | Grok-4.20-Non-Reasoning | xAI | the `-reasoning` sibling is a **different model** and is not in this corpus; never mix the two inside one lambda curve |
| `qwen3-235b-a22b-instruct-2507` | Qwen3-235B-A22B | Alibaba | mixture-of-experts |

Every game is **self-play**: both agents in a row are the same model. There are no
cross-model dyads in this corpus (verified: `agent1_llm == agent2_llm` in all 12,000 rows).

`MODEL_MAP` is indexed directly by `_iter_files`, so **a model directory that is not
registered there raises `KeyError` and kills the ingest**; it is not skipped. Register any
new sweep in `MODEL_MAP` before running analysis.

---

## 4. CSV schema (20 columns, in file order)

The header is identical in all 300 files.

| # | column | dtype as read by `pd.read_csv` | domain in this corpus | meaning |
|---|---|---|---|---|
| 1 | `game_id` | `str` | `game_0` ... `game_39`, unique within a file | index of the game inside the file; encodes the design cell (see section 5) |
| 2 | `language` | `str` | `ar`, `cn`, `en`, `fr`, `vn` | prompt language; equals the `<lang>` token in the file name |
| 3 | `n_rounds_is_known` | `bool` | always `True` | the prompt disclosed the horizon ("There are 10 rounds to decide") |
| 4 | `max_rounds` | `int` | always `10` | horizon written into the prompt |
| 5 | `played_rounds` | `int` | always `10` | rounds actually played; equal to `max_rounds` everywhere, so no game aborted early |
| 6 | `agents_communicate` | `bool` | always `False` | no cheap-talk stage |
| 7 | `agent1_name` | `str` | always `agent1` | placeholder identity substituted into the prompt |
| 8 | `agent1_llm` | `str` | one of the 6 slugs | model behind agent 1; equals the parent directory name |
| 9 | `agent1_personality` | `str` | localised, see section 6 | persona injected into agent 1's prompt |
| 10 | `agent1_knows_opponent_with_prob` | `int` | always `0` | agent 1 was told nothing about the opponent's persona (0%) |
| 11 | `agent1_strategies` | `str` holding a Python list | `"['OptionB', 'OptionA', ...]"`, length exactly 10 | agent 1's action in each round, in round order |
| 12 | `agent1_scores` | `str` holding a Python list | `"[2, 2, 10, ...]"`, length exactly 10 | **penalty** agent 1 received in each round |
| 13 | `agent1_messages` | `str` | always `"[]"` | empty because `agents_communicate` is `False` |
| 14 | `agent2_name` | `str` | always `agent2` | |
| 15 | `agent2_llm` | `str` | same value as `agent1_llm` | |
| 16 | `agent2_personality` | `str` | localised, see section 6 | |
| 17 | `agent2_knows_opponent_with_prob` | `int` | always `0` | |
| 18 | `agent2_strategies` | `str` holding a Python list | length exactly 10 | |
| 19 | `agent2_scores` | `str` holding a Python list | length exactly 10 | |
| 20 | `agent2_messages` | `str` | always `"[]"` | |

### Parsing the list columns

`agent{1,2}_strategies`, `agent{1,2}_scores` and `agent{1,2}_messages` are **Python
literals with single quotes**, not JSON. Use `ast.literal_eval`, not `json.loads`:

```python
import ast
actions = ast.literal_eval(row["agent1_strategies"])   # ['OptionB', 'OptionA', ...]
scores  = ast.literal_eval(row["agent1_scores"])       # [2, 2, 10, ...]
```

`agent1_scores[t]` is the penalty agent 1 paid in round `t`, aligned index for index with
`agent1_strategies[t]` and with `agent2_strategies[t]` (the opponent's move in the same round).

### Nine columns are constant

`n_rounds_is_known`, `max_rounds`, `played_rounds`, `agents_communicate`, `agent1_name`,
`agent2_name`, `agent1_knows_opponent_with_prob`, `agent2_knows_opponent_with_prob`,
`agent1_messages`, `agent2_messages` never vary. They are FAIRGAME schema slots that this
sweep left fixed. Do not use them as analysis factors; they exist so the files stay
schema-compatible with the other corpora in `Dataset/`, some of which do vary them.

---

## 5. Row layout inside a file

Every file holds the same 40 rows in the same order: 4 personality pairings, 10
repetitions each, contiguous.

| rows | `game_id` | agent1 persona | agent2 persona | dyad code |
|---|---|---|---|---|
| 0-9 | `game_0` ... `game_9` | cooperative | cooperative | `CvC` |
| 10-19 | `game_10` ... `game_19` | cooperative | selfish | `CvS` |
| 20-29 | `game_20` ... `game_29` | selfish | cooperative | `SvC` |
| 30-39 | `game_30` ... `game_39` | selfish | selfish | `SvS` |

So, with `k = int(game_id.split("_")[1])`:

```python
pairing_index = k // 10     # 0 = CvC, 1 = CvS, 2 = SvC, 3 = SvS
repetition    = k % 10      # 0 .. 9
```

Verified: this holds for all 12,000 rows, in every language, at every scale.

`CvS` and `SvC` are the **same dyad composition with the roles swapped**. Pooling them is
legitimate for dyad-level statistics; keeping them apart is what lets you test whether
being agent 1 versus agent 2 matters. Both agents see structurally identical prompts, so
any asymmetry is sampling noise, not design.

> `game_id` is unique inside a file here (40 rows, 40 distinct ids). The rep-counter in
> `ingest.py::build_master` exists because *other* corpora in `Dataset/` repeat ids; it is
> a no-op on this one. Across files, `game_id` is **not** unique, so the join key for a
> game is the tuple `(scale, model, language, game_id)`, or the `game_uid` that
> `ingest.py` builds as `f"{csv.stem}#{row_index:03d}"`.

---

## 6. Personality strings, per language

The persona is written into the prompt in the prompt's own language, so the raw string
differs by file. Normalise before grouping. Each of the 10 strings below appears exactly
2,400 times in the corpus.

| language | cooperative | selfish |
|---|---|---|
| `en` | `cooperative` | `selfish` |
| `fr` | `coopératif` | `égoïste` |
| `vn` | `một người hợp tác` | `một người ích kỷ` |
| `cn` | `合作型的` | `自私型的` |
| `ar` | `متعاون` | `أناني` |

`Analysis/pdlib/ingest.py::PERSONALITY_MAP` is the canonical normaliser and also accepts
the uppercase English variants used by older corpora. Read the CSVs with
`encoding="utf-8"` explicitly: the Windows default `cp1252` mangles four of the five
languages.

---

## 7. Game semantics: payoff matrix and action coding

### The prompt is stated as penalties, and the agent is told to minimise

The PD template (`FAIRGAME/resources/game_templates/prisoner_dilemma_*`, inlined into
`kaggle/benchmarks/pd_task.py`) frames the game as arrest and imprisonment: "you both get
a penalty of ...", "Your goal is to minimize your penalty". Base weights at scale 1 are
`weight1 = 6`, `weight2 = 10`, `weight3 = 0`, `weight4 = 2`, laid out as:

| | opponent `OptionA` | opponent `OptionB` |
|---|---|---|
| focal `OptionA` | 6, 6 | 0, 10 |
| focal `OptionB` | 10, 0 | 2, 2 |

### Therefore: `OptionA` = DEFECT, `OptionB` = COOPERATE

`OptionA` yields the lower penalty in **both** columns (0 < 2 and 6 < 10), so it is the
strictly dominant action, which is the definition of defection. Mutual `OptionB` (2, 2) is
better for both than mutual `OptionA` (6, 6), which makes `OptionB` the cooperative action.

```python
ACTION_MAP = {"OptionA": "D", "OptionB": "C"}   # from Analysis/pdlib/ingest.py
```

**This is the single most consequential fact in this data card.** Getting it backwards
inverts every cooperation rate in the corpus (`x -> 1 - x`) and flips the sign of the
payoff-scale and endgame effects. The mapping above was validated by reproducing Fig. 2a
of arXiv:2601.19082 from these tables: mean absolute error 0.002 under this mapping,
0.171 under the inverted one.

> Historical trap, still worth knowing: `pd_task.py` itself hard-coded `OptionA` as
> cooperation until 2026-09-03, so **cooperation rates printed in old Kaggle run logs and
> hand-copied into old notes are defection rates**, equal to `1 - true value`. The CSVs
> themselves were never affected; only the log summaries were. Anything routed through
> `ingest.py` has always been correct.

### T / R / P / S in penalty space

```
focal C, opp C  ->  R = 2      (mutual cooperation)
focal C, opp D  ->  S = 10     (sucker)
focal D, opp C  ->  T = 0      (temptation)
focal D, opp D  ->  P = 6      (mutual defection)
```

Read as penalties this is `T < R < P < S`, the penalty-space image of the usual
`T > R > P > S`. To get a conventional higher-is-better utility, negate: `u = -penalty`.
`ingest.py::payoff_matrix("frontier")` returns these plus the greed, fear and Rapoport-k
indices computed on the negated scale so they compare with the standard PD literature.

### The payoff scale lambda

The directory name multiplies all four cell values. At scale lambda the observable score
values are exactly `{0, 2*lambda, 6*lambda, 10*lambda}`:

| lambda | R | S | T | P | distinct values seen in the files |
|---|---|---|---|---|---|
| 0.01 | 0.02 | 0.1 | 0 | 0.06 | `0, 0.02, 0.06, 0.1` |
| 1 | 2 | 10 | 0 | 6 | `0, 2, 6, 10` |
| 1000 | 2000 | 10000 | 0 | 6000 | `0, 2000, 6000, 10000` |

The full ladder is `0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 100, 1000`, five orders of
magnitude. Since a positive affine rescaling of payoffs leaves the game strategically
identical (von Neumann-Morgenstern invariance), any dependence of behaviour on lambda is a
deviation from expected-utility rationality, which is what this corpus was built to measure.

**Verified corpus-wide:** every file's realised scale equals its directory name exactly,
no file mixes scales, and the `T` cell is `0` in all 240,000 agent-rounds regardless of
lambda. So on this corpus `scale_nominal == scale_eff` and you may trust the folder name.
`ingest.py` still re-derives the scale from the round outcomes because other corpora in
`Dataset/` do contain mixed-scale files.

---

## 8. Common random numbers across lambda

`pd_task.py` seeds each game from a cell index
`(lang_index * 4 + pairing_index) * 10 + repetition`, which **deliberately does not depend
on lambda**. The same `(model, language, game_id)` therefore starts from the same sampling
seed at every scale.

Practical consequence: games are **matched across lambda**. Comparisons along the scale
axis can and should be paired on `(model, language, game_id)` rather than treated as
independent samples. This is the design's main source of statistical power, since LLM
sampling variance across cells is large relative to the lambda effect.

---

## 9. Verified invariants and a validation snippet

All of these were checked over the full 300 files on 2026-09-06:

1. 300 CSV files, 40 data rows each, 12,000 rows total.
2. Every `agent{1,2}_strategies` and `agent{1,2}_scores` list has length exactly 10.
3. Action tokens are only `OptionA` (98,510) and `OptionB` (141,490);
   there are no `None` values or truncated strings such as `'Option'`. The released
   per-game CSVs do not store retry counts, and the protocol's terminal fallback is
   itself encoded as `OptionA`, so these files alone cannot establish a zero fallback
   rate.
4. `agent1_llm == agent2_llm == <parent directory name>` in every row.
5. `n_rounds_is_known=True`, `max_rounds=10`, `played_rounds=10`,
   `agents_communicate=False`, both `knows_opponent_with_prob=0`, both `messages=[]`
   in every row.
6. Each file contains 4 contiguous blocks of 10 games in the fixed order CvC, CvS, SvC, SvS.
7. Scores reproduce the payoff matrix exactly: for every round,
   `score == lambda * matrix[own_action][opp_action]`, with no mixed-scale file.
8. Each of the 10 localised personality strings occurs exactly 2,400 times.

Re-run the check before trusting a modified copy:

```python
import ast, csv, pathlib

ROOT   = pathlib.Path("Dataset/data_fairgame_frontier_llm")
ACTION = {"OptionA": "D", "OptionB": "C"}
BASE   = {"CC": 2.0, "CD": 10.0, "DC": 0.0, "DD": 6.0}   # penalties at lambda = 1

files = sorted(ROOT.glob("*/*/*.csv"))
assert len(files) == 300, len(files)

for f in files:
    lam = float(f.parent.parent.name)
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    assert len(rows) == 40, (f, len(rows))
    assert f.parent.name == rows[0]["agent1_llm"]
    for i, r in enumerate(rows):
        a1 = [ACTION[x] for x in ast.literal_eval(r["agent1_strategies"])]
        a2 = [ACTION[x] for x in ast.literal_eval(r["agent2_strategies"])]
        s1 = ast.literal_eval(r["agent1_scores"])
        s2 = ast.literal_eval(r["agent2_scores"])
        assert len(a1) == len(a2) == len(s1) == len(s2) == 10
        assert int(r["game_id"].split("_")[1]) == i
        for own, opp, sc in ((a1, a2, s1), (a2, a1, s2)):
            for t in range(10):
                assert abs(sc[t] - lam * BASE[own[t] + opp[t]]) < 1e-9
print("ok")
```

---

## 10. Canonical loader: `Analysis/pdlib/ingest.py`

Prefer this over ad-hoc parsing. It already handles the action coding, the personality
normalisation, the scale recovery and the lagged features, and every figure and table in
the manuscripts is built on it.

```python
import sys; sys.path.insert(0, "Analysis")
from pdlib.ingest import build_master

rounds, games = build_master(families=("frontier",))   # this corpus only
```

`families=("frontier",)` restricts the walk to `data_fairgame_frontier_llm`; the mapping
lives in `ingest.py::FAMILY_DIR` (`frontier` -> this corpus, `small` ->
`data_fairgame_small_llm`). Omitting the argument loads both.

### `rounds`: one row per (game, round, focal agent)

240,000 rows for this corpus. Both agents of a dyad appear as focal, so agent-level
statistics are symmetric by construction and you must **cluster on `game_uid`**, never
treat rounds as independent.

| column | meaning |
|---|---|
| `family` | `"frontier"` |
| `model` | display name from `MODEL_MAP`, e.g. `Gemini-3.5-Flash-Lite` |
| `language` | `ar` / `cn` / `en` / `fr` / `vn` |
| `scale_nominal` | float from the directory name |
| `scale_eff` | scale re-derived from the round payoffs; equals `scale_nominal` throughout this corpus |
| `game_uid` | `f"{csv_stem}#{row_index:03d}"`, unique across the whole corpus |
| `game_id`, `rep` | id inside the file, and the replicate counter |
| `agent` | `1` or `2`, which agent is focal in this row |
| `personality`, `opp_personality` | normalised to `cooperative` / `selfish` |
| `dyad` | `CvC`, `CvS`, `SvC`, `SvS`, oriented from the focal agent |
| `rounds_known`, `max_rounds` | `True`, `10` |
| `round`, `round_frac` | 1-based round index and `round / n_rounds`, for endgame analysis |
| `action`, `opp_action` | `C` or `D` |
| `coop`, `opp_coop` | 0/1 versions of the above |
| `outcome` | `CC`, `CD`, `DC`, `DD` from the focal agent's viewpoint |
| `payoff_raw` | penalty as recorded, still multiplied by lambda |
| `payoff_base` | `payoff_raw / scale_eff`, so scales are comparable |
| `payoff_norm` | penalty rescaled to [0, 1] on the S..T span, **higher is better** |
| `efficiency` | `(S - payoff_base) / (S - R)`; 1.0 = mutual cooperation, 0.5 = mutual defection, > 1 = successful exploitation |
| `prev_action`, `prev_opp_action`, `prev_outcome` | one-round lags within `(game_uid, agent)`; `NaN` in round 1 |
| `prev_letter` | `R`/`S`/`T`/`P` for the previous outcome, `E` at the start of a game |
| `token` | `prev_letter + action`, the 10-symbol alphabet the strategy classifier consumes |

### `games`: one row per (game, focal agent)

24,000 rows. Design keys plus `n_rounds`, `coop_rate`, `opp_coop_rate`,
`first_move_coop`, `last_move_coop`, `payoff_base`, `payoff_per_round`, `efficiency`,
`cc_rate`, `dd_rate`, `cd_rate`, `dc_rate`, `longest_D_run`, `dd_absorbed`
(1 when the dyad reaches mutual defection and never leaves it).

### Interval helpers in the same module

* `wilson(p, n)` for proportion error bars.
* `cluster_bootstrap_ci(df, value, cluster="game_uid")` resamples whole games. Rounds
  inside a game are strongly autocorrelated, so a round-level interval is far too narrow;
  every interval in the manuscripts uses the clustered version.

---

## 11. Standalone loader, no repo dependency

If you need a tidy frame without importing `pdlib`:

```python
import ast, pathlib
import pandas as pd

ROOT   = pathlib.Path("Dataset/data_fairgame_frontier_llm")
ACTION = {"OptionA": "D", "OptionB": "C"}
PERS   = {
    "cooperative": "cooperative", "selfish": "selfish",
    "coopératif": "cooperative", "égoïste": "selfish",
    "một người hợp tác": "cooperative", "một người ích kỷ": "selfish",
    "合作型的": "cooperative", "自私型的": "selfish",
    "متعاون": "cooperative", "أناني": "selfish",
}

recs = []
for f in sorted(ROOT.glob("*/*/*.csv")):
    scale, model = float(f.parent.parent.name), f.parent.name
    lang = f.stem.split("_")[1]
    df = pd.read_csv(f, encoding="utf-8")
    for i, r in df.iterrows():
        acts = [[ACTION[x] for x in ast.literal_eval(r[f"agent{k}_strategies"])] for k in (1, 2)]
        scs  = [ast.literal_eval(r[f"agent{k}_scores"]) for k in (1, 2)]
        pers = [PERS[r[f"agent{k}_personality"]] for k in (1, 2)]
        k = int(r.game_id.split("_")[1])
        for a in (0, 1):
            o = 1 - a
            for t in range(len(acts[a])):
                recs.append({
                    "model": model, "language": lang, "scale": scale,
                    "game_uid": f"{f.stem}#{i:03d}", "game_id": r.game_id,
                    "pairing": k // 10, "rep": k % 10,
                    "agent": a + 1,
                    "personality": pers[a], "opp_personality": pers[o],
                    "dyad": f"{pers[a][0].upper()}v{pers[o][0].upper()}",
                    "round": t + 1,
                    "action": acts[a][t], "opp_action": acts[o][t],
                    "coop": int(acts[a][t] == "C"),
                    "outcome": acts[a][t] + acts[o][t],
                    "penalty_raw": float(scs[a][t]),
                    "penalty_base": float(scs[a][t]) / scale,
                })

rounds = pd.DataFrame(recs)          # 240,000 rows
assert len(rounds) == 240_000
```

---

## 12. Sanity-check statistics

If your pipeline reproduces these, the action coding and the parse are right. If your
cooperation rates come out as `1 - x` of these, you have `OptionA` and `OptionB` swapped.

Overall cooperation rate (`OptionB`), pooled over everything: **0.590**.

| model | coop rate | | payoff scale | coop rate | | language | coop rate |
|---|---|---|---|---|---|---|---|
| Claude-Haiku-4.5 | 0.415 | | 0.01 | 0.505 | | ar | 0.573 |
| Gemini-3.1-Flash-Lite-Preview | 0.617 | | 0.1 | 0.533 | | cn | 0.582 |
| Gemini-3.5-Flash-Lite | 0.751 | | 0.25 | 0.630 | | en | 0.625 |
| GPT-5.4-Nano | 0.606 | | 0.5 | 0.588 | | fr | 0.600 |
| Grok-4.20-Non-Reasoning | 0.713 | | 1 | 0.610 | | vn | 0.569 |
| Qwen3-235B-A22B | 0.435 | | 2 | 0.624 | | | |
| | | | 5 | 0.615 | | | |
| | | | 10 | 0.620 | | | |
| | | | 100 | 0.574 | | | |
| | | | 1000 | 0.598 | | | |

Each model cell rests on 40,000 agent-rounds, each scale cell on 24,000, each language
cell on 48,000.

---

## 13. Caveats for analysis

1. **Action coding.** Section 7. Everything else is secondary to this.
2. **Not independent observations.** 240,000 rounds are only 12,000 games and 24,000
   agent-games. Cluster on `game_uid` (or on the dyad) for every interval and test.
   Both agents of a dyad appear as focal, so the two rows of a game are the same
   interaction seen twice.
3. **Self-play only.** Every result is about a model playing itself. Nothing here
   supports claims about cross-model interaction.
4. **Known horizon.** `n_rounds_is_known` is `True` everywhere, so backward induction is
   available to the agent and endgame defection is expected. There is no
   unknown-horizon arm in this corpus, which means the horizon effect cannot be
   identified from this corpus alone.
5. **A translation quirk inherited from FAIRGAME.** In the `ar` and `cn` PD templates the
   final goal sentence says *maximise your rewards* while the payoff sentences above it
   say *penalty*; `en`, `fr` and `vn` correctly say *minimise your penalty*. This is
   verbatim from `FAIRGAME/resources/`, is enforced byte-identical by
   `kaggle/benchmarks/test_pd_task_parity.py`, and is deliberately not corrected so the
   corpus stays comparable with the published FAIRGAME baseline. Treat any `ar`/`cn`
   language effect with this in mind: part of it may be prompt inconsistency rather than
   language.
6. **`gemini-3.1-flash-lite-preview` is a preview build**, and the manuscripts keep it in
   the supplement rather than the main text for that reason.
7. **Grok variant.** Only `-non-reasoning` was collected. The `-reasoning` sibling is a
   different model; never splice them into one curve.
8. **Nine constant columns** (section 4) carry no information here. Do not put them on an
   axis or into a regression.

---

## 14. Related corpora in `Dataset/`

| directory | what it is | relation |
|---|---|---|
| `data_fairgame_frontier_llm` | **this corpus**: 6 frontier models, 10 scales, PD | primary |
| `data_fairgame_small_llm` | open-weight models, PD; base matrix has `P = 8`, not 6 | different payoff matrix, so `_BASE_MATRIX["small"]` applies, family `small` in `ingest.py` |
| `data_replicate_frontier` | test-retest replicate of a subset, different `BASE_SEED` and a `RUN_TAG` suffix | reliability estimate |
| `data_stag_hunt_frontier` | stag hunt, **reward** framing, so `OptionA` is the cooperative action there | the action coding is the mirror image; never reuse this card's mapping on it |
| `data_fairgame_e2_notation` | payoff numbers written in a different notation, same games | notation-invariance experiment |
| `noise_dataset`, `noise_dataset_30round` | synthetic strategy-labelled games | trains and validates the strategy classifier |
| `archive_data_fairgame_frontier_llm` | superseded earlier frontier run | do not mix into current analyses |

---

## 15. Provenance

* **Generator:** `kaggle/benchmarks/pd_task.py`, run server-side as the Kaggle Benchmarks
  task `prisoner-dilemma-fairgame`. Prompts are byte-identical copies of
  `FAIRGAME/resources/game_templates/prisoner_dilemma_{en,fr,ar,cn,vn}`, enforced by
  `kaggle/benchmarks/test_pd_task_parity.py`.
* **Config lineage:** FAIRGAME
  `prisoner_dilemma_nocomm_round_known_conventional.json`, weights
  `w1=6, w2=10, w3=0, w4=2`.
* **Ingestion into `Dataset/`:** `kaggle/benchmarks/collect_run.py`, which rejects any
  cell with fewer than 200 rows and strips the `model_tag` suffix from both the file name
  and the `agent1_llm` column. This matters because Kaggle mounts the previous run's
  output into the next run so `RESUME` can work, which means a downloaded bundle also
  carries fragments of failed runs.
* **Collection order:** Gemini-3.5-Flash-Lite first, then
  Gemini-3.1-Flash-Lite-Preview (2026-09-02), GPT-5.4-Nano (2026-09-03),
  Claude-Haiku-4.5 and Qwen3-235B-A22B (2026-09-05), Grok-4.20-Non-Reasoning (2026-09-05).
* **Sampling:** the task explicitly sets `temperature=1.0` and a deterministic
  common-random-number seed for each cell/round/agent, and caps output with
  `max_tokens=128`; optional provider parameters are dropped when an endpoint rejects
  them. The released CSVs contain only valid action tokens, but they do not retain
  retry/fallback counters, so the terminal fallback rate cannot be reconstructed from
  this corpus alone.
* **This card:** written 2026-09-06 from a full scan of all 300 files.
