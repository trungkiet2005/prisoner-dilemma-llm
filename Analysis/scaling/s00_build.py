"""Build the tidy master tables for the payoff-scaling study.

Scope is fixed by decision (2026-09-05): the frontier prisoner's-dilemma corpus
`Dataset/data_fairgame_frontier_llm` and nothing else.  The design is a complete
crossing of

    10 payoff scales  x  5 models  x  5 prompt languages  x  4 persona pairings
    x 10 replicates  x  10 rounds  x  2 agents

= 10,000 dyads, 20,000 agent-games, 200,000 decisions, with every cell at
exactly 200 dyads.  `assert_complete()` refuses to write anything if that is not
what is on disk, because a silently truncated cell is the failure mode this
corpus has actually suffered.

Outputs (parquet, under Analysis/scaling/data/):
    rounds.parquet  one row per (dyad, round, focal agent)
    games.parquet   one row per (dyad, focal agent)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdlib.ingest import build_master           # noqa: E402
from pdlib.style import DATASET                 # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

SCALES = [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0]
LANGS = ["ar", "cn", "en", "fr", "vn"]
DYADS_PER_CELL = 40          # 4 persona pairings x 10 replicates
N_ROUNDS = 10


def assert_complete(games: pd.DataFrame) -> None:
    """Refuse to proceed on a partial grid.

    Every check below has a failure it is guarding against, not a hypothetical:
    a short cell means a run stopped early on an exhausted account, a missing
    scale means a sweep never ran, and a short game means a truncated reply was
    written as if it were play.
    """
    n_models = games["model"].nunique()
    cells = (games.groupby(["scale_nominal", "model", "language"])["game_uid"]
             .nunique().rename("n_dyads").reset_index())

    problems = []
    missing = set(SCALES) - set(games["scale_nominal"].unique())
    if missing:
        problems.append(f"missing payoff scales: {sorted(missing)}")
    bad_lang = set(games["language"].unique()) - set(LANGS)
    if bad_lang:
        problems.append(f"unexpected languages: {sorted(bad_lang)}")
    short = cells[cells.n_dyads != DYADS_PER_CELL]
    if len(short):
        problems.append(f"{len(short)} cells not at {DYADS_PER_CELL} dyads:\n"
                        f"{short.to_string(index=False)}")
    expected_cells = len(SCALES) * n_models * len(LANGS)
    if len(cells) != expected_cells:
        problems.append(f"{len(cells)} cells present, expected {expected_cells}")
    ragged = games[games.n_rounds != N_ROUNDS]
    if len(ragged):
        problems.append(f"{len(ragged)} agent-games not {N_ROUNDS} rounds long")

    if problems:
        raise SystemExit("GRID INCOMPLETE - refusing to write:\n  "
                         + "\n  ".join(problems))


def main() -> None:
    print(f"reading {DATASET / 'data_fairgame_frontier_llm'}")
    rounds, games = build_master(families=("frontier",))
    assert_complete(games)

    # `scale_eff` is recovered from the realised payoffs rather than trusted
    # from the folder name; they must agree, or the folder is mislabelled.
    drift = (games.merge(rounds[["game_uid", "scale_eff"]].drop_duplicates(),
                         on="game_uid")
             .assign(rel=lambda d: (d.scale_eff - d.scale_nominal).abs()
                     / d.scale_nominal))
    if (drift.rel > 1e-6).any():
        raise SystemExit("recovered payoff scale disagrees with folder name for "
                         f"{(drift.rel > 1e-6).sum()} agent-games")

    rounds.to_parquet(OUT / "rounds.parquet", index=False)
    games.to_parquet(OUT / "games.parquet", index=False)

    print(f"  models      {games.model.nunique()}: {sorted(games.model.unique())}")
    print(f"  scales      {games.scale_nominal.nunique()}: {sorted(games.scale_nominal.unique())}")
    print(f"  languages   {games.language.nunique()}: {sorted(games.language.unique())}")
    print(f"  dyads       {games.game_uid.nunique():,}")
    print(f"  agent-games {len(games):,}")
    print(f"  decisions   {len(rounds):,}")
    print(f"  personas    {sorted(games.dyad.unique())}")
    print(f"  overall cooperation {games.coop_rate.mean():.4f}")
    print(f"wrote {OUT/'rounds.parquet'} and {OUT/'games.parquet'}")


if __name__ == "__main__":
    main()
