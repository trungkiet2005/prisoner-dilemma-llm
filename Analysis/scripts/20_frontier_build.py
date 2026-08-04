"""Build the frontier-only master tables that the FR figure suite reads.

Writes `Analysis/data/frontier_{rounds,games}.parquet` and the frontier corpus
tables.  Deliberately separate from `00_build_master.py`: the legacy figures
(F01-F34) were computed on the three-model frontier arm plus the open-weight
arm, and rebuilding those from a corpus that now contains a fourth model would
silently change published numbers.  This script touches nothing they read.

Design note carried into every figure
-------------------------------------
The Gemini arm was collected with a *known* horizon (`n_rounds_is_known=True`,
`PD_ROUNDS_KNOWN=1` in `kaggle/benchmarks/pd_task.py`), while the Claude, GPT
and Mistral arms were collected with an *unknown* horizon.  A known finite
horizon is exactly the condition under which backward induction predicts
unravelling to mutual defection, so the Gemini-vs-rest contrast confounds
model identity with horizon knowledge.  Every figure that compares models
marks this, and `T_FR01` records it per arm.  Treat the Gemini column as
"model x horizon", not "model".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pdlib.ingest import build_master, payoff_matrix
from pdlib.natstyle import DATADIR, FRONTIER, HORIZON, LANG_ORDER, TABDIR


def main():
    rounds, games = build_master(families=("frontier",))

    missing = sorted(set(FRONTIER) - set(games.model.unique()))
    extra = sorted(set(games.model.unique()) - set(FRONTIER))
    if missing:
        raise SystemExit(f"frontier arm is missing model(s): {missing}")
    if extra:
        print(f"!! frontier folder holds unexpected model(s), ignored: {extra}")
        games = games[games.model.isin(FRONTIER)].copy()
        rounds = rounds[rounds.model.isin(FRONTIER)].copy()

    # horizon condition recovered from the logs, not assumed
    horizon = (games.groupby("model")["rounds_known"].agg(["mean", "size"]))
    for m in FRONTIER:
        obs = "known" if horizon.loc[m, "mean"] > 0.5 else "unknown"
        if not np.isclose(horizon.loc[m, "mean"] % 1, 0):
            print(f"!! {m}: horizon flag is mixed within the arm "
                  f"({horizon.loc[m, 'mean']:.2f} known)")
        if obs != HORIZON[m]:
            raise SystemExit(
                f"{m}: logs say horizon={obs} but natstyle.HORIZON says "
                f"{HORIZON[m]} -- fix the constant before plotting")

    rounds.to_parquet(DATADIR / "frontier_rounds.parquet", index=False)
    games.to_parquet(DATADIR / "frontier_games.parquet", index=False)
    print(f"rounds: {len(rounds):,}   games (agent-rows): {len(games):,}   "
          f"dyads: {games.game_uid.nunique():,}")

    # ---- T_FR01  corpus ----------------------------------------------------
    corpus = (games.groupby("model")
              .agg(dyads=("game_uid", "nunique"),
                   agent_games=("game_uid", "size"),
                   scales=("scale_nominal", "nunique"),
                   languages=("language", "nunique"),
                   rounds_per_game=("n_rounds", "max"),
                   coop=("coop_rate", "mean"))
              .reindex(FRONTIER))
    corpus["decisions"] = corpus.agent_games * corpus.rounds_per_game
    corpus["horizon"] = [HORIZON[m] for m in corpus.index]
    corpus.to_csv(TABDIR / "T_FR01_frontier_corpus.csv")
    print()
    print(corpus.to_string())

    # ---- T_FR02  payoff geometry -------------------------------------------
    geo = pd.DataFrame([{"family": "frontier", **payoff_matrix("frontier")}])
    geo.to_csv(TABDIR / "T_FR02_payoff_geometry.csv", index=False)

    # ---- T_FR03  cell counts (model x scale x language) --------------------
    cells = (games.pivot_table(index=["model", "scale_nominal"],
                               columns="language", values="game_uid",
                               aggfunc="nunique")
             .reindex(columns=LANG_ORDER))
    cells.to_csv(TABDIR / "T_FR03_cell_counts.csv")
    if cells.nunique().nunique() != 1 or cells.stack().nunique() != 1:
        print("\n!! design is unbalanced -- cell sizes differ:")
        print(cells.to_string())
    else:
        print(f"\nbalanced design: {int(cells.iloc[0, 0])} dyads in every "
              f"model x scale x language cell")

    # ---- integrity: realised vs nominal payoff scale ------------------------
    mism = ~np.isclose(rounds.scale_eff, rounds.scale_nominal, rtol=1e-6)
    if mism.any():
        odd = (rounds.loc[mism, ["model", "language", "scale_nominal", "scale_eff"]]
               .drop_duplicates())
        odd.to_csv(TABDIR / "T_FR04_scale_anomalies.csv", index=False)
        print(f"\n!! {mism.sum():,} rounds whose realised payoffs disagree with "
              f"the folder's nominal scale (written to T_FR04)")
    else:
        print("payoff integrity: realised scale matches the nominal scale everywhere")


if __name__ == "__main__":
    main()
