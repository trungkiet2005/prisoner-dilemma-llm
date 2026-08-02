"""Parse every FAIRGAME log into Analysis/data/{rounds,games}.parquet.

Run this first -- every figure script reads the two tables it writes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pdlib.ingest import build_master, payoff_matrix
from pdlib.style import DATADIR, TABDIR


def main():
    rounds, games = build_master()

    rounds.to_parquet(DATADIR / "rounds.parquet", index=False)
    games.to_parquet(DATADIR / "games.parquet", index=False)

    print(f"rounds: {len(rounds):,} rows   games: {len(games):,} rows")
    print()

    # ---- corpus description table -----------------------------------------
    desc = (games.groupby(["family", "model", "scale_nominal"])
            .agg(n_games=("game_uid", "nunique"),
                 n_agents=("game_uid", "size"),
                 rounds=("n_rounds", "max"),
                 languages=("language", "nunique"),
                 coop=("coop_rate", "mean"))
            .reset_index())
    desc.to_csv(TABDIR / "T01_corpus_description.csv", index=False)
    print(desc.to_string(index=False))
    print()

    # ---- payoff geometry ---------------------------------------------------
    geo = pd.DataFrame([{"family": f, **payoff_matrix(f)} for f in ("frontier", "small")])
    geo.to_csv(TABDIR / "T02_payoff_geometry.csv", index=False)
    print(geo.to_string(index=False))
    print()

    # ---- data-integrity flags ---------------------------------------------
    import numpy as np
    mism = ~np.isclose(rounds.scale_eff, rounds.scale_nominal, rtol=1e-6)
    odd = (rounds.loc[mism, ["family", "model", "language", "scale_nominal", "scale_eff"]]
           .drop_duplicates())
    if len(odd):
        odd.to_csv(TABDIR / "T03_scale_anomalies.csv", index=False)
        print("!! games whose realised payoffs disagree with the folder's nominal scale:")
        print(odd.to_string(index=False))
    else:
        print("no scale anomalies")


if __name__ == "__main__":
    main()
