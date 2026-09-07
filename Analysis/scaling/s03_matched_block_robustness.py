"""Sensitivity analysis that preserves recorded game-ID scale ladders.

The released tables retain matching by game identifier but not API seed values.
This script therefore tests robustness to the documented matching structure,
not provider-specific common-random-number semantics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "games.parquet"
OUT = HERE / "tables" / "T18_matched_block_sensitivity.csv"
N_PERM = 10000
SEED = 20260907


def spread(d: pd.DataFrame) -> float:
    means = d.groupby("scale_nominal").coop_rate.mean()
    return float(means.max() - means.min())


def permutation_p(d: pd.DataFrame, blocks: list[pd.DataFrame],
                  rng: np.random.Generator) -> float:
    observed = spread(d)
    scales = np.sort(d.scale_nominal.unique())
    values_by_block = [x.sort_values("scale_nominal").coop_rate.to_numpy()
                       for x in blocks]
    values = np.asarray(values_by_block, dtype=float)
    # Draw all block permutations at once. The indexed gather is equivalent
    # to the loop above but keeps the 10,000-draw sensitivity run in numpy.
    keys = rng.random((N_PERM, len(values), len(scales)))
    perms = np.argsort(keys, axis=2)
    assigned = np.take_along_axis(values[None, :, :], perms, axis=2)
    means = assigned.sum(axis=1) / len(values)
    hits = int(np.count_nonzero((means.max(axis=1) - means.min(axis=1)) >= observed))
    return (hits + 1) / (N_PERM + 1)


def main() -> None:
    g = pd.read_parquet(DATA)
    rows = []
    rng = np.random.default_rng(SEED)
    for model, d in g.groupby("model", sort=True):
        # There are two agent-games per dyad. Aggregate first, then preserve a
        # ten-scale ladder for every model, language and recorded game ID.
        dyads = (d.groupby(["language", "game_id", "scale_nominal", "game_uid"],
                           as_index=False).coop_rate.mean())
        ladders = [x for _, x in dyads.groupby(["language", "game_id"], sort=False)]
        if not all(x.scale_nominal.nunique() == 10 for x in ladders):
            raise ValueError(f"incomplete recorded ladder for {model}")
        paired = pd.concat(ladders, ignore_index=True)
        rows.append({"model": model, "n_blocks": len(ladders),
                     "range": spread(paired),
                     "p_block_permutation": permutation_p(paired, ladders, rng),
                     "n_permutations": N_PERM})
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
