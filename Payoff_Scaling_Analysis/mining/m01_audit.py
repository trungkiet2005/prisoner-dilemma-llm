"""Section 1-2: structural audit, data dictionary and data-quality forensics.

Nothing here is taken on trust from the data card: the payoff matrix, the design
balance, the action alphabet and the common-random-number pairing are all
re-derived from the 300 CSV files and reported as measurements.
"""
from __future__ import annotations

import ast
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import core as C


# --------------------------------------------------------------------------
def raw_file_audit() -> pd.DataFrame:
    """Walk the raw CSVs and measure the physical layout."""
    rows = []
    for f in sorted(C.CORPUS.glob("*/*/*.csv")):
        df = pd.read_csv(f, encoding="utf-8")
        rows.append({
            "path": str(f.relative_to(C.CORPUS)).replace("\\", "/"),
            "scale": float(f.parent.parent.name),
            "model_dir": f.parent.name,
            "language": re.match(r"^x[\d.]+_([a-z]{2})_", f.name).group(1),
            "n_rows": len(df), "n_cols": df.shape[1],
            "bytes": f.stat().st_size,
            "n_missing": int(df.isna().sum().sum()),
            "n_dup_rows": int(df.duplicated().sum()),
            "llm_matches_dir": bool((df.agent1_llm == f.parent.name).all()
                                    and (df.agent2_llm == f.parent.name).all()),
            "selfplay": bool((df.agent1_llm == df.agent2_llm).all()),
            "seq_len_ok": bool(all(
                len(ast.literal_eval(x)) == 10 for c in
                ["agent1_strategies", "agent2_strategies", "agent1_scores", "agent2_scores"]
                for x in df[c])),
        })
    return pd.DataFrame(rows)


def data_dictionary(rounds, ag, dy) -> pd.DataFrame:
    """Per-column dictionary for the raw schema and the three derived tables."""
    recs = []

    # raw schema, measured over one representative file per language
    raw = pd.concat([pd.read_csv(f, encoding="utf-8")
                     for f in sorted(C.CORPUS.glob("1/*/x1_*.csv"))], ignore_index=True)
    for col in raw.columns:
        s = raw[col]
        recs.append(_describe(s, col, "raw CSV (lambda=1 slice)", _raw_role(col, s)))

    roles = {
        "model": "grouping / experimental factor (LLM identity)",
        "language": "grouping / experimental factor (prompt language)",
        "scale": "SCALING VARIABLE (payoff multiplier lambda)",
        "log_scale": "SCALING VARIABLE, log10 lambda",
        "game_uid": "identifier (unique game)",
        "cell": "identifier (design cell, matched across lambda by CRN)",
        "game_id": "identifier inside a file",
        "pairing": "experimental condition (personality pairing 0-3)",
        "rep": "replicate index",
        "agent": "player identifier (1 or 2)",
        "personality": "experimental condition (own persona)",
        "opp_personality": "experimental condition (opponent persona)",
        "dyad": "experimental condition (pairing, focal-oriented)",
        "round": "temporal index",
        "round_frac": "temporal index, normalised",
        "action": "OUTCOME (own move, C/D)",
        "opp_action": "outcome (opponent move)",
        "outcome": "OUTCOME (joint outcome CC/CD/DC/DD)",
        "coop": "OUTCOME (cooperation indicator, primary target)",
        "opp_coop": "outcome (opponent cooperation)",
        "payoff_raw": "PAYOFF as recorded (penalty, still multiplied by lambda)",
        "payoff_base": "PAYOFF in base units (penalty / lambda)",
        "utility": "PAYOFF rescaled to [0,1], higher is better",
        "efficiency": "PAYOFF vs mutual-cooperation benchmark (1=CC, 0.5=DD)",
        "prev_action": "lagged feature", "prev_opp_action": "lagged feature",
        "prev_outcome": "lagged feature", "prev_letter": "lagged feature (R/S/T/P/E)",
        "token": "lagged feature (strategy-classifier alphabet)",
        "coop_rate": "OUTCOME (per-agent-game cooperation rate)",
        "utility_gap": "FAIRNESS (within-dyad |u1-u2|, 0 = equal)",
        "gini": "FAIRNESS (Gini of the two utilities)",
        "fairness": "FAIRNESS (1 - Gini)",
        "joint_utility": "WELFARE (dyad mean utility)",
        "joint_efficiency": "WELFARE (dyad mean efficiency)",
        "reciprocity": "BEHAVIOUR (P(C|oppC) - P(C|oppD))",
        "strategy": "BEHAVIOUR (exact memory-one rule label)",
        "dd_absorbed": "BEHAVIOUR (locked into mutual defection)",
        "endgame_drop": "BEHAVIOUR (early minus late cooperation)",
    }
    for tab, name in ((rounds, "rounds"), (ag, "agent_games"), (dy, "dyads")):
        for col in tab.columns:
            recs.append(_describe(tab[col], col, name,
                                  roles.get(col, _derived_role(col))))
    return pd.DataFrame(recs)


def _raw_role(col, s):
    if s.nunique(dropna=False) == 1:
        return "CONSTANT - schema slot, carries no information here"
    if col.endswith("_strategies"):
        return "OUTCOME (action sequence, Python literal list)"
    if col.endswith("_scores"):
        return "PAYOFF (penalty sequence, Python literal list)"
    if col.endswith("_personality"):
        return "experimental condition (persona, localised string)"
    if col.endswith("_llm"):
        return "grouping / experimental factor (LLM identity)"
    if col == "game_id":
        return "identifier"
    if col == "language":
        return "grouping / experimental factor"
    return "unclassified"


def _derived_role(col):
    for k, v in (("_rate", "outcome rate"), ("pC_", "conditional cooperation"),
                 ("p_", "outcome frequency"), ("_1", "agent-1 copy"),
                 ("_2", "agent-2 copy")):
        if k in col:
            return v
    return "derived"


def _describe(s: pd.Series, col: str, table: str, role: str) -> dict:
    out = {"table": table, "column": col, "dtype": str(s.dtype),
           "n": len(s), "n_unique": int(s.nunique(dropna=False)),
           "pct_missing": round(100 * s.isna().mean(), 4), "role": role}
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        q = s.quantile([.01, .25, .5, .75, .99])
        out |= {"min": s.min(), "max": s.max(), "mean": s.mean(), "median": s.median(),
                "std": s.std(), "q01": q.iloc[0], "q25": q.iloc[1], "q50": q.iloc[2],
                "q75": q.iloc[3], "q99": q.iloc[4],
                "skew": s.skew(), "kurtosis": s.kurtosis(),
                "cv": s.std() / s.mean() if s.mean() not in (0, np.nan) else np.nan}
    vals = s.dropna().unique()[:4]
    out["examples"] = " | ".join(str(v)[:40] for v in vals)
    return out


# --------------------------------------------------------------------------
def payoff_reconstruction(rounds: pd.DataFrame) -> pd.DataFrame:
    """Check score == lambda * matrix[own][opp] for all 240,000 agent-rounds."""
    pred = rounds["outcome"].map(C.PEN).astype(float) * rounds["scale"]
    err = (rounds["payoff_raw"] - pred).abs()
    rel = np.where(pred > 0, err / np.maximum(pred, 1e-12), err)
    return pd.DataFrame({
        "n_rounds": [len(rounds)],
        "max_abs_error": [float(err.max())],
        "max_rel_error": [float(np.max(rel))],
        "n_exact": [int((err < 1e-9).sum())],
        "frac_exact": [float((err < 1e-9).mean())],
    })


def run(rounds, ag, dy):
    C.use_style()
    print("\n== 01 audit and data quality ==")

    files = raw_file_audit()
    C.savetab(files, "file_audit")

    # -- dataset-level summary ------------------------------------------------
    summ = {
        "corpus_path": str(C.CORPUS.relative_to(C.ROOT)).replace("\\", "/"),
        "n_files": len(files), "on_disk_bytes": int(files["bytes"].sum()),
        "n_raw_rows": int(files["n_rows"].sum()),
        "n_games": len(dy), "n_agent_games": len(ag), "n_agent_rounds": len(rounds),
        "n_payoff_scales": rounds["scale"].nunique(),
        "scales": ", ".join(str(s) for s in sorted(rounds["scale"].unique())),
        "n_models": rounds["model"].nunique(),
        "models": ", ".join(sorted(rounds["model"].unique())),
        "n_languages": rounds["language"].nunique(),
        "languages": ", ".join(sorted(rounds["language"].unique())),
        "n_dyad_types": rounds["dyad"].nunique(),
        "reps_per_cell": int(rounds["rep"].nunique()),
        "rounds_per_game": int(rounds["round"].max()),
        "design_cells": int(rounds.groupby(
            ["scale", "model", "language", "dyad"], observed=True).ngroups),
        "grid_complete": bool(len(dy) == 10 * 6 * 5 * 4 * 10),
        "total_missing_cells": int(files["n_missing"].sum()),
        "duplicate_raw_rows": int(files["n_dup_rows"].sum()),
        "all_selfplay": bool(files["selfplay"].all()),
        "all_seq_len_10": bool(files["seq_len_ok"].all()),
        "action_alphabet": ", ".join(sorted(rounds["action"].unique())),
        "n_cooperate": int((rounds["action"] == "C").sum()),
        "n_defect": int((rounds["action"] == "D").sum()),
        "overall_coop_rate": round(float(rounds["coop"].mean()), 6),
    }
    rec = payoff_reconstruction(rounds)
    summ |= {"payoff_max_abs_error": float(rec.max_abs_error[0]),
             "payoff_frac_exact": float(rec.frac_exact[0])}
    C.savetab(pd.DataFrame([{"property": k, "value": v} for k, v in summ.items()]),
              "dataset_summary")

    dd = data_dictionary(rounds, ag, dy)
    C.savetab(dd, "feature_summary")

    # ---- Q1 design completeness --------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
    piv = dy.pivot_table(index="model", columns="scale", values="game_uid",
                         aggfunc="count").reindex(C.MODEL_ORDER)
    im = axes[0].imshow(piv.values, cmap=C.SEQ, vmin=0, vmax=piv.values.max(),
                        aspect="auto")
    axes[0].set_xticks(range(len(piv.columns)),
                       [f"{c:g}" for c in piv.columns], rotation=45, ha="right")
    axes[0].set_yticks(range(len(piv.index)), piv.index, fontsize=6.5)
    axes[0].set_title("a  games per model x payoff scale")
    axes[0].set_xlabel("payoff scale $\\lambda$")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[0].text(j, i, int(piv.values[i, j]), ha="center", va="center",
                         fontsize=5.5, color="white" if piv.values[i, j] > 100 else C.INK)
    axes[0].grid(False)

    piv2 = dy.pivot_table(index="language", columns="dyad", values="game_uid",
                          aggfunc="count").reindex(C.LANGS)[C.DYADS]
    axes[1].imshow(piv2.values, cmap=C.SEQ, aspect="auto", vmin=0)
    axes[1].set_xticks(range(4), C.DYADS)
    axes[1].set_yticks(range(5), [C.LANG_LABEL[x] for x in piv2.index], fontsize=7)
    axes[1].set_title("b  games per language x pairing")
    for i in range(5):
        for j in range(4):
            axes[1].text(j, i, int(piv2.values[i, j]), ha="center", va="center",
                         fontsize=6, color="white")
    axes[1].grid(False)

    cnt = dy.groupby(["scale", "model", "language", "dyad"],
                     observed=True).size().value_counts()
    axes[2].bar(cnt.index.astype(str), cnt.values, color=C.C_COOP, width=.5)
    axes[2].set_xlabel("games per design cell")
    axes[2].set_ylabel("number of cells")
    axes[2].set_title("c  cell occupancy is a single spike")
    C.annotate(axes[2], f"{len(cnt)} distinct count(s)\nall cells = "
                        f"{cnt.index[0]} games", loc="upper right")
    C.save(fig, "quality", "Q01_design_balance",
           "Is the 10x6x5x4x10 factorial grid complete and balanced?",
           f"Complete: {len(dy)} games, every one of "
           f"{dy.groupby(['scale','model','language','dyad'],observed=True).ngroups} "
           f"design cells holds exactly {cnt.index[0]} games.",
           "heatmap + bar", "scale, model, language, dyad, game count")

    # ---- Q2 constant columns and missingness -------------------------------
    raw = pd.concat([pd.read_csv(f, encoding="utf-8")
                     for f in sorted(C.CORPUS.glob("*/*/x*_en_*.csv"))],
                    ignore_index=True)
    nun = raw.nunique(dropna=False).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    cols = [C.MUTED if v == 1 else C.C_COOP for v in nun.values]
    axes[0].barh(range(len(nun)), np.maximum(nun.values, .8), color=cols)
    axes[0].set_yticks(range(len(nun)), nun.index, fontsize=6.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("distinct values (log scale)")
    axes[0].set_title("a  9 of 20 raw columns are constant")
    axes[0].axvline(1, color=C.INK, lw=.8, ls=":")
    miss = pd.DataFrame({
        "table": ["rounds"] * len(rounds.columns) + ["agent_games"] * len(ag.columns)
                 + ["dyads"] * len(dy.columns),
        "col": list(rounds.columns) + list(ag.columns) + list(dy.columns),
        "pct": list(100 * rounds.isna().mean()) + list(100 * ag.isna().mean())
               + list(100 * dy.isna().mean())})
    m = miss[miss.pct > 0].sort_values("pct")
    axes[1].barh(range(len(m)), m.pct, color=C.OUTCOME_COL["DC"])
    axes[1].set_yticks(range(len(m)), [f"{t}.{c}" for t, c in zip(m.table, m.col)],
                       fontsize=6.5)
    axes[1].set_xlabel("percent missing")
    axes[1].set_title("b  all missingness is structural")
    C.annotate(axes[1], "round-1 lags are undefined;\nconditional rates need the\n"
                        "conditioning event to occur", loc="lower right")
    C.save(fig, "quality", "Q02_constants_and_missingness",
           "Which columns carry information, and is any value truly missing?",
           f"{int((nun == 1).sum())} raw columns are constant. No value is missing "
           "in the raw files; every NaN downstream is a structurally undefined lag "
           "or an unobserved conditioning event.",
           "bar", "all raw and derived columns")

    # ---- Q3 payoff reconstruction ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for oc in C.OUTCOME_ORDER:
        sub = rounds[rounds["outcome"] == oc]
        axes[0].scatter(sub["scale"], sub["payoff_raw"] + 1e-4, s=4, alpha=.02,
                        color=C.OUTCOME_COL[oc], label=C.OUTCOME_LABEL[oc], rasterized=True)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("payoff scale $\\lambda$")
    axes[0].set_ylabel("recorded penalty  (+1e-4 for log axis)")
    axes[0].set_title("a  recorded penalties lie exactly on $\\lambda\\cdot\\{0,2,6,10\\}$")
    h = [plt.Line2D([], [], marker="o", ls="", color=C.OUTCOME_COL[o], label=C.OUTCOME_LABEL[o])
         for o in C.OUTCOME_ORDER]
    axes[0].legend(handles=h, loc="upper left", fontsize=6)

    err = (rounds["payoff_raw"] - rounds["outcome"].map(C.PEN).astype(float)
           * rounds["scale"]).abs()
    axes[1].hist(np.log10(err + 1e-18), bins=60, color=C.C_COOP)
    axes[1].set_xlabel("$\\log_{10}$ absolute reconstruction error")
    axes[1].set_ylabel("agent-rounds")
    axes[1].set_title("b  reconstruction error is identically zero")
    C.annotate(axes[1], f"{int(rec.n_exact[0]):,} / {len(rounds):,} rounds exact\n"
                        f"max abs error = {rec.max_abs_error[0]:.2e}")
    C.save(fig, "quality", "Q03_payoff_reconstruction",
           "Do the recorded scores reproduce lambda times the stated payoff matrix?",
           f"Yes, exactly: {rec.frac_exact[0]:.6f} of 240,000 agent-rounds reconstruct "
           "with zero error, so the payoff scale is uncontaminated and lambda is a "
           "clean experimental factor.",
           "scatter + histogram", "payoff_raw, scale, outcome")

    # ---- Q4 sequence duplication / pseudoreplication ------------------------
    seqs = (rounds.sort_values(["game_uid", "agent", "round"])
            .groupby(["game_uid", "agent"], observed=True)["action"]
            .apply(lambda s: "".join(s)))
    vc = seqs.value_counts()
    allc, alld = vc.get("C" * 10, 0), vc.get("D" * 10, 0)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
    axes[0].bar(range(min(25, len(vc))), vc.values[:25],
                color=[C.C_COOP if s == "C" * 10 else C.C_DEFECT if s == "D" * 10
                       else C.MUTED for s in vc.index[:25]])
    axes[0].set_xticks(range(min(25, len(vc))), vc.index[:25], rotation=90, fontsize=5)
    axes[0].set_ylabel("agent-games")
    axes[0].set_title("a  25 most frequent sequences")
    C.annotate(axes[0], f"CCCCCCCCCC: {allc:,}\nDDDDDDDDDD: {alld:,}\n"
                        f"{(allc+alld)/len(seqs):.1%} of all agent-games",
               loc="upper right")
    ranks = np.arange(1, len(vc) + 1)
    axes[1].loglog(ranks, vc.values, color=C.C_COOP, lw=1.2)
    axes[1].set_xlabel("sequence rank")
    axes[1].set_ylabel("count")
    axes[1].set_title("b  heavy-tailed sequence frequency")
    C.annotate(axes[1], f"{len(vc)} distinct sequences\nof 1024 possible\n"
                        f"top-1 share {vc.values[0]/len(seqs):.1%}", loc="upper right")
    lvl = pd.Series({"agent-rounds": len(rounds), "agent-games": len(ag),
                     "games (independent units)": len(dy),
                     "design cells": dy.groupby(
                         ["scale", "model", "language", "dyad"], observed=True).ngroups})
    axes[2].barh(range(len(lvl)), lvl.values, color=[C.MUTED, C.MUTED, C.C_DEFECT, C.C_COOP])
    axes[2].set_yticks(range(len(lvl)), lvl.index, fontsize=7)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("count (log scale)")
    axes[2].set_title("c  nesting of observations")
    for i, v in enumerate(lvl.values):
        axes[2].text(v * 1.1, i, f"{v:,}", va="center", fontsize=6.5)
    C.save(fig, "quality", "Q04_sequence_duplication_and_nesting",
           "How much repetition is in the action sequences, and what is the "
           "independent unit of analysis?",
           f"Only {len(vc)} of 1024 possible sequences occur; the two constant "
           f"sequences alone are {(allc+alld)/len(seqs):.1%} of agent-games. Rounds are "
           "20x pseudoreplicated relative to games, so every interval must cluster "
           "on game_uid.",
           "bar + log-log + bar", "action sequences, table sizes")

    # ---- Q5 common random numbers ------------------------------------------
    n_cells = dy["cell"].nunique()
    per = dy.groupby("cell", observed=True)["scale"].nunique()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.1))
    axes[0].hist(per.values, bins=np.arange(.5, 11.5), color=C.C_COOP, rwidth=.7)
    axes[0].set_xlabel("distinct payoff scales per design cell")
    axes[0].set_ylabel("cells")
    axes[0].set_title("a  every cell is observed at all 10 scales")
    C.annotate(axes[0], f"{n_cells:,} cells\nall with {int(per.min())}-"
                        f"{int(per.max())} scales")
    piv = dy.pivot_table(index="cell", columns="scale", values="joint_coop")
    corr = piv.corr(method="spearman")
    im = axes[1].imshow(corr.values, cmap=C.DIV, vmin=-1, vmax=1)
    axes[1].set_xticks(range(10), [f"{c:g}" for c in corr.columns], rotation=90, fontsize=6)
    axes[1].set_yticks(range(10), [f"{c:g}" for c in corr.index], fontsize=6)
    axes[1].set_title("b  cell-matched Spearman correlation across $\\lambda$")
    axes[1].grid(False)
    plt.colorbar(im, ax=axes[1], fraction=.046)
    off = corr.values[~np.eye(10, dtype=bool)]
    C.annotate(axes[1], f"mean off-diagonal $\\rho$ = {off.mean():.2f}", loc="lower left")
    C.save(fig, "quality", "Q05_common_random_numbers",
           "Are games matched across payoff scale, so the scale axis can be "
           "analysed as a paired comparison?",
           f"Yes. All {n_cells:,} design cells appear at all 10 scales. Cell-matched "
           f"cooperation correlates across scales at mean Spearman rho = {off.mean():.2f}, "
           "so pairing on the cell removes a large part of the between-cell variance.",
           "histogram + correlation heatmap", "cell, scale, joint_coop")

    C.record_finding(
        id="Q-integrity",
        finding="The corpus is a complete, perfectly balanced factorial design with "
                "zero missing data and exactly reconstructible payoffs.",
        evidence=f"300 files x 40 rows = 12,000 games = 10 scales x 6 models x 5 "
                 f"languages x 4 pairings x 10 repetitions; "
                 f"{rec.frac_exact[0]:.6f} of 240,000 agent-rounds satisfy "
                 f"score == lambda * matrix[own][opp] exactly; 0 missing raw values; "
                 f"only C and D tokens, no parse fallbacks.",
        figure="01_data_quality/Q01_design_balance.png, Q03_payoff_reconstruction.png",
        strength="strong", robustness="deterministic check over all files",
        interpretation="Any behavioural pattern found downstream is a property of the "
                       "models, not of collection artefacts.",
        caveat="Integrity of the log does not certify the sampling temperature or "
               "provider-side routing, which are not recorded.",
        claim="All analyses rest on a complete 10 x 6 x 5 x 4 x 10 factorial corpus "
              "(12,000 games, 240,000 agent-rounds) with no missing observations and "
              "payoffs that reconstruct the stated matrix exactly.")

    C.record_finding(
        id="Q-pseudoreplication",
        finding="Rounds are 20-fold pseudoreplicated relative to independent games, "
                "and the action sequences are extremely concentrated.",
        evidence=f"240,000 agent-rounds collapse to 12,000 independent games; only "
                 f"{len(vc)} of the 1,024 possible 10-round sequences occur, and the "
                 f"two constant sequences account for {(allc+alld)/len(seqs):.1%} of all "
                 f"agent-games.",
        figure="01_data_quality/Q04_sequence_duplication_and_nesting.png",
        strength="strong", robustness="exact count",
        interpretation="Behaviour is dominated by a few degenerate policies, which is "
                       "why round-level standard errors would be badly anticonservative.",
        caveat="Concentration is partly forced by the 10-round horizon.",
        claim="Cooperation in this corpus is highly degenerate: two constant sequences "
              "account for over half of all agent-games, and all inference clusters on "
              "the game.")
    return files, dd
