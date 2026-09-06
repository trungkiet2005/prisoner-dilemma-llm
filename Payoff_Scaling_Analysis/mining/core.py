"""Shared loading, derivation and plotting infrastructure for the mining run.

Everything downstream imports from here so that the corpus is parsed once, the
derived quantities are defined once, and every figure uses the same typography.

Design facts this module encodes (all verified in Dataset/DATA_CARD_frontier_llm.md):

* `OptionA` = DEFECT, `OptionB` = COOPERATE.  The prompt states the numbers as
  penalties and tells the agent to minimise them, so the dominant action is the
  one with the lower number in both columns, which is `OptionA`.
* The corpus is a fully balanced 10 x 6 x 5 x 4 x 10 grid: payoff scale lambda,
  model, language, personality pairing, repetition.  12,000 games.
* Games are matched across lambda by common random numbers, so the scale axis is
  a within-cell paired comparison, not independent samples.
* Rounds are not independent observations.  Cluster on `game_uid` everywhere.
"""
from __future__ import annotations

import ast
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "Dataset" / "data_fairgame_frontier_llm"
OUT = ROOT / "Payoff_Scaling_Analysis"
CACHE = OUT / "cache"
TABLES = OUT / "tables"

DIRS = {
    "quality": OUT / "01_data_quality",
    "dist": OUT / "02_distributions",
    "payoff": OUT / "03_payoff",
    "scaling": OUT / "04_scaling",
    "fairness": OUT / "05_fairness",
    "llm": OUT / "06_llm_comparison",
    "egt": OUT / "07_game_theory",
    "frontier": OUT / "08_frontier",
    "robust": OUT / "09_robustness",
    "advanced": OUT / "10_advanced",
}
for _d in [CACHE, TABLES, *DIRS.values()]:
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# game constants (penalty space, scale = 1)
# --------------------------------------------------------------------------
ACTION_MAP = {"OptionA": "D", "OptionB": "C"}
PERSONALITY_MAP = {
    "cooperative": "cooperative", "COOPERATIVE": "cooperative",
    "selfish": "selfish", "SELFISH": "selfish",
    "coopératif": "cooperative", "égoïste": "selfish",
    "một người hợp tác": "cooperative", "một người ích kỷ": "selfish",
    "合作型的": "cooperative", "自私型的": "selfish",
    "متعاون": "cooperative", "أناني": "selfish",
}
MODEL_MAP = {
    "claude-haiku-4-5-20251001": "Claude-Haiku-4.5",
    "gemini-3.1-flash-lite-preview": "Gemini-3.1-Flash-Lite",
    "gemini-3.5-flash-lite": "Gemini-3.5-Flash-Lite",
    "gpt-5.4-nano-2026-03-17": "GPT-5.4-Nano",
    "grok-4.20-0309-non-reasoning": "Grok-4.20-Non-Reasoning",
    "qwen3-235b-a22b-instruct-2507": "Qwen3-235B-A22B",
}
VENDOR = {
    "Claude-Haiku-4.5": "Anthropic",
    "Gemini-3.1-Flash-Lite": "Google",
    "Gemini-3.5-Flash-Lite": "Google",
    "GPT-5.4-Nano": "OpenAI",
    "Grok-4.20-Non-Reasoning": "xAI",
    "Qwen3-235B-A22B": "Alibaba",
}
MODEL_ORDER = ["Claude-Haiku-4.5", "Qwen3-235B-A22B", "GPT-5.4-Nano",
               "Gemini-3.1-Flash-Lite", "Grok-4.20-Non-Reasoning",
               "Gemini-3.5-Flash-Lite"]

# penalties the agent is shown, at lambda = 1
PEN = {"CC": 2.0, "CD": 10.0, "DC": 0.0, "DD": 6.0}
T_PEN, R_PEN, P_PEN, S_PEN = 0.0, 2.0, 6.0, 10.0
OUTCOME_LETTER = {"CC": "R", "CD": "S", "DC": "T", "DD": "P"}

SCALES = [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0]
LANGS = ["en", "fr", "ar", "cn", "vn"]
LANG_LABEL = {"en": "English", "fr": "French", "ar": "Arabic",
              "cn": "Chinese", "vn": "Vietnamese"}
DYADS = ["CvC", "CvS", "SvC", "SvS"]

# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8b8983"
GRID = "#e3e2db"

C_COOP = "#2a78d6"
C_DEFECT = "#e34948"
OUTCOME_COL = {"CC": "#2a78d6", "CD": "#1baf7a", "DC": "#eda100", "DD": "#e34948"}
OUTCOME_ORDER = ["CC", "CD", "DC", "DD"]
OUTCOME_LABEL = {"CC": "CC  mutual cooperation (R)", "CD": "CD  exploited (S)",
                 "DC": "DC  exploiting (T)", "DD": "DD  mutual defection (P)"}

# six qualitatively separated hues, one per model
MODEL_COL = {
    "Claude-Haiku-4.5": "#c1440e",
    "Qwen3-235B-A22B": "#8250b8",
    "GPT-5.4-Nano": "#0f8a72",
    "Gemini-3.1-Flash-Lite": "#d99400",
    "Grok-4.20-Non-Reasoning": "#3b6fd4",
    "Gemini-3.5-Flash-Lite": "#c02c6b",
}
LANG_COL = {"en": "#2a78d6", "fr": "#1baf7a", "ar": "#eda100",
            "cn": "#e34948", "vn": "#8250b8"}
DYAD_COL = {"CvC": "#2a78d6", "CvS": "#1baf7a", "SvC": "#eda100", "SvS": "#e34948"}

SEQ = mpl.colors.LinearSegmentedColormap.from_list(
    "pd_seq", ["#f7f5ef", "#cfe0f2", "#7fb0e0", "#3b6fd4", "#1d3f86"])
DIV = mpl.colors.LinearSegmentedColormap.from_list(
    "pd_div", ["#c1440e", "#e8a480", "#f4f2ea", "#8fb6e6", "#1d3f86"])


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 300, "figure.dpi": 120,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 7.0,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.7,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "lines.linewidth": 1.5,
        "lines.markersize": 4.5, "errorbar.capsize": 0,
        "figure.constrained_layout.use": True,
    })


_FIG_LOG: list[dict] = []


def save(fig, key: str, name: str, question: str, finding: str,
         plot_type: str, variables: str, worthy: str = "yes", pdf: bool = True):
    """Save a figure into its section directory and record it in the inventory."""
    d = DIRS[key]
    png = d / f"{name}.png"
    fig.savefig(png, bbox_inches="tight")
    if pdf:
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    _FIG_LOG.append({
        "figure_id": f"{d.name}/{name}",
        "filename": str(png.relative_to(OUT)).replace("\\", "/"),
        "section": d.name, "plot_type": plot_type, "variables": variables,
        "scientific_question": question, "main_finding": finding,
        "publication_worthy": worthy,
    })
    print(f"  [fig] {d.name}/{name}.png")


def flush_inventory(path: Path | None = None):
    path = path or (OUT / "figure_inventory.csv")
    if not _FIG_LOG:
        return
    df = pd.DataFrame(_FIG_LOG)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old.figure_id.isin(df.figure_id)], df], ignore_index=True)
    df.sort_values(["section", "figure_id"]).to_csv(path, index=False)
    _FIG_LOG.clear()


def annotate(ax, text, loc="upper left", **kw):
    xy = {"upper left": (0.02, 0.97), "upper right": (0.98, 0.97),
          "lower left": (0.02, 0.03), "lower right": (0.98, 0.03),
          "upper center": (0.5, 0.97), "lower center": (0.5, 0.03),
          "center left": (0.02, 0.5), "center right": (0.98, 0.5)}[loc]
    ha = "left" if "left" in loc else "right" if "right" in loc else "center"
    va = "top" if "upper" in loc else "bottom" if "lower" in loc else "center"
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=kw.pop("fontsize", 7), color=kw.pop("color", INK2), **kw)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def _parse_corpus() -> pd.DataFrame:
    """One row per (game, round, focal agent).  240,000 rows."""
    files = sorted(CORPUS.glob("*/*/*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV under {CORPUS}")
    recs = []
    for f in files:
        scale = float(f.parent.parent.name)
        model = MODEL_MAP[f.parent.name]
        lang = re.match(r"^x[\d.]+_([a-z]{2})_", f.name).group(1)
        df = pd.read_csv(f, encoding="utf-8")
        for i, r in enumerate(df.itertuples(index=False)):
            acts = [[ACTION_MAP[x] for x in ast.literal_eval(r.agent1_strategies)],
                    [ACTION_MAP[x] for x in ast.literal_eval(r.agent2_strategies)]]
            scs = [ast.literal_eval(r.agent1_scores), ast.literal_eval(r.agent2_scores)]
            pers = [PERSONALITY_MAP[r.agent1_personality],
                    PERSONALITY_MAP[r.agent2_personality]]
            k = int(r.game_id.split("_")[1])
            uid = f"{f.stem}#{i:03d}"
            # cell id: identical across lambda thanks to common random numbers
            cell = f"{model}|{lang}|{r.game_id}"
            n = len(acts[0])
            for a in (0, 1):
                o = 1 - a
                for t in range(n):
                    own, opp = acts[a][t], acts[o][t]
                    recs.append((
                        model, lang, scale, uid, cell, r.game_id, k // 10, k % 10,
                        a + 1, pers[a], pers[o],
                        f"{pers[a][0].upper()}v{pers[o][0].upper()}",
                        t + 1, own, opp, own + opp, float(scs[a][t]),
                    ))
    cols = ["model", "language", "scale", "game_uid", "cell", "game_id", "pairing",
            "rep", "agent", "personality", "opp_personality", "dyad", "round",
            "action", "opp_action", "outcome", "payoff_raw"]
    rounds = pd.DataFrame.from_records(recs, columns=cols)

    rounds["log_scale"] = np.log10(rounds["scale"])
    rounds["coop"] = (rounds["action"] == "C").astype(np.int8)
    rounds["opp_coop"] = (rounds["opp_action"] == "C").astype(np.int8)
    # dividing by lambda leaves float residue (6.0 -> 5.999999999999999), and the
    # support is known to be exactly {0, 2, 6, 10}, so snap it back.
    rounds["payoff_base"] = np.round(rounds["payoff_raw"] / rounds["scale"], 9)
    # higher-is-better utility in [0, 1]: S -> 0, T -> 1
    rounds["utility"] = (S_PEN - rounds["payoff_base"]) / (S_PEN - T_PEN)
    # efficiency against the mutual-cooperation benchmark: CC = 1, DD = 0.5
    rounds["efficiency"] = (S_PEN - rounds["payoff_base"]) / (S_PEN - R_PEN)
    rounds["round_frac"] = rounds["round"] / 10.0

    rounds = rounds.sort_values(["game_uid", "agent", "round"]).reset_index(drop=True)
    g = rounds.groupby(["game_uid", "agent"], sort=False)
    rounds["prev_action"] = g["action"].shift(1)
    rounds["prev_opp_action"] = g["opp_action"].shift(1)
    rounds["prev_outcome"] = g["outcome"].shift(1)
    rounds["prev_letter"] = rounds["prev_outcome"].map(OUTCOME_LETTER).fillna("E")
    rounds["token"] = rounds["prev_letter"] + rounds["action"]

    for c in ["model", "language", "dyad", "personality", "opp_personality",
              "action", "opp_action", "outcome", "prev_letter", "token"]:
        rounds[c] = rounds[c].astype("category")
    return rounds


# --------------------------------------------------------------------------
# strategy classification (memory-one rules + GRIM, exact match)
# --------------------------------------------------------------------------
RULES = {
    "AllC": {"E": "C", "R": "C", "S": "C", "T": "C", "P": "C"},
    "AllD": {"E": "D", "R": "D", "S": "D", "T": "D", "P": "D"},
    "TFT":  {"E": "C", "R": "C", "S": "D", "T": "C", "P": "D"},
    "WSLS": {"E": "C", "R": "C", "S": "D", "T": "D", "P": "C"},
}
STRAT_ORDER = ["AllC", "TFT", "WSLS", "GRIM", "AllD"]
STRAT_COL = {"AllC": "#2a78d6", "TFT": "#1baf7a", "WSLS": "#eda100",
             "GRIM": "#8250b8", "AllD": "#e34948", "unclassified": "#b9b7ae"}


def _classify_agent_game(actions: list[str], opp: list[str]) -> tuple:
    """Exact memory-one consistency plus GRIM, and the nearest-rule distance."""
    n = len(actions)
    prev = ["E"] + [OUTCOME_LETTER[actions[t] + opp[t]] for t in range(n - 1)]
    fits = []
    dev = {}
    for name, rule in RULES.items():
        d = sum(rule[prev[t]] != actions[t] for t in range(n))
        dev[name] = d
        if d == 0:
            fits.append(name)
    # GRIM: cooperate until the opponent defects once, then defect forever
    grim = []
    triggered = False
    for t in range(n):
        grim.append("D" if triggered else "C")
        if opp[t] == "D":
            triggered = True
    dgrim = sum(grim[t] != actions[t] for t in range(n))
    dev["GRIM"] = dgrim
    if dgrim == 0:
        fits.append("GRIM")
    nearest = min(dev, key=lambda k: dev[k])
    return (frozenset(fits), dev, nearest, dev[nearest])


def _build_agent_games(rounds: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, focal agent).  24,000 rows."""
    keys = ["model", "language", "scale", "log_scale", "game_uid", "cell", "game_id",
            "pairing", "rep", "agent", "personality", "opp_personality", "dyad"]
    ag = (rounds.groupby(keys, observed=True, sort=False)
          .agg(coop_rate=("coop", "mean"),
               opp_coop_rate=("opp_coop", "mean"),
               first_coop=("coop", "first"),
               last_coop=("coop", "last"),
               utility=("utility", "mean"),
               efficiency=("efficiency", "mean"),
               payoff_base=("payoff_base", "mean"),
               payoff_raw=("payoff_raw", "sum"),
               cc_rate=("outcome", lambda x: (x == "CC").mean()),
               cd_rate=("outcome", lambda x: (x == "CD").mean()),
               dc_rate=("outcome", lambda x: (x == "DC").mean()),
               dd_rate=("outcome", lambda x: (x == "DD").mean()))
          .reset_index())

    # early / late cooperation, for the endgame effect
    early = (rounds[rounds["round"] <= 3].groupby(["game_uid", "agent"], observed=True)
             ["coop"].mean().rename("coop_early"))
    late = (rounds[rounds["round"] >= 8].groupby(["game_uid", "agent"], observed=True)
            ["coop"].mean().rename("coop_late"))
    ag = ag.merge(early, on=["game_uid", "agent"], how="left")
    ag = ag.merge(late, on=["game_uid", "agent"], how="left")
    ag["endgame_drop"] = ag["coop_early"] - ag["coop_late"]

    # conditional-cooperation fingerprint per agent-game
    sub = rounds[rounds["prev_letter"] != "E"]
    piv = (sub.pivot_table(index=["game_uid", "agent"], columns="prev_letter",
                           values="coop", aggfunc="mean", observed=True)
           .rename(columns=lambda c: f"pC_{c}").reset_index())
    ag = ag.merge(piv, on=["game_uid", "agent"], how="left")
    # reciprocity: P(C | opp C last) - P(C | opp D last)
    rec = (sub.assign(after=np.where(sub["prev_opp_action"] == "C", "c", "d"))
           .pivot_table(index=["game_uid", "agent"], columns="after", values="coop",
                        aggfunc="mean", observed=True)
           .rename(columns={"c": "pC_after_C", "d": "pC_after_D"}).reset_index())
    ag = ag.merge(rec, on=["game_uid", "agent"], how="left")
    ag["reciprocity"] = ag["pC_after_C"] - ag["pC_after_D"]

    # strategy labels
    seq = (rounds.sort_values(["game_uid", "agent", "round"])
           .groupby(["game_uid", "agent"], observed=True, sort=False)
           .agg(a=("action", list), o=("opp_action", list)).reset_index())
    labels, nearest, ndev, exact = [], [], [], []
    for a, o in zip(seq["a"], seq["o"]):
        fits, dev, near, dmin = _classify_agent_game(list(a), list(o))
        labels.append("+".join(sorted(fits, key=STRAT_ORDER.index)) if fits else "unclassified")
        nearest.append(near)
        ndev.append(dmin)
        exact.append(len(fits))
    seq["strategy_set"] = labels
    seq["nearest_rule"] = nearest
    seq["rule_distance"] = ndev
    seq["n_rules_fit"] = exact
    ag = ag.merge(seq.drop(columns=["a", "o"]), on=["game_uid", "agent"], how="left")

    # a single readable label: unique rule if unique, else the set, else residual
    def _canon(s):
        if s == "unclassified":
            return "unclassified"
        parts = s.split("+")
        return parts[0] if len(parts) == 1 else "ambiguous"
    ag["strategy"] = ag["strategy_set"].map(_canon)
    return ag


def _build_dyads(rounds: pd.DataFrame, ag: pd.DataFrame) -> pd.DataFrame:
    """One row per game.  12,000 rows.  Fairness lives here."""
    a1 = ag[ag["agent"] == 1].set_index("game_uid")
    a2 = ag[ag["agent"] == 2].set_index("game_uid")
    keep = ["model", "language", "scale", "log_scale", "cell", "game_id",
            "pairing", "rep", "dyad"]
    d = a1[keep].copy()
    for c in ["coop_rate", "utility", "efficiency", "payoff_base", "payoff_raw",
              "first_coop", "coop_early", "coop_late", "reciprocity", "strategy"]:
        d[f"{c}_1"] = a1[c]
        d[f"{c}_2"] = a2[c]

    d["joint_utility"] = (d["utility_1"] + d["utility_2"]) / 2
    d["joint_efficiency"] = (d["efficiency_1"] + d["efficiency_2"]) / 2
    d["joint_coop"] = (d["coop_rate_1"] + d["coop_rate_2"]) / 2
    # fairness: absolute utility gap inside the dyad, 0 = perfectly equal
    d["utility_gap"] = (d["utility_1"] - d["utility_2"]).abs()
    d["signed_gap"] = d["utility_1"] - d["utility_2"]
    tot = d["utility_1"] + d["utility_2"]
    # Gini for two agents reduces to |u1 - u2| / (u1 + u2)
    d["gini"] = np.where(tot > 0, d["utility_gap"] / tot, np.nan)
    d["fairness"] = 1.0 - d["gini"]

    oc = (rounds[rounds["agent"] == 1]
          .pivot_table(index="game_uid", columns="outcome", values="round",
                       aggfunc="count", observed=True).fillna(0) / 10.0)
    for c in OUTCOME_ORDER:
        d[f"p_{c}"] = oc[c] if c in oc else 0.0
    # exploitation events in the dyad, either direction
    d["p_exploit"] = d["p_CD"] + d["p_DC"]

    seq = (rounds[rounds["agent"] == 1].sort_values(["game_uid", "round"])
           .groupby("game_uid", observed=True)["outcome"].apply(list))
    d["dd_absorbed"] = seq.map(_dd_absorbing)
    d["longest_dd"] = seq.map(lambda s: _longest_run(s, "DD"))
    d["first_dd"] = seq.map(lambda s: s.index("DD") + 1 if "DD" in s else np.nan)
    d["n_switch"] = (rounds[rounds["agent"] == 1].sort_values(["game_uid", "round"])
                     .groupby("game_uid", observed=True)["action"]
                     .apply(lambda s: int((s.values[1:] != s.values[:-1]).sum())))
    return d.reset_index()


def _longest_run(seq, val):
    best = cur = 0
    for x in seq:
        cur = cur + 1 if x == val else 0
        best = max(best, cur)
    return best


def _dd_absorbing(outcomes) -> int:
    if "DD" not in outcomes:
        return 0
    i = outcomes.index("DD")
    return int(all(o == "DD" for o in outcomes[i:]))


# --------------------------------------------------------------------------
# public loader
# --------------------------------------------------------------------------
def load(rebuild: bool = False):
    """Return (rounds, agent_games, dyads), parsing the corpus once and caching."""
    fr, fa, fd = (CACHE / "rounds.parquet", CACHE / "agent_games.parquet",
                  CACHE / "dyads.parquet")
    if not rebuild and fr.exists() and fa.exists() and fd.exists():
        return (pd.read_parquet(fr), pd.read_parquet(fa), pd.read_parquet(fd))
    print("parsing corpus ...")
    rounds = _parse_corpus()
    print(f"  rounds: {len(rounds):,}")
    ag = _build_agent_games(rounds)
    print(f"  agent-games: {len(ag):,}")
    dy = _build_dyads(rounds, ag)
    print(f"  dyads: {len(dy):,}")
    rounds.to_parquet(fr, index=False)
    ag.to_parquet(fa, index=False)
    dy.to_parquet(fd, index=False)
    return rounds, ag, dy


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------
def cluster_boot_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000,
                    seed: int = 0, stat=np.mean):
    """Bootstrap CI resampling whole clusters (games), not rows."""
    df = pd.DataFrame({"v": values, "g": clusters}).dropna()
    if df.empty:
        return np.nan, np.nan, np.nan
    grp = df.groupby("g", sort=False)["v"].mean().to_numpy()
    if len(grp) < 2:
        return float(grp.mean()) if len(grp) else np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(grp), size=(n_boot, len(grp)))
    boots = stat(grp[idx], axis=1)
    return float(grp.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def wilson(p, n, z=1.96):
    p = np.asarray(p, float)
    n = np.asarray(n, float)
    n = np.where(n == 0, np.nan, n)
    den = 1 + z**2 / n
    ctr = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return ctr - half, ctr + half


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def cliffs_delta(a, b):
    """Non-parametric effect size, robust to the bounded rate distributions here."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    if len(a) * len(b) > 4_000_000:  # subsample for tractability, deterministic
        rng = np.random.default_rng(0)
        a = rng.choice(a, 2000, replace=False) if len(a) > 2000 else a
        b = rng.choice(b, 2000, replace=False) if len(b) > 2000 else b
    bs = np.sort(b)
    gt = np.searchsorted(bs, a, side="left").sum()
    lt = (len(b) - np.searchsorted(bs, a, side="right")).sum()
    return float((gt - lt) / (len(a) * len(b)))


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    q = p[ok]
    n = len(q)
    if n == 0:
        return out
    order = np.argsort(q)
    ranked = q[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


_TESTS: list[dict] = []


def record_test(**kw):
    _TESTS.append(kw)


def flush_tests(path: Path | None = None):
    path = path or (TABLES / "statistical_tests.csv")
    if not _TESTS:
        return
    df = pd.DataFrame(_TESTS)
    if "p_value" in df:
        df["p_fdr"] = bh_fdr(df["p_value"].to_numpy())
    if path.exists():
        old = pd.read_csv(path)
        key = ["section", "test", "comparison"]
        have = [c for c in key if c in df.columns and c in old.columns]
        if have:
            merged = old.merge(df[have].drop_duplicates(), on=have, how="left",
                               indicator=True)
            old = old[merged["_merge"].to_numpy() == "left_only"]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    _TESTS.clear()


_FINDINGS: list[dict] = []


def record_finding(**kw):
    _FINDINGS.append(kw)


def flush_findings(path: Path | None = None):
    path = path or (CACHE / "findings.json")
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    ids = {f["id"] for f in _FINDINGS}
    keep = [f for f in old if f["id"] not in ids]
    path.write_text(json.dumps(keep + _FINDINGS, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    _FINDINGS.clear()


def savetab(df: pd.DataFrame, name: str, index: bool = False):
    p = TABLES / f"{name}.csv"
    df.to_csv(p, index=index)
    print(f"  [tab] tables/{name}.csv  ({len(df)} rows)")
    return p
