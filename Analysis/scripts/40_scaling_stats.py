"""Statistics for the payoff-scaling manuscript.

The question is a single one: does multiplying every cell of the payoff matrix
by a positive constant change what a language model plays?  A positive
rescaling of a von Neumann-Morgenstern utility leaves the ordering of outcomes,
the best-reply correspondence, the equilibrium set and the replicator dynamics
untouched, so the predicted effect is exactly zero for any agent whose
behaviour is a function of the game.

The corpus is a single frontier arm, split by how densely each model samples
the scale:

    ten-scale     3 models x 10 scales (1e-2 .. 1e3) x 5 languages, 10 rounds
    three-scale   3 models x  3 scales (1e-1 .. 1e1) x 5 languages, 10 rounds

Both are balanced at 80 agent-games per model x language x scale cell.  The
open-weight arm of the earlier preprint is no longer read here; it lives in the
electronic supplementary material only.

Everything written here lands in ``tables/T_PS*.csv`` and is read back by
``41_fig_scaling.py``, so no figure panel computes a number of its own.

    python Analysis/scripts/40_scaling_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pdlib.style import DATADIR, TABDIR

TABDIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
N_PERM = 2000
SEED = 20260830

# The FAIRGAME templates do not state the objective identically in every
# language: en/fr/vn ask the agent to *minimise a penalty*, ar/cn ask it to
# *maximise a reward* while still calling the cell values penalties.  Language
# is therefore confounded with objective framing, and the subset that shares
# one framing is the only place a clean language contrast can be measured.
MINIMISE = ("en", "fr", "vn")

# Base cells, in the penalty units the agent is shown, before the scale is
# applied.  The two arms were run against different base matrices.
# Mot ma tran goc duy nhat cho ca corpus: (T, R, P, S) trong khong gian HINH PHAT.
# Hai khoa duoi day la hai NHOM DO PHAN GIAI, khong phai hai tro choi khac nhau, nen
# chung tro toi cung mot bo so.
_PD_BASE = (0.0, 2.0, 6.0, 10.0)
BASE = {"ten-scale": _PD_BASE, "three-scale": _PD_BASE}

ARM_MODELS = {
    "ten-scale": ["Gemini-3.5-Flash-Lite", "Gemini-3.1-Flash-Lite-Preview",
                  "GPT-5.4-Nano"],
    "three-scale": ["Claude-3.5-Haiku", "GPT-4o", "Mistral-Large"],
}
# Tra nguoc model -> nhom. Model chua dang ky se KeyError ngay o load(), co y: im
# lang bo qua mot model moi la cach hong ma khong ai phat hien duoc.
MODEL_ARM = {m: a for a, ms in ARM_MODELS.items() for m in ms}

STRAT_ORDER = ["AllC", "TFT", "WSLS", "AllD", "Ambiguous"]
# The four canonical memory-one rules.  "Ambiguous" is not one of them: it
# records that several rules reproduce the run of play equally well, which is
# a fact about identifiability rather than about which strategy was played.
NAMED = ["AllC", "TFT", "WSLS", "AllD"]
# Posterior floor below which the learned half of the read-out is reporting a
# nearest neighbour rather than a recognition.  Same value as the sister
# manuscript's abstention threshold, so the two are directly comparable.
LSTM_FLOOR = 0.90


# --------------------------------------------------------------------------
# notation: what the agent actually reads
# --------------------------------------------------------------------------
def printed(value: float) -> str:
    """The payoff string as the prompt template renders it."""
    return str(int(round(value))) if abs(value - round(value)) < 1e-9 else f"{value:g}"


def notation_features(arm: str, lam: float) -> dict:
    """Properties of the four printed cell values at this scale."""
    cells = [printed(c * lam) for c in BASE[arm]]
    digits = [len(c.replace(".", "").replace("-", "").lstrip("0")) or 1 for c in cells]
    return {
        "cells": " / ".join(cells),
        "is_fractional": int(any("." in c for c in cells)),
        "mean_glyphs": float(np.mean([len(c) for c in cells])),
        "max_digits": int(max(digits)),
        "regime": ("fractional" if lam < 1 else "unit" if lam <= 10 else "large"),
        # `regime` above is a function of lambda, not of what the prompt prints.  On
        # the decade grid the two coincide, so the distinction never mattered; off
        # the decade grid they come apart, and lambda = 0.5 is the case that matters:
        # it prints 0 / 1 / 4 / 5, all integers, at sub-unit magnitude.  Labelling it
        # "fractional" would throw away the one condition that separates a notation
        # account from a magnitude account.  `regime_printed` reads the cells instead.
        "regime_printed": _regime_printed(cells),
    }


def _regime_printed(cells: list[str]) -> str:
    """Regime read off the printed strings rather than off lambda.

    Reproduces `regime` exactly on the decade grid: 0.01 and 0.1 carry a decimal
    point; 1 and 10 top out at two and three digits; 100 and 1000 at four and five.
    """
    if any("." in c for c in cells):
        return "fractional"
    digits = max(len(c.replace("-", "").lstrip("0")) or 1 for c in cells)
    return "unit" if digits <= 3 else "large"


def load() -> pd.DataFrame:
    """Corpus frontier trong mot bang agent-game.

    Nhanh open-weight da bi bo khoi bai (2026-09-02), nen ham nay khong doc
    ``games.parquet`` nua. Cot ``arm`` giu lai ten cu nhung mang y nghia moi: no chia
    corpus theo DO PHAN GIAI lambda, vi do la doi lap con lai duy nhat co that.
    """
    front = pd.read_parquet(DATADIR / "frontier_games.parquet").copy()
    front["arm"] = front.model.map(MODEL_ARM)
    unknown = sorted(front[front.arm.isna()].model.unique())
    if unknown:
        raise KeyError(f"model chua dang ky trong ARM_MODELS: {unknown}")
    keep = ["arm", "model", "language", "scale_nominal", "game_uid", "agent",
            "personality", "opp_personality", "dyad", "n_rounds", "coop_rate",
            "opp_coop_rate", "first_move_coop", "last_move_coop", "efficiency",
            "cc_rate", "dd_rate", "cd_rate", "dc_rate"]
    df = front[keep].copy()
    df["loglam"] = np.log10(df.scale_nominal)
    df["framing"] = np.where(df.language.isin(MINIMISE), "minimise", "maximise")
    feats = {(a, l): notation_features(a, l)
             for a in BASE for l in sorted(df[df.arm == a].scale_nominal.unique())}
    key = list(zip(df.arm, df.scale_nominal))
    for f in ("regime", "is_fractional", "mean_glyphs", "max_digits"):
        df[f] = [feats[k][f] for k in key]
    df["lam_f"] = df.scale_nominal.astype(str)
    return df


def load_rounds() -> pd.DataFrame:
    front = pd.read_parquet(DATADIR / "frontier_rounds.parquet").copy()
    front["arm"] = front.model.map(MODEL_ARM)
    unknown = sorted(front[front.arm.isna()].model.unique())
    if unknown:
        raise KeyError(f"model chua dang ky trong ARM_MODELS: {unknown}")
    keep = ["arm", "model", "language", "scale_nominal", "scale_eff", "game_uid",
            "agent", "personality", "round", "coop", "opp_coop", "payoff_raw"]
    out = front[keep].copy()
    # CHU Y: nhan nay doc tu lambda chu khong tu o in ra, nen no SAI o lambda=0.5
    # (in ra "0 / 1 / 3 / 5" - toan so nguyen, tuc la `unit`). Giu lai vi moi bang da
    # cong bo deu dung no; cot `regime_printed` ngay duoi moi la cach doc dung, va
    # phan tich nao noi ve notation thi phai dung cot do.
    out["regime"] = np.where(out.scale_nominal < 1, "fractional",
                             np.where(out.scale_nominal <= 10, "unit", "large"))
    # Parallel column read off the printed cells (see `_regime_printed`).  Kept
    # separate from `regime` so every published table stays byte-identical.
    out["regime_printed"] = [
        notation_features(a, s)["regime_printed"]
        for a, s in zip(out.arm, out.scale_nominal)
    ]
    return out


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------
def dyad_boot_mean(frame: pd.DataFrame, col: str, *, n=N_BOOT, seed=SEED):
    """Bootstrap mean of `col`, resampling whole dyads.

    Rounds inside a game are autocorrelated and the two agent-rows of a dyad
    share a history, so resampling rows would narrow the interval by roughly
    a factor of sqrt(2).
    """
    rng = np.random.default_rng(seed)
    by = frame.groupby("game_uid")[col].mean().to_numpy()
    if by.size == 0:
        return np.nan, np.nan, np.nan
    draws = rng.choice(by, size=(n, by.size), replace=True).mean(axis=1)
    return float(by.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def dyad_boot_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, *, n=N_BOOT, seed=SEED):
    """Bootstrap difference of two group means, resampling whole dyads."""
    rng = np.random.default_rng(seed)
    x = a.groupby("game_uid")[col].mean().to_numpy()
    y = b.groupby("game_uid")[col].mean().to_numpy()
    dx = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    dy = rng.choice(y, size=(n, y.size), replace=True).mean(axis=1)
    d = dx - dy
    return float(x.mean() - y.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def dyad_boot_did(hi: pd.DataFrame, lo: pd.DataFrame, split: str, a: str, b: str,
                  col: str, *, n=N_BOOT, seed=SEED):
    """Bootstrap CI on (effect inside `hi`) minus (effect inside `lo`).

    The effect is the `a` minus `b` contrast on `split`.  This is the
    difference in differences the pooled curve cannot express: a persona
    effect can shift by the same amount in every model while never once
    changing sign, and only a per-model contrast distinguishes the two.
    """
    rng = np.random.default_rng(seed)

    def arms(frame):
        return (frame[frame[split] == a].groupby("game_uid")[col].mean().to_numpy(),
                frame[frame[split] == b].groupby("game_uid")[col].mean().to_numpy())

    def draw(pair):
        x, y = pair
        return (rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
                - rng.choice(y, size=(n, y.size), replace=True).mean(axis=1))

    ph, pl = arms(hi), arms(lo)
    obs = (ph[0].mean() - ph[1].mean()) - (pl[0].mean() - pl[1].mean())
    d = draw(ph) - draw(pl)
    return float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def perm_gap(frame: pd.DataFrame, factor: str, col: str, *, n=N_PERM, seed=SEED):
    """Largest between-level gap in `col`, against a dyad-level permutation null.

    The label is shuffled across dyads rather than across rows, so a dyad keeps
    both of its agent-rows together and the null preserves the dependence that
    the design creates.
    """
    rng = np.random.default_rng(seed)
    dy = frame.groupby("game_uid").agg(f=(factor, "first"), v=(col, "mean")).reset_index()
    codes, uniq = pd.factorize(dy.f)
    k = len(uniq)
    v = dy.v.to_numpy(float)
    cnt = np.bincount(codes, minlength=k).astype(float)
    obs = float((np.bincount(codes, weights=v, minlength=k) / cnt).ptp())
    null = np.empty(n)
    for i in range(n):
        c = rng.permutation(codes)
        null[i] = (np.bincount(c, weights=v, minlength=k)
                   / np.bincount(c, minlength=k)).ptp()
    return obs, float((null >= obs).mean()), float(np.percentile(null, 95))


def mix(frame: pd.DataFrame) -> np.ndarray:
    v = frame.archetype.value_counts(normalize=True)
    return np.array([v.get(k, 0.0) for k in STRAT_ORDER])


def perm_mix_tv(frame: pd.DataFrame, factor: str, *, cats=None, n=N_PERM, seed=SEED):
    """Largest total-variation distance between level mixes, and its null.

    `cats` selects the vocabulary.  With the default five it includes
    ``Ambiguous``, which is a statement about how identifiable the play is and
    not about which strategy was played; passing the four named rules instead
    measures the movement between strategies alone, on the subset the read-out
    was willing to name.
    """
    cats = list(STRAT_ORDER if cats is None else cats)
    frame = frame[frame.archetype.isin(cats)]
    rng = np.random.default_rng(seed)
    levels = sorted(frame[factor].unique())
    codes = pd.factorize(pd.Categorical(frame[factor], categories=levels))[0]
    onehot = np.zeros((len(frame), len(cats)))
    idx = pd.Categorical(frame.archetype, categories=cats).codes
    onehot[np.arange(len(frame)), idx] = 1.0

    def stat(c):
        ms = np.vstack([onehot[c == j].mean(axis=0) for j in range(len(levels))])
        return max(0.5 * np.abs(ms[i] - ms[j]).sum()
                   for i in range(len(levels)) for j in range(i + 1, len(levels)))

    obs = stat(codes)
    # one label per dyad, so both agent-rows move together under the shuffle
    dyads, inv = np.unique(frame.game_uid.to_numpy(), return_inverse=True)
    dyad_code = np.zeros(len(dyads), dtype=int)
    dyad_code[inv] = codes
    null = np.empty(n)
    for i in range(n):
        null[i] = stat(rng.permutation(dyad_code)[inv])
    return float(obs), float((null >= obs).mean()), float(np.percentile(null, 95))


def check_polarity(rounds: pd.DataFrame) -> pd.DataFrame:
    """Verify that Option A is the defecting action, on every logged round.

    The direction of the payoffs is load-bearing: reading Option A as
    cooperation inverts every cooperation rate and reverses the sign of the
    scale effect.  It is also checkable rather than a matter of interpretation,
    because FAIRGAME logs the realised payoff next to every action.
    """
    rows = []
    for arm, d in rounds.groupby("arm"):
        T, R, P, S = BASE[arm]
        c, o = d.coop.to_numpy(), d.opp_coop.to_numpy()

        def expected(ci, oi):
            return np.where((ci == 1) & (oi == 1), R,
                            np.where((ci == 1) & (oi == 0), S,
                                     np.where((ci == 0) & (oi == 1), T, P)))

        lam = d.scale_eff.to_numpy()
        stated = np.isclose(expected(c, o) * lam, d.payoff_raw, atol=1e-6).mean()
        inverted = np.isclose(expected(1 - c, 1 - o) * lam, d.payoff_raw,
                              atol=1e-6).mean()
        rows.append({"arm": arm, "rounds": len(d),
                     "match_option_a_defects": float(stated),
                     "match_option_a_cooperates": float(inverted)})
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_PS00_polarity.csv", index=False)
    assert (out.match_option_a_defects == 1.0).all(), out
    return out


# --------------------------------------------------------------------------
# T_PS01  the corpus
# --------------------------------------------------------------------------
def t01_corpus(df):
    rows = []
    for arm, d in df.groupby("arm"):
        for mdl, dm in d.groupby("model"):
            rows.append({
                "arm": arm, "model": mdl,
                "scales": len(dm.scale_nominal.unique()),
                "scale_min": dm.scale_nominal.min(), "scale_max": dm.scale_nominal.max(),
                "languages": dm.language.nunique(), "rounds": int(dm.n_rounds.iloc[0]),
                "dyads": dm.game_uid.nunique(), "agent_games": len(dm),
                "decisions": int(dm.n_rounds.sum()),
                "cells": dm.groupby(["language", "scale_nominal"]).ngroups,
                "per_cell": int(len(dm) / dm.groupby(["language", "scale_nominal"]).ngroups),
                "coop_rate": dm.coop_rate.mean(),
            })
    out = pd.DataFrame(rows).sort_values(["arm", "model"])
    out.to_csv(TABDIR / "T_PS01_corpus.csv", index=False)
    return out


# --------------------------------------------------------------------------
# T_PS02  what changes and what does not when lambda changes
# --------------------------------------------------------------------------
def t02_notation(df):
    rows = []
    for arm in ARM_MODELS:
        T, R, P, S = BASE[arm]
        for lam in sorted(df[df.arm == arm].scale_nominal.unique()):
            f = notation_features(arm, lam)
            # on the utility scale u = -penalty the dilemma indices are
            # invariant to lambda by construction; we compute them anyway so
            # the table shows the invariance rather than asserting it
            uT, uR, uP, uS = (-T * lam, -R * lam, -P * lam, -S * lam)
            rows.append({
                "arm": arm, "lambda": lam, "printed_cells": f["cells"],
                "regime": f["regime"], "is_fractional": f["is_fractional"],
                "mean_glyphs": f["mean_glyphs"], "max_digits": f["max_digits"],
                "greed": (uT - uR) / (uT - uS), "fear": (uP - uS) / (uT - uS),
                "dominant_action": "A", "nash": "(A,A)",
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_PS02_notation.csv", index=False)
    return out


# --------------------------------------------------------------------------
# T_PS03  the response curve
# --------------------------------------------------------------------------
def t03_response(df):
    rows = []
    for (arm, mdl), d in df.groupby(["arm", "model"]):
        for lam, dl in d.groupby("scale_nominal"):
            m, lo, hi = dyad_boot_mean(dl, "coop_rate")
            rows.append({"arm": arm, "model": mdl, "lambda": lam,
                         "coop": m, "lo": lo, "hi": hi, "n": len(dl)})
    for arm, d in df.groupby("arm"):
        for lam, dl in d.groupby("scale_nominal"):
            m, lo, hi = dyad_boot_mean(dl, "coop_rate")
            rows.append({"arm": arm, "model": "pooled", "lambda": lam,
                         "coop": m, "lo": lo, "hi": hi, "n": len(dl)})
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_PS03_response.csv", index=False)
    return out


# --------------------------------------------------------------------------
# T_PS04  sensitivity, rank reversals, and the omnibus test
# --------------------------------------------------------------------------
def t04_sensitivity(df, response):
    rows = []
    for (arm, mdl), d in df.groupby(["arm", "model"]):
        curve = (response[(response.arm == arm) & (response.model == mdl)]
                 .sort_values("lambda"))
        obs, p, q95 = perm_gap(d, "scale_nominal", "coop_rate")
        fit = np.polyfit(np.log10(curve["lambda"]), curve.coop, 1)
        rows.append({
            "arm": arm, "model": mdl,
            "swing": curve.coop.max() - curve.coop.min(),
            "argmin": curve.loc[curve.coop.idxmin(), "lambda"],
            "argmax": curve.loc[curve.coop.idxmax(), "lambda"],
            "monotone": int(bool(np.all(np.diff(curve.coop) >= 0)
                                 or np.all(np.diff(curve.coop) <= 0))),
            "slope_log10": fit[0],
            "gap_obs": obs, "gap_p": p, "gap_null95": q95,
        })
    out = pd.DataFrame(rows)

    # how often the choice of lambda reverses which of two models cooperates
    # more.  A benchmark that fixes one lambda reports one side of these.
    rev = []
    for arm, d in df.groupby("arm"):
        p = d.pivot_table(index="model", columns="scale_nominal", values="coop_rate")
        models, lams, n_rev, n_tot = list(p.index), list(p.columns), 0, 0
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                for a in range(len(lams)):
                    for b in range(a + 1, len(lams)):
                        n_tot += 1
                        d1 = p.iloc[i, a] - p.iloc[j, a]
                        d2 = p.iloc[i, b] - p.iloc[j, b]
                        n_rev += int(np.sign(d1) != np.sign(d2))
        spread = (p.max(axis=0) - p.min(axis=0))
        rev.append({"arm": arm, "reversals": n_rev, "comparisons": n_tot,
                    "reversal_rate": n_rev / n_tot,
                    "max_within_model_swing": float((p.max(axis=1) - p.min(axis=1)).max()),
                    "min_within_model_swing": float((p.max(axis=1) - p.min(axis=1)).min()),
                    "median_between_model_spread": float(spread.median())})
    # What a two-condition design would have seen.  The reporting standard of
    # the discussion costs one extra condition, so the question is whether one
    # decade of separation is enough; we compare it with the null of the full
    # sweep, which is conservative because a two-level design has a smaller one.
    two = []
    for (arm, mdl), d in df.groupby(["arm", "model"]):
        curve = (response[(response.arm == arm) & (response.model == mdl)]
                 .set_index("lambda").coop)
        null95 = float(out[out.model == mdl].gap_null95.iloc[0])
        for lo in sorted(curve.index):
            hi = lo * 10
            if hi not in curve.index:
                continue
            gap = abs(curve[hi] - curve[lo])
            two.append({"arm": arm, "model": mdl, "lambda_lo": lo,
                        "lambda_hi": hi, "gap": gap, "sweep_null95": null95,
                        "clears_null": int(gap > null95)})
    two = pd.DataFrame(two)
    two.to_csv(TABDIR / "T_PS04c_one_decade.csv", index=False)

    out.to_csv(TABDIR / "T_PS04_sensitivity.csv", index=False)
    pd.DataFrame(rev).to_csv(TABDIR / "T_PS04b_reversals.csv", index=False)
    return out, pd.DataFrame(rev)


# --------------------------------------------------------------------------
# T_PS05  magnitude or notation
# --------------------------------------------------------------------------
CTRL = " + C(model) + C(language) + C(personality) + C(opp_personality)"
CTRL_1M = " + C(language) + C(personality) + C(opp_personality)"


def _ladder(ctrl: str) -> dict:
    return {
        "controls only": "coop_rate ~ 1" + ctrl,
        "linear in log10 lambda": "coop_rate ~ loglam" + ctrl,
        "quadratic in log10 lambda": "coop_rate ~ loglam + I(loglam**2)" + ctrl,
        "cubic in log10 lambda": "coop_rate ~ loglam + I(loglam**2) + I(loglam**3)" + ctrl,
        "fractional flag": "coop_rate ~ is_fractional" + ctrl,
        "fractional + glyph count": "coop_rate ~ is_fractional + mean_glyphs" + ctrl,
        "notation regime (3 levels)": "coop_rate ~ C(regime)" + ctrl,
        "saturated in lambda": "coop_rate ~ C(lam_f)" + ctrl,
    }


def t05_magnitude_or_notation(df):
    rows = []
    for arm, d in df.groupby("arm"):
        for name, formula in _ladder(CTRL).items():
            m = smf.ols(formula, data=d).fit()
            rows.append({"arm": arm, "scope": "pooled", "model_spec": name,
                         "k": int(m.df_model), "bic": m.bic, "aic": m.aic,
                         "r2": m.rsquared})
        for mdl, dm in d.groupby("model"):
            for name, formula in _ladder(CTRL_1M).items():
                m = smf.ols(formula, data=dm).fit()
                rows.append({"arm": arm, "scope": mdl, "model_spec": name,
                             "k": int(m.df_model), "bic": m.bic, "aic": m.aic,
                             "r2": m.rsquared})
    out = pd.DataFrame(rows)
    best = (out.loc[out.groupby(["arm", "scope"]).bic.idxmin(),
                    ["arm", "scope", "model_spec", "bic"]]
            .rename(columns={"model_spec": "best_by_bic", "bic": "best_bic"}))
    out = out.merge(best, on=["arm", "scope"], how="left")
    out["delta_bic"] = out.bic - out.best_bic
    out.to_csv(TABDIR / "T_PS05_magnitude_or_notation.csv", index=False)

    # the three regimes, as levels
    reg = []
    for (arm, r), d in df.groupby(["arm", "regime"]):
        m, lo, hi = dyad_boot_mean(d, "coop_rate")
        reg.append({"arm": arm, "regime": r, "coop": m, "lo": lo, "hi": hi, "n": len(d)})
    for (arm, mdl, r), d in df.groupby(["arm", "model", "regime"]):
        m, lo, hi = dyad_boot_mean(d, "coop_rate")
        reg.append({"arm": arm, "regime": r, "model": mdl,
                    "coop": m, "lo": lo, "hi": hi, "n": len(d)})
    reg = pd.DataFrame(reg)
    reg["model"] = reg.model.fillna("pooled")
    reg.to_csv(TABDIR / "T_PS05b_regime_means.csv", index=False)
    return out, reg


# --------------------------------------------------------------------------
# T_PS06  where in the game the effect lives
# --------------------------------------------------------------------------
def t06_rounds(rounds):
    rows = []
    for (arm, reg), d in rounds.groupby(["arm", "regime"]):
        horizon = int(d["round"].max())
        edges = ([0, 1, 5, 15, 30] if horizon == 30 else [0, 1, 3, 6, 10])
        labels = [f"r{edges[i] + 1}-{edges[i + 1]}" if edges[i + 1] > edges[i] + 1
                  else f"r{edges[i + 1]}" for i in range(len(edges) - 1)]
        blk = pd.cut(d["round"], edges, labels=labels)
        for b, db in d.groupby(blk, observed=True):
            rows.append({"arm": arm, "regime": reg, "block": str(b),
                         "block_order": labels.index(str(b)),
                         "coop": db.coop.mean(), "n": len(db)})
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_PS06_round_blocks.csv", index=False)

    # the opening move on its own, per scale: no history exists yet, so
    # anything here is a prior the prompt induced rather than a response to play
    first = []
    r1 = rounds[rounds["round"] == 1]
    for (arm, lam), d in r1.groupby(["arm", "scale_nominal"]):
        m, lo, hi = dyad_boot_mean(d, "coop")
        first.append({"arm": arm, "model": "pooled", "lambda": lam,
                      "coop_r1": m, "lo": lo, "hi": hi})
    for (arm, mdl, lam), d in r1.groupby(["arm", "model", "scale_nominal"]):
        m, lo, hi = dyad_boot_mean(d, "coop")
        first.append({"arm": arm, "model": mdl, "lambda": lam,
                      "coop_r1": m, "lo": lo, "hi": hi})
    first = pd.DataFrame(first)
    first.to_csv(TABDIR / "T_PS06b_opening_move.csv", index=False)

    # share of the whole-game effect that the opening move already carries
    share = []
    for arm, d in rounds.groupby("arm"):
        f = d[d["round"] == 1].groupby("regime").coop.mean()
        a = d.groupby("regime").coop.mean()
        share.append({"arm": arm,
                      "gap_r1": float(f.get("fractional", np.nan) - f.get("unit", np.nan)),
                      "gap_all": float(a.get("fractional", np.nan) - a.get("unit", np.nan))})
    share = pd.DataFrame(share)
    share["r1_share_of_gap"] = share.gap_r1 / share.gap_all
    share.to_csv(TABDIR / "T_PS06c_opening_share.csv", index=False)
    return out, first, share


# --------------------------------------------------------------------------
# T_PS07  what the scale gates: the persona instruction
# --------------------------------------------------------------------------
def t07_persona(df):
    rows = []
    for (arm, lam), d in df.groupby(["arm", "scale_nominal"]):
        e, lo, hi = dyad_boot_diff(d[d.personality == "cooperative"],
                                   d[d.personality == "selfish"], "coop_rate")
        rows.append({"arm": arm, "model": "pooled", "lambda": lam,
                     "effect": e, "lo": lo, "hi": hi,
                     "regime": d.regime.iloc[0]})
    for (arm, mdl, lam), d in df.groupby(["arm", "model", "scale_nominal"]):
        e, lo, hi = dyad_boot_diff(d[d.personality == "cooperative"],
                                   d[d.personality == "selfish"], "coop_rate")
        rows.append({"arm": arm, "model": mdl, "lambda": lam,
                     "effect": e, "lo": lo, "hi": hi,
                     "regime": d.regime.iloc[0]})
    out = pd.DataFrame(rows)
    out["compliant"] = (out.lo > 0).astype(int)
    out["inverted"] = (out.hi < 0).astype(int)
    out.to_csv(TABDIR / "T_PS07_persona_by_scale.csv", index=False)

    # the interaction itself, as one number per arm
    inter = []
    for arm, d in df.groupby("arm"):
        m = smf.ols("coop_rate ~ C(regime) * C(personality)" + CTRL, data=d).fit()
        m0 = smf.ols("coop_rate ~ C(regime) + C(personality)" + CTRL, data=d).fit()
        f = m.compare_f_test(m0)
        inter.append({"arm": arm, "F": f[0], "p": f[1], "df": f[2],
                      "bic_with": m.bic, "bic_without": m0.bic})
    inter = pd.DataFrame(inter)
    inter.to_csv(TABDIR / "T_PS07b_persona_interaction.csv", index=False)

    # ---- the per-model statement the pooled curve cannot make --------------
    # Pooling averages models whose persona effect carries opposite constant
    # signs, so the pooled curve can cross zero without any model crossing it.
    # What is common across models is the *shift*, not the sign, and the two
    # have to be separated before either is claimed.
    reg = []
    for (arm, mdl), d in df.groupby(["arm", "model"]):
        for regime, dr in d.groupby("regime"):
            e, lo, hi = dyad_boot_diff(dr[dr.personality == "cooperative"],
                                       dr[dr.personality == "selfish"], "coop_rate")
            reg.append({"arm": arm, "model": mdl, "regime": regime,
                        "effect": e, "lo": lo, "hi": hi, "n": len(dr),
                        "compliant": int(lo > 0), "inverted": int(hi < 0)})
    reg = pd.DataFrame(reg)
    reg.to_csv(TABDIR / "T_PS07c_persona_by_regime.csv", index=False)

    # the shift across each regime boundary, per model, as a difference in
    # differences with its own dyad bootstrap
    shifts = []
    for (arm, mdl), d in df.groupby(["arm", "model"]):
        present = set(d.regime.unique())
        for lo_r, hi_r in (("fractional", "unit"), ("unit", "large")):
            if not {lo_r, hi_r} <= present:
                continue
            s, slo, shi = dyad_boot_did(d[d.regime == hi_r], d[d.regime == lo_r],
                                        "personality", "cooperative", "selfish",
                                        "coop_rate")
            shifts.append({"arm": arm, "model": mdl, "from": lo_r, "to": hi_r,
                           "shift": s, "lo": slo, "hi": shi,
                           "toward_compliance": int(s > 0),
                           "significant": int(slo * shi > 0)})
    shifts = pd.DataFrame(shifts)
    shifts.to_csv(TABDIR / "T_PS07d_persona_shift.csv", index=False)

    # does the regime x persona interaction itself differ across models?  If
    # it does, the pooled two-way interaction is not a summary of a shared
    # pattern and must not be reported as one.
    three = []
    for arm, d in df.groupby("arm"):
        full = ("coop_rate ~ C(model)*C(regime)*C(personality)"
                " + C(language) + C(opp_personality)")
        red = ("coop_rate ~ C(model)*C(regime) + C(model)*C(personality)"
               " + C(regime)*C(personality) + C(language) + C(opp_personality)")
        m = smf.ols(full, data=d).fit()
        m0 = smf.ols(red, data=d).fit()
        f = m.compare_f_test(m0)
        n_const = (reg[reg.arm == arm].groupby("model")
                   .apply(lambda g: int(g.compliant.all() or g.inverted.all()),
                          include_groups=False).sum())
        three.append({"arm": arm, "F": f[0], "p": f[1], "df": f[2],
                      "models": d.model.nunique(),
                      "models_constant_sign": int(n_const)})
    three = pd.DataFrame(three)
    three.to_csv(TABDIR / "T_PS07e_persona_threeway.csv", index=False)
    return out, inter, reg, shifts, three


# --------------------------------------------------------------------------
# T_PS08  language, framing, and their interaction with the scale
# --------------------------------------------------------------------------
def t08_language(df):
    cells = (df.groupby(["arm", "language", "scale_nominal"])
             .coop_rate.mean().reset_index())
    cells.to_csv(TABDIR / "T_PS08_language_cells.csv", index=False)

    rows = []
    for arm, d in df.groupby("arm"):
        # language spread at each scale, and the scale swing within each language
        p = d.pivot_table(index="language", columns="scale_nominal", values="coop_rate")
        m = smf.ols("coop_rate ~ C(regime) * C(language)" + CTRL, data=d).fit()
        m0 = smf.ols("coop_rate ~ C(regime) + C(language)" + CTRL, data=d).fit()
        f = m.compare_f_test(m0)
        # framing subset: the same contrast with the objective held fixed
        fr = d.groupby("framing").coop_rate.mean()
        mf = smf.ols("coop_rate ~ C(regime) * C(framing)" + CTRL, data=d).fit()
        mf0 = smf.ols("coop_rate ~ C(regime) + C(framing)" + CTRL, data=d).fit()
        ff = mf.compare_f_test(mf0)
        rows.append({
            "arm": arm,
            "lang_spread_min": float((p.max(axis=0) - p.min(axis=0)).min()),
            "lang_spread_max": float((p.max(axis=0) - p.min(axis=0)).max()),
            "scale_swing_min": float((p.max(axis=1) - p.min(axis=1)).min()),
            "scale_swing_max": float((p.max(axis=1) - p.min(axis=1)).max()),
            "regime_x_language_F": f[0], "regime_x_language_p": f[1],
            "framing_gap": float(fr.get("minimise", np.nan) - fr.get("maximise", np.nan)),
            "regime_x_framing_F": ff[0], "regime_x_framing_p": ff[1],
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_PS08b_language_tests.csv", index=False)

    # the response curve under each objective framing, since the reward-reading
    # account predicts the two should differ in shape and not only in level
    fr_rows = []
    for (arm, fr, lam), d in df.groupby(["arm", "framing", "scale_nominal"]):
        m, lo, hi = dyad_boot_mean(d, "coop_rate")
        fr_rows.append({"arm": arm, "framing": fr, "lambda": lam,
                        "coop": m, "lo": lo, "hi": hi})
    pd.DataFrame(fr_rows).to_csv(TABDIR / "T_PS08c_framing_curves.csv", index=False)
    return out


# --------------------------------------------------------------------------
# T_PS09  what does not move
# --------------------------------------------------------------------------
def t09_invariants(df):
    """Statistics measured on the same games, scale by scale.

    A corpus in which everything moves is a noisy corpus.  The point of this
    table is that the coordination structure of play is flat across five orders
    of magnitude while the level of cooperation is not.
    """
    measures = {
        "cooperation rate": "coop_rate",
        "mutual cooperation (CC)": "cc_rate",
        "mutual defection (DD)": "dd_rate",
        "miscoordination (CD+DC)": "_mis",
        "efficiency": "efficiency",
        "opening cooperation": "first_move_coop",
    }
    d = df.copy()
    d["_mis"] = d.cd_rate + d.dc_rate
    rows = []
    for arm, da in d.groupby("arm"):
        for label, col in measures.items():
            obs, p, q95 = perm_gap(da, "scale_nominal", col)
            by = da.groupby("scale_nominal")[col].mean()
            rows.append({"arm": arm, "measure": label, "column": col,
                         "min": by.min(), "max": by.max(), "swing": by.max() - by.min(),
                         "gap_obs": obs, "p": p, "null95": q95,
                         "moves": int(p < 0.05)})
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_PS09_invariants.csv", index=False)

    # role asymmetry: the first-listed agent against the second, at each scale
    role = (d.groupby(["arm", "scale_nominal", "agent"]).coop_rate.mean()
            .unstack("agent"))
    role.columns = [f"agent{c}" for c in role.columns]
    role["gap"] = role.iloc[:, 0] - role.iloc[:, 1]
    role.reset_index().to_csv(TABDIR / "T_PS09b_role_asymmetry.csv", index=False)
    return out


# --------------------------------------------------------------------------
# T_PS10  the strategy mix a standard read-out returns
# --------------------------------------------------------------------------
def t10_mix():
    # Nhanh open-weight da bi bo khoi bai (2026-09-02): read-out chi chay tren
    # corpus frontier, va chia theo dung hai nhom do phan giai nhu moi bang khac.
    a = pd.read_parquet(DATADIR / "frontier_archetypes.parquet").copy()
    a["arm"] = a.model.map(MODEL_ARM)
    unknown = sorted(set(a.model[a.arm.isna()]))
    if unknown:
        raise KeyError(f"model chua dang ky trong ARM_MODELS: {unknown}")
    keep = ["arm", "model", "scale_nominal", "game_uid", "agent", "archetype",
            "assignment", "confidence", "language", "personality"]
    a = a[keep]

    # Where each label came from.  `assignment` records the stage of the
    # read-out that produced it: an exact rule match, several rules at once,
    # or the LSTM.  A label from the network above the 0.90 posterior floor is
    # a different kind of evidence from one below it, and the composition is
    # reported rather than folded away, because it is not constant across the
    # sweep.
    a["source"] = np.where(a.assignment == "exact", "rule-exact",
                  np.where(a.assignment == "ambiguous", "rule-ambiguous",
                  np.where(a.confidence >= LSTM_FLOOR, "lstm-confident",
                           "lstm-below-floor")))

    shares = (a.groupby(["arm", "scale_nominal"]).archetype
              .value_counts(normalize=True).rename("share").reset_index())
    shares = shares.merge(a.groupby(["arm", "scale_nominal"]).size().rename("n"),
                          on=["arm", "scale_nominal"])
    shares.to_csv(TABDIR / "T_PS10_mix_by_scale.csv", index=False)

    rows = []
    for arm, d in a.groupby("arm"):
        for mdl, dm in [("pooled", d)] + sorted(d.groupby("model")):
            tv, p, q = perm_mix_tv(dm, "scale_nominal")
            tvn, pn, qn = perm_mix_tv(dm, "scale_nominal", cats=NAMED)
            rows.append({"arm": arm, "model": mdl,
                         "max_tv": tv, "p": p, "null95": q,
                         "max_tv_named": tvn, "p_named": pn, "null95_named": qn,
                         "named_share": float(dm.archetype.isin(NAMED).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_PS10b_mix_tests.csv", index=False)

    per_model = (a.groupby(["arm", "model", "scale_nominal"]).archetype
                 .value_counts(normalize=True).rename("share").reset_index())
    per_model.to_csv(TABDIR / "T_PS10c_mix_by_model.csv", index=False)

    # ---- where the labels come from, and whether that moves with the scale -
    comp = (a.groupby(["arm", "scale_nominal"]).source
            .value_counts(normalize=True).rename("share").reset_index())
    tot = (a.groupby("arm").source.value_counts(normalize=True)
           .rename("share").reset_index())
    tot["scale_nominal"] = "all"
    comp = pd.concat([comp, tot], ignore_index=True)
    comp.to_csv(TABDIR / "T_PS10d_readout_source.csv", index=False)

    # ---- one curve per strategy, so the shape is visible and not just the TV
    # A single total-variation number says the mix moved; it cannot say which
    # labels moved, and the answer to that turns out to be the whole result.
    curves, swings = [], []
    for arm, d in a.groupby("arm"):
        for strat in STRAT_ORDER:
            d = d.copy()
            d["ind"] = (d.archetype == strat).astype(float)
            for lam, dl in d.groupby("scale_nominal"):
                m, lo, hi = dyad_boot_mean(dl, "ind")
                curves.append({"arm": arm, "archetype": strat, "lambda": lam,
                               "share": m, "lo": lo, "hi": hi})
            obs, p, q95 = perm_gap(d, "scale_nominal", "ind")
            swings.append({"arm": arm, "archetype": strat, "swing": obs,
                           "p": p, "null95": q95, "moves": int(p < 0.05)})
    pd.DataFrame(curves).to_csv(TABDIR / "T_PS10e_strategy_curves.csv", index=False)
    swings = pd.DataFrame(swings)
    swings.to_csv(TABDIR / "T_PS10f_strategy_swings.csv", index=False)
    return shares, out, comp, swings


# --------------------------------------------------------------------------
def main():
    print("[load]")
    df = load()
    rounds = load_rounds()
    print(f"  {len(df):,} agent-games, {df.game_uid.nunique():,} dyads, "
          f"{int(df.n_rounds.sum()):,} decisions")

    print("[T_PS00] payoff polarity")
    print(check_polarity(rounds).to_string(index=False))
    print("[T_PS01] corpus");        print(t01_corpus(df).to_string(index=False))
    print("[T_PS02] notation");      print(t02_notation(df).to_string(index=False))
    print("[T_PS03] response curve")
    resp = t03_response(df)
    print(resp[resp.model == "pooled"].to_string(index=False))
    print("[T_PS04] sensitivity")
    sens, rev = t04_sensitivity(df, resp)
    print(sens.to_string(index=False)); print(rev.to_string(index=False))
    print("[T_PS05] magnitude or notation")
    lad, reg = t05_magnitude_or_notation(df)
    print(lad[lad.scope == "pooled"].to_string(index=False))
    print(reg[reg.model == "pooled"].to_string(index=False))
    print("[T_PS06] where the effect lives")
    blocks, first, share = t06_rounds(rounds)
    print(blocks.to_string(index=False)); print(share.to_string(index=False))
    print("[T_PS07] persona gating")
    per, inter, reg, shifts, three = t07_persona(df)
    print(per[per.model == "pooled"].to_string(index=False))
    print(inter.to_string(index=False))
    print("  per model and regime:")
    print(reg.pivot_table(index=["arm", "model"], columns="regime",
                          values="effect").round(3).to_string())
    print("  shift across each boundary (positive = toward compliance):")
    print(shifts.round(3).to_string(index=False))
    print(three.to_string(index=False))
    print("[T_PS08] language and framing")
    print(t08_language(df).to_string(index=False))
    print("[T_PS09] what does not move")
    print(t09_invariants(df).to_string(index=False))
    print("[T_PS10] strategy mix")
    shares, tests, comp, swings = t10_mix()
    print(tests.to_string(index=False))
    print("  where the labels come from:")
    print(comp.pivot_table(index=["arm", "scale_nominal"], columns="source",
                           values="share").round(3).to_string())
    print("  per strategy, across the sweep:")
    print(swings.round(3).to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
