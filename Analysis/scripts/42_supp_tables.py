"""Electronic supplementary material tables for the payoff-scaling manuscript.

Reads ``tables/T_PS*.csv`` and writes ``paper_scaling/supp_tables.tex``, a
fragment that ``paper_scaling/supplementary.tex`` inputs.  Nothing here
computes a statistic: every number is copied from the table that
``40_scaling_stats.py`` wrote, for the same reason the figure script computes
nothing of its own.  A supplementary table therefore cannot drift from the
main text, and re-running the pipeline updates both.

The corpus is frontier-only since 2026-09-02.  The two groups below are two
sampling densities of one arm, not two arms: the same base matrix, the same
horizon, the same decoding settings.  Every three-scale row is padded into the
ten-scale lambda columns by value, so a column always means the same lambda.

    python Analysis/scripts/42_supp_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pdlib.style import TABDIR

OUT = Path(__file__).resolve().parents[2] / "paper_scaling" / "supp_tables.tex"

ARM_ORDER = ["ten-scale", "three-scale"]
ARM_LABEL = {"ten-scale": "ten scales", "three-scale": "three scales"}
ARM_BLOCK = {"ten-scale": "the three models that met ten scales",
             "three-scale": "the three models that met three scales"}
# Reference grid: every wide table lays its columns out on these lambdas, and a
# model that did not meet one of them gets a dash in that column rather than a
# shifted number.
LAMS = [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0]

MODEL_ORDER = ["Gemini-3.5-Flash-Lite", "Gemini-3.1-Flash-Lite-Preview",
               "GPT-5.4-Nano", "Claude-3.5-Haiku", "GPT-4o", "Mistral-Large"]
MODEL_LABEL = {
    "Gemini-3.5-Flash-Lite": "Gemini 3.5 Flash-Lite",
    "Gemini-3.1-Flash-Lite-Preview": "Gemini 3.1 Flash-Lite Prev.",
    "GPT-5.4-Nano": "GPT-5.4 Nano",
    "Claude-3.5-Haiku": "Claude 3.5 Haiku",
    "GPT-4o": "GPT-4o",
    "Mistral-Large": "Mistral Large",
}
MODEL_SHORT = {
    "Gemini-3.5-Flash-Lite": "Gem 3.5",
    "Gemini-3.1-Flash-Lite-Preview": "Gem 3.1",
    "GPT-5.4-Nano": "Nano",
    "Claude-3.5-Haiku": "Claude",
    "GPT-4o": "GPT-4o",
    "Mistral-Large": "Mistral",
}
LANG_LABEL = {"en": "English", "fr": "French", "vn": "Vietnamese",
              "cn": "Chinese", "ar": "Arabic"}
SPEC_LABEL = {
    "controls only": "no $\\lambda$ term",
    "linear in log10 lambda": "linear in $\\log_{10}\\lambda$",
    "quadratic in log10 lambda": "quadratic in $\\log_{10}\\lambda$",
    "cubic in log10 lambda": "cubic in $\\log_{10}\\lambda$",
    "fractional flag": "fractional flag",
    "fractional + glyph count": "fractional $+$ glyph count",
    "notation regime (3 levels)": "notation regime",
    "saturated in lambda": "saturated in $\\lambda$",
}
SPEC_ORDER = list(SPEC_LABEL)
STRAT_ORDER = ["AllC", "TFT", "WSLS", "AllD", "Ambiguous"]
SOURCE_LABEL = {"rule-exact": "provable rule", "rule-ambiguous": "several rules",
                "lstm-confident": "LSTM $\\geq 0.90$",
                "lstm-below-floor": "LSTM $< 0.90$"}
SOURCE_ORDER = list(SOURCE_LABEL)


def T(name: str) -> pd.DataFrame:
    return pd.read_csv(TABDIR / f"T_PS{name}.csv")


def lam(v) -> str:
    return "%g" % float(v)


def num(v, dp: int = 3) -> str:
    return "-" if v is None or pd.isna(v) else f"{float(v):.{dp}f}"


def pval(p: float) -> str:
    """Exact enough to be re-checked, never rounded onto a decision boundary."""
    p = float(p)
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def on_grid(series_by_lam: dict, dp: int = 3) -> list[str]:
    """One cell per reference lambda; a dash where the model never met it."""
    return [num(series_by_lam.get(l), dp) for l in LAMS]


def lam_header() -> list[str]:
    return [f"${lam(l)}$" for l in LAMS]


def models_in(d: pd.DataFrame, arm: str, col: str = "model") -> list[str]:
    present = set(d[d.arm == arm][col])
    return [m for m in MODEL_ORDER if m in present]


def table(caption: str, label: str, colspec: str, header: list[str],
          body: list[str], *, note: str = "", tabcolsep: str = "5pt",
          size: str = "footnotesize") -> str:
    head = " & ".join(header) + " \\\\"
    lines = [
        "\\begin{table}[htbp]", "\\centering", f"\\{size}",
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}", "\\toprule", head,
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}",
    ]
    if note:
        lines += ["", "\\vspace{3pt}", f"{{\\footnotesize {note}}}"]
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


def block_row(arm: str, ncol: int) -> str:
    return (f"\\multicolumn{{{ncol}}}{{@{{}}l}}"
            f"{{\\emph{{{ARM_BLOCK[arm]}}}}}\\\\")


# --------------------------------------------------------------------------
def s_corpus() -> str:
    d = T("01_corpus")
    body = []
    for arm in ARM_ORDER:
        for m in models_in(d, arm):
            r = d[(d.arm == arm) & (d.model == m)].iloc[0]
            body.append(" & ".join([
                ARM_LABEL[arm], MODEL_LABEL.get(r.model, r.model),
                str(int(r.scales)),
                f"{lam(r.scale_min)}--{lam(r.scale_max)}", str(int(r.languages)),
                str(int(r.rounds)), f"{int(r.dyads):,}", f"{int(r.agent_games):,}",
                f"{int(r.decisions):,}", str(int(r.per_cell)), num(r.coop_rate),
            ]) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    return table(
        "\\textbf{The corpus, by model.} ``Cell $n$'' is the number of "
        "agent-games in every model $\\times$ language $\\times$ scale cell; "
        "the design is balanced, with none missing. The two groups are two "
        "sampling densities of one arm and share the base matrix, the horizon "
        "and the decoding settings; they differ in whether the round count was "
        "disclosed in the prompt, so no level statistic is compared across the "
        "rule.",
        "tab:S-corpus", "llccccrrrcc",
        ["Sweep", "Model", "Scales", "Range", "Lang.", "Rnds", "Dyads",
         "Games", "Decis.", "Cell $n$", "Coop."],
        body, tabcolsep="3pt")


def s_notation() -> str:
    d = T("02_notation")
    body = []
    for arm in ARM_ORDER:
        for _, r in d[d.arm == arm].sort_values("lambda").iterrows():
            body.append(" & ".join([
                ARM_LABEL[arm], f"${lam(r['lambda'])}$",
                f"\\texttt{{{r.printed_cells}}}",
                r.regime, str(int(r.is_fractional)), num(r.mean_glyphs, 2),
                str(int(r.max_digits)), num(r.greed, 2), num(r.fear, 2),
                r.dominant_action, r.nash.strip("()"),
            ]) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    return table(
        "\\textbf{What the rescaling changes, and what it cannot.} The cell "
        "values as the prompt template prints them at each scale, with the "
        "notation features derived from that string on the left and the "
        "game-theoretic invariants on the right. Greed, fear, the dominant "
        "action and the equilibrium are constant down every row by "
        "construction; only the printed string moves. The row for $\\lambda = "
        "0.5$ is the one on which the regime read from the printed cells and "
        "the regime read from $\\lambda$ disagree.",
        "tab:S-notation", "llllccccccc",
        ["Sweep", "$\\lambda$", "Printed cells", "Regime", "Frac.",
         "Glyphs", "Digits", "Greed", "Fear", "Dom.", "Nash"],
        body, tabcolsep="3.5pt")


def s_ladder() -> str:
    """Transposed: specifications down the side, models across."""
    d = T("05_magnitude_or_notation")
    d = d[d.scope != "pooled"]
    cols, best = [], {}
    for arm in ARM_ORDER:
        for mdl in models_in(d, arm, "scope"):
            dm = d[d.scope == mdl].set_index("model_spec")
            cols.append((arm, mdl, dm))
            best[mdl] = dm.delta_bic.idxmin()
    body = []
    for spec in SPEC_ORDER:
        cells = []
        for _, mdl, dm in cols:
            if spec not in dm.index:
                cells.append("-")
                continue
            v = num(dm.loc[spec, "delta_bic"], 1)
            cells.append(f"\\textbf{{{v}}}" if spec == best[mdl] else v)
        body.append(" & ".join([SPEC_LABEL[spec]] + cells) + " \\\\")
    header = ["How $\\lambda$ enters"] + [MODEL_SHORT[m] for _, m, _ in cols]
    n_ten = sum(1 for a, _, _ in cols if a == "ten-scale")
    rule = (f"\\cmidrule(lr){{2-{1 + n_ten}}}"
            f"\\cmidrule(lr){{{2 + n_ten}-{1 + len(cols)}}}")
    group = (f" & \\multicolumn{{{n_ten}}}{{c}}{{ten scales}}"
             f" & \\multicolumn{{{len(cols) - n_ten}}}{{c}}{{three scales}}"
             " \\\\\n" + rule)
    winners = ", ".join(f"{MODEL_SHORT[m]}: {SPEC_LABEL[best[m]]}"
                        for _, m, _ in cols)
    return table(
        "\\textbf{The description ladder, model by model.} $\\Delta$BIC from "
        "the best specification for that model, every row fitted to identical "
        "data with language and both personas controlled. Bold marks the best "
        "specification in each column. No notation specification is best for "
        "any of the three models that met ten scales; on three scales the "
        "fractional flag and the regime factor are the same specification, and "
        "a cubic is collinear with a quadratic, so the rungs that appear to "
        "rank are not distinguishable.",
        "tab:S-ladder", "l" + "c" * len(cols),
        [group + "\n" + " & ".join(header)], body, tabcolsep="4pt",
        note="Best description by BIC: " + winners + ".")


def s_persona() -> str:
    d = T("07_persona_by_scale")
    d = d[d.model != "pooled"]
    ncol = 1 + len(LAMS)
    body = []
    for arm in ARM_ORDER:
        body.append(block_row(arm, ncol))
        for mdl in models_in(d, arm):
            dm = d[d.model == mdl].set_index("lambda")
            cells = []
            for l in LAMS:
                if l not in dm.index:
                    cells.append("-")
                    continue
                r = dm.loc[l]
                mark = "^{\\dagger}" if (r.compliant or r.inverted) else ""
                cells.append(f"${r.effect:+.3f}{mark}$")
            body.append(" & ".join([MODEL_LABEL.get(mdl, mdl)] + cells) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    return table(
        "\\textbf{The persona instruction, model by model and scale by "
        "scale.} The effect is the cooperation rate of an agent told to be "
        "cooperative minus that of an agent told to be selfish. A dagger "
        "marks an estimate whose 95\\% dyad bootstrap interval excludes zero. "
        "Read across a row: three of the six models hold one sign over the "
        "whole sweep and three cross zero, which is why the average over "
        "models describes no individual model.",
        "tab:S-persona", "l" + "c" * len(LAMS),
        ["Model"] + lam_header(), body, tabcolsep="2.2pt",
        size="scriptsize")


def s_mixtests() -> str:
    d = T("10b_mix_tests")
    body = []
    for arm in ARM_ORDER:
        sub = d[d.arm == arm]
        order = ["pooled"] + models_in(d, arm)
        for mdl in order:
            rows = sub[sub.model == mdl]
            if rows.empty:
                continue
            r = rows.iloc[0]
            name = "pooled" if mdl == "pooled" else MODEL_LABEL.get(mdl, mdl)
            body.append(" & ".join([
                ARM_LABEL[arm], name, num(r.max_tv), pval(r.p), num(r.null95),
                num(r.max_tv_named), pval(r.p_named), num(r.null95_named),
                num(r.named_share),
            ]) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    return table(
        "\\textbf{Strategy-mix shifts on both vocabularies.} The largest "
        "total-variation distance between the mixes at two payoff scales, "
        "against a null that shuffles the scale label across whole dyads. "
        "``5-cat.'' uses the five categories the read-out emits; ``named'' "
        "drops \\emph{ambiguous} and renormalises over the four canonical "
        "rules, so that a change in how identifiable the play is cannot "
        "contribute. ``Named share'' is the fraction of that model's "
        "agent-games the restriction keeps. Every model is significant on "
        "both vocabularies, and the restriction raises three of the six "
        "estimates rather than lowering them, so none of these shifts is an "
        "artefact of play becoming more or less nameable.",
        "tab:S-mixtests", "llcccccccc",
        ["Sweep", "Model", "5-cat.", "$P$", "null", "named", "$P$", "null",
         "Named share"],
        body, tabcolsep="4.5pt")


def s_mixmodel() -> str:
    d = T("10c_mix_by_model")
    ncol = 2 + len(LAMS)
    body = []
    for arm in ARM_ORDER:
        body.append(block_row(arm, ncol))
        for mdl in models_in(d, arm):
            dm = d[d.model == mdl]
            for i, strat in enumerate(STRAT_ORDER):
                ds = dm[dm.archetype == strat].set_index("scale_nominal")
                have = set(dm.scale_nominal)
                cells = [num(ds.share.get(l, 0.0)) if l in have else "-"
                         for l in LAMS]
                name = MODEL_LABEL.get(mdl, mdl) if i == 0 else ""
                body.append(" & ".join([name, strat] + cells) + " \\\\")
            body.append("\\addlinespace")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    return table(
        "\\textbf{The strategy mix, model by model.} Share of that model's "
        "agent-games carrying each label at each payoff scale. A dash marks a "
        "scale the model never met.",
        "tab:S-mixmodel", "ll" + "c" * len(LAMS),
        ["Model", "Label"] + lam_header(), body, tabcolsep="2.2pt")


def s_source() -> str:
    d = T("10d_readout_source")
    body = []
    for arm in ARM_ORDER:
        sub = d[d.arm == arm].copy()
        scales = sorted(float(s) for s in sub.scale_nominal.unique()
                        if s != "all")
        for s in [lam(x) for x in scales] + ["all"]:
            row = sub[sub.scale_nominal.astype(str).str.rstrip("0").str.rstrip(".")
                      .eq(s) | sub.scale_nominal.eq(s)]
            row = row.set_index("source")
            label = "all scales" if s == "all" else f"$\\lambda={s}$"
            cells = [num(row.share.get(k, 0.0)) for k in SOURCE_ORDER]
            body.append(" & ".join([ARM_LABEL[arm], label] + cells) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    return table(
        "\\textbf{Where each strategy label came from.} Share of agent-games "
        "resolved at each stage of the read-out. ``Provable rule'' means "
        "exactly one canonical rule reproduces the run of play; ``several "
        "rules'' means more than one does and the run is recorded as "
        "ambiguous; the last two are the LSTM's nearest rule, split at the "
        "0.90 posterior floor below which a read-out designed to abstain "
        "would decline to answer. None of the four shares is constant across "
        "the sweep, so how nameable the play is depends on the payoff scale.",
        "tab:S-source", "ll" + "c" * len(SOURCE_ORDER),
        ["Sweep", "Scale"] + [SOURCE_LABEL[k] for k in SOURCE_ORDER], body)


def s_language() -> str:
    d = T("08_language_cells")
    ncol = 2 + len(LAMS)
    body = []
    for arm in ARM_ORDER:
        sub = d[d.arm == arm]
        have = set(sub.scale_nominal)
        body.append(block_row(arm, ncol))
        for lg in ["en", "fr", "vn", "cn", "ar"]:
            dm = sub[sub.language == lg].set_index("scale_nominal")
            cells = [num(dm.coop_rate.get(l)) if l in have else "-" for l in LAMS]
            framing = "minimise" if lg in ("en", "fr", "vn") else "maximise"
            body.append(" & ".join([LANG_LABEL[lg], framing] + cells) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    return table(
        "\\textbf{Cooperation by prompt language and payoff scale.} The "
        "FAIRGAME templates do not state the objective identically in every "
        "language, so language is confounded with objective framing and the "
        "framing each template uses is given alongside it. The five languages "
        "do not move in parallel across the sweep.",
        "tab:S-language", "ll" + "c" * len(LAMS),
        ["Language", "Framing"] + lam_header(), body, tabcolsep="2.2pt")


def s_decade() -> str:
    d = T("04c_one_decade")
    body = []
    for arm in ARM_ORDER:
        for mdl in models_in(d, arm):
            for _, r in d[d.model == mdl].iterrows():
                body.append(" & ".join([
                    ARM_LABEL[arm], MODEL_LABEL.get(r.model, r.model),
                    f"${lam(r.lambda_lo)}$ vs ${lam(r.lambda_hi)}$",
                    num(r.gap), num(r.sweep_null95),
                    "yes" if int(r.clears_null) else "no",
                ]) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    return table(
        "\\textbf{What one decade of separation buys.} The cooperation gap "
        "between two scales a decade apart, plus the off-decade pair "
        "$\\lambda = 0.5$ against $\\lambda = 5$, held against the 95th "
        "percentile of the permutation null for that model's \\emph{full} "
        "sweep, which is conservative because a two-level design generates a "
        "smaller null. This is the evidence behind the reporting standard, and "
        "behind the qualification it carries: two conditions a decade apart "
        "detect the effect, but which decade is not a free choice.",
        "tab:S-decade", "lllccc",
        ["Sweep", "Model", "Contrast", "Gap", "Null (full sweep)", "Clears"],
        body, tabcolsep="4.5pt")


def s_blocks() -> str:
    d = T("06_round_blocks")
    blocks = (d[d.arm == "ten-scale"].sort_values("block_order")
              .block.unique().tolist())
    body = []
    for arm in ARM_ORDER:
        sub = d[d.arm == arm]
        for rg in ["fractional", "unit", "large"]:
            dm = sub[sub.regime == rg].set_index("block")
            if dm.empty:
                continue
            cells = [num(dm.coop.get(b)) for b in blocks]
            body.append(" & ".join([ARM_LABEL[arm], rg] + cells) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    return table(
        "\\textbf{Cooperation across the game, by notation regime.} Round one "
        "on its own, then three equal blocks of the remaining rounds. On the "
        "three-scale grid the gap between the regimes is open on round one, on "
        "which no history exists, and stays open; on the ten-scale grid, where "
        "the regime and the magnitude no longer coincide, the three regimes "
        "are within a point of each other at every position.",
        "tab:S-blocks", "ll" + "c" * len(blocks),
        ["Sweep", "Regime"] + [f"\\texttt{{{b}}}" for b in blocks], body)


BUILDERS = [s_corpus, s_notation, s_ladder, s_persona, s_mixtests,
            s_mixmodel, s_source, s_language, s_decade, s_blocks]


def main() -> None:
    parts = ["% Generated by Analysis/scripts/42_supp_tables.py -- do not edit.",
             "% Every number is copied from tables/T_PS*.csv.", ""]
    for build in BUILDERS:
        parts.append(build())
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"{len(BUILDERS)} supplementary tables written to {OUT}")


if __name__ == "__main__":
    main()
