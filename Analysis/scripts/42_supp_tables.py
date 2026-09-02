"""Electronic supplementary material tables for the payoff-scaling manuscript.

Reads ``tables/T_PS*.csv`` and writes ``paper_scaling/supp_tables.tex``, a
fragment that ``paper_scaling/supplementary.tex`` inputs.  Nothing here
computes a statistic: every number is copied from the table that
``40_scaling_stats.py`` wrote, for the same reason the figure script computes
nothing of its own.  A supplementary table therefore cannot drift from the
main text, and re-running the pipeline updates both.

    python Analysis/scripts/42_supp_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pdlib.style import TABDIR

OUT = Path(__file__).resolve().parents[2] / "paper_scaling" / "supp_tables.tex"

ARM_ORDER = ["open-weight", "frontier"]
MODEL_LABEL = {
    "Gemma-3-12B": "Gemma 3 12B", "Llama-3.1-8B": "Llama 3.1 8B",
    "Qwen3-8B": "Qwen3 8B", "Claude-3.5-Haiku": "Claude 3.5 Haiku",
    "Gemini-3.5-Flash-Lite": "Gemini 3.5 Flash-Lite", "GPT-4o": "GPT-4o",
    "Mistral-Large": "Mistral Large",
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
    return "-" if pd.isna(v) else f"{float(v):.{dp}f}"


def pval(p: float) -> str:
    """Exact enough to be re-checked, never rounded onto a decision boundary."""
    p = float(p)
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def table(caption: str, label: str, colspec: str, header: list[str],
          body: list[str], *, note: str = "", tabcolsep: str = "5pt") -> str:
    head = " & ".join(header) + " \\\\"
    lines = [
        "\\begin{table}[htbp]", "\\centering", "\\footnotesize",
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}", "\\toprule", head,
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}",
    ]
    if note:
        lines += ["", "\\vspace{3pt}", f"{{\\footnotesize {note}}}"]
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
def s_corpus() -> str:
    d = T("01_corpus")
    body = []
    for arm in ARM_ORDER:
        for _, r in d[d.arm == arm].iterrows():
            body.append(" & ".join([
                arm, MODEL_LABEL.get(r.model, r.model), str(int(r.scales)),
                f"{lam(r.scale_min)}--{lam(r.scale_max)}", str(int(r.languages)),
                str(int(r.rounds)), f"{int(r.dyads):,}", f"{int(r.agent_games):,}",
                f"{int(r.decisions):,}", str(int(r.per_cell)), num(r.coop_rate),
            ]) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    return table(
        "\\textbf{The corpus, by model.} ``Cell $n$'' is the number of "
        "agent-games in every model $\\times$ language $\\times$ scale cell; "
        "the design is balanced, with none missing. The two arms are never "
        "pooled.",
        "tab:S-corpus", "llccccrrrcc",
        ["Arm", "Model", "Scales", "Range", "Lang.", "Rnds", "Dyads",
         "Games", "Decis.", "Cell $n$", "Coop."],
        body, tabcolsep="3pt")


def s_notation() -> str:
    d = T("02_notation")
    body = []
    for arm in ARM_ORDER:
        for _, r in d[d.arm == arm].iterrows():
            body.append(" & ".join([
                arm, f"${lam(r['lambda'])}$", f"\\texttt{{{r.printed_cells}}}",
                r.regime, str(int(r.is_fractional)), num(r.mean_glyphs, 2),
                str(int(r.max_digits)), num(r.greed, 2), num(r.fear, 2),
                r.dominant_action, f"({r.nash.strip('()')})".replace("(", "").replace(")", ""),
            ]) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    return table(
        "\\textbf{What the rescaling changes, and what it cannot.} The cell "
        "values as the prompt template prints them at each scale, with the "
        "notation features derived from that string on the left and the "
        "game-theoretic invariants on the right. Greed, fear, the dominant "
        "action and the equilibrium are constant down each arm by "
        "construction; only the printed string moves.",
        "tab:S-notation", "llllccccccc",
        ["Arm", "$\\lambda$", "Printed cells", "Regime", "Frac.",
         "Glyphs", "Digits", "Greed", "Fear", "Dom.", "Nash"],
        body, tabcolsep="3.5pt")


def s_ladder() -> str:
    """Transposed: specifications down the side, models across.

    The specification names are the long strings, so they belong in the stub
    column; eight of them across the top does not fit the text block.
    """
    d = T("05_magnitude_or_notation")
    d = d[d.scope != "pooled"]
    cols, best = [], {}
    for arm in ARM_ORDER:
        for mdl in sorted(d[d.arm == arm].scope.unique()):
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
    short = {"Gemma 3 12B": "Gemma", "Llama 3.1 8B": "Llama",
             "Qwen3 8B": "Qwen3", "Claude 3.5 Haiku": "Claude",
             "Gemini 3.5 Flash-Lite": "Gemini", "GPT-4o": "GPT-4o",
             "Mistral Large": "Mistral"}
    header = ["How $\\lambda$ enters"] + [
        short[MODEL_LABEL.get(m, m)] for _, m, _ in cols]
    n_ow = sum(1 for a, _, _ in cols if a == "open-weight")
    rule = (f"\\cmidrule(lr){{2-{1 + n_ow}}}"
            f"\\cmidrule(lr){{{2 + n_ow}-{1 + len(cols)}}}")
    group = (f" & \\multicolumn{{{n_ow}}}{{c}}{{open-weight, 6 scales}}"
             f" & \\multicolumn{{{len(cols) - n_ow}}}{{c}}{{frontier, 3 scales}}"
             " \\\\\n" + rule)
    return table(
        "\\textbf{The description ladder, model by model.} $\\Delta$BIC from "
        "the best specification for that model, every row fitted to identical "
        "data with language and both personas controlled. Bold marks the best "
        "specification in each column. Five of the seven models prefer a "
        "description built from how the cells are printed; Gemma 3 12B "
        "prefers a cubic in $\\log_{10}\\lambda$, the one case in the corpus "
        "where the notation account loses to a polynomial, and Mistral Large "
        "prefers no $\\lambda$ term at all, which is the correct description "
        "of a model whose cooperation rate did not move. Over three scales a "
        "cubic is collinear with a quadratic and the fractional flag is the "
        "same specification as the regime factor.",
        "tab:S-ladder", "l" + "c" * len(cols),
        [group + "\n" + " & ".join(header)], body, tabcolsep="4pt")


def s_persona() -> str:
    d = T("07_persona_by_scale")
    d = d[d.model != "pooled"]
    body = []
    for arm in ARM_ORDER:
        sub = d[d.arm == arm]
        lams = sorted(sub["lambda"].unique())
        for mdl in sorted(sub.model.unique()):
            dm = sub[sub.model == mdl].set_index("lambda")
            cells = []
            for l in lams:
                r = dm.loc[l]
                mark = "^{\\dagger}" if (r.compliant or r.inverted) else ""
                cells.append(f"${r.effect:+.3f}{mark}$")
            body.append(" & ".join([arm, MODEL_LABEL.get(mdl, mdl)] + cells) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    lams_ow = sorted(d[d.arm == "open-weight"]["lambda"].unique())
    header = ["Arm", "Model"] + [f"$\\lambda={lam(l)}$" for l in lams_ow]
    # the frontier rows carry three scales; pad them into the matching columns
    fixed = []
    for line in body:
        parts = line.replace(" \\\\", "").split(" & ")
        if parts[0] == "frontier" and len(parts) == 5:
            parts = parts[:2] + ["-"] + parts[2:] + ["-", "-"]
        fixed.append(" & ".join(parts) + " \\\\")
    return table(
        "\\textbf{The persona instruction, model by model and scale by "
        "scale.} The effect is the cooperation rate of an agent told to be "
        "cooperative minus that of an agent told to be selfish. A dagger "
        "marks an estimate whose 95\\% dyad bootstrap interval excludes zero. "
        "Read across a row: five of the seven models hold one sign over the "
        "whole sweep, which is why the average over models, which does cross "
        "zero, describes no individual model.",
        "tab:S-persona", "ll" + "c" * len(lams_ow), header, fixed,
        tabcolsep="4pt")


def s_mixtests() -> str:
    d = T("10b_mix_tests")
    body = []
    for arm in ARM_ORDER:
        sub = d[d.arm == arm]
        for _, r in sub.iterrows():
            name = ("pooled" if r.model == "pooled"
                    else MODEL_LABEL.get(r.model, r.model))
            body.append(" & ".join([
                arm, name, num(r.max_tv), pval(r.p), num(r.null95),
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
        "both vocabularies. Gemma 3 12B is the one whose estimate changes "
        "materially, because almost half of its five-category distance is "
        "movement into the ambiguous bucket.",
        "tab:S-mixtests", "llcccccccc",
        ["Arm", "Model", "5-cat.", "$P$", "null", "named", "$P$", "null",
         "Named share"],
        body, tabcolsep="4.5pt")


def s_mixmodel() -> str:
    d = T("10c_mix_by_model")
    body = []
    for arm in ARM_ORDER:
        sub = d[d.arm == arm]
        lams = sorted(sub.scale_nominal.unique())
        for mdl in sorted(sub.model.unique()):
            for i, strat in enumerate(STRAT_ORDER):
                cells = []
                for l in lams:
                    m = sub[(sub.model == mdl) & (sub.scale_nominal == l)
                            & (sub.archetype == strat)]
                    cells.append(num(m.share.iloc[0]) if len(m) else "0.000")
                if arm == "frontier":
                    cells = ["-"] + cells + ["-", "-"]
                name = MODEL_LABEL.get(mdl, mdl) if i == 0 else ""
                body.append(" & ".join([name, strat] + cells) + " \\\\")
            body.append("\\addlinespace")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    lams_ow = sorted(d[d.arm == "open-weight"].scale_nominal.unique())
    return table(
        "\\textbf{The strategy mix, model by model.} Share of that model's "
        "agent-games carrying each label at each payoff scale. The frontier "
        "arm sweeps only $\\lambda \\in \\{0.1, 1, 10\\}$.",
        "tab:S-mixmodel", "ll" + "c" * len(lams_ow),
        ["Model", "Label"] + [f"$\\lambda={lam(l)}$" for l in lams_ow],
        body, tabcolsep="4pt")


def s_source() -> str:
    d = T("10d_readout_source")
    body = []
    for arm in ARM_ORDER:
        sub = d[d.arm == arm]
        scales = [s for s in sub.scale_nominal.unique() if s != "all"] + ["all"]
        for s in scales:
            row = sub[sub.scale_nominal == s].set_index("source")
            label = "all scales" if s == "all" else f"$\\lambda={lam(s)}$"
            cells = [num(row.share.get(k, 0.0)) for k in SOURCE_ORDER]
            body.append(" & ".join([arm, label] + cells) + " \\\\")
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
        ["Arm", "Scale"] + [SOURCE_LABEL[k] for k in SOURCE_ORDER], body)


def s_language() -> str:
    d = T("08_language_cells")
    body = []
    for arm in ARM_ORDER:
        sub = d[d.arm == arm]
        lams = sorted(sub.scale_nominal.unique())
        for lg in ["en", "fr", "vn", "cn", "ar"]:
            dm = sub[sub.language == lg].set_index("scale_nominal")
            cells = [num(dm.coop_rate.get(l)) for l in lams]
            if arm == "frontier":
                cells = ["-"] + cells + ["-", "-"]
            framing = "minimise" if lg in ("en", "fr", "vn") else "maximise"
            body.append(" & ".join([arm, LANG_LABEL[lg], framing] + cells) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    lams_ow = sorted(d[d.arm == "open-weight"].scale_nominal.unique())
    return table(
        "\\textbf{Cooperation by prompt language and payoff scale.} The "
        "FAIRGAME templates do not state the objective identically in every "
        "language, so language is confounded with objective framing and the "
        "framing each template uses is given alongside it. The five languages "
        "do not move in parallel across the sweep.",
        "tab:S-language", "lll" + "c" * len(lams_ow),
        ["Arm", "Language", "Framing"] + [f"$\\lambda={lam(l)}$" for l in lams_ow],
        body, tabcolsep="4.5pt")


def s_decade() -> str:
    d = T("04c_one_decade")
    body = []
    for arm in ARM_ORDER:
        for _, r in d[d.arm == arm].iterrows():
            body.append(" & ".join([
                arm, MODEL_LABEL.get(r.model, r.model),
                f"${lam(r.lambda_lo)}$ vs ${lam(r.lambda_hi)}$",
                num(r.gap), num(r.sweep_null95),
                "yes" if int(r.clears_null) else "no",
            ]) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    return table(
        "\\textbf{What one decade of separation buys.} The cooperation gap "
        "between two adjacent scales, held against the 95th percentile of the "
        "permutation null for that model's \\emph{full} sweep, which is "
        "conservative because a two-level design generates a smaller null. "
        "This is the evidence behind the reporting standard: a single extra "
        "condition one decade away detects the effect in six of the seven "
        "models.",
        "tab:S-decade", "lllccc",
        ["Arm", "Model", "Contrast", "Gap", "Null (full sweep)", "Clears"],
        body)


def s_blocks() -> str:
    d = T("06_round_blocks")
    body = []
    for arm in ARM_ORDER:
        sub = d[d.arm == arm]
        blocks = (sub.sort_values("block_order").block.unique().tolist())
        for rg in ["fractional", "unit", "large"]:
            dm = sub[sub.regime == rg].set_index("block")
            if dm.empty:
                continue
            cells = [num(dm.coop.get(b)) for b in blocks]
            body.append(" & ".join([arm, rg] + cells) + " \\\\")
        if arm != ARM_ORDER[-1]:
            body.append("\\midrule")
    blocks_ow = (d[d.arm == "open-weight"].sort_values("block_order")
                 .block.unique().tolist())
    return table(
        "\\textbf{Cooperation across the game, by notation regime.} Round one "
        "on its own, then three equal blocks of the remaining rounds. The gap "
        "between the regimes is open on round one, on which no history "
        "exists. The two arms divide different horizons into the same four "
        "positions and are not comparable to each other.",
        "tab:S-blocks", "ll" + "c" * len(blocks_ow),
        ["Arm", "Regime"] + [f"\\texttt{{{b}}}" for b in blocks_ow], body)


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
