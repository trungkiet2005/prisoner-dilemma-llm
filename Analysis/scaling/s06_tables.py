"""Emit the manuscript tables as LaTeX, so no number is ever retyped by hand.

Writes paper_scaling/tables_auto.tex, which main.tex \\input's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import MODEL_LABEL, MODEL_ORDER      # noqa: E402

HERE = Path(__file__).resolve().parent
TAB = HERE / "tables"
OUT = HERE.parents[1] / "paper_scaling" / "tables_auto.tex"


def fmt_p(p):
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def table_effect(t03, t09):
    t03 = t03.set_index("model")
    t09 = t09.set_index("model")
    rows = []
    for m in MODEL_ORDER:
        a, b = t03.loc[m], t09.loc[m]
        rows.append(
            f"{MODEL_LABEL[m]} & {a['coop_min']:.3f} & {a['coop_max']:.3f} & "
            f"{a['range']:.3f} & [{a['lo']:.3f},\\,{a['hi']:.3f}] & "
            f"{fmt_p(a['p_perm'])} & {b['wald_chi2_scale']:.1f} & {fmt_p(b['p_scale'])} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{\\textbf{{Cooperation moves with the payoff scale in every model.}}
Minimum and maximum cooperation rate over the ten scales, their range, a
95\\% confidence interval on that range from a bootstrap that resamples whole
dyads, and a permutation test that shuffles the scale label within language and
persona (2{{,}}000 draws, so $0.0005$ is the smallest attainable value). The last
two columns give a Wald test on the nine scale contrasts of a per-model logistic
regression of the round-level decision, with standard errors clustered on the
dyad ($n=40{{,}}000$ decisions and $2{{,}}000$ clusters per model).}}
\\label{{tab:effect}}
\\small
\\begin{{tabular}}{{lccccccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{cooperation}} & & & & \\multicolumn{{2}}{{c}}{{Wald test}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{7-8}}
Model & min & max & range & 95\\% CI & $p_{{\\text{{perm}}}}$ & $\\chi^2_9$ & $p$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_persona(t07):
    t07 = t07.set_index("model")
    rows = []
    for m in MODEL_ORDER:
        r = t07.loc[m]
        flip = "yes" if r["sign_flip"] else "no"
        rows.append(
            f"{MODEL_LABEL[m]} & ${r['persona_effect_subunit']:+.3f}$ & "
            f"${r['persona_effect_suprunit']:+.3f}$ & {r['shift']:.3f} & "
            f"[{r['shift_lo']:.3f},\\,{r['shift_hi']:.3f}] & {flip} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{\\textbf{{The payoff scale gates the persona instruction.}}
The persona effect is the cooperation rate of an agent told it is cooperative
minus that of an agent told it is selfish. It is reported separately for the two
scales at which every payoff printed is at most 1 ($\\lam \\leq 0.1$) and for the
eight at which some payoff exceeds it. The shift between the two regimes carries
a 95\\% confidence interval from the dyad bootstrap; the last column records
whether the persona effect changes sign.}}
\\label{{tab:persona}}
\\small
\\begin{{tabular}}{{lccccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{persona effect}} & & & \\\\
\\cmidrule(lr){{2-3}}
Model & $\\lam \\leq 0.1$ & $\\lam \\geq 0.25$ & shift & 95\\% CI & sign flip \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_strategy(t10, t11):
    t10 = t10.set_index("model")
    t11 = t11.set_index("model")
    rows = []
    for m in MODEL_ORDER:
        a, b = t10.loc[m], t11.loc[m]
        rows.append(
            f"{MODEL_LABEL[m]} & {a['deduced_subunit']:.1f} & "
            f"{a['deduced_suprunit']:.1f} & ${a['beta_logscale']:+.3f}$ & "
            f"{fmt_p(a['p'])} & {b['dist_at_min_scale']:.2f} & "
            f"{b['dist_at_max_scale']:.2f} & ${b['beta_logscale']:+.3f}$ & "
            f"{fmt_p(b['p'])} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{\\textbf{{Larger payoffs make play less rule-governed in three models of
five.}} The left block gives the share of agent-games matched exactly by one of
the four canonical rules, below and above the sub-unit boundary, and the
coefficient of a logistic regression of that indicator on $\\log_{{10}}\\lam$. The
right block gives the mean number of rounds that violate the nearest canonical
rule at the smallest and largest scale, and the same trend fitted by least
squares. Both use standard errors clustered on the dyad. Neither quantity
depends on the learned classifier.}}
\\label{{tab:strategy}}
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lcccccccc}}
\\toprule
& \\multicolumn{{4}}{{c}}{{exactly one rule fits (\\%)}}
& \\multicolumn{{4}}{{c}}{{rounds violating the nearest rule}} \\\\
\\cmidrule(lr){{2-5}} \\cmidrule(lr){{6-9}}
Model & $\\lam \\leq 0.1$ & $\\lam \\geq 0.25$ & $\\beta$ & $p$
      & $\\lam{{=}}0.01$ & $\\lam{{=}}10^3$ & $\\beta$ & $p$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main():
    t03 = pd.read_csv(TAB / "T03_effect_size.csv")
    t07 = pd.read_csv(TAB / "T07_persona_gating.csv")
    t09 = pd.read_csv(TAB / "T09_regression.csv")
    t10 = pd.read_csv(TAB / "T10_provenance_trend.csv")
    t11 = pd.read_csv(TAB / "T11_rule_distance.csv")

    text = ("% Generated by Analysis/scaling/s06_tables.py - do not edit by hand.\n"
            + table_effect(t03, t09) + "\n"
            + table_persona(t07) + "\n"
            + table_strategy(t10, t11))
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
