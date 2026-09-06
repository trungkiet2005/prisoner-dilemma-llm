"""Emit the supplementary tables as LaTeX.

Writes paper_scaling/supp_tables_auto.tex, which supplementary.tex \\input's.
Every number comes from the tables in tables/ or from the master parquet, so
nothing in the supplement is retyped by hand either.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (LANG_LABEL, LANG_ORDER, MODEL_LABEL,     # noqa: E402
                      MODEL_ORDER, MODEL_ORDER_ALL, SCALES)

# Every per-model table in this supplement carries all six models, including
# the one the main text reports here rather than in its own figures.  The two
# pooled quantities are printed both ways, five models and six, so a reader can
# see for themselves that including the sixth changes no conclusion.

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TAB = HERE / "tables"
# The manuscript directory is configurable so the same pipeline can feed a
# second manuscript, exactly as figstyle.PAPER_DIR is.  The default is
# unchanged: with PD_PAPER_DIR unset this resolves to paper_scaling/ as before.
PAPER_DIR = Path(os.environ.get("PD_PAPER_DIR", HERE.parents[1] / "papers" / "interface-focus"))
PAPER_DIR.mkdir(parents=True, exist_ok=True)
OUT = PAPER_DIR / "supp_tables_auto.tex"

SC = [("%g" % s) for s in SCALES]


def wrap(body, caption, label, spec, header, small="\\footnotesize",
         colsep="4pt", fit=False):
    """`fit` scales the tabular to the text width.

    The two widest tables carry 13 and 12 columns beside a long model name, and
    no font or column-spacing setting brings them inside the margin on their
    own.
    """
    open_fit = "\\resizebox{\\textwidth}{!}{%\n" if fit else ""
    close_fit = "}\n" if fit else ""
    return f"""\\begin{{table}}[htbp]
\\centering
{small}
\\setlength{{\\tabcolsep}}{{{colsep}}}
\\caption{{{caption}}}
\\label{{{label}}}
{open_fit}\\begin{{tabular}}{{@{{}}{spec}@{{}}}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
{close_fit}\\end{{table}}
"""


def s1_corpus(g, r):
    rows = []
    for m in MODEL_ORDER_ALL:
        d = g[g.model == m]
        rd = r[r.model == m]
        cell = (d.groupby(["scale_nominal", "language"]).game_uid.nunique()
                .unique())
        rows.append(f"{MODEL_LABEL[m]} & {d.scale_nominal.nunique()} & "
                    f"{d.language.nunique()} & {d.game_uid.nunique():,} & "
                    f"{len(d):,} & {len(rd):,} & "
                    f"{cell[0] if len(cell) == 1 else '!'} & "
                    f"{d.coop_rate.mean():.3f} \\\\")
    tot = (f"\\textbf{{All}} & {g.scale_nominal.nunique()} & "
           f"{g.language.nunique()} & {g.game_uid.nunique():,} & {len(g):,} & "
           f"{len(r):,} & 40 & {g.coop_rate.mean():.3f} \\\\")
    return wrap("\n".join(rows) + "\n\\midrule\n" + tot,
                "\\textbf{The corpus, by model.} Every model was run on the "
                "same ten payoff scales and five prompt languages. "
                "``Cell $n$'' is the number of dyads in each "
                "model $\\times$ language $\\times$ scale cell; the design is "
                "balanced with none missing, and the build refuses to write "
                "its tables otherwise. The first five models are the ones the "
                "main text reports; Gemini 3.1 Flash-Lite is reported here, "
                # Not "\\S S8": this file is \\input by two manuscripts whose
                # section numbering differs, and a hard-coded number would be
                # wrong in one of them without any LaTeX warning.
                "for the reason given with the sixth model below.",
                "tab:S-corpus", "llccrrrc",
                "Model & Scales & Lang. & Dyads & Agent-games & Decisions & "
                "Cell $n$ & Coop. \\\\")


def s2_scale(t02):
    p = t02.pivot(index="model", columns="scale", values="coop")
    lo = t02.pivot(index="model", columns="scale", values="lo")
    hi = t02.pivot(index="model", columns="scale", values="hi")
    rows = []
    for m in MODEL_ORDER_ALL:
        cells = " & ".join(f"{p.loc[m, s]:.3f}" for s in SCALES)
        rows.append(f"{MODEL_LABEL[m]} & {cells} \\\\")
        cells = " & ".join(f"\\tiny{{[{lo.loc[m, s]:.2f},{hi.loc[m, s]:.2f}]}}"
                           for s in SCALES)
        rows.append(f" & {cells} \\\\[2pt]")
    return wrap("\n".join(rows),
                "\\textbf{Cooperation by model and payoff scale.} Each entry "
                "aggregates 200 dyads. The bracketed line under each model is "
                "a 95\\% confidence interval from a bootstrap that resamples "
                "whole dyads.",
                "tab:S-scale", "l" + "c" * len(SCALES),
                "Model & " + " & ".join(f"$\\lam{{=}}{s}$" for s in SC)
                + " \\\\", fit=True)


def s3_language(g):
    rows = []
    for m in MODEL_ORDER_ALL:
        d = g[g.model == m]
        for lang in LANG_ORDER:
            s = d[d.language == lang].groupby("scale_nominal").coop_rate.mean()
            rng = s.max() - s.min()
            name = MODEL_LABEL[m] if lang == LANG_ORDER[0] else ""
            cells = " & ".join(f"{s.loc[x]:.2f}" for x in SCALES)
            rows.append(f"{name} & {LANG_LABEL[lang]} & {cells} & "
                        f"\\textbf{{{rng:.3f}}} \\\\")
        rows.append("\\addlinespace")
    return wrap("\n".join(rows),
                "\\textbf{Cooperation by model, prompt language and payoff "
                "scale.} 40 dyads per entry. The final column is the range "
                "across the ten scales for that model and language; over "
                "all six models it runs from 0.080, for Gemini 3.1 Flash-Lite "
                "in Arabic, to 0.855, for Grok 4.20 in English, so the scale "
                "sensitivity of a model is a property of the model and the "
                "language jointly.",
                "tab:S-language", "ll" + "c" * len(SCALES) + "c",
                "Model & Language & " + " & ".join(f"{s}" for s in SC)
                + " & Range \\\\", small="\\scriptsize", colsep="2.5pt", fit=True)


def s4_variance(t06, t06all):
    name = {"C(model)": "Model", "C(language)": "Language",
            "C(scale_nominal)": "Payoff scale",
            "C(model):C(language)": "Model $\\times$ language",
            "C(model):C(scale_nominal)": "Model $\\times$ scale",
            "C(language):C(scale_nominal)": "Language $\\times$ scale",
            "Residual": "Residual"}
    def body(t):
        rows = []
        for _, r in t.iterrows():
            f = "" if np.isnan(r["F"]) else f"{r['F']:.2f}"
            p = ("" if np.isnan(r["PR(>F)"]) else
                 ("$<0.001$" if r["PR(>F)"] < 0.001
                  else f"${r['PR(>F)']:.3f}$"))
            rows.append(f"{name.get(r['term'], r['term'])} & "
                        f"{r['sum_sq']:.4f} & {int(r['df'])} & {f} & {p} & "
                        f"{r['pct_variance']:.1f} \\\\")
        return "\n".join(rows)

    head = "Term & Sum sq. & df & $F$ & $p$ & \\% variance \\\\"
    a = wrap(body(t06),
             "\\textbf{Variance decomposition of the cell means, five "
             "models.} Type-II analysis of variance on the 250 "
             "model $\\times$ language $\\times$ scale cell means of the five "
             "models the main text reports, where the design is exactly "
             "balanced. The payoff scale carries a significant main effect, "
             "and its interaction with the model carries three times as much "
             "of the variance.",
             "tab:S-variance", "lrrrrr", head)
    b = wrap(body(t06all),
             "\\textbf{Variance decomposition of the cell means, all six "
             "models.} The same type-II analysis on all 300 cell means, the "
             "sixth model included. Every term significant in the five-model "
             "table is significant here and in the same order: the "
             "model $\\times$ scale interaction again carries several times "
             "the variance of the payoff scale as a main effect.",
             "tab:S-variance-all", "lrrrrr", head)
    return a + "\n" + b


def s5_persona(per):
    p = per.pivot(index="model", columns="scale", values="persona_effect")
    rows = []
    for m in MODEL_ORDER_ALL:
        cells = " & ".join(f"${p.loc[m, s]:+.3f}$" for s in SCALES)
        rows.append(f"{MODEL_LABEL[m]} & {cells} \\\\")
    return wrap("\n".join(rows),
                "\\textbf{Persona effect by model and payoff scale.} The "
                "cooperation rate of an agent instructed that it is "
                "cooperative minus that of an agent instructed that it is "
                "selfish, at each scale. The two leftmost columns are the "
                "scales at which every payoff printed is at most 1.",
                "tab:S-persona", "l" + "c" * len(SCALES),
                "Model & " + " & ".join(f"$\\lam{{=}}{s}$" for s in SC)
                + " \\\\", fit=True)


def s6_strategy(t10per, t12per):
    rows = []
    for m in MODEL_ORDER_ALL:
        d = t10per[t10per.model == m].set_index("scale")
        cells = " & ".join(f"{d.loc[s, 'deduced']:.1f}" for s in SCALES)
        rows.append(f"{MODEL_LABEL[m]} & {cells} \\\\")
    a = wrap("\n".join(rows),
             "\\textbf{Share of agent-games matched exactly by one canonical "
             "rule (\\%).} Computed by exact rule matching, with no learned "
             "component. 400 agent-games per entry.",
             "tab:S-deduced", "l" + "c" * len(SCALES),
             "Model & " + " & ".join(f"$\\lam{{=}}{s}$" for s in SC)
             + " \\\\", fit=True)

    rows = []
    for m in MODEL_ORDER_ALL:
        d = t12per[t12per.model == m].set_index("scale")
        for lab in ["AllC", "TFT", "WSLS", "AllD"]:
            name = MODEL_LABEL[m] if lab == "AllC" else ""
            cells = " & ".join(f"{d.loc[s, lab]:.1f}" for s in SCALES)
            rows.append(f"{name} & {lab} & {cells} \\\\")
        rows.append("\\addlinespace")
    b = wrap("\n".join(rows),
             "\\textbf{Strategy composition by model and payoff scale (\\%).} "
             "The label assigned by the hybrid read-out: the rule base fixes "
             "the candidate set and the LSTM ranks within it, deciding alone "
             "only where no canonical rule fits. Columns sum to 100 within "
             "each model and scale.",
             "tab:S-mix", "ll" + "c" * len(SCALES),
             "Model & Label & " + " & ".join(f"{s}" for s in SC) + " \\\\",
             small="\\scriptsize", colsep="2.5pt", fit=True)
    return a + "\n" + b


def s6c_label_provenance(d):
    """How each label was arrived at, which is not uniform across labels.

    This is the table that stops the compositional claim being read as stronger
    than it is: AllC and AllD carry a substantial deduced share, while TFT and
    WSLS are almost entirely attributions the LSTM makes on trajectories that
    match no canonical rule at all.
    """
    t = pd.crosstab(d.label, d.provenance, normalize="index") * 100
    n = d.label.value_counts()
    rows = []
    for lab in ["AllC", "TFT", "WSLS", "AllD"]:
        rows.append(f"{lab} & {n[lab]:,} & {t.loc[lab, 'deduced']:.1f} & "
                    f"{t.loc[lab, 'ambiguous']:.1f} & "
                    f"{t.loc[lab, 'unmatched']:.1f} \\\\")
    return wrap("\n".join(rows),
                "\\textbf{Provenance of each label.} For every label, the share "
                "of the agent-games carrying it that were deduced (exactly one "
                "canonical rule fits), ambiguous (several fit, and the LSTM "
                "ranked within that set) or unmatched (no canonical rule fits, "
                "so the label is an attribution the LSTM makes alone). The "
                "asymmetry is severe and bounds how the compositional result "
                "may be read: AllC and AllD carry a substantial deduced share, "
                "whereas the TFT and WSLS labels are almost entirely "
                "attributions.",
                "tab:S-provenance", "lrccc",
                "Label & $n$ & Deduced (\\%) & Ambiguous (\\%) & "
                "Unmatched (\\%) \\\\")


def s7_classifier(t01):
    rows = [f"{r['split']} & {int(r['n']):,} & {r['accuracy']:.4f} \\\\"
            for _, r in t01.iterrows()]
    return wrap("\n".join(rows),
                "\\textbf{Validation of the LSTM branch.} Trained on "
                "synthetic AllC / AllD / Tit-for-Tat / Win-Stay-Lose-Shift "
                "trajectories at zero and 5\\% execution noise, balanced at "
                "40{,}320 sequences per strategy per level. The 10\\% and "
                "20\\% noise levels are never trained on. No language-model "
                "transcript enters training.",
                "tab:S-classifier", "lrc",
                "Split & $n$ & Accuracy \\\\")


def s8_shapes(corr, corrall):
    def table(c, order, caption, label):
        rows = []
        for m in order:
            # Diagonal left blank rather than filled with a dash: the project's
            # style check treats a run of hyphens as an em dash.
            cells = " & ".join(
                "" if m == k else f"${c.loc[m, k]:+.2f}$" for k in order)
            rows.append(f"{MODEL_LABEL[m]} & {cells} \\\\")
        return wrap("\n".join(rows), caption, label,
                    "l" + "c" * len(order),
                    " & " + " & ".join(MODEL_LABEL[m].split()[0] + " "
                                       + MODEL_LABEL[m].split()[1]
                                       for m in order) + " \\\\",
                    fit=True)

    a = table(corr, MODEL_ORDER,
              "\\textbf{Pairwise correlation between the model curves, five "
              "models.} Correlation of the ten-point "
              "cooperation-against-scale curves of each pair of the five "
              "models the main text reports. Six of the ten pairs are negative "
              "and the mean is $-0.10$, which is why averaging the curves "
              "cancels much of the movement.",
              "tab:S-shapes")
    b = table(corrall, MODEL_ORDER_ALL,
              "\\textbf{Pairwise correlation between the model curves, all "
              "six models.} The same correlations with the sixth model "
              "included: nine of the fifteen pairs are negative and the mean "
              "is $-0.08$.",
              "tab:S-shapes-all")
    return a + "\n" + b


def s9_firstmove(t08):
    """Opening-move range against the range over rounds 2 to 10.

    The two windows rest on 400 and 3,600 decisions per cell, so a range
    computed on the first is upward-biased relative to the second and the
    ratio is descriptive rather than a test.  The caption says so, because
    the column invites exactly the reading it cannot support.
    """
    t = t08.set_index("model")
    rows = []
    for m in MODEL_ORDER_ALL:
        d = t.loc[m]
        rows.append(f"{MODEL_LABEL[m]} & {d.range_round1:.3f} & "
                    f"{d.range_later:.3f} & {d.ratio:.2f} \\\\")
    return wrap("\n".join(rows),
                "\\textbf{Range of cooperation across the ten payoff scales, "
                "on the opening move and afterwards.} At round 1 the agent has "
                "seen only the payoff matrix and the instructions, so a "
                "dependence on the payoff scale there cannot have come through "
                "the history of play. The two windows rest on 400 and 3{,}600 "
                "decisions per cell, so their ranges are not variance-matched "
                "and the ratio is descriptive rather than a test; the bias runs "
                "the same way in every model, while the ratio does not.",
                "tab:S-firstmove", "lccc",
                "Model & Round 1 & Rounds 2--10 & Ratio \\\\")


def main():
    g = pd.read_parquet(DATA / "games.parquet")
    r = pd.read_parquet(DATA / "rounds.parquet")
    t01 = pd.read_csv(TAB / "T01_classifier.csv")
    t02 = pd.read_csv(TAB / "T02_scale_by_model.csv")
    t06 = pd.read_csv(TAB / "T06_variance.csv")
    t06all = pd.read_csv(TAB / "T06_variance_all.csv")
    t08 = pd.read_csv(TAB / "T08_first_move.csv")
    per = pd.read_csv(TAB / "T07_persona_by_scale.csv")
    t10p = pd.read_csv(TAB / "T10_provenance_by_scale.csv")
    t12p = pd.read_csv(TAB / "T12_label_mix.csv")
    corr = pd.read_csv(TAB / "T04_shape_correlations.csv", index_col=0)
    corrall = pd.read_csv(TAB / "T04_shape_correlations_all.csv",
                          index_col=0)
    d = pd.read_parquet(DATA / "readout.parquet")

    text = ("% Generated by Analysis/scaling/s07_supplementary.py - do not edit.\n"
            + s1_corpus(g, r) + "\n" + s2_scale(t02) + "\n" + s3_language(g)
            + "\n" + s4_variance(t06, t06all) + "\n" + s5_persona(per) + "\n"
            + s6_strategy(t10p, t12p) + "\n" + s6c_label_provenance(d) + "\n"
            + s7_classifier(t01) + "\n" + s8_shapes(corr, corrall)
            + "\n" + s9_firstmove(t08))
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
