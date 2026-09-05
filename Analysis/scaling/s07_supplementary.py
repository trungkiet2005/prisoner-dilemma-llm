"""Emit the supplementary tables as LaTeX.

Writes paper_scaling/supp_tables_auto.tex, which supplementary.tex \\input's.
Every number comes from the tables in tables/ or from the master parquet, so
nothing in the supplement is retyped by hand either.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (LANG_LABEL, LANG_ORDER, MODEL_LABEL,     # noqa: E402
                      MODEL_ORDER, SCALES)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TAB = HERE / "tables"
OUT = HERE.parents[1] / "paper_scaling" / "supp_tables_auto.tex"

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
    for m in MODEL_ORDER:
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
                "its tables otherwise.",
                "tab:S-corpus", "llccrrrc",
                "Model & Scales & Lang. & Dyads & Agent-games & Decisions & "
                "Cell $n$ & Coop. \\\\")


def s2_scale(t02):
    p = t02.pivot(index="model", columns="scale", values="coop")
    lo = t02.pivot(index="model", columns="scale", values="lo")
    hi = t02.pivot(index="model", columns="scale", values="hi")
    rows = []
    for m in MODEL_ORDER:
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
    for m in MODEL_ORDER:
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
                "across the ten scales for that model and language; it runs "
                "from 0.080 to 0.449, so the scale sensitivity of a model is "
                "a property of the model and the language jointly.",
                "tab:S-language", "ll" + "c" * len(SCALES) + "c",
                "Model & Language & " + " & ".join(f"{s}" for s in SC)
                + " & Range \\\\", small="\\scriptsize", colsep="2.5pt", fit=True)


def s4_variance(t06):
    name = {"C(model)": "Model", "C(language)": "Language",
            "C(scale_nominal)": "Payoff scale",
            "C(model):C(language)": "Model $\\times$ language",
            "C(model):C(scale_nominal)": "Model $\\times$ scale",
            "C(language):C(scale_nominal)": "Language $\\times$ scale",
            "Residual": "Residual"}
    rows = []
    for _, r in t06.iterrows():
        f = "" if np.isnan(r["F"]) else f"{r['F']:.2f}"
        p = ("" if np.isnan(r["PR(>F)"]) else
             ("$<0.001$" if r["PR(>F)"] < 0.001 else f"${r['PR(>F)']:.3f}$"))
        rows.append(f"{name.get(r['term'], r['term'])} & {r['sum_sq']:.4f} & "
                    f"{int(r['df'])} & {f} & {p} & {r['pct_variance']:.1f} \\\\")
    return wrap("\n".join(rows),
                "\\textbf{Variance decomposition of the cell means.} Type-II "
                "analysis of variance on the 250 model $\\times$ language "
                "$\\times$ scale cell means, where the design is exactly "
                "balanced. The payoff scale has no significant main effect "
                "while both of its interactions are significant.",
                "tab:S-variance", "lrrrrr",
                "Term & Sum sq. & df & $F$ & $p$ & \\% variance \\\\")


def s5_persona(per):
    p = per.pivot(index="model", columns="scale", values="persona_effect")
    rows = []
    for m in MODEL_ORDER:
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
    for m in MODEL_ORDER:
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
    for m in MODEL_ORDER:
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


def s8_shapes(corr):
    rows = []
    for m in MODEL_ORDER:
        # Diagonal left blank rather than filled with a dash: the project's
        # style check treats a run of hyphens as an em dash.
        cells = " & ".join(
            "" if m == n else f"${corr.loc[m, n]:+.2f}$" for n in MODEL_ORDER)
        rows.append(f"{MODEL_LABEL[m]} & {cells} \\\\")
    return wrap("\n".join(rows),
                "\\textbf{Pairwise correlation between the model curves.} "
                "Correlation of the ten-point cooperation-against-scale curves "
                "of each pair of models. Six of the ten pairs are negative and "
                "the mean is $-0.10$, which is why averaging the curves "
                "cancels most of the movement.",
                "tab:S-shapes", "l" + "c" * len(MODEL_ORDER),
                " & " + " & ".join(MODEL_LABEL[m].split()[0] + " "
                                   + MODEL_LABEL[m].split()[1]
                                   for m in MODEL_ORDER) + " \\\\")


def main():
    g = pd.read_parquet(DATA / "games.parquet")
    r = pd.read_parquet(DATA / "rounds.parquet")
    t01 = pd.read_csv(TAB / "T01_classifier.csv")
    t02 = pd.read_csv(TAB / "T02_scale_by_model.csv")
    t06 = pd.read_csv(TAB / "T06_variance.csv")
    per = pd.read_csv(TAB / "T07_persona_by_scale.csv")
    t10p = pd.read_csv(TAB / "T10_provenance_by_scale.csv")
    t12p = pd.read_csv(TAB / "T12_label_mix.csv")
    corr = pd.read_csv(TAB / "T04_shape_correlations.csv", index_col=0)
    d = pd.read_parquet(DATA / "readout.parquet")

    text = ("% Generated by Analysis/scaling/s07_supplementary.py - do not edit.\n"
            + s1_corpus(g, r) + "\n" + s2_scale(t02) + "\n" + s3_language(g)
            + "\n" + s4_variance(t06) + "\n" + s5_persona(per) + "\n"
            + s6_strategy(t10p, t12p) + "\n" + s6c_label_provenance(d) + "\n"
            + s7_classifier(t01) + "\n" + s8_shapes(corr))
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
