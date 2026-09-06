# Superseded drafts

Kept, not deleted, because their numbers are cited in the current manuscript as
prior work by the same authors and because the earlier figure stems are still
referenced from the analysis pipeline.

## `paper_scaling/`

The pre-restructure draft of *Payoff scale reshapes how language models play the
prisoner's dilemma*, superseded on 2026-09-06 by
[`../papers/interface-focus`](../papers/interface-focus/README.md), which carries
the same title and the same manuscript ID, RSFS-2026-0050. The restructure moved
Methods ahead of Results, merged six Results subsections into four and added a
Conclusion. No claim and no number changed except five prose figures corrected
to agree with the generated tables; `README.md` in this directory lists them.

`RUN_PLAN.md` is still live reading: it is the collection plan whose experiment
numbers name the run directories in [`../results/kbench`](../results/README.md).

To rebuild this draft rather than the current one:

```bash
cd Analysis/scaling
export PD_PAPER_DIR="$(pwd)/../../legacy/paper_scaling"
export PD_FIGDIR="$PD_PAPER_DIR/figures"
python s05_figures.py
```

Pointing the pipeline here is what switches `LEGACY_ALIAS` on, so the figures are
also written under their old `fig1_`..`fig6_` names and this draft keeps building
from an unedited `main.tex`.
