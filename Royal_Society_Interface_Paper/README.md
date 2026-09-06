# Royal Society Interface submission

Single-column manuscript for *J. R. Soc. Interface* (class `rsproca_new`), generated
from the Elsevier manuscript in `../els-cas-paper/` by `build_rsi_from_els.py`
(body content is ported verbatim; all deliberate content fixes are documented in
that script's `transform()`).

## Files

| File | Role |
|---|---|
| `main.tex` / `main.pdf` | Main manuscript (21 pp.) |
| `appendix.tex` / `appendix.pdf` | Standalone Supplementary Appendix / ESM (17 pp.) |
| `mybib.bib` | Bibliography database (biblatex/biber, style fixed by the class) |
| `rsproca_new.cls` | Journal class |
| `TemplateFigs/` | Journal logos **required by the class** (`\maketitle` first page) — do not delete |
| `all_figures/` | The 30 figure PDFs referenced by the two `.tex` files |
| `*.bbl` | Pre-built bibliographies (lets the sources compile without running biber) |
| `build_rsi_from_els.py` | Regenerates `main.tex`/`appendix.tex` from `../els-cas-paper/` |

## Build order

`main.tex` pulls the appendix's labels via `xr`, so the appendix must be built first:

```
pdflatex appendix && biber appendix && pdflatex appendix && pdflatex appendix
pdflatex main     && biber main     && pdflatex main     && pdflatex main
```

The `\RequirePackage[2025-06-01]{latexrelease}` line at the top of both sources
works around this machine's MiKTeX (2026 kernel + biblatex 3.21); it is harmless
on Overleaf/TeX Live and can be removed once the local MiKTeX is fully updated.

## Regenerating from the Elsevier source

```
python build_rsi_from_els.py   # rewrites main.tex + appendix.tex
```

Figure note: `all_figures/fig_unfit_vs_threshold_small_llm_by_model.pdf` is
produced by `analysis/appendix_figures/fig_unfit_vs_threshold_small_llm_by_model.py`
(self-checks against the hybrid pipeline's retained counts); the remaining
figures are copies from `../els-cas-paper/all_figures/`.
