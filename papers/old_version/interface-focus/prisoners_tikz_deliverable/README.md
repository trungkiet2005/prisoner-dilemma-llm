# TikZ prisoner-dilemma overview

Files:
- `prisoners_dilemma_overview.pdf`: final figure; all labels are live/selectable PDF text.
- `prisoners_dilemma_overview.tex`: TikZ/LuaLaTeX source.
- `assets/`: transparent PNG assets extracted from the supplied raster figure.
- `extract_assets.py`: reproducible asset-extraction script (coordinates correspond to the supplied 1448x1086 image).

Compile from this directory with:

```bash
lualatex -interaction=nonstopmode prisoners_dilemma_overview.tex
```

The source uses the installed `Inter` font and standard TikZ/graphicx/amsmath packages.
