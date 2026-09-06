# Interface Focus submission checklist

Target: *Interface Focus*, RSFS-2026-0050  
Manuscript: *Payoff scale reshapes how language models play the prisoner's dilemma*

This checklist separates checks that are already reproducible from the repository
from decisions that must be confirmed by the authors before submission.

## Completed in this revision

- [x] Main-text figure suite consolidated to the 11 redesigned figures.
- [x] Legacy f1-f8 figure references removed from `main.tex`.
- [x] Appendix cross-references repaired after the figure integration.
- [x] Source-level reference audit: no undefined `\ref{}` targets.
- [x] Source-level bibliography audit: every cited key is present in `mybib.bib`.
- [x] Public GitHub repository linked in the data-accessibility statement.
- [x] Corpus collection window recorded as 2-5 September 2026.
- [x] Root README corrected to OptionA = defect, OptionB = cooperate.
- [x] Root README corrected to the 10-round frontier corpus.
- [x] Benchmark task documentation corrected to the collected 10-scale, 10-round design.
- [x] Data card aligned with the actual sampling parameters
      (temperature 1.0, CRN seed, max_tokens cap).
- [x] Data-card fallback language qualified: released per-game CSVs do not retain
      retry/fallback counters.
- [x] AI-use statement is present and describes coding assistance and language editing.
- [x] Strict `preflight_submission.py` added.

## Author decisions required before submission

- [ ] **Choose and add a data licence.** Royal Society data-sharing guidance expects
      reusable data under an open licence; CC0 or CC BY are typical choices.
- [ ] **Choose and add a code licence** for the repository's original analysis and
      collection code. Keep the upstream FAIRGAME licence/notice intact and make the
      scope of the new licence explicit.
- [ ] Replace `[DATA LICENCE]` and `[CODE LICENCE]` in `main.tex`.
- [ ] Confirm the CRediT roles for every author.
- [ ] Confirm whether funding exists beyond EPSRC EP/Y00857X/1; if not, state that
      no other funding was received.
- [ ] Confirm acknowledgement wording for Kaggle model-access credits and any
      contributors below the authorship threshold.
- [ ] Confirm the three equal-contribution authors match the three daggers in the
      author list.

## Final-file checks

- [ ] Re-run the full statistics pipeline and `s08_verify_paper.py`.
- [ ] Re-run all 11 scripts in `Analysis/figures/` and the four appendix figures.
- [ ] Inspect final-size figure lettering. The current style includes note/tick text
      below 7.5 pt, so either enlarge it and regenerate or verify with the editorial
      office that the final files are acceptable.
- [ ] Build appendix first, then main, then appendix again if cross-references moved.
- [ ] Visually inspect both compiled PDFs for clipped panels, bad glyphs, stale
      cross-references and float placement.
- [ ] Upload final figures as separate production files at >=300 DPI where raster
      output is used, with fonts embedded in vector files.
- [ ] Run `python preflight_submission.py`; it should exit 0.
- [ ] Archive the exact submission commit in a DOI-issuing repository (for example
      Zenodo) and add the DOI to the data-accessibility statement and repository
      citation when available.

## Current expected main-text figures

1. `f_overview`
2. `f_landscape`
3. `f_language`
4. `f_persona`
5. `f_strategy_mix`
6. `f_firstmove`
7. `f_simplex`
8. `f_invasion`
9. `f_egt_vs_llm`
10. `f_robustness`
11. `f_strategy_space`
