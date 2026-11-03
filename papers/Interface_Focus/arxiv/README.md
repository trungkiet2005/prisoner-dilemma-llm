# arXiv package

`make_arxiv.py` turns `../main.tex` and `../appendix.tex` into one flat arXiv
source bundle:

- merges the supplement after the main text (arXiv accepts one `\documentclass`)
- switches revision highlighting off and drops the journal manuscript number
- strips every comment with `arxiv_latex_cleaner`
- compiles the bundle in an empty directory and refuses on any undefined
  reference, undefined citation, missing file or fatal error
- writes `arxiv_submission.zip` (main.tex, main.bbl, mybib.bib, figures) and
  `preview.pdf`

```bash
pip install arxiv-latex-cleaner   # or pass the binary path as the first argument
python3 make_arxiv.py
```

`build/`, `preview.pdf` and the zip are regenerated outputs and are not tracked.
Paste-ready form fields are in `metadata.txt`.

## Posting

1. Log in at https://arxiv.org, choose "Start New Submission".
2. Licence: CC BY 4.0. Primary category cs.AI, cross-lists in `metadata.txt`.
3. Upload `arxiv_submission.zip`; arXiv unpacks it and builds with TeX Live.
4. Paste title, authors, abstract and comments from `metadata.txt`.
5. Open the PDF arXiv built and check: 57 pages, no `[?]` in citations, no
   `??` in references, all 17 figures present.
6. Submit. It appears after the next announcement cycle (about one working day).

After the journal publishes, add the DOI and journal reference to the arXiv
record (no new version needed).
