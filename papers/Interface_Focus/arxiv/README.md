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

## Posting (replacement of arXiv:2601.19082, becomes v3)

1. Log in at https://arxiv.org as the submitter of record and open the user page.
2. On 2601.19082 choose "Replace". Do not start a new submission.
3. Upload `arxiv_submission.zip`.
4. Update authors (The Anh Han first), abstract and comments from `metadata.txt`;
   title, categories and licence stay as they are.
5. Open the PDF arXiv built and check: 57 pages, no `[?]` in citations, no
   `??` in references, all 17 figures present, the accepted-manuscript line on page 1.
6. Submit. The new version appears after the next announcement cycle.

After the journal publishes, add the DOI and journal reference from the arXiv
user page ("Add journal ref and DOI"); this does not create a new version.
