"""Build a flat, single-document arXiv bundle from main.tex + appendix.tex.

arXiv accepts one \\documentclass per submission, so the supplement is merged
after the main text. Revision highlighting is switched off and the journal
manuscript number is dropped. The bundle is compiled in an empty directory and
the build is refused on any undefined reference, citation or fatal error.
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
OUT = HERE / "build"
ZIP = HERE / "arxiv_submission.zip"

main = (SRC / "main.tex").read_text(encoding="utf-8")
app = (SRC / "appendix.tex").read_text(encoding="utf-8")

main = main.replace("\\highlightrevisionstrue", "\\highlightrevisionsfalse")
main = main.replace("\\usepackage{xcolor}",
                    "\\usepackage{xcolor}\n\\usepackage{multirow}\n\\usepackage{chngcntr}", 1)

main = main.replace("electronic supplementary table~", "table~")
main = main.replace("in the electronic supplementary material", "in the supplementary material")

body = app.split("\\begin{document}", 1)[1]
body = body.split("\\bibliographystyle{unsrtnat}", 1)[0]
body = re.sub(r"\\begin\{center\}.*?\\end\{center\}",
              r"\\begin{center}\n  {\\LARGE\\bfseries Supplementary material}\n\\end{center}",
              body, count=1, flags=re.S)
body = re.sub(r"^.*\\tableofcontents.*$", "", body, flags=re.M)
body = re.sub(r"\\setcounter\{tocdepth\}\{2\}\s*\\clearpage", "", body, count=1)
body = body.replace(
    "\\appendix",
    "\\appendix\n\\renewcommand\\thesubsection{\\thesection.\\arabic{subsection}}\n"
    "\\renewcommand\\thesubsubsection{\\thesubsection.\\arabic{subsubsection}}",
    1,
)

tail = "\\end{document}"
assert main.rstrip().endswith(tail)
merged = main.rstrip()[: -len(tail)] + "\n\\clearpage\n" + body.strip() + "\n\n" + tail + "\n"

figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", merged)))

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "figures").mkdir(parents=True)
(OUT / "main.tex").write_text(merged, encoding="utf-8")
shutil.copy(SRC / "mybib.bib", OUT / "mybib.bib")
for f in figs:
    shutil.copy(SRC / f, OUT / f)


def run(*cmd, cwd=OUT):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


run("pdflatex", "-interaction=nonstopmode", "main.tex")
run("bibtex", "main")
blg = (OUT / "main.blg").read_text(encoding="latin-1")
bibwarn = [l for l in blg.splitlines() if l.startswith(("Warning--I didn't find", "I couldn't open"))]

cleaner = shutil.which("arxiv_latex_cleaner") or (len(sys.argv) > 1 and sys.argv[1])
if not cleaner:
    sys.exit("arxiv_latex_cleaner not found: pip install arxiv-latex-cleaner, or pass its path")
for p in OUT.glob("main.*"):
    if p.suffix not in (".tex", ".bbl"):
        p.unlink()
staged = OUT.with_name("staged")
shutil.rmtree(staged, ignore_errors=True)
OUT.rename(staged)
r = run(cleaner, "--keep_bib", str(staged), cwd=HERE)
cleaned = HERE / "staged_arXiv"
if r.returncode or not cleaned.exists():
    sys.exit("cleaner failed\n" + r.stderr)
cleaned.rename(OUT)
shutil.rmtree(staged)
assert not re.search(r"(?m)^\s*%", (OUT / "main.tex").read_text(encoding="utf-8"))

run("pdflatex", "-interaction=nonstopmode", "main.tex")
run("pdflatex", "-interaction=nonstopmode", "main.tex")

log = (OUT / "main.log").read_text(encoding="latin-1")
bad = [l for l in log.splitlines()
       if re.search(r"undefined|Emergency stop|Fatal error|File `.*' not found|Missing \$", l)]
if bad or bibwarn or not (OUT / "main.pdf").exists():
    print("BUILD REFUSED")
    print("\n".join(bad + bibwarn))
    sys.exit(1)

shutil.copy(OUT / "main.pdf", HERE / "preview.pdf")
keep = ["main.tex", "main.bbl", "mybib.bib"] + figs
if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for k in keep:
        z.write(OUT / k, k)
pages = re.search(r"Output written on main.pdf \((\d+) pages", log)
print(f"OK  pages={pages.group(1) if pages else '?'}  figures={len(figs)}  "
      f"zip={ZIP.stat().st_size / 1e6:.2f} MB")
