"""Assemble the files to upload to ScholarOne for revision RSFS-2026-0050.R1.

The working sources in revised_paper/ stay as they are (red highlighting,
\\input of generated tables, cross-document references through xr), because
that is the version edited on Overleaf. This script derives from them what the
journal asks for at the final stage:

  manuscript_source/   main.tex as ONE complete file: tables inlined, every
                       reference into the supplement replaced by its number,
                       xr removed, highlighting off; plus main.bbl, mybib.bib,
                       the figure files it calls, and the PDF it compiles to.
  figures/             one file per main-text figure, Figure_1.png ... Figure_7.png.
  supplementary/       the electronic supplementary material as a clean PDF.
  tracked_changes/     manuscript and supplement with every change in red.
  Response_to_Reviewers_RSFS-2026-0050.docx

It refuses to finish if any build has an error or an undefined reference, and
it reports the remaining ">>> FILL IN <<<" placeholders. Re-run it after every
edit of revised_paper/:

    python make_submission_package.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "revised_paper"
OUT = HERE / "submission_package"
BUILD = OUT / "_build"
LETTER = HERE / "Response_to_Reviewers_RSFS-2026-0050.docx"

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
REF_RE = re.compile(r"\\(ref|eqref|pageref)\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
NEWLABEL_RE = re.compile(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_log(log: Path) -> None:
    text = log.read_text(encoding="latin-1")
    bad = [l for l in text.splitlines()
           if l.startswith("!") or "Warning: Reference" in l or "Warning: Citation" in l
           or "There were undefined references" in l]
    if bad:
        sys.exit(f"{log.name}: build problems\n  " + "\n  ".join(bad[:10]))


def build_pair(folder: Path) -> None:
    """Appendix first, then main, then the appendix again, as CLAUDE.md requires."""
    for f in folder.glob("*.bbl"):
        f.unlink()
    pdf = ["pdflatex", "-interaction=nonstopmode"]
    run(pdf + ["appendix"], folder)
    run(["biber", "appendix"], folder)
    run(pdf + ["appendix"], folder)
    run(pdf + ["main"], folder)
    run(["bibtex", "main"], folder)
    run(pdf + ["main"], folder)
    run(pdf + ["main"], folder)
    run(pdf + ["appendix"], folder)
    check_log(folder / "main.log")
    check_log(folder / "appendix.log")


def copy_sources(dest: Path, highlight: bool) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(SRC, dest, ignore=shutil.ignore_patterns("pdf"))
    if not highlight:
        for name in ("main.tex", "appendix.tex"):
            p = dest / name
            s = p.read_text(encoding="utf-8")
            assert s.count("\\highlightrevisionstrue") == 1, name
            p.write_text(s.replace("\\highlightrevisionstrue", "\\highlightrevisionsfalse"),
                         encoding="utf-8")


def labels_from_aux(aux: Path) -> dict[str, str]:
    out = {}
    for key, num, _page in NEWLABEL_RE.findall(aux.read_text(encoding="latin-1")):
        if not key.endswith("@cref"):
            out[key] = num
    return out


def flatten_main(build: Path, dest: Path) -> list[str]:
    """Write a self-contained main.tex into dest and return the figure paths it calls."""
    s = (build / "main.tex").read_text(encoding="utf-8")

    def inline(m: re.Match) -> str:
        name = m.group(1)
        body = (build / (name if name.endswith(".tex") else name + ".tex")).read_text(encoding="utf-8")
        return f"% ---- begin inlined {name}.tex ----\n{body}\n% ---- end inlined {name}.tex ----\n"

    # only \input lines that are not commented out
    lines = []
    for line in s.splitlines(keepends=True):
        code = line.split("%", 1)[0]
        lines.append(INPUT_RE.sub(inline, line) if INPUT_RE.search(code) else line)
    s = "".join(lines)

    own = set(LABEL_RE.findall(s))
    ext = labels_from_aux(build / "appendix.aux")
    missing = []

    def resolve(m: re.Match) -> str:
        cmd, key = m.groups()
        if key in own:
            return m.group(0)
        if key not in ext:
            missing.append(key)
            return m.group(0)
        num = ext[key]
        return f"({num})" if cmd == "eqref" else num

    s = REF_RE.sub(resolve, s)
    if missing:
        sys.exit("references found in neither document: " + ", ".join(sorted(set(missing))))
    s = re.sub(r"^\\usepackage\{xr\}\s*\n", "", s, flags=re.M)
    s = re.sub(r"^\\externaldocument(\[[^\]]*\])*\{[^}]*\}\s*\n", "", s, flags=re.M)
    s = s.replace("% Revision RSFS-2026-0050.R1:",
                  "% Final source for RSFS-2026-0050.R1, generated by make_submission_package.py:\n"
                  "% generated tables are inlined and references to the electronic supplementary\n"
                  "% material are written as numbers, so this file compiles on its own.\n"
                  "% Revision RSFS-2026-0050.R1:", 1)
    (dest / "main.tex").write_text(s, encoding="utf-8")
    return [m for m in GRAPHICS_RE.findall(s.split("\\end{document}")[0])]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    BUILD.mkdir(parents=True)

    hi, clean = BUILD / "highlighted", BUILD / "clean"
    copy_sources(hi, highlight=True)
    copy_sources(clean, highlight=False)
    print("building highlighted and clean versions ...")
    build_pair(hi)
    build_pair(clean)

    # aux files go back to the working folder: Overleaf cannot produce appendix.aux
    for name in ("main.aux", "appendix.aux"):
        shutil.copy2(hi / name, SRC / name)

    tracked = OUT / "tracked_changes"
    tracked.mkdir()
    shutil.copy2(hi / "main.pdf", tracked / "RSFS-2026-0050.R1_manuscript_tracked.pdf")
    shutil.copy2(hi / "appendix.pdf", tracked / "RSFS-2026-0050.R1_ESM_tracked.pdf")

    supp = OUT / "supplementary"
    supp.mkdir()
    shutil.copy2(clean / "appendix.pdf", supp / "RSFS-2026-0050.R1_ESM.pdf")

    # self-contained manuscript source, compiled once more on its own
    ms = OUT / "manuscript_source"
    ms.mkdir()
    graphics = flatten_main(clean, ms)
    shutil.copy2(clean / "mybib.bib", ms / "mybib.bib")
    for g in graphics:
        (ms / g).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clean / g, ms / g)
    run(["pdflatex", "-interaction=nonstopmode", "main"], ms)
    run(["bibtex", "main"], ms)
    run(["pdflatex", "-interaction=nonstopmode", "main"], ms)
    run(["pdflatex", "-interaction=nonstopmode", "main"], ms)
    check_log(ms / "main.log")
    for f in ms.iterdir():
        if f.suffix in {".aux", ".log", ".blg", ".out", ".toc"}:
            f.unlink()
    (ms / "main.pdf").rename(ms / "RSFS-2026-0050.R1_manuscript.pdf")
    if (ms / "main.tex").read_text(encoding="utf-8").count("\\input{"):
        sys.exit("an \\input survived flattening")

    # one file per main-text figure, in the order the text calls them
    figs = OUT / "figures"
    figs.mkdir()
    for i, g in enumerate(graphics, 1):
        png = (clean / g).with_suffix(".png")
        if not png.exists():
            sys.exit(f"no PNG for {g}")
        shutil.copy2(png, figs / f"Figure_{i}.png")
        print(f"  Figure_{i}.png  <-  {png.name}")

    if LETTER.exists():
        shutil.copy2(LETTER, OUT / LETTER.name)
    shutil.rmtree(BUILD)

    fills = []
    for name in ("main.tex", "appendix.tex", "mybib.bib"):
        for n, line in enumerate((SRC / name).read_text(encoding="utf-8").splitlines(), 1):
            if ">>> FILL IN <<<" in line and "pre-submission search" not in line:
                fills.append(f"{name}:{n}: {line.strip()[:110]}")
    print(f"\nwrote {OUT}")
    print(f"{len(fills)} placeholder(s) still to fill:" if fills else "no placeholders left")
    for f in fills:
        print("  " + f)


if __name__ == "__main__":
    main()
