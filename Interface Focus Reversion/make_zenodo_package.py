"""Build the Zenodo deposit for RSFS-2026-0050: data, prompts and analysis code.

The deposit mirrors the repository layout (Dataset/, results/, Analysis/,
kaggle/, FAIRGAME/resources/) so that the analysis pipeline runs from the
unpacked archive without editing a path. Only what the article uses is
included, and nothing that carries an account name or a credential: the
multi-account launch scripts and the quota probe of kaggle/benchmarks are left
out on purpose.

    python make_zenodo_package.py

writes  zenodo_upload/payoff-scaling-llm-pd-v1.0/      (the unpacked deposit)
        zenodo_upload/payoff-scaling-llm-pd-v1.0.zip   (the file to upload)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NAME = "payoff-scaling-llm-pd-v1.0"
OUT = HERE / "zenodo_upload"
DEST = OUT / NAME

# (source relative to the repository, destination relative to the deposit)
TREES = [
    ("Dataset/data_fairgame_frontier_llm", "Dataset/data_fairgame_frontier_llm"),
    ("Dataset/data_replicate_frontier", "Dataset/data_replicate_frontier"),
    ("Dataset/data_fairgame_e2_notation", "Dataset/data_fairgame_e2_notation"),
    ("results/kbench/exp2/run359522", "results/kbench/exp2/run359522"),
    ("results/kbench/exp2/run971021", "results/kbench/exp2/run971021"),
    ("results/kbench/exp4/run1029391", "results/kbench/exp4/run1029391"),
    ("Analysis/pdlib", "Analysis/pdlib"),
    ("Analysis/scaling/data", "Analysis/scaling/data"),
    ("Analysis/scaling/tables", "Analysis/scaling/tables"),
    ("Analysis/scaling/models", "Analysis/scaling/models"),
]
FILES = [
    ("Dataset/DATA_CARD_frontier_llm.md", "Dataset/DATA_CARD_frontier_llm.md"),
    ("results/README.md", "results/README.md"),
    ("LICENSE", "LICENSE"),
    ("DATA_LICENSE.md", "DATA_LICENSE.md"),
    ("kaggle/benchmarks/pd_task.py", "kaggle/benchmarks/pd_task.py"),
    ("kaggle/benchmarks/collect_run.py", "kaggle/benchmarks/collect_run.py"),
    ("kaggle/benchmarks/run_and_watch.py", "kaggle/benchmarks/run_and_watch.py"),
    ("kaggle/benchmarks/test_pd_task_parity.py", "kaggle/benchmarks/test_pd_task_parity.py"),
    ("FAIRGAME/LICENSE", "FAIRGAME/LICENSE"),
    ("FAIRGAME/NOTICE.md", "FAIRGAME/NOTICE.md"),
]
# the synthetic corpus the strategy classifier is trained and tested on
NOISE = [f"Dataset/noise_dataset/{lvl}/4stratsAllCAllDTFTWSLS_{suf}.txt"
         for lvl, suf in [("NoNoise", "nonoise"), ("Noise005", "noise005"),
                          ("Noise01", "noise01"), ("Noise02", "noise02")]]
# run reports, with token counts and metered cost, of the three archived runs
RUN_REPORTS = {
    "results/kbench_download/prisoner-dilemma-fairgame/2/gemini-3.5-flash-lite/359522": "results/kbench/exp2/run359522",
    "results/kbench_download/prisoner-dilemma-fairgame/2/gemini-3.5-flash-lite/971021": "results/kbench/exp2/run971021",
    "results/kbench_download/prisoner-dilemma-fairgame/4/gemini-3.5-flash-lite/1029391": "results/kbench/exp4/run1029391",
}
LANGS = {"en": "English", "fr": "French", "vn": "Vietnamese", "cn": "Chinese", "ar": "Arabic"}
PERSONALITIES = {  # copied from kaggle/benchmarks/pd_task.py, checked below
    "en": ["cooperative", "selfish"], "fr": ["coopératif", "égoïste"],
    "ar": ["متعاون", "أناني"], "cn": ["合作型的", "自私型的"],
    "vn": ["một người hợp tác", "một người ích kỷ"],
}
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "*.log")

AUTHORS = [
    ("Han", "The Anh", "School of Computing, Engineering and Digital Technologies, Teesside University, Middlesbrough, United Kingdom"),
    ("Huynh", "Trung-Kiet", "University of Science (HCMUS), Vietnam National University Ho Chi Minh City"),
    ("Duy-Minh", "Dao-Sy", "University of Science (HCMUS), Vietnam National University Ho Chi Minh City"),
    ("Cao", "Thanh-Bang", "Ho Chi Minh City University of Technology (HCMUT), Vietnam National University Ho Chi Minh City"),
    ("Le", "Phong-Hao", "Ho Chi Minh City University of Technology (HCMUT), Vietnam National University Ho Chi Minh City"),
    ("Nguyen", "Hong-Dan", "Ho Chi Minh City University of Technology (HCMUT), Vietnam National University Ho Chi Minh City"),
    ("Nguyen-Lam", "Phu-Quy", "University of Science (HCMUS), Vietnam National University Ho Chi Minh City"),
    ("Nguyen-Vo", "Minh-Luan", "Ho Chi Minh City University of Technology (HCMUT), Vietnam National University Ho Chi Minh City"),
    ("Pham", "Hong-Phat", "Ho Chi Minh City University of Technology (HCMUT), Vietnam National University Ho Chi Minh City"),
    ("Pham", "Phu-Hoa", "University of Science (HCMUS), Vietnam National University Ho Chi Minh City"),
    ("Than", "Thien-Kim", "Ho Chi Minh City University of Technology (HCMUT), Vietnam National University Ho Chi Minh City"),
    ("Tran", "Chi-Nguyen", "Ho Chi Minh City University of Technology (HCMUT), Vietnam National University Ho Chi Minh City"),
    ("Tran", "Huy", "Ho Chi Minh City University of Technology (HCMUT), Vietnam National University Ho Chi Minh City"),
    ("Tran-Le", "Gia-Thoai", "Ho Chi Minh City University of Technology (HCMUT), Vietnam National University Ho Chi Minh City"),
    ("Buscemi", "Alessio", "Luxembourg Institute of Science and Technology (LIST), Luxembourg"),
    ("Trang", "Le Hong", "Ho Chi Minh City University of Technology (HCMUT), Vietnam National University Ho Chi Minh City"),
]
TITLE = ("Payoff scaling shapes cooperation in LLM agents across languages: "
         "game transcripts, prompts and analysis code")


def copy_tree(src: str, dst: str) -> None:
    shutil.copytree(REPO / src, DEST / dst, ignore=SKIP)


def copy_file(src: str, dst: str) -> None:
    (DEST / dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / src, DEST / dst)


def prompts() -> None:
    """Templates, persona lines and rendered example prompts, in one place."""
    p = DEST / "prompts"
    (p / "fairgame_templates").mkdir(parents=True)
    for f in sorted((REPO / "FAIRGAME/resources/game_templates").glob("prisoner_dilemma_*")):
        shutil.copy2(f, p / "fairgame_templates" / f.name)
    src = (REPO / "kaggle/benchmarks/pd_task.py").read_text(encoding="utf-8")
    for lang, pair in PERSONALITIES.items():
        assert all(f"'{x}'" in src for x in pair), f"persona lines for {lang} drifted"
    lines = ["# Persona lines", "",
             "Each agent receives one line naming its own persona, in the language of the",
             "prompt. The opponent's persona is never disclosed. The four ordered pairings",
             "(cooperative/cooperative, cooperative/selfish, selfish/cooperative,",
             "selfish/selfish) are indexed p0 to p3 in the per-decision records.", "",
             "| language | cooperative | selfish |", "|---|---|---|"]
    lines += [f"| {LANGS[l]} (`{l}`) | {a} | {b} |" for l, (a, b) in PERSONALITIES.items()]
    (p / "persona_lines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ex = p / "example_prompts"
    ex.mkdir()
    ck = REPO / "results/kbench/exp2/run971021/gemini-3.5-flash-lite/checkpoints"
    ck2 = REPO / "results/kbench/exp2/run359522/gemini-3.5-flash-lite/checkpoints"
    for lang in LANGS:
        for lam, folder in (("0p01", ck), ("1", ck2), ("1000", ck)):
            rec = json.loads((folder / f"x{lam}__lang-{lang}__p0__rep-000.json").read_text(encoding="utf-8"))
            for t in rec["turns"]:
                if t["agent"] == "agent1" and t["round"] in (1, 5):
                    (ex / f"{lang}_lambda{lam.replace('p', '.')}_round{t['round']}.txt").write_text(
                        t["prompt"], encoding="utf-8")


def requirements() -> None:
    (DEST / "requirements.txt").write_text(
        "# Versions under which the published results were produced (ESM, appendix E).\n"
        "# Python 3.12.3. egttools has no wheel for Python 3.14.\n"
        "numpy==1.26.4\npandas==2.2.2\npyarrow==16.1.0\nscipy==1.17.1\n"
        "statsmodels==0.14.6\nscikit-learn==1.5.1\ntorch==2.11.0\nmatplotlib==3.9.2\n"
        "mpmath==1.3.0\negttools==0.1.14.2\n"
        "# collection only\nkaggle-benchmarks==0.6.1\n", encoding="utf-8")


def citation(doi: str | None) -> None:
    a = []
    for fam, giv, aff in AUTHORS:
        a += [f'  - family-names: "{fam}"', f'    given-names: "{giv}"', f'    affiliation: "{aff}"']
    (DEST / "CITATION.cff").write_text("\n".join([
        "cff-version: 1.2.0",
        'message: "If you use this deposit, please cite the article and the deposit."',
        f'title: "{TITLE}"',
        "version: 1.0",
        'license: "CC-BY-4.0"',
        'repository-code: "https://github.com/trungkiet2005/prisoner-dilemma-llm"',
        "authors:", *a,
        "keywords:",
        *[f'  - "{k}"' for k in ["machine behaviour", "large language models", "iterated prisoner's dilemma",
                                  "payoff scaling", "utility invariance", "cross-lingual behaviour", "game theory"]],
        f'doi: "{doi}"' if doi else "# doi: rerun make_zenodo_package.py --doi <reserved DOI>",
        ""]), encoding="utf-8")


README = """# {title}

Data, prompts and code for the article

> Han T.A., Huynh T.-K., Duy-Minh D.-S., *et al.* **Payoff scaling shapes cooperation in
> LLM agents across languages.** *Interface Focus* (theme issue *Machine behaviour in the
> age of large language models*), manuscript RSFS-2026-0050.
{doi_line}

Five commercial language models (plus a sixth reported in the supplement) play a ten-round
iterated prisoner's dilemma against themselves at ten payoff scales
λ ∈ {{0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 100, 1000}}, in five prompt languages, under four
persona pairings, ten replicates each: 12,000 dyads, 24,000 agent-games, 240,000 decisions.
No new model calls are needed to reproduce any number in the article.

## Contents

| path | what it is |
|---|---|
| `Dataset/data_fairgame_frontier_llm/` | **the corpus**: 300 CSV files, one per (scale, model, language), 40 games each. Schema, action coding and verified invariants: `Dataset/DATA_CARD_frontier_llm.md` |
| `Dataset/data_replicate_frontier/` | independent repeat of Gemini 3.5 Flash-Lite at λ ∈ {{0.1, 0.25, 0.5, 1}} (ESM D.5) |
| `Dataset/data_fairgame_e2_notation/` | notation control: payoffs printed with two or three decimals at fixed magnitude (ESM D.5) |
| `Dataset/noise_dataset/` | synthetic AllC/AllD/TFT/WSLS trajectories at four execution-noise levels, used only to train and test the strategy classifier |
| `results/kbench/` | raw output of the three archived Kaggle Benchmarks runs of Gemini 3.5 Flash-Lite: per-game CSVs and **one JSON record per game holding every prompt, raw reply, requested sampling seed, parse-retry count and fallback flag**, plus each run's report with token counts and metered cost |
| `prompts/` | the FAIRGAME prisoner's-dilemma templates in five languages, the persona lines, and rendered example prompts at λ = 0.01, 1 and 1000 |
| `kaggle/benchmarks/` | the collection task (`pd_task.py`), its template-parity test, and the scripts that download a run and promote it into `Dataset/` |
| `Analysis/scaling/` | the analysis pipeline, s00 to s15; `data/` holds the tidy game, round and read-out tables, `tables/` every result table, `models/` the trained classifier |
| `Analysis/pdlib/` | shared library: ingest, rule base, sequence coding, classifier |
| `Analysis/figures/` | one script per figure of the article |
| `FAIRGAME/` | licence and notice of the FAIRGAME framework, whose templates are reproduced in `prompts/` |

What the deposit does **not** contain: per-decision records for the models other than
Gemini 3.5 Flash-Lite, which were not retained from the collection runs, and the earlier
open-weight corpus, which the article does not use.

## Reproducing the article

```bash
pip install -r requirements.txt          # Python 3.12
python Analysis/scaling/rebuild_interface_focus.py            # everything, including the evolutionary grid
python Analysis/scaling/rebuild_interface_focus.py --skip-egt # if egttools cannot be installed
```

On Windows, unpack into a short path (for example `C:\\data\\`): the longest file path in the
deposit is 156 characters, and Windows refuses paths longer than 260.

The rebuild reads only `Dataset/` and `results/kbench/`, regenerates every table and figure,
and ends with `s08_verify_paper.py`, which recomputes the numbers quoted in the article and
fails if any of them disagrees. Seeds are fixed (ESM appendix E). No GPU is needed.

Two cautions from the data card: the action **Option B is cooperation** (the prompt states
penalties to be minimised), and the Arabic and Chinese templates phrase the goal as
maximising a reward although the cells are penalties. Both are reproduced from FAIRGAME
on purpose and discussed in the article.

## Licences

Data (`Dataset/`, `results/`, `Analysis/scaling/data`, `Analysis/scaling/tables`, `prompts/`
except the FAIRGAME templates): CC BY 4.0, see `DATA_LICENSE.md`. Code: MIT, see `LICENSE`.
The FAIRGAME templates in `prompts/fairgame_templates/` remain under the Apache License 2.0
of FAIRGAME, see `FAIRGAME/LICENSE` and `FAIRGAME/NOTICE.md`.

## Integrity

`SHA256SUMS.txt` lists a checksum for every file. Verify with `sha256sum -c SHA256SUMS.txt`.
"""


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi", help="the DOI Zenodo reserved for this deposit, e.g. 10.5281/zenodo.1234567")
    doi = ap.parse_args().doi
    if doi and not doi.startswith("10.5281/zenodo."):
        raise SystemExit("expected a Zenodo DOI of the form 10.5281/zenodo.<number>")
    if OUT.exists():
        shutil.rmtree(OUT)
    DEST.mkdir(parents=True)
    for s, d in TREES:
        copy_tree(s, d)
    for s, d in FILES:
        copy_file(s, d)
    for f in NOISE:
        copy_file(f, f)
    for src, dst in RUN_REPORTS.items():
        for f in (REPO / src).glob("*.run.json"):
            shutil.copy2(f, DEST / dst / f.name)
    sc = DEST / "Analysis/scaling"
    for f in sorted((REPO / "Analysis/scaling").glob("*.py")):
        shutil.copy2(f, sc / f.name)
    fg = DEST / "Analysis/figures"
    fg.mkdir(parents=True)
    for f in sorted((REPO / "Analysis/figures").glob("*.py")):
        shutil.copy2(f, fg / f.name)
    prompts()
    requirements()
    citation(doi)
    doi_line = (f"\nThis deposit: [doi:{doi}](https://doi.org/{doi}).\n" if doi else "")
    (DEST / "README.md").write_text(README.format(title=TITLE, doi_line=doi_line), encoding="utf-8")

    # nothing that looks like a credential may leave the machine
    bad = []
    for f in DEST.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".md", ".sh", ".txt", ".json", ".cff"} and f.stat().st_size < 5_000_000:
            t = f.read_text(encoding="utf-8", errors="ignore")
            if any(k in t for k in ("KGAT_", "KAGGLE_API_TOKEN=", "vinhdinhthien", "kaggle.json\"", "sk-proj-")):
                bad.append(str(f.relative_to(DEST)))
    if bad:
        raise SystemExit("possible credential or account name in: " + ", ".join(bad))

    files = sorted(p for p in DEST.rglob("*") if p.is_file())
    with open(DEST / "SHA256SUMS.txt", "w", encoding="utf-8", newline="\n") as out:
        for p in files:
            out.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(DEST).as_posix()}\n")

    zpath = OUT / f"{NAME}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(DEST.rglob("*")):
            if p.is_file():
                z.write(p, f"{NAME}/{p.relative_to(DEST).as_posix()}")
    raw = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    print(f"{len(files) + 1} files, {raw / 1e6:.1f} MB unpacked, {zpath.stat().st_size / 1e6:.1f} MB zipped")
    print(zpath)


if __name__ == "__main__":
    main()
