"""Turn the internal Zenodo deposit into the public release, and refuse it if a gate fails.

    python make_zenodo_release.py

reads   zenodo_upload/payoff-scaling-llm-pd-v1.0.zip           (make_zenodo_package.py)
writes  zenodo_upload/release/payoff-scaling-llm-pd-v1.0.zip   (the file to upload)

The release names no collection platform. kaggle/benchmarks/pd_task.py ships as
collection/pd_game.py with the evaluation harness swapped out: the harness import, its
re-authentication, the push-time smoke guard, the task decorator and the harness assertion
are removed, and a client for one OpenAI-compatible chat-completions endpoint (OpenRouter
by default) takes their place. Comments and docstrings, which are Vietnamese and describe
the harness, are dropped. results/kbench/ becomes results/runs/. The download and ingestion
scripts, the template-parity test (it reads the FAIRGAME repository) and the platform's
three *.run.json run records are left out; the per-game checkpoints keep every prompt,
reply, seed, token count and metered cost. The documentation is reworded to match, and
says where the article names the collection proxy (ESM, appendix A.2).

Gates on the built release:
  1. no path and no text file mentions the platform;
  2. every top-level definition of pd_game.py outside the swapped layer has the same
     syntax tree as in pd_task.py (docstrings aside), and run_sweep() differs from the
     task function only by the documented edits;
  3. replaying the recorded replies of the lambda = 1 run through a local stand-in
     endpoint reproduces those games: same prompts, seeds, actions and scores;
  4. Analysis/scaling/s15_parse_fallback.py, run inside the unpacked release, rewrites
     its two tables byte for byte;
  5. no credential pattern appears, and the written zip reads back byte for byte.
Also left out: SHA256SUMS.txt, the turns.jsonl files (they duplicate the checkpoints),
the legacy s05_figures.py and the pdlib modules that the rebuild never imports.
Nothing is uploaded.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import http.server
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tokenize
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "payoff-scaling-llm-pd-v1.0"
SRC_ZIP = HERE / "zenodo_upload" / f"{NAME}.zip"
OUT_DIR = HERE / "zenodo_upload" / "release"
OUT_ZIP = OUT_DIR / f"{NAME}.zip"
ROOT = NAME + "/"
# Short scratch path: the deepest file of the deposit is 156 characters long.
WORK = Path(os.environ.get("PD_RELEASE_WORK", "D:/tmp/pdrel"))

TASK = "kaggle/benchmarks/pd_task.py"
GAME = "collection/pd_game.py"
DROP = {"kaggle/benchmarks/collect_run.py", "kaggle/benchmarks/run_and_watch.py",
        "kaggle/benchmarks/test_pd_task_parity.py", "SHA256SUMS.txt",
        # legacy figure driver and library modules that the rebuild never imports
        "Analysis/scaling/s05_figures.py", "Analysis/pdlib/egtstyle.py", "Analysis/pdlib/metrics.py",
        "Analysis/pdlib/natstyle.py", "Analysis/pdlib/residual.py", "Analysis/pdlib/unclassified.py"}
PLATFORM = re.compile(r"kaggle|kbench", re.I)
VIETNAMESE = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)


def fail(msg: str) -> None:
    sys.exit(f"RELEASE REFUSED: {msg}")


def rename(path: str) -> str | None:
    """Release path of a deposit path, or None when the file is left out."""
    # turns.jsonl concatenates the turns of the run's checkpoints, so it is a duplicate
    if path in DROP or path.endswith(".run.json") or path.endswith("/turns.jsonl"):
        return None
    if path == TASK:
        return GAME
    if path.startswith("results/kbench/"):
        return "results/runs/" + path[len("results/kbench/"):]
    return path


def sub_once(text: str, old: str, new: str, where: str) -> str:
    n = text.count(old)
    if n != 1:
        fail(f"{where}: expected one occurrence of {old[:60]!r}, found {n}")
    return text.replace(old, new)


# ---------------------------------------------------------------- game code ---------
GAME_DOC = '''"""FAIRGAME iterated prisoner's dilemma for language-model agents, payoff-scaled.

The game code that collected the corpus of "Payoff scaling shapes cooperation in LLM
agents across languages" (Interface Focus, RSFS-2026-0050). Two agents of one model play
ten simultaneous rounds; the FAIRGAME templates in five languages state the payoff matrix
as penalties multiplied by lambda; each reply is parsed with FAIRGAME's matcher, retried
twice on a parse failure and otherwise recorded as Option A. Run `python pd_game.py`;
the PD_* environment variables select the slice of the experiment (collection/README.md).

Model calls go through one OpenAI-compatible chat-completions endpoint (OpenRouter by
default, see ROUTER_URL). Everything else is the code that played the recorded games.

Seeds: sampling seed = ((BASE + cell) * 100000 + round * 100 + agent) mod (2^31 - 1), with
cell = (language index * 4 + persona pairing) * REPS + rep, deliberately independent of
lambda so that every scale requests the same seed schedule.
"""'''

ROUTER = '''

# %% =====================  MODEL ROUTER  ===============================================
# Every model call goes to ONE OpenAI-compatible chat-completions endpoint. OpenRouter is
# the default; point PD_ROUTER_URL at a LiteLLM proxy, a vLLM server or a provider's own
# OpenAI-compatible API to use those instead.
ROUTER_URL = os.environ.get("PD_ROUTER_URL", "https://openrouter.ai/api/v1").rstrip("/")
ROUTER_KEY = (os.environ.get("PD_ROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
              or os.environ.get("OPENAI_API_KEY") or "")
ROUTER_TIMEOUT_S = float(os.environ.get("PD_ROUTER_TIMEOUT", "300"))
_IS_OPENROUTER = "openrouter.ai" in ROUTER_URL


class Usage(object):
    """Token counts of one call, and its cost in nanodollars when the router reports it."""

    def __init__(self, input_tokens=0, output_tokens=0, total_cost_nanodollars=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_cost_nanodollars = total_cost_nanodollars


def _router_chat(prompt, seed, params):
    """One chat completion. Returns (text, Usage); raises RuntimeError naming the HTTP code."""
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "seed": int(seed)}
    body.update(params)
    if _IS_OPENROUTER:
        body["usage"] = {"include": True}
    request = urllib.request.Request(
        ROUTER_URL + "/chat/completions", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + ROUTER_KEY})
    try:
        with urllib.request.urlopen(request, timeout=ROUTER_TIMEOUT_S) as response:
            reply = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError("HTTP %d: %s" % (exc.code, detail[:300]))
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError("503 connection error: %s" % exc)
    if "error" in reply and not reply.get("choices"):
        raise RuntimeError("router error: %s" % str(reply["error"])[:300])
    choice = (reply.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    usage = reply.get("usage") or {}
    return text, Usage(input_tokens=usage.get("prompt_tokens") or 0,
                       output_tokens=usage.get("completion_tokens") or 0,
                       total_cost_nanodollars=int(round((usage.get("cost") or 0) * 1e9)))


def _call_llm(prompt, seed, max_attempts=6):
    """One free-text generation. Returns (text, usage).

    Out of credit ends the sweep (QuotaExhausted; finished games stay checkpointed). An
    optional parameter the provider refuses is dropped once for the whole run. 429 and 5xx
    replies back off with jitter. Any other error ends the call loudly.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            params = {"max_tokens": MAX_OUTPUT_TOKENS}
            if not _NO_TOOL_CHOICE.is_set():
                params["tool_choice"] = "none"
            if REASONING_EFFORT and not _NO_REASONING_PARAM.is_set():
                params["reasoning_effort"] = REASONING_EFFORT
            return _router_chat(prompt, seed, params)
        except Exception as e:
            msg = str(e).lower()
            if any(t in msg for t in _QUOTA_ERR):
                raise QuotaExhausted(str(e)) from e
            _param_status = _drop_optional_param(msg)
            if _param_status == "dropped":
                attempt -= 1
                continue
            if _param_status == "stale":
                if attempt >= max_attempts:
                    raise
                time.sleep(min(2 ** attempt, 12))
                continue
            transient = any(t in msg for t in _TRANSIENT)
            budget = TRANSIENT_MAX_ATTEMPTS if transient else max_attempts
            if attempt >= budget or not transient:
                raise
            time.sleep(min(2 ** attempt, 45) * (0.5 + random.random()))
'''

ASSERT_NEW = '''    if fallback_rate > FALLBACK_RATE_TOLERANCE:
        raise SystemExit(f"check failed: fallback rate {fallback_rate:.4f} > "
                         f"{FALLBACK_RATE_TOLERANCE}; the replies were not parsed")
    return result'''


def node_names(node) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {t.id for t in targets if isinstance(t, ast.Name)}
    return set()


def node_span(node) -> tuple[int, int]:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return start, node.end_lineno


def edit_task_function(seg: str) -> str:
    """The task function with the harness edits applied, as run_sweep()."""
    seg = re.sub(r"^@kbench\.task\(.*?^\)\n", "", seg, flags=re.S | re.M)
    seg = sub_once(seg, "def prisoner_dilemma_fairgame(llm) -> dict:", "def run_sweep() -> dict:", "task fn")
    seg = sub_once(seg, "    global _LLM\n    _LLM = llm\n", "", "task fn")
    seg = sub_once(seg, 'f"results/kbench/{model_tag}"', 'f"results/runs/{model_tag}"', "task fn")
    tail = re.search(r"    kbench\.assertions\.assert_true\(.*?\n    return result", seg, re.S)
    if not tail:
        fail("task fn: harness assertion not found")
    return seg[:tail.start()] + ASSERT_NEW + seg[tail.end():]


def swap_harness(src: str) -> str:
    """pd_task.py with the harness layer replaced; comments still in place."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    cut: list[tuple[int, int, str]] = []  # (start, end, replacement), 1-based inclusive
    smoke = {"_force_full", "_has_overrides", "_slug", "_SMOKE_ONLY", "PROBE_ALL_MODELS"}
    for node in tree.body:
        s, e = node_span(node)
        seg = "".join(lines[s - 1:e])
        names = node_names(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node is tree.body[0]:
            cut.append((s, e, GAME_DOC + "\n"))
        elif isinstance(node, ast.Import) and [a.name for a in node.names] == ["subprocess"]:
            cut.append((s, e, ""))
        elif isinstance(node, ast.Import) and [a.name for a in node.names] == ["time"]:
            cut.append((s, e, seg + "import urllib.error\nimport urllib.request\n"))
        elif isinstance(node, ast.Import) and any(a.name == "kaggle_benchmarks" for a in node.names):
            cut.append((s, e, ""))
        elif names == {"MODEL"}:
            cut.append((s, e, 'MODEL = os.environ.get("PD_MODEL") or "google/gemini-3.5-flash-lite"\n'))
        elif isinstance(node, ast.Assign) and "LLM_DEFAULT" in seg:
            cut.append((s, e, ""))
        elif names & smoke or (isinstance(node, ast.If) and "PROBE_ALL_MODELS" in seg):
            cut.append((s, e, ""))
        elif names & {"_LLM", "_LLM_LOCK", "_reauth"}:
            cut.append((s, e, ""))
        elif names == {"_call_llm"}:
            cut.append((s, e, ROUTER.lstrip("\n")))
        elif names == {"prisoner_dilemma_fairgame"}:
            cut.append((s, e, edit_task_function(seg)))
        elif isinstance(node, ast.If) and "PD_SKIP_RUN" in seg:
            cut.append((s, e, 'if __name__ == "__main__":\n    run_sweep()\n'))
    out, prev = [], 1
    for s, e, rep in sorted(cut):
        out.append("".join(lines[prev - 1:s - 1]))
        out.append(rep)
        prev = e + 1
    out.append("".join(lines[prev - 1:]))
    return "".join(out)


def strip_comments(src: str) -> str:
    """Drop every comment; a line that held only a comment goes with it."""
    cut: dict[int, int] = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            cut[tok.start[0]] = tok.start[1]
    kept = []
    for i, line in enumerate(src.splitlines(keepends=True), 1):
        if i in cut:
            head = line[:cut[i]].rstrip()
            if not head:
                continue
            line = head + "\n"
        kept.append(line)
    return "".join(kept)


# English docstrings for definitions whose whole body is a Vietnamese docstring
DOC_EN = {"QuotaExhausted": "Out of credit: retrying cannot help. Change the key and resume "
                            "from the checkpoints."}


def strip_foreign_docstrings(src: str) -> str:
    """Drop docstrings that are Vietnamese or name the platform; keep English ones."""
    tree = ast.parse(src)
    edits = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and isinstance(body, list) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            doc = body[0].value.value
            if VIETNAMESE.search(doc) or PLATFORM.search(doc):
                new = ""
                if len(body) == 1:
                    if node.name not in DOC_EN:
                        fail(f"{node.name}: docstring is the whole body and has no English text")
                    new = " " * body[0].col_offset + f'"""{DOC_EN[node.name]}"""\n'
                edits.append((body[0].lineno, body[0].end_lineno, new))
    lines = src.splitlines(keepends=True)
    for s, e, new in sorted(edits, reverse=True):
        lines[s - 1:e] = [new] if new else []
    return "".join(lines)


def tidy(src: str) -> str:
    src = re.sub(r"\n{3,}", "\n\n", src)
    src = re.sub(r"\n+(?=(def |class |@|if __name__|# %%))", "\n\n\n", src)
    return src.strip("\n") + "\n"


# Runtime log and error messages, translated; only these string fragments change.
MESSAGES = [
    ("PD_WEIGHT_FORMAT không hợp lệ: ", "invalid PD_WEIGHT_FORMAT: "),
    ("(chỉ nhận 'native' hoặc 'decN')", "(accepted: 'native' or 'decN')"),
    ("[warn] provider từ chối `reasoning_effort` -> bỏ tham số này. ",
     "[warn] the provider refused `reasoning_effort` -> no longer sending it. "),
    ("CẢNH BÁO: model thinking có thể ăn hết max_tokens rồi trả text ",
     "WARNING: a thinking model may spend max_tokens on reasoning and return truncated "),
    ("cụt -> kiểm tra fallback_rate cuối run.", "text -> check fallback_rate at the end of the run."),
    ("[warn] provider từ chối `tool_choice` -> bỏ tham số này. Model có thể ",
     "[warn] the provider refused `tool_choice` -> no longer sending it. The model may "),
    ("trả tool_call rỗng; nhánh 'SDK tool-call parse bug' bên dưới đỡ.",
     "return an empty tool call, which is parsed as a failed reply and retried."),
    ("[checkpoint] bỏ shard không khớp signature: ",
     "[checkpoint] skipping a shard whose signature does not match: "),
    ('"thiếu lượt"', '"missing turns"'),
    ("[checkpoint] shard hỏng ", "[checkpoint] unreadable shard "),
    ("tổ hợp \"", "pairings \""),
    ("vòng × 2 agent = ", "rounds × 2 agents = "),
    ("lượt gọi model (concurrency=", "model calls (concurrency="),
    ("[resume] khôi phục ", "[resume] restored "),
    (" game từ {ckpt_dir}", " games from {ckpt_dir}"),
    ("[warmup] 1 lượt gọi tuần tự OK -> mở pool.", "[warmup] one sequential call OK -> opening the pool."),
    ("[warmup] lượt gọi khởi động lỗi (", "[warmup] the warm-up call failed ("),
    ("vẫn mở pool vì retry trong _call_llm có thể qua được.",
     "opening the pool anyway, since the retries in _call_llm may succeed."),
    ("[quota] DỪNG SỚM — hết credit: ", "[quota] stopped early, out of credit: "),
    ("[quota] Đã giữ ", "[quota] kept "),
    (" game trong {ckpt_dir}. ", " games in {ckpt_dir}. "),
    ("Đổi sang API key khác rồi chạy lại đúng lệnh này để resume.",
     "Switch to another API key and rerun the same command to resume."),
]


def translate(src: str) -> str:
    for old, new in MESSAGES:
        src = sub_once(src, old, new, "message")
    return src


def build_game(task_src: str) -> str:
    # comments go first, so the English comments of the new client layer survive
    return tidy(strip_foreign_docstrings(swap_harness(translate(strip_comments(task_src)))))


def _bare(tree):
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


SWAPPED = {"MODEL", "_force_full", "_has_overrides", "_slug", "_SMOKE_ONLY", "PROBE_ALL_MODELS",
           "_LLM", "_LLM_LOCK", "_reauth", "_call_llm", "prisoner_dilemma_fairgame", "run_sweep",
           "ROUTER_URL", "ROUTER_KEY", "ROUTER_TIMEOUT_S", "_IS_OPENROUTER", "Usage", "_router_chat"}


def gate_same_logic(task_src: str, game_src: str) -> int:
    # the original with only the documented message translations applied
    task_src = translate(strip_comments(task_src))

    def kept(src):
        out = []
        for node in _bare(ast.parse(src)).body:
            seg_names = node_names(node)
            dump = ast.dump(node)
            if seg_names & SWAPPED:
                continue
            if isinstance(node, ast.Import) and any(a.name in ("kaggle_benchmarks", "subprocess", "urllib.error",
                                                               "urllib.request") for a in node.names):
                continue
            if isinstance(node, ast.Assign) and "LLM_DEFAULT" in dump:
                continue
            if isinstance(node, ast.If) and ("PROBE_ALL_MODELS" in dump or "PD_SKIP_RUN" in dump
                                             or "__main__" in dump):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            out.append((sorted(seg_names), dump))
        return out
    a, b = kept(task_src), kept(game_src)
    if a != b:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                fail(f"game logic differs at top-level definition {x[0] or i} vs {y[0] or i}")
        fail(f"top-level definition count differs: {len(a)} vs {len(b)}")
    # run_sweep is the task function plus exactly the documented edits
    t = ast.parse(task_src)
    fn = next(n for n in t.body if node_names(n) == {"prisoner_dilemma_fairgame"})
    s, e = node_span(fn)
    want = _bare(ast.parse(edit_task_function("".join(task_src.splitlines(keepends=True)[s - 1:e]))))
    got = next(n for n in _bare(ast.parse(game_src)).body if node_names(n) == {"run_sweep"})
    if ast.dump(want.body[0]) != ast.dump(got):
        fail("run_sweep differs from the task function beyond the documented edits")
    return len(a)


# ---------------------------------------------------------------- documentation -----
def edit_readme(t: str) -> str:
    t = re.sub(r"^\| `results/kbench/` \|.*$",
               "| `results/runs/` | raw output of the three archived collection runs of Gemini 3.5 "
               "Flash-Lite: per-game CSVs and **one JSON record per game holding every prompt, raw "
               "reply, requested sampling seed, parse-retry count, fallback flag, token counts and "
               "metered cost** |", t, count=1, flags=re.M)
    t = re.sub(r"^\| `kaggle/benchmarks/` \|.*$",
               "| `collection/` | the game code that collected the corpus (`pd_game.py`), with its model "
               "client set to any OpenAI-compatible endpoint; see `collection/README.md` |",
               t, count=1, flags=re.M)
    t = sub_once(t, "`Dataset/` and `results/kbench/`", "`Dataset/` and `results/runs/`", "README.md")
    t = sub_once(t, "deposit is 156 characters", "deposit is 154 characters", "README.md")
    t = sub_once(t, "python Analysis/scaling/rebuild_interface_focus.py --skip-egt # if egttools cannot be installed\n",
                 "python Analysis/scaling/rebuild_interface_focus.py --skip-egt # if egttools cannot be installed\n"
                 "python Analysis/scaling/rebuild_interface_focus.py --retrain-classifier  "
                 "# also retrain the strategy classifier\n", "README.md")
    t = sub_once(t, "regenerates every table and figure,\nand ends with",
                 "regenerates every table and figure\n(written to `papers/interface-focus/`, or to "
                 "`PD_PAPER_DIR` when set), and ends with", "README.md")
    return sub_once(t, "\n## Integrity\n\n`SHA256SUMS.txt` lists a checksum for every file. "
                       "Verify with `sha256sum -c SHA256SUMS.txt`.\n", "", "README.md")


def edit_card(t: str) -> str:
    w = "DATA_CARD"
    t = sub_once(t, "produced by the Kaggle Benchmarks task\n`prisoner-dilemma-fairgame` (`kaggle/benchmarks/pd_task.py`)",
                 "produced by the collection task\n`prisoner-dilemma-fairgame` (`collection/pd_game.py`)", w)
    t = sub_once(t, "inlined into\n`kaggle/benchmarks/pd_task.py`)", "inlined into\n`collection/pd_game.py`)", w)
    t = sub_once(t, "> Historical trap, still worth knowing: `pd_task.py` itself hard-coded",
                 "> Historical trap, still worth knowing: the collection task itself hard-coded", w)
    t = sub_once(t, "printed in old Kaggle run logs and", "printed in old run logs and", w)
    t = sub_once(t, "is enforced byte-identical by\n   `kaggle/benchmarks/test_pd_task_parity.py`, and is",
                 "is enforced byte-identical by\n   a template-parity test at collection time, and is", w)
    t = sub_once(t, "* **Generator:** `kaggle/benchmarks/pd_task.py`, run server-side as the Kaggle Benchmarks\n"
                    "  task `prisoner-dilemma-fairgame`.",
                 "* **Generator:** the collection task `prisoner-dilemma-fairgame`, released as\n"
                 "  `collection/pd_game.py` and run through a hosted model proxy (ESM, appendix A.2).", w)
    t = sub_once(t, "enforced by\n  `kaggle/benchmarks/test_pd_task_parity.py`.",
                 "enforced by\n  a template-parity test at collection time.", w)
    t = sub_once(t, "* **Ingestion into `Dataset/`:** `kaggle/benchmarks/collect_run.py`, which rejects any",
                 "* **Ingestion into `Dataset/`:** an ingestion script, which rejects any", w)
    t = sub_once(t, "This matters because Kaggle mounts the previous run's",
                 "This matters because the collection platform mounts the previous run's", w)
    # internal wording and corpora that are not part of the deposit
    t = sub_once(t, "every figure and table in\nthe manuscripts is built on it.",
                 "every figure and table in\nthe article is built on it.", w)
    t = sub_once(t, "and the manuscripts keep it in", "and the article keeps it in", w)
    t = sub_once(t, "the\n   manuscript's primary intervals", "the\n   article's primary intervals", w)
    t = sub_once(t, "`data_fairgame_small_llm`). Omitting the argument loads both.",
                 "`data_fairgame_small_llm`, which is not part of this deposit).", w)
    table = re.search(r"\| `data_fairgame_frontier_llm` \| \*\*this corpus\*\*.*?\n\n", t, re.S)
    if not table:
        fail(f"{w}: related-corpora table not found")
    t = t[:table.start()] + (
        "| `data_fairgame_frontier_llm` | **this corpus**: 6 frontier models, 10 scales, PD | primary |\n"
        "| `data_replicate_frontier` | test-retest replicate of a subset, different `BASE_SEED` and a "
        "`RUN_TAG` suffix | reliability estimate (ESM D.5) |\n"
        "| `data_fairgame_e2_notation` | payoff numbers written in a different notation, same games "
        "| notation control (ESM D.5) |\n"
        "| `noise_dataset` | synthetic strategy-labelled games | trains and validates the strategy "
        "classifier |\n\n") + t[table.end():]
    return t


def edit_s15(t: str) -> str:
    w = "s15_parse_fallback.py"
    t = sub_once(t, "checkpoints that the Kaggle Benchmarks task writes do",
                 "checkpoints that the collection task writes do", w)
    t = sub_once(t, "under results/kbench, so the table grows as further\n"
                    "runs are downloaded with kaggle/benchmarks/run_and_watch.py.",
                 "under results/runs, so the table grows as further\nruns are added there.", w)
    t = sub_once(t, 'KBENCH = REPO / "results" / "kbench"', 'RUNS = REPO / "results" / "runs"', w)
    return sub_once(t, "KBENCH.glob(", "RUNS.glob(", w)


def edit_rebuild(t: str) -> str:
    """Use the shipped classifier unless asked to retrain, and create the output folder
    first (s09 and s14 write into it without creating it)."""
    w = "rebuild_interface_focus.py"
    t = sub_once(t, "import subprocess\n", "import os\nimport subprocess\n", w)
    t = sub_once(t, '        help="reuse committed EGT outputs when egttools is unavailable",\n    )\n',
                 '        help="reuse committed EGT outputs when egttools is unavailable",\n    )\n'
                 "    parser.add_argument(\n"
                 '        "--retrain-classifier", action="store_true",\n'
                 '        help="retrain the strategy classifier instead of using the shipped "\n'
                 '             "Analysis/scaling/models/strategy_lstm.pt",\n'
                 "    )\n", w)
    t = sub_once(t, '    for name in ("s00_build.py", "s01_train_lstm.py", "s02_readout.py",\n'
                    '                 "s03_stats.py", "s04_strategy_stats.py"):\n'
                    "        run(scaling / name)\n",
                 '    steps = ["s00_build.py", "s02_readout.py", "s03_stats.py", "s04_strategy_stats.py"]\n'
                 "    if args.retrain_classifier:\n"
                 '        steps.insert(1, "s01_train_lstm.py")\n'
                 "    for name in steps:\n"
                 "        run(scaling / name)\n", w)
    return sub_once(t, "    args = parser.parse_args()\n",
                    "    args = parser.parse_args()\n"
                    "    # Generated LaTeX tables and figures go to PD_PAPER_DIR "
                    "(default papers/interface-focus).\n"
                    '    paper = Path(os.environ.get("PD_PAPER_DIR", REPO / "papers" / "interface-focus"))\n'
                    "    paper.mkdir(parents=True, exist_ok=True)\n", w)


def edit_requirements(t: str) -> str:
    return sub_once(t, "# collection only\nkaggle-benchmarks==0.6.1\n", "", "requirements.txt")


RESULTS_README = """# Collection runs

Raw output of the three archived collection runs of Gemini 3.5 Flash-Lite, as written by
the game code in `collection/pd_game.py`. This is the unprocessed record: the CSVs that
`Dataset/` was built from, and one JSON checkpoint per game holding the full transcript.

## Layout

```
results/runs/exp<E>/run<ID>/<model_tag>/
├── <lambda>/<model_tag>/x<lambda>_<lang>_<model_tag>.csv   # one row per dyad
└── checkpoints/x<lambda>__lang-<l>__p<persona>__rep-<n>.json
```

| exp | run id | model tag | payoff scales λ | role |
|---|---|---|---|---|
| 2 | 359522 | `gemini-3.5-flash-lite` | 0.1, 1, 10 | main grid |
| 2 | 971021 | `gemini-3.5-flash-lite` | 0.01, 0.25, 0.5, 2, 5, 100, 1000 | main grid |
| 4 | 1029391 | `gemini-3.5-flash-lite-rep2` | 0.1, 0.25, 0.5, 1 | independent repeat (ESM D.5) |

Each λ directory holds 200 dyads: five languages by forty games. The two main-grid runs
together hold the 2,000 Gemini 3.5 Flash-Lite dyads of `Dataset/data_fairgame_frontier_llm`;
in `Dataset/` the `-rep2` suffix of the repeat run is stripped from directory names, file
names and the `agent1_llm` / `agent2_llm` columns.

## Checkpoint records

Each checkpoint holds `signature` (the settings a resumed run must match), `condition_key`,
`game` (the CSV row), `turns` and `stats`. One entry of `turns` per decision gives
`game_id`, `round`, `agent`, `personality`, `strategy`, `parse_failed`, `fell_back`,
`raw_response`, `prompt`, `sampling_seed`, `language`, `scale`, `rep` and `perm_idx`.
`stats` sums the game's `parse_failed`, `fell_back`, input tokens (`tok_in`), output
tokens (`tok_out`) and metered cost (`cost`, in nanodollars). Summing `cost` over the
2,000 main-grid games gives the US$5.54 reported in ESM appendix A.2.

The game code also writes `turns.jsonl` next to `checkpoints/`, the turns of every
checkpoint in one file; it is left out here because it duplicates the checkpoints.
"""

CODE_README = """# Game code

`pd_game.py` is the game code that collected the corpus. It builds each agent's prompt
from the FAIRGAME prisoner's-dilemma templates in five languages (inlined byte for byte),
scales the payoff matrix by λ, parses each free-text reply with FAIRGAME's matcher,
retries twice on a parse failure and otherwise records Option A, scores the round, and
checkpoints every finished game so that an interrupted sweep resumes where it stopped.
It needs only the Python standard library.

Models are called through one OpenAI-compatible chat-completions endpoint, so a single key
reaches every provider. The default is [OpenRouter](https://openrouter.ai); set
`PD_ROUTER_URL` to use a LiteLLM proxy, a vLLM server or a provider's own
OpenAI-compatible API instead.

```sh
export OPENROUTER_API_KEY=...                  # or PD_ROUTER_KEY for another endpoint
PD_MODEL=google/gemini-3.5-flash-lite python pd_game.py
PD_LAMBDAS=1 PD_LANGS=en PD_REPS=1 python pd_game.py      # a 4-game smoke test first
```

| Variable | Meaning | Default |
|---|---|---|
| `PD_MODEL` | model, as the endpoint names it | `google/gemini-3.5-flash-lite` |
| `PD_ROUTER_URL` | OpenAI-compatible base URL | `https://openrouter.ai/api/v1` |
| `PD_ROUTER_KEY` | API key (else `OPENROUTER_API_KEY`, else `OPENAI_API_KEY`) | |
| `PD_LAMBDAS` | payoff scales | `0.01,0.1,0.25,0.5,1,2,5,10,100,1000` |
| `PD_LANGS` | prompt languages | `en,fr,ar,cn,vn` |
| `PD_REPS` | games per (scale, language, persona pairing) | `10` |
| `PD_ROUNDS` | rounds per game | `10` |
| `PD_TEMPERATURE` | sampling temperature | `1.0` |
| `PD_BASE_SEED` | base of the seed schedule | `12345` |
| `PD_MAX_OUTPUT_TOKENS` | output cap per reply | `128` |
| `PD_REASONING_EFFORT` | sent as `reasoning_effort` when set | not sent |
| `PD_CONCURRENCY` | games played in parallel; each game is sequential, so results do not depend on it | `8` |
| `PD_OUT` | output folder | `results/runs/<model tag>` |
| `PD_RESUME` | reuse finished games from the checkpoints | `1` |

The output has the layout of `results/runs/` (see `results/README.md`). At the end of a
sweep the code prints a summary and stops with an error if more than 2% of decisions fell
back to Option A, since systematic fallback would bias the cooperation rate.
"""


# ---------------------------------------------------------------- gates ----------------
class _StandIn(http.server.BaseHTTPRequestHandler):
    replies: dict = {}
    by_prompt: dict = {}
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        text = self.replies.get((prompt, body["seed"])) or self.by_prompt.get(prompt, "")
        type(self).seen.append((prompt, body["seed"]))
        out = json.dumps({"choices": [{"message": {"content": text}}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


def gate_replay(root: Path) -> int:
    """Replay the recorded lambda = 1 replies; the games must come out as recorded."""
    run = root / "results/runs/exp2/run359522/gemini-3.5-flash-lite"
    records = {}
    for f in sorted((run / "checkpoints").glob("x1__*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec["turns"][0]["rep"] in (0, 1):
            records[rec["game"]["game_id"]] = rec
            for t in rec["turns"]:
                _StandIn.replies[(t["prompt"], t["sampling_seed"])] = t["raw_response"]
                _StandIn.by_prompt[t["prompt"]] = t["raw_response"]
    if len(records) != 40:
        fail(f"replay: expected 40 recorded games at lambda = 1, found {len(records)}")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StandIn)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    out = WORK / "replay_out"
    shutil.rmtree(out, ignore_errors=True)
    # REPS stays 10 because it enters the seed formula; the sweep is narrowed from outside
    # to reps 0 and 1 of every language and pairing, without touching pd_game.py.
    env = dict(os.environ, PD_ROUTER_URL=f"http://127.0.0.1:{server.server_port}",
               PD_ROUTER_KEY="stand-in", PD_LAMBDAS="1", PD_REPS="10", PD_CONCURRENCY="1",
               PD_OUT=str(out), PD_RESUME="0", PYTHONUTF8="1")
    driver = (
        "import runpy\n"
        "g = runpy.run_path(r'%s', run_name='pd_game')\n"
        "live = g['run_sweep'].__globals__\n"
        "full = live['_sweep_order']\n"
        "live['_sweep_order'] = lambda: [c for c in full() if c[3] in (0, 1)]\n"
        "g['run_sweep']()\n" % (root / GAME))
    proc = subprocess.run([sys.executable, "-c", driver], cwd=WORK, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    server.shutdown()
    if proc.returncode:
        fail(f"replay run failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    got = {}
    for f in (out / "checkpoints").glob("*.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        got[rec["game"]["game_id"]] = rec
    if set(got) != set(records):
        fail(f"replay: game ids differ ({len(got)} played, {len(records)} recorded)")
    for gid, rec in records.items():
        g = got[gid]
        for k in ("agent1_strategies", "agent2_strategies", "agent1_scores", "agent2_scores"):
            if str(g["game"][k]) != str(rec["game"][k]):
                fail(f"replay {gid}: {k} differs")
        mine = [(t["round"], t["agent"], t["prompt"], t["sampling_seed"]) for t in g["turns"]]
        theirs = [(t["round"], t["agent"], t["prompt"], t["sampling_seed"]) for t in rec["turns"]]
        if mine != theirs:
            fail(f"replay {gid}: prompts or seeds differ")
        if g["signature"] != rec["signature"]:
            fail(f"replay {gid}: signature differs")
    return len(records)


def gate_s15(root: Path) -> None:
    tab = root / "Analysis/scaling/tables"
    names = ["T27_parse_fallback.csv", "T27_parse_fallback_summary.csv"]
    before = {n: (tab / n).read_bytes() for n in names}
    proc = subprocess.run([sys.executable, str(root / "Analysis/scaling/s15_parse_fallback.py")],
                          cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode:
        fail(f"s15 failed in the release:\n{proc.stderr[-1500:]}")
    for n in names:
        if (tab / n).read_bytes() != before[n]:
            fail(f"s15 in the release rewrote {n} differently")


SECRET = re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}|\bgh[pousr]_[A-Za-z0-9]{30,}|\bhf_[A-Za-z0-9]{30,}"
                    r"|\bAIza[0-9A-Za-z_\-]{35}|\bAKIA[0-9A-Z]{16}|\"key\"\s*:\s*\"[0-9a-f]{32}\"")


def main() -> None:
    if not SRC_ZIP.exists():
        fail(f"{SRC_ZIP} not found; run make_zenodo_package.py first")
    src = zipfile.ZipFile(SRC_ZIP)
    files: dict[str, bytes] = {}
    for info in src.infolist():
        if info.is_dir():
            continue
        rel = info.filename[len(ROOT):]
        new = rename(rel)
        if new is not None:
            files[new] = src.read(info)

    def lf(data: bytes) -> str:  # the files edited here are matched and written with \n
        return data.decode("utf-8").replace("\r\n", "\n")

    task_src = lf(src.read(ROOT + TASK))
    game_src = build_game(task_src)
    files[GAME] = game_src.encode("utf-8")
    files["collection/README.md"] = CODE_README.encode("utf-8")
    files["results/README.md"] = RESULTS_README.encode("utf-8")
    for path, fn in (("README.md", edit_readme), ("Dataset/DATA_CARD_frontier_llm.md", edit_card),
                     ("Analysis/scaling/s15_parse_fallback.py", edit_s15),
                     ("Analysis/scaling/rebuild_interface_focus.py", edit_rebuild),
                     ("requirements.txt", edit_requirements)):
        files[path] = fn(lf(files[path])).encode("utf-8")

    # gate 1 and 5: no platform name, no credential
    for p, d in files.items():
        if PLATFORM.search(p):
            fail(f"path names the platform: {p}")
        if b"\0" in d[:4096]:
            continue
        text = d.decode("utf-8", "replace")
        m = PLATFORM.search(text)
        if m:
            fail(f"{p} mentions the platform: ...{text[max(0, m.start() - 60):m.end() + 60]!r}...")
        if SECRET.search(text):
            fail(f"{p} contains a credential pattern")
    print(f"gate 1/5: {len(files)} files, no platform name, no credential pattern")

    n = gate_same_logic(task_src, game_src)
    print(f"gate 2: {n} top-level definitions outside the client layer have the task's syntax tree; "
          "run_sweep differs only by the documented edits")

    shutil.rmtree(WORK, ignore_errors=True)
    root = WORK / NAME
    for p, d in files.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_bytes(d)
    games = gate_replay(root)
    print(f"gate 3: replaying the recorded replies reproduced {games} games "
          "(prompts, seeds, actions, scores)")
    gate_s15(root)
    print("gate 4: s15_parse_fallback.py rewrites T27 byte for byte inside the release")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p, d in files.items():
            z.writestr(ROOT + p, d)
    z = zipfile.ZipFile(OUT_ZIP)
    if z.testzip() is not None or {i.filename[len(ROOT):] for i in z.infolist()} != set(files):
        fail("the written zip does not hold exactly the release files")
    bad = sum(hashlib.sha256(z.read(ROOT + p)).digest() != hashlib.sha256(d).digest()
              for p, d in files.items())
    if bad:
        fail(f"{bad} files differ after zipping")
    print(f"gate 5: the zip reads back byte for byte\nwrote {OUT_ZIP} ({OUT_ZIP.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
