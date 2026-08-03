"""
=====================================================================
FAIRGAME - Kaggle Offline Notebook (Internet OFF, GPU ON)
=====================================================================
Chạy FAIRGAME với local LLM trên GPU Kaggle (RTX 6000, 96GB VRAM).

CÁCH CHẠY:
  1. Tạo notebook mới trên Kaggle
     - GPU: ON (chọn RTX 6000 / GPU accelerator)
     - Internet: OFF

  2. Add Input (nút "+ Add Input" bên phải):
     a) Code: gắn notebook/repo — toàn bộ source là folder read-only (không cần zip).
        Path mặc định: .../fairgame-hackathon-research/ (sửa KAGGLE_CODE_INPUT ở Cell 3 nếu khác).
        Nên dùng bản repo có ``offline_patch_assets/`` và ``kaggle_game_round_patch.b64``
        (Cell 3 tự ghi đè ``src/game_round.py`` nếu dataset cũ còn ``retry``).
     b) Model: HuggingFace model — mount tại /kaggle/input/<tên-model>/

  3. Copy nội dung file này vào notebook, chia cells theo "# CELL X"

  4. Sửa MODEL_PATH ở Cell 1 cho đúng path model của bạn

  5. Chạy theo thứ tự Cell 1 → 2 → 3 → … (Cell 2 định nghĩa ensure_src_importable cho import src.*).

Input paths trên Kaggle:
  /kaggle/input/notebooks/trungkiet/git-fairgame/fairgame-hackathon-research/  ← code (folder)
  /kaggle/input/<tên-model>/            ← model weights
=====================================================================
"""

# =====================================================================
# CELL 1: Cấu hình - SỬA Ở ĐÂY
# =====================================================================

# === Path tới model HuggingFace (đã add làm input trên Kaggle) ===
# Chạy "!ls /kaggle/input/" để xem path thực tế, rồi điền vào đây.
# Ví dụ format Kaggle: /kaggle/input/models/<org>/<model>/transformers/<variant>/<version>
MODEL_PATH = "/kaggle/input/datasets/foundnotkiet/llama-3-1-8b/model_weights"
# MODEL_PATH = "/kaggle/input/models/google/gemma-2/transformers/gemma-2-9b-it/1"
# MODEL_PATH = "/kaggle/input/models/google/gemma-2/transformers/gemma-2-27b-it/1"
# MODEL_PATH = "/kaggle/input/models/qwen/qwen2.5/transformers/7b-instruct/1"

# ⚠️ TIP: Nên dùng model INSTRUCT (vd: gemma-7b-it, gemma-2-9b-it) thay vì base model.
# Model instruct hiểu prompt tốt hơn → trả lời game theory chính xác hơn.

# === Engine chạy LLM ===
ENGINE = "transformers"  # "transformers" (ổn định trên Kaggle) hoặc "vllm" (nếu cài được)

# === Tham số model ===
MAX_MODEL_LEN = 4096   # Context length tối đa
TEMPERATURE = 1.0      # Sampling temperature
MAX_TOKENS = 512       # Số token output tối đa mỗi response
GPU_UTIL = 0.90        # % GPU memory sử dụng

# === Batched runner (tận dụng RTX 6000 96GB) ===
# True  = chạy tất cả games (langs × personalities) trong lockstep, gọi LLM theo batch
#         → tốc độ tăng 5-15× trên transformers/vLLM với cùng workload
# False = chạy tuần tự game-by-game (legacy, dùng cho API connectors như Anthropic)
USE_BATCHED_RUNNER = True
# Cap số prompt/batch để tránh OOM. 0 = một batch chứa tất cả games của 1 rep.
# Với llama-3-1-8b bf16 + RTX 6000 96GB: 0 (≈40 prompts) chạy ổn.
# Nếu OOM → giảm xuống 16 hoặc 8.
BATCH_SIZE = 0
# Retry batches cho prompts trả về response không khớp strategy nào (rồi fallback).
BATCH_STRATEGY_RETRIES = 2

# === Tên ngắn của model dùng cho filename output ===
# Output sẽ ra dạng paper: results/<lambda>/<short>/x<lambda>_<lang>_<short>.csv
# vd: "qwen3_8b" → results/1/qwen3_8b/x1_en_qwen3_8b.csv
MODEL_SHORT_NAME = "llama-3-1-8b"

# === Log verbosity ===
# Mặc định tắt log prompt/response ồn ào trong src/; bật = "1" khi cần debug.
FAIRGAME_VERBOSE_LOGS = "0"

# =====================================================================
# CELL 2: Chuẩn bị (Internet OFF — không pip) + path cho import src.*
# =====================================================================
import os
import sys
from pathlib import Path

# Force exact verbosity setting from Cell 1 (avoid inheriting stale env="1").
os.environ["FAIRGAME_VERBOSE_LOGS"] = FAIRGAME_VERBOSE_LOGS

WORK_COPY = Path("/kaggle/working/FAIRGAME")
MARKER_ROOT = Path("/kaggle/working/.fairgame_project_root")


def ensure_src_importable():
    """chdir + sys.path tới thư mục có src/ (marker từ Cell 3 hoặc quét WORK_COPY)."""
    if MARKER_ROOT.exists():
        root = Path(MARKER_ROOT.read_text(encoding="utf-8").strip())
    else:
        root = None
        for c in (WORK_COPY, WORK_COPY / "FAIRGAME", WORK_COPY / "fairgame"):
            if c.is_dir() and (c / "src").is_dir():
                root = c
                break
        if root is None and WORK_COPY.is_dir():
            for ch in WORK_COPY.iterdir():
                if ch.is_dir() and (ch / "src").is_dir():
                    root = ch
                    break
        if root is None:
            raise RuntimeError(
                "Không thấy src/: chạy Cell 3 trước (copy repo), hoặc kiểm tra tree dưới "
                f"{WORK_COPY}."
            )
    os.chdir(root)
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


print("Internet OFF: dùng thư viện có sẵn trên image Kaggle.")

# =====================================================================
# CELL 3: Setup FAIRGAME source code
# =====================================================================
import os
import shutil
import sys
from pathlib import Path

# Folder code đã add làm input (read-only) — copy sang /kaggle/working để chạy ghi file.
# Toàn bộ repo đặt tại path này; không cần zip. Nếu src/ nằm trong con FAIRGAME/: thêm / "FAIRGAME".
KAGGLE_CODE_INPUT = Path(
    "/kaggle/input/notebooks/trungkiet/git-fairgame/fairgame-hackathon-research"
)

FAIRGAME_WORK = Path("/kaggle/working/FAIRGAME")
MARKER_ROOT = Path("/kaggle/working/.fairgame_project_root")


def resolve_fairgame_root(base: Path) -> Path:
    """Thư mục gốc có package src/ (repo có thể là base hoặc base/FAIRGAME/)."""
    candidates = [base, base / "FAIRGAME", base / "fairgame"]
    for c in candidates:
        if c.is_dir() and (c / "src").is_dir():
            return c.resolve()
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "src").is_dir():
                return child.resolve()
    hint = [p.name for p in base.iterdir()] if base.is_dir() else []
    raise FileNotFoundError(
        f"Không thấy src/ dưới {base}. Các mục con: {hint}"
    )


def apply_game_round_no_retry_patch(project_root: Path) -> None:
    """
    Ghi đè src/game_round.py bản không cần PyPI ``retry``.
    Dataset Kaggle Input đôi khi là snapshot cũ vẫn còn ``from retry import retry``;
    patch này chạy sau copy để luôn đúng dù không thêm internet/re-sync dataset.

    Thứ tự: (1) ``offline_patch_assets/game_round.py`` nếu có;
    (2) ``kaggle_game_round_patch.b64`` (base64) nếu Code Input có file này;
    (3) bỏ qua nếu ``game_round.py`` đã sạch ``retry``.
    """
    import base64

    dest = project_root / "src" / "game_round.py"
    bundled = project_root / "offline_patch_assets" / "game_round.py"
    if bundled.is_file():
        shutil.copyfile(bundled, dest)
        print("✅ Đã áp patch offline: src/game_round.py ← offline_patch_assets/")
        return
    text = dest.read_text(encoding="utf-8") if dest.is_file() else ""
    if "from retry import retry" not in text:
        return
    b64_path = project_root / "kaggle_game_round_patch.b64"
    if b64_path.is_file():
        raw = base64.b64decode(b64_path.read_text(encoding="ascii").strip())
        dest.write_bytes(raw)
        print("✅ Đã áp patch offline: src/game_round.py ← kaggle_game_round_patch.b64")
        return
    raise RuntimeError(
        "src/game_round.py vẫn import ``retry`` nhưng thiếu "
        "offline_patch_assets/game_round.py hoặc kaggle_game_round_patch.b64 — "
        "cập nhật Kaggle Code Input từ repo mới nhất."
    )


# Copy code sang working directory (input là read-only)
if FAIRGAME_WORK.exists():
    shutil.rmtree(FAIRGAME_WORK)
shutil.copytree(KAGGLE_CODE_INPUT, FAIRGAME_WORK)

FAIRGAME_ROOT = resolve_fairgame_root(FAIRGAME_WORK)
apply_game_round_no_retry_patch(FAIRGAME_ROOT)
os.chdir(FAIRGAME_ROOT)
if str(FAIRGAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FAIRGAME_ROOT))
(FAIRGAME_ROOT / "resources" / "results").mkdir(parents=True, exist_ok=True)

MARKER_ROOT.write_text(str(FAIRGAME_ROOT), encoding="utf-8")
FAIRGAME_DIR = FAIRGAME_ROOT

print(f"✅ Code ready — project root (có src/): {FAIRGAME_ROOT}")

# Kiểm tra model path
print(f"📁 Model path: {MODEL_PATH} | exists={Path(MODEL_PATH).exists()}")

# =====================================================================
# CELL 4: Load model vào GPU
# =====================================================================
ensure_src_importable()

from FAIRGAME.src.llm_connectors.local_vllm_connector import init_local_llm

print(f"🚀 Loading model ({ENGINE})...")

if ENGINE == "vllm":
    init_local_llm(
        MODEL_PATH, engine="vllm",
        max_model_len=MAX_MODEL_LEN,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        gpu_memory_utilization=GPU_UTIL,
        tensor_parallel_size=1,
    )
else:
    init_local_llm(
        MODEL_PATH, engine="transformers",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

print("✅ Model loaded!")

# =====================================================================
# CELL 5: Test nhanh
# =====================================================================
ensure_src_importable()

from FAIRGAME.src.llm_connectors.local_vllm_connector import LocalVLLMConnector

test = LocalVLLMConnector(provider_model="test")
print(f"🧪 Test response: {test.send_prompt('What is 2+2? Answer with just the number.')}")

# =====================================================================
# CELL 6: Chạy FAIRGAME experiments
# =====================================================================
from pathlib import Path

ensure_src_importable()

from FAIRGAME.src.fairgame_factory import FairGameFactory
from FAIRGAME.src.io_managers.file_manager import FileManager
from FAIRGAME.src.results_processing.results_processor import ResultsProcessor
import copy
import json

# ==== CHỌN CONFIGS ====
CONFIG_FILES = [
    "prisoner_dilemma_nocomm_round_known_mild.json",
    # "prisoner_dilemma_nocomm_round_known_harsh.json",
    # "prisoner_dilemma_nocomm_round_known_conventional.json",
    # "prisoner_dilemma_nocomm_round_not_known_mild.json",
    # "battle_sexes_nocomm_round_known_conventional.json",
    # "harmony_game_nocomm_round_known_conventional.json",
    # "snow_drift_nocomm_round_known_conventional.json",
    # "stag_hunt_nocomm_round_known_conventional.json",
]

# =====================================================================
# ==== TÙY CHỈNH CONFIG (để None = giữ giá trị mặc định từ file) ====
# =====================================================================

# --- Số round ---
OVERRIDE_N_ROUNDS = 30           # vd: 5, 10, 20, 50
OVERRIDE_ROUNDS_KNOWN = None     # True = agent biết số round, False = không biết

# --- Payoff Scaling (nhân tất cả weights với hệ số) ---
# Prisoner's Dilemma mặc định: weight1=8, weight2=10, weight3=0, weight4=2
# Sweep cả LAMBDAS — output: results/<lambda>/<short>/x<lambda>_<lang>_<short>.csv
LAMBDAS = [0.1, 1.0, 10.0]       # khớp paper (full sweep)

# Hoặc ghi đè weights cụ thể (ưu tiên hơn LAMBDAS, áp cho mọi λ):
OVERRIDE_WEIGHTS = None          # vd: {"weight1": 3, "weight2": 5, "weight3": 0, "weight4": 1}

# --- Ngôn ngữ ---
OVERRIDE_LANGUAGES = ["en", "fr", "ar", "cn", "vn"]   # 5 langs từ paper

# --- Agent settings ---
OVERRIDE_PERSONALITIES = None    # vd: {"en": ["cooperative", "competitive"]}
OVERRIDE_COMMUNICATE = None      # True = agents gửi message trước khi chọn

# --- Chạy tất cả permutations hay chỉ 1 config? ---
OVERRIDE_ALL_PERMUTATIONS = None # True / False

# --- Stop conditions ---
OVERRIDE_STOP_WHEN = None        # vd: ["combination4"] hoặc [] (không dừng sớm)

# --- Số lần lặp (repetitions) cho statistical significance ---
# Mỗi config tạo ra ~4 games (4 personality combos).
# N_REPETITIONS = 10 → 4 × 10 = 40 games tổng cộng (giống dataset gốc)
N_REPETITIONS = 10               # vd: 1, 5, 10, 20

# =====================================================================


def apply_overrides(config, lam):
    """Áp dụng các override lên config đã load từ file. ``lam`` là payoff scale hiện tại."""
    # LLM override (luôn áp dụng cho Kaggle)
    config["llm"] = "LocalModel"
    config.pop("llms", None)

    if OVERRIDE_N_ROUNDS is not None:
        config["nRounds"] = OVERRIDE_N_ROUNDS
    if OVERRIDE_ROUNDS_KNOWN is not None:
        config["nRoundsIsKnown"] = OVERRIDE_ROUNDS_KNOWN
    if OVERRIDE_COMMUNICATE is not None:
        config["agentsCommunicate"] = OVERRIDE_COMMUNICATE
    if OVERRIDE_ALL_PERMUTATIONS is not None:
        config["allAgentPermutations"] = OVERRIDE_ALL_PERMUTATIONS
    if OVERRIDE_LANGUAGES is not None:
        config["languages"] = OVERRIDE_LANGUAGES
    if OVERRIDE_STOP_WHEN is not None:
        config["stopGameWhen"] = OVERRIDE_STOP_WHEN
    if OVERRIDE_PERSONALITIES is not None:
        config["agents"]["personalities"] = OVERRIDE_PERSONALITIES

    # Payoff: scale trước, override sau (override ưu tiên hơn)
    weights = config["payoffMatrix"]["weights"]
    config["payoffMatrix"]["weights"] = {k: v * lam for k, v in weights.items()}
    if OVERRIDE_WEIGHTS is not None:
        config["payoffMatrix"]["weights"] = OVERRIDE_WEIGHTS

    return config


def load_templates(config_file, config):
    """Load prompt templates cho tất cả languages (hỗ trợ .txt và .rtf)."""
    game_name = config_file.rsplit("_nocomm", 1)[0]
    templates = {}
    for lang in config["languages"]:
        tpl_txt = Path("resources/game_templates") / f"{game_name}_{lang}.txt"
        tpl_rtf = Path("resources/game_templates") / f"{game_name}_{lang}.rtf"
        if tpl_txt.exists():
            templates[lang] = tpl_txt.read_text(encoding="utf-8")
        elif tpl_rtf.exists():
            from FAIRGAME.src.utils.rtf_to_text import rtf_to_text
            templates[lang] = rtf_to_text(tpl_rtf.read_text(encoding="utf-8"))
    if templates:
        config["promptTemplate"] = templates
    else:
        config["templateFilename"] = game_name
    return config


def shorten_strings_for_log(obj, max_len=1500):
    """Rút gọn chuỗi dài (prompt templates) để log không tràn notebook."""
    if isinstance(obj, dict):
        return {k: shorten_strings_for_log(v, max_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [shorten_strings_for_log(x, max_len) for x in obj]
    if isinstance(obj, str) and len(obj) > max_len:
        return f"{obj[:max_len]}... [truncated, {len(obj)} chars total]"
    return obj


def merge_results(all_rep_results):
    """Gộp results từ nhiều repetitions, đánh lại game_id liên tục."""
    merged = {}
    game_counter = 0
    for rep_results in all_rep_results:
        for old_game_id, game_data in rep_results.items():
            merged[f"game_{game_counter}"] = game_data
            game_counter += 1
    return merged


def fmt_lambda(scale):
    """Format lambda như paper dataset: 0.1, 1, 10 (no trailing zero)."""
    lam = 1.0 if scale is None else float(scale)
    if lam.is_integer():
        return str(int(lam))
    return str(lam)


CONFIG_DIR = Path("resources/config")
all_results = {}

SHOW_CONFIG_PREVIEW = False
SHOW_PER_REP_LOGS = False
SHOW_EXPORT_FILE_LIST = False

for lam in LAMBDAS:
    lam_str = fmt_lambda(lam)
    for config_file in CONFIG_FILES:
        print(f"\n🎮 Running: {config_file} | λ={lam_str} | reps={N_REPETITIONS}")

        # Load config gốc
        with open(CONFIG_DIR / config_file, "r", encoding="utf-8") as f:
            base_config = json.load(f)

        # Áp dụng overrides (kèm scale theo λ hiện tại)
        base_config = apply_overrides(base_config, lam=lam)

        # Full CONFIG (giống mỗi lần repetition sau load_templates)
        preview_config = load_templates(config_file, copy.deepcopy(base_config))
        if SHOW_CONFIG_PREVIEW:
            print("\n📋 CONFIG (effective — chuỗi dài trong promptTemplate bị rút gọn khi in):\n")
            print(json.dumps(
                shorten_strings_for_log(preview_config),
                indent=2,
                ensure_ascii=False,
                default=str,
            ))
            print()

        try:
            # Chạy N_REPETITIONS lần, mỗi lần tạo ~4 games (tùy permutations)
            rep_results_list = []
            for rep in range(N_REPETITIONS):
                config = copy.deepcopy(base_config)
                config = load_templates(config_file, config)

                factory = FairGameFactory()
                if USE_BATCHED_RUNNER:
                    rep_results = factory.create_and_run_games_batched(
                        config,
                        batch_size=BATCH_SIZE,
                        max_strategy_retries=BATCH_STRATEGY_RETRIES,
                    )
                else:
                    rep_results = factory.create_and_run_games(config)
                rep_results_list.append(rep_results)

                n_games = len(rep_results)
                if SHOW_PER_REP_LOGS:
                    print(f"   ✅ Rep {rep+1}/{N_REPETITIONS}: {n_games} games completed")

            # Gộp tất cả repetitions → game_0, game_1, ..., game_N
            merged_results = merge_results(rep_results_list)
            total_games = len(merged_results)
            print(f"   📊 Total: {total_games} games (~{total_games // N_REPETITIONS} permutations × {N_REPETITIONS} reps)")

            # Save: paper format Dataset/data_fairgame/<lambda>/<llm>/x<lambda>_<lang>_<llm>.csv
            name = config_file.replace(".json", "")
            df = ResultsProcessor().process(merged_results)
            results_root = Path("resources/results")
            csv_dir = results_root / lam_str / MODEL_SHORT_NAME
            csv_dir.mkdir(parents=True, exist_ok=True)

            for lang_code, df_lang in df.groupby("language"):
                csv_path = csv_dir / f"x{lam_str}_{lang_code}_{MODEL_SHORT_NAME}.csv"
                df_lang.to_csv(csv_path, index=False)
                if SHOW_PER_REP_LOGS:
                    print(f"   ✅ {csv_path.relative_to(results_root)} ({len(df_lang)} games)")

            # JSON gộp full history per-(λ, config) (debug / XAI fallback)
            json_dir = results_root / lam_str / MODEL_SHORT_NAME
            json_dir.mkdir(parents=True, exist_ok=True)
            (json_dir / f"results_{name}.json").write_text(
                json.dumps(merged_results, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8"
            )
            all_results[f"lam{lam_str}__{name}"] = merged_results

        except Exception as e:
            print(f"❌ Error (λ={lam_str}, config={config_file}): {e}")
            import traceback; traceback.print_exc()

# =====================================================================
# CELL 7: Export results
# =====================================================================
import shutil
from pathlib import Path

FAIRGAME_DIR = ensure_src_importable()

output = Path("/kaggle/working/results")
output.mkdir(exist_ok=True)

# Copy preserving directory structure: <lambda>/<llm>/x<lambda>_<lang>_<llm>.csv + JSON debug
src_results = FAIRGAME_DIR / "resources" / "results"
for src in src_results.rglob("*"):
    if src.is_file():
        dst = output / src.relative_to(src_results)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

# Save combined results JSON
combined_path = output / "all_results_combined.json"
combined_path.write_text(
    json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8"
)

print(f"\n📦 Results → {output}")
if SHOW_EXPORT_FILE_LIST:
    for f in sorted(output.rglob("*")):
        if f.is_file():
            print(f"   {f.relative_to(output)}  ({f.stat().st_size/1024:.1f} KB)")

print("""
📊 CSV format khớp với Dataset/data_fairgame/ — sẵn sàng cho infer_strategies.py:
   • agent{1,2}_strategies  - List 'OptionA'/'OptionB' từng round
   • agent{1,2}_scores      - Penalty mỗi round (đã scale theo lambda)
   • agent{1,2}_messages    - Messages giữa agents (nếu agents_communicate=True)
   • language, max_rounds, agent personalities, ... (xem Dataset/ làm reference)

📁 JSON full history per-config (results_*.json) — chứa raw rounds (dùng debug).
""")
print("🎉 Done! Vào Output tab để download.")

# =====================================================================
# CELL 8: Zip results folder (tùy chọn, để download dễ hơn)
# =====================================================================
import shutil
import zipfile
from pathlib import Path

output = Path("/kaggle/working/results")
zip_path = Path("/kaggle/working/results.zip")

print(f"\n📦 Zipping results folder...")

# Tạo zip file
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path in output.rglob("*"):
        if file_path.is_file():
            arcname = file_path.relative_to(output.parent)  # relative path inside zip
            zipf.write(file_path, arcname)
            
zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
print(f"✅ Zip created: {zip_path} ({zip_size_mb:.2f} MB)")
print(f"📥 Download từ Output tab: results.zip")

# (Tùy chọn) Hiển thị danh sách file bên trong zip
print("\n📋 Contents of results.zip:")
with zipfile.ZipFile(zip_path, 'r') as zipf:
    for info in zipf.filelist[:20]:  # show first 20 files
        print(f"   {info.filename}  ({info.file_size/1024:.1f} KB)")
    if len(zipf.filelist) > 20:
        print(f"   ... and {len(zipf.filelist) - 20} more files")
