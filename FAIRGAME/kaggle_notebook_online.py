"""
=====================================================================
FAIRGAME - Kaggle ONLINE Notebook (Internet ON, GPU ON)
=====================================================================
Phiên bản notebook kết hợp: tự download model từ HuggingFace rồi chạy
FAIRGAME ngay trong cùng notebook (không cần dataset model dựng sẵn).

CÁCH CHẠY:
  1. Tạo notebook mới trên Kaggle
     - GPU: ON (RTX 6000 / GPU accelerator)
     - Internet: ON   ← BẮT BUỘC để tải model
     - Persistence (Settings): có thể bật để cache model giữa các session

  2. Add Input (nút "+ Add Input" bên phải):
     a) Code: gắn notebook/repo FAIRGAME (read-only).
        Mặc định: /kaggle/input/notebooks/trungkiet/git-fairgame/fairgame-hackathon-research/
        Sửa KAGGLE_CODE_INPUT ở Cell 4 nếu khác.

  3. (Tuỳ chọn) Add Kaggle Secret tên ``HF_TOKEN`` cho gated models
     (Llama, Gemma...). Settings → Add-ons → Secrets.

  4. Copy nội dung file này vào notebook, chia cells theo "# CELL X".

  5. Chỉnh Cell 1 (model muốn tải) + Cell 2 (siêu tham số) → Run All.

So sánh với ``kaggle_notebook.py``:
  • kaggle_notebook.py     : Internet OFF, model nạp từ /kaggle/input/<model>/
  • kaggle_notebook_online.py (file này): Internet ON, tự ``snapshot_download``
    về /kaggle/working/model_weights/ rồi load như bình thường.
=====================================================================
"""

# =====================================================================
# CELL 1: Cấu hình DOWNLOAD MODEL — SỬA Ở ĐÂY
# =====================================================================


# === Private GitHub repo chứa FAIRGAME source ===
# Repo sẽ được clone vào /kaggle/working ở Cell 4 (không cần Add Code Input).
GITHUB_TOKEN = ""  # dán Personal Access Token khi chạy, hoặc dùng Kaggle Secrets — KHÔNG commit token thật
GITHUB_USER = "trungkiet2005"
GITHUB_REPO = "fairgame-hackathon-research"
GITHUB_BRANCH = None  # None = default branch; hoặc "main", "dev", ...

# === Model HuggingFace muốn tải (namespace/repo hoặc full URL) ===
# Với 96GB VRAM có thể chạy tới ~72B params; chọn theo nhu cầu:
MODEL_ID = "meta-llama/Llama-3.1-8B"
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"            # ~15GB, nhanh, test trước
# MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"           # ~140GB, best reasoning
# MODEL_ID = "google/gemma-2-9b-it"                # ~18GB, balanced
# MODEL_ID = "google/gemma-2-27b-it"               # ~54GB
# MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"    # ~16GB, gated
# MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"   # ~140GB, gated
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # ~15GB

# === HuggingFace token (cần cho Llama, Gemma) ===
# Để None → script tự đọc từ Kaggle Secrets (key "HF_TOKEN"). Đây là cách
# khuyến nghị; chỉ paste trực tiếp khi test cá nhân.
HF_TOKEN = ""  # dán HuggingFace token khi chạy, hoặc dùng Kaggle Secrets — KHÔNG commit token thật
# HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# === Nơi lưu model sau khi tải ===
# /kaggle/working    : ~20GB, persist trong session (chỉ đủ cho model ≤ 14GB sau khi lọc).
# /tmp/model_weights : ~70GB, ephemeral (mất khi session restart) — DÙNG CHO MODEL ≥15GB.
# Llama 3.1 8B safetensors ~16GB → KHÔNG vừa /kaggle/working, phải dùng /tmp.
MODEL_DOWNLOAD_DIR = "/tmp/model_weights"
# MODEL_DOWNLOAD_DIR = "/kaggle/working/model_weights"  # cho model nhỏ (≤14GB)

# Bỏ qua các file không cần (TIẾT KIỆM ĐĨA — quan trọng vì /kaggle/working chỉ 20GB).
# Llama 3.1 base có cả safetensors VÀ original .pth → bỏ .pth để giảm 50% dung lượng.
HF_IGNORE_PATTERNS = [
    "*.gguf",                     # GGUF (llama.cpp format, không dùng với vllm)
    "*.bin", "*.bin.index*",      # PyTorch .bin (đã có safetensors)
    "*.pth", "*.pt",              # Original PyTorch checkpoint (Llama Meta format)
    "consolidated.*",             # Meta's consolidated checkpoint
    "original/*",                 # Repos để original format trong dir này
    "*.msgpack", "*.h5", "*.ot",  # Flax / TF / Rust
]

# =====================================================================
# CELL 2: Cấu hình FAIRGAME — SỬA Ở ĐÂY
# =====================================================================

# Tự động trỏ MODEL_PATH về thư mục vừa tải; có thể override thủ công nếu cần.
MODEL_PATH = MODEL_DOWNLOAD_DIR

# === Engine chạy LLM ===
# vllm: nhanh hơn transformers ~3-5x trên T4/T4x2 (Internet ON nên có thể pip install).
# Đổi về "transformers" nếu vllm fail cài (hiếm trên image Kaggle mới).
ENGINE = "vllm"

# === Tham số model ===
MAX_MODEL_LEN = 4096
TEMPERATURE = 1.0
MAX_TOKENS = 512
GPU_UTIL = 0.90

# === Tensor parallelism cho vllm ===
# T4x2 (2 GPUs)             → 2
# Single GPU (T4 / RTX6000) → 1
# vLLM yêu cầu num_attention_heads chia hết cho TP_SIZE
# (Llama 3.1 8B có 32 heads → OK với 1/2/4; Qwen3 8B có 32 heads → OK với 1/2/4).
TP_SIZE = 2

# === Tên ngắn của model dùng cho filename output ===
# Output: results/<lambda>/<short>/x<lambda>_<lang>_<short>.csv
MODEL_SHORT_NAME = "lamma_31_8B"

# === Log verbosity ===
FAIRGAME_VERBOSE_LOGS = "0"

# =====================================================================
# CELL 3: Download model từ HuggingFace (Internet ON)
# =====================================================================
import subprocess
import sys

# CRITICAL: upgrade huggingface_hub TRƯỚC mọi import của nó.
# vllm 0.20+ cần ``BucketNotFoundError`` (thêm vào hf_hub 0.34) — image Kaggle
# có thể vẫn dùng bản cũ. Một khi đã ``import huggingface_hub`` thì pip upgrade
# không có tác dụng (Python cache sys.modules) → phải làm ngay đây, trước imports.
print("📥 Upgrading huggingface_hub (cần cho vllm 0.20+)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-U",
    "huggingface_hub>=0.34",
])

import os
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import huggingface_hub  # noqa: F401
print(f"✅ huggingface_hub {huggingface_hub.__version__}")

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError

# Force exact verbosity setting from Cell 2
os.environ["FAIRGAME_VERBOSE_LOGS"] = FAIRGAME_VERBOSE_LOGS

# Lấy HF_TOKEN từ Kaggle Secrets nếu chưa set ở Cell 1
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        if HF_TOKEN:
            print("✅ HF_TOKEN loaded from Kaggle Secrets")
    except Exception:
        HF_TOKEN = None
        print("ℹ️ Không có HF_TOKEN (Kaggle Secrets / biến trực tiếp) — chỉ tải được public model.")


def normalize_hf_repo_id(model_ref: str) -> str:
    """Accept HF repo_id hoặc full URL, normalize về 'namespace/repo'."""
    cleaned = model_ref.strip()
    if "huggingface.co" not in cleaned:
        return cleaned
    parsed = urlparse(cleaned)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if parts and parts[0] in {"models", "datasets", "spaces"}:
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(
            f"Không parse được MODEL_ID='{model_ref}'. "
            "Dùng 'namespace/repo' hoặc full URL HuggingFace."
        )
    return f"{parts[0]}/{parts[1]}"


MODEL_REPO_ID = normalize_hf_repo_id(MODEL_ID)
if MODEL_REPO_ID != MODEL_ID:
    print(f"ℹ️ Normalized MODEL_ID: {MODEL_ID} -> {MODEL_REPO_ID}")

if (MODEL_REPO_ID.startswith("meta-llama/") or MODEL_REPO_ID.startswith("google/gemma")) and not HF_TOKEN:
    print("⚠️ Model có thể bị gated. Add HF_TOKEN trong Kaggle Secrets để tránh 401/403.")

model_dir = Path(MODEL_DOWNLOAD_DIR)
model_dir.mkdir(parents=True, exist_ok=True)


def _disk_free_gb(path: Path) -> float:
    """Free space (GB) ở mount chứa ``path`` (path phải tồn tại)."""
    return shutil.disk_usage(path).free / (1024 ** 3)


# Skip nếu đã có config.json + ít nhất 1 safetensors (cache hit hoặc đã download xong)
weight_files = list(model_dir.glob("*.safetensors"))
if (model_dir / "config.json").exists() and weight_files:
    print(f"♻️ Model đã có sẵn tại {model_dir} ({len(weight_files)} safetensors) — skip download.")
else:
    # Dọn partial download (file .incomplete / .lock từ lần fail trước) để tránh tràn đĩa
    if any(model_dir.iterdir()):
        print(f"🧹 Cleaning partial download in {model_dir}...")
        shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

    free_gb = _disk_free_gb(model_dir)
    print(f"💾 Free space at {model_dir}: {free_gb:.1f} GB")
    if free_gb < 18:
        print(
            f"⚠️ Free < 18GB. Llama/Qwen 8B safetensors ~16GB. "
            f"Đổi MODEL_DOWNLOAD_DIR sang /tmp/model_weights (~70GB) ở Cell 1."
        )

    print(f"📥 Downloading {MODEL_REPO_ID} → {model_dir}")
    print(f"   ignore_patterns = {HF_IGNORE_PATTERNS}")
    t0 = time.time()
    try:
        snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=str(model_dir),
            token=HF_TOKEN,
            ignore_patterns=HF_IGNORE_PATTERNS,
        )
    except HFValidationError as e:
        raise ValueError(
            "MODEL_ID sai format. Dùng 'namespace/repo_name' "
            "(vd: 'meta-llama/Llama-3.1-8B-Instruct') hoặc full URL HF."
        ) from e
    print(f"✅ Downloaded in {(time.time() - t0) / 60:.1f} minutes")

# Kích thước & danh sách file
total = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())
print(f"📦 Model size: {total / (1024 ** 3):.2f} GB")
print(f"📁 Files (top-level):")
for p in sorted(model_dir.iterdir()):
    if p.is_file():
        print(f"   {p.name} ({p.stat().st_size / (1024 ** 2):.1f} MB)")

# =====================================================================
# CELL 4: Clone private GitHub repo + copy về /kaggle/working
# =====================================================================
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Repo được clone vào đây (read-write); sau đó copytree sang FAIRGAME_WORK.
KAGGLE_CODE_INPUT = Path(f"/kaggle/working/{GITHUB_REPO}")

FAIRGAME_WORK = Path("/kaggle/working/FAIRGAME")
MARKER_ROOT = Path("/kaggle/working/.fairgame_project_root")


def clone_or_update_repo(dest: Path, user: str, repo: str, token: str, branch: str | None) -> None:
    """Clone private repo về ``dest``. Nếu đã có .git thì fetch + reset để đồng bộ."""
    clone_url = f"https://{user}:{token}@github.com/{user}/{repo}.git"
    safe_url = f"https://{user}:***@github.com/{user}/{repo}.git"

    if (dest / ".git").is_dir():
        print(f"♻️ Repo đã tồn tại tại {dest} — fetch + reset để đồng bộ remote.")
        subprocess.check_call(["git", "-C", str(dest), "remote", "set-url", "origin", clone_url])
        subprocess.check_call(["git", "-C", str(dest), "fetch", "--depth=1", "origin"])
        target = branch or subprocess.check_output(
            ["git", "-C", str(dest), "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
            text=True,
        ).strip().split("/")[-1]
        subprocess.check_call(["git", "-C", str(dest), "reset", "--hard", f"origin/{target}"])
        return

    if dest.exists():
        print(f"🧹 {dest} tồn tại nhưng không phải git repo — xoá để clone lại.")
        shutil.rmtree(dest)

    print(f"📥 Cloning {safe_url} → {dest}")
    cmd = ["git", "clone", "--depth=1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [clone_url, str(dest)]
    subprocess.check_call(cmd)


clone_or_update_repo(KAGGLE_CODE_INPUT, GITHUB_USER, GITHUB_REPO, GITHUB_TOKEN, GITHUB_BRANCH)


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
    raise FileNotFoundError(f"Không thấy src/ dưới {base}. Mục con: {hint}")


def apply_game_round_no_retry_patch(project_root: Path) -> None:
    """Ghi đè src/game_round.py bản không cần PyPI ``retry`` (bản online thường đã sạch).
    Vẫn giữ logic này để an toàn khi Code Input là snapshot cũ.
    """
    import base64
    dest = project_root / "src" / "game_round.py"
    bundled = project_root / "offline_patch_assets" / "game_round.py"
    if bundled.is_file():
        shutil.copyfile(bundled, dest)
        print("✅ Đã áp patch: src/game_round.py ← offline_patch_assets/")
        return
    text = dest.read_text(encoding="utf-8") if dest.is_file() else ""
    if "from retry import retry" not in text:
        return
    b64_path = project_root / "kaggle_game_round_patch.b64"
    if b64_path.is_file():
        raw = base64.b64decode(b64_path.read_text(encoding="ascii").strip())
        dest.write_bytes(raw)
        print("✅ Đã áp patch: src/game_round.py ← kaggle_game_round_patch.b64")
        return
    # Internet ON → cài retry từ PyPI thay vì patch
    print("ℹ️ src/game_round.py còn import ``retry``; cài từ PyPI (Internet ON).")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "retry"])


def ensure_src_importable():
    """chdir + sys.path tới thư mục có src/."""
    if MARKER_ROOT.exists():
        root = Path(MARKER_ROOT.read_text(encoding="utf-8").strip())
    else:
        root = resolve_fairgame_root(FAIRGAME_WORK)
    os.chdir(root)
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


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

print(f"✅ Code ready — project root: {FAIRGAME_ROOT}")
print(f"📁 Model path: {MODEL_PATH} | exists={Path(MODEL_PATH).exists()}")

# =====================================================================
# CELL 5: Cài vLLM (Internet ON) — skip nếu đã có sẵn
# =====================================================================
if ENGINE == "vllm":
    try:
        import vllm  # noqa: F401
        print(f"♻️ vllm đã có sẵn (version {vllm.__version__}) — skip install.")
    except ImportError:
        print("📥 Installing vllm... (~2-5 phút)")
        # Pin huggingface_hub>=0.34 cùng lúc để vllm không tự cài bản cũ chèn lại.
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "vllm",
            "huggingface_hub>=0.34",
        ])
        import vllm  # noqa: F401
        print(f"✅ vllm installed (version {vllm.__version__})")

    # Verify huggingface_hub vẫn ≥ 0.34 sau khi cài vllm
    import huggingface_hub
    from packaging.version import Version
    if Version(huggingface_hub.__version__) < Version("0.34"):
        raise RuntimeError(
            f"huggingface_hub bị vllm install downgrade về {huggingface_hub.__version__}. "
            "Cần ≥0.34 cho vllm 0.20+. Restart kernel và Run All lại."
        )
    print(f"✅ huggingface_hub {huggingface_hub.__version__} (compatible với vllm)")

# =====================================================================
# CELL 6: Load model vào GPU
# =====================================================================
ensure_src_importable()

# Kaggle T4 fix: vLLM v1 mặc định chọn FlashInfer, JIT-compile kernel rồi
# fail với `/usr/bin/ld: cannot find -lcuda` (libcuda stub thiếu trên image
# Kaggle). T4 (SM 7.5) không support FlashAttention2 fast path nên
# XFORMERS là backend tốt nhất — không cần JIT, perf tương đương.
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from FAIRGAME.src.llm_connectors.local_vllm_connector import init_local_llm

print(f"🚀 Loading model ({ENGINE}, TP={TP_SIZE if ENGINE == 'vllm' else 'n/a'})...")

if ENGINE == "vllm":
    init_local_llm(
        MODEL_PATH, engine="vllm",
        max_model_len=MAX_MODEL_LEN,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        gpu_memory_utilization=GPU_UTIL,
        tensor_parallel_size=TP_SIZE,
    )
else:
    init_local_llm(
        MODEL_PATH, engine="transformers",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

print("✅ Model loaded!")

# =====================================================================
# CELL 7: Test nhanh
# =====================================================================
ensure_src_importable()

from FAIRGAME.src.llm_connectors.local_vllm_connector import LocalVLLMConnector

test = LocalVLLMConnector(provider_model="test")
print(f"🧪 Test response: {test.send_prompt('What is 2+2? Answer with just the number.')}")

# =====================================================================
# CELL 8: Chạy FAIRGAME experiments
# =====================================================================
import copy
import json
from pathlib import Path

ensure_src_importable()

from FAIRGAME.src.fairgame_factory import FairGameFactory
from FAIRGAME.src.io_managers.file_manager import FileManager
from FAIRGAME.src.results_processing.results_processor import ResultsProcessor

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

# ==== TÙY CHỈNH CONFIG (None = giữ giá trị mặc định từ file) ====
OVERRIDE_N_ROUNDS = 30
OVERRIDE_ROUNDS_KNOWN = None

LAMBDAS = [0.1, 1.0, 10.0]
OVERRIDE_WEIGHTS = None

OVERRIDE_LANGUAGES = ["en", "fr", "ar", "cn", "vn"]
OVERRIDE_PERSONALITIES = None
OVERRIDE_COMMUNICATE = None
OVERRIDE_ALL_PERMUTATIONS = None
OVERRIDE_STOP_WHEN = None

N_REPETITIONS = 10


def apply_overrides(config, lam):
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

    weights = config["payoffMatrix"]["weights"]
    config["payoffMatrix"]["weights"] = {k: v * lam for k, v in weights.items()}
    if OVERRIDE_WEIGHTS is not None:
        config["payoffMatrix"]["weights"] = OVERRIDE_WEIGHTS

    return config


def load_templates(config_file, config):
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
    if isinstance(obj, dict):
        return {k: shorten_strings_for_log(v, max_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [shorten_strings_for_log(x, max_len) for x in obj]
    if isinstance(obj, str) and len(obj) > max_len:
        return f"{obj[:max_len]}... [truncated, {len(obj)} chars total]"
    return obj


def merge_results(all_rep_results):
    merged = {}
    counter = 0
    for rep_results in all_rep_results:
        for _old, game_data in rep_results.items():
            merged[f"game_{counter}"] = game_data
            counter += 1
    return merged


def fmt_lambda(scale):
    lam = 1.0 if scale is None else float(scale)
    return str(int(lam)) if lam.is_integer() else str(lam)


CONFIG_DIR = Path("resources/config")
all_results = {}

SHOW_CONFIG_PREVIEW = False
SHOW_PER_REP_LOGS = False
SHOW_EXPORT_FILE_LIST = False

for lam in LAMBDAS:
    lam_str = fmt_lambda(lam)
    for config_file in CONFIG_FILES:
        print(f"\n🎮 Running: {config_file} | λ={lam_str} | reps={N_REPETITIONS}")

        with open(CONFIG_DIR / config_file, "r", encoding="utf-8") as f:
            base_config = json.load(f)

        base_config = apply_overrides(base_config, lam=lam)

        preview_config = load_templates(config_file, copy.deepcopy(base_config))
        if SHOW_CONFIG_PREVIEW:
            print("\n📋 CONFIG (effective):\n")
            print(json.dumps(
                shorten_strings_for_log(preview_config),
                indent=2, ensure_ascii=False, default=str,
            ))
            print()

        try:
            rep_results_list = []
            for rep in range(N_REPETITIONS):
                config = copy.deepcopy(base_config)
                config = load_templates(config_file, config)

                factory = FairGameFactory()
                rep_results = factory.create_and_run_games(config)
                rep_results_list.append(rep_results)

                if SHOW_PER_REP_LOGS:
                    print(f"   ✅ Rep {rep + 1}/{N_REPETITIONS}: {len(rep_results)} games")

            merged_results = merge_results(rep_results_list)
            total_games = len(merged_results)
            print(f"   📊 Total: {total_games} games (~{total_games // N_REPETITIONS} permutations × {N_REPETITIONS} reps)")

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

            json_dir = results_root / lam_str / MODEL_SHORT_NAME
            json_dir.mkdir(parents=True, exist_ok=True)
            (json_dir / f"results_{name}.json").write_text(
                json.dumps(merged_results, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            all_results[f"lam{lam_str}__{name}"] = merged_results

        except Exception as e:
            print(f"❌ Error (λ={lam_str}, config={config_file}): {e}")
            import traceback; traceback.print_exc()

# =====================================================================
# CELL 9: Export results
# =====================================================================
import shutil
from pathlib import Path

FAIRGAME_DIR = ensure_src_importable()

output = Path("/kaggle/working/results")
output.mkdir(exist_ok=True)

src_results = FAIRGAME_DIR / "resources" / "results"
for src in src_results.rglob("*"):
    if src.is_file():
        dst = output / src.relative_to(src_results)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

combined_path = output / "all_results_combined.json"
combined_path.write_text(
    json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)

print(f"\n📦 Results → {output}")
if SHOW_EXPORT_FILE_LIST:
    for f in sorted(output.rglob("*")):
        if f.is_file():
            print(f"   {f.relative_to(output)}  ({f.stat().st_size / 1024:.1f} KB)")

print("""
📊 CSV format khớp với Dataset/data_fairgame/ — sẵn sàng cho infer_strategies.py.
📁 JSON full history per-config (results_*.json) — chứa raw rounds (debug).
""")
print("🎉 Done! Vào Output tab để download.")

# =====================================================================
# CELL 10: Zip results folder (tuỳ chọn, tải về dễ hơn)
# =====================================================================
import zipfile
from pathlib import Path

output = Path("/kaggle/working/results")
zip_path = Path("/kaggle/working/results.zip")

print(f"\n📦 Zipping results folder...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for fp in output.rglob("*"):
        if fp.is_file():
            zipf.write(fp, fp.relative_to(output.parent))

print(f"✅ Zip created: {zip_path} ({zip_path.stat().st_size / (1024 ** 2):.2f} MB)")
print(f"📥 Download từ Output tab: results.zip")

print("\n📋 Contents (first 20):")
with zipfile.ZipFile(zip_path, "r") as zipf:
    for info in zipf.filelist[:20]:
        print(f"   {info.filename}  ({info.file_size / 1024:.1f} KB)")
    if len(zipf.filelist) > 20:
        print(f"   ... and {len(zipf.filelist) - 20} more files")
