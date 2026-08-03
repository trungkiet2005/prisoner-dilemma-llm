"""
=====================================================================
FAIRGAME Prisoner's Dilemma — Kaggle OFFLINE notebook (Internet OFF, GPU ON)
BASELINE CHUẨN | 7 MODEL OPEN-SOURCE | 6 MỨC PAYOFF SCALING
=====================================================================
Đây là notebook BASELINE CHÍNH THỨC của nhánh open-source cho Prisoner's Dilemma.
Nạp LẦN LƯỢT 7 model mã nguồn mở, mỗi model chạy TOÀN BỘ lưới thí nghiệm
(6 λ × 5 ngôn ngữ × 4 tổ hợp tính cách × 10 rep), lưu kết quả RIÊNG theo
(λ × model) đúng layout của `Dataset/data_fairgame_small_llm/`.

⚠️  VÌ SAO CÓ NOTEBOOK NÀY — SỬA LỖI PAYOFF CỦA LẦN CHẠY TRƯỚC
─────────────────────────────────────────────────────────────────────
Lần chạy open-source trước dùng NHẦM config `..._mild.json` (weight1 = 8).
Nhánh frontier (claude / gpt / mistral trong `Dataset/data_fairgame_frontier_llm/`)
chạy bằng `..._conventional.json` (weight1 = **6**). Hai nhánh vì thế KHÔNG so
sánh được: R = 8 (mild) vs R = 6 (conventional) đổi hẳn cả greed lẫn k-index của
ma trận PD.

  conventional (ĐÚNG — dùng ở đây):  w1=6  w2=10  w3=0  w4=2   → T=10 R=6 P=2 S=0
  mild         (SAI — lần trước):    w1=8  w2=10  w3=0  w4=2   → T=10 R=8 P=2 S=0

Notebook này CHỐT `prisoner_dilemma_nocomm_round_known_conventional.json` để nhánh
open-source khớp payoff với nhánh frontier, và mở rộng payoff scaling ra 6 mức
(frontier chỉ có 3) — đó là đóng góp riêng của nhánh open-source:

  frontier:     λ ∈ {0.1, 1, 10}
  open-source:  λ ∈ {0.01, 0.1, 1, 10, 100, 1000}      ← 6 mức, rộng hơn 2 bậc mỗi đầu

Mọi thứ khác giữ nguyên thiết kế của nhánh open-source đã có (30 vòng, agent BIẾT
số vòng, 5 ngôn ngữ, không giao tiếp, 10 rep) nên dữ liệu mới thay thế trực tiếp
`Dataset/data_fairgame_small_llm/` mà không phá pipeline `Analysis/`.

QUY MÔ (mỗi model)
─────────────────────────────────────────────────────────────────────
  6 λ × 5 lang × 4 tổ hợp tính cách × 10 rep = 1200 game
  1200 game × 30 vòng × 2 agent            = 72.000 lượt sinh
Batched runner gom 5 lang × 4 tổ hợp × 10 rep = 200 game chạy lockstep →
400 prompt/bước × 30 bước cho MỖI λ. Trên RTX PRO 6000 96GB với model 7–12B
(vLLM) ≈ 1–2 h/model; 70B AWQ chậm hơn ~8–10×.

⚠️  CHIA PHIÊN (1 phiên Kaggle ~9–12h): 7 model một phiên là QUÁ GIỜ. Nên chạy
1–2 model/phiên (comment bớt `MODELS[]`), 70B/72B AWQ chạy riêng hẳn. Output ghi
riêng theo <λ>/<model> nên gộp nhiều phiên không đụng nhau — RESUME (Cell 1) tự
bỏ qua các ô (λ, model) đã có đủ 5 file CSV.

CÁCH CHẠY
─────────────────────────────────────────────────────────────────────
  1. (nếu image Kaggle chưa có vLLM) chạy `kaggle/setup/build_quant_wheels.py`
     (Internet ON) → Output → New Dataset (vd "pd-quant-wheels").
  2. + Add Input mỗi model: model bf16 từ Kaggle Models hub; model AWQ tải bằng
     `FAIRGAME/download_model.py` → New Dataset.
  3. Tạo notebook mới — GPU ON, Internet OFF. + Add Input:
       (a) repo này (thư mục chứa `FAIRGAME/`), (b) dataset wheels, (c) các model.
  4. Copy file này vào notebook, chia cell theo "# CELL N".
  5. Sửa `MODELS[]` theo path thực (`!ls /kaggle/input/`). Run Cell 1 → 8.

OUTPUT
─────────────────────────────────────────────────────────────────────
  /kaggle/working/pd_results/<λ>/<model_short>/x<λ>_<lang>_<model_short>.csv
  + results_<config>.json (full history, debug/XAI) cùng thư mục
  + run_manifest.json + pd_results.zip ở Output tab.
Layout này TRÙNG `Dataset/data_fairgame_small_llm/<λ>/<model>/x<λ>_<lang>_<model>.csv`
→ giải nén đè thẳng vào Dataset/ là `Analysis/run_all.py` chạy được ngay.
(Nhớ thêm short_name mới vào `Analysis/pdlib/ingest.py::MODEL_MAP`, và sửa
`_BASE_MATRIX["small"]` R: 8.0 → 6.0 khi dữ liệu mới thay dữ liệu cũ.)
=====================================================================
"""

# =====================================================================
# CELL 1: CẤU HÌNH — SỬA Ở ĐÂY
# =====================================================================

# --- Danh sách 7 model open-source. Mỗi model đã add làm Kaggle input. ------ #
# path:          thư mục model trong /kaggle/input/... (xem bằng "!ls /kaggle/input/")
# short_name:    tên thư mục output + phần <model> trong tên file CSV (phải DUY NHẤT,
#                và phải là key hợp lệ khi thêm vào Analysis/pdlib/ingest.py::MODEL_MAP).
# engine:        "vllm" (khuyên dùng) | "transformers".
# quantization:  None = vLLM TỰ nhận từ checkpoint (bf16 nhỏ để None; AWQ/GPTQ cũng để
#                None → vLLM tự đọc quantization_config). Ép "awq"/"gptq" nếu kernel lỗi.
# (tuỳ chọn)     dtype / kv_cache_dtype / max_num_seqs / cpu_offload_gb / gpu_util
#                / temperature / max_tokens / max_model_len: override riêng model đó.
#
# GỢI Ý CHIA PHIÊN: comment bớt để mỗi phiên vừa giờ (3 model nhỏ 1 phiên; 2 model
# vừa 1 phiên; mỗi model AWQ 70B/72B chạy riêng 1 phiên).
MODELS = [
    # --- nhỏ (bf16, GPU thường đủ) ---
    {
        "path": "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1",
        "short_name": "qwen25-7b-instruct",
        "engine": "vllm",
    },
    {
        "path": "/kaggle/input/models/google/gemma-2/transformers/gemma-2-9b-it/2",
        "short_name": "gemma2-9b-it",
        "engine": "vllm",
    },
    {
        "path": "/kaggle/input/datasets/foundnotkiet/llama-3-1-8b/model_weights",
        "short_name": "llama-3-1-8b",
        "engine": "vllm",
    },
    # --- vừa (bf16, cần VRAM lớn hơn) ---
    {
        "path": "/kaggle/input/models/qwen-lm/qwen2.5/transformers/32b-instruct/1",
        "short_name": "qwen25-32b-instruct",
        "engine": "vllm",
    },
    {
        "path": "/kaggle/input/models/google/gemma-2/transformers/gemma-2-27b-it/2",
        "short_name": "gemma2-27b-it",
        "engine": "vllm",
    },
    # --- lớn (AWQ int4, cần GPU 96GB + dataset wheels) ---
    {
        "path": "/kaggle/input/qwen25-72b-instruct-awq/model_weights",
        "short_name": "qwen25-72b-instruct-awq",
        "engine": "vllm",
        "quantization": None,     # AWQ quantize sẵn → vLLM tự nhận (awq_marlin)
        "dtype": "auto",          # auto → float16 (chuẩn cho AWQ)
    },
    {
        # Llama 3.3 70B Instruct AWQ int4 — KAGGLE MODELS hub (không phải dataset).
        # Nguồn: kaggle.com/models/jagatkiran/meta-llama-3.3-70b/Transformers/ibnzterrell-instruct-awq-int4/1
        # NHỚ verify bằng `!ls /kaggle/input/...` (Cell 3 in exists=) — sai path thì sửa lại đây.
        "path": "/kaggle/input/models/jagatkiran/meta-llama-3.3-70b/transformers/ibnzterrell-instruct-awq-int4/1",
        "short_name": "llama-3-3-70b-instruct-awq",
        "engine": "vllm",
        "quantization": None,
        "dtype": "auto",
    },
]

# --- Config game (tên file trong FAIRGAME/resources/config/) ---------------- #
# ⚠️ conventional = weight1 6 → KHỚP nhánh frontier. KHÔNG đổi sang mild/harsh trừ
# khi cố tình chạy một ablation riêng (lúc đó nhớ đổi cả OUTPUT_DIR để khỏi lẫn).
CONFIG_FILES = [
    "prisoner_dilemma_nocomm_round_known_conventional.json",
    # "prisoner_dilemma_nocomm_round_known_mild.json",        # w1=8 — lần chạy SAI trước đây
    # "prisoner_dilemma_nocomm_round_known_harsh.json",       # w1=8 w4=5
    # "prisoner_dilemma_nocomm_round_not_known_conventional.json",
]

# --- Payoff scaling: nhân TẤT CẢ weights với λ ------------------------------ #
# frontier chỉ có {0.1, 1, 10}; nhánh open-source mở rộng 2 bậc mỗi đầu.
LAMBDAS = [0.01, 0.1, 1, 10, 100, 1000]

# Ghi đè weights tuyệt đối (ưu tiên hơn LAMBDAS, áp cho MỌI λ). None = dùng λ.
OVERRIDE_WEIGHTS = None   # vd {"weight1": 6, "weight2": 10, "weight3": 0, "weight4": 2}

# λ×weight ra float (6 × 1 = 6.0). True = ép về int khi giá trị nguyên, để prompt
# ghi "6" chứ không phải "6.0" → λ=1 khớp ĐÚNG TỪNG KÝ TỰ prompt của nhánh frontier.
NORMALIZE_INTEGER_WEIGHTS = True

# --- Lưới thí nghiệm (giữ nguyên thiết kế nhánh open-source đã có) ---------- #
LANGUAGES = ["en", "fr", "ar", "cn", "vn"]
N_ROUNDS = 30            # nhánh frontier dùng 10; open-source dùng 30 (đã cố định)
ROUNDS_KNOWN = True      # agent BIẾT tổng số vòng
N_REPETITIONS = 10       # 4 tổ hợp tính cách × 10 rep = 40 game / (λ, lang)
AGENTS_COMMUNICATE = None   # None = theo config (False). True = bật pha nhắn tin.
STOP_WHEN = None            # None = theo config ([] = không dừng sớm)

# --- Tham số sinh MẶC ĐỊNH (model có thể override từng cái trong MODELS[]) --- #
DEFAULT_ENGINE = "vllm"
MAX_MODEL_LEN = 8192     # 30 vòng history dài hơn CRSD → cần context rộng hơn 4096
TEMPERATURE = 1.0        # FAIRGAME gốc dùng 1.0
MAX_TOKENS = 512
GPU_UTIL = 0.92
TP_SIZE = 1
BATCH_SIZE = 0           # 0 = 1 batch/bước (≈400 prompt). Giảm 128/64 nếu OOM.
BATCH_STRATEGY_RETRIES = 2   # retry các response không khớp strategy nào, rồi fallback

# --- Knob quantize MẶC ĐỊNH (override per-model trong MODELS[]) ------------- #
QUANTIZATION = None      # None = tự nhận từ checkpoint
DTYPE = "auto"           # "auto" | "float16" | "bfloat16"
KV_CACHE_DTYPE = None    # None/"auto" | "fp8" (nén KV-cache khi VRAM sát nút)
MAX_NUM_SEQS = None      # None = mặc định vLLM; vd 64 để ghìm VRAM
CPU_OFFLOAD_GB = 0       # >0 = đẩy trọng số sang RAM (chậm — van xả cuối)
ENFORCE_EAGER = True     # True = an toàn + tiết kiệm VRAM (không CUDA graph)

# --- Nút TRIM (giảm tải cho vừa 1 phiên) ----------------------------------- #
LAMBDAS_OVERRIDE = None      # vd [1] để soi nhanh 1 mức trước khi chạy full
LANGUAGES_OVERRIDE = None    # vd ["en"] để chỉ chạy 1 ngôn ngữ
REPS_OVERRIDE = None         # vd 2 để chạy thử

# --- Resume / output ------------------------------------------------------- #
RESUME = True            # True = bỏ qua ô (λ, model) đã có đủ CSV mọi ngôn ngữ
SMOKE_TEST = True        # in 1 reply mẫu + ETA thô khi nạp mỗi model
FAIRGAME_VERBOSE_LOGS = "0"

from pathlib import Path  # noqa: E402

OUTPUT_DIR = Path("/kaggle/working/pd_results")

# =====================================================================
# CELL 2: Helpers path (Internet OFF — không pip trừ Cell 2.5)
# =====================================================================
import os  # noqa: E402
import sys  # noqa: E402

os.environ["FAIRGAME_VERBOSE_LOGS"] = FAIRGAME_VERBOSE_LOGS

WORK_COPY = Path("/kaggle/working/pd_repo")
MARKER_ROOT = Path("/kaggle/working/.pd_project_root")


def resolve_repo_root(base: Path) -> Path:
    """Thư mục gốc để import `FAIRGAME.src.*` — tức thư mục CHỨA `FAIRGAME/`."""
    cands = [base]
    if base.is_dir():
        cands += [c for c in sorted(base.iterdir()) if c.is_dir()]
    for c in cands:
        if (c / "FAIRGAME" / "src").is_dir():
            return c.resolve()
    hint = [p.name for p in base.iterdir()] if base.is_dir() else []
    raise FileNotFoundError(f"Không thấy FAIRGAME/src/ dưới {base}. Mục con: {hint}")


def ensure_importable():
    """chdir + sys.path tới repo root (đọc marker do Cell 3 ghi)."""
    if not MARKER_ROOT.exists():
        raise RuntimeError("Chưa có marker — chạy Cell 3 trước.")
    root = Path(MARKER_ROOT.read_text(encoding="utf-8").strip())
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


print("Internet OFF: dùng thư viện có sẵn trên image Kaggle (trừ Cell 2.5 nếu cần vLLM).")

# =====================================================================
# CELL 2.5: (TUỲ CHỌN) Cài vLLM OFFLINE từ wheels đã build sẵn
# =====================================================================
# Chỉ cần khi muốn engine="vllm" mà image Kaggle CHƯA có vllm. Yêu cầu một Dataset
# chứa toàn bộ .whl (build bằng kaggle/setup/build_quant_wheels.py, Internet ON,
# CÙNG image GPU), đã + Add Input.
import importlib.util  # noqa: E402
import subprocess  # noqa: E402


def find_wheels_dir(root="/kaggle/input", max_depth=6):
    """Tự dò thư mục chứa vllm*.whl dưới /kaggle/input/."""
    root = Path(root)
    if not root.is_dir():
        return None
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            if any(d.glob("vllm*.whl")):
                return d
        except OSError:
            pass
        if depth < max_depth:
            try:
                for c in sorted(d.iterdir()):
                    if c.is_dir() and not c.name.startswith("."):
                        stack.append((c, depth + 1))
            except OSError:
                pass
    return None


VLLM_WHEELS_DIR = find_wheels_dir("/kaggle/input") or Path("/kaggle/input/pd-quant-wheels")
VLLM_VERSION = ""   # "" = bản trong wheels; hoặc ghim "0.6.x"

_want_vllm = (DEFAULT_ENGINE == "vllm") or any(
    m.get("engine", DEFAULT_ENGINE) == "vllm" for m in MODELS)
_have_vllm = importlib.util.find_spec("vllm") is not None

if _want_vllm:
    # Tắt FlashInfer sampler: trên GPU mới (Blackwell sm_120) nó JIT-compile và ngã.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    print("VLLM_USE_FLASHINFER_SAMPLER=0 (sampler PyTorch-native).")

if not _want_vllm:
    print("Không model nào dùng vllm — bỏ qua cài đặt.")
elif _have_vllm:
    print("vLLM đã có sẵn trên image — không cần cài.")
elif not VLLM_WHEELS_DIR.is_dir():
    raise FileNotFoundError(
        f"Cần engine vllm nhưng không thấy wheels ở {VLLM_WHEELS_DIR}. "
        "Hãy + Add Input dataset wheels, sửa VLLM_WHEELS_DIR, hoặc đổi engine='transformers'.")
else:
    _spec = "vllm" + (f"=={VLLM_VERSION}" if VLLM_VERSION else "")
    _cmd = [sys.executable, "-m", "pip", "install", "--no-index",
            f"--find-links={VLLM_WHEELS_DIR}", _spec]
    print(f"Cài vLLM offline từ {VLLM_WHEELS_DIR} ... (ẩn log pip, chỉ hiện khi lỗi)")
    _r = subprocess.run(_cmd, capture_output=True, text=True)
    if _r.returncode != 0:
        print(_r.stdout[-3000:])
        print(_r.stderr[-3000:])
        raise RuntimeError("pip install vllm offline thất bại — xem log phía trên.")
    importlib.invalidate_caches()
    _probe = "import torch; torch.zeros(1).cuda(); print('GPU_OK', torch.__version__)"
    _rp = subprocess.run([sys.executable, "-c", _probe], capture_output=True, text=True)
    if _rp.returncode != 0:
        print(_rp.stdout)
        print(_rp.stderr)
        raise RuntimeError(
            "torch vừa cài KHÔNG init được GPU — gần như chắc do wheels build SAI CUDA so với "
            "driver Kaggle. Build lại wheels khớp torch của Kaggle, hoặc tạm đổi engine='transformers'.")
    print(_rp.stdout.strip())
    print("vLLM đã cài từ wheels và torch init GPU OK.")

# =====================================================================
# CELL 3: Setup source (copy repo, patch offline, đặt marker)
# =====================================================================
import shutil  # noqa: E402


def find_repo_input(root="/kaggle/input", max_depth=6):
    """Tự dò thư mục input chứa FAIRGAME/src/ (dataset / GitHub repo / notebook output)."""
    root = Path(root)
    if not root.is_dir():
        return None
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            if (d / "FAIRGAME" / "src").is_dir():
                return d
        except OSError:
            pass
        if depth < max_depth:
            try:
                for c in sorted(d.iterdir()):
                    if c.is_dir() and not c.name.startswith("."):
                        stack.append((c, depth + 1))
            except OSError:
                pass
    return None


KAGGLE_CODE_INPUT = find_repo_input("/kaggle/input") or Path("/kaggle/input/prisoner-dilemma-game")
print(f"Repo input: {KAGGLE_CODE_INPUT}  (exists={Path(KAGGLE_CODE_INPUT).exists()})")


def apply_game_round_no_retry_patch(fairgame_dir: Path) -> None:
    """
    Gỡ phụ thuộc PyPI ``retry`` khỏi FAIRGAME/src/game_round.py khi Internet OFF.

    ⚠️ CHỈ patch KHI CẦN. `offline_patch_assets/game_round.py` là snapshot CŨ hơn
    src/ (matcher của nó chỉ so chuỗi con, không hiểu "Option A"/"A"/"1" và không có
    fallback). Bản notebook cũ copy đè vô điều kiện → tự hạ cấp chính file đang tốt.
    Thứ tự đúng: nếu src/ đã sạch ``retry`` thì KHÔNG đụng vào.
      (1) src/ đã sạch  → không làm gì;
      (2) kaggle_game_round_patch.b64 (bản mới, base64) nếu có;
      (3) offline_patch_assets/game_round.py — phương án chót, có cảnh báo.
    """
    import base64

    dest = fairgame_dir / "src" / "game_round.py"
    text = dest.read_text(encoding="utf-8") if dest.is_file() else ""
    if "from retry import retry" not in text:
        print("✅ src/game_round.py đã sạch ``retry`` — không cần patch.")
        return

    b64_path = fairgame_dir / "kaggle_game_round_patch.b64"
    if b64_path.is_file():
        dest.write_bytes(base64.b64decode(b64_path.read_text(encoding="ascii").strip()))
        print("✅ Đã áp patch offline: src/game_round.py ← kaggle_game_round_patch.b64")
        return

    bundled = fairgame_dir / "offline_patch_assets" / "game_round.py"
    if bundled.is_file():
        shutil.copyfile(bundled, dest)
        print("⚠️  Đã áp patch offline: src/game_round.py ← offline_patch_assets/ "
              "(snapshot CŨ — matcher yếu hơn; batched runner không dùng matcher này "
              "nên vẫn chạy đúng, nhưng nên cập nhật Kaggle Input từ repo mới nhất).")
        return

    raise RuntimeError(
        "src/game_round.py vẫn import ``retry`` nhưng thiếu kaggle_game_round_patch.b64 "
        "và offline_patch_assets/ — cập nhật Kaggle Code Input từ repo mới nhất.")


if WORK_COPY.exists():
    shutil.rmtree(WORK_COPY)
shutil.copytree(KAGGLE_CODE_INPUT, WORK_COPY, ignore=shutil.ignore_patterns(".git"))

REPO_ROOT = resolve_repo_root(WORK_COPY)
FAIRGAME_DIR = REPO_ROOT / "FAIRGAME"
apply_game_round_no_retry_patch(FAIRGAME_DIR)
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MARKER_ROOT.write_text(str(REPO_ROOT), encoding="utf-8")

_need = [FAIRGAME_DIR / "src" / "fairgame_factory.py",
         FAIRGAME_DIR / "src" / "batch_runner.py",
         FAIRGAME_DIR / "src" / "llm_connectors" / "local_vllm_connector.py",
         FAIRGAME_DIR / "resources" / "config" / CONFIG_FILES[0],
         FAIRGAME_DIR / "resources" / "game_templates"]
_missing = [str(p) for p in _need if not p.exists()]
print(f"Repo root: {REPO_ROOT}")
print("Models khai báo:")
for m in MODELS:
    print(f"   - {m['short_name']:<28} exists={Path(m['path']).exists()}  ({m['path']})")
if _missing:
    print("THIẾU file (push & re-add input):")
    for _m in _missing:
        print("   -", _m)
else:
    print("OK — FAIRGAME/ + config + game_templates đầy đủ.")

# =====================================================================
# CELL 4: Import FAIRGAME + dựng kế hoạch chạy
# =====================================================================
ensure_importable()

import copy  # noqa: E402
import json  # noqa: E402

from FAIRGAME.src.fairgame_factory import FairGameFactory                  # noqa: E402
from FAIRGAME.src.results_processing.results_processor import ResultsProcessor  # noqa: E402
import FAIRGAME.src.llm_connectors.local_vllm_connector as conn            # noqa: E402

CONFIG_DIR = FAIRGAME_DIR / "resources" / "config"
TEMPLATE_DIR = FAIRGAME_DIR / "resources" / "game_templates"

EFF_LAMBDAS = list(LAMBDAS_OVERRIDE) if LAMBDAS_OVERRIDE else list(LAMBDAS)
EFF_LANGUAGES = list(LANGUAGES_OVERRIDE) if LANGUAGES_OVERRIDE else list(LANGUAGES)
EFF_REPS = int(REPS_OVERRIDE) if REPS_OVERRIDE else int(N_REPETITIONS)


def fmt_lambda(scale) -> str:
    """Format λ như layout dataset: 0.01, 0.1, 1, 10, 100, 1000 (không thừa .0)."""
    lam = float(scale)
    return str(int(lam)) if lam.is_integer() else str(lam)


def scale_weight(value, lam):
    """value × λ, làm tròn hết nhiễu float; ép int khi nguyên (nếu bật knob)."""
    scaled = round(float(value) * float(lam), 10)
    if NORMALIZE_INTEGER_WEIGHTS and float(scaled).is_integer():
        return int(scaled)
    return scaled


def load_templates(config_file: str, config: dict) -> dict:
    """Nạp prompt template cho mọi ngôn ngữ (hỗ trợ .txt và .rtf)."""
    game_name = config_file.rsplit("_nocomm", 1)[0]
    templates = {}
    for lang in config["languages"]:
        tpl_txt = TEMPLATE_DIR / f"{game_name}_{lang}.txt"
        tpl_rtf = TEMPLATE_DIR / f"{game_name}_{lang}.rtf"
        if tpl_txt.exists():
            templates[lang] = tpl_txt.read_text(encoding="utf-8")
        elif tpl_rtf.exists():
            from FAIRGAME.src.utils.rtf_to_text import rtf_to_text
            templates[lang] = rtf_to_text(tpl_rtf.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(f"Không thấy template {game_name}_{lang}.(txt|rtf)")
    config["promptTemplate"] = templates
    config.pop("templateFilename", None)
    return config


def build_config(config_file: str, lam) -> dict:
    """Load config file → áp mọi override của Cell 1 → gắn template. Trả config sẵn chạy."""
    with open(CONFIG_DIR / config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # LLM: luôn dùng connector local trên Kaggle.
    config["llm"] = "LocalModel"
    config.pop("llms", None)

    config["nRounds"] = N_ROUNDS
    config["nRoundsIsKnown"] = ROUNDS_KNOWN
    config["languages"] = EFF_LANGUAGES
    if AGENTS_COMMUNICATE is not None:
        config["agentsCommunicate"] = AGENTS_COMMUNICATE
    if STOP_WHEN is not None:
        config["stopGameWhen"] = STOP_WHEN

    # Payoff: scale theo λ trước, override tuyệt đối sau (override ưu tiên hơn).
    weights = config["payoffMatrix"]["weights"]
    config["payoffMatrix"]["weights"] = {k: scale_weight(v, lam) for k, v in weights.items()}
    if OVERRIDE_WEIGHTS is not None:
        config["payoffMatrix"]["weights"] = dict(OVERRIDE_WEIGHTS)

    return load_templates(config_file, config)


def csv_paths_for(lam_str: str, short: str):
    """Các file CSV mà một ô (λ, model) phải sinh ra — dùng cho RESUME."""
    d = OUTPUT_DIR / lam_str / short
    return [d / f"x{lam_str}_{lang}_{short}.csv" for lang in EFF_LANGUAGES]


_games_per_cell = len(EFF_LANGUAGES) * 4 * EFF_REPS      # 4 = tổ hợp tính cách (2 personality ^ 2 agent)
_cells = len(EFF_LAMBDAS) * len(CONFIG_FILES)
GAMES_PER_MODEL = _games_per_cell * _cells
GEN_PER_MODEL = GAMES_PER_MODEL * N_ROUNDS * 2

print(f"Kế hoạch / model: {len(EFF_LAMBDAS)} λ × {len(CONFIG_FILES)} config × "
      f"{len(EFF_LANGUAGES)} lang × 4 tổ hợp × {EFF_REPS} rep = {GAMES_PER_MODEL} game "
      f"({GEN_PER_MODEL} lượt sinh, {_games_per_cell * 2} prompt/bước batch).")
print(f"Tổng {len(MODELS)} model = {len(MODELS) * GAMES_PER_MODEL} game.")
print(f"λ chạy: {[fmt_lambda(x) for x in EFF_LAMBDAS]} | lang: {EFF_LANGUAGES} | "
      f"config: {CONFIG_FILES}")

# =====================================================================
# CELL 5: Helpers — nạp model, smoke test, chạy 1 ô (λ × config), lưu
# =====================================================================
import time  # noqa: E402


def engine_kwargs_for(model_cfg: dict) -> tuple:
    """(engine, kwargs) truyền vào init_local_llm — knob chỉ gửi khi đặt tường minh."""
    engine = model_cfg.get("engine", DEFAULT_ENGINE)
    kwargs = {
        "temperature": float(model_cfg.get("temperature", TEMPERATURE)),
        "max_tokens": int(model_cfg.get("max_tokens", MAX_TOKENS)),
    }
    if engine == "vllm":
        # `_init_vllm_engine` tự lọc các knob rỗng/"auto" trước khi dựng vLLM LLM(),
        # nên truyền thẳng cả cụm là an toàn với cả vLLM đời cũ.
        kwargs.update(
            max_model_len=int(model_cfg.get("max_model_len", MAX_MODEL_LEN)),
            gpu_memory_utilization=float(model_cfg.get("gpu_util", GPU_UTIL)),
            tensor_parallel_size=TP_SIZE,
            enforce_eager=bool(model_cfg.get("enforce_eager", ENFORCE_EAGER)),
            quantization=model_cfg.get("quantization", QUANTIZATION),
            dtype=model_cfg.get("dtype", DTYPE),
            kv_cache_dtype=model_cfg.get("kv_cache_dtype", KV_CACHE_DTYPE),
            max_num_seqs=model_cfg.get("max_num_seqs", MAX_NUM_SEQS),
            cpu_offload_gb=model_cfg.get("cpu_offload_gb", CPU_OFFLOAD_GB),
        )
    elif "quantization" in model_cfg:
        kwargs["quantization"] = model_cfg["quantization"]
    return engine, kwargs


def load_model(model_cfg: dict) -> None:
    """Nạp model vào GPU (force=True để thay model trong cùng tiến trình)."""
    engine, kwargs = engine_kwargs_for(model_cfg)
    print(f"Loading {model_cfg['short_name']} ({engine}) <- {model_cfg['path']}")
    conn.init_local_llm(model_cfg["path"], engine=engine, force=True, **kwargs)
    print(f"{model_cfg['short_name']} loaded.")


def smoke_test(model_cfg: dict) -> None:
    """1 prompt thật của game → xem model có trả 'OptionA/OptionB' không + ETA thô."""
    from FAIRGAME.src.game_round import GameRound

    config = build_config(CONFIG_FILES[0], EFF_LAMBDAS[0])
    factory = FairGameFactory()
    processed = factory.io_manager.process_and_validate_configuration(copy.deepcopy(config))
    games = factory.create_games(processed)
    game = games[0]
    agent = list(game.agents.values())[0]
    prompt = GameRound(game).create_prompt(agent, phase="choose")

    t0 = time.time()
    resp = conn.send_prompts_global([prompt])[0]
    dt = time.time() - t0

    from FAIRGAME.src.batch_runner import _match_strategy_key
    key = _match_strategy_key(resp, game.payoff_matrix.strategies)
    print(f"[{model_cfg['short_name']}] prompt mẫu (200 ký tự cuối):\n...{prompt[-200:]}")
    print(f"[{model_cfg['short_name']}] reply: {resp[:200]!r}  → strategy={key}")
    if key is None:
        print("⚠️  Reply KHÔNG khớp strategy nào — batched runner sẽ retry rồi fallback "
              "OptionA. Nếu tỉ lệ này cao thì dữ liệu hỏng: đổi model instruct / tăng MAX_TOKENS.")
    print(f"ETA THÔ: 1 lượt sinh tuần tự ~{dt:.1f}s → CẬN TRÊN "
          f"~{dt * GEN_PER_MODEL / 3600:.1f} h/model cho {GEN_PER_MODEL} lượt "
          f"(thực tế nhanh hơn NHIỀU vì batch {_games_per_cell * 2} prompt/lần). "
          f"Quá lâu → hạ REPS_OVERRIDE / LANGUAGES_OVERRIDE / LAMBDAS_OVERRIDE ở Cell 1.")


def create_games_with_reps(factory: FairGameFactory, config: dict, reps: int) -> list:
    """
    Dựng `reps` bản sao của toàn bộ lưới permutation trong CÙNG một factory, để
    batched runner chạy tất cả rep lockstep (batch to hơn reps lần → nhanh hơn hẳn
    so với vòng lặp rep tuần tự).

    `create_games` cộng dồn `config_all_langs_df` rồi dựng lại `self.games` từ TOÀN
    BỘ df, nên gọi `reps` lần → `reps × (n_lang × 4)` game độc lập (mỗi game là
    object mới, history/score riêng).
    """
    processed = factory.io_manager.process_and_validate_configuration(config)
    games = []
    for _ in range(reps):
        games = factory.create_games(processed)
    return games


def run_cell(model_cfg: dict, config_file: str, lam) -> dict:
    """Chạy MỘT ô (model × config × λ) và ghi CSV/JSON. Model phải đã nạp sẵn."""
    short = model_cfg["short_name"]
    lam_str = fmt_lambda(lam)
    out_dir = OUTPUT_DIR / lam_str / short
    name = config_file.replace(".json", "")

    expected = csv_paths_for(lam_str, short)
    if RESUME and all(p.exists() for p in expected):
        print(f"[resume] bỏ qua {short} λ={lam_str} — đã đủ {len(expected)} CSV.")
        return {"model": short, "config": name, "lambda": lam_str, "status": "resumed"}

    config = build_config(config_file, lam)
    weights = config["payoffMatrix"]["weights"]
    print(f"\n🎮 {short} | {name} | λ={lam_str} | weights={weights} | "
          f"{len(EFF_LANGUAGES)} lang × 4 tổ hợp × {EFF_REPS} rep")

    factory = FairGameFactory()
    games = create_games_with_reps(factory, config, EFF_REPS)
    print(f"   dựng {len(games)} game → batch {len(games) * 2} prompt/bước, {N_ROUNDS} bước")

    t0 = time.time()
    factory.run_games_batched(batch_size=BATCH_SIZE,
                              max_strategy_retries=BATCH_STRATEGY_RETRIES)
    results = factory.results_games()
    elapsed = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    df = ResultsProcessor().process(results)
    written = []
    for lang_code, df_lang in df.groupby("language"):
        csv_path = out_dir / f"x{lam_str}_{lang_code}_{short}.csv"
        df_lang.to_csv(csv_path, index=False)
        written.append(f"{csv_path.name} ({len(df_lang)})")

    (out_dir / f"results_{name}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"   ✅ {len(results)} game trong {elapsed / 60:.1f} phút → {out_dir}")
    print(f"      {', '.join(written)}")
    return {"model": short, "config": name, "lambda": lam_str, "status": "ok",
            "n_games": len(results), "weights": weights,
            "elapsed_sec": round(elapsed, 1), "output_dir": str(out_dir)}


def run_one_model(model_cfg: dict) -> list:
    """Nạp → smoke test → chạy mọi (config × λ). KHÔNG free ở đây (Cell 6 free trong finally)."""
    short = model_cfg["short_name"]
    if not Path(model_cfg["path"]).exists():
        print(f"BỎ QUA {short}: path không tồn tại ({model_cfg['path']}).")
        return []

    # RESUME: model đã xong hết mọi ô thì khỏi nạp GPU.
    if RESUME and all(all(p.exists() for p in csv_paths_for(fmt_lambda(lam), short))
                      for lam in EFF_LAMBDAS):
        print(f"[resume] {short} đã có đủ CSV cho mọi λ — bỏ qua (không nạp model).")
        return [{"model": short, "status": "resumed"}]

    load_model(model_cfg)
    if SMOKE_TEST:
        smoke_test(model_cfg)

    summaries = []
    for config_file in CONFIG_FILES:
        for lam in EFF_LAMBDAS:
            summaries.append(run_cell(model_cfg, config_file, lam))
    return summaries

# =====================================================================
# CELL 6: Chạy LẦN LƯỢT tất cả model
# =====================================================================
import traceback  # noqa: E402

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
manifest = {
    "config_files": CONFIG_FILES,
    "lambdas": [fmt_lambda(x) for x in EFF_LAMBDAS],
    "languages": EFF_LANGUAGES,
    "n_rounds": N_ROUNDS,
    "n_rounds_known": ROUNDS_KNOWN,
    "repetitions": EFF_REPS,
    "temperature": TEMPERATURE,
    "batch_size": BATCH_SIZE,
    "normalize_integer_weights": NORMALIZE_INTEGER_WEIGHTS,
    "runs": [],
}


def _write_manifest():
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


# Mỗi model bọc try/except/finally: 1 model lỗi (vd OOM) KHÔNG làm hỏng cả run —
# vẫn free GPU, ghi manifest, chạy tiếp model sau. Ô nào xong đã ghi đĩa rồi.
for _cfg in MODELS:
    _short = _cfg["short_name"]
    try:
        _summaries = run_one_model(_cfg)
        manifest["runs"].extend(_summaries or [{"model": _short, "status": "skipped"}])
    except Exception as _e:  # noqa: BLE001
        traceback.print_exc()
        manifest["runs"].append({"model": _short, "status": "failed",
                                 "error": f"{type(_e).__name__}: {_e}"})
        print(f"⚠️  {_short} LỖI — bỏ qua, chạy tiếp model sau.")
    finally:
        try:
            conn.free_local_llm()
        except Exception:  # noqa: BLE001
            pass
        _write_manifest()   # cập nhật sau MỖI model (sống sót cả khi crash muộn)

_done = sum(1 for r in manifest["runs"] if r.get("n_games"))
print(f"\nHoàn tất {_done} ô (model × config × λ). Chi tiết: run_manifest.json")

# =====================================================================
# CELL 7: Gộp mọi CSV để soi nhanh (so sánh chéo model × λ × lang)
# =====================================================================
import ast  # noqa: E402

import pandas as pd  # noqa: E402

_frames = []
for csv_path in sorted(OUTPUT_DIR.rglob("x*.csv")):
    rel = csv_path.relative_to(OUTPUT_DIR)          # <λ>/<model>/x<λ>_<lang>_<model>.csv
    if len(rel.parts) < 3:
        continue
    df = pd.read_csv(csv_path)
    df.insert(0, "scale", float(rel.parts[0]))
    df.insert(1, "model", rel.parts[1])
    _frames.append(df)

if _frames:
    combined = pd.concat(_frames, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "pd_all_models.csv", index=False)
    print(f"Gộp {len(_frames)} file → pd_all_models.csv ({len(combined)} game).")

    def _coop_rate(row):
        """Tỉ lệ hợp tác (OptionA) trên cả 2 agent của 1 game."""
        picks = ast.literal_eval(row["agent1_strategies"]) + ast.literal_eval(row["agent2_strategies"])
        return sum(p == "OptionA" for p in picks) / len(picks) if picks else float("nan")

    combined["coop_rate"] = combined.apply(_coop_rate, axis=1)
    piv = combined.pivot_table(index=["model", "language"], columns="scale",
                               values="coop_rate", aggfunc="mean")
    print("\n=== Tỉ lệ hợp tác (OptionA) theo model × language × payoff scale ===")
    print(piv.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nĐối chiếu nhánh frontier: Dataset/data_fairgame_frontier_llm/ (cùng payoff "
          "conventional T=10 R=6 P=2 S=0, nhưng 10 vòng & agent KHÔNG biết số vòng).")
else:
    print("Không có CSV nào — kiểm tra path trong MODELS[] / log Cell 6.")

# =====================================================================
# CELL 8: Zip để download
# =====================================================================
import zipfile  # noqa: E402

zip_path = Path("/kaggle/working/pd_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fp in OUTPUT_DIR.rglob("*"):
        if fp.is_file():
            z.write(fp, fp.relative_to(OUTPUT_DIR.parent))
print(f"{zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB) — tải ở Output tab.")
print("Giải nén rồi copy thư mục <λ>/<model>/ vào Dataset/data_fairgame_small_llm/ "
      "là chạy được Analysis/run_all.py.")
