"""
=====================================================================
FAIRGAME - Download & Package Model for Kaggle (RUN WITH INTERNET ON)
=====================================================================
Chạy notebook này trên Kaggle với Internet=ON để:
  1. Download model từ HuggingFace
  2. Download pip wheels cần thiết
  3. Zip tất cả lại
  4. Lưu vào /kaggle/working/ → sau đó "Save & Run" sẽ tự tạo output dataset

Sau khi chạy xong, vào Output tab → "New Dataset" để biến output thành dataset.
=====================================================================
"""

# =====================================================================
# CELL 1: Configuration - CHỈNH SỬA Ở ĐÂY
# =====================================================================

# === Chọn model muốn download ===
# 96GB VRAM: bf16 chỉ vừa tới ~32B (Qwen2.5-32B ≈ 64GB). Model 70B+ bf16 ≈ 140GB
# KHÔNG vừa -> dùng checkpoint QUANTIZE SẴN 4-bit (AWQ/GPTQ, ~40GB) bên dưới.
# Uncomment model bạn muốn:

MODEL_ID = "meta-llama/Llama-3.1-8B"          # ~15GB, nhanh, test trước
# --- bf16 thường (vừa 96GB tới ~32B) ---
# MODEL_ID = "google/gemma-2-27b-it"             # ~54GB, balanced
# MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"         # ~64GB, lớn nhất còn vừa bf16
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"# ~15GB, fast
# --- QUANTIZE SẴN 4-bit (70B+ trên 1 GPU 96GB; chạy bằng kaggle_exp_baseline_72b.py) ---
# MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-AWQ"                       # ~41GB, AWQ int4 (vLLM tự nhận)
# MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" # ~40GB, AWQ int4
# MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"                 # ~41GB, GPTQ int4 (phương án B)
# MODEL_ID = "unsloth/Qwen2.5-72B-Instruct-bnb-4bit"               # ~40GB, bnb NF4 (cần wheel bitsandbytes)
# --- bf16 70B+ (~140GB): CHỈ tải nếu muốn tự quantize fp8, KHÔNG chạy trực tiếp được trên 96GB ---
# MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"         # ~140GB
# MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct" # ~140GB

# HuggingFace token (cần cho Llama, Gemma - lấy từ https://huggingface.co/settings/tokens)
# Cách 1: Paste trực tiếp
HF_TOKEN = ""  # dán HuggingFace token khi chạy, hoặc dùng Kaggle Secrets — KHÔNG commit token thật
# Cách 2: Dùng Kaggle Secrets (recommended) - add secret tên "HF_TOKEN" trong Settings

# Có download pip wheels không? (True nếu Kaggle chưa có sẵn vllm)
DOWNLOAD_WHEELS = False

# Không zip để tiết kiệm thời gian và tránh tốn disk khi nén model lớn.
ZIP_MODEL = False

# =====================================================================
# CELL 2: Install download tools
# =====================================================================
import subprocess
import sys
import os
import shutil
from pathlib import Path

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "huggingface_hub"])

# Try to get HF_TOKEN from Kaggle Secrets ("" hoặc None đều = chưa có token —
# check `is None` cũ khiến nhánh Secrets KHÔNG BAO GIỜ chạy vì mặc định là "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        HF_TOKEN = secrets.get_secret("HF_TOKEN")
        if HF_TOKEN:
            print("✅ HF_TOKEN loaded from Kaggle Secrets")
    except Exception:
        print("ℹ️ No Kaggle Secrets found, proceeding without HF_TOKEN")
HF_TOKEN = HF_TOKEN or None  # "" -> None để snapshot_download dùng token cached/env nếu có

# =====================================================================
# CELL 3: Download Model
# =====================================================================
from urllib.parse import urlparse

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError
import time

OUTPUT_DIR = Path("/kaggle/working")
MODEL_DIR = OUTPUT_DIR / "model_weights"


def normalize_hf_repo_id(model_ref: str) -> str:
    """Accept either a Hugging Face repo_id or full URL and normalize to namespace/repo."""
    cleaned = model_ref.strip()

    if "huggingface.co" not in cleaned:
        return cleaned

    parsed = urlparse(cleaned)
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]

    # Support both:
    # - https://huggingface.co/namespace/repo
    # - https://huggingface.co/models/namespace/repo
    if path_parts and path_parts[0] in {"models", "datasets", "spaces"}:
        path_parts = path_parts[1:]

    if len(path_parts) < 2:
        raise ValueError(
            f"Cannot parse Hugging Face repo from MODEL_ID='{model_ref}'. "
            "Use 'namespace/repo_name' or full URL."
        )

    return f"{path_parts[0]}/{path_parts[1]}"


MODEL_REPO_ID = normalize_hf_repo_id(MODEL_ID)

if MODEL_REPO_ID != MODEL_ID:
    print(f"ℹ️ Normalized MODEL_ID: {MODEL_ID} -> {MODEL_REPO_ID}")

if (MODEL_REPO_ID.startswith("meta-llama/") or MODEL_REPO_ID.startswith("google/gemma")) and not HF_TOKEN:
    print("⚠️ This model is likely gated. Add HF_TOKEN in Kaggle Secrets to avoid 401/403 errors.")

print(f"📥 Downloading model: {MODEL_ID}")
print(f"📂 Saving to: {MODEL_DIR}")
start_time = time.time()

try:
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN,
        resume_download=True,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "tokenizer*",
            "*.model",
        ],
        ignore_patterns=[
            "*.pth",
            "*.bin",
            "original/*",
            "consolidated*",
        ],
    )
except HFValidationError as e:
    raise ValueError(
        "Invalid MODEL_ID format. Use 'namespace/repo_name' "
        "(example: 'meta-llama/Llama-3.1-8B-Instruct') "
        "or a full Hugging Face URL."
    ) from e

elapsed = time.time() - start_time
print(f"✅ Model downloaded in {elapsed/60:.1f} minutes")

# Print size
total_size = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, fns in os.walk(MODEL_DIR)
    for f in fns
)
print(f"📦 Model size: {total_size / (1024**3):.2f} GB")

# List files
print("\n📁 Model files:")
for p in sorted(MODEL_DIR.rglob("*")):
    if p.is_file():
        size_mb = p.stat().st_size / (1024**2)
        print(f"   {p.relative_to(MODEL_DIR)} ({size_mb:.1f} MB)")

# =====================================================================
# CELL 4: Download Pip Wheels (optional)
# =====================================================================
if DOWNLOAD_WHEELS:
    WHEELS_DIR = OUTPUT_DIR / "pip_wheels"
    WHEELS_DIR.mkdir(exist_ok=True)

    print("\n📥 Downloading pip wheels for offline install...")

    # Packages that Kaggle might NOT have pre-installed
    extra_packages = [
        "vllm",
        "python-dotenv",
        "decorator",
        "py",
    ]

    # Download binary wheels for Linux
    for pkg in extra_packages:
        print(f"   Downloading {pkg}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "download",
                "--dest", str(WHEELS_DIR),
                pkg,
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ Failed to download {pkg}: {e.stderr[:200]}")

    wheel_files = list(WHEELS_DIR.glob("*"))
    total_wheels_size = sum(f.stat().st_size for f in wheel_files if f.is_file())
    print(f"\n✅ Downloaded {len(wheel_files)} wheel files ({total_wheels_size/(1024**2):.1f} MB)")
    for wf in sorted(wheel_files):
        print(f"   {wf.name}")

# =====================================================================
# CELL 5: Skip zip (keep folders as-is)
# =====================================================================
if ZIP_MODEL:
    print("\nℹ️ ZIP_MODEL=True nhưng bản script hiện đang để workflow không nén.")
else:
    print("\nℹ️ Skipping zip step. Keeping model folder and wheel folder uncompressed.")

# =====================================================================
# CELL 6: Summary & Next Steps
# =====================================================================
print("\n" + "=" * 60)
print("🎉 DONE! Output files in /kaggle/working/:")
print("=" * 60)

for f in sorted(OUTPUT_DIR.iterdir()):
    if f.is_file():
        size = f.stat().st_size
        if size > 1024**3:
            print(f"   📦 {f.name} ({size/(1024**3):.2f} GB)")
        else:
            print(f"   📦 {f.name} ({size/(1024**2):.1f} MB)")

print(f"""
╔══════════════════════════════════════════════════════════╗
║                    NEXT STEPS                            ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. Click "Save Version" (phải trên) → "Save & Run All" ║
║                                                          ║
║  2. Sau khi notebook chạy xong, vào tab "Output"         ║
║                                                          ║
║  3. Click "New Dataset" để tạo dataset từ output          ║
║     - Đặt tên: "fairgame-model" cho folder model_weights  ║
║     - Đặt tên: "fairgame-wheels" cho folder pip_wheels    ║
║                                                          ║
║  4. Dataset sẽ available tại:                             ║
║     /kaggle/input/fairgame-model/model_weights/          ║
║     /kaggle/input/fairgame-wheels/pip_wheels/            ║
║                                                          ║
║  5. Trong notebook chạy FAIRGAME (Internet OFF),          ║
║     load model trực tiếp từ folder:                       ║
║                                                          ║
║     MODEL_PATH = '/kaggle/input/fairgame-model/model_weights' ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

Model: {MODEL_ID}
""")
