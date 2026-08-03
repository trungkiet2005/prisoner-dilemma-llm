"""
=====================================================================
FAIRGAME PD — Build wheels dataset cho QUANTIZED inference (RUN INTERNET ON)
=====================================================================
Tải TOÀN BỘ .whl cần cho chạy model quantize (AWQ/GPTQ/bnb 4-bit) trên notebook
GPU Internet OFF: vllm (kèm cả cây phụ thuộc: torch cu12x, triton, nvidia-*...)
+ bitsandbytes. Sau đó Output -> "New Dataset" -> add dataset đó làm Input cho
notebook offline (kaggle/experiments/baseline.py Cell 2.5 tự dò thư mục wheels).

CÁCH CHẠY:
  1. Tạo notebook Kaggle mới — Internet: ON. Accelerator: NÊN chọn ĐÚNG loại
     image sẽ dùng ở notebook offline (cùng Python version -> wheel tag khớp).
     Không cần GPU thật để tải wheels, nhưng image GPU đảm bảo torch/CUDA khớp.
  2. Copy file này vào notebook, chia cell theo "# CELL N", Save & Run All.
  3. Xong: tab Output -> "New Dataset", đặt tên vd "pd-quant-wheels".
  4. Notebook offline: + Add Input dataset đó. Cell cài đặt sẽ tự tìm
     thư mục chứa vllm*.whl và `pip install --no-index --find-links=...`.

LƯU Ý:
  - Wheels PHẢI build cùng Python version với notebook offline (cùng image
    Kaggle thì tự khớp). Đổi image -> chạy lại notebook này.
  - Cây phụ thuộc vllm ~ 5–10 GB (torch + nvidia-* cu12) — trong hạn mức
    dataset Kaggle, nhưng tải mất ~10–20 phút.
=====================================================================
"""

# =====================================================================
# CELL 1: CẤU HÌNH — SỬA Ở ĐÂY
# =====================================================================

# Các package cần cho quantized inference. Version: để trống = bản mới nhất
# (khuyên dùng: RTX PRO 6000 Blackwell sm_120 cần vllm/torch MỚI có kernel cu128).
# Muốn ghim thì viết "vllm==0.11.0" v.v.
PACKAGES = [
    "vllm",           # engine chính — tự nhận AWQ/GPTQ từ checkpoint (kernel marlin built-in)
    "bitsandbytes",   # quantize on-the-fly / checkpoint bnb-4bit (vLLM & transformers)
]

# Tuỳ chọn thêm (mặc định tắt cho nhẹ):
INCLUDE_AUTOAWQ = False   # chỉ cần nếu chạy checkpoint AWQ bằng backend TRANSFORMERS (vLLM thì KHÔNG cần)
INCLUDE_GPTQMODEL = False  # chỉ cần nếu chạy checkpoint GPTQ bằng backend TRANSFORMERS

WHEELS_DIR_NAME = "quant_wheels"   # tên thư mục output (sẽ thành thư mục trong dataset)

# =====================================================================
# CELL 2: Tải wheels (pip download — ưu tiên binary-only để offline cài chắc chắn)
# =====================================================================
import json
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
WHEELS_DIR = OUTPUT_DIR / WHEELS_DIR_NAME
WHEELS_DIR.mkdir(parents=True, exist_ok=True)

packages = list(PACKAGES)
if INCLUDE_AUTOAWQ:
    packages.append("autoawq")
if INCLUDE_GPTQMODEL:
    packages.append("gptqmodel")


def pip_download(specs, only_binary=True):
    """pip download vào WHEELS_DIR. Trả (ok, log-đuôi)."""
    cmd = [sys.executable, "-m", "pip", "download", "--dest", str(WHEELS_DIR)]
    if only_binary:
        # Chỉ lấy wheel: sdist cài offline phải build tại chỗ, dễ vỡ vì thiếu build deps.
        cmd += ["--only-binary", ":all:"]
    cmd += list(specs)
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = (r.stdout + "\n" + r.stderr)[-3000:]
    return r.returncode == 0, tail


print(f"Tải wheels cho: {packages}\n-> {WHEELS_DIR}")
ok, log_tail = pip_download(packages, only_binary=True)
if not ok:
    # Một dep nào đó không có wheel -> thử lại cho phép sdist, nhưng CẢNH BÁO vì
    # sdist cài offline cần build tại chỗ (thường vẫn ổn với package thuần Python).
    print("⚠️ --only-binary thất bại (một dep không có wheel). Log đuôi:")
    print(log_tail)
    print("Thử lại cho phép sdist (package thuần Python thì cài offline vẫn ổn)...")
    ok, log_tail = pip_download(packages, only_binary=False)
    if not ok:
        print(log_tail)
        raise RuntimeError("pip download thất bại — xem log phía trên.")

files = sorted(p for p in WHEELS_DIR.iterdir() if p.is_file())
total_gb = sum(f.stat().st_size for f in files) / 1024**3
n_sdist = sum(1 for f in files if f.suffix in (".gz", ".zip") and not f.name.endswith(".whl"))
print(f"✅ {len(files)} file ({total_gb:.2f} GB), trong đó sdist: {n_sdist}")
for f in files:
    print(f"   {f.name} ({f.stat().st_size / 1024**2:.1f} MB)")

# =====================================================================
# CELL 3: KIỂM CHỨNG cài offline được (pip --dry-run, không đụng env hiện tại)
# =====================================================================
_verify = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--dry-run", "--no-index",
     f"--find-links={WHEELS_DIR}"] + [p.split("==")[0] for p in packages],
    capture_output=True, text=True,
)
if _verify.returncode == 0:
    print("✅ Dry-run OK: bộ wheels TỰ ĐỦ — cài offline được không cần mạng.")
else:
    print("❌ Dry-run FAIL — bộ wheels THIẾU dep, notebook offline sẽ vỡ:")
    print((_verify.stdout + "\n" + _verify.stderr)[-3000:])
    raise RuntimeError("Bộ wheels chưa tự đủ — xem log để biết thiếu package nào.")

# =====================================================================
# CELL 4: Manifest + hướng dẫn cài, rồi tổng kết
# =====================================================================
import platform

manifest = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages_requested": packages,
    "n_files": len(files),
    "total_gb": round(total_gb, 2),
    "files": [f.name for f in files],
}
(WHEELS_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

(WHEELS_DIR / "INSTALL_OFFLINE.md").write_text(f"""# Cài offline từ bộ wheels này

Notebook GPU (Internet OFF), sau khi + Add Input dataset chứa thư mục này:

```python
import subprocess, sys
WHEELS = "/kaggle/input/<ten-dataset>/{WHEELS_DIR_NAME}"   # sửa theo path thực (!ls /kaggle/input)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index",
                       f"--find-links={{WHEELS}}"] + {json.dumps([p.split('==')[0] for p in packages])})
```

Wheels build với Python {platform.python_version()} — notebook offline phải cùng
Python version (dùng cùng image Kaggle là tự khớp).

`kaggle/experiments/baseline.py` (Cell 2.5) tự dò thư mục chứa `vllm*.whl` dưới
/kaggle/input/ nên thường KHÔNG phải sửa path gì.
""", encoding="utf-8")

print(f"""
╔══════════════════════════════════════════════════════════════╗
║                        BƯỚC TIẾP THEO                        ║
╠══════════════════════════════════════════════════════════════╣
║ 1. "Save Version" -> "Save & Run All" và chờ chạy xong.      ║
║ 2. Tab "Output" -> "New Dataset" (vd tên: pd-quant-wheels).  ║
║ 3. Notebook offline (kaggle/experiments/baseline.py):        ║
║    + Add Input dataset vừa tạo — Cell 2.5 tự tìm wheels      ║
║    và cài: pip install --no-index --find-links=... vllm ...  ║
╚══════════════════════════════════════════════════════════════╝
Wheels: {WHEELS_DIR} ({total_gb:.2f} GB, {len(files)} file)
""")
