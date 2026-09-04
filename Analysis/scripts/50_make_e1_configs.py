"""E1: sinh config FAIRGAME cho các mức λ ngoài decade (0.25, 0.5, 2, 5).

Vì sao cần bốn mức này
----------------------
Trong lưới decade hiện tại (0.01 ... 1000), "λ < 1" và "ô payoff in ra có dấu thập
phân" là CÙNG MỘT biến cố, nên notation và magnitude không tách được. Bốn mức dưới
đây phá vỡ đồng nhất thức đó mà không cần đổi engine:

    λ = 0.25  ->  0 / 0.5 / 1.5 / 2.5     thập phân, độ lớn giữa 0.1 và 1
    λ = 0.5   ->  0 / 1 / 3 / 5           SỐ NGUYÊN, độ lớn dưới đơn vị  <- ô quyết định
    λ = 2     ->  0 / 4 / 12 / 20         số nguyên
    λ = 5     ->  0 / 10 / 30 / 50        số nguyên

λ = 0.5 là ô mà hai giả thuyết cho dự đoán trái ngược nhau. Nếu notation chi phối thì
coop(0.5) ≈ coop(1); nếu magnitude chi phối thì coop(0.5) nằm giữa coop(0.1) và coop(1).

Dùng cho nhánh nào
------------------
Chỉ nhánh chạy FAIRGAME native (Claude35Haiku / MistralLarge / OpenAIGPT4o) mới đọc
file config. Nhánh Kaggle proxy (`kaggle/benchmarks/pd_task.py`) và nhánh open-weight
(`kaggle/experiments/baseline.py`) đều nhân λ bằng code, nên chỉ cần truyền tham số:

    PD_LAMBDAS=0.25,0.5,2,5          # pd_task.py
    LAMBDAS_OVERRIDE = [0.25, 0.5, 2, 5]   # baseline.py, Cell 1

Chạy
----
    python Analysis/scripts/50_make_e1_configs.py            # sinh + kiểm
    python Analysis/scripts/50_make_e1_configs.py --check    # chỉ kiểm, không ghi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "FAIRGAME" / "resources" / "config"
BASE_CONFIG = "prisoner_dilemma_nocomm_round_known_conventional.json"
OUT_DIR = CONFIG_DIR / "e1_offdecade"

# λ mới của E1. Cố ý KHÔNG gồm các mức decade đã có dữ liệu.
E1_LAMBDAS = [0.25, 0.5, 2, 5]

# Kỳ vọng notation tại từng λ, tính từ ma trận conventional (w1=6, w2=10, w3=0, w4=2).
# Test ở kaggle/benchmarks/test_pd_task_parity.py khoá đúng bảng này.
EXPECTED = {
    0.25: {"cells": "0 / 0.5 / 1.5 / 2.5", "fractional": True},
    0.5: {"cells": "0 / 1 / 3 / 5", "fractional": False},
    2: {"cells": "0 / 4 / 12 / 20", "fractional": False},
    5: {"cells": "0 / 10 / 30 / 50", "fractional": False},
}


def fmt_lambda(lam: float) -> str:
    """Tên thư mục/scale y hệt layout Dataset: 0.25, 0.5, 2, 5 (không thừa '.0').

    Phải khớp `fmt_lambda` của pd_task.py và baseline.py, vì `ingest.py` đọc scale
    bằng `float(scale_dir.name)`.
    """
    lam = float(lam)
    return str(int(lam)) if lam.is_integer() else str(lam)


def scale_weight(value, lam):
    """value × λ, khử nhiễu float, ép int khi nguyên.

    Port đúng `scale_weight` của baseline.py và `scaled_weights` của pd_task.py. Phép
    ép int là thứ quyết định notation: 6×0.5 = 3.0 -> in ra "3", không phải "3.0".
    """
    scaled = round(float(value) * float(lam), 10)
    return int(scaled) if float(scaled).is_integer() else scaled


def printed_cells(weights: dict) -> list[str]:
    """Bốn ô payoff theo thứ tự S / P / R / T, y như prompt hiển thị."""
    return [str(weights[k]) for k in ("weight3", "weight4", "weight1", "weight2")]


def build_config(base: dict, lam: float) -> dict:
    cfg = json.loads(json.dumps(base))          # deep copy
    cfg["payoffMatrix"]["weights"] = {
        k: scale_weight(v, lam) for k, v in base["payoffMatrix"]["weights"].items()
    }
    cfg["name"] = f"{base['name']} (E1, lambda={fmt_lambda(lam)})"
    return cfg


def check(lam: float, weights: dict) -> list[str]:
    """Trả về danh sách lỗi; rỗng nghĩa là đạt."""
    errs = []
    cells = printed_cells(weights)
    got = " / ".join(cells)
    exp = EXPECTED[lam]

    if got != exp["cells"]:
        errs.append(f"ô in ra là {got!r}, kỳ vọng {exp['cells']!r}")

    is_frac = any("." in c for c in cells)
    if is_frac != exp["fractional"]:
        errs.append(f"thập phân={is_frac}, kỳ vọng {exp['fractional']}")

    # Thứ tự PD phải giữ nguyên: T > R > P > S. Đây là cái làm phép nhân λ trở thành
    # một null có định lý bảo chứng - nếu nó gãy thì không còn là cùng một trò chơi.
    T, R, P, S = (weights["weight2"], weights["weight1"],
                  weights["weight4"], weights["weight3"])
    if not (T > R > P > S):
        errs.append(f"thứ tự PD gãy: T={T} R={R} P={P} S={S}")

    # 2R > T + S: điều kiện để hợp tác lặp lại có lợi hơn luân phiên bóc lột.
    if not (2 * R > T + S):
        errs.append(f"2R > T+S gãy: 2*{R} <= {T}+{S}")

    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="chỉ kiểm notation, không ghi file")
    args = ap.parse_args()

    base = json.loads((CONFIG_DIR / BASE_CONFIG).read_text(encoding="utf-8"))
    bw = base["payoffMatrix"]["weights"]
    print(f"Config gốc : {BASE_CONFIG}")
    print(f"Weights gốc: {bw}  ->  ô in ra {' / '.join(printed_cells(bw))}\n")

    if not args.check:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    failed = False
    hdr = f"{'λ':>6}  {'ô in ra':<22} {'thập phân':>10}  {'file':<38} kiểm tra"
    print(hdr)
    print("-" * len(hdr))

    for lam in E1_LAMBDAS:
        cfg = build_config(base, lam)
        w = cfg["payoffMatrix"]["weights"]
        cells = " / ".join(printed_cells(w))
        frac = any("." in c for c in printed_cells(w))
        errs = check(lam, w)

        name = f"prisoner_dilemma_nocomm_round_known_conventional_x{fmt_lambda(lam)}.json"
        if not args.check and not errs:
            (OUT_DIR / name).write_text(
                json.dumps(cfg, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

        status = "OK" if not errs else "LỖI: " + "; ".join(errs)
        if errs:
            failed = True
        print(f"{fmt_lambda(lam):>6}  {cells:<22} {str(frac):>10}  {name:<38} {status}")

    print()
    if failed:
        print("Có cell không đạt kỳ vọng notation - KHÔNG chạy sweep cho tới khi sửa xong.")
        return 1

    if args.check:
        print("Chỉ kiểm, không ghi file (--check).")
    else:
        print(f"Đã ghi {len(E1_LAMBDAS)} config vào {OUT_DIR.relative_to(REPO)}")

    print("\nNhánh không đọc config (nhân λ bằng code) thì truyền tham số:")
    lams = ",".join(fmt_lambda(x) for x in E1_LAMBDAS)
    print(f"  pd_task.py   :  PD_LAMBDAS={lams}")
    print(f"  baseline.py  :  LAMBDAS_OVERRIDE = {E1_LAMBDAS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
