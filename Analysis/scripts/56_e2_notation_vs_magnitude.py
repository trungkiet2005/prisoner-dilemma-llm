"""E2: tách hiệu ứng NOTATION khỏi hiệu ứng MAGNITUDE, và tách cả hai khỏi nhiễu.

Vấn đề E2 giải quyết
--------------------
Trên lưới decade, "λ < 1" và "ô payoff in ra có dấu chấm thập phân" là **cùng một sự
kiện**. Mọi phân tích quan sát trên lưới đó - kể cả thang BIC ở `53_e1_ladder.py` - đều
không tách được hai giả thuyết, vì không có ô nào mà hai thứ đó rời nhau.

`PD_WEIGHT_FORMAT` trong `pd_task.py` cắt đúng ràng buộc đó. Ba nhánh:

===========  ===============================  ==========================================
cách in      λ=1 in ra                        vai trò
===========  ===============================  ==========================================
``native``   ``0 / 2 / 10 / 6``               gốc FAIRGAME
``dec2``     ``0.00 / 2.00 / 10.00 / 6.00``   ô nguyên thành thập phân
``dec3``     ``0.000 / 2.000 / 10.000 / ...`` thêm đúng một số 0 so với dec2
===========  ===============================  ==========================================

Ba phép so, và phép thứ hai là phép phân định
---------------------------------------------
- **A1 - ranh giới notation** (`native` với `dec2`): cùng λ nên payoff y hệt; ô nguyên
  chuyển thành thập phân.
- **A2 - PLACEBO THẬT** (`dec2` với `dec3`): cùng λ, và **mọi ô ở cả hai bên đều đã là
  thập phân**. Khác biệt duy nhất là một số 0 ở đuôi. Không có ranh giới nào bị vượt.
- **B - magnitude với notation cố định**: trong cùng một cách in, so các bước λ.

Quy tắc đọc:

- |A1| lớn và |A2| ~ 0  ⇒ ranh giới thập phân thật sự quan trọng.
- |A1| ~ |A2|           ⇒ thứ đo được chỉ là **độ nhạy với chuỗi ký tự payoff nói
  chung**, không phải ranh giới. Khi đó phát biểu "notation quan trọng" là quá mạnh.

Vì sao cần A2: bản đầu của script này dùng λ=0.1 làm placebo, nhưng đó **không phải
placebo sạch**. S = 0 trong mọi cấu hình, nên `native` luôn in `0` còn `dec2` in `0.00`
- tức là mọi λ đều có ít nhất một ô vượt ranh giới nguyên→thập phân. Không có λ nào
tránh được điều đó, nên placebo phải nằm ở trục CÁCH IN chứ không phải trục λ.

Vì sao KHÔNG dùng Analysis/pdlib/ingest.py
------------------------------------------
Cùng lý do với `52_e7_staghunt_stats.py`: thư mục model ở đây mang hậu tố `-dec2` /
`-dec3`, mà `ingest.py:147` tra `MODEL_MAP[model_dir.name]` trực tiếp nên sẽ KeyError và
làm hỏng cả lần ingest. Dữ liệu E2 vì thế để riêng ngoài
`Dataset/data_fairgame_frontier_llm/`, để pipeline chính không bao giờ trộn hai cách in
vào cùng một ô λ.

Chạy
----
    python Analysis/scripts/56_e2_notation_vs_magnitude.py
"""
from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
NATIVE_ROOT = REPO / "Dataset" / "data_fairgame_frontier_llm"
FMT_ROOT = REPO / "Dataset" / "data_fairgame_e2_notation"
OUT = REPO / "Analysis" / "tables" / "T_E2_notation.csv"

# PD dùng khung HÌNH PHẠT và mục tiêu là tối thiểu hoá, nên OptionA (6/0) trội tuyệt
# đối so với OptionB (10/2) => OptionA = phản bội, OptionB = HỢP TÁC. Ngược với Stag
# Hunt. Xem CLAUDE.md BẪY 8.
COOP = "OptionB"
BASE = {"S": 0, "P": 2, "T": 10, "R": 6}   # weight3 / weight4 / weight2 / weight1
LAMBDAS = [0.1, 1, 10]
FORMATS = ["native", "dec2", "dec3"]
# Cặp so ở trục CÁCH IN, và ý nghĩa của từng cặp.
FMT_PAIRS = [("native", "dec2", "ranh gioi: o nguyen -> thap phan"),
             ("dec2", "dec3", "PLACEBO THAT: ca hai deu thap phan, chi khac mot so 0")]
MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite-preview"]
N_BOOT = 8000
SEED = 20260903
# Sàn nhiễu đo từ replicate (55_replicate_testretest.py): |Δ| trung vị 0,008, lớn nhất
# 0,038. Hiệu ứng dưới mức này không được diễn giải.
NOISE_FLOOR = 0.04


def printed(lam: float, fmt: str) -> str:
    """Bốn ô đúng như prompt in ra, thứ tự S / P / T / R."""
    def f(v):
        x = round(v * lam, 10)
        if fmt == "native":
            return str(int(x)) if float(x).is_integer() else f"{x:g}"
        return f"{float(x):.{int(fmt[3:])}f}"
    return " / ".join(f(BASE[k]) for k in ("S", "P", "T", "R"))


def dyad_rates(d: Path) -> np.ndarray:
    """Một phần tử = một dyad = tỉ lệ hợp tác trung bình của cả hai agent."""
    out = []
    for f in sorted(d.glob("x*.csv")):
        with open(f, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                moves = (ast.literal_eval(row["agent1_strategies"])
                         + ast.literal_eval(row["agent2_strategies"]))
                out.append(np.mean([m == COOP for m in moves]))
    return np.array(out)


def boot_diff(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float]:
    """Chênh lệch trung bình b - a, và p hai phía từ bootstrap theo dyad."""
    d = np.array([rng.choice(b, len(b), True).mean() - rng.choice(a, len(a), True).mean()
                  for _ in range(N_BOOT)])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(b.mean() - a.mean()), float(min(p, 1.0))


def load(model: str, lam: float, fmt: str):
    lam_s = str(int(lam)) if float(lam).is_integer() else str(lam)
    d = (NATIVE_ROOT / lam_s / model if fmt == "native"
         else FMT_ROOT / lam_s / f"{model}-{fmt}")
    if not d.exists():
        return None, str(d.relative_to(REPO))
    v = dyad_rates(d)
    return (v if len(v) else None), str(d.relative_to(REPO))


def main() -> int:
    rng = np.random.default_rng(SEED)
    cells, missing = {}, []
    for model in MODELS:
        for lam in LAMBDAS:
            for fmt in FORMATS:
                v, path = load(model, lam, fmt)
                if v is None:
                    missing.append(path)
                else:
                    cells[(model, lam, fmt)] = v

    if missing:
        print("THIEU du lieu (cac phep so lien quan se bi bo qua):")
        for m in missing:
            print("  ", m)
        print()

    rows = []
    for model in MODELS:
        have_fmt = [f for f in FORMATS
                    if all((model, l, f) in cells for l in LAMBDAS)]
        if len(have_fmt) < 2:
            continue
        print("=" * 78)
        print(f"{model}  -  ti le hop tac (OptionB = hop tac)")
        print("=" * 78)
        print(f"{'lambda':>7} {'cach in':>7} {'o in ra (S/P/T/R)':<36} {'coop':>7}  n")
        print("-" * 78)
        for lam in LAMBDAS:
            for fmt in have_fmt:
                v = cells[(model, lam, fmt)]
                print(f"{lam:>7g} {fmt:>7} {printed(lam, fmt):<36} "
                      f"{v.mean():>7.3f}  {len(v)}")

        print("\nA. CACH IN (cung lambda -> payoff Y HET, chi khac chuoi in ra)")
        print(f"{'lambda':>7} {'cap':>14} {'delta':>8} {'p':>7}  y nghia cua cap")
        print("-" * 78)
        summary = {}
        for lo, hi, meaning in FMT_PAIRS:
            if lo not in have_fmt or hi not in have_fmt:
                continue
            deltas = []
            for lam in LAMBDAS:
                delta, p = boot_diff(cells[(model, lam, lo)], cells[(model, lam, hi)], rng)
                deltas.append(delta)
                print(f"{lam:>7g} {f'{lo}->{hi}':>14} {delta:>+8.3f} {p:>7.3f}  {meaning}")
                rows.append(dict(contrast="format", model=model, lam=lam,
                                 pair=f"{lo}->{hi}", printed_lo=printed(lam, lo),
                                 printed_hi=printed(lam, hi),
                                 coop_lo=round(float(cells[(model, lam, lo)].mean()), 4),
                                 coop_hi=round(float(cells[(model, lam, hi)].mean()), 4),
                                 delta=round(delta, 4), p=round(p, 4)))
            summary[(lo, hi)] = (float(np.mean(np.abs(deltas))),
                                 all(d > 0 for d in deltas)
                                 or all(d < 0 for d in deltas))

        if ("native", "dec2") in summary and ("dec2", "dec3") in summary:
            a1, a1_same_sign = summary[("native", "dec2")]
            a2, _ = summary[("dec2", "dec3")]
            print("")
            print("  ranh gioi (native->dec2): |delta| tb = %.3f, dau %s"
                  % (a1, "NHAT QUAN" if a1_same_sign else "DOI CHIEU"))
            print("  PLACEBO   (dec2->dec3)  : |delta| tb = %.3f" % a2)
            # Ba dieu kien, va DAU la dieu kien khong duoc bo qua. Mot hieu ung that
            # phai day ve CUNG MOT PHIA o moi lambda; |delta| trung binh lon ma dau
            # doi chieu chinh la dang cua NHIEU, khong phai cua hieu ung. Ban dau
            # script chi so |delta| nen ket luan "notation THAT" cho ca model co dau
            # doi chieu - qua rong rai.
            if not a1_same_sign:
                print("  => dau DOI CHIEU giua cac lambda: day khong phai hinh dang")
                print("     cua mot hieu ung ranh gioi. Khong ket luan ve notation.")
            elif a1 <= NOISE_FLOOR:
                print("  => duoi san nhieu %.2f: khong ket luan duoc gi." % NOISE_FLOOR)
            elif a2 >= 0.5 * a1:
                print("  => PLACEBO cung dich gan bang ranh gioi: cai do duoc la DO")
                print("     NHAY VOI CHUOI PAYOFF noi chung, KHONG phai ranh gioi.")
            else:
                print("  => dau nhat quan, tren san nhieu, placebo im: hieu ung ranh")
                print("     gioi notation la THAT tren model nay.")

        print("\nB. MAGNITUDE (trong CUNG mot cach in -> glyph cung loai)")
        hdr = f"{'buoc':>14} {'x mag':>6}"
        for f in have_fmt:
            hdr += f" {f + ' delta':>13} {'p':>7}"
        print(hdr)
        print("-" * 78)
        for x, y in zip(LAMBDAS, LAMBDAS[1:]):
            line = f"{f'{x:g} -> {y:g}':>14} {y / x:>6g}"
            for f in have_fmt:
                d, p = boot_diff(cells[(model, x, f)], cells[(model, y, f)], rng)
                line += f" {d:>+13.3f} {p:>7.3f}"
                rows.append(dict(contrast="magnitude", model=model,
                                 lam=f"{x:g}->{y:g}", pair=f,
                                 printed_lo=printed(x, f), printed_hi=printed(y, f),
                                 coop_lo=round(float(cells[(model, x, f)].mean()), 4),
                                 coop_hi=round(float(cells[(model, y, f)].mean()), 4),
                                 delta=round(d, 4), p=round(p, 4)))
            print(line)
        print()

    if not rows:
        print("khong du du lieu de so sanh")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"da ghi {OUT.relative_to(REPO)}")
    print(f"\nSan nhieu do tu replicate: |delta| ~ {NOISE_FLOOR}. Moi hieu ung duoi muc")
    print("do khong duoc dien giai, du p co nho den may.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
