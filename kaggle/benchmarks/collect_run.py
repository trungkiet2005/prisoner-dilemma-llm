"""Chép output một run Kaggle vào `Dataset/`, có kiểm tra trước khi chép.

Vì sao cần script này thay vì `cp`
----------------------------------
Bước chép tay này đã hỏng hai lần liên tiếp, và cả hai lần đều hỏng IM LẶNG:

1. **Mảnh vỡ của run trước đi kèm** (CLAUDE.md BẪY 11). Kaggle mount output của run
   trước vào run sau để `RESUME` chạy được, nên gói tải về chứa cả thư mục λ mà run
   này không hề chạy. Một lần đã có `0.5/` với đúng 15 dòng lẫn vào giữa 5 thư mục
   1000 dòng.
2. **Hậu tố `model_tag` dính vào tên** (BẪY 9). `RUN_TAG` mặc định `-rep2` bám vào tên
   thư mục model, tên file CSV VÀ cột `agent1_llm`/`agent2_llm` bên trong. Chép nguyên
   xi là `ingest.py` KeyError, hoặc tệ hơn là model bị đếm thành hai model khác nhau.

Cả hai lỗi đều không làm gì báo lỗi cả - chỉ làm bảng phân tích sai.

Dùng
----
    python kaggle/benchmarks/collect_run.py D:/tmp/kbf1 \\
        --dest Dataset/data_fairgame_frontier_llm \\
        --strip="-rep2" \\
        --lambdas 0.01 0.25 2 5 1000

    # xem trước, không chép:
    python kaggle/benchmarks/collect_run.py D:/tmp/kbe2 --dest ... --dry-run

Mặc định **từ chối** thư mục λ không đủ `--expect-rows` dòng (200 = 5 ngôn ngữ × 40
game). Muốn chép một cell chưa đủ thì phải nói rõ bằng `--allow-partial`.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def find_cells(src: Path):
    """(lambda, thư mục model) cho mọi cell trong gói tải về.

    Layout: <bất kỳ>/results/kbench/<model_tag>/<lambda>/<model_tag>/x*.csv
    """
    for d in sorted(src.rglob("x*.csv")):
        model_dir = d.parent
        lam_dir = model_dir.parent
        yield lam_dir.name, model_dir


def count_rows(model_dir: Path) -> int:
    n = 0
    for f in model_dir.glob("x*.csv"):
        with open(f, encoding="utf-8") as fh:
            n += sum(1 for _ in csv.reader(fh)) - 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="thư mục đã tải về (kaggle b t download -o)")
    ap.add_argument("--dest", type=Path, required=True,
                    help="gốc dataset, ví dụ Dataset/data_fairgame_frontier_llm")
    ap.add_argument("--strip", default="",
                    help="hậu tố cần bỏ khỏi tên thư mục, tên file VÀ nội dung CSV. "
                         "Phải viết dạng --strip=\"-rep2\": bắt đầu bằng dấu gạch "
                         "nên argparse hiểu nhầm là một cờ nếu tách bằng khoảng trắng")
    ap.add_argument("--lambdas", nargs="*", default=None,
                    help="chỉ chép các λ này; bỏ trống là chép tất cả")
    ap.add_argument("--expect-rows", type=int, default=200,
                    help="số dòng mỗi λ phải có (mặc định 200 = 5 lang × 40 game)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="chép cả cell thiếu dòng (mặc định TỪ CHỐI)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    dest_root = a.dest if a.dest.is_absolute() else REPO / a.dest
    cells = {}
    for lam, model_dir in find_cells(a.src):
        cells[(lam, model_dir)] = count_rows(model_dir)
    if not cells:
        print(f"khong tim thay CSV nao trong {a.src}")
        return 1

    plan, skipped = [], []
    for (lam, model_dir), rows in sorted(cells.items(), key=lambda kv: float(kv[0][0])):
        if a.lambdas and lam not in a.lambdas:
            skipped.append((lam, model_dir.name, rows, "khong nam trong --lambdas"))
            continue
        if rows < a.expect_rows and not a.allow_partial:
            skipped.append((lam, model_dir.name, rows,
                            f"MANH VO: {rows} < {a.expect_rows} dong"))
            continue
        plan.append((lam, model_dir, rows))

    name_of = lambda s: s.replace(a.strip, "") if a.strip else s  # noqa: E731

    print(f"{'lambda':>7} {'model (sau khi bo hau to)':<40} {'dong':>6}")
    print("-" * 60)
    for lam, model_dir, rows in plan:
        print(f"{lam:>7} {name_of(model_dir.name):<40} {rows:>6}")
    if skipped:
        print("\nBO QUA:")
        for lam, name, rows, why in skipped:
            print(f"  λ={lam:<7} {name:<40} {rows:>5} dong  <- {why}")
    if not plan:
        print("\nkhong co gi de chep")
        return 1
    if a.dry_run:
        print("\n--dry-run: chua chep gi")
        return 0

    n_files = 0
    for lam, model_dir, _ in plan:
        out = dest_root / lam / name_of(model_dir.name)
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(model_dir.glob("x*.csv")):
            target = out / name_of(f.name)
            text = f.read_text(encoding="utf-8")
            if a.strip:
                # Cột agent1_llm/agent2_llm cũng mang hậu tố; để nguyên là cùng một
                # model bị ghi thành hai tên khác nhau trong mọi bảng downstream.
                text = text.replace(a.strip, "")
            target.write_text(text, encoding="utf-8", newline="")
            n_files += 1
    print(f"\nda chep {n_files} file vao {dest_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
