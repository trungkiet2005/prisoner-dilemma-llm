"""E7: nhúng 5 template Stag Hunt vào kaggle/benchmarks/pd_task.py.

Vì sao phải sinh bằng script chứ không gõ tay
---------------------------------------------
Kaggle Benchmarks push MỘT file duy nhất nên `pd_task.py` không import được repo;
các template phải nằm inline. Nhưng chúng chứa tiếng Ả Rập, Trung, Việt và Pháp,
và bẫy số 1 của dự án này là mở/lưu bằng editor không phải UTF-8 sẽ làm hỏng
**chính prompt gửi cho model**, không chỉ comment. Bản PD đã dính lỗi đó một lần.

Script này đọc thẳng file nguồn rồi phát ra `repr()` của Python, nên không có bước
nào con người gõ lại ký tự non-ASCII. `test_staghunt_templates_match_fairgame_sources`
trong test_pd_task_parity.py so lại từng byte với cùng các file nguồn đó.

Chạy
----
    python Analysis/scripts/51_embed_staghunt.py          # kiểm, không ghi
    python Analysis/scripts/51_embed_staghunt.py --write  # ghi đè block trong pd_task.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO / "FAIRGAME" / "resources" / "game_templates"
PD_TASK = REPO / "kaggle" / "benchmarks" / "pd_task.py"
LANGS = ["en", "fr", "ar", "cn", "vn"]

HEADER = (
    "# Sinh tu FAIRGAME/resources/game_templates/stag_hunt_*.txt bang\n"
    "# Analysis/scripts/51_embed_staghunt.py - KHONG SUA TAY, chay lai script de cap nhat.\n"
    "# Test parity so tung byte voi cac file nguon do.\n"
)


def build_block() -> str:
    lines = [HEADER + "SH_TEMPLATES = {"]
    for lg in LANGS:
        p = TEMPLATE_DIR / f"stag_hunt_{lg}.txt"
        if not p.exists():
            raise SystemExit(f"thieu {p}")
        lines.append(f"    # <- FAIRGAME/resources/game_templates/{p.name}")
        lines.append(f"    {lg!r}: {p.read_text(encoding='utf-8')!r},")
    lines.append("}")
    return "\n".join(lines) + "\n"


def verify(block: str) -> None:
    """Đọc ngược block vừa sinh và so từng byte với file nguồn."""
    ns: dict = {}
    exec(compile(block, "<sh_block>", "exec"), ns)
    for lg in LANGS:
        want = (TEMPLATE_DIR / f"stag_hunt_{lg}.txt").read_text(encoding="utf-8")
        if ns["SH_TEMPLATES"][lg] != want:
            raise SystemExit(f"block sinh ra KHONG khop nguon o ngon ngu {lg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="ghi đè block SH_TEMPLATES trong pd_task.py")
    args = ap.parse_args()

    block = build_block()
    verify(block)
    print(f"sinh {len(block)} ky tu, byte-exact voi {len(LANGS)} file nguon")

    for lg in LANGS:
        t = (TEMPLATE_DIR / f"stag_hunt_{lg}.txt").read_text(encoding="utf-8")
        ph = sorted(set(re.findall(r"\{(\w+)\}", t)))
        print(f"  {lg}: {len(t):>5} ky tu | {len(ph)} placeholder")

    if not args.write:
        print("\n(chi kiem, khong ghi - dung --write de cap nhat pd_task.py)")
        return 0

    src = PD_TASK.read_text(encoding="utf-8")
    pat = re.compile(r"# Sinh tu FAIRGAME.*?\nSH_TEMPLATES = \{.*?\n\}\n", re.S)
    if not pat.search(src):
        raise SystemExit("khong tim thay block SH_TEMPLATES trong pd_task.py")
    # lambda chứ không truyền `block` thẳng: template chứa   và re.sub sẽ
    # diễn giải dấu gạch chéo ngược trong chuỗi thay thế thành escape rồi nổ.
    PD_TASK.write_text(pat.sub(lambda _m: block, src, count=1),
                       encoding="utf-8", newline="\n")
    print(f"\nda ghi vao {PD_TASK.relative_to(REPO)}")
    print("chay lai: PD_SKIP_RUN=1 pytest kaggle/benchmarks/test_pd_task_parity.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
