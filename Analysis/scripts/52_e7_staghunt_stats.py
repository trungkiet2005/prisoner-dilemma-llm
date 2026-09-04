"""E7: tỉ lệ chọn Stag theo payoff scale, nhánh Stag Hunt.

Vì sao KHÔNG dùng Analysis/pdlib/ingest.py
------------------------------------------
`ingest.py:51` cố định `ACTION_MAP = {"OptionA": "D", "OptionB": "C"}` ở phạm vi
module, vì trong Prisoner's Dilemma OptionA là *phản bội*. Trong Stag Hunt thì
ngược lại: weight1 (phần thưởng cao nhất) rơi vào combination1 = cả hai chọn
strategy1, nên **OptionA = Stag = hợp tác**.

Đẩy log Stag Hunt qua ingest.py mà không sửa sẽ cho ra `1 - x` cho mọi tỉ lệ, và
**không có lỗi nào được ném ra**. Vì thế script này tự đọc CSV với bảng ánh xạ
riêng, và dữ liệu Stag Hunt được để ngoài `Dataset/data_fairgame_*` để pipeline
của bài không bao giờ chạm vào nó.

Chạy
----
    python Analysis/scripts/52_e7_staghunt_stats.py
"""
from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SH_ROOT = REPO / "Dataset" / "data_stag_hunt_frontier"
OUT = REPO / "Analysis" / "tables" / "T_E7_staghunt.csv"

# Stag Hunt: OptionA = Stag = hợp tác. NGƯỢC với PD.
STAG = "OptionA"

# FAIRGAME/resources/config/stag_hunt_nocomm_round_known_conventional.json
BASE = {"R": 8, "T": 6, "P": 4, "S": 0}
N_BOOT = 4000
SEED = 12345


def printed_cells(lam: float) -> tuple[str, str]:
    """Bốn ô như prompt in ra, theo thứ tự S / P / T / R, và nhãn notation."""
    def f(v):
        x = round(v * lam, 10)
        return str(int(x)) if float(x).is_integer() else f"{x:g}"
    cells = [f(BASE[k]) for k in ("S", "P", "T", "R")]
    return " / ".join(cells), ("decimal" if any("." in c for c in cells) else "integer")


def dyad_stag_rates(model_dir: Path) -> np.ndarray:
    """Một phần tử = một dyad = tỉ lệ chọn Stag trung bình của cả hai agent."""
    out = []
    for f in sorted(model_dir.glob("x*.csv")):
        with open(f, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                moves = (ast.literal_eval(row["agent1_strategies"])
                         + ast.literal_eval(row["agent2_strategies"]))
                out.append(np.mean([m == STAG for m in moves]))
    return np.array(out)


def main() -> int:
    if not SH_ROOT.exists():
        print(f"chua co du lieu: {SH_ROOT}")
        return 1

    rng = np.random.default_rng(SEED)
    rows = []
    for lam_dir in sorted(SH_ROOT.iterdir(), key=lambda p: float(p.name)):
        for model_dir in sorted(lam_dir.iterdir()):
            v = dyad_stag_rates(model_dir)
            if not len(v):
                continue
            bs = np.array([rng.choice(v, len(v), True).mean() for _ in range(N_BOOT)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells, nota = printed_cells(float(lam_dir.name))
            rows.append(dict(game="stag_hunt", model=model_dir.name,
                             lam=float(lam_dir.name), printed_cells=cells,
                             notation=nota, stag_rate=round(float(v.mean()), 4),
                             lo=round(float(lo), 4), hi=round(float(hi), 4),
                             n_dyads=len(v)))

    if not rows:
        print("khong tim thay CSV nao")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("STAG HUNT - ti le chon Stag (OptionA = Stag = hop tac)\n")
    print(f"{'lambda':>7} {'in ra (S/P/T/R)':<22} {'notation':>9} {'stag':>7}  95% CI")
    print("-" * 70)
    for r in rows:
        print(f"{r['lam']:>7g} {r['printed_cells']:<22} {r['notation']:>9} "
              f"{r['stag_rate']:>7.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]")

    # Phep kiem phan dinh: buoc nhay khi vuot ranh gioi notation so voi cac buoc khac.
    print("\nBUOC NHAY GIUA CAC MUC LIEN TIEP")
    for a, b in zip(rows, rows[1:]):
        if a["model"] != b["model"]:
            continue
        flip = a["notation"] != b["notation"]
        print(f"  {a['lam']:>5g} -> {b['lam']:<5g} {b['stag_rate']-a['stag_rate']:+.3f}"
              f"   {'<<< VUOT RANH GIOI NOTATION' if flip else ''}")
    print(f"\nda ghi {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
