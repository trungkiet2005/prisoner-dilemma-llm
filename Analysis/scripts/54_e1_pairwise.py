"""E1/E3: lambda ngoai decade - tach notation khoi magnitude.

Bang T_E1_offdecade.csv da co tu truoc nhung KHONG kem kiem dinh nao, nen khong
tra loi duoc cau hoi thuc su cua E1: buoc nhay khi vuot ranh gioi notation co lon
hon cac buoc trong CUNG mot notation hay khong.

Gia thuyet notation du doan hai dieu, va ca hai deu kiem duoc o day:
  (a) coop(0.5) ~ coop(1)                - hai o cung in so nguyen thi phai gan nhau
  (b) buoc 0.25 -> 0.5 la buoc lon nhat  - vi do la buoc duy nhat doi notation

Neu mot buoc TRONG cung notation lon hon buoc vuot ranh gioi, gia thuyet notation
khong duoc du lieu ung ho.

Vi sao KHONG dung Analysis/pdlib/ingest.py: ingest phuc vu master table cua bai
chinh va gan MODEL_MAP cung. Script nay chi doc raw CSV. Nhan hanh dong lay theo
ingest.py:51 - OptionB = hop tac (prompt dien dat payoff la hinh phat, nen OptionA
la chien luoc troi = phan boi).

Chay
----
    python Analysis/scripts/53_e1_offdecade_stats.py
"""
from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "Dataset" / "data_fairgame_frontier_llm"
TABDIR = REPO / "Analysis" / "tables"
OUT_CELLS = TABDIR / "T_E1_offdecade.csv"
OUT_PAIRS = TABDIR / "T_E1_offdecade_pairs.csv"

COOP = "OptionB"

# prisoner_dilemma_nocomm_round_known_conventional.json, da sap xep tang dan.
BASE = (0, 2, 6, 10)
N_BOOT = 8000
SEED = 12345

# TU DO TIM model, khong hard-code. Danh sach cung o day tung khien mot sweep moi
# chay xong ma khong xuat hien trong bang, va khong co gi bao - bang van in ra binh
# thuong voi cac model cu. Dieu kien duy nhat de vao bang la co du buoc de so: it
# nhat MIN_LAMBDAS muc lambda. Ba model cu (claude, gpt, mistra) chi co 3 muc tren
# luoi decade nen tu rot ra.
MIN_LAMBDAS = 5


def discover_models(root):
    """Ten thu muc model co it nhat MIN_LAMBDAS muc lambda, sap xep on dinh."""
    count = {}
    for lam_dir in root.iterdir():
        if not lam_dir.is_dir():
            continue
        for mdir in lam_dir.iterdir():
            if mdir.is_dir():
                count[mdir.name] = count.get(mdir.name, 0) + 1
    return tuple(sorted(m for m, n in count.items() if n >= MIN_LAMBDAS))

# Luoi 3 decade cua ban thao goc. Moi muc khac la o chay them cho E1/E3, va cot
# `source` giu lai phan biet do de bang con doc duoc khi tach khoi RUN_PLAN.
ORIGINAL_GRID = (0.1, 1.0, 10.0)


def printed_cells(lam):
    """Bon o dung nhu prompt in ra, va nhan notation doc tu chinh cac o do."""
    def f(v):
        x = round(v * lam, 10)
        return str(int(x)) if float(x).is_integer() else "%g" % x
    cells = [f(w) for w in BASE]
    return " / ".join(cells), ("decimal" if any("." in c for c in cells) else "integer")


def dyad_coop_rates(model_dir):
    """Mot phan tu = mot dyad = ti le hop tac trung binh cua ca hai agent."""
    out = []
    for f in sorted(model_dir.glob("x*.csv")):
        with open(f, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                moves = (ast.literal_eval(row["agent1_strategies"])
                         + ast.literal_eval(row["agent2_strategies"]))
                if moves:
                    out.append(np.mean([m == COOP for m in moves]))
    return np.array(out)


def boot_means(v, rng):
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    return v[idx].mean(axis=1)


def two_sided_p(dbs):
    return float(min(2 * min((dbs <= 0).mean(), (dbs >= 0).mean()), 1.0))


def main():
    if not ROOT.exists():
        print("chua co du lieu: %s" % ROOT)
        return 1

    rng = np.random.default_rng(SEED)
    MODELS = discover_models(ROOT)
    if not MODELS:
        print("khong model nao co du %d muc lambda" % MIN_LAMBDAS)
        return 1
    print("model co du do phan giai lambda: " + ", ".join(MODELS))
    print()
    cells = dict((m, []) for m in MODELS)

    for lam_dir in sorted(ROOT.iterdir(), key=lambda p: float(p.name)):
        lam = float(lam_dir.name)
        for model in MODELS:
            mdir = lam_dir / model
            if not mdir.is_dir():
                continue
            v = dyad_coop_rates(mdir)
            if not len(v):
                continue
            bs = boot_means(v, rng)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            printed, nota = printed_cells(lam)
            cells[model].append(dict(
                model=model, lam=lam, printed_cells=printed, notation=nota,
                coop=round(float(v.mean()), 4), lo=round(float(lo), 4),
                hi=round(float(hi), 4), n_dyads=len(v),
                source=("existing" if lam in ORIGINAL_GRID else "E1/E3 new"),
                _bs=bs))

    rows = [dict((k, r[k]) for k in r if not k.startswith("_"))
            for m in MODELS for r in cells[m]]
    if not rows:
        print("khong tim thay CSV nao")
        return 1

    TABDIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CELLS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- buoc nhay giua cac muc lambda lien tiep -----------------------------
    pairs = []
    for model in MODELS:
        seq = cells[model]
        for a, b in zip(seq, seq[1:]):
            dbs = b["_bs"] - a["_bs"]
            pairs.append(dict(
                model=model, lam_from=a["lam"], lam_to=b["lam"],
                magnitude_ratio=round(b["lam"] / a["lam"], 3),
                notation_from=a["notation"], notation_to=b["notation"],
                crosses_notation=int(a["notation"] != b["notation"]),
                delta=round(b["coop"] - a["coop"], 4),
                lo=round(float(np.percentile(dbs, 2.5)), 4),
                hi=round(float(np.percentile(dbs, 97.5)), 4),
                p=round(two_sided_p(dbs), 4)))

    with open(OUT_PAIRS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)

    # ---- bao cao ------------------------------------------------------------
    for model in MODELS:
        seq = cells[model]
        if not seq:
            continue
        print("\n" + "=" * 78)
        print("%s  -  ti le hop tac (OptionB = hop tac)" % model)
        print("=" * 78)
        print("%8s %-26s %9s %7s  95%% CI" % ("lambda", "in ra", "notation", "coop"))
        print("-" * 78)
        for r in seq:
            print("%8g %-26s %9s %7.3f  [%.3f, %.3f]"
                  % (r["lam"], r["printed_cells"], r["notation"],
                     r["coop"], r["lo"], r["hi"]))

        pm = [p for p in pairs if p["model"] == model]
        print("\n%-18s %7s %8s %7s   notation" % ("buoc", "x mag", "delta", "p"))
        print("-" * 78)
        for p in pm:
            mark = "<<< VUOT RANH GIOI" if p["crosses_notation"] else ""
            print("%g -> %-13g %7g %+8.3f %7.3f   %s"
                  % (p["lam_from"], p["lam_to"], p["magnitude_ratio"],
                     p["delta"], p["p"], mark))

        # Phep kiem phan dinh cua E1.
        cross = [p for p in pm if p["crosses_notation"]]
        within = [p for p in pm if not p["crosses_notation"]]
        if cross and within:
            c = max(cross, key=lambda p: abs(p["delta"]))
            wn = max(within, key=lambda p: abs(p["delta"]))
            print("\n  buoc vuot notation lon nhat : %g->%g  |delta|=%.3f  p=%.3f"
                  % (c["lam_from"], c["lam_to"], abs(c["delta"]), c["p"]))
            print("  buoc trong cung notation lon: %g->%g  |delta|=%.3f  p=%.3f"
                  % (wn["lam_from"], wn["lam_to"], abs(wn["delta"]), wn["p"]))
            ok = abs(c["delta"]) > abs(wn["delta"])
            print("  => %s" % ("notation DUOC ung ho" if ok else
                               "notation KHONG duoc ung ho: buoc trong cung "
                               "notation lon hon buoc vuot ranh gioi"))

        # Du doan (a): coop(0.5) ~ coop(1).
        by_lam = dict((r["lam"], r) for r in seq)
        if 0.5 in by_lam and 1.0 in by_lam:
            dbs = by_lam[1.0]["_bs"] - by_lam[0.5]["_bs"]
            p = two_sided_p(dbs)
            print("  coop(0.5)=%.3f vs coop(1)=%.3f  delta=%+.3f  p=%.3f   (%s)"
                  % (by_lam[0.5]["coop"], by_lam[1.0]["coop"], dbs.mean(), p,
                     "gan nhau nhu notation du doan" if p > 0.05 else "khac nhau"))

    print("\nda ghi %s" % OUT_CELLS.relative_to(REPO))
    print("da ghi %s" % OUT_PAIRS.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
