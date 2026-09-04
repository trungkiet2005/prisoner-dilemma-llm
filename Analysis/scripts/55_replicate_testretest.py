"""Replicate: do san nhieu giua hai lan chay doc lap o CUNG mot lambda.

Vi sao can
----------
Thang BIC tren luoi lambda mo rong chon `saturated in lambda` (thang 45,7 diem so
voi dac ta tot thu hai). Ket luan rut ra la "phan ung theo lambda co that nhung
khong tron". Nhung ca corpus KHONG co replicate nao o cung mot lambda, nen chua co
gi phan biet ket luan do voi "mo hinh bao hoa dang fit nhieu cua tung o".

Script nay so dot goc (BASE_SEED=12345) voi dot replicate (BASE_SEED=67890) o bon
muc lambda 0.1 / 0.25 / 0.5 / 1, va tra loi hai cau:

  1. San nhieu. |delta| giua hai lan chay o cung lambda lon co nao?
  2. Buoc 0.1 -> 0.25 (+0,178 o dot goc, buoc lon nhat toan luoi va la tru cot cua
     ket luan "khong tron") co lap lai khong?

Neu san nhieu ngang bien do cac buoc thi dong gop (i) cua ban thao khong dung vung,
va thang BIC phai duoc doc lai voi mot thanh phan nhieu theo o.

Chay
----
    python Analysis/scripts/55_replicate_testretest.py
"""
from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ORIG_ROOT = REPO / "Dataset" / "data_fairgame_frontier_llm"
REP_ROOT = REPO / "Dataset" / "data_replicate_frontier"
OUT = REPO / "Analysis" / "tables" / "T_R1_testretest.csv"

COOP = "OptionB"
ORIG_MODEL = "gemini-3.5-flash-lite"
REP_MODEL = "gemini-3.5-flash-lite-rep2"
LAMS = (0.1, 0.25, 0.5, 1.0)
N_BOOT = 8000
SEED = 12345


def lam_dirname(lam):
    return str(int(lam)) if float(lam).is_integer() else ("%g" % lam)


def dyad_coop_rates(d):
    out = []
    for f in sorted(d.glob("x*.csv")):
        with open(f, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                moves = (ast.literal_eval(row["agent1_strategies"])
                         + ast.literal_eval(row["agent2_strategies"]))
                if moves:
                    out.append(np.mean([m == COOP for m in moves]))
    return np.array(out)


def boot(v, rng):
    return v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)


def two_sided_p(d):
    return float(min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0))


def main():
    if not REP_ROOT.exists():
        print("chua co du lieu replicate: %s" % REP_ROOT)
        print("Thu thap bang: python run_and_watch.py --attach "
              "--collect-into Dataset/data_replicate_frontier")
        return 1

    rng = np.random.default_rng(SEED)
    rows, keep = [], {}

    for lam in LAMS:
        dn = lam_dirname(lam)
        a_dir = ORIG_ROOT / dn / ORIG_MODEL
        b_dir = REP_ROOT / dn / REP_MODEL
        if not a_dir.is_dir() or not b_dir.is_dir():
            print("bo qua lambda=%g (thieu %s)"
                  % (lam, a_dir if not a_dir.is_dir() else b_dir))
            continue
        a, b = dyad_coop_rates(a_dir), dyad_coop_rates(b_dir)
        if not len(a) or not len(b):
            continue
        ba, bb = boot(a, rng), boot(b, rng)
        d = bb - ba
        keep[lam] = (ba, bb)
        rows.append(dict(
            lam=lam, coop_run1=round(float(a.mean()), 4),
            coop_run2=round(float(b.mean()), 4),
            delta=round(float(b.mean() - a.mean()), 4),
            lo=round(float(np.percentile(d, 2.5)), 4),
            hi=round(float(np.percentile(d, 97.5)), 4),
            p=round(two_sided_p(d), 4), n1=len(a), n2=len(b)))

    if not rows:
        print("khong ghep duoc o nao")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("SAN NHIEU GIUA HAI LAN CHAY DOC LAP (cung lambda, seed khac)\n")
    print("%8s %10s %10s %9s %9s   %s"
          % ("lambda", "run1", "run2", "delta", "p", "95% CI cua delta"))
    print("-" * 74)
    for r in rows:
        flag = "  <-- KHAC NHAU" if r["p"] < 0.05 else ""
        print("%8g %10.3f %10.3f %+9.3f %9.3f   [%+.3f, %+.3f]%s"
              % (r["lam"], r["coop_run1"], r["coop_run2"], r["delta"],
                 r["p"], r["lo"], r["hi"], flag))

    noise = float(np.median([abs(r["delta"]) for r in rows]))
    worst = max(rows, key=lambda r: abs(r["delta"]))
    print("\n  |delta| trung vi   : %.3f" % noise)
    print("  |delta| lon nhat   : %.3f (lambda=%g, p=%.3f)"
          % (abs(worst["delta"]), worst["lam"], worst["p"]))
    print("  so o khac nhau     : %d/%d" % (sum(r["p"] < 0.05 for r in rows), len(rows)))

    # ---- Buoc 0.1 -> 0.25 co lap lai khong? ---------------------------------
    if 0.1 in keep and 0.25 in keep:
        s1 = keep[0.25][0] - keep[0.1][0]      # buoc o dot goc
        s2 = keep[0.25][1] - keep[0.1][1]      # buoc o dot replicate
        print("\nBUOC 0.1 -> 0.25 (buoc lon nhat toan luoi, tru cot cua ket luan"
              " \"khong tron\")")
        print("  dot goc      : %+.3f  [%+.3f, %+.3f]  p=%.3f"
              % (s1.mean(), np.percentile(s1, 2.5), np.percentile(s1, 97.5),
                 two_sided_p(s1)))
        print("  dot replicate: %+.3f  [%+.3f, %+.3f]  p=%.3f"
              % (s2.mean(), np.percentile(s2, 2.5), np.percentile(s2, 97.5),
                 two_sided_p(s2)))
        diff = s2 - s1
        print("  chenh lech   : %+.3f  p=%.3f" % (diff.mean(), two_sided_p(diff)))
        same_sign = (s1.mean() > 0) == (s2.mean() > 0)
        both_sig = two_sided_p(s1) < 0.05 and two_sided_p(s2) < 0.05
        if same_sign and both_sig:
            verdict = "LAP LAI - dong gop (i) dung vung"
        elif same_sign:
            verdict = "cung dau nhung khong ca hai deu co y nghia - yeu"
        else:
            verdict = "KHONG LAP LAI - dong gop (i) khong dung vung"
        print("  => %s" % verdict)

        print("\n  So sanh voi san nhieu: buoc goc %.3f vs |delta| trung vi %.3f"
              % (abs(s1.mean()), noise))
        if abs(s1.mean()) <= noise:
            print("  => buoc khong lon hon nhieu giua hai lan chay. Thang BIC dang"
                  " doc nhieu theo o chu khong phai phan ung theo lambda.")

    print("\nda ghi %s" % OUT.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
