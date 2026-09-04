"""E1: thang BIC dựng lại trên lưới λ mở rộng (10 mức thay vì 3).

Ghi ra bảng RIÊNG (`T_E1_ladder.csv`); `T_PS05` của bản thảo không bị đụng tới.

Điểm mới so với T_PS05
----------------------
1. Nhánh frontier có 10 mức λ thay vì 3, nên đa thức bậc 2/3 và mô hình bão hoà
   không còn trùng nhau - ở 3 điểm chúng là CÙNG một mô hình, và toàn bộ ΔBIC=8,43
   mà §3.2 báo cáo chỉ là hình phạt `ln(N)` cho một tham số thừa, không phải chênh
   lệch độ khớp.
2. Thêm đặc tả `notation regime PRINTED`, đọc regime từ ô in ra thay vì từ λ. Hai
   cách chỉ khác nhau ở λ=0.5 (in ra `0 / 1 / 3 / 5`, số nguyên ở độ lớn dưới đơn
   vị) - đúng ô phân định giữa hai giả thuyết.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis"))
from pdlib import ingest  # noqa: E402

BASE = [0, 2, 6, 10]          # frontier: S / P / R / T theo thu tu ingest
OUT = REPO / "Analysis" / "tables" / "T_E1_ladder.csv"


def printed(v):
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:g}"


def feats(lam):
    cells = [printed(c * lam) for c in BASE]
    frac = any("." in c for c in cells)
    digits = max(len(c.replace(".", "").replace("-", "").lstrip("0")) or 1 for c in cells)
    pdigits = max(len(c.replace("-", "").lstrip("0")) or 1 for c in cells)
    return dict(
        is_fractional=int(frac),
        mean_glyphs=float(np.mean([len(c) for c in cells])),
        regime="fractional" if lam < 1 else "unit" if lam <= 10 else "large",
        regime_printed=("fractional" if frac else "unit" if pdigits <= 3 else "large"),
        max_digits=digits,
    )


LADDER = {
    "controls only": "coop_rate ~ 1",
    "linear in log10 lambda": "coop_rate ~ loglam",
    "quadratic in log10 lambda": "coop_rate ~ loglam + I(loglam**2)",
    "cubic in log10 lambda": "coop_rate ~ loglam + I(loglam**2) + I(loglam**3)",
    "fractional flag": "coop_rate ~ is_fractional",
    "fractional + glyph count": "coop_rate ~ is_fractional + mean_glyphs",
    "notation regime (lambda)": "coop_rate ~ C(regime)",
    "notation regime (PRINTED)": "coop_rate ~ C(regime_printed)",
    "saturated in lambda": "coop_rate ~ C(lam_f)",
}
CTRL = " + C(language) + C(personality) + C(opp_personality)"


def main():
    rounds, games = ingest.build_master(families=("frontier",))
    rows = []
    for model, d in games.groupby("model"):
        lams = sorted(d.scale_nominal.unique())
        if len(lams) < 4:
            print(f"bo qua {model}: chi co {len(lams)} muc lambda")
            continue
        d = d.copy()
        f = pd.DataFrame([feats(x) for x in d.scale_nominal], index=d.index)
        d = pd.concat([d, f], axis=1)
        d["loglam"] = np.log10(d.scale_nominal)
        d["lam_f"] = d.scale_nominal.astype(str)

        for name, formula in LADDER.items():
            m = smf.ols(formula + CTRL, data=d).fit()
            rows.append(dict(model=model, n_lambdas=len(lams), spec=name,
                             k=int(m.df_model), bic=m.bic, r2=m.rsquared))

    out = pd.DataFrame(rows)
    out["delta_bic"] = out.bic - out.groupby("model").bic.transform("min")
    out = out.sort_values(["model", "delta_bic"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    for model, d in out.groupby("model"):
        print(f"\n=== {model}  ({d.n_lambdas.iloc[0]} muc lambda) ===")
        print(f"{'spec':<28}{'k':>4}{'BIC':>11}{'r2':>9}{'dBIC':>9}")
        print("-" * 61)
        for _, r in d.iterrows():
            star = "  <-- tot nhat" if r.delta_bic == 0 else ""
            print(f"{r.spec:<28}{r.k:>4}{r.bic:>11.1f}{r.r2:>9.4f}{r.delta_bic:>9.1f}{star}")
    print(f"\nda ghi {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
