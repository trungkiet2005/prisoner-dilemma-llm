#!/usr/bin/env bash
# Ngày 1 của paper_scaling/RUN_PLAN.md: E1 (λ ngoài decade) + E3 (mở rộng decade).
#
#   E1  λ = 0.25, 0.5, 2, 5      phá vỡ đồng nhất thức "λ<1 <=> ô có dấu thập phân"
#   E3  λ = 0.01, 100, 1000      cho nhánh frontier đủ bậc tự do để thang BIC phân biệt
#
# Cách dùng:
#   ./run_e1_day1.sh <account> <model-slug> [e1|e3|both]
#
#   ./run_e1_day1.sh trungkiet google/gemini-3.5-flash-lite both
#
# Mỗi account MỘT thư mục KAGGLE_CONFIG_DIR riêng, vì credential của các account sẽ
# đè lên nhau nếu dùng chung ~/.kaggle. Xem CLAUDE.md muc "MODEL PROXY".

set -euo pipefail

ACCOUNT="${1:?thiếu tên account, ví dụ: trungkiet}"
MODEL="${2:?thiếu model slug, ví dụ: google/gemini-3.5-flash-lite}"
WHICH="${3:-both}"

CRED_ROOT="${PD_CRED_ROOT:-D:/AI_PhD/GameTheory/kaggle_for_research}"
CFG_DIR="${PD_CFG_ROOT:-$HOME/.kaggle-accounts}/$ACCOUNT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

E1_LAMBDAS="0.25,0.5,2,5"
E3_LAMBDAS="0.01,100,1000"

case "$WHICH" in
  e1)   LAMBDAS="$E1_LAMBDAS" ;;
  e3)   LAMBDAS="$E3_LAMBDAS" ;;
  both) LAMBDAS="$E1_LAMBDAS,$E3_LAMBDAS" ;;
  *)    echo "tham số 3 phải là e1 | e3 | both" >&2; exit 2 ;;
esac

n_lam=$(awk -F, '{print NF}' <<<"$LAMBDAS")
echo "=============================================================="
echo " account   : $ACCOUNT   (config dir: $CFG_DIR)"
echo " model     : $MODEL"
echo " λ         : $LAMBDAS   ($n_lam cell)"
echo " ước tính  : $n_lam × 200 game = $((n_lam * 200)) game, $((n_lam * 4000)) lượt gọi"
echo "=============================================================="

# ---- 1. Cổng chặn: notation phải đúng trước khi tiêu tiền --------------------
# λ=0.5 in ra "0 / 1 / 3 / 5" là toàn bộ lý do tồn tại của E1. Nếu nó in ra
# "0.0 / 1.0 / 3.0 / 5.0" thì sweep vẫn chạy, vẫn ra số, nhưng vô nghĩa.
echo "[1/4] kiểm notation + parity ..."
( cd "$HERE" && PD_SKIP_RUN=1 PYTHONUTF8=1 python -m pytest test_pd_task_parity.py -q ) \
  || { echo "PARITY FAIL - dừng, không chạy sweep." >&2; exit 1; }

# ---- 2. Nạp credential cho đúng account -------------------------------------
echo "[2/4] nạp credential cho '$ACCOUNT' ..."
mkdir -p "$CFG_DIR"
export KAGGLE_CONFIG_DIR="$CFG_DIR"

if [[ -f "$CRED_ROOT/kaggle-api/$ACCOUNT.txt" ]]; then
  KAGGLE_API_TOKEN="$(grep -o 'KGAT_[A-Za-z0-9_-]*' "$CRED_ROOT/kaggle-api/$ACCOUNT.txt" | head -1)"
  export KAGGLE_API_TOKEN
  echo "      -> token KGAT_… từ kaggle-api/$ACCOUNT.txt"
elif [[ -f "$CRED_ROOT/$ACCOUNT.json" ]]; then
  cp "$CRED_ROOT/$ACCOUNT.json" "$CFG_DIR/kaggle.json"
  echo "      -> kaggle.json kiểu cũ"
else
  echo "không tìm thấy credential cho '$ACCOUNT' trong $CRED_ROOT" >&2
  exit 1
fi

# Proxy key hết hạn sau ~2 tiếng. Sweep dài hơn thế thì chạy lại lệnh này cùng
# account để làm mới, KHÔNG cần đổi sang account khác.
kaggle benchmarks auth -y --env-file "$HERE/.env" \
  || { echo "auth thất bại - account hết quota hoặc chưa verify?" >&2; exit 1; }

# ---- 3. Chạy sweep ----------------------------------------------------------
# PD_LAMBDAS được đặt -> _has_overrides = true -> pd_task tự mở full sweep, không
# cần thêm slug vào FULL_SWEEP_MODELS. Mặt trái: gõ sai slug cũng chạy full chứ
# không rơi về smoke, nên kiểm kỹ dòng "model" in ở đầu.
echo "[3/4] chạy sweep ..."
( cd "$HERE" && \
  PD_LAMBDAS="$LAMBDAS" \
  PD_FULL=1 \
  PYTHONUTF8=1 \
  python run_and_watch.py -m "$MODEL" )

# ---- 4. Cổng chặn cuối: fallback_rate ---------------------------------------
# fallback_rate cao nghĩa là model không parse được và bị gán OptionA - tỉ lệ hợp
# tác khi đó là rác. Đây là bẫy đã làm hỏng dữ liệu một lần rồi.
echo "[4/4] kiểm fallback_rate ..."
( cd "$HERE" && PYTHONUTF8=1 python - "$MODEL" <<'PY'
import glob, json, sys
slug = sys.argv[1]
tag = slug.split("/")[-1]
files = sorted(glob.glob(f"*{tag}*.run.json"))
if not files:
    print("      chưa thấy .run.json - kiểm thủ công bằng: kaggle b t status")
    raise SystemExit(0)
bad = False
for f in files[-3:]:
    d = json.load(open(f, encoding="utf-8"))
    for r in (d.get("results") or []):
        fr = r.get("fallback_rate")
        pf = r.get("parse_fail_rate")
        if fr is None:
            continue
        flag = "  <-- ĐIỀU TRA" if fr > 0.02 else ""
        bad = bad or fr > 0.02
        print(f"      {f}: fallback={fr:.4f} parse_fail={pf}{flag}")
raise SystemExit(1 if bad else 0)
PY
) || { echo "fallback_rate vượt 0.02 - KHÔNG dùng dữ liệu này cho phân tích." >&2; exit 1; }

echo
echo "Xong. Chép CSV về Dataset/ bằng:"
echo "  python run_and_watch.py --attach --collect-into Dataset/data_fairgame_frontier_llm"
