#!/usr/bin/env bash
# Ngày 2 (E5): 14 model × 5 λ, phân bổ qua 5 account.
#
# Mỗi account push TASK RIÊNG của mình rồi chạy phần model được giao, vì quota là
# theo account chứ không phải một pool chung (~$10/account/ngày). Một account chạy
# cả 14 model sẽ vượt hạn mức.
#
# Chạy:  bash run_day2_e5.sh submit     # push + submit, không chờ
#        bash run_day2_e5.sh collect    # tải kết quả về sau khi xong

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CR="D:/AI_PhD/GameTheory/kaggle_for_research"
TASK="prisoner-dilemma-fairgame"
DEST="$REPO/results/kbench"          # tuyệt đối: đường dẫn tương đối từng
                                              # làm dữ liệu rơi ra ngoài repo
# account:model1,model2,model3
ASSIGN=(
  "chunaiu:gemini-3.1-flash-lite-preview,gemini-3.5-flash,gemini-3.6-flash"
  "trunkdabest:gemini-3.7-flash,gemma-4-26b-a4b-it,gemma-4-31b-it"
  "vinhdinhthien:gpt-oss-20b,gpt-oss-120b,gpt-5.4-nano-2026-03-17"
  "acc1:claude-haiku-4-5-20251001,deepseek-v3.1,glm-5"
  "acc2:qwen3-next-80b-a3b-instruct,grok-4.20-0309-non-reasoning"
)

load_creds() {                                # $1 = tên account
  local acc="$1" cfg="$HOME/.kaggle-accounts/$1" tok=""
  mkdir -p "$cfg"
  export KAGGLE_CONFIG_DIR="$cfg"
  for f in "$CR/kaggle-api/$acc.txt" "$CR/kaggle-api-2/$acc.md"; do
    [[ -f "$f" ]] && tok="$(grep -o 'KGAT_[A-Za-z0-9_-]*' "$f" | head -1)" && break
  done
  [[ -n "$tok" ]] || { echo "  [!] không tìm thấy token cho $acc"; return 1; }
  export KAGGLE_API_TOKEN="$tok"
}

kg() { timeout 900 python -m kaggle benchmarks tasks "$@" 2>&1 \
       | grep -viE "cryptography|TripleDES|Blowfish|\"class\"|\"cipher\""; }

cd "$HERE"

case "${1:-submit}" in

submit)
  # Cổng chặn: notation phải đúng và λ phải là bộ của Ngày 2 trước khi tiêu tiền.
  PD_SKIP_RUN=1 PYTHONUTF8=1 python -m pytest test_pd_task_parity.py -q >/dev/null \
    || { echo "PARITY FAIL - dừng."; exit 1; }
  PD_SKIP_RUN=1 PYTHONUTF8=1 python -c "
import pd_task as T
want = [0.1, 0.5, 1, 10, 100]
assert sorted(map(float, T.LAMBDAS)) == sorted(want), f'LAMBDAS={T.LAMBDAS}, can {want}'
print(f'  λ={T.LAMBDAS}  max_tokens={T.MAX_OUTPUT_TOKENS}  allowlist={len(T.FULL_SWEEP_MODELS)} model')
" || exit 1

  for row in "${ASSIGN[@]}"; do
    acc="${row%%:*}"; models="${row#*:}"
    echo "── $acc ── ${models//,/ }"
    load_creds "$acc" || continue
    # Push tạo task riêng của account này. Push cũng chạy thử 1 lần trên model mặc
    # định của server; model đó không nằm trong allowlist nên rơi về smoke 40 lượt.
    kg push "$TASK" -f pd_task.py --wait | tail -2
    # `-m` chi nhan MOT model moi lan; nhieu model thi PHAI lap co: -m a -m b.
    # Viet "-m a b c" lam argparse bao loi va KHONG submit gi ca.
    margs=(); for m in ${models//,/ }; do margs+=(-m "$m"); done
    kg run "$TASK" "${margs[@]}" | grep -iE "scheduled|skipped|error|queued" | head -8
  done
  echo; echo "Đã submit. Chờ rồi chạy:  bash run_day2_e5.sh collect"
  ;;

collect)
  for row in "${ASSIGN[@]}"; do
    acc="${row%%:*}"
    echo "── $acc ──"
    load_creds "$acc" || continue
    kg status "$TASK" | sed -n '/^Model/,$p' | head -20
    kg download "$TASK" -o "$DEST" | tail -3
  done
  ;;

*) echo "dùng: $0 {submit|collect}" >&2; exit 2 ;;
esac
