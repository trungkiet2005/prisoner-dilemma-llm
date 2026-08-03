# `kaggle/` — chạy lại FAIRGAME Prisoner's Dilemma trên Kaggle

Hai nhánh chạy **cùng một thí nghiệm**, khác nhau chỗ agent gọi model ở đâu:

| Nhánh | File | Model | Hạ tầng |
|---|---|---|---|
| Open-source | [`experiments/baseline.py`](experiments/baseline.py) | 7 model mã nguồn mở | Kaggle Notebook, GPU ON, Internet OFF, vLLM local |
| API | [`benchmarks/pd_task.py`](benchmarks/pd_task.py) | Gemini (và mọi model trên proxy Kaggle Benchmarks) | Kaggle Benchmarks, `kaggle b t push/run` |

Cả hai ghi ra **đúng schema CSV** và **đúng layout thư mục** của
`Dataset/data_fairgame_small_llm/` nên kết quả ghép thẳng vào cùng một bảng
phân tích.

---

## ⚠️ Vì sao có thư mục này — sửa lỗi payoff

Lần chạy open-source trước dùng nhầm config **`..._mild.json` (weight1 = 8)**.
Nhánh frontier (`claude` / `gpt` / `mistral`) chạy bằng
**`..._conventional.json` (weight1 = 6)**. Hai nhánh vì thế không so sánh được —
R đổi từ 6 sang 8 làm đổi cả `greed` lẫn `k`-index của ma trận PD:

|  | T | R | P | S | greed = (T−R)/(T−S) | k = (R−P)/(T−S) |
|---|---|---|---|---|---|---|
| conventional (**đúng**) | 10 | **6** | 2 | 0 | 0.40 | 0.40 |
| mild (lần trước, sai) | 10 | **8** | 2 | 0 | 0.20 | 0.60 |

Cả hai file trong thư mục này **chốt cứng `conventional`** và mở rộng payoff
scaling ra 6 mức thay vì 3:

```
frontier     λ ∈ {0.1, 1, 10}
nhánh này    λ ∈ {0.01, 0.1, 1, 10, 100, 1000}
```

Test [`benchmarks/test_pd_task_parity.py`](benchmarks/test_pd_task_parity.py)
có một case (`test_conventional_payoff_is_the_frontier_one`) chỉ để chặn đúng
lỗi này tái diễn.

---

## Thiết kế thí nghiệm (giống hệt ở cả hai nhánh)

* Config: `FAIRGAME/resources/config/prisoner_dilemma_nocomm_round_known_conventional.json`
* Payoff base: `w1=6 w2=10 w3=0 w4=2` → T=10, R=6, P=2, S=0
* λ ∈ {0.01, 0.1, 1, 10, 100, 1000} — nhân TẤT CẢ weights
* 30 vòng, agent **biết** tổng số vòng, **không** giao tiếp, không dừng sớm
* 5 ngôn ngữ: `en, fr, ar, cn, vn`
* 4 tổ hợp tính cách (cooperative/selfish × 2 agent) × 10 rep → **40 game / (λ, lang)**
* Tổng: **1200 game/model** = 72.000 lượt gọi model

λ×weight ra float (`6 × 1 = 6.0`); cả hai nhánh ép về `int` khi giá trị nguyên
(`NORMALIZE_INTEGER_WEIGHTS`) nên prompt ghi `6` chứ không phải `6.0` — tại λ=1
prompt khớp **đúng từng ký tự** với prompt nhánh frontier.

---

## Nhánh 1 — 7 model open-source (`experiments/baseline.py`)

| # | `short_name` | Nguồn |
|---|---|---|
| 1 | `qwen25-7b-instruct` | Kaggle Models `qwen-lm/qwen2.5` |
| 2 | `gemma2-9b-it` | Kaggle Models `google/gemma-2` |
| 3 | `llama-3-1-8b` | Dataset `foundnotkiet/llama-3-1-8b` |
| 4 | `qwen25-32b-instruct` | Kaggle Models `qwen-lm/qwen2.5` |
| 5 | `gemma2-27b-it` | Kaggle Models `google/gemma-2` |
| 6 | `qwen25-72b-instruct-awq` | Dataset AWQ int4 (tải bằng `FAIRGAME/download_model.py`) |
| 7 | `llama-3-3-70b-instruct-awq` | Kaggle Models `jagatkiran/meta-llama-3.3-70b` |

Path giống hệt bản `Colective_Risk_Game/kaggle/experiments/baseline.py` để khỏi
phải add lại input.

**Chạy**

1. (nếu image chưa có vLLM) chạy [`setup/build_quant_wheels.py`](setup/build_quant_wheels.py)
   trên notebook **Internet ON** → Output → *New Dataset* (vd `pd-quant-wheels`).
2. Tạo notebook mới: **GPU ON, Internet OFF**. `+ Add Input`:
   repo này (thư mục chứa `FAIRGAME/`), dataset wheels, và từng model.
3. Copy `experiments/baseline.py` vào notebook, chia cell theo `# CELL N`.
4. Sửa `MODELS[]` cho khớp path thật (`!ls /kaggle/input/`), rồi Run Cell 1 → 8.

**Chia phiên.** 1 phiên Kaggle ~9–12h; 7 model một phiên là quá giờ. Chạy 1–2
model/phiên (comment bớt `MODELS[]`), hai model AWQ 70B/72B chạy riêng.
`RESUME = True` tự bỏ qua ô `(λ, model)` đã có đủ 5 file CSV, nên chạy nhiều
phiên rồi `+ Add Input` output cũ là gộp được.

**Nút trim khi hết giờ:** `LAMBDAS_OVERRIDE`, `LANGUAGES_OVERRIDE`,
`REPS_OVERRIDE` ở Cell 1 — không phải sửa file config.

**Output**

```
/kaggle/working/pd_results/<λ>/<model_short>/x<λ>_<lang>_<model_short>.csv
/kaggle/working/pd_results/<λ>/<model_short>/results_<config>.json   # full history, debug/XAI
/kaggle/working/pd_results/run_manifest.json
/kaggle/working/pd_results.zip
```

---

## Nhánh 2 — Gemini trên Kaggle Benchmarks (`benchmarks/pd_task.py`)

Kaggle Benchmarks chỉ push **một file**, nên `pd_task.py` nhúng sẵn bản sao của
prompt template + config payoff và viết lại logic dựng prompt / parse / chấm
điểm. `test_pd_task_parity.py` so **từng byte** bản sao đó với `FAIRGAME/` —
chạy test sau mỗi lần sửa một trong hai bên.

```bash
# 0. parity test (không tốn tiền, không gọi model)
PYTHONUTF8=1 python -m pytest kaggle/benchmarks/test_pd_task_parity.py -q

# 1. smoke test RẺ — xem parse_fail_rate / fallback_rate trước khi mở full sweep
cd kaggle/benchmarks
PD_LAMBDAS=1 PD_LANGS=en PD_REPS=1 PD_ROUNDS=5 PYTHONUTF8=1 python pd_task.py

# 2. push + chạy thật
kaggle b auth -y
kaggle b t push prisoner-dilemma-fairgame -f pd_task.py --wait
kaggle b t run  prisoner-dilemma-fairgame -m google/gemini-3.1-flash-lite-preview --wait
kaggle b t status   prisoner-dilemma-fairgame
kaggle b t download prisoner-dilemma-fairgame -o ./results
```

**Biến môi trường**

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PD_MODEL` | `google/gemini-3.1-flash-lite-preview` | ghi đè model khi chạy local |
| `PD_MODEL_TAG` | tên model đã sanitize | tên thư mục + phần `<model>` trong tên file CSV |
| `PD_LAMBDAS` | `0.01,0.1,1,10,100,1000` | các mức payoff scaling |
| `PD_LANGS` | `en,fr,ar,cn,vn` | ngôn ngữ |
| `PD_REPS` | `10` | rep / (λ, lang, tổ hợp) |
| `PD_ROUNDS` | `30` | số vòng mỗi game |
| `PD_CONCURRENCY` | `8` | số game chạy song song (hạ về 1 nếu proxy 429) |
| `PD_RESUME` | `1` | `0` = chạy lại sạch, bỏ checkpoint |
| `PD_OUT` | `results/kbench/<tag>` | thư mục output |
| `PD_FULL` / `PD_FULL_MODELS` | — | mở full sweep cho model ngoài allowlist |

**Chốt chặn chi phí (allowlist, không phải blocklist).** `kaggle b t push` chạy
task một lần trên **model mặc định của server** — và ta không kiểm soát model đó
là gì. Nên `pd_task.py` mặc định chỉ chạy **smoke 40 lượt gọi**, trừ khi model
nằm trong hằng `FULL_SWEEP_MODELS` (hiện có
`google/gemini-3.1-flash-lite-preview`) hoặc có `PD_FULL=1` / `PD_FULL_MODELS` /
override `PD_LAMBDAS…`. Thêm model mới cho lần chạy thật thì thêm slug vào hằng
đó rồi mới `kaggle b t run -m <slug>`.

**Chi phí.** Full sweep = 72.000 lượt gọi model (prompt ~400–700 token, output
~5–20 token). Mỗi game xong ghi một shard checkpoint nên chạy lại là resume —
không trả tiền lần hai cho phần đã xong. `PD_LAMBDAS`/`PD_LANGS` cho phép chia
nhỏ thành nhiều lần chạy; checkpoint dùng chung được vì signature **không** chứa
λ/ngôn ngữ (chỉ chứa những thứ ảnh hưởng seed).

Đo thực tế (smoke `λ=1, en, 1 rep, 3 vòng`, gemini-3.1-flash-lite-preview):
24 lượt gọi = **$0.0016**, 11 s, `parse_fail_rate = 0`. Prompt dài dần theo
history nên full 30 vòng đắt hơn mỗi lượt ~3×.

**Seed / CRN.** `seed = ((BASE + cell)*1e5 + round*100 + agent) mod 2³¹−1` với
`cell = (lang_idx*4 + perm_idx)*REPS + rep`. `cell` **cố ý không chứa λ**: mọi mức
payoff scaling dùng chung một dãy số ngẫu nhiên, nên chênh lệch giữa các λ là do
payoff chứ không phải nhiễu sampling.

**Health check.** Task assert `fell_back == 0` — tức mọi quyết định đều parse ra
được `OptionA`/`OptionB`, không cái nào phải rơi về fallback. Tỉ lệ hợp tác *tự
nó là kết quả*, không phải điều kiện pass/fail. Luôn xem `parse_fail_rate` và
`fallback_rate` **trước** khi diễn giải số liệu.

---

## Sau khi chạy xong — nối vào `Analysis/`

1. Giải nén `pd_results.zip` (hoặc `results/kbench/<tag>/`) rồi copy các thư mục
   `<λ>/<model>/` vào `Dataset/data_fairgame_small_llm/`.
2. Sửa `Analysis/pdlib/ingest.py`:
   * thêm mọi `short_name` mới vào `MODEL_MAP` (hàm `_iter_files` sẽ `KeyError`
     nếu thiếu);
   * đổi `_BASE_MATRIX["small"]` từ `R: 8.0` → **`R: 6.0`** khi dữ liệu mới thay
     dữ liệu cũ — đây chính là con số phản ánh lỗi payoff đã nêu ở trên.
3. `python Analysis/run_all.py`.

---

## Cấu trúc thư mục

```
kaggle/
├── README.md                          # file này
├── experiments/
│   └── baseline.py                    # nhánh open-source, 7 model, 6 λ
├── benchmarks/
│   ├── pd_task.py                     # nhánh API (Gemini) cho Kaggle Benchmarks
│   ├── test_pd_task_parity.py         # so từng byte pd_task.py ↔ FAIRGAME
│   └── reference.md                   # cú pháp kaggle-benchmarks
└── setup/
    └── build_quant_wheels.py          # build wheels vLLM/bitsandbytes cho notebook offline
```
