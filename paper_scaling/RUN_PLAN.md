# Kế hoạch chạy bổ sung cho `paper_scaling`

Xếp theo tỉ lệ **tác động / chi phí**. Ngân sách khả dụng: **17 account Kaggle
(16 dùng được) ~ $170/ngày**, nên toàn bộ Ưu tiên 1-3 nằm gọn trong khoảng một tuần.

Neo chi phí đã đo thật: `gemini-3.5-flash-lite`, sweep 3 λ đầy đủ = 600 game
= 12.000 lượt gọi = **$1,64 / 17 phút**.

---

## 0. Ba chỗ dữ liệu hiện tại chưa đỡ nổi khẳng định của bài

Cả ba đọc thẳng ra từ `Analysis/tables/T_PS*.csv`, không phải suy đoán.

### 0.1 Thang BIC của nhánh frontier bị suy biến

Nhánh frontier chỉ có **3 mức λ**. Với 3 điểm, đa thức bậc hai đã bão hoà, nên bốn
đặc tả sau **là cùng một mô hình**:

| spec | k | BIC | r² |
|---|---|---|---|
| quadratic in log10 λ | 11 | 2911.101 | 0.172070 |
| cubic in log10 λ | 11 | 2911.101 | 0.172070 |
| fractional + glyph count | 11 | 2911.101 | 0.172070 |
| saturated in λ | 11 | 2911.101 | 0.172070 |
| **fractional flag** | 10 | **2902.668** | 0.172062 |
| **notation regime (3 levels)** | 10 | **2902.668** | 0.172062 |

Hai điều rút ra:

1. `fractional flag` và `notation regime` cũng là **một** mô hình ở nhánh này.
2. ΔBIC = 8,43 mà §3.2 báo cáo **không phải chênh lệch độ khớp**. r² của notation
   thực ra thấp hơn một chút (0,172062 so với 0,172070); toàn bộ khoảng cách là
   hình phạt `ln(N)` cho một tham số thừa.

Hệ quả: câu "năm trong bảy model ưa mô tả theo notation" thực chất là **hai**
(Llama, Qwen3). Ba model frontier "ưa notation" chỉ vì thiết kế không đủ bậc tự do
để phân biệt. Gemma - model duy nhất trong nhánh 6 mức thực sự chọn được - lại **ưa
đa thức bậc ba**. Nền chứng cứ thật cho khẳng định trung tâm là 2/3, không phải 5/7.

### 0.2 Không có mức λ nào nằm ngoài decade

Toàn bộ corpus là lưới log10 đúng decade: `0.01, 0.1, 1, 10, 100, 1000`. Trong lưới
đó, "λ < 1" và "ô in ra có dấu thập phân" là **cùng một biến cố** - bài thừa nhận ở
§2.3 và §4.2.

Nhưng nó rẻ hơn bài tưởng rất nhiều. Ma trận gốc open-weight `0 / 2 / 8 / 10`:

| λ | Ô in ra trong prompt | Thập phân? | Trạng thái |
|---|---|---|---|
| 0.01 | `0 / 0.02 / 0.08 / 0.1` | có | đã có |
| 0.1 | `0 / 0.2 / 0.8 / 1` | có | đã có |
| **0.25** | `0 / 0.5 / 2 / 2.5` | có | **thiếu** |
| **0.5** | `0 / 1 / 4 / 5` | **KHÔNG** | **thiếu - ô quyết định** |
| 1 | `0 / 2 / 8 / 10` | không | đã có |
| **2** | `0 / 4 / 16 / 20` | không | **thiếu** |
| **5** | `0 / 10 / 40 / 50` | không | **thiếu** |
| 10 | `0 / 20 / 80 / 100` | không | đã có |
| 100 | `0 / 200 / 800 / 1000` | không | đã có |
| 1000 | `0 / 2000 / 8000 / 10000` | không | đã có |

**λ = 0.5 in ra toàn số nguyên dù độ lớn dưới đơn vị.** Code hiện tại vẫn gán nhãn
`fractional` cho nó (vì `lam < 1`), nhưng prompt thì không có dấu thập phân nào. Đó
chính xác là ô mà hai giả thuyết cho dự đoán trái ngược nhau - và nó **không cần sửa
một dòng code nào**.

### 0.3 n = 7, và không model nào biết suy luận

"Sáu trong bảy model dịch chuyển" là giai thoại, chưa phải phân phối. Bài tự gọi
reasoning model là "phần mở rộng giá trị nhất" và "thí nghiệm duy nhất mà chúng tôi
không muốn dự đoán kết quả".

---

## 1. Ưu tiên 1 - phá vỡ confound trung tâm

> Không sửa code (trừ E2). Làm trước tất cả. Tổng ~$65-180.

### E1. Thêm λ ngoài decade: 0.25, 0.5, 2, 5

Thí nghiệm rẻ nhất và quyết định nhất. Chỉ cần một config FAIRGAME mới với
`payoffMatrix.weights` đã nhân sẵn, và thư mục đặt tên theo λ - `ingest.py` đọc scale
bằng `float(scale_dir.name)` nên `0.5` parse bình thường.

Bốn mức này cho hai thang tách rời nhau lần đầu tiên:

- **độ lớn ở notation cố định**: 0.5 → 1 → 2 → 5 → 10, tất cả in số nguyên
- **notation ở độ lớn gần nhau**: 0.25 (thập phân) so với 0.5 (số nguyên)

```bash
# frontier: qua benchmarks API
PD_LAMBDAS=0.25,0.5,2,5 PD_FULL=1 \
  python run_and_watch.py -m google/gemini-3.5-flash-lite

# open-weight: qua kaggle/experiments/baseline.py (vLLM, tốn GPU chứ không tốn credit)
LAMBDAS_OVERRIDE = [0.25, 0.5, 2, 5]
```

**Dự đoán phân định.** Nếu *notation* chi phối: coop(0.5) ≈ coop(1), và có bậc nhảy
sắc nét giữa 0.25 và 0.5. Nếu *magnitude* chi phối: coop(0.5) nằm giữa coop(0.1) và
coop(1), còn 0.25 với 0.5 gần như trùng nhau. Hai kết quả không thể nhầm lẫn.

| Sửa code | Khối lượng | Chi phí | Tác động |
|---|---|---|---|
| Không | 4 λ × 7 model | ~$25-60 | Rất cao |

### E3. Mở nhánh frontier lên đủ 6 decade

Cách duy nhất cứu khẳng định "notation thắng magnitude" ở nhánh frontier. Thêm
λ = `0.01, 100, 1000` cho cả bốn model là đủ để thang BIC có bậc tự do phân biệt đa
thức với bậc thang notation - hiện tại nó không có (xem §0.1).

Phần thưởng kèm theo: nhánh frontier sẽ có regime `large`, nên kiểm chứng được phần
*đi xuống* của đường cong - hiện chỉ quan sát ở open-weight, và đó là bằng chứng gián
tiếp mạnh nhất chống lại lối giải thích theo "mức cược".

```bash
PD_LAMBDAS=0.01,100,1000 PD_FULL=1 \
  python run_and_watch.py -m google/gemini-3.5-flash-lite <3 slug frontier còn lại>
```

| Sửa code | Khối lượng | Chi phí | Tác động |
|---|---|---|---|
| Không | 3 λ × 4 model | ~$20-70 | Rất cao |

### E2. Notation cố định / magnitude cố định - thiết kế 2×2 đầy đủ

Bài nêu ở §4.2 nhưng cho là tốn kém. Thực ra engine không cần sửa: template dùng
`str.format()` với format spec rỗng, nên fork template và đổi `{weight1}` thành
`{weight1:.1f}` là in được `0.0 / 2.0 / 8.0 / 10.0` trong khi giá trị chấm điểm giữ
nguyên số nguyên.

Cách sạch hơn: thêm trường `weightFormat` vào config và áp ở `prompt_creator.py:127`
- khoảng ba dòng, tách hẳn *notation hiển thị* khỏi *magnitude chấm điểm*.

Bốn ô: {độ lớn 1×, 0.1×} × {in số nguyên, in một chữ số thập phân}.

> ⚠️ **Va chạm đường dẫn.** Trong `pd_task.py`, hai điều kiện cùng magnitude sẽ có
> cùng `_condition_key`, cùng tên file checkpoint và cùng đường dẫn CSV - điều kiện
> thứ hai bị resume bỏ qua hoặc ghi đè lên điều kiện thứ nhất. Phải nhét nhãn notation
> vào `PD_MODEL_TAG` hoặc `fmt_lambda` **trước khi chạy**.

| Sửa code | Khối lượng | Chi phí | Tác động |
|---|---|---|---|
| ~3 dòng + tag | 4 ô × N model | ~$20-50 | Rất cao |

---

## 2. Ưu tiên 2 - biến giai thoại thành phân phối

> Nơi tác động trên mỗi đô-la cao nhất. Tổng ~$180-480.

### E5. Nâng từ 7 lên ~20 model

Tác động trên mỗi đô-la cao nhất trong cả danh sách. Sweep 3 λ tốn $1,64/model, nên
thêm 13 model rẻ chỉ tầm vài chục đô.

Cái nó mua được không phải "n lớn hơn" mà là một **trục mới**: hồi quy biên độ swing
lên số tham số, ngày phát hành, điểm benchmark. "Sáu trong bảy" trở thành **"hiệu ứng
không co lại theo năng lực"** - một câu mạnh hơn hẳn.

CLAUDE.md đã xác nhận 23 slug lên lịch được. Ưu tiên phủ kín các họ model và các bậc
kích thước, đừng lấy nhiều biến thể cùng một họ.

> ⚠️ **Bẫy im lặng.** Slug ngoài `FULL_SWEEP_MODELS` (`pd_task.py:213`) sẽ **âm thầm
> tụt xuống smoke test 40 lượt gọi** (1 λ, 1 ngôn ngữ, 5 vòng) mà không báo lỗi. Thêm
> slug vào danh sách đó hoặc đặt `PD_FULL=1`, rồi **push lại file** vì server chạy bản
> snapshot đã push.

| Sửa code | Khối lượng | Chi phí | Tác động |
|---|---|---|---|
| 1 dòng allowlist | +10-13 model | ~$30-80 | Rất cao |

### E4. Reasoning models

Cả hai kết quả đều đăng được, và đó là điều làm nó đáng chạy. Nếu model biết suy luận
**khôi phục** được bất biến → kết quả dương sạch: suy luận tại thời điểm suy diễn trả
lại định lý. Nếu **không** → khẳng định của bài mạnh lên một bậc: hiệu ứng sống sót
qua năng lực.

Để có công suất mà không đốt ngân sách, cắt các chiều không phải λ:
3-4 λ × 2 ngôn ngữ × 4 cặp persona × 5 lượt lặp = 120-160 game/model (thay vì 600).

```bash
PD_REASONING_EFFORT=medium \
PD_MAX_OUTPUT_TOKENS=2048 \
PD_LANGS=en,vn PD_REPS=5 PD_FULL=1 \
  python run_and_watch.py -m openai/gpt-5.6-sol \
    anthropic/claude-opus-5-default \
    google/gemini-3.1-pro-preview \
    deepseek/deepseek-r1-0528
```

> ⚠️ **BẪY CHẾT NGƯỜI - đọc trước khi chạy.** Mặc định repo là
> `PD_REASONING_EFFORT="none"` và `PD_MAX_OUTPUT_TOKENS=16`. Chạy reasoning model với
> hai giá trị đó thì phần suy luận ăn hết token, nội dung bị cắt thành `'Option'`,
> parse hỏng, retry, rồi **fallback OptionA 100%** - ra "hợp tác 100%" hoàn toàn giả.
> **Luôn kiểm `fallback_rate` trước khi tin bất kỳ con số nào.** Assertion ngưỡng 0.02
> ở cuối `pd_task.py` là chốt chặn.

| Sửa code | Khối lượng | Chi phí | Tác động |
|---|---|---|---|
| Chỉ env var | 4-6 model, sweep rút gọn | ~$150-400 | Cao nhất |

---

## 3. Ưu tiên 3 - tổng quát hoá

> Trả lời "có phải chỉ đúng với PD hai người không?". Tổng ~$35-90 cho E6+E7.

### E6. Nửa cộng của bất biến: u → a + u

Bài mới kiểm nửa nhân của phép biến đổi affine. Referee sẽ hỏi nửa còn lại, và nó vừa
rẻ vừa là **đối chứng notation gần như hoàn hảo**: cộng a = 1000 cho mọi ô ra
`1000 / 1002 / 1008 / 1010` - dịch điểm tham chiếu mà *không* đổi số chữ số giữa các
ô, cũng không tạo dấu thập phân.

Nếu hợp tác dịch chuyển dưới phép cộng trong khi notation gần như đứng yên → lối giải
thích thuần notation yếu đi. Nếu nó đứng yên → đối chứng âm rất mạnh cho E1 và E2.

| Sửa code | Khối lượng | Chi phí | Tác động |
|---|---|---|---|
| Không | 2-3 mức a | ~$15-40 | Cao |

### E7. Game thứ hai - Stag Hunt, gần như miễn phí

Repo **đã có sẵn** config và template đủ năm ngôn ngữ cho `stag_hunt`, `snow_drift`,
`harmony_game`, `battle_sexes`. Đổi một dòng tên config là chạy được.

Stag Hunt là lựa chọn tốt nhất vì nó cho **biến phụ thuộc sắc hơn tỉ lệ hợp tác**:
phép nhân λ không được phép dịch chuyển cân bằng trội theo payoff so với cân bằng trội
theo rủi ro. Nếu λ làm dịch chuyển *việc chọn cân bằng*, đó là khẳng định mạnh hơn hẳn
"tỉ lệ hợp tác thay đổi", và nâng bài từ một kết quả về PD lên kết quả về trò chơi ma
trận nói chung.

> Lưu ý: template PD tiếng Trung và tiếng Việt tồn tại dưới dạng `.rtf` chứ không phải
> `.txt`, mà `io_manager.py:53` hard-code đuôi `.txt`. Bốn game kia có đủ `.txt` cả năm
> ngôn ngữ nên không dính vấn đề này.

| Sửa code | Khối lượng | Chi phí | Tác động |
|---|---|---|---|
| Không | 1 game × 3-6 λ | ~$20-50 | Cao |

### E8. 3-4 người chơi

**Engine không hard-code hai người.** `payoff_matrix.py`, `game_round.py` và
`fairgame_factory.py` đều lặp trên số agent tuỳ ý, và repo đã có sẵn một cấu hình
**ba người chơi chạy được** (`volunteer_dilemma`) kèm unit test pass.

Việc thật nằm ở chỗ khác: viết template N người (25 template hiện tại đều liệt kê bảng
2×2 bằng văn xuôi và chỉ tham chiếu `{opponent1}`), và liệt kê đủ `S^N` tổ hợp trong
config - 8 ô với N=3, 16 ô với N=4. Mẫu để chép là `volunteer_dilemma_en.txt`, mô tả
payoff *đối xứng* ("những ai chọn X thì...") thay vì liệt kê từng ô.

**Đánh giá thẳng:** đáng làm, nhưng xếp sau E1-E5. Với riêng khẳng định về bất biến,
nó thêm ít hơn nhiều so với chi phí, và chồng lấn một phần với E7. Nếu làm, hãy làm
dưới dạng **public goods game ba người** - vì hệ số nhân r cho **một null thứ hai cũng
được định lý bảo chứng**, và vì các hệ multi-agent thực tế đều là N-agent nên chuẩn
báo cáo của bài trở nên liên quan trực tiếp hơn.

> ⚠️ **Bug tiềm ẩn chặn đường.** `prompt_creator.py:86-89` sinh khoá có đánh số
> (`opponentPersonality1`), nhưng cả 23 template sản xuất đều dùng dạng *không* đánh
> số. Hiện bị che vì mọi config đặt `opponentPersonalityProb: [0]`, làm khối đó bị xoá
> khỏi prompt. Đặt giá trị khác 0 là `.format()` ném `KeyError` ngay.

| Sửa code | Khối lượng | Chi phí | Tác động |
|---|---|---|---|
| Template + config mới | Trung bình-lớn | ~1-2 tuần người | Trung bình |

---

## 4. Ưu tiên 4 - đóng các phản biện rẻ tiền

> Mỗi cái vài giờ, mỗi cái bịt một lỗ hổng cụ thể. Tổng ~$20-30.

### E9. Gỡ confound giữa hai nhánh

Hai nhánh khác nhau ở horizon (30 với 10), ở việc có công bố số vòng hay không, ở nhiệt
độ và ở ma trận gốc - nên bài phải nói "không so sánh gì giữa hai nhánh". Chạy **một
model duy nhất** ở cả hai cấu hình là xoá được câu cảnh báo đó.

```bash
PD_ROUNDS=30 PD_ROUNDS_KNOWN=1 PD_FULL=1 \
  python run_and_watch.py -m google/gemini-3.5-flash-lite
```

### E10. Nhiệt độ và seed

"Đây có phải chỉ là nhiễu lấy mẫu không?" là câu hỏi rẻ nhất referee có thể đặt, và
cũng rẻ nhất để bịt. Một model, một ngôn ngữ, toàn bộ sweep λ, ở `PD_TEMPERATURE` = 0
và = 1. Ở nhiệt độ 0 hiệu ứng λ phải vẫn còn; nếu nó biến mất thì bạn cần biết trước
reviewer.

### E11. Hình thức chữ số theo ngôn ngữ

FAIRGAME **không có bất kỳ cơ chế locale nào**. Mọi ngôn ngữ đều in chữ số Ả Rập
phương Tây với dấu chấm thập phân - template tiếng Ả Rập in `6` chứ không phải `٦`,
tiếng Việt in `0.8` chứ không phải `0,8`.

Đó vừa là đối chứng tốt (nhất quán giữa các ngôn ngữ) vừa là thí nghiệm đang chờ sẵn:
in `٦` trong template Ả Rập, dấu phẩy thập phân trong fr/vn, hoặc viết bằng chữ ("tám
điểm"). Nó **nối kết quả notation với kết quả ngôn ngữ** - hai kết quả hiện đang rời
nhau trong bài - và rẻ.

---

## 5. Lịch chạy theo ngày, ngân sách $170/ngày

### 5.1 Đơn vị tính và ràng buộc

**1 cell = 1 cặp (model, λ) = 200 game = 4.000 lượt gọi.** Đó là đơn vị nhỏ nhất còn
giữ được thiết kế cân bằng: 5 ngôn ngữ × 4 cặp persona × 10 lượt lặp.

Neo giá đo thật: `gemini-3.5-flash-lite`, 600 game = 12.000 lượt gọi = **$1,64**.
Suy ra **1 cell hạng flash-lite = $0,55**.

Bảng hệ số dưới đây là **ước tính từ bảng giá công khai, chưa hiệu chuẩn**:

| Hạng model | ~$/cell | Ghi chú |
|---|---|---|
| flash-lite, gpt-oss-20b | 0,55 | đã đo thật |
| flash, gemma, qwen3-next | 0,8-1,5 | ngoại suy |
| claude-3.5-haiku | ~2,5 | ngoại suy |
| mistral-large | ~4,4 | ngoại suy |
| gpt-4o | ~5,5 | ngoại suy |
| reasoning (rút gọn 160 game) | 8-25 | **rất không chắc**, xem Ngày 3 |

> ⚠️ **Ngân sách là theo account, không phải một pool chung.** 16 account sống × ~$10
> mỗi account mỗi ngày = **~$160/ngày dùng được**. Một model đắt không thể dồn vào một
> account - `gpt-4o` chạy 7 cell là đã $38, phải trải qua 4 account. Mỗi account một
> `KAGGLE_CONFIG_DIR` riêng.

Ràng buộc khác: proxy key hết hạn sau ~2 tiếng (xin lại **cùng account**), và trần
phiên batch là 20 session đồng thời.

### 5.2 Ngày 0 - chuẩn bị, $0

Không chạy sweep nào. Làm hết những thứ sau rồi mới sang Ngày 1:

1. Dựng 16 thư mục `KAGGLE_CONFIG_DIR`, mỗi account một cái. Bỏ `trnnguynchis`.
2. Probe liveness: gọi `gpt-5.4-nano` prompt 1 chữ cho từng account, xác nhận có
   `"choices"`.
3. Viết config cho E1 (λ = 0.25, 0.5, 2, 5) và E3 (λ = 0.01, 100, 1000).
4. Thêm slug vào `FULL_SWEEP_MODELS`, chạy
   `PD_SKIP_RUN=1 pytest kaggle/benchmarks/test_pd_task_parity.py -q`, rồi **push lại**
   `pd_task.py`.
5. **Hiệu chuẩn giá.** Chạy đúng 1 cell rẻ nhất (flash-lite, λ=1, en), đọc
   `usage.cost` (nanodollars) trong response, nhân ra bảng giá thật cho từng model.
   Đừng tin bảng ngoại suy ở §5.1 trước khi làm bước này.

### 5.3 Ngày 1 - E1 + E3, nhánh frontier

**Đây là ngày quan trọng nhất.** Hết ngày 1 bạn đã trả lời được câu hỏi trung tâm của
bài: λ = 0.5 rơi về phía `λ = 1` hay phía `λ = 0.1`.

Mỗi model 7 cell: 4 cell của E1 (0.25, 0.5, 2, 5) + 3 cell của E3 (0.01, 100, 1000).

| Model | ~$/cell | 7 cell | Account cần |
|---|---|---|---|
| gemini-3.5-flash-lite | 0,55 | $3,83 | 1 |
| claude-3.5-haiku | 2,46 | $17,23 | 2 |
| mistral-large | 4,38 | $30,63 | 4 |
| gpt-4o | 5,47 | $38,29 | 4 |
| **Tổng** | | **~$90** | **11 / 16** |

Song song và **không tốn credit**: E1 nhánh open-weight (4 λ × 3 model = 12 cell) chạy
trên vLLM qua `kaggle/experiments/baseline.py`, tốn GPU chứ không đụng model proxy.

Cuối ngày: kiểm `fallback_rate` cả 28 cell. Cell nào > 0,02 thì điều tra trước khi
chạy tiếp - đừng để một model hỏng lặng lẽ trôi sang ngày sau.

### 5.4 Ngày 2 - E5, mở rộng bộ model

13 model mới × 5 λ = **65 cell**, phân bổ **1 model = 1 account = 5 cell ≈ $6**.

λ chọn: `0.1, 0.5, 1, 10, 100`. Lý do: `0.1 → 0.5 → 1` bắt ranh giới fractional/unit
cộng ô số nguyên dưới đơn vị; `1, 10` bắt vùng phẳng; `100` bắt khúc đi xuống. Đủ để
tính swing và đủ để vẽ hình dạng đường cong.

Slug đề xuất, ưu tiên phủ nhiều họ thay vì nhiều biến thể cùng họ:

```
gemini-3.1-flash-lite-preview   gemma-4-26b-a4b-it    qwen3-235b-a22b-instruct-2507
gemini-3.5-flash                gemma-4-31b-it        qwen3-next-80b-a3b-instruct
gemini-3.6-flash                gpt-oss-20b           glm-5
claude-haiku-4-5                gpt-oss-120b          deepseek-v3.1
grok-4.20-0309-non-reasoning
```

Ước tính **~$78, 13 account**.

### 5.5 Ngày 3 - E4 hiệu chuẩn rồi đợt 1

**Sáng: hiệu chuẩn, không bỏ qua.** Chạy 1 cell rút gọn (λ=1, en, 5 rep = 20 game) cho
*mỗi* model reasoning, đọc `usage.cost`. Ngân sách ~$5-15. Reasoning token làm chi phí
lệch 10-30 lần so với model thường và bạn **không biết trước hệ số** - đó là lý do
bước này đứng trước, không phải sau.

**Chiều: đợt 1, 2 model.** Thiết kế rút gọn:
4 λ (`0.25, 0.5, 1, 10`) × 2 ngôn ngữ (en, vn) × 4 cặp persona × 5 lượt lặp
= 160 game/model. Trải mỗi model qua 3-4 account.

```bash
PD_REASONING_EFFORT=medium PD_MAX_OUTPUT_TOKENS=2048 \
PD_LAMBDAS=0.25,0.5,1,10 PD_LANGS=en,vn PD_REPS=5 PD_FULL=1 \
  python run_and_watch.py -m openai/gpt-5.6-sol
```

⚠️ Kiểm `fallback_rate` **ngay sau cell đầu tiên**, đừng đợi hết đợt. Đây là chỗ bẫy
reasoning đã làm hỏng dữ liệu một lần rồi.

Ước tính **~$60-120**.

### 5.6 Ngày 4 - E4 đợt 2, và sửa code cho E2

Chạy 2-4 model reasoning còn lại, cùng thiết kế rút gọn. Ước tính **~$60-150**.

Tối, không tốn credit: sửa code cho E2 - thêm `weightFormat` (~3 dòng ở
`prompt_creator.py:127`) và nhãn chống va chạm trong `PD_MODEL_TAG`. Chạy parity test,
push lại.

### 5.7 Ngày 5 - E2, thao tác notation trực tiếp

4 điều kiện × 4 model frontier = **16 cell, ~$50, 8 account**.

| Điều kiện | Ô in ra | Vai trò |
|---|---|---|
| 1×, số nguyên | `0 / 2 / 8 / 10` | đối chứng, đã có |
| 1×, một chữ số thập phân | `0.0 / 2.0 / 8.0 / 10.0` | cùng độ lớn, đổi notation |
| 0.1×, thập phân | `0 / 0.2 / 0.8 / 1` | đối chứng, đã có |
| 0.1×, số nguyên qua đổi đơn vị | `0 / 2 / 8 / 10` "xu" | cùng notation, đổi độ lớn |

E1 đã gợi ý câu trả lời từ Ngày 1; E2 khoá nó lại bằng thao tác trực tiếp thay vì suy
ra từ một mức λ.

### 5.8 Ngày 6 - E6 + E7, tổng quát hoá

- **E6** additive: a ∈ {10, 100, 1000} × 4 model = 12 cell, ~$40
- **E7** stag hunt: 4 λ (`0.1, 0.5, 1, 10`) × 4 model = 16 cell, ~$50

Tổng **~$90, 12 account**.

### 5.9 Ngày 7 - E9 + E10 + E11, robustness

Chỉ dùng model rẻ nhất, nên cả ngày rất nhẹ.

- **E9** gỡ confound hai nhánh: 1 model × 6 λ ở cấu hình nhánh kia (30 vòng,
  `PD_ROUNDS_KNOWN=1`) = 6 cell
- **E10** nhiệt độ: 1 model × 4 λ × T ∈ {0, 1} = 8 cell
- **E11** hình thức chữ số: 1 model × 3 λ × 3 biến thể = 9 cell

Tổng **~$25, 5 account**.

### 5.10 Ngày 8 - không chạy gì, dựng lại phân tích

```bash
python Analysis/scripts/40_scaling_stats.py
python Analysis/scripts/41_fig_scaling.py
python Analysis/scripts/42_supp_tables.py
```

Kiểm bắt buộc: thang BIC ở nhánh frontier phải **hết suy biến** - số đặc tả có BIC
phân biệt được phải lớn hơn 4 (xem §0.1). Nếu vẫn bằng 4 thì E3 chưa vào dữ liệu.

Rồi viết lại §3.2 và §4.2 theo kết quả E1/E2.

### 5.11 Tổng kết ngân sách

| Ngày | Chạy | Ước tính | Account | % ngân sách ngày |
|---|---|---|---|---|
| 0 | chuẩn bị + hiệu chuẩn giá | ~$1 | 1 | <1% |
| 1 | **E1 + E3** frontier | ~$90 | 11 | 56% |
| 2 | **E5** +13 model | ~$78 | 13 | 49% |
| 3 | **E4** hiệu chuẩn + đợt 1 | ~$60-120 | 8-12 | 38-75% |
| 4 | **E4** đợt 2 | ~$60-150 | 8-14 | 38-94% |
| 5 | **E2** notation 2×2 | ~$50 | 8 | 31% |
| 6 | **E6 + E7** | ~$90 | 12 | 56% |
| 7 | **E9 + E10 + E11** | ~$25 | 5 | 16% |
| 8 | dựng lại phân tích | $0 | 0 | 0% |

**Tổng ~$455-605 trên 7 ngày chạy**, trung bình $65-86/ngày - tức khoảng **một nửa
ngân sách**. Phần đệm còn lại dùng để chạy lại cell hỏng và để E4 vượt dự toán mà
không phải cắt thiết kế.

### 5.12 Nếu chỉ có 3 ngày

Chạy Ngày 1, Ngày 2, rồi nhảy thẳng sang Ngày 8. Tổng ~$170, đúng một ngày ngân sách.

Ba ngày đó cho bạn E1 + E3 + E5, tức là: phá vỡ confound trung tâm, cứu thang BIC ở
nhánh frontier, và biến kết quả tiêu đề thành phát biểu về phân phối. Đó đã là một bản
thảo mạnh hơn hẳn bản hiện tại. E4 và E2 thêm vào sau khi có kết quả của ba ngày này -
và kết quả E1 sẽ nói cho bạn biết E2 còn cần thiết đến đâu.

## 6. Bốn cái bẫy đã ghi nhận

| Bẫy | Nội dung |
|---|---|
| **fallback** | Reasoning model làm hỏng dữ liệu trong im lặng. `max_tokens` nhỏ khiến nội dung bị cắt thành `'Option'`, parse hỏng, rồi fallback OptionA 100% - hiện ra thành "hợp tác 100%". Luôn kiểm `fallback_rate` trước khi tin bất kỳ con số nào. |
| **smoke** | Slug ngoài allowlist tụt xuống 40 lượt gọi mà không báo lỗi. `FULL_SWEEP_MODELS` ở `pd_task.py:213` - thêm slug hoặc đặt `PD_FULL=1`, và nhớ push lại file vì server chạy bản snapshot đã push. |
| **utf-8** | Template ar/cn/vn nằm inline trong `pd_task.py`. Mở và lưu bằng editor không phải UTF-8 sẽ làm hỏng chính prompt gửi cho model. Chạy `PD_SKIP_RUN=1 pytest kaggle/benchmarks/test_pd_task_parity.py -q` trước mỗi lần push. |
| **va chạm** | Hai điều kiện cùng magnitude sẽ ghi đè lên nhau. `_condition_key`, tên file checkpoint và đường dẫn CSV đều chỉ khoá theo λ. Bất kỳ thí nghiệm notation nào cũng phải thêm nhãn riêng trước khi chạy. |

---

## 7. Những thứ KHÔNG nên chạy

- **Thêm ngôn ngữ.** Năm ngôn ngữ đã đủ mang kết quả framing; ngôn ngữ thứ sáu không
  trả lời câu hỏi nào đang mở.
- **Thêm lượt lặp ở các λ đã có.** Khoảng tin cậy hiện tại đã đủ hẹp để tách các
  regime; chi thêm ở đó không mua được gì.
- **Đối chứng người thật.** Thú vị, nhưng là một bài khác, và không làm mạnh thêm
  khẳng định về bất biến.
- **Read-out biết từ chối trả lời.** Bài đã nêu ở phần hạn chế - nhưng đó là việc
  *phân tích*, không cần dữ liệu mới. Làm khi viết lại, không phải khi chạy.

---

## 8. Quyết định cần chốt trước khi tiêu tiền

`paper/` và `paper_scaling/` dùng chung corpus và **chỉ một trong hai được nộp**. E4
(reasoning models) là phần mở rộng dùng chung cho cả hai, nên quyết định đó nên chốt
trước khi phân bổ ngân sách - nó đổi thứ tự ưu tiên của mọi thứ còn lại.

---

*Dựng từ `Analysis/tables/T_PS*.csv`, `kaggle/benchmarks/pd_task.py`, `FAIRGAME/src/`
và `paper_scaling/main.tex`.*
