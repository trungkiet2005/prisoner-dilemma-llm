# Kế hoạch chạy - `paper_scaling`

> **Hướng đã chốt 2026-09-02: bài đi theo NHÁNH FRONTIER, bỏ nhánh open-weight.**
> Mọi ưu tiên dưới đây viết lại theo hướng đó.

Ngân sách: dò thực tế 2026-09-03 còn **6 account dùng thoải mái** (`vinhdinhthien`,
`acc1`-`acc5`), 3 account gần cạn, 3 account token hỏng phải regenerate. Dò bằng
`kaggle/benchmarks/probe_quota.py`.
Đã tiêu: **$28,40 / 9 run** (2026-09-02) + **$2,19** (replicate) + **$2,30** (F1)
+ **$1,31** (E2 lượt hỏng vì quota) + **$3,68** (F2 GPT-5.4-Nano)
+ **$3,10** (E2 `dec2`) + **$3,14** (E2 `dec3` placebo) ≈ **$44,10**.

---

## 0. Frontier-only đổi những gì

**Được:**

- Bỏ hẳn cảnh báo "hai nhánh khác horizon, khác nhiệt độ, khác ma trận gốc nên không
  so sánh gì được" - đó là hạn chế nặng nhất về thiết kế của bản hiện tại.
- Model frontier là thứ người ta thật sự triển khai, nên chuẩn báo cáo ở §4.7 có
  trọng lượng ứng dụng hơn hẳn.
- Toàn bộ dữ liệu mới thu ngày 2026-09-02 đều thuộc nhánh frontier.
- Nhánh frontier **to hơn trước**: từ 4.800 lên **9.600 agent-game** nhờ các sweep mới.
  (Nhánh open-weight bị bỏ có 7.200 agent-game / 216.000 lượt quyết định.)

**Mất, và phải xử lý:**

- **§3.2 mất nền chứng cứ.** Khẳng định "notation thắng magnitude" chủ yếu tựa vào
  Llama và Qwen3, cả hai đều open-weight. Bỏ nhánh đó là mất hai model duy nhất thật
  sự ủng hộ nó. Tệ hơn, dữ liệu frontier mở rộng **bác nó** - xem §2.
- **Mất phép lặp lại giữa hai nhánh.** Bài dùng "cùng chiều, cùng hình dạng ở cả hai
  nhánh" làm bằng chứng vững chắc. Phải thay bằng nguồn khác - xem §2.
- **Mất regime `large` ở 4/5 model.** Chỉ `Gemini-3.5-Flash-Lite` có λ ≥ 100.

---

## 1. Corpus frontier hiện có

| Model | số λ | khoảng λ | agent-game | chạy được trên Kaggle? |
|---|---|---|---|---|
| **Gemini-3.5-Flash-Lite** | **10** | 0.01 - 1000 | 4.000 | có |
| **Gemini-3.1-Flash-Lite-Preview** | **10** | 0.01 - 1000 | 4.000 | có (F1 xong 2026-09-03) |
| **GPT-5.4-Nano** | **10** | 0.01 - 1000 | 4.000 | có (F2 xong 2026-09-03) |
| Claude-3.5-Haiku | 3 | 0.1 - 10 | 1.200 | **không** |
| GPT-4o | 3 | 0.1 - 10 | 1.200 | **không** |
| Mistral-Large | 3 | 0.1 - 10 | 1.200 | **không** |

**Bộ frontier không còn thuần Google.** `GPT-5.4-Nano` vào được sau khi gỡ `tool_choice`
(F2), và nó là model có phụ thuộc λ **mạnh nhất** trong ba model.

Cộng thêm **Stag Hunt** (`Gemini-3.5-Flash-Lite`, 4 λ, 800 game) ở
`Dataset/data_stag_hunt_frontier/`.

**Vấn đề trung tâm của hướng frontier-only:** ba model cuối kẹt ở 3 mức λ. Chúng
**không có trong danh sách 42 slug của Kaggle Benchmarks** - dữ liệu gốc thu qua
connector native của FAIRGAME với API key nhà cung cấp (`llm_factory_connector.py:29-31`).
Đó là vấn đề *danh sách model*, không phải vấn đề proxy: proxy server-side phục vụ
~28/38 model (§4), chỉ là không có `claude-3-5-haiku`, `gpt-4o`, `mistral-large` trong
đó. Model Claude/GPT thế hệ mới thì CÓ, nhưng chúng là model khác, không nối tiếp được
dữ liệu 3 λ đã có.

Với 3 điểm λ, thang BIC **suy biến**: đa thức bậc 2, bậc 3, `fractional + glyph` và mô
hình bão hoà đều là **cùng một mô hình** (BIC 2911.101, r² 0.172070 giống hệt nhau).
Ba model đó không đóng góp được gì cho bất kỳ khẳng định nào về *hình dạng*.

---

## 2. Khẳng định nào sống, khẳng định nào chết

Thang BIC dựng lại trên lưới λ mở rộng - `Analysis/scripts/53_e1_ladder.py`,
`Analysis/tables/T_E1_ladder.csv`:

**Ba model giờ đều có 10 mức λ.** Đặt cạnh nhau, thang BIC cho **ba** câu trả lời khác
nhau về hình dạng - và đó là kết quả, không phải trục trặc:

| spec | k | **Gemini-3.5-Flash-Lite** | **Gemini-3.1-Flash-Lite-Prev.** | **GPT-5.4-Nano** |
|---|---|---|---|---|
| saturated in λ | 15 | **0,0** | 57,9 *(tệ nhất)* | 10,7 |
| linear in log10 λ | 7 | 49,1 | **0,0** | 67,3 |
| quadratic in log10 λ | 8 | 56,0 | 7,6 | **0,0** |
| cubic in log10 λ | 9 | 63,8 | 13,3 | 5,2 |
| fractional + glyph count | 8 | 62,5 | 8,4 | 124,7 |
| notation regime (λ) | 8 | 45,7 | 11,0 | 129,7 |
| notation regime (**PRINTED**) | 8 | 62,0 | 15,7 | 120,3 |
| fractional flag | 7 | 64,2 | 29,7 | 121,3 |
| controls only | 6 | 80,9 | 37,1 | 220,9 |

### Sống, và mạnh lên

**Bất biến thất bại ở CẢ BA model.** `controls only` xếp bét ở cả ba bảng, cách đặc tả
thắng cuộc 80,9 / 37,1 / **220,9** điểm BIC. Ba model độc lập, hai họ nhà cung cấp,
10 mức λ mỗi model - đây là đóng góp chính và nó có phép lặp lại vững.

**Nhưng HÌNH DẠNG của thất bại đó phụ thuộc model, và khác nhau rõ rệt.**

| λ | ô in ra | Gemini-3.5-Flash-Lite | Gemini-3.1-Flash-Lite-Prev. | GPT-5.4-Nano |
|---|---|---|---|---|
| 0.01 | `0 / 0.02 / 0.06 / 0.1` | 0,792 | 0,641 | **0,438** |
| 0.1 | `0 / 0.2 / 0.6 / 1` | 0,694 | 0,648 | 0,540 |
| 0.25 | `0 / 0.5 / 1.5 / 2.5` | **0,872** | 0,621 | 0,644 |
| 0.5 | `0 / 1 / 3 / 5` | 0,800 | 0,640 | 0,598 |
| 1 | `0 / 2 / 6 / 10` | 0,774 | 0,635 | 0,576 |
| 2 | `0 / 4 / 12 / 20` | 0,710 | 0,614 | 0,637 |
| 5 | `0 / 10 / 30 / 50` | 0,697 | 0,614 | **0,680** |
| 10 | `0 / 20 / 60 / 100` | 0,754 | 0,598 | 0,633 |
| 100 | `0 / 200 / 600 / 1000` | 0,734 | 0,584 | 0,663 |
| 1000 | `0 / 2000 / 6000 / 10000` | 0,681 | **0,577** | 0,653 |
| | **biên độ** | 0,178 | 0,071 | **0,242** |

- **Gemini-3.5-Flash-Lite - dao động không đơn điệu.** Biên độ 0,178, đổi chiều năm
  lần, **năm bước liền kề có ý nghĩa** và chúng ngược dấu nhau. Mô hình bão hoà thắng
  mọi đa thức 49-64 điểm BIC.
- **Gemini-3.1-Flash-Lite-Preview - trôi đơn điệu và trơn.** 0,648 xuống 0,577 qua năm
  bậc độ lớn, biên độ 0,071, **không một bước liền kề nào có ý nghĩa** (p nhỏ nhất
  0,25). `linear in log10 λ` thắng; mô hình bão hoà là đặc tả TỆ NHẤT, tệ hơn cả
  `controls only` 20 điểm BIC.
- **GPT-5.4-Nano - đơn điệu tăng rồi bão hoà.** Biên độ **0,242**, lớn nhất trong ba.
  Hai bước đầu mỗi bước +0,103 (p<0,001), sau λ=5 thì phẳng ra. `quadratic in log10 λ`
  thắng - đúng hình dạng của một đường cong bão hoà, và nó bỏ `linear` 67,3 điểm BIC.

Ba kiểu vi phạm này chỉ phân biệt được nhờ sàn nhiễu đo từ replicate (|Δ| trung vị
0,008, lớn nhất 0,038): các bước của Gemini-3.5 và GPT-5.4-Nano nằm **trên** sàn, các
bước của Gemini-3.1 nằm **dưới** - nên với model đó, chỉ xu hướng cộng dồn mới có thật,
còn từng bước thì không.

Phát biểu đúng cho bài: **bất biến thất bại một cách đáng tin cậy; cách nó thất bại thì
không.** Điều đó mạnh hơn "phản ứng không trơn", vì nó chặn trước phản biện "các anh
chỉ fit nhiễu của một model".

#### Đã chống được phản biện chí mạng: replicate (2026-09-03)

Mô hình bão hoà thắng cũng là điều một mô hình fit nhiễu theo ô sẽ làm, và **cả corpus
không có replicate nào ở cùng một λ** để phân biệt hai khả năng. Đó là câu hỏi rẻ nhất
reviewer có thể đặt, và nó nhắm đúng vào đóng góp duy nhất còn sống.

Chạy lại `gemini-3.5-flash-lite` ở λ = 0.1 / 0.25 / 0.5 / 1 với `BASE_SEED` 67890
(đợt gốc 12345), 800 game, `fallback_rate` 0.0, `parse_fail_rate` 0.0, **$2,19**,
16 phút. `Analysis/scripts/55_replicate_testretest.py` → `T_R1_testretest.csv`.

| λ | lần 1 | lần 2 | Δ | p |
|---|---|---|---|---|
| 0.1 | 0,694 | 0,696 | +0,002 | 0,96 |
| 0.25 | 0,872 | 0,858 | -0,014 | 0,43 |
| 0.5 | 0,800 | 0,802 | +0,002 | 0,92 |
| 1 | 0,774 | 0,812 | +0,038 | 0,029 |

**Sàn nhiễu: |Δ| trung vị 0,008, lớn nhất 0,038.** Bước 0.1 → 0.25 - bước lớn nhất
toàn lưới và là trụ cột của kết luận "không trơn" - lặp lại sạch: **+0,178** ở đợt gốc
so với **+0,162** ở replicate, chênh lệch p = 0,70. Biên độ đó gấp **22 lần** sàn nhiễu.

⇒ Phản ứng theo λ có thật, không trơn, và **không phải nhiễu theo ô**. Đóng góp (i)
đứng vững.

Phải nói thẳng trong bài: ô λ=1 lệch 0,038 với p = 0,029. Với 4 phép so sánh thì ngưỡng
Bonferroni là 0,0125 nên chưa gọi là khác nhau được, nhưng **sàn nhiễu đo được là ~0,04
chứ không phải 0**. Mọi hiệu ứng dưới mức đó không được diễn giải.

#### Kiểm định cặp cho E1 (`54_e1_pairwise.py` → `T_E1_offdecade_pairs.csv`)

Thang BIC nói đặc tả nào thắng nhưng không nói hiệu ứng nằm ở đâu. Bootstrap dyad
8000 lần trên từng bước λ liền kề, cùng phương pháp E7:

| bước | ×độ lớn | notation | Δ | p |
|---|---|---|---|---|
| 0.1 → 0.25 | 2,5 | giữ | **+0,178** | **<0,001** |
| 0.25 → 0.5 | 2 | **đổi** | -0,071 | <0,001 |
| 1 → 2 | 2 | giữ | -0,064 | 0,002 |
| 10 → 100 | 10 | giữ | -0,020 | 0,40 |

Bước lớn nhất toàn lưới **không vượt ranh giới notation nào** và chỉ đổi độ lớn 2,5
lần, trong khi bước 10 lần ở 10 → 100 không dịch gì. Trên `Gemini-3.1`, bước vượt ranh
giới cho Δ = -0,008 (p = 0,72), null sạch. Đây là bằng chứng độc lập với thang BIC cho
cùng kết luận ở mục "Chết" dưới đây.

### Chết

**"Notation thắng magnitude" - đã bác được ở CẢ HAI model, trong CẢ HAI cách vận hành
hoá.** Không đặc tả notation nào thắng ở model nào có đủ 10 mức λ. Với 3 điểm thì
notation thắng 8,4 BIC, nhưng đó là hình phạt tham số chứ không phải độ khớp - r² của
nó còn *thấp hơn*.

Chi tiết quyết định, và nó lặp lại: `notation regime (PRINTED)` - cách vận hành hoá
**đúng**, đọc regime từ ô in ra nên λ=0.5 (`0 / 1 / 3 / 5`) là `unit` chứ không phải
`fractional` - lại **tệ hơn** cách dựa vào λ ở cả hai model (62,0 so với 45,7 ở model
1; 15,7 so với 11,0 ở model 2). Phần "thắng" của notation trong bản cũ đến từ chỗ nó
tình cờ trùng một hàm của λ trên lưới decade, không từ cách in số. Đó là điều kiểm
được: cách in số là thứ agent thật sự nhìn thấy, và nó khớp *kém hơn*.

Kiểm định cặp nói cùng một điều, độc lập với BIC: ở **cả ba** model, **bước liền kề lớn
nhất không vượt ranh giới notation nào**, và bước duy nhất có vượt ranh giới (0.25 →
0.5) thì nhỏ hơn:

| model | bước lớn nhất (cùng notation) | bước vượt ranh giới 0.25 → 0.5 |
|---|---|---|
| Gemini-3.5-Flash-Lite | **+0,178** (0.1→0.25, p<0,001) | -0,071 (p<0,001) |
| Gemini-3.1-Flash-Lite-Prev. | -0,028 (0.1→0.25, p=0,25) | +0,020 (p=0,40) |
| GPT-5.4-Nano | **+0,103** (0.01→0.1 và 0.1→0.25, p<0,001) | -0,046 (p=0,08) |

### Nhưng E2 cho notation một kết quả KHẲNG ĐỊNH, trên một model

Mọi thứ trên đây là quan sát trên lưới λ, nơi notation và magnitude dính chặt nhau. E2
tách chúng ra bằng cách giữ nguyên λ và chỉ đổi cách IN payoff, với một **placebo thật**
để trừ đi nhiễu giữa các lần chạy:

| | λ=1 in ra | vai trò |
|---|---|---|
| `native` | `0 / 2 / 10 / 6` | gốc |
| `dec2` | `0.00 / 2.00 / 10.00 / 6.00` | ô nguyên thành thập phân |
| `dec3` | `0.000 / 2.000 / 10.000 / 6.000` | **placebo**: đã thập phân sẵn, chỉ thêm một số 0 |

| model | native → dec2 (ranh giới) | dec2 → dec3 (placebo) | kết luận |
|---|---|---|---|
| **Gemini-3.1-Flash-Lite-Prev.** | +0,046 / **+0,070** / **+0,065** - cùng dấu, tb 0,060 | -0,027 / +0,008 / -0,028 - không ô nào có ý nghĩa, tb 0,021 | **hiệu ứng ranh giới THẬT** |
| Gemini-3.5-Flash-Lite | -0,059 / **-0,047** / **+0,051** - **đổi dấu**, tb 0,052 | +0,007 / +0,011 / **-0,058** | không kết luận được |

Trên `Gemini-3.1`, in payoff dưới dạng thập phân **nâng hợp tác lên ~0,06 ở mọi λ**,
trong khi thêm một số 0 nữa thì không làm gì. Đó là bằng chứng nhân quả, không phải
tương quan. Trên `Gemini-3.5` thì dấu đổi chiều giữa λ=1 và λ=10 và chính placebo lại
có một ô p=0,005, nên không đọc được gì - đúng dạng của nhiễu.

**Chỗ này hoà giải hai kết quả tưởng như mâu thuẫn.** Hiệu ứng notation trên
`Gemini-3.1` là một **dịch chuyển mức gần như hằng số**, không phụ thuộc λ. Một hằng số
thì không giải thích được sự phụ thuộc vào λ - nên nó có thật mà vẫn thua sạch trong
thang BIC, và cả hai điều đó đều đúng cùng lúc.

**Và magnitude sống sót khi giữ notation cố định** - đây mới là điều quan trọng nhất
của E2 với đóng góp (i):

| bước (10× độ lớn) | Gemini-3.5 `dec2` | Gemini-3.1 `dec3` |
|---|---|---|
| 0.1 → 1 | **+0,092** (p=0,004) | +0,046 (p=0,024) |
| 1 → 10 | **+0,078** (p<0,001) | **-0,078** (p<0,001) |

Mọi ô ở cả hai đầu đều in cùng một kiểu glyph, nên phụ thuộc λ **không phải là hệ quả
của việc ô đổi từ nguyên sang thập phân**.

Một hạn chế phải nói: ba nhánh chạy trên ba account và ba thời điểm khác nhau, nên một
độ lệch ở mức lần-chạy bị lẫn vào phép so. Nhánh placebo chính là thứ đo độ lệch đó
(tb 0,021), và hiệu ứng ranh giới trên `Gemini-3.1` lớn gấp ba lần nó.

### Thay thế cho phép lặp lại giữa hai nhánh

**Trò chơi thứ hai, và nó đã có phép lặp lại trên model thứ hai (F3, 2026-09-03).**
Stag Hunt cho kết quả **ngược hẳn** với Prisoner's Dilemma:

| bước | ×độ lớn | notation | Gemini-3.1-Flash-Lite-Preview | Gemini-3.5-Flash-Lite |
|---|---|---|---|---|
| 0.1 → 0.5 | 5 | **đổi** | **-0,201** (p<0,0001) | **-0,117** (p=0,005) |
| 0.5 → 1 | 2 | giữ | +0,044 (p=0,25) | +0,055 (p=0,15) |
| 1 → 10 | 10 | giữ | **+0,175** (p<0,0001) | +0,028 (p=0,46) |

Bước vượt ranh giới notation có ý nghĩa ở **cả hai** model, cùng dấu, cùng cỡ. Bước
0.5 → 1 giữ nguyên notation thì không có ý nghĩa ở model nào.

Phải nói thẳng một chỗ: `Gemini-3.1` **cũng** có một bước magnitude thuần có ý nghĩa
(1 → 10, +0,175), nên với model đó không phải "chỉ notation mới quan trọng" - cả hai
đều quan trọng, và bước notation lớn hơn (0,201 so với 0,175) dù thay đổi độ lớn nhỏ
hơn (5× so với 10×). Model kia thì bước magnitude không có ý nghĩa.

Kết luận trung thực: **trong Stag Hunt, notation có tác dụng ở cả hai model; magnitude
thì tuỳ model. Trong Prisoner's Dilemma, notation không có tác dụng ở model nào.** Hiệu
ứng notation là đặc tính của **cặp model × trò chơi**, không phải của cách in số nói
chung - và đó là phát biểu mạnh hơn cả hai phía tranh luận trong bản cũ.

Đó là phát biểu mạnh hơn và trung thực hơn cả hai phía tranh luận trong bản cũ, và nó
biến "lặp lại theo quy mô model" thành **"lặp lại theo trò chơi"** - phù hợp với hướng
frontier-only, vì frontier không có trục quy mô để mà lặp.

---

## 3. Việc cần chạy tiếp

### F1. Mở `Gemini-3.1-Flash-Lite-Preview` lên đủ 10 mức λ - **XONG 2026-09-03**

Chạy 5 λ còn thiếu (`0.01, 0.25, 2, 5, 1000`) trên account `vinhdinhthien`: 1000 game,
`fallback_rate` 0,0, `parse_fail_rate` 0,0, **$2,30**, 29 phút. Kết quả: §2.

Hai chi tiết đáng ghi lại:

- **Bỏ λ=0.5 khỏi lần chạy lại** dù RUN_PLAN cũ đề nghị giữ nó làm điểm kiểm chống lặp.
  Lý do: `BASE_SEED` không đổi nên CRN sẽ trả lại gần đúng số cũ - đó không phải phép
  test-retest, mà chỉ là rủi ro ghi đè bản gốc (BẪY 9). Phép test-retest thật đã làm
  riêng ở replicate với seed 67890.
- **Kaggle mount output của run TRƯỚC vào run sau** để resume. Run 429 hỏng đã để lại
  một mảnh 15 dòng ở `0.5/`, và nó đi theo gói tải về. Luôn đếm số dòng từng λ trước
  khi chép vào `Dataset/` - 200 dòng mỗi λ, không đủ thì là mảnh vỡ.

```
# sua default trong pd_task.py (env var KHONG toi duoc server - xem §5):
GAME    = "prisoner_dilemma"
LAMBDAS = [0.25, 0.5, 2, 5, 0.01, 1000]   # 0.5 lam diem kiem chong lap
FULL_SWEEP_MODELS = ["gemini-3.1-flash-lite-preview"]
RUN_TAG = ""        # dang la "-rep2" tu dot replicate 09-03, PHAI dat lai
BASE_SEED = 12345   # dang la 67890 tu dot replicate, tra ve gia tri goc
```

### F2. Gỡ lỗi `tool_choice` - **XONG 2026-09-03**

Chẩn đoán cũ ghi "lỗi SDK kbench" là **sai**. `pd_task.py` tự gửi `tool_choice="none"`
(để model khỏi trả tool_call rỗng). Endpoint OpenAI/xAI từ chối tham số đó khi không
kèm `tools`; proxy Google thì nhận, nên lỗi chỉ lộ ra ở model ngoài Google:

```
Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.
```

Sửa mất hai vòng, và vòng đầu **hỏng theo cách đáng ghi lại**:

1. Thêm cờ một lần cho cả run: thấy lỗi thì bỏ `tool_choice` rồi thử lại. Run vẫn chết
   ngay - `CONCURRENCY=8` nên tám worker cùng bay vào cùng một lỗi 400. Worker đầu gỡ
   được tham số; các worker sau nhận đúng lỗi đó nhưng **không còn gì để gỡ**, hàm trả
   "không nhận ra lỗi này" và ngoại lệ rơi thẳng ra ngoài, giết cả run (run 1034690).
2. Sửa đúng: phân biệt **"vừa gỡ"** (thử lại ngay, không tính lượt) với **"đã gỡ từ
   trước"** (request bay song song gửi trước lúc cờ kịp bật -> retry có backoff). Thêm
   một **lượt gọi khởi động tuần tự** trước khi mở pool, để cờ tham số ổn định xong mới
   chạy song song. Một lượt gọi, và lỗi tham số nào cũng lộ ra ở dòng log đầu tiên.

Bài học tổng quát hơn: **cờ "sửa một lần cho cả run" và pool worker là tổ hợp nguy
hiểm.** Cứ có cờ kiểu đó thì phải trả lời được câu "worker thứ hai gặp lỗi đó thì sao"
- và có một lượt gọi tuần tự trước khi mở pool là cách rẻ nhất để không phải trả lời.

Kết quả: `gpt-5.4-nano-2026-03-17` chạy sweep thật 10 λ × 200 game, `parse_fail=0`,
`fallback=0`. Đây là **model ngoài Google đầu tiên** trong corpus frontier. `grok-4.20`
dính đúng lỗi đó nên giờ cũng mở được.

### F3. Stag Hunt trên model thứ hai - **XONG 2026-09-03**

800 game, `fallback_rate` 0,0. Kết quả ở §2 "Thay thế cho phép lặp lại". Bẫy vấp phải:
`RUN_TAG` mặc định `-rep2` (còn sót từ đợt replicate) dính vào `model_tag` nên 4 thư
mục ra sai tên dù đây là lần chạy Stag Hunt ĐẦU TIÊN của model này - đã đổi tên lại.
**Đặt lại `RUN_TAG=""` ngay sau mỗi đợt replicate.**

### F4. Ba model kẹt ở 3 λ - quyết định về ngân sách

`Claude-3.5-Haiku`, `GPT-4o`, `Mistral-Large` chỉ mở rộng được qua **API nhà cung cấp**,
không phải credit Kaggle. Ba lựa chọn:

1. **Bỏ tiền vendor API** mở chúng lên 6-10 λ. Đắt nhất nhưng cứu được cả ba và giữ
   được đa dạng họ model.
2. **Giữ nguyên 3 λ**, dùng chúng chỉ cho các kết quả *không* cần hình dạng: kiểm bất
   biến thất bại (cần ≥2 điểm), đảo thứ hạng, persona, nhãn chiến lược. Nói rõ trong
   bài rằng chúng không đóng góp cho §3.2.
3. **Bỏ hẳn**, chạy bài trên 2-3 model Google. Gọn nhưng hẹp.

Khuyến nghị **lựa chọn 2**, và nếu ngân sách vendor cho phép thì thêm `GPT-4o` theo
lựa chọn 1 - nó là model ngoài Google duy nhất đã có sẵn dữ liệu 3 λ.

### F5. E2 - notation cố định / magnitude cố định - **XONG 2026-09-03**

Đây là cách duy nhất phát biểu điều gì đó **khẳng định** về notation thay vì chỉ bác bỏ,
nên nó quan trọng hơn hẳn sau khi §3.2 phải viết lại.

Đã cài `PD_WEIGHT_FORMAT` trong `pd_task.py`: `native` (mặc định, y như FAIRGAME) hoặc
`decN` (luôn N chữ số thập phân). Điểm số vẫn đi bằng SỐ - chỉ prompt đổi cách in. Hậu
tố `format_tag()` tự nối vào `model_tag` nên `dec2` ở λ=1 không ghi đè `native` ở λ=1;
hậu tố là **hàm** chứ không phải hằng số để nó không thể lệch khỏi `WEIGHT_FORMAT`.
Thêm 5 test (23/23 qua), và fixture parity ghim `native` để bộ test không phụ thuộc vào
cấu hình của lần push đang chuẩn bị.

**Thiết kế chỉ tốn 3 ô mới mỗi model**, vì nhánh `native` đã có sẵn: chạy λ ∈ {0.1, 1,
10} ở `dec2`, cùng `BASE_SEED = 12345` với dữ liệu gốc nên hai nhánh ghép cặp được.

| | ô in ra | so với |
|---|---|---|
| **A. notation** (magnitude cố định) | λ=1: `6` với `6.00`; λ=10: `60` với `60.00` | payoff Y HỆT -> chênh lệch nào cũng là notation |
| **B. magnitude** (notation cố định) | `0.60` / `6.00` / `60.00` | glyph cùng loại -> chênh lệch nào cũng là magnitude |
| **placebo** | λ=0.1: `0.6` với `0.60` | chỉ thêm một số 0 -> phải ra ~0, nếu không thì ta đang đo nhiễu giữa hai lần chạy |

Placebo là chỗ quan trọng nhất của thiết kế: nếu ô λ=0.1 cũng lệch bằng ô λ=1 thì hiệu
ứng "notation" chỉ là nhiễu run-to-run, và sàn nhiễu đã đo được từ replicate (|Δ| ~0,04)
là thước đo để phán.

Phân tích: `Analysis/scripts/56_e2_notation_vs_magnitude.py` -> `T_E2_notation.csv`.
Dữ liệu để riêng ở `Dataset/data_fairgame_e2_notation/`, KHÔNG vào
`data_fairgame_frontier_llm/`, để `ingest.py` không trộn hai cách in vào cùng một ô λ.

**Thiết kế ban đầu SAI ở chỗ placebo, và phải chạy thêm một nhánh để sửa.** Bản đầu lấy
λ=0.1 làm placebo với lý do `0.6` → `0.60` chỉ thêm một số 0. Nhưng S = 0 trong mọi cấu
hình, nên `native` luôn in `0` còn `dec2` in `0.00`: **mọi λ đều có ít nhất một ô vượt
ranh giới nguyên→thập phân**, không λ nào tránh được. Placebo vì thế phải nằm trên trục
CÁCH IN chứ không phải trục λ - nhánh `dec3` (`6.000` so với `6.00`, cả hai đã thập
phân) mới là placebo sạch. Thêm 600 game/model, ~$1,5/model.

**Và tiêu chí đọc kết quả cũng phải sửa.** Bản đầu của script chỉ so `|Δ|` trung bình
giữa nhánh ranh giới và nhánh placebo, nên nó tuyên "hiệu ứng notation THẬT" cho cả
`Gemini-3.5` - model có Δ **đổi dấu** giữa λ=1 và λ=10. `|Δ|` trung bình lớn mà dấu đổi
chiều chính là hình dạng của NHIỄU. Tiêu chí giờ đòi cả ba: dấu nhất quán ở mọi λ, trên
sàn nhiễu, và placebo dưới một nửa.

### F6. Mở rộng bộ model - THỬ LẠI LẦN HAI 2026-09-03, VẪN CHỦ YẾU HỎNG

Thử lại 6 model sau khi gỡ được `tool_choice`, mỗi model một account, quota đã dò sạch
trước khi giao việc. Kết quả chia thành **hai loại hỏng khác hẳn nhau**, và phân biệt
được chúng là thứ có giá trị nhất từ đợt này:

| model | kết quả | loại |
|---|---|---|
| `gpt-oss-20b` | `choices[0].message=None` | **hỏng vĩnh viễn ở tầng proxy** |
| `glm-5` | `choices[0].message=None` (sau khi warm-up ĐÃ qua) | **hỏng vĩnh viễn ở tầng proxy** |
| `gemma-4-26b-a4b-it`, `gemma-4-31b-it` | `message=None` (đợt trước) | **hỏng vĩnh viễn ở tầng proxy** |
| `grok-4.20-0309-non-reasoning` | warm-up OK, 1 game chạy thật, rồi 429 | tạm thời + **quá chậm** |
| `qwen3-next-80b-a3b-instruct` | 429 heavy load ngay ở warm-up | tạm thời |
| `deepseek-v3.1` | 503 not reachable ngay ở warm-up | tạm thời |
| `gpt-oss-120b` | 429 ở warm-up, **và vẫn 429 khi thử lại với 14 lượt retry** | tạm thời nhưng kéo dài |

**Bốn model `message=None` nên coi là chết.** Proxy trả về một response hợp lệ nhưng
không có `message`, ta không sửa được gì từ phía mình, và nó lặp lại qua nhiều ngày và
nhiều account. Đừng thử lại nữa trừ khi Kaggle thông báo đã sửa.

**`grok-4.20` thì `tool_choice` ĐÃ thông** - warm-up qua, một game chạy trọn với dữ
liệu thật. Nhưng nhịp gọi đo được là ~75 giây/lượt, tức là 40.000 lượt sẽ mất hơn 40
giờ, vượt xa giới hạn thời gian của một run Kaggle. Không phải vấn đề sửa được bằng
code: model này không dùng được qua proxy ở nhịp đó.

**Lượt gọi khởi động tuần tự chứng minh giá trị ngay lần đầu.** Cả bốn model hỏng đều
lộ nguyên nhân ở **dòng log đầu tiên** thay vì sau vài trăm game, nên mỗi lần thử hỏng
chỉ tốn vài xu và vài giây.

**Đã nới ngân sách retry cho lỗi TẠM THỜI** (`PD_TRANSIENT_ATTEMPTS`, mặc định 14, kèm
jitter) tách khỏi ngân sách 6 lượt dùng cho lỗi thật - vì 429 thì *chờ chính là cách
xử lý đúng*, còn lỗi thật thì retry chỉ đốt thời gian.

**Phép kiểm đã chạy, và câu trả lời là KHÔNG.** `gpt-oss-120b` chạy lại với ngân sách
14 lượt vẫn `Errored`, vẫn 429 ngay ở warm-up. Nên 429 ở đây **không phải một đợt tăng
tải ngắn** mà là trạng thái kéo dài của model đó trên proxy, và không có chiến lược
retry nào ở phía client cứu được. Thay đổi này vẫn giữ (nó không tốn gì và có ích cho
đợt tải ngắn thật), nhưng **đừng trông vào nó để mở model mới**.

Hệ quả cho việc lên lịch: khác biệt duy nhất còn đáng thử là **thời điểm**, không phải
tham số. Nếu thử lại thì thử vào giờ khác trong ngày, và dừng ngay sau warm-up nếu vẫn
429 - phép thử tốn vài xu và vài giây.

**Kết luận cho bài:** bộ model khả dụng qua Kaggle Model Proxy hiện là Google
flash-lite cộng `gpt-5.4-nano`. Đa dạng nhà cung cấp bị giới hạn bởi hạ tầng chứ không
phải bởi ngân sách, và §4.6 của bài phải nói đúng như vậy.

### Đã bỏ

- **E1 nhánh open-weight** - ngoài phạm vi theo hướng mới.
- ~~**E5 mở rộng 20 model**~~ - **KHÔI PHỤC**, xem F6. Lý do loại trước đây ("proxy chỉ
  phục vụ 7 model") dựa trên phép dò local và **không đúng cho server-side** (§4).
- **E6 phép cộng `u -> a + u`** - confound bốn hướng cùng lúc: thang `a` làm số chữ số
  đều nhau, làm tương phản `(max-min)/mean` sụp từ 2,2222 xuống 0,0100 (giảm 223 lần),
  xoá mất ô số 0, và tạo tiền tố chung. Thêm nữa prompt diễn đạt payoff là *hình phạt*
  nên `+a` thực chất là `u -> u - a`, ngược dấu với §2.1.

---

## 4. Trạng thái hạ tầng (ĐÍNH CHÍNH 2026-09-02)

> ⚠️ **Đính chính một kết luận sai.** Bản trước của mục này viết "proxy chỉ phục vụ
> model Google" dựa trên phép dò từ máy local. **Sai.** Proxy local và proxy
> server-side là **hai thứ khác nhau**: local phục vụ ~6/38 model, server-side ~28/38.
> **503 ở local không nói được gì về việc model có chạy trên server hay không.** Phép
> kiểm duy nhất có thẩm quyền là submit một task run thật.

### Đã đo được gì, và bằng đường nào

**Server-side (có thẩm quyền)** - từ các lần submit thật:

| Model | Kết quả server-side |
|---|---|
| gemini-3.5-flash-lite | chạy trọn 1400 + 800 + 200 game, fallback 0.0 |
| gemini-3.1-flash-lite-preview | chạy trọn 1000 game, fallback 0.0 |
| gemini-3.5/3.6/3.7-flash | chạy nhưng **thinking**: parse_fail 20-34%, đắt gấp 16 lần |
| gemma-4-26b-a4b-it, gemma-4-31b-it | proxy trả `choices[0].message=None` → **fallback 1.0** |
| gpt-oss-20b, glm-5, qwen3-next-80b | **429 heavy load** |
| deepseek-v3.1 | **503 not reachable** |
| gpt-oss-120b | **403** - đặt cọc theo `max_output_tokens` vượt quota account |
| gpt-5.4-nano, grok-4.20-non-reasoning | **400 `tool_choice`** (lỗi SDK, xem F2) |

**F6 đã chạy 2026-09-03 và cho kết luận dứt điểm: 0/7 model ra được dữ liệu dùng
được**, kể cả khi làm đúng cách (một model một account, giãn 30 giây, trần token 128).
Chi phí thất bại: ~$0,03.

Khác biệt quan trọng so với kết luận sai trước đây: giờ đây bằng chứng là **server-side**
nên có thẩm quyền, không phải suy từ phép dò local. Nhưng bản chất lỗi khác nhau:

- **gemma** hỏng thật ở tầng proxy (`message=None`), sửa gì cũng không chạy;
- **429 / 503** là quá tải và mất kết nối phía Kaggle - **có thể thử lại lúc khác**,
  đây không phải kết luận vĩnh viễn;
- **403** là vấn đề account, không phải model - dò quota trước khi giao việc.

**Local (chỉ để tham khảo, KHÔNG dùng để loại model):** dò tuần tự cho thấy chỉ 7 model
trả lời, nhưng con số đó là của proxy local nên không áp cho server.

### Ba sự thật về proxy (đừng phát hiện lại)

1. **Local ≠ server-side.** Xem trên.
2. **503 là lỗi phía Kaggle, không phải hết quota.** Kiểm bằng 3 account cho ra cùng
   một tập 503. Đổi account để né 503 là vô ích.
3. **Proxy đặt cọc tiền trước theo `max_output_tokens`, không theo token thực tiêu.**
   Không cap hoặc cap rộng thì model đắt bị **403** dù thực tế chỉ tốn vài xu. Luôn đặt
   trần chặt, và đọc 403 trên model đắt là vấn đề đặt cọc trước khi nghĩ tới credit.

### Bảng giá đã hiệu chuẩn (8 run, $26,13)

| Hạng | $/cell (200 game) |
|---|---|
| Google flash-lite (không-thinking) | **$0,50** |
| Google flash (**thinking**) | **$8,23** - đắt gấp 16 lần |

Đã thử 5 cách tắt reasoning: `reasoning_effort=none` (400), `reasoning_effort=low`,
`reasoning={effort:none}`, `thinking={type:disabled}`, `thinking_budget=0`. Bốn cách
sau được **nhận nhưng bỏ qua**. Model thinking vẫn coi như không dùng được.

## 5. Bẫy đã ghi nhận (đọc trước mỗi lần chạy)

| Bẫy | Nội dung |
|---|---|
| **env không tới server** | `kaggle b t run` chỉ gửi tên task + tên model. Task chạy từ **snapshot đã push**, nên `PD_LAMBDAS`, `PD_GAME`... đặt local **không có tác dụng**. Phải sửa default rồi push lại. |
| **guard slug** | Server đặt `MODEL = anthropic/claude-haiku-4-5@20251001` - có tiền tố provider và dùng `@` thay `-`. Guard phải chuẩn hoá cả hai, nếu không sweep **âm thầm tụt xuống smoke 40 lượt gọi** mà vẫn báo `Completed`. Đã sửa. |
| **`-m` một model** | `-m a b c` làm argparse lỗi và **không submit gì cả** nhưng script vẫn chạy tiếp. Phải lặp cờ: `-m a -m b`. |
| **fallback** | Model thinking cắt cụt câu trả lời → parse hỏng → fallback OptionA → "hợp tác 100%" giả. Luôn kiểm `fallback_rate` **từng model**, không gộp. |
| **utf-8** | Template ar/cn/vn nằm inline trong `pd_task.py`. Sinh bằng `51_embed_staghunt.py`, đừng gõ tay. Chạy parity test trước mỗi lần push. |
| **va chạm game** | `pd_task` chỉ khoá output theo λ/lang/model. Hậu tố `GAME_TAG` (`-sh`) là thứ duy nhất ngăn Stag Hunt ghi đè PD. Với **replicate cùng λ** thì `GAME_TAG` không đủ - dùng `RUN_TAG` (mặc định `-rep2`), cũng nối vào `model_tag` nên tách hẳn out_dir, tên thư mục model và tên file CSV. |
| **khối `{choose}`** | `assemble_prompt` **xoá mọi khối không khai báo** trong `enabled`. Template Stag Hunt bọc câu lệnh chọn trong `{choose}`; quên khai báo là prompt mất câu yêu cầu chọn mà không lỗi ở đâu cả. |
| **MODEL_MAP** | `ingest.py:143` tra dict trực tiếp → model chưa đăng ký gây **KeyError làm hỏng cả lần ingest**. Đăng ký mọi model mới trước khi phân tích. |
| **hợp tác đếm ngược** | `pd_task.py` từng hard-code `OptionA` là hợp tác cho **cả hai** game. Đúng với Stag Hunt (khung phần thưởng) nhưng **ngược với PD**: prompt PD nói về *hình phạt* và mục tiêu là *tối thiểu hoá*, OptionA cho 6/0 so với OptionB cho 10/2 nên OptionA trội tuyệt đối = phản bội. Mọi `overall_coop_rate` in ra ở nhánh PD là **tỉ lệ phản bội** (`1 - giá trị thật`); assertion cuối run chỉ dùng `fallback_rate` nên không có gì chặn lại. Bảng đã công bố KHÔNG bị ảnh hưởng vì chúng đi qua `ingest.py`, vốn ánh xạ đúng. Đã thay bằng `COOP_STRATEGY` suy ra từ `GAME_FRAMING` (2026-09-03). |
| **MAX_PATH** | Tên model dài lặp 3 lần trong đường dẫn tải về → vượt 260 ký tự của Windows, CLI báo `FileNotFoundError` không rõ lý do. Tải về `D:/tmp/kb` rồi chép sang. |
| **cờ một lần + pool** | `_call_llm` gỡ tham số bị từ chối "một lần cho cả run". Với `CONCURRENCY=8`, worker đầu gỡ được, worker thứ hai nhận đúng lỗi đó nhưng **không còn gì để gỡ** → ngoại lệ rơi ra ngoài, **giết cả run**. Phải tách "vừa gỡ" (thử lại ngay) khỏi "đã gỡ từ trước" (retry có backoff), và gọi **một lượt tuần tự** trước khi mở pool. |
| **mảnh vỡ run trước** | Kaggle mount output run trước vào run sau để `RESUME` chạy được → gói tải về chứa cả thư mục λ mà run này không chạy. Dùng `kaggle/benchmarks/collect_run.py`, nó từ chối cell không đủ 200 dòng và bỏ hậu tố `model_tag` khỏi cả tên file lẫn cột `agent1_llm`. |
| **quota đổi trong ngày** | Account "còn nhiều" buổi sáng có thể cạn buổi chiều và ngược lại. Triệu chứng khi giao nhầm: run báo `Completed` nhưng `complete: False`, `stopped_early_quota` có 403, `coop_by_scale` có ô `None`. Dò lại bằng `kaggle/benchmarks/probe_quota.py` **trước mỗi đợt**, đừng tin danh sách cũ. |

---

## 6. Việc viết lại trong `main.tex`

**Trạng thái 2026-09-03.** Toàn bộ pipeline số liệu đã dựng lại trên corpus
frontier-only: `20_frontier_build.py` (6 model, 15.600 agent-game, 156.000 quyết định)
và `40_scaling_stats.py`. Trục `arm` trong `40_scaling_stats.py` đã đổi nghĩa từ "hai
nhánh corpus" sang **độ phân giải λ** (`ten-scale` / `three-scale`), giữ nguyên tên cột
để mọi hàm downstream không phải sửa.

Đã viết lại: **abstract**, §2.2 corpus, §2.3 notation regimes, §2.6 confounds,
§3.1 response, §3.2 notation, **hai mục Results mới** (§3.3 thí nghiệm cách in,
§3.4 Stag Hunt), §4.2 discussion, §4.6 limitations, caption hình 1-3, và phần
đóng góp ở Introduction.

**Còn lại, và một chỗ trong đó là vấn đề thật chứ không phải chép số:**

1. §3.5-§3.9 (opening move, what does not move, persona gating, language, strategy
   labels) và §4.3-§4.5, §4.7 vẫn còn **74 dòng** nhắc nhánh open-weight hoặc
   Gemma/Llama/Qwen, và số liệu phải lấy lại từ `T_PS06`-`T_PS10` đã dựng lại.
2. **§3.5 "The effect is set at the opening move" không còn phát biểu được như cũ.**
   Thống kê đó dựa trên khoảng cách hợp tác giữa regime `fractional` và `unit` **gộp
   qua các model**. Trên nhóm 10 mức λ, khoảng cách gộp đó giờ là **0,0005** - ba model
   có ba hình dạng khác nhau nên chúng triệt tiêu nhau - và tỉ lệ "phần khoảng cách nằm
   ở vòng 1" thành tỉ số của hai số gần bằng 0, tức là vô nghĩa. Trên nhóm 3 mức λ thì
   khoảng cách là -0,1725 và **103,6% của nó đã có ngay ở vòng 1**, tức khẳng định cũ
   vẫn đúng ở đó. Mục này phải viết lại theo TỪNG MODEL, không gộp - cùng bài học với
   §3.1.
3. `41_fig_scaling.py` mới chỉ cập nhật bảng mã hoá; năm hàm dựng hình vẫn dựng quanh
   trục hai nhánh và chưa chạy được. Caption hình 1-3 đã viết theo bố cục panel MỚI,
   nên hình phải được dựng lại cho khớp trước khi nộp.
4. `42_supp_tables.py` chưa chạy lại.



1. **§2.2** - bỏ mô tả nhánh open-weight và bảng hai nhánh; bỏ phần "confounds we
   declare" nói về khác biệt giữa hai nhánh (không còn nữa).
2. **§3.2 và §4.2** - viết lại hoàn toàn. Kết luận mới: phản ứng là hàm của λ mà không
   đặc tả rút gọn nào bắt được (mô hình bão hoà thắng), và mô tả theo notation **không**
   thắng khi lưới λ đủ dày. Nêu rõ ΔBIC = 8,4 của thiết kế 3 điểm là hình phạt tham số.
3. **§3.9 / §4.4** - thêm Stag Hunt làm phép lặp lại thay cho phép lặp lại giữa hai
   nhánh, và nêu phát hiện notation phụ thuộc trò chơi.
4. **§4.6 hạn chế** - bỏ mục "hai nhánh khác nhau"; thêm mục "ba trong năm model chỉ có
   3 mức λ nên không đóng góp cho khẳng định về hình dạng".
5. **Abstract** - hiện 299 từ, sẽ ngắn lại đáng kể khi bỏ nhánh open-weight.

---

*Cập nhật 2026-09-02 sau 9 run / $28,40. Bảng: `T_E1_offdecade.csv`,
`T_E1_ladder.csv`, `T_E7_staghunt.csv`.*
