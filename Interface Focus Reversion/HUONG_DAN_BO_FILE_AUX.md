# Bỏ file aux: ghi thẳng số cho các tham chiếu giữa main và appendix

Hiện tại `main.tex` và `appendix.tex` tham chiếu chéo nhau qua gói `xr`:

- `main.tex` đọc `appendix.aux` để biết số của các mục trong appendix (A.2, D.6, ...);
- `appendix.tex` đọc `main.aux` để biết số của các mục trong main (§3.4, Hình 2, Bảng 1, ...).

Overleaf không tự sinh `appendix.aux` (nó biên dịch với `-jobname=output`), nên trước giờ phải upload aux bằng tay. Sau khi làm theo file này, **mỗi file tự biên dịch độc lập, không cần aux nào**.

Có **38 chỗ** cần thay: 29 chỗ trong `main.tex` (22 label khác nhau) và 9 chỗ trong `appendix.tex` (5 label). Tất cả đều là lệnh `\ref{...}`; không có `\eqref` hay `\pageref` trỏ sang file kia.

> **Đã kiểm tra:** mình áp đúng các thay đổi dưới đây lên bản sao của bài (bản đầy đủ: bản zip thứ ba cộng Bước 7, phần Cost và ba chỗ `\rev` của R1), xoá hết file aux, rồi biên dịch từng file riêng. Kết quả: không còn ref nào undefined, và chữ trong PDF **giống hệt từng ký tự** bản dùng `xr`.

---

## 0. Làm trước: hoàn tất các sửa đổi còn thiếu

Các số dưới đây là số của **bản cuối**. Phải làm xong **Bước 7 (Table D.1 ở mục D.6)** trước, vì Bước 7 đẩy bảng conditioning từ D.1 thành **D.2**. Nếu chưa làm Bước 7 mà thay số trước, chỗ `tab:conditioning-ci` sẽ ghi sai.

Đoạn Cost (`\paragraph{Cost.}`) không làm đổi số mục hay số bảng nào.

## 1. Cách làm trên Overleaf

1. Mở file, nhấn **Ctrl+F**, rồi bấm mũi tên mở ô **Replace**.
2. Tìm chính xác chuỗi ở cột "Tìm", ví dụ `\ref{appendix:corpus}`, và thay bằng chuỗi ở cột "Thay bằng", ví dụ `A.1`. Bấm **Replace all**.
3. Kiểm tra số chỗ được thay khớp với cột "Số chỗ".

Chữ đứng trước giữ nguyên. Ví dụ `appendix~\ref{appendix:corpus}` thành `appendix~A.1`, `\S\ref{sec:persona}` thành `\S3.4`, `table~\ref{tab:effect}` thành `table~1`.

> ⚠️ Chỉ thay trong **đúng file** của bảng. Mỗi label trong bảng main chỉ xuất hiện ở main, và ngược lại; các `\ref{...}` trỏ tới label trong **chính file đó** (ví dụ `\ref{fig:egtllm}` trong main) thì **giữ nguyên**, vì LaTeX tự đánh số được.

---

## 2. `main.tex`

### 2a. Bỏ gói xr

Xoá hai dòng này ở phần preamble (ngay dưới dòng `hyperref`):

```latex
\usepackage{xr}
\externaldocument[][nocite]{appendix}
```

### 2b. Thay 29 tham chiếu sang appendix

| # | Tìm (Replace all) | Thay bằng | Là gì | Số chỗ |
|---|---|---|---|---|
| 1 | `\ref{appendix:conditioning_ci}` | `D.7` | D.7 Uncertainty in the conditioning contrasts | 1 |
| 2 | `\ref{appendix:corpus}` | `A.1` | A.1 Scope of the corpus and the completeness guard | 2 |
| 3 | `\ref{appendix:payoff_scaling}` | `A.3` | A.3 The scaled payoff matrix | 1 |
| 4 | `\ref{appendix:endpoints}` | `A.2` | A.2 Endpoints, seeds, parsing and fallback | 3 |
| 5 | `\ref{appendix:templates}` | `A.4` | A.4 Prompt templates and a framing artefact | 1 |
| 6 | `\ref{appendix:robust_parsing}` | `D.4` | D.4 Parsing, fallback | 1 |
| 7 | `\ref{appendix:classifier}` | `A.6` | A.6 Classifier training and validation | 1 |
| 8 | `\ref{appendix:robust_ladders}` | `D.6` | D.6 Matched scale ladders | 1 |
| 9 | `\ref{appendix:variance}` | `B.3` | B.3 Variance decomposition | 3 |
| 10 | `\ref{appendix:repro}` | `E` | E Reproduction | 1 |
| 11 | `\ref{tab:S-language}` | `B.3` | Bảng B.3 (cooperation theo model, ngôn ngữ, scale) | 2 |
| 12 | `\ref{appendix:pooling}` | `B.6` | B.6 Why the model curves cancel under pooling | 1 |
| 13 | `\ref{tab:S-variance}` | `B.4` | Bảng B.4 (variance decomposition, 5 model) | 1 |
| 14 | `\ref{appendix:robustness}` | `D` | D Robustness | 1 |
| 15 | `\ref{appendix:persona_full}` | `B.4` | B.4 Persona, in full | 1 |
| 16 | `\ref{appendix:readout_full}` | `B.5` | B.5 The strategy read-out, in full | 2 |
| 17 | `\ref{appendix:egt_simplex}` | `C.2` | C.2 Deterministic dynamics and simplex orbits | 1 |
| 18 | `\ref{fig:app_simplex}` | `C.1` | Hình C.1 (replicator flow trên simplex) | 1 |
| 19 | `\ref{appendix:egt}` | `C` | C The evolutionary baseline | 1 |
| 20 | `\ref{tab:conditioning-ci}` | `D.2` | Bảng D.2 (bảng conditioning) | 1 |
| 21 | `\ref{appendix:discussion}` | `F` | F Supplementary discussion | 1 |
| 22 | `\ref{appendix:verbal}` | `F.3` | F.3 A verbal baseline | 1 |

### 2c. Danh sách từng chỗ, để đối chiếu

Số dòng là của bản cuối, lấy gần đúng; nên tìm theo chữ, không theo số dòng.

| Dòng | Trước | Sau |
|---|---|---|
| ~170 | `ng (Section \ref{sec:conditioning}, appendix~\ref{appendix:conditioning_ci}).` | `ng (Section \ref{sec:conditioning}, appendix~D.7).` |
| ~181 | `ce is enforced rather than assumed (appendix~\ref{appendix:corpus}).` | `ce is enforced rather than assumed (appendix~A.1).` |
| ~211 | `model comprehension of the dilemma (appendix~\ref{appendix:corpus}).` | `model comprehension of the dilemma (appendix~A.1).` |
| ~237 | `x for all ten scales is provided in appendix~\ref{appendix:payoff_scaling}.` | `x for all ten scales is provided in appendix~A.3.` |
| ~254 | `eir model identifiers are listed in appendix~\ref{appendix:endpoints}.` | `eir model identifiers are listed in appendix~A.2.` |
| ~258 | `the same requested sampling seeds (appendix~\ref{appendix:endpoints}). Decisions are elicited as f` | `the same requested sampling seeds (appendix~A.2). Decisions are elicited as f` |
| ~258 | `st of the collection is reported in appendix~\ref{appendix:endpoints}.}` | `st of the collection is reported in appendix~A.2.}` |
| ~256 | `rather than five renderings of one (appendix~\ref{appendix:templates}). Each agent also receives on` | `rather than five renderings of one (appendix~A.4). Each agent also receives on` |
| ~258 | `he other models were not retained} (appendix~\ref{appendix:robust_parsing}). \rev{The cost of the collec` | `he other models were not retained} (appendix~D.4). \rev{The cost of the collec` |
| ~264 | `held-out execution-noise condition (appendix~\ref{appendix:classifier}). It ranks only within the se` | `held-out execution-noise condition (appendix~A.6). It ranks only within the se` |
| ~268 | `rm a matched ladder, 200 per model. Appendix~\ref{appendix:robust_ladders} repeats the headline inferenc` | `rm a matched ladder, 200 per model. Appendix~D.6 repeats the headline inferenc` |
| ~270 | `e supplementary decomposition is in appendix~\ref{appendix:variance}. The persona result contrasts` | `e supplementary decomposition is in appendix~B.3. The persona result contrasts` |
| ~385 | `e game (table~\ref{tab:S-variance}, appendix~\ref{appendix:variance}). Model identity and the mode` | `e game (table~\ref{tab:S-variance}, appendix~B.3). Model identity and the mode` |
| ~387 | `remaining model continues to move (appendix~\ref{appendix:variance}). A complementary dyad-level` | `remaining model continues to move (appendix~B.3). A complementary dyad-level` |
| ~270 | `orded with the archived pipeline in appendix~\ref{appendix:repro}.` | `orded with the archived pipeline in appendix~E.` |
| ~364 | `ges appear in electronic supplementary table~\ref{tab:S-language}.` | `ges appear in electronic supplementary table~B.3.` |
| ~368 | `nguage cells (electronic supplementary table~\ref{tab:S-language}). Figure~\ref{fig:language}b` | `nguage cells (electronic supplementary table~B.3). Figure~\ref{fig:language}b` |
| ~383 | `xture over distinct policy regimes (appendix~\ref{appendix:pooling}). Across the corpus, uncondit` | `xture over distinct policy regimes (appendix~B.6). Across the corpus, uncondit` |
| ~385 | `odel individuality dominates the game (table~\ref{tab:S-variance}, appendix~\ref{appendix:varia` | `odel individuality dominates the game (table~B.4, appendix~\ref{appendix:varia` |
| ~387 | `including the subset without Grok (appendix~\ref{appendix:robustness}). The two analyses are theref` | `including the subset without Grok (appendix~D). The two analyses are theref` |
| ~397 | `erification across payoff contexts (appendix~\ref{appendix:persona_full}).` | `erification across payoff contexts (appendix~B.4).` |
| ~408 | `cleanly (table~\ref{tab:strategy}, appendix~\ref{appendix:readout_full}).` | `cleanly (table~\ref{tab:strategy}, appendix~B.5).` |
| ~416 | `atched play (figure~\ref{fig:mix}c, appendix~\ref{appendix:readout_full}). When models appear to incre` | `atched play (figure~\ref{fig:mix}c, appendix~B.5). When models appear to incre` |
| ~429 | `d all rest points remain invariant (appendix~\ref{appendix:egt_simplex}, figure~\ref{fig:app_simplex}` | `d all rest points remain invariant (appendix~C.2, figure~\ref{fig:app_simplex}` |
| ~429 | `(appendix~\ref{appendix:egt_simplex}, figure~\ref{fig:app_simplex}).` | `(appendix~\ref{appendix:egt_simplex}, figure~C.1).` |
| ~431 | `tage of defection over cooperation (appendix~\ref{appendix:egt}). \rev{With $\beta$ fixed, p` | `tage of defection over cooperation (appendix~C). \rev{With $\beta$ fixed, p` |
| ~454 | `five models (electronic supplementary table~\ref{tab:conditioning-ci}). Thus the headline compariso` | `five models (electronic supplementary table~D.2). Thus the headline compariso` |
| ~494 | `n stake response learned from text. Appendix~\ref{appendix:discussion} gives the evidence for each a` | `n stake response learned from text. Appendix~F gives the evidence for each a` |
| ~501 | `ayoffs themselves change behaviour (appendix~\ref{appendix:verbal}).}` | `ayoffs themselves change behaviour (appendix~F.3).}` |

---

## 3. `appendix.tex`

### 3a. Bỏ gói xr

Xoá bốn dòng này (hai dòng chú thích và hai dòng lệnh, ngay dưới `\hypersetup{...}`):

```latex
% Pull in the main text's labels so that \ref{} to its sections, figures and
% tables resolve.  [nocite] keeps main's citation numbers out of this file.
\usepackage{xr}
\externaldocument[][nocite]{main}
```

### 3b. Thay 9 tham chiếu về main

| # | Tìm (Replace all) | Thay bằng | Là gì | Số chỗ |
|---|---|---|---|---|
| 1 | `\ref{sec:protocol}` | `2.3` | §2.3 Models, prompt languages, personas and protocol | 1 |
| 2 | `\ref{sec:persona}` | `3.4` | §3.4 Payoff scale modulates persona | 3 |
| 3 | `\ref{fig:landscape}` | `2` | Hình 2 | 1 |
| 4 | `\ref{fig:language}` | `3` | Hình 3 | 1 |
| 5 | `\ref{tab:effect}` | `1` | Bảng 1 | 3 |

### 3c. Danh sách từng chỗ, để đối chiếu

| Dòng | Trước | Sau |
|---|---|---|
| ~99 | `the distinction the main text draws on in \S\ref{sec:protocol}: an undated identifier can in` | `the distinction the main text draws on in \S2.3: an undated identifier can in` |
| ~132 | `he sub-unit boundary the persona result in \S\ref{sec:persona} is organised around, and it i` | `he sub-unit boundary the persona result in \S3.4 is organised around, and it i` |
| ~1014 | `kes in Claude and Gemini (main text, Section~\ref{sec:persona}). Removing the narrative or t` | `kes in Claude and Gemini (main text, Section~3.4). Removing the narrative or t` |
| ~1018 | `struction at every scale (main text, Section~\ref{sec:persona}), which points to such differ` | `struction at every scale (main text, Section~3.4), which points to such differ` |
| ~239 | `e model-by-scale estimates underlying figure~\ref{fig:landscape}c.` | `e model-by-scale estimates underlying figure~2c.` |
| ~243 | `y-language-by-scale estimates used in figure~\ref{fig:language}.` | `y-language-by-scale estimates used in figure~3.` |
| ~249 | `eparts from the invariance on its own (table~\ref{tab:effect}). The pooled main effect is t` | `eparts from the invariance on its own (table~1). The pooled main effect is t` |
| ~919 | `create one, so the smallest ranges in table~\ref{tab:effect} are the ones such a correlati` | `create one, so the smallest ranges in table~1 are the ones such a correlati` |
| ~941 | `and the round-level logistic model of table~\ref{tab:effect} with standard errors clustere` | `and the round-level logistic model of table~1 with standard errors clustere` |

---

## 4. Sau khi thay xong

1. **Xoá `main.aux` và `appendix.aux`** khỏi project Overleaf. Chúng không còn được dùng.
2. Recompile `main.tex`, rồi `appendix.tex`. Kiểm tra:
   - không còn `??` ở đâu;
   - main §2.3 ghi "appendix D.4" và "appendix A.2"; §2.5 ghi "Appendix D.6"; §3.7 ghi "table D.2";
   - appendix A.2 ghi "\S2.3"; D.4 và D.6 ghi "table 1".
3. Tìm `\ref{appendix:` trong `main.tex`: kết quả phải là **0**. Tìm `\ref{sec:`, `\ref{fig:landscape}`, `\ref{fig:language}`, `\ref{tab:effect}` trong `appendix.tex`: cũng phải là **0**.

## 5. Đánh đổi cần biết

- **Số giờ là chữ cố định.** Nếu sau này thêm, xoá hay đổi chỗ một mục, bảng hay hình ở file này, các số tương ứng ở **file kia** không tự cập nhật. Khi đó phải sửa tay theo bảng ở mục 2b và 3b.
  - Ví dụ: thêm một bảng mới trước Bảng D.2 trong appendix thì phải đổi `table~D.2` trong main thành `table~D.3`.
- Trước đây các chỗ này cũng không phải link bấm được (gói `xr` không tạo hyperlink), nên bỏ `xr` **không mất link nào**.
- Trích dẫn tài liệu không bị ảnh hưởng: mỗi file vẫn có bibliography riêng chạy bằng bibtex như cũ.
- Khi đổi giữa bản highlight và bản sạch (`\highlightrevisionsfalse`), số không đổi, nên không phải sửa gì thêm.
