# Kế hoạch Ablation Study — Khung đánh giá 1 (retriever) + Khung 3 (system)

> **Ngày**: 2026-07-05 · **Branch**: feature/validate_old_data
> **Nguồn yêu cầu**: `docs/note_danh_gia.md` §"Đánh giá 4 lớp" (#1 retriever, #3 ablation)
> **Trạng thái**: Khung 1 = **SẴN SÀNG CHẠY** — full-FT (−PEFT) đã implement (USE_LORA=0) +
> smoke-test CPU (2026-07-05); chỉ cần thuê server + chạy `.agents/scripts/run_khung1_ablation.sh`.
> Khung 3 = plan (cần dựng downstream + MedRAG trước khi chạy).

---

## 0. Bối cảnh & nguyên tắc

- **Khung 1** đo ở tầng **retriever** (embedding), metric Hit@1/Hit@10/MRR trên bộ financial
  (val=50, test=100), giao thức paper-protocol + LlamaIndex — **cùng pipeline `eval_paper_protocol.py`** đã dùng cho ε-sweep.
- **Khung 3** đo ở tầng **hệ thống end-to-end**, metric **accuracy trắc nghiệm** trên MedRAG/MIRAGE
  (không quan tâm retriever, chỉ đúng/sai đáp án) — cùng chuẩn với C-FedRAG, RAG² để so sánh.
- **Insight then chốt (từ ε-sweep)**: dưới DP, AdamW-invariance khiến trọng số gần như bất động
  ⇒ **mọi ablation trên NHÁNH DP đều ≈ pretrained (62/62)**. Tín hiệu ablation thật nằm ở
  **nhánh Non-DP** (nơi model thực sự học). Vì vậy ablation Khung 1 phải chạy **song song 2 nhánh
  DP/Non-DP** để cho thấy "DP nuốt chửng ảnh hưởng của recipe" — đó là một kết quả ablation có giá trị.

---

## 1. KHUNG 1 — Ablation retriever

### 1.1 Ma trận ablation (one-factor-at-a-time từ recipe đầy đủ)

Recipe đầy đủ = **DP(ε=20) + PEFT(LoRA) + InfoNCE + KD-GLE**. Bỏ từng thành phần:

| # | Cấu hình | DP | PEFT | KD-GLE | Val/Test Hit@1 | Trạng thái |
|---|---|:--:|:--:|:--:|---|---|
| 0 | Pretrained (không train) | – | – | – | 62 / 62 | ✅ có |
| A | **Full** (DP+LoRA+KD) | ✓ | LoRA | ✓ | 62 / 62 | ✅ có (ε=20) |
| B | − DP | ✗ | LoRA | ✓ | 58 / 56 | ✅ có (Run 10) |
| C | − KD-GLE (kd_weight=0) | ✓ | LoRA | ✗ | ? | ⏳ 1 run |
| D | − PEFT → **full fine-tune** | ✓ | full | ✓ | ? | ⚠️ code + 1 run |
| C' | Non-DP − KD-GLE | ✗ | LoRA | ✗ | ? | ⏳ 1 run |
| D' | Non-DP full fine-tune | ✗ | full | ✓ | ? | ⚠️ code + 1 run |
| — | **DP-strength** (ε=1/20/50/100) | ✓ | LoRA | ✓ | 62 toàn dải | ✅ **ε-sweep xong** |

→ **Cần chạy mới**: C, C' (toggle `kd_weight=0`, rẻ) + D, D' (full-FT, cần sửa code).
Tùy chọn thêm: **qLoRA** thay LoRA (`USE_QLORA=1`) để ablation biến thể PEFT.

### 1.2 Vấn đề "− PEFT" (cell D, D') — cần sửa code

`config.get_model()` (config.py:48-93) **luôn** `get_peft_model(base, lora_config)` — không có nhánh
full fine-tune. Để ablate PEFT cần:
1. Thêm cờ `USE_LORA=0` → trả về BGE-base với `requires_grad=True` toàn bộ (bỏ freeze + bỏ LoRA wrap).
2. Sửa `fedrag_lora.py`: `_is_lora_key`/`_lora_state_only`/`aggregate`/checkpoint-save đang giả định
   **LoRA-only state** → cần path lưu/tổng hợp full state khi không có LoRA.
3. **Cảnh báo compute**: per-sample DP-SGD trên **109M params** (thay vì 295K) → gradient per-sample
   cho toàn model → **nặng RAM/VRAM & chậm hơn nhiều**. Batch có thể phải giảm. Trên 4090 24GB
   khả thi cho BGE-base nhưng chậm — ước lượng ~3-4× thời gian mỗi run.

> **Ý nghĩa khoa học của D**: paper gốc dùng **full fine-tune** (không PEFT). Cell D/D' chính là
> tái lập recipe của paper (± DP) → cho thấy PEFT đánh đổi bao nhiêu so với full-FT. Đáng làm.

### 1.3 Lệnh chạy (toggle-only, cell C & C') — sẵn sàng

```bash
# trên server, trong FedE/, đã setup như ε-sweep:
# C  : DP + LoRA, KD off
DP_ENABLED=1 KD_WEIGHT=0 TARGET_EPS=20 python -X utf8 -u main_lora.py   # (cần thêm env KD_WEIGHT)
# C' : nonDP + LoRA, KD off
DP_ENABLED=0 KD_WEIGHT=0 python -X utf8 -u main_lora.py
# rồi eval như cũ:
python eval_paper_protocol.py --checkpoint <ckpt> --use-llama-index --name <tag> --split val/test
```
⚠️ `main_lora.py` hiện hard-code `DEFAULT_KD_WEIGHT = 1.0` (dòng 34) — cần thêm đọc env `KD_WEIGHT`
(sửa 1 dòng) để toggle được. Nhỏ.

### 1.4 Chi phí ước lượng (server Ryzen 5 3600 kiểu đã dùng, ~$0.40/h)

| Nhóm run | Số run | Sửa code | Thời gian | Chi phí |
|---|---|---|---|---|
| C, C' (kd_weight=0) | 2 | 1 dòng env | ~4h | ~$1.6 |
| qLoRA (tùy chọn) | 1-2 | không | ~2-4h | ~$1-1.6 |
| D, D' (full-FT) | 2 | ~30-50 dòng | ~8h | ~$3.2 |
| **Tối thiểu (C,C')** | **2** | 1 dòng | **~4h** | **~$1.6** |

---

## 2. KHUNG 3 — Ablation system-level (downstream, MedRAG)

### 2.1 Ma trận ablation (bỏ từng thành phần kiến trúc)

Metric: **accuracy trắc nghiệm** trên MedRAG/MIRAGE (đúng/sai đáp án), LLM = llama3-8b-instruct
(cùng chuẩn C-FedRAG / RAG²).

| # | Cấu hình | FedRAG-down | RagRouter | DP-embed | PEFT-embed |
|---|---|:--:|:--:|:--:|:--:|
| S0 | **Full system** | ✓ | ✓ | ✓ | ✓ |
| S1 | − FedRAG-downstream (RAG tập trung, 1 corpus) | ✗ | ✓ | ✓ | ✓ |
| S2 | − RagRouter/MoE (1 retriever, không route) | ✓ | ✗ | ✓ | ✓ |
| S3 | − DP (embed non-DP) | ✓ | ✓ | ✗ | ✓ |
| S4 | − PEFT (embed full-FT) | ✓ | ✓ | ✓ | ✗ |
| Sb | Baseline: BGE pretrained + LLM (không FL/DP/PEFT) | ✗ | ✗ | ✗ | ✗ |

### 2.2 Điều kiện tiên quyết (chưa sẵn sàng — đây là phần nặng)

1. **Downstream `finsaferag` chạy end-to-end**: Flower server/client + FastAPI + retriever + LLM generation.
   Hiện có code nhưng chưa có pipeline eval tự động trên benchmark.
2. **Bộ MedRAG/MIRAGE**: tải + index corpus y tế (theo note Excel: ~**600GB** sau embed) → cần disk lớn + thời gian.
3. **LLM generation**: llama3-8b-instruct (VRAM ~16GB fp16, chạy được trên 4090 24GB hoặc A100).
4. **Embed checkpoints**: DP / non-DP / full-FT (một phần từ Khung 1) — nhưng **embed hiện train trên
   financial**, còn MedRAG là **y tế** → cần train lại embed trên dữ liệu y tế (bộ 5-client medical
   trong Excel Khung 2: WEBMD/COVID/MEDQUAD, 59K mẫu) hoặc dùng MedCPT.
5. ⚠️ **"RagRouter (MoE)" CHƯA TỒN TẠI**: code downstream hiện là `api/router.py` = **QueryRouter phân
   loại domain theo keyword**, không phải MoE học được. Muốn ablate S2 cho đúng nghĩa "MoE" thì **phải
   xây RagRouter MoE trước**, hoặc định nghĩa lại S2 = "bỏ định tuyến, broadcast tất cả" (khả thi ngay
   với code hiện tại).

### 2.3 Lộ trình đề xuất cho Khung 3 (theo pha)

- **Pha 0 — Quyết định phạm vi**: chốt (a) benchmark = MIRAGE hay MedRAG toolkit; (b) LLM; (c)
  embed y tế = train lại hay dùng MedCPT/pretrained; (d) S2 = MoE thật hay "no-routing".
- **Pha 1 — Dựng harness eval end-to-end** trên `finsaferag`: input câu trắc nghiệm → RAG → LLM →
  parse đáp án → accuracy. Chạy được 1 cấu hình (S0 hoặc Sb) trên 1 subset nhỏ (~100 câu) để verify.
- **Pha 2 — Chuẩn bị data**: tải MIRAGE, index corpus (chú ý disk 600GB), cache embed.
- **Pha 3 — Chạy ablation S0-S4 + Sb**: mỗi cấu hình 1 lần trên full test, ghi accuracy + so bảng
  với C-FedRAG/RAG².
- **Pha 4 — Phân tích**: đóng góp từng lớp; đặc biệt kỳ vọng **−DP không giảm accuracy** (khớp
  finding Khung 1) → củng cố "DP miễn phí".

### 2.4 Chi phí ước lượng Khung 3 (thô)

Rất phụ thuộc quy mô. Nếu train lại embed y tế + index 600GB + chạy 6 cấu hình × ~vài nghìn câu qua
LLM 8B: **nhiều ngày GPU + hàng chục $** (cần A100/40GB cho thoải mái, hoặc 4090 với subset nhỏ). Nên
**bắt đầu bằng subset MIRAGE nhỏ** để kiểm thử harness trước khi chạy full.

---

## 3. Thứ tự thực hiện đề xuất

1. **(Ngay, rẻ)** Khung 1: sửa 1 dòng `KD_WEIGHT` env → thuê server → chạy C, C' (kd_weight=0) +
   tùy chọn qLoRA → điền bảng ablation retriever vào Excel/báo cáo. **~$1.6, ~4h.**
2. **(Quyết định)** Có làm cell D/D' (full-FT) không? Nếu có → tôi sửa `config.get_model` +
   `fedrag_lora` cho path no-LoRA, test trên CPU trước, rồi chạy. **+~$3.2, +code.**
3. **(Song song)** Khung 3: chốt Pha 0 → tôi dựng harness eval end-to-end (Pha 1) trên subset nhỏ.
   Đây là phần lớn nhất, làm dần.

---

## 4. Câu hỏi mở / quyết định cần chốt

- [x] Khung 1: **full fine-tune (−PEFT) = CÓ** → đã implement `USE_LORA=0` (config/main_lora/fedrag_lora) +
  smoke-test CPU. Cells D/D' nằm trong `run_khung1_ablation.sh`.
- [ ] Khung 3: benchmark **MIRAGE hay MedRAG toolkit**? LLM nào? Embed y tế **train lại hay MedCPT**?
- [ ] Khung 3: **S2** = xây **RagRouter MoE** thật, hay định nghĩa lại = "no-routing / broadcast"?
- [ ] Ngân sách/thời gian trần cho toàn bộ ablation?
```
