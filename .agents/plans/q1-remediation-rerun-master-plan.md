# KẾ HOẠCH KHẮC PHỤC & CHẠY LẠI — FedE4RAG theo chuẩn Q1

> Nguồn: `docs/Bao_cao_tham_dinh_Q1_FedE4RAG_2026-07-18.docx` (điểm 31/100, Reject & Resubmit).
> Ngày lập: 2026-07-18. Branch làm việc: `feature/validate_old_data` → tạo branch mới `fix/q1-remediation`.
> **Cập nhật 2026-07-22:** audit code + data hoàn tất — xem mục "KẾT QUẢ AUDIT SOURCE CODE" bên dưới. Cả 4 P0 xác nhận đúng; A2/A3/C1/0 đã được điều chỉnh theo bằng chứng mới (168-page excision).
>
> **4 lỗi P0 phải sửa trước mọi claim khoa học:**
> - P0.1 Oracle/qrel injection: evaluator append golden passages vào index; 0/62 val + 0/127 test qrels tồn tại tự nhiên trong corpus phát hành.
> - P0.2 Metric sai semantics: Hit@1 cắt golden thay vì cắt retrieved; MAP/IDCG tự chế; duplicate IDs không dedupe.
> - P0.3 DP accounting sai: noise mỗi local step nhưng accountant đếm mỗi round (50×50=2500 mechanisms, không phải 50); InfoNCE batch-coupled phá sensitivity per-example.
> - P0.4 CKKS threat model: secret key cùng process với server; claim "multi-key" sai.

---

## KẾT QUẢ AUDIT SOURCE CODE (2026-07-22) — bằng chứng & điều chỉnh

Đối chiếu từng phát hiện của báo cáo với code thực tế trên branch `feature/validate_old_data`. **Tất cả 4 P0 xác nhận đúng**, kèm 5 phát hiện mới làm thay đổi thiết kế Giai đoạn A/C.

### Xác nhận từng lỗi (file:line thực tế)

| Lỗi | Vị trí xác minh | Kết luận |
|---|---|---|
| P0.1 append refs | `RAGTest/data/loader.py:31-35`; `FedE/eval_paper_protocol.py:227-245` (char) + `302-319` (llama-index), default `append_refs=True` | ĐÚNG. Điểm cộng: flag `--no-append-refs` đã tồn tại → ablation B3 chạy được ngay |
| P0.2/E2 Hit cắt golden | `FedE/eval_paper_protocol.py:568` — `paper_hit(retrieved_ids, golden_ids[0:k])` | ĐÚNG |
| P0.2/E3/E4 metric tự chế | `RAGTest/eval/evaluate_TRT.py:170-171` (hit1/hit10 cắt golden), `230-239` (MAP sai công thức), `254-264` (IDCG tự chế, báo như metric ở `:175`), không dedupe retrieved | ĐÚNG. Thêm: mỗi page sinh nhiều chunk cùng `metadata['id']` → retrieved_ids chứa duplicate → precision/AP double-count |
| P0.3/E5 accountant | `fedrag_lora.py:97-102` calibrate σ với `num_steps=num_rounds`, `sample_rate=clients_per_round/n_clients` (=1.0); noise cộng **mỗi local step** `:529`; accountant `.step()` **mỗi round** `:268`. `main_lora.py:87,93`: rounds=50, num_steps=50 → 2500 mechanisms vs 50 được đếm | ĐÚNG chính xác |
| P0.3/E6 InfoNCE coupling | `fedrag_lora.py:464-471`: `ref_embs` encode cả batch (chứa reference của sample i) rồi dùng trong sim_row của MỌI sample khác → bỏ sample i làm đổi loss của các sample còn lại; docstring claim sensitivity preserved là SAI | ĐÚNG |
| D3 clip-after-noise | `fedrag_lora.py:532` — `clip_grad_norm_(params, 1.0)` SAU khi cộng noise | ĐÚNG |
| P0.4/E7 CKKS | `fedrag_ckks_primitives.py:41-77` — `_SECRET_CTX`/`_PUBLIC_CTX` module singleton cùng process; `fedrag_lora_ckks.py:81` server lấy public ctx từ CHÍNH singleton đó. Server không update plaintext (`:169-173`) nhưng chưa có luồng decrypt-eval end-to-end | ĐÚNG (assert `has_secret_key()==False` chỉ che ciphertext object, không che process) |
| E8 pooling | `flgo/benchmark/fedrag_classification/core.py:196,202,234,241` — `torch.mean(last_hidden_state, dim=1)` không mask | ĐÚNG |
| FedAvg uniform | `fedrag_lora.py:239-244` — mean đều, không weighted n_k/n | ĐÚNG |
| Partition IID | `main_lora.py:71` — `IIDPartitioner`, num_clients=5, hard-code rounds/steps | ĐÚNG |
| E10 test CKKS | `tests/test_ckks_correctness.py:41` — `sys.exit(0)` ở module level khi thiếu tenseal | ĐÚNG |
| A2 route 1 tái dùng | `eval_paper_faithful.py:139-162` — `iter_corpus`/`golden_pages` đọc (doc_name, evidence_page_num) như plan nói | ĐÚNG |

### 5 PHÁT HIỆN MỚI (thay đổi thiết kế plan)

**M1 — Cơ chế thật của "0/62, 0/127" là 168-page excision, KHÔNG phải khác namespace.**
Đo trực tiếp trên data (2026-07-22):
- `test_corpus.json`: 368 docs, 30.829 pages, index 0..54119 (thưa). `train_corpus.json`: 368 docs (CÙNG bộ), 23.123 pages, overlap index = 0.
- Union = 53.952/54.120 → khuyết đúng **168 index**. Số `reference_idx` duy nhất (val∪test) = **168**. Giao 168∩168 = **168/168 khớp tuyệt đối**.
- ⇒ Bộ dữ liệu gốc là MỘT collection 54.120 trang; nhà phát hành **cắt đúng 168 trang golden ra khỏi cả hai corpus** và chỉ ship text của chúng trong QA (`key_content.reference`). Vì thế loader gốc phải append refs lúc eval (chính là P0.1).
- Hệ quả: chuỗi mapping A2 cũ (doc+page → substring → fuzzy) **không thể đạt 100%**: đo được (doc,page) khớp trực tiếp chỉ 19/62 val, 37/127 test (~30%, và cả khi khớp thì CONTENT trang không chứa reference — pagination evidence khác hệ với corpus); substring sau normalize = **0/62, 0/127**; fuzzy containment ≥0.8 chỉ 7/62 (~11%).

**M2 — Reference là EXCERPT, không phải full page.** Median reference = 1.271 chars vs median page = 4.283 chars. Khôi phục 168 trang từ reference text tạo passage ngắn hơn hệ thống → phải khai báo length-bias trong paper; đường nâng cấp là trích lại full page từ raw PDF (10-K/10-Q public, kiểu FinanceBench).

**M3 — Toàn bộ 368 documents xuất hiện ở CẢ train_corpus lẫn test_corpus** (chia theo trang, không theo document). Gate A3 "0 parent_document_id trùng train↔test" theo định nghĩa cũ sẽ **fail vĩnh viễn by design**. Phải định nghĩa lại gate (xem A3 sửa đổi).

**M4 — Training data có 43 công ty, không phải 5.** `selected_data.json` (33.649 records) chứa 43 company (kèm typo trùng: `ACTIVISIONBLIZZARD`/`ACTIVSIONBLIZZARD`, `PFIZER`/`Pfizer`); test QA trải 24–30 công ty. "Natural company split" cho 5 clients phải là ánh xạ 43 công ty → 5 client (normalize tên trước), không phải 1 công ty = 1 client.

**M5 — Rủi ro môi trường: Python 3.14 trên Windows.** `ranx`, `pytrec_eval`, `rapidfuzz` chưa cài; ranx (numba) và pytrec_eval (C-ext) nhiều khả năng chưa có wheel cp314/win. → Giai đoạn 0 phải tạo venv Python 3.11/3.12 riêng cho eval tooling (hoặc chạy tooling trên WSL/Linux box), không dùng env 3.14 hiện tại.

### Điểm code TỐT có thể tái dùng (không phải viết mới từ đầu)

- `--no-append-refs` + JSON per-query output của `eval_paper_protocol.py` → B3 protocol-sensitivity chạy được ngay hôm nay.
- `fedrag_ckks_primitives.py` pattern serialize-without-secret-key đúng chuẩn; chỉ cần tách process + xóa singleton chung.
- `eval_paper_faithful.py:139-162` (iter_corpus/golden_pages) tái dùng cho build_qrels route xác thực chéo (doc,page).
- Per-sample clipping loop trong `_train_dp` viết sạch — vấn đề là accounting + loss coupling, không phải mechanics clip.

---

## Nguyên tắc xuyên suốt (không được vi phạm)

1. **Không vá metric trên corpus sai.** `RAGTest/` và `FedE/eval_paper_protocol.py` chuyển sang chế độ read-only/reproduction; output luôn gắn nhãn `P1_ORIGINAL` hoặc `P2_POOL`, không bao giờ vào bảng chính.
2. **Ba protocol tách biệt:**
   - **P1** — reproduction paper gốc (6.066 pages + appended qrels, metric gốc): chỉ để chứng minh tái tạo được.
   - **P2** — disclosed candidate pool (pool cố định = distractors ∪ qrels, công bố rõ): chỉ để so sánh tương đối.
   - **P3** — clean full corpus (30.829 pages + 168 trang golden restored-before-freeze = 30.997 passages, KHÔNG append lúc eval-time, qrel coverage 100%): duy nhất được dùng cho main claim. (Xem A2a — restore ≠ oracle injection.)
3. **Fail-fast:** coverage < 100% → dừng, không skip query âm thầm.
4. **Freeze bằng SHA-256:** mỗi run lưu git commit, command, seed, model revision, CUDA/PyTorch, hash corpus/qrels/split.
5. **Thứ tự bắt buộc:** Data (A) → Evaluator (B) → Training/FL (C) → DP (D) → CKKS (E) → Rerun matrix (F) → Statistics (G) → Downstream + paper (H). Không nhảy cóc: không chạy GPU khi A/B chưa qua gate.

---

## GIAI ĐOẠN 0 — Chuẩn bị (nửa ngày, local, không GPU)

```powershell
# 0.1 Branch mới
git checkout -b fix/q1-remediation

# 0.2 Freeze snapshot dữ liệu + commit
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "artifacts/manifests/$runId"
New-Item -ItemType Directory -Force $out | Out-Null
git rev-parse HEAD | Set-Content "$out/git_commit.txt" -Encoding utf8
git status --porcelain=v1 | Set-Content "$out/git_status.txt" -Encoding utf8
git diff | Out-File "$out/git_diff.txt" -Encoding utf8
Get-FileHash FedE/paper_test_data/test_corpus.json -Algorithm SHA256 | Format-List | Out-File "$out/data_hashes.txt" -Encoding utf8
Get-FileHash FedE/paper_test_data/val_qa_data_50.json -Algorithm SHA256 | Format-List | Out-File "$out/data_hashes.txt" -Append -Encoding utf8
Get-FileHash FedE/paper_test_data/test_qa_data_100.json -Algorithm SHA256 | Format-List | Out-File "$out/data_hashes.txt" -Append -Encoding utf8
Get-FileHash FedE/selected_data.json -Algorithm SHA256 | Format-List | Out-File "$out/data_hashes.txt" -Append -Encoding utf8

# 0.3 Cài thư viện eval chuẩn — LƯU Ý (M5): env hiện tại là Python 3.14/Windows,
# ranx (numba) + pytrec_eval (C-ext) nhiều khả năng KHÔNG có wheel cp314.
# Tạo venv 3.11/3.12 riêng cho eval tooling:
py -3.12 -m venv .venv-eval           # hoặc py -3.11; nếu không có → cài từ python.org / chạy trên WSL
.venv-eval\Scripts\pip install ranx pytrec_eval-terrier rapidfuzz
```

Bổ sung freeze (M1): hash thêm `FedE/train_corpus.json` vào `data_hashes.txt` — nó là nửa còn lại của collection 54.120 trang và là input cho audit leakage A3.

**Cấu trúc file mới sẽ tạo** (theo Bảng 33 của báo cáo — hiện CHƯA tồn tại trong repo):

| File mới | Trách nhiệm duy nhất |
|---|---|
| `FedE/tools/build_corpus_manifest.py` | test_corpus.json → corpus.jsonl với passage_id ổn định. KHÔNG nhận tham số QA/qrel. |
| `FedE/tools/build_qrels.py` | Ánh xạ annotation → passage_id tự nhiên. Xuất qrels.tsv + unresolved_qrels.tsv. |
| `FedE/tools/audit_protocol.py` | Coverage/duplicate/leakage/hash audit. Exit ≠ 0 nếu coverage < 100%. |
| `FedE/eval_clean.py` | Encode frozen corpus → retrieve → dedupe → TREC run + per-query JSON. Không import loader có append_refs. |
| `FedE/tests/test_ir_metrics.py` | Toy cases có expected values, đối chiếu ranx/pytrec_eval. |
| `FedE/configs/*.yaml` | Một nguồn config cho data/model/FL/DP/HE/eval. |
| `FedE/run_experiment.py` | Train/eval từ YAML, tạo run manifest, resume-safe. |
| `FedE/tools/aggregate_results.py` | Gộp seeds, bootstrap CI, paired tests, Holm. |

DoD Giai đoạn 0: branch mới + manifest hash + ranx/pytrec_eval import được.

---

## GIAI ĐOẠN A — Sửa dữ liệu & qrel (Tuần 1, local/CPU, KHÔNG GPU)

Đây là nút cổ chai quyết định: chừng nào 0/62 và 0/127 chưa thành 100% coverage thì mọi kết quả P3 đều vô nghĩa.

### A1. Canonical passage ID — `tools/build_corpus_manifest.py`

Schema mỗi dòng `corpus_v1.jsonl`:
```json
{"passage_id": "sha256('v1|'+norm(doc_name)+'|'+page_num+'|'+norm(text))[:24]",
 "parent_document_id": "sha256('v1|'+norm(doc_name))[:16]",
 "doc_name": "...", "page_num": 3, "company": "...",
 "text": "<raw text giữ nguyên>", "text_sha256": "...", "source_split": "test_corpus"}
```
Quy tắc:
- KHÔNG dùng index tăng dần (thứ tự JSON có thể đổi). Hash có version `v1|`.
- Normalize (NFC unicode, collapse whitespace, lowercase doc_name) TRƯỚC khi hash; giữ raw text trong manifest; ghi version hàm normalize vào header file.
- Nếu chunk một page: passage_id thêm `|chunk_start|chunk_end`; parent giữ (doc, page).
- Dedupe exact theo text_sha256 (log các dòng bị loại); near-duplicate chỉ audit bằng MinHash, không tự xóa.
- Chạy 2 lần → phải cho đúng cùng tập passage_id (test stable ID).

### A2. Khôi phục 168 trang golden + ánh xạ qrel — `tools/build_qrels.py` (SỬA ĐỔI 2026-07-22 theo M1/M2)

Nguyên nhân thật của 0/62, 0/127 (đã đo, xem mục AUDIT/M1): `reference_idx` **cùng namespace** với corpus `index`, nhưng đúng 168 trang golden đã bị nhà phát hành **cắt khỏi cả train_corpus lẫn test_corpus**; text của chúng chỉ còn trong `key_content.reference` của QA files. Chuỗi mapping cũ (doc+page → substring → fuzzy) đã đo được trần ~11–30% — **không dùng làm đường chính nữa**.

**Thiết kế mới — A2a (đường chính): restore-before-freeze.**
1. `build_corpus_manifest.py` nhận thêm input QA files ở **chế độ restore riêng biệt** (một lần, trước freeze, được log): với mỗi cặp `(reference, reference_idx)` + `(doc_name, evidence_page_num)` (align theo vị trí, validate độ dài 2 list bằng nhau), tạo passage mới:
   - `passage_id` theo đúng schema A1 (hash từ doc_name + page + text) — KHÔNG dùng reference_idx làm ID;
   - metadata: `source_split="qa_reference_restored"`, `original_index=reference_idx`, `restored=true`.
2. Corpus P3 v1 = 30.829 trang test_corpus + 168 trang restored = **30.997 passages**, freeze SHA-256 xong mới build index.
3. Qrel = ánh xạ trực tiếp `qid → passage_id(restored)` — coverage 100% **by construction**; các route (doc,page)/fuzzy cũ hạ xuống làm **cross-check** (ghi `match_method=restored+xcheck_page_ok|xcheck_page_mismatch`), không quyết định coverage.
4. Vẫn xuất `unresolved_qrels.tsv` cho mọi record có reference/evidence align lỗi (2 list lệch độ dài, reference rỗng, evidence_page_num vượt max trang của doc — đã thấy 1 val + 3 test) → adjudicate thủ công.

**Vì sao restore KHÔNG phải oracle injection như P0.1:** (i) làm MỘT lần lúc build corpus, trước freeze, độc lập model/score; (ii) 168 trang là trang hợp lệ của chính 368 documents (có provenance doc/page/index gốc); (iii) được khai báo rõ trong paper + manifest; (iv) index build từ corpus frozen, evaluator không bao giờ đọc QA để thêm document (gate Index independence A3 vẫn giữ nguyên). Khác biệt then chốt với P0.1: P0.1 append lúc **eval-time trong evaluator**, không khai báo, và corpus thay đổi theo QA split được chấm.

**Caveat bắt buộc khai báo (M2 — length bias):** reference là excerpt (median 1.271 chars) ngắn hơn page thường (median 4.283 chars) → 168 passage restored ngắn hơn hệ thống. Ghi rõ trong paper §limitations + đo ảnh hưởng: report phân phối độ dài passage retrieved vs golden.

**A2b (nâng cấp, tùy chọn — làm sau khi pipeline chạy):** trích lại **full page** cho 168 trang từ raw PDF gốc (10-K/10-Q public, bộ doc kiểu FinanceBench: PEPSICO_2022_10K, BOEING_2022_10K, …) → corpus P3 v2 loại bỏ length bias. Chỉ làm nếu còn thời gian; v1 restored-excerpt là đủ cho main claim nếu caveat được khai báo.

Output: `artifacts/data/qrels_val_v1.tsv`, `qrels_test_v1.tsv` (format TREC: `qid 0 passage_id rel`), `queries_{val,test}_v1.jsonl`, `unresolved_{val,test}.tsv`, `restored_pages_v1.jsonl` (168 dòng, có provenance đầy đủ).

### A3. Cổng nghiệm thu — `tools/audit_protocol.py` (exit ≠ 0 nếu fail)

| Kiểm tra | Ngưỡng qua | Nếu fail |
|---|---|---|
| Qrel coverage | 100% query có ≥1 relevant trong corpus | Dừng; sửa restore/annotation |
| Restore integrity | Đúng 168 passage `restored=true`; mỗi qid map đủ số qrel; 0 restored passage lọt vào bất kỳ training corpus/pairs nào | Sửa build_corpus_manifest |
| Index independence | Static test: eval_clean không import/mở QA file; build_corpus_manifest chỉ đọc QA trong restore-mode được log | Tách module |
| Page-level leakage (SỬA theo M3) | 0 `text_sha256` trùng giữa client-train data và test corpus/restored pages; đặc biệt: 0 record nào trong `selected_data.json` (33.649) có reference trùng/near-dup với 168 trang restored | Loại record khỏi train trước freeze |
| Parent overlap (SỬA theo M3) | KHÔNG còn là gate fail — 368 docs vốn xuất hiện cả train↔test corpus **by design của dataset gốc**. Đo và BÁO CÁO tỷ lệ; ghi vào paper §limitations; tùy chọn: robustness split document-disjoint ở appendix | Báo cáo, không dừng |
| Exact dup trong corpus | 0 text_sha256 trùng nội bộ corpus P3 | Dedupe trước freeze |
| Stable ID | Rebuild 2 lần → cùng passage_id set | Sửa ID derivation |
| Reachability | Dùng chính reference text làm query → passage restored tương ứng vào top-10 (oracle sanity, đặc biệt quan trọng vì restored là excerpt) | Kiểm tra tokenizer/chunking |
| Company normalize (M4) | Bảng map tên company chuẩn hóa (43 → canonical, gộp `ACTIVSIONBLIZZARD`, `Pfizer`/`PFIZER`) được commit | Sửa bảng map |

### Lệnh chạy Giai đoạn A (cập nhật theo A2a restore-mode)
```powershell
cd FedE
# Restore-mode: đọc QA MỘT LẦN để khôi phục 168 trang golden (log + provenance), rồi freeze
python -X utf8 tools/build_corpus_manifest.py --input paper_test_data/test_corpus.json `
    --restore-from-qa paper_test_data/val_qa_data_50.json paper_test_data/test_qa_data_100.json `
    --restored-out ../artifacts/data/restored_pages_v1.jsonl `
    --output ../artifacts/data/corpus_v1.jsonl
# build_qrels: map trực tiếp qid → restored passage_id; route (doc,page)/fuzzy chỉ cross-check
python -X utf8 tools/build_qrels.py --qa paper_test_data/val_qa_data_50.json  --corpus ../artifacts/data/corpus_v1.jsonl --output ../artifacts/data/qrels_val_v1.tsv  --unresolved ../artifacts/data/unresolved_val.tsv
python -X utf8 tools/build_qrels.py --qa paper_test_data/test_qa_data_100.json --corpus ../artifacts/data/corpus_v1.jsonl --output ../artifacts/data/qrels_test_v1.tsv --unresolved ../artifacts/data/unresolved_test.tsv
python -X utf8 tools/audit_protocol.py --corpus ../artifacts/data/corpus_v1.jsonl --qrels ../artifacts/data/qrels_val_v1.tsv --train-data selected_data.json --train-corpus train_corpus.json --require-coverage 1.0 --fail-on-page-leak
```

**DoD Tuần 1:** coverage 100% (hoặc danh sách loại query pre-registered), unresolved = 0, audit pass, hash freeze. **Không được sang Giai đoạn B/C nếu chưa đạt.**

---

## GIAI ĐOẠN B — Evaluator chuẩn (Tuần 2, local/CPU + 1 lần GPU nhẹ)

### B1. `eval_clean.py` — metric chuẩn IR

Sửa các lỗi E3/E4 (evaluate_TRT.py:163–200, 230–264) và E2 (eval_paper_protocol.py:548–570 lặp lại lỗi cắt golden):

- **Dedupe retrieved passage_id trước metric**; document-level: map sang parent rồi dedupe lần nữa.
- `Hit@k = 1 nếu set(retrieved[:k]) ∩ qrels ≠ ∅` — cắt **retrieved**, không cắt golden.
- `Recall@k = |relevant duy nhất được lấy| / |qrels|`; `MRR@k` = 1/rank của relevant đầu tiên trong top-k (0 nếu không có).
- `AP@k` theo quy ước TREC `map_cut` (chia cho TỔNG |qrels|, không phải min(|qrels|,k)) — bắt buộc để khớp ranx/pytrec_eval trong 1e-9 (đã verify 2026-07-22; hai quy ước trùng nhau khi k ≥ |qrels|, luôn đúng với k=10 và 1–3 qrels/query); `nDCG@k` với IDCG từ labels độc lập retrieved list. **Không báo IDCG như metric.**
- Set-EM cũ → đổi tên `Set-EM@|qrels|`, chỉ appendix.
- Primary endpoint khai báo trước: **MRR@10** (qrel binary) — mọi metric khác là secondary.
- Không skip query: nếu qrel thiếu → raise, đúng tinh thần fail-fast.
- Output: TREC run file + per-query JSON (để bootstrap ở Giai đoạn G) + manifest (hash corpus/qrels, model revision, k, seed).

### B2. `tests/test_ir_metrics.py` — toy suite bắt buộc

| Case | Input | Expected |
|---|---|---|
| Relevant rank 1 | run=[A,B,C], qrels={A} | Hit@1=1; MRR=1; AP=1; nDCG=1 |
| Relevant rank 3 | run=[X,Y,A], qrels={A} | Hit@1=0; Hit@3=1; MRR@10=1/3 |
| Hai qrel | run=[A,X,B], qrels={A,B} | Recall@1=0.5; Recall@3=1 |
| Duplicate | run=[A,A,B], qrels={A,B} | dedupe → [A,B], không double-count |
| Missing qrel ID | qrels={Z}∉corpus | evaluator **fail trước retrieval** |
| Empty qrel | qrels={} | schema validation fail |

Cross-check từng metric với `ranx` (hoặc `pytrec_eval`) trong CI — assert bằng nhau trong 1e-9.

Sửa luôn E10: `tests/test_ckks_correctness.py` thay `sys.exit(0)` bằng `pytest.importorskip('tenseal')` để pytest collection không vỡ.

### B3. Protocol sensitivity (chạy được ngay với source hiện có — nhãn reproduction only)

```powershell
cd FedE
# P1 reproduction (append refs) — 2 lệnh
python -X utf8 eval_paper_protocol.py --split val  --name p1_pretrained_val  --use-llama-index --query-expansion 0
python -X utf8 eval_paper_protocol.py --split test --name p1_pretrained_test --use-llama-index --query-expansion 0
# P1 no-append ablation — chứng minh oracle sensitivity
python -X utf8 eval_paper_protocol.py --split val  --name p1_no_append_val  --use-llama-index --query-expansion 0 --no-append-refs
python -X utf8 eval_paper_protocol.py --split test --name p1_no_append_test --use-llama-index --query-expansion 0 --no-append-refs
# P3 clean — LỆNH MỚI sau khi A xong
python -X utf8 eval_clean.py --corpus ../artifacts/data/corpus_v1.jsonl --qrels ../artifacts/data/qrels_val_v1.tsv --queries ../artifacts/data/queries_val_v1.jsonl --model BAAI/bge-base-en-v1.5 --top-k 100 --output ../artifacts/eval/p3_pretrained_val
```
Ghi chú: dùng `--query-expansion 0` cho main claim (QE dùng Phi-3 không deterministic — nếu muốn giữ QE phải khóa prompt/revision/decoding/seed và publish expanded queries).

**DoD Tuần 2:** toy tests pass + khớp ranx; P3 pretrained cho metric non-zero hợp lý (nếu = 0 → quay lại A, chẩn đoán mapping, KHÔNG train); bảng protocol-sensitivity P1-append vs P1-no-append vs P3.

---

## GIAI ĐOẠN C — Sửa training & FL (Tuần 3, GPU smoke/pilot ~$1–3 Vast.ai)

### C1. Sửa code train (các lỗi E8, E9 + Bảng 37)

| Hạng mục | File | Sửa | Test bắt buộc |
|---|---|---|---|
| Pooling | `FedE/core.py:192–203,229–241` | Mean đang tính trên toàn padded tensor → **masked mean** (hoặc CLS theo model card BGE); normalize sau pooling; khóa query instruction của BGE | Embedding không đổi khi pad thêm token; cosine self-match ≈ 1 |
| Loss | `flgo/algorithm/fedrag_lora.py` | Chốt một công thức: temperature InfoNCE + λ·MSE-KD; xóa comment "KL" nếu dùng MSE; λKD tune trên validation | Toy batch loss; gradient ≠ 0 cho LoRA-B; λ=0 tắt hẳn KD |
| FedAvg | `flgo/algorithm/*` | Uniform average → **weighted n_k/n** (đúng paper equation); giữ uniform như ablation | 2 client kích thước khác nhau → closed-form đúng |
| Partition | `main_lora.py` | `IIDPartitioner` → company split (chính): normalize 43 tên company (M4) → gán company→client cố định cho 5 clients (cân bằng theo số record, seed-fixed, commit bảng gán); Dirichlet α chỉ là stress test; `selected_data.json` thêm parent/company/split metadata | Page-level: 0 text hash leak giữa client-train và test/restored (gate A3 mới); bảng company→client reproducible |
| Config | mới `run_experiment.py` + `configs/*.yaml` | Bỏ hard-code rounds/seed/epsilon/path trong entrypoint; seed đầy đủ Python/NumPy/Torch/CUDA + deterministic flags | 2 smoke runs cùng seed → metric/checkpoint khớp trong tolerance |
| Selection | `run_experiment.py` | Best round theo validation metric khai báo trước; lưu best-val + last; KHÔNG đọc test trong early stopping | CI test phát hiện code đọc test metric |

### C2. Thứ tự baseline (mỗi bước phải ổn mới bước tiếp)

B0 pretrained frozen (xác nhận evaluator) → B1 centralized upper bound → B2 local-only per company → B3 FedAvg (full + LoRA) → B4 FedProx/SCAFFOLD (cùng budget tune) → B5 LoRA+KD (proposed). **Chỉ sau khi B0–B5 ổn mới bật DP, cuối cùng mới HE.**

### C3. Smoke → Pilot → Confirmatory

| Tầng | Config | Cổng qua |
|---|---|---|
| Smoke | 1 seed, 2 clients, 2–3 rounds, 100–500 samples, corpus nhỏ nhưng qrel hợp lệ | Pipeline hoàn tất; loss giảm; update ≠ 0; evaluator không skip |
| Pilot | 1 seed, full train + full P3 validation, 10–25 rounds | Chọn LR/LoRA rank/λKD/local steps/pooling; tuyệt đối không đụng test |
| Confirmatory | 5 seeds, config khóa | Để Giai đoạn F |

Lưu ý hạ tầng (từ kinh nghiệm Phase 7C): trước khi thuê máy Vast.ai chạy DP, kiểm tra CPU gate — `cpu MHz ≥ 2500` + microbench 10M-loop < 1.2s (tránh EPYC oversubscribed); dùng `.agents/scripts/complete_eps_sweep.sh` làm mẫu gated runner.

**DoD Tuần 3:** centralized (B1) phải cho retrieval gain > pretrained trên P3 validation — nếu objective không học được ở chế độ centralized thì FL/DP không có ý nghĩa; non-DP pipeline chạy ổn định, reproducible cùng seed.

---

## GIAI ĐOẠN D — Thiết kế lại DP (Tuần 4, GPU pilot)

### D1. Đường chính: **client-level DP** (khuyến nghị của báo cáo — Bảng 39)

Lý do: 5 clients, InfoNCE batch-coupled (E6) làm record-level sensitivity không chứng minh được; client-level tránh hoàn toàn vấn đề này.

Thiết kế (thay cho cơ chế hiện tại ở `fedrag_lora.py:90–102, 266–274, 449–533` — E5):
1. Adjacency: hai federated datasets khác nhau bởi toàn bộ dữ liệu một client. Khai báo threat model + released objects (noisy aggregate/checkpoints/metrics).
2. Client k train local (KHÔNG noise trong local steps) → `delta_k = theta_k − theta_global`.
3. Clip **một lần** toàn vector LoRA delta: `delta_bar = delta · min(1, C/‖delta‖₂)`.
4. Server average clipped deltas + **Gaussian noise một lần mỗi round**. Xóa local-step noise hiện tại — không được có cả hai khi accountant chỉ compose một mechanism.
5. Accountant: `q = clients_sampled/total_clients`, `T = số rounds`. Với 5 clients q=1 → không có amplification; nêu rõ hạn chế này trong paper, KHÔNG giả lập 20 clients bằng chia record.
6. Cross-check ε bằng **hai implementation độc lập** (RDP tự cài + Opacus accountant hoặc dp-accounting của Google); ghi α order tối ưu; δ biện minh theo protected population; báo ε spent tại best-val round và final round.

### D2. Record-level DP — chỉ giữ nếu cần claim này (không khuyến nghị cho bản đầu)

Nếu giữ: sampling rate = batch_size/N_client (không phải clients_per_round/num_clients); compose TOÀN BỘ local steps × rounds; đổi InfoNCE → pairwise loss với public/fixed negative bank (hoặc có proof sensitivity cho batch-coupled loss); adaptive clipping phải privatize hoặc dùng schedule cố định; privacy unit test đối chiếu Opacus trên lịch sampling nhỏ.

### D3. Controls bắt buộc trước ε-sweep (Bảng 40)

- `noise=0` cùng DP code path; `clip-only`; so với non-DP thuần → tách lỗi implementation / clipping bias / noise.
- **Adaptation diagnostics:** update norm theo round, ‖θ−θ₀‖, LoRA-B norm, ranking agreement với pretrained, update SNR. Phát hiện Phase 7 "DP ≡ pretrained bit-identical" nghĩa là adapter KHÔNG học — kết quả DP chỉ được gọi "utility retention" nếu adaptation thực sự ≠ 0. Bỏ mọi diễn giải "DP improves utility"/"privacy for free".
- Lưu ý: `clip_grad_norm_(..., 1.0)` SAU khi cộng noise (hiện tại) làm grad norm luôn = 1.0 và triệt tiêu khác biệt noise scale — xóa trong thiết kế mới.
- ε-sweep đề xuất: {1, 3, 8, 20}, δ cố định, cùng rounds/sampling/clip policy/seed matrix.

**DoD Tuần 4 (DP gate — Bảng 30):** adjacency/unit/N/sampling/clipping/δ được văn bản hóa; accountant khớp 2 implementation; số accountant.step == số mechanism release; noise=0 control ≈ non-DP.

---

## GIAI ĐOẠN E — CKKS đúng threat model (Tuần 4, song song với D, CPU là đủ)

Sửa E7 (`fedrag_ckks_primitives.py:41–77` — secret/public context singleton cùng process):

1. **Tách ≥ 2 process:** server process chỉ có public context (serialize context KHÔNG kèm secret key); key-holder/client service giữ secret key. Integration test: server process cố decrypt → PHẢI fail (negative test).
2. **Sửa terminology:** gọi đúng "shared-key CKKS aggregation" (mọi client cùng 1 secret key). Xóa comment "mỗi tenant một key vẫn homomorphic sum được" — sai với single-key CKKS; muốn multi-key phải dùng multi-key/threshold CKKS thực sự.
3. **Correctness:** so `plaintext_aggregate` vs `decrypt(HE_aggregate)` trên cùng clipped deltas; báo max abs error + relative L2 (< 1e-4); ảnh hưởng lên retrieval metric trong numerical tolerance.
4. **Security parameters công bố:** poly_modulus_degree, coeff modulus chain, scale, slots, multiplicative depth, ước lượng ≥128-bit (lattice-estimator).
5. **End-to-end:** checkpoint đánh giá cuối PHẢI tạo từ `decrypt(final_cipher)` của luồng thật (kiểm tra `fedrag_lora_ckks.py` ~169–173), không đánh giá plaintext checkpoint song song rồi suy ra.
6. **Systems benchmark:** encryption/client, serialize bytes, upload, aggregation, decryption, peak RAM, end-to-end round time (p50/p95); mô phỏng client dropout + ciphertext malformed.

**DoD (HE gate — Bảng 41):** 0 secret-key material trong server process/log/checkpoint; negative decrypt test pass; L2 error ≤ 1e-4; overhead table; final model đi qua decrypt path thật.

---

## GIAI ĐOẠN F — Ma trận chạy lại (Tuần 5–6, GPU chính — ước lượng $10–30 Vast.ai)

### F1. Seed & selection protocol
- Seed set khai báo trước: **{13, 29, 47, 71, 101}**, ghép cặp cùng seed giữa các method (paired).
- Hyperparameter search CHỈ trên validation (seed dev riêng nếu cần); sau khi chốt config → chạy đủ 5 confirmatory seeds.
- Freeze `config.yaml` + corpus/qrels hashes + checkpoint list → **locked test chạy đúng 1 lần/seed**, không quay lại chỉnh.

### F2. Ma trận runs tối thiểu (Bảng 42)

| Nhóm | Runs | Đầu ra |
|---|---|---|
| Protocol sensitivity | P1-append, P1-no-append, P3 × (pretrained + 1 checkpoint) | Chứng minh metric inflation |
| Core baselines | pretrained, centralized, local-only, FedAvg, FedProx, FedE-orig, Ours-LoRA × 5 seeds | Main utility table trên P3 |
| Ablation | LoRA/full-FT × KD on/off × masked pooling | Đóng góp từng thành phần |
| Privacy | No-DP + ε∈{1,3,8,20} + noise=0/clip-only × 5 paired seeds | Privacy-utility curve + adaptation diagnostics |
| Heterogeneity | company split + Dirichlet α∈{0.1,0.5,1.0}; clients 5/10/20 nếu data cho phép | Robustness/fairness per-client |
| HE systems | plain vs CKKS cùng deltas, ≥5 repetitions | Correctness + overhead |
| Downstream RAG | retriever top-k → generator cố định | Giai đoạn H |

### F3. Lệnh chạy (sau refactor)
```powershell
python -X utf8 run_experiment.py --config configs/p3/fedavg_lora.yaml    --seed 13 --run-id fedavg_lora_s13
python -X utf8 run_experiment.py --config configs/p3/client_dp_eps8.yaml --seed 13 --run-id clientdp_e8_s13
python -X utf8 eval_clean.py --run-id clientdp_e8_s13 --split val --locked-config artifacts/runs/clientdp_e8_s13/manifest.json
# ... lặp cho 5 seeds × mỗi config; test chỉ sau freeze
python -X utf8 tools/aggregate_results.py --runs "artifacts/runs/*" --primary mrr@10 --bootstrap 10000 --holm --output artifacts/tables
```

---

## GIAI ĐOẠN G — Thống kê (Tuần 7, local)

`tools/aggregate_results.py` đọc per-query artifacts (không chép tay số):
- Mean ± SD qua seeds; paired bootstrap theo query ≥ 10.000 resamples → CI 95%; hierarchical bootstrap seed×query; permutation test cho paired differences.
- Primary endpoint MRR@10 khai báo trước; Holm correction cho family các comparison chính; effect size + minimum practically important difference (không chỉ p-value).
- Slice: company, document length, qrel count, answer type, difficulty; báo worst-client/macro-client/micro-query; kèm failure examples.

**Bảng chẩn đoán kết quả bất thường (Bảng 43 — dùng khi đọc kết quả rerun):**

| Hiện tượng | Chẩn đoán ưu tiên | Hành động |
|---|---|---|
| P3 pretrained = 0 | Qrel mapping/namespace sai | DỪNG training; oracle reachability; inspect 20 queries |
| P1 cao, P3 thấp | Oracle pool làm bài dễ hơn | Trình bày protocol sensitivity; cải thiện trên P3; không che P1 |
| Non-DP < pretrained | Overfit 5 công ty / pooling / drift | Check centralized bound, LR, learning curve held-out company |
| DP ≈ pretrained | Adapter không học | Check LoRA-B norm, SNR, noise=0 control; KHÔNG claim DP tốt |
| CI chứa 0 | Bằng chứng chưa đủ | Không dùng ngôn ngữ superiority; báo inconclusive |

---

## GIAI ĐOẠN H — Downstream RAG & viết lại paper (Tuần 7–8)

- Cố định generator/prompt/decoding/context budget/k khi so retriever. Đo: answer correctness (exact/numeric tolerance), citation precision/recall, evidence recall, faithfulness; human subset blinded + inter-annotator agreement. **KHÔNG so PPL giữa model/tokenizer khác nhau.**
- Manuscript: main paper CHỈ dùng P3; P1/P2 vào appendix với nhãn rõ; bảng cũ ghi "invalidated for realistic retrieval claims" (không xóa dấu vết).
- Định vị contribution: *"communication-efficient federated retriever adaptation with formally accounted client-level DP and isolated encrypted adapter aggregation, validated on a frozen full-corpus enterprise non-IID benchmark."*
- Mẫu claim đúng/sai: theo Bảng 48 của báo cáo (ví dụ: thay "ε=1 DP with no utility loss" bằng "client-level adjacency, composition over T mechanisms, ε=… δ=…, ΔMRR@10 = … [CI]").

---

## Cổng quyết định nộp lại (Bảng 45 — cả 5 phải đạt)

1. P3 qrel coverage 100%.
2. Main comparison ≥ 5 seeds + paired CI.
3. DP guarantee khớp chính xác code/accountant (cross-checked).
4. Server không giữ secret key (integration test).
5. Final evaluated checkpoint đi qua encrypted aggregation/decryption thật.

Điểm tiềm năng sau khi hoàn tất: ~72–82/100 (theo ước lượng của reviewer).

## Checklist tổng (đánh dấu khi xong)

- [x] 0. Branch + freeze hash + cài ranx/pytrec_eval — DONE 2026-07-22 (branch `fix/q1-remediation`, manifest `artifacts/manifests/20260722-153303`, `.venv-eval` py3.14 với ranx 0.3.21 + pytrec_eval 0.5.10 + rapidfuzz — numba 0.66 đã hỗ trợ cp314, rủi ro M5 giải trừ)
- [x] A1. corpus_v1.jsonl với canonical passage_id, stable qua 2 lần build — DONE 2026-07-22: 30.496 passages (30.316 corpus + 180 restored; 522 exact-dup dropped có log; stable-ID diff=0)
- [x] A2. restore 168 trang golden + qrels coverage 100% by construction — DONE 2026-07-22: val 50/50 (70 qrels), test 100/100 (142 qrels), unresolved=0. Chi tiết mới: 189 cặp (ref,idx) → 168 idx duy nhất; 12 idx có 2 excerpt khác nhau (val vs test) → giữ cả hai làm golden variant; 9 dup-resolved. Length-bias caveat phải vào paper
- [x] A3. audit_protocol pass — DONE 2026-07-22: ALL HARD GATES PASS (coverage/restore-integrity/exact-dup/page-leak-0/stable-ID/index-independence). REPORT: parent overlap 354/354 by design; **96 selected_data records trích từ trang golden restored → PHẢI loại 96 records này khỏi training data ở Giai đoạn C**
- [x] B1. eval_clean.py + tools/ir_metrics.py (metric chuẩn + dedupe, AP@k theo TREC map_cut, fail-fast, TREC run + per-query JSON + manifest hash)
- [x] B2. test_ir_metrics.py 11/11 pass, khớp ranx + pytrec_eval trong 1e-9; test_ckks sys.exit(0) → pytest.importorskip (cả torch lẫn tenseal), thêm pytest entry point
- [~] B3. P1 append/no-append/P3 sensitivity table — SMOKE PASS 2026-07-22: eval_clean end-to-end trên index rút gọn (568 passages = 68 qrel + 500 filler, CPU) cho MRR@10=62.47, Hit@1/5/10=54/74/80, recall@10=76.33 → restored pages RETRIEVABLE, pipeline + fail-fast + manifest hoạt động. CÒN LẠI: full P3 pretrained (30.496 passages — GPU hoặc ~1-2h CPU) + 4 lệnh P1 append/no-append để hoàn thành bảng sensitivity
- [ ] C1. masked pooling + weighted FedAvg + company split + config/seed
- [ ] C2. B0→B5 baselines qua smoke/pilot; centralized học được
- [ ] D. client-level DP + accountant cross-check + noise=0/clip-only controls
- [ ] E. CKKS process isolation + negative decrypt test + params + overhead
- [ ] F. freeze config → 5 seeds × ma trận → locked test 1 lần
- [ ] G. aggregate CI/Holm/effect size tự động từ per-query artifacts
- [ ] H. downstream RAG + viết lại claim; P1/P2 vào appendix
