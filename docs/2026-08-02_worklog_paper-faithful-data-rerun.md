# Worklog 2026-08-02 — Paper-faithful data rerun

## 1. Mục tiêu trong ngày

Tạm dừng luồng thí nghiệm client-level DP đang thực hiện trên tập dữ liệu mở
rộng, chuyển sang một nhánh Git riêng và chuẩn bị chạy lại bằng đúng tập dữ
liệu huấn luyện được công bố cùng paper FedE4RAG.

Paper và nguồn dữ liệu:

- Paper: https://arxiv.org/abs/2504.19101
- Dataset: https://huggingface.co/datasets/DocAILab/FedE4RAG_Dataset
- Dataset revision đã đóng băng:
  `398304846743f184d36f2c35a3db58fa9be70a9d`

## 2. Trạng thái task cũ: client-level DP

Task DP cũ đã được dừng và lưu lại trước khi hủy Vast.ai server.

### Kết quả đã hoàn thành

- Client-level DP primitives và thuật toán riêng đã được tích hợp.
- Noise-zero control đã PASS:
  - FedAvg MRR@10: `25.52`
  - Client-DP noise0 MRR@10: `25.52`
  - Metric delta: `0`
  - Privacy releases/accountant steps: `0/0`
- Clip-only với `C=8` đã hoàn thành:
  - MRR@10: `25.52`
  - Có `5/15` client updates bị clip
  - Hệ số clip nhỏ nhất: `0.7059817`
- RDP accountant đã được sửa theo công thức chuyển đổi sử dụng bởi Opacus.
- Cross-check native accountant với Opacus 1.6 khớp tuyệt đối cho:
  - epsilon: `{1, 3, 8, 20}`
  - số vòng kiểm tra: `{3, 15}`
- DP epsilon 20:
  - Train smoke đã hoàn thành
  - Checkpoint và privacy report đã lưu
  - Validation bị dừng giữa quá trình encode corpus
  - Epsilon 8, 3 và 1 chưa chạy

### File phục hồi task DP

- Gói handoff:
  `artifacts/sda-client-dp-handoff-20260730.tar.gz`
- SHA256:
  `b870cc98f53b4f4a536af7c293af357710465f78b51a77d879821ef03d19edec`
- Resume note:
  `artifacts/training/d_client_dp_epsilon_smoke_seed13/RESUME_STATE.md`
- Resume runner:
  `scripts/vast/job-d-client-dp-epsilon-smoke-sweep.sh`

Task DP này hiện tạm dừng, không bị xóa và có thể tiếp tục sau.

## 3. Phát hiện về tập dữ liệu cũ

Pipeline trước đó dùng:

`FedE/selected_data_clean.json`

Thông tin:

- Bản gốc `selected_data.json`: `33,649` records
- Bản clean: `33,553` records
- Đã loại `96` records có nguy cơ leak golden pages
- Có `41` công ty
- Các công ty được cân bằng vào 5 client bằng company-LPT

Đây là dữ liệu cùng họ FedE4RAG nhưng không trùng cấu hình train trong paper.
Paper sử dụng đúng 5 công ty và 43,658 query-chunk pairs.

Trong tập cũ:

- Records thuộc đúng 5 công ty paper: `10,406`
- Records thuộc các công ty ngoài paper: `23,147`

Vì vậy kết quả cũ phải được mô tả là benchmark dẫn xuất/mở rộng, không phải
reproduction chính xác dataset split của paper.

## 4. Nhánh Git mới

Đã tạo và checkout nhánh:

`paper-faithful-data-rerun`

Các thay đổi và artifacts chưa commit của task trước được giữ nguyên, không
stash, reset hoặc xóa.

## 5. Dữ liệu chính thức đã tải

### Tập train đúng paper

Đường dẫn:

`artifacts/data/paper_faithful_source/FEDE4FIN/train_data/data_50000_random.json`

Mặc dù tên file là `data_50000_random.json`, nội dung thực tế có đúng:

- `43,658` records
- `5` công ty
- Schema: `company`, `page`, `index`, `reference`, `question`
- Không có query/reference rỗng

SHA256:

`e10e402a2189948eb11759f3cec65901bfff6471c6577e3726c240827f1f176a`

Phân bố records:

| Client | Công ty | Records |
|---:|---|---:|
| 0 | AES | 4,842 |
| 1 | BOEING | 5,302 |
| 2 | ACTIVISIONBLIZZARD | 6,328 |
| 3 | PG | 8,382 |
| 4 | PEPSICO | 18,804 |
| | Tổng | 43,658 |

### Train corpus chính thức

Đường dẫn:

`artifacts/data/paper_faithful_source/FEDE4FIN/train_corpus.json`

Thông tin:

- `368` documents
- `23,123` pages
- Năm công ty train tương ứng với `51` documents được paper mô tả

SHA256:

`009a967f9472ec71c42497ac11aae12db82417e9c4279ccaafa689de2d75f165`

### Manifest dữ liệu

`artifacts/data/paper_faithful/manifest.json`

Manifest lưu source revision, hash, record counts, company counts, schema và
kết quả các data gates.

## 6. Federated task paper-faithful

Đã thêm `PaperFiveCompanyPartitioner` với quy tắc nghiêm ngặt:

1. AES -> Client0
2. BOEING -> Client1
3. ACTIVISIONBLIZZARD -> Client2
4. PG -> Client3
5. PEPSICO -> Client4

Partitioner sẽ báo lỗi nếu:

- Số client khác 5
- Thiếu một trong năm công ty
- Xuất hiện công ty ngoài roster paper

Task đã tạo tại:

`artifacts/training/paper_faithful_task_v1_seed13/task/data.json`

Task metadata xác nhận:

- Strategy: `paper_five_company`
- Client sizes: `4842, 5302, 6328, 8382, 18804`
- Tổng client records: `43,658`
- Mỗi công ty chỉ thuộc đúng một client

## 7. Thay đổi code trong ngày

### `FedE/flgo/benchmark/partition.py`

- Thêm `PaperFiveCompanyPartitioner`
- Đóng băng thứ tự năm client theo Figure 3 của paper
- Thêm roster gate để ngăn dùng nhầm dữ liệu mở rộng

### `FedE/main_lora.py`

- Thêm biến môi trường:
  `PARTITIONER=paper_five`
- Giữ mặc định cũ:
  `PARTITIONER=company_lpt`
- Paper mode bắt buộc `NUM_CLIENTS=5`
- Task mặc định của paper mode dùng tên riêng, tránh tái sử dụng task cũ

### `FedE/flgo/benchmark/fedrag_classification/core.py`

- Task metadata ghi đúng tên partition strategy thay vì hard-code
  `company_lpt`

### `FedE/tools/validate_paper_faithful_data.py`

- Kiểm tra SHA256
- Kiểm tra 43,658 records
- Kiểm tra exact company counts
- Kiểm tra schema và empty pairs
- Kiểm tra train corpus hash, document count và page count
- Xuất manifest JSON và fail-fast nếu dữ liệu sai

### `FedE/tests/test_company_partitioner.py`

- Thêm test thứ tự client paper
- Thêm test từ chối company roster mở rộng

### Runner GPU

`scripts/vast/job-paper-faithful-data-smoke.sh`

Runner thực hiện:

1. Chạy data validator trước khi dùng GPU
2. Train FedAvg LoRA+KD bằng exact 43,658-pair dataset
3. Dùng B5 config hiện tại để cô lập tác động của việc đổi dữ liệu
4. Chạy smoke `3 rounds x 3 local steps`
5. Đánh giá bằng frozen full-corpus validation evaluator hiện tại
6. Lưu checkpoint, hash, manifest, log và result

## 8. Kết quả kiểm tra

### Unit tests

`FedE/tests/test_company_partitioner.py`: `5 passed`

### Data validation

Tất cả gates PASS:

- Train SHA256
- Train record count
- Exact company counts
- Record schema
- Non-empty query/reference pairs
- Train corpus SHA256
- Train corpus document/page counts

### Leakage và evaluator audit

Tất cả hard gates PASS:

- Validation qrels coverage: `50/50`
- Test qrels coverage: `100/100`
- Golden restore integrity: `168/168`
- Internal exact duplicates: `0`
- Train corpus vs evaluator corpus shared page fingerprints: `0`
- Train corpus vs restored golden pages: `0`
- Train references vs restored golden pages: `0`
- Evaluator index-independence scan: PASS

## 9. Chính sách evaluation

Lượt chạy đầu dùng frozen full-corpus evaluator của Q1 remediation để có thể
so sánh trực tiếp với các baseline cũ.

Cần tách riêng ba protocol:

1. Frozen full corpus hiện tại: 30,496 passages
2. Paper mô tả validation corpus khoảng 6,656 pages
3. Public code của paper cap ở 6,066 pages rồi append query references

Không được trộn kết quả của ba protocol vào cùng một bảng mà không ghi rõ.

## 10. Trạng thái hiện tại và việc tiếp theo

### Đã hoàn thành

- Checkout nhánh mới
- Tải đúng dữ liệu paper
- Đóng băng hash và revision
- Tạo exact five-client partition
- Tạo federated task
- Chạy unit tests
- Chạy data/leakage/evaluator gates
- Chuẩn bị GPU smoke runner

### Chưa thực hiện

- Chưa train smoke trên GPU
- Chưa chạy full budget
- Chưa chạy lại baseline matrix trên dataset mới
- Chưa quay lại epsilon sweep DP

### Lý do chưa train

Máy local hiện dùng PyTorch CPU-only. GPU laptop RTX 3050 Ti chỉ có 4 GB VRAM,
không đủ an toàn cho pipeline BGE federated hiện tại.

### Bước tiếp theo khi có server

1. Thuê GPU server và cung cấp SSH
2. Upload source data, code mới và frozen evaluator artifacts
3. Chạy `job-paper-faithful-data-smoke.sh`
4. Xác nhận task metadata vẫn là exact five-company partition
5. So sánh smoke với baseline cũ trên cùng evaluator
6. Nếu smoke đạt gate, chạy full budget và baseline matrix
7. Chỉ sau khi non-DP ổn định mới chạy lại client-level DP và CKKS
