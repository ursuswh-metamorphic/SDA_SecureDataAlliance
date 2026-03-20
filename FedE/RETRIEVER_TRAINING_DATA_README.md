"""
Hướng dẫn sử dụng: Generate Retriever Training Data từ PubMed
===============================================================

TỔNG QUAN (Overview):
- Tạo dữ liệu huấn luyện cho retriever từ corpus PubMed
- Format output: (query, positive_chunk, negative_chunks)
- Một sample training pair sẽ có:
  - query: câu hỏi sinh tổng hợp
  - positive: đoạn văn bản tương ứng
  - negatives: 3 đoạn văn bản từ các tài liệu khác (hard negatives)


QUY TRÌNH HOẠT ĐỘNG (Pipeline):
==============================

1. Load PubMed: Đọc file pubmed23n0001.json
2. Chunk: Chia tài liệu thành các đoạn text (256 tokens, overlap 64)
3. Filter: Loại bỏ chunks chất lượng thấp
   - Kiểm tra độ dài
   - Kiểm tra từ khóa y tế
   - Loại bỏ văn bản lặp lại

4. Generate Queries: Dùng LLM để tạo câu hỏi cho mỗi chunk
5. Build Pairs: Tạo (query, positive, negatives) từ chunks
6. Save: Lưu dataset thành JSON


CÁC FILE CẦN:
==============

1. generate_retriever_training_data.py
   - File chính với class MedicalDatasetGenerator
   - Pipeline hoàn chỉnh để tạo dataset

2. test_retriever_generation.py
   - Script để test với 5 tài liệu
   - Kiểm tra format output

3. pubmed23n0001.json
   - File input PubMed (sẵn có)
   - ~34MB, chứa hàng nghìn medical documents


HƯỚNG DẪN SỬ DỤNG (Tutorial):
===============================

[1] TEST VỚI DỮ LIỆU NHỎ:
-----------
cd d:\lab\SDA_SecureDataAlliance\FedE

# Test với 5 tài liệu
python test_retriever_generation.py

Output:
- test_retriever_train_small.json (tinyDatasensor ~50-100 pairs)
- Log sẽ hiển thị sample pair để xem kết quả


[2] GENERATE TOÀN BỘ DATASET:
-----------
python generate_retriever_training_data.py \\
  --pubmed pubmed23n0001.json \\
  --output pubmed_retriever_train.json \\
  --max-docs 1000 \\
  --queries-per-chunk 1 \\
  --chunk-size 256

Tham số:
  --pubmed: Đường dẫn file PubMed input
  --output: Đường dẫn file output
  --max-docs: Giới hạn số tài liệu (None = dùng tất cả)
  --queries-per-chunk: Số query sinh cho mỗi chunk (1 hoặc 2)
  --chunk-size: Kích thước chunk (256 tokens/sentences)

Example outputs:
  - 100 docs → ~500-1000 training pairs
  - 1000 docs → ~5000-10000 training pairs
  - All docs → ~50000-100000 training pairs


[3] CÓ THỂ CHẠY VỚI CONFIG KHÁC:
-----------

# Nhanh (quick): 200 docs, 1 query per chunk
python generate_retriever_training_data.py \\
  --max-docs 200 \\
  --queries-per-chunk 1 \\
  --output pubmed_retriever_train_quick.json

# Chi tiết (detailed): 500 docs, 2 queries per chunk
python generate_retriever_training_data.py \\
  --max-docs 500 \\
  --queries-per-chunk 2 \\
  --output pubmed_retriever_train_detailed.json

# Toàn bộ (full): Dùng tất cả docs
python generate_retriever_training_data.py \\
  --output pubmed_retriever_train_full.json


FORMAT OUTPUT JSON:
===================

[
  {
    "query": "What is the effect of bisabolol on peptic activity?",
    "positive": {
      "id": "pubmed23n0001_0_chunk_0",
      "doc_id": "pubmed23n0001_0",
      "text": "[full chunk text here]",
      "title": "[document title]"
    },
    "negatives": [
      {
        "id": "pubmed23n0001_5_chunk_2",
        "doc_id": "pubmed23n0001_5",
        "text": "[negative chunk text]",
        "title": "[title]"
      },
      // ... more negatives
    ],
    "metadata": {
      "doc_id": "pubmed23n0001_0",
      "pmid": "21",
      "chunk_id": "pubmed23n0001_0_chunk_0"
    }
  },
  // ... more pairs
]

Mỗi pair có:
- query: string - câu hỏi y tế
- positive: object - đoạn text đúng (gold passage)
- negatives: array of objects - hard negatives (3 đoạn khác)
- metadata: object - thông tin meta


YÊU CẦU CẤU HÌNH (Requirements):
==================================

Config file (finsaferag/config.toml):
- Phải có LLM đã cấu hình (NVIDIA, OpenAI, local model, etc.)
- Ví dụ với NVIDIA NIM:
  llm = "nvidia"
  nvidia_api_key = "your-key"
  nvidia_api_base = "https://integrate.api.nvidia.com/v1"
  nvidia_model = "meta/llama-3.1-70b-instruct"

Tested LLM:
- NVIDIA (meta/llama-3.1-70b-instruct) ✓
- OpenAI (gpt-3.5-turbo) ✓
- Hugging Face local models ✓


MỘT SỐ GHI CHÚ (Notes):
=======================

1. QUERY GENERATION CHỈ BẰNG TIẾNG ANH:
   - Hiện tại script tạo queries bằng Tiếng Anh
   - Có thể chỉnh sửa prompt để tạo queries bằng Tiếng Việt nếu cần

2. LLM API CALLS:
   - Nếu có N chunks, sẽ gọi LLM N lần
   - Tính toán: 1000 chunks ≈ 1000 API calls
   - Nếu dùng API trả phí, cần lưu ý chi phí

3. MEMORY & PERFORMANCE:
   - Full dataset với 10000+ pairs: ~500MB+ RAM
   - Nên chạy trên GPU nếu dùng local LLM

4. NEGATIVES SAMPLING:
   - Lấy random từ chunks khác doc_id
   - Có thể customize để chọn hard negatives thông minh hơn

5. CHUNK SIZE:
   - 256 tokens ≈ 200-300 từ
   - Có thể điều chỉnh --chunk-size tuỳ yêu cầu
   - Nhỏ hơn: chi tiết hơn nhưng nhiều chunks hơn
   - Lớn hơn: ít chunks nhưng context lớn hơn


KHẮC PHỤC SỰ CỐ (Troubleshooting):
===================================

❌ "LLM connection failed"
✓ Kiểm tra LLM config trong config.toml
✓ Kiểm tra API key, network connectivity

❌ "Out of memory"
✓ Giảm --max-docs
✓ Chạy nhiều lần với batch nhỏ hơn

❌ "Queries không hợp lý"
✓ Chỉnh sửa prompt trong generate_query_for_chunk()
✓ Thay đổi parameter nhiệt độ của LLM

❌ "Format lỗi"
✓ Chạy test_retriever_generation.py để kiểm tra
✓ Xem sample JSON output


TÍCH HỢP VỚI FEDERATED TRAINING:
==================================

Sử dụng dataset trong FedE:

1. Chia dataset thành nhiều flux (1 flux/client)
   - Client 0: pairs [0:N/k]
   - Client 1: pairs [N/k:2N/k]
   - ...

2. Format cho FedE retriever task:
   {
     "client_data": {...},
     "query": query,
     "positive": positive_text,
     "negatives": [neg1_text, neg2_text, ...]
   }

3. Train contrastive loss (ở client)
4. Aggregate embeddings (ở server)


LIÊN HỆ & SUPPORT:
==================
- Thắc mắc về LLM calls: Xem llms/llm.py
- Về chunking/filtering: Xem generate_retriever_training_data.py
- Về format: Xem data structure trong tệp này


VERSION HISTORY:
================
v1.0: Initial release
  - Load PubMed
  - Chunk + Filter
  - Query generation với LLM
  - Negative sampling
  - JSON output

v1.1 (planned):
  - Hard negative mining
  - Multi-language query generation
  - Batch processing
  - Dataset visualization
"""

# Đây là file README, không phải Python executable
# Sử dụng làm tài liệu hướng dẫn
