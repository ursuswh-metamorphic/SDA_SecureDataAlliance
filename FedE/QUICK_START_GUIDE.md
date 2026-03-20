"""
TỔNG HỢP HƯỚNG DẪN: Tạo Synthetic Training Data cho FedE Retriever
==================================================================

Ngày tạo: March 20, 2026
Mục đích: Thiết lập đầy đủ pipeline tạo training data từ PubMed cho retriever
Định dạng output: (query, positive_chunk, negative_chunks)


QUI TRÌNH (PIPELINE OVERVIEW)
==============================

Input: pubmed23n0001.json (34 MB, ~10000 documents)
                ↓
[1] LOAD: Đọc tài liệu
                ↓
[2] CHUNK: Chia thành đoạn text (sliding window, 256 tokens)
                ↓
[3] FILTER: Loại bỏ chunks chất lượng thấp
                ↓
[4] GENERATE QUERIES: Dùng LLM tạo câu hỏi
                ↓
[5] BUILD PAIRS: Tạo (query, positive, negatives)
                ↓
[6] SAVE: Lưu dưới dạng JSON
                ↓
Output: pubmed_retriever_train_*.json


QUICK START (Nhanh nhất)
=======================

cd d:\lab\SDA_SecureDataAlliance\FedE

# 1. Test với dataset nhỏ (20 docs)
python quick_start.py --preset tiny

# 2. Generate medium dataset (500 docs, 2500 pairs) - RECOMMENDED
python quick_start.py --preset medium

# 3. Generate + split cho 4 federated clients
python quick_start.py --preset medium --split-clients 4


STEP-BY-STEP GUIDE (Chi tiết từng bước)
========================================


STEP 1: VERIFY SETUP
-------------------
cd d:\lab\SDA_SecureDataAlliance\FedE

# Kiểm tra files tồn tại
ls pubmed23n0001.json  # input
ls generate_retriever_training_data.py  # main script
ls quick_start.py  # quick start script
ls dataset_utils.py  # utilities
ls test_retriever_generation.py  # test script

# Kiểm tra LLM config (trong finsaferag/config.toml)
# - Đảm bảo API key đã set
# - Đảm bảo network connectivity


STEP 2: TEST VỚI DATASET NHỎ
----------------------------
python test_retriever_generation.py

Output:
  ✓ Kiểm tra format JSON
  ✓ Kiểm tra LLM connection
  ✓ Tạo test_retriever_train_small.json (5 docs, ~25 pairs)

Nếu thành công, có thể tiếp tục.
Nếu lỗi, xem troubleshooting bên dưới.


STEP 3: GENERATE MAIN DATASET (MỘT TRONG CÁC CÁCH)
--------------------------------------------------

CÁCH 1: QUICK START (Khuyến nghị)
-----------
# Medium dataset (500 docs, 2500 pairs) - 15-30 phút
python quick_start.py --preset medium

# Hoặc large (1000 docs, 5000 pairs) - 30-60 phút
python quick_start.py --preset large


CÁCH 2: CUSTOM PARAMETERS
-----------
python generate_retriever_training_data.py \
  --pubmed pubmed23n0001.json \
  --output pubmed_retriever_train_custom.json \
  --max-docs 300 \
  --queries-per-chunk 1 \
  --chunk-size 256

Parameters:
  --pubmed: đường dẫn file PubMed
  --output: đường dẫn output
  --max-docs: số docs (None = tất cả)
  --queries-per-chunk: queries/chunk (1 hoặc 2)
  --chunk-size: cỡ chunk (tokens/sentences)


CÁCH 3: BATCH MODE (Nếu API limited)
-----------
# Tạo từng batch để tránh API limit

# Batch 1: 200 docs
python generate_retriever_training_data.py \
  --max-docs 200 \
  --output pubmed_retriever_train_batch1.json

# Batch 2: 200 docs khác (từ partition của PubMed)
python generate_retriever_training_data.py \
  --max-docs 200 \
  --output pubmed_retriever_train_batch2.json

# Merge
python dataset_utils.py merge <batch1> <batch2> merged.json


STEP 4: VALIDATE DATASET
------------------------
# Kiểm tra integrity
python dataset_utils.py validate pubmed_retriever_train_medium.json

# Analyze statistics
python dataset_utils.py analyze pubmed_retriever_train_medium.json

Output:
  ✓ Format check
  ✓ Statistics (query length, chunk size, etc.)
  ✓ Doc coverage


STEP 5: SPLIT CHO FEDERATED CLIENTS (Tùy chọn)
----------------------------------------------
# Split cho 4 clients
python dataset_utils.py split pubmed_retriever_train_medium.json 4

Output:
  fede_client_datasets/
    ├── client_0_data.json  (~625 pairs)
    ├── client_1_data.json  (~625 pairs)
    ├── client_2_data.json  (~625 pairs)
    └── client_3_data.json  (~625 pairs)

Hoặc trong quick_start:
python quick_start.py --preset medium --split-clients 4


STEP 6: USE IN FEDERATED TRAINING
---------------------------------
# Copy to clients
cp fede_client_datasets/client_0_data.json ~/client_0/retriever_train.json
cp fede_client_datasets/client_1_data.json ~/client_1/retriever_train.json
...

# Trong training script (pseudo-code):
from finsaferag.lms import Retriever

retriever = Retriever()
for pair in training_data:
    query = pair['query']
    positive = pair['positive']['text']
    negatives = [n['text'] for n in pair['negatives']]
    
    # Train with contrastive loss
    loss = retriever.train_step(query, positive, negatives)


FORMAT CHI TIẾT
================

INPUT - PubMed JSON:
[
  {
    "id": "pubmed23n0001_0",
    "title": "...",
    "content": "...",
    "contents": "...",
    "PMID": 21
  },
  ...
]


OUTPUT - Training Dataset JSON:
[
  {
    "query": "What is the effect of bisabolol on peptic activity?",
    
    "positive": {
      "id": "pubmed23n0001_0_chunk_0",
      "doc_id": "pubmed23n0001_0",
      "text": "[full chunk 256 tokens]",
      "title": "[document title]"
    },
    
    "negatives": [
      {
        "id": "pubmed23n0001_5_chunk_2",
        "doc_id": "pubmed23n0001_5",
        "text": "[negative chunk from different doc]",
        "title": "[title]"
      },
      // ... 2 more negatives
    ],
    
    "metadata": {
      "doc_id": "pubmed23n0001_0",
      "pmid": "21",
      "chunk_id": "pubmed23n0001_0_chunk_0"
    }
  },
  // ... more pairs
]


PRESET SIZES (quick_start.py)
=============================

tiny:   20 docs   →  ~100 pairs   (2-5 min)   - Test only
small:  100 docs  →  ~500 pairs   (5-10 min)
medium: 500 docs  →  ~2500 pairs  (15-30 min) - RECOMMENDED ⭐
large:  1000 docs →  ~5000 pairs  (30-60 min)
xl:     all docs  →  ~50000+ pairs (2-4 hours) - SLOW!


TRẠNG THÁI LLM API CALLS
========================

Query Generation Strategy:
  - 1 LLM call per chunk (to generate query)
  - Cost estimates (OpenAI pricing):
    - 500 docs → 2500 chunks → 2500 calls → ~$1-3
    - 1000 docs → 5000 chunks → 5000 calls → ~$2-5
    - Full → ~50000 calls → ~$20-50

  - NVIDIA NIM: Miễn phí/rẻ hơn
  - Local LLM: Miễn phí (chỉ cần GPU)


LỖI THƯỜNG GẶP & CÁCH KHẮC PHỤC
==================================

❌ "ModuleNotFoundError: No module named 'llms'"
✓ Chạy lệnh: cd FedE && python (chứ không chạy từ ngoài)
✓ Hoặc: PYTHONPATH=../finsaferag python script.py


❌ "LLM connection timeout"
✓ Kiểm tra API key trong finsaferag/config.toml
✓ Kiểm tra network connectivity
✓ Thử với local LLM (NVIDIA, Ollama) thay vì API


❌ "CUDA out of memory"
✓ Nếu dùng local LLM: giảm batch size
✓ Hoặc dùng quantized model (4-bit, 8-bit)


❌ "Queries không hợp lý / không phải tiếng Anh"
✓ Chỉnh sửa prompt trong generate_retriever_training_data.py
✓ Độc lập 'generate_query_for_chunk()' function


❌ "Too few/many pairs generated"
✓ Số pairs = num_docs * chunks_per_doc * queries_per_chunk
✓ Adjust --max-docs hoặc --queries-per-chunk


❌ "Out of memory khi load full dataset"
✓ Dùng batch processing
✓ Dùng --max-docs để giới hạn


TINH CHỈNH (OPTIMIZATION)
==========================

1. QUERY QUALITY:
   - Chỉnh sửa prompt trong generate_query_for_chunk()
   - Test với sample trước khi chạy toàn bộ

2. CHUNK SIZE:
   - Nhỏ hơn (128 tokens): chi tiết, có nhiều chunks hơn
   - Lớn hơn (512 tokens): ít chunks, context lớn

3. NEGATIVE SAMPLING:
   - Hiện tại: random từ khác doc_id
   - Có thể upgrade: hard negative mining (khoảng cách embedding)

4. SPEED:
   - Parallel query generation (thread/async)
   - Cache LLM responses

5. DATASET BALANCE:
   - Nếu cần balanced negatives: custom sampling logic


INTEGRATION VỚI FEDERATED TRAINING
===================================

File structure cho FedE:
  FedE/
  ├── pubmed_retriever_train_medium.json
  └── fede_client_datasets/
      ├── client_0_data.json
      ├── client_1_data.json
      ├── client_2_data.json
      └── client_3_data.json

Training pipeline:
  1. Server: tạo dataset từ PubMed
  2. Server: split cho clients
  3. Clients: độc lập train retriever (contrastive learning)
  4. Server: aggregate embeddings
  5. Server: save federated retriever


STATISTICS EXAMPLE
==================

Dataset tạo từ 500 docs:

Total pairs: 2,450

Query Statistics (words):
  Min: 8, Max: 45, Mean: 18.5, Median: 18

Positive Chunk Statistics (characters):
  Min: 120, Max: 2,046, Mean: 856.3, Median: 780

Negatives per Pair:
  Min: 3, Max: 3, Mean: 3.0

Document Coverage:
  Unique docs: 500
  Pairs per doc (mean): 4.9


NEXT STEPS AFTER GENERATION
=============================

1. ✓ Qua retriever training stage
2. ✓ Fine-tuning on specific domains
3. ✓ Integrate với medical QA system
4. ✓ Deploy trên federated clients
5. ✓ Monitor & update periodically


SUPPORT & RESOURCES
====================

Scripts dalam folder:
  - generate_retriever_training_data.py: Main pipeline
  - test_retriever_generation.py: Test small dataset
  - quick_start.py: Quick generation with presets
  - dataset_utils.py: Utilities (split, validate, analyze)
  - RETRIEVER_TRAINING_DATA_README.md: Detailed docs

Docs tham khảo:
  - COMPLETE_TRAINING_TO_RAG_PIPELINE.md
  - RAG_DETAILED_EXPLANATION.md
  - finsaferag/llms/llm.py: LLM configuration
  - finsaferag/config.toml: Config file


CHANGELOG
=========

v1.0 (2026-03-20):
  ✓ Load PubMed
  ✓ Chunk + Filter
  ✓ LLM query generation
  ✓ Negative sampling
  ✓ JSON output
  ✓ Quick start
  ✓ Dataset utilities
  ✓ Test scripts

Future:
  - Hard negative mining
  - Multi-language support
  - Batch API optimization
  - Visualization tools
"""

# This is markdown documentation
# Use as reference guide
