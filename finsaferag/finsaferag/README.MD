# 💰 FinSafeRag: Financial Q&A Chatbot with Privacy Protection

**FinSafeRag** là một hệ thống chatbot RAG (Retrieval-Augmented Generation) dành riêng cho lĩnh vực tài chính, tích hợp bảo vệ dữ liệu cá nhân (Privacy) và hỗ trợ học tập liên kết (Federated Learning) thông qua Flower Framework.

## 🌟 Tính năng chính

✅ **RAG Chatbot** - Trả lời câu hỏi dựa trên tài liệu tài chính  
✅ **Privacy Protection** - Phát hiện & che dấu thông tin cá nhân (PII)  
✅ **Multi-Retriever** - 5 loại retriever: financial, general, technical, legal, default  
✅ **Federated Learning** - Hỗ trợ đưa ra câu trả lời toàn diện dựa treo tri thức liên minh vẫn đảm bảo tính bảo mật  
✅ **Web UI** - Giao diện Streamlit thân thiện  
✅ **REST API** - FastAPI backend  
✅ **GPU Support** - Tối ưu CUDA  

## 📋 Yêu cầu hệ thống

```
Python >= 3.10
CUDA 11.8+ (tuỳ chọn, nhưng khuyến nghị)
8GB RAM tối thiểu
2GB Disk cho models
```

## 🚀 Hướng dẫn cài đặt

### Step 1: Clone project

```bash
git clone https://github.com/ursuswh/FinSafeRag.git
cd FinSafeRag/Fedrag/finsaferag/finsaferag
```

### Step 2: Tạo Virtual Environment

```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

## 🎯 Cách sử dụng

./start_chatbot.sh

---

## 📊 API Endpoints

### **1. Query Endpoint**

```
POST /api/query

Request:
{
  "question": "What is 3M revenue?",
  "apply_privacy": true,
  "retriever_type": "financial",
  "use_federated": false
}

Response:
{
  "question": "What is 3M revenue?",
  "answer": "3M's revenue in 2024 was $36.8 billion",
  "sources": [
    {
      "content": "3M's revenue...",
      "score": 0.95
    }
  ],
  "privacy_applied": true,
  "privacy_stats": {
    "pii_detected": 0,
    "pii_density": 0.0,
    "sentences_removed": 0,
    "average_risk": 0.0,
    "encrypted": false
  },
  "response_time": 2.34
}
```

### **2. Health Check**

```
GET /api/health

Response:
{
  "status": "healthy",
  "timestamp": "2025-12-03T10:30:00Z",
  "version": "1.0.0"
}
```

### **3. Stats**

```
GET /api/stats

Response:
{
  "privacy_enabled": true,
  "total_queries": 42,
  "average_response_time": 2.15,
  "available_retrievers": ["financial", "general", "technical", "legal", "default"]
}
```

---

## 🔧 Cấu hình chính

### **Chỉnh Model LLM**

Sửa `config.toml`:

```toml
[settings]
llm = "gpt-4o"  # Các option: gpt-4o, gpt-4-turbo, gpt-4o-mini, gpt-3.5-turbo
```

### **Chỉnh Embedding Model**

```toml
[settings]
embeddings = "BAAI/bge-large-en"  # Hoặc: BAAI/bge-base-en, sentence-transformers/all-mpnet-base-v2
```

### **Bật/Tắt Privacy**

```toml
[privacy]
enable_privacy_summary = true  # true = BẬT, false = TẮT
```

### **Chỉnh Chunk Size**

```toml
[settings]
chunk_size = 2048  # Tăng → Chất lượng cao hơn nhưng chậm hơn
```

---

## 📁 Cấu trúc Project

```
finsaferag/
├── api/
│   ├── main.py              # FastAPI main server
│   ├── models.py            # Request/Response models
│   └── router.py            # Query routing logic
├── ui/
│   └── streamlit_app.py     # Web UI
├── llms/
│   └── llm.py               # LLM integration
├── privacy/
│   └── privacy_summary.py   # Privacy protection
├── embs/
│   └── embedding.py         # Embedding models
├── retriever/
│   └── retrievers.py        # Multi-retrievers
├── client_app.py            # Flower client
├── server_app.py            # Flower server
├── config.py                # Configuration loader
├── config.toml              # Configuration file
└── storage/                 # Vector index storage
    └── client_0/
        └── default__vector_store.json
```

---

## 🧪 Testing

### **Test Config**

```bash
python -c "
from config import Config
cfg = Config()
print(f'LLM: {cfg.llm}')
print(f'Privacy: {cfg.privacy}')
print(f'Embeddings: {cfg.embeddings}')
"
```

### **Test API**

```bash
# Health check
curl http://localhost:8000/api/health

# Test query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Apple stock price?",
    "apply_privacy": true,
    "retriever_type": "financial"
  }'
```

### **Test Streamlit UI**

```bash
streamlit run ui/streamlit_app.py --server.port 8501
# Truy cập: http://localhost:8501
```

---

## 🔒 Privacy Protection

### **Entities được phát hiện:**

- ✅ PERSON (Tên người)
- ✅ PHONE_NUMBER (Số điện thoại)
- ✅ EMAIL_ADDRESS (Email)
- ✅ CREDIT_CARD (Số thẻ)
- ✅ LOCATION (Địa điểm)
- ✅ US_SSN (Số an sinh xã hội)

### **Cách Privacy hoạt động:**

1. **Presidio Detection** - Phát hiện PII trong text
2. **Eraser4RAG** - Xóa sentences có risk cao
3. **Anonymization** - Thay thế `<PERSON>`, `<EMAIL>`, etc.
4. **TenSEAL Encryption** (tuỳ chọn) - Mã hóa end-to-end

---

## 🚨 Troubleshooting

### **Lỗi: `ModuleNotFoundError: No module named 'llama_index.llms.gemini'`**

**Fix:**
```bash
# Sửa llms/llm.py - Thay Gemini bằng OpenAI
# Hoặc cài đặt: pip install llama-index-llms-gemini
```

### **Lỗi: `Privacy module: DISABLED`**

**Fix:**
```bash
# 1. Kiểm tra config.toml có [privacy] section
# 2. Kiểm tra enable_privacy_summary = true
# 3. Xóa debug code ở cuối api/main.py (dòng 272+)
# 4. Restart API
```

### **Lỗi: `CUDA out of memory`**

**Fix:**
```bash
# Giảm chunk_size
chunk_size = 1024  # từ 2048

# Hoặc dùng CPU model nhẹ hơn
llm = "gpt-3.5-turbo"
```

### **Lỗi: `Index not found`**

**Fix:**
```bash
# Rebuild index
python -c "
from index import build_index
build_index()
print('Index built successfully')
"
```

---

## 📊 Performance Tips

### **Tăng tốc độ:**

```toml
[settings]
chunk_size = 1024           # Giảm từ 2048
llm = "gpt-3.5-turbo"       # Nhanh nhất
embeddings = "all-MiniLM-L6-v2"  # Model nhẹ
```

### **Tăng chất lượng:**

```toml
[settings]
chunk_size = 2048           # Tăng lên
llm = "gpt-4o"              # Chất lượng cao nhất
embeddings = "BAAI/bge-large-en"  # Model mạnh
```

---

## 🤝 Contributing

Pull requests được hoan nghênh! Vui lòng:

1. Fork repository
2. Tạo branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

Dự án này được cấp phép dưới MIT License - xem file `LICENSE` để chi tiết.

---

## 📞 Contact & Support

**Author:** UrsuSWH  
**Email:** ursuswh@example.com  
**GitHub:** [@ursuswh](https://github.com/ursuswh)

---

## 🎓 References

- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Flower Federated Learning](https://flower.ai/)
- [Presidio Privacy](https://microsoft.github.io/presidio/)

---

## 📈 Roadmap

- [ ] Support for more LLMs (Claude, Llama)
- [ ] Advanced query transformation (HyDE, StepBack)
- [ ] Multi-language support
- [ ] Vector database optimization (Pinecone, Weaviate)
- [ ] Monitoring & logging dashboard
- [ ] Docker support
- [ ] Production deployment guide

---

**Last Updated:** December 3, 2025  
**Version:** 1.0.0