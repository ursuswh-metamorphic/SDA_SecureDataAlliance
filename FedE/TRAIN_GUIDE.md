# Hướng Dẫn Chạy Upstream FedE — DP-FedRAG Training

> **Phiên bản:** v3.0 (Phase 1-6: LoRA + DP + InfoNCE/KL + FHE + qLoRA)
> **Cập nhật:** 2026-05-05

> Có sẵn **2 lộ trình**:
> * Lộ trình cũ (v2.0) — main.py / main_dp.py / main_dp_lora_eps20.py — vẫn chạy được, để regression
> * Lộ trình mới (v3.0) — main_lora.py / main_full.py — pipeline thống nhất qua flgo
>
> Xem [Section 11](#11-phase-runbook-v30) để chạy lộ trình mới.

---

## Mục Lục

1. [Tổng quan pipeline](#1-tổng-quan-pipeline)
2. [Yêu cầu môi trường](#2-yêu-cầu-môi-trường)
3. [Chuẩn bị dữ liệu](#3-chuẩn-bị-dữ-liệu)
4. [Chạy training — Các chế độ](#4-chạy-training--các-chế-độ)
5. [Cấu hình DP chi tiết](#5-cấu-hình-dp-chi-tiết)
6. [Theo dõi Privacy Budget](#6-theo-dõi-privacy-budget)
7. [Output & Checkpoints](#7-output--checkpoints)
8. [Convert model sang finsaferag](#8-convert-model-sang-finsaferag)
9. [Xử lý lỗi thường gặp](#9-xử-lý-lỗi-thường-gặp)
10. [Bảng tham số đầy đủ](#10-bảng-tham-số-đầy-đủ)
11. [Phase Runbook (v3.0 — pipeline thống nhất)](#11-phase-runbook-v30)

---

## 1. Tổng Quan Pipeline

```
finsaferag/data/
├── data_50.json            ← Nguồn dữ liệu QA
├── test_corpus_backup.json ← Nguồn corpus
└── rag_corpus.json         ← (generated) corpus cho Flower

FedE/
├── new_select_data.json    ← Dữ liệu training FedE (format: question/company/reference)
├── main.py                 ← Training KHÔNG có formal DP (baseline)
├── main_dp.py              ← Training VỚI formal DP (KHUYẾN NGHỊ)
└── flgo/algorithm/
    ├── fedrag.py           ← Baseline algorithm (batch-level clip, σ=0.1)
    └── fedrag_dp.py        ← DP algorithm mới (per-sample clip, RDP accounting)
```

**Luồng dữ liệu:**
```
data_50.json → new_select_data.json → FEDRAG Dataset → FedE Training → x-model_*.bin
                                                                              ↓
                                                         finsaferag/config.toml (model_path)
```

---

## 2. Yêu Cầu Môi Trường

### 2.1 Cài đặt dependencies

```bash
cd FedE
pip install -r requirements.txt

# Dependencies chính:
# torch==2.1.0, transformers, flgo, numpy
```

### 2.2 Kiểm tra GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 2.3 Kiểm tra dữ liệu đã sẵn sàng

```bash
# Phải tồn tại:
ls FedE/new_select_data.json       # ← file dữ liệu training
ls FedE/select_data.json           # ← file dữ liệu phụ (backup)

# Kiểm tra format:
python -c "
import json
with open('new_select_data.json') as f:
    d = json.load(f)
print(f'Records: {len(d)}')
print('Keys:', list(d[0].keys()))
print('Sample:', d[0]['question'][:60])
"
```

**Format yêu cầu của `new_select_data.json`:**
```json
[
  {
    "question": "What is the restructuring cost for PepsiCo in FY2022?",
    "company": "PepsiCo",
    "reference": "Note 3 Restructuring... [đoạn văn bản tài chính]"
  }
]
```

---

## 3. Chuẩn Bị Dữ Liệu

### 3.1 Trường hợp `new_select_data.json` chưa có / muốn regenerate

Tạo từ `finsaferag/data/data_50.json`:

```bash
cd FedE
python -c "
import json, pathlib

# Đọc data_50.json
src = pathlib.Path('../finsaferag/finsaferag/data/data_50.json')
with open(src, encoding='utf-8') as f:
    data = json.load(f)

records = []
for entry in data:
    question = entry['question']
    company = entry['other_info']['doc_name']
    refs = entry['key_content']['reference']
    # Gộp tất cả references thành 1 chuỗi
    merged_ref = ' '.join(refs) if isinstance(refs, list) else refs
    records.append({'question': question, 'company': company, 'reference': merged_ref})

with open('new_select_data.json', 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f'Created new_select_data.json: {len(records)} records')
"
```

### 3.2 Kiểm tra task directory

```bash
cd FedE
# Task directory tự động tạo nếu chưa có khi chạy main.py
# Nếu muốn regenerate task (xoá partition cũ):
rm -rf ./num5_alpha05
```

---

## 4. Chạy Training — Các Chế Độ

### Chế độ A: Baseline (KHÔNG DP) — So sánh upper-bound utility

```bash
cd FedE
python main.py
```

Config trong `main.py`:
```python
option={
    'num_rounds': 25,
    'num_epochs': 1,
    'gpu': 0,           # GPU ID, dùng [] cho CPU
    'batch_size': 8,
    'learning_rate': 0.00001,
    'dp_enabled': False,   # Tắt DP
}
```

---

### Chế độ B: Formal DP Training — **KHUYẾN NGHỊ** (dùng cho paper)

```bash
cd FedE
python main_dp.py
```

`main_dp.py` sẽ tự động:
1. Tính σ cần thiết để đạt `target_epsilon=8.0`
2. Verify privacy guarantee
3. Chạy training với RDP accounting
4. In privacy report sau mỗi round

**Output mẫu khi chạy:**
```
============================================================
[DP Calibration]
  Target ε     = 8.0
  Target δ     = 1e-05
  Num rounds   = 25
  Sampling q   = 0.600 (3/5 clients)
  → Calibrated σ = 1.6701
============================================================

[Verification] σ=1.6701 → ε=8.0089 at α=2.0
               δ=1e-05, rounds=25, q=0.600

[DP-FedRAG] Starting training with formal privacy guarantee...
[DP-Server] Round 5: ε_spent=1.8234 / ε_target=8.0  ← tracking real-time
[DP-Server] Round 10: ε_spent=3.7102 / ε_target=8.0
[DP-Server] Round 25: ε_spent=8.0089 / ε_target=8.0
```

---

### Chế độ C: Baseline DP (fedrag.py cũ) — để ablation study

```bash
cd FedE
# Sửa main.py: dùng fedrag thay vì fedrag_dp
python -c "
import flgo
import flgo.algorithm.fedrag as fedrag   # ← baseline algorithm
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
task = './num5_alpha05'
config = {'benchmark': {'name': 'flgo.benchmark.fedrag_classification'},
          'partitioner': {'name': 'IDPartitioner', 'para': {'num_clients': 5}}}
if not os.path.exists(task): flgo.gen_task(config, task_path=task)
runner = flgo.init(task=task, algorithm=fedrag,
    option={'num_rounds': 25, 'num_epochs': 1, 'gpu': 0, 'batch_size': 8,
            'learning_rate': 0.00001, 'dp_enabled': True,
            'dp_clip_norm': 1.0, 'dp_noise_multiplier': 0.1})  # ← baseline config
runner.run()
"
```

---

### Chế độ D: Chạy trên CPU (khi không có GPU)

```bash
cd FedE
python -c "
import flgo
import flgo.algorithm.fedrag_dp as fedrag_dp
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
task = './num5_alpha05'
config = {'benchmark': {'name': 'flgo.benchmark.fedrag_classification'},
          'partitioner': {'name': 'IDPartitioner', 'para': {'num_clients': 5}}}
if not os.path.exists(task): flgo.gen_task(config, task_path=task)
runner = flgo.init(task=task, algorithm=fedrag_dp,
    option={'num_rounds': 10, 'num_epochs': 1,
            'gpu': [],           # ← CPU mode
            'batch_size': 4,     # ← giảm batch size
            'learning_rate': 0.00001,
            'dp_enabled': True,
            'target_epsilon': 8.0,
            'target_delta': 1e-5,
            'server_noise_multiplier': 1.67,
            'dp_noise_multiplier': 1.67})
runner.run()
"
```

---

## 5. Cấu Hình DP Chi Tiết

### 5.1 Bảng σ khuyến nghị theo target ε

Chạy calibration để tìm σ phù hợp:

```bash
cd FedE
python -c "
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from privacy.rdp_accountant import find_noise_multiplier, compute_epsilon

print('Calibration Table: sigma for target eps=8.0, delta=1e-5')
print(f'{\"Config\":<35} {\"sigma\":>8} {\"verified_eps\":>14}')
print('-' * 60)

configs = [
    ('T=25, q=0.2 (1/5 clients)',  25, 0.2),
    ('T=25, q=0.4 (2/5 clients)',  25, 0.4),
    ('T=25, q=0.6 (3/5 clients)',  25, 0.6),
    ('T=10, q=0.2 (1/5 clients)',  10, 0.2),
    ('T=10, q=0.6 (3/5 clients)',  10, 0.6),
    ('T=50, q=0.2 (1/5 clients)',  50, 0.2),
]
for name, T, q in configs:
    sigma = find_noise_multiplier(8.0, T, q, 1e-5)
    eps, _ = compute_epsilon(T, sigma, q, 1e-5)
    print(f'{name:<35} {sigma:>8.4f} {eps:>14.4f}')
"
```

**Kết quả mẫu:**
```
Config                              sigma    verified_eps
------------------------------------------------------------
T=25, q=0.2 (1/5 clients)         0.7012         8.0032
T=25, q=0.4 (2/5 clients)         1.1696         8.0037
T=25, q=0.6 (3/5 clients)         1.6701         8.0089
T=10, q=0.2 (1/5 clients)         0.5250         8.0034
T=10, q=0.6 (3/5 clients)         1.1696         7.9987
T=50, q=0.2 (1/5 clients)         0.9213         8.0041
```

> **Tip:** q nhỏ hơn (ít clients/round) → privacy amplification mạnh hơn → σ nhỏ hơn → utility tốt hơn.

### 5.2 Cách đổi target ε trong `main_dp.py`

Mở `main_dp.py` và chỉnh:

```python
TARGET_EPSILON    = 3.0    # Thay 8.0 → 3.0 (mạnh hơn, utility thấp hơn)
TARGET_DELTA      = 1e-5
NUM_ROUNDS        = 25
CLIENTS_PER_ROUND = 1      # Giảm từ 3 → 1 để tăng amplification
```

### 5.3 Tắt adaptive clipping (dùng fixed C)

```python
option={
    ...
    'dp_adaptive_clip': False,     # Tắt adaptive
    'dp_clip_norm': 0.5,           # Fixed C = 0.5
    'server_clip_norm': 0.5,
}
```

---

## 6. Theo Dõi Privacy Budget

### 6.1 Check ε sau khi train xong

```bash
cd FedE
python -c "
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from privacy.rdp_accountant import RDPAccountant

# Điền vào các thông số đã dùng khi train:
accountant = RDPAccountant(
    noise_multiplier=1.6701,   # sigma đã dùng
    sample_rate=0.6,           # clients_per_round / total_clients
    delta=1e-5,
)
accountant._num_steps = 25     # Số rounds đã chạy

report = accountant.get_privacy_report()
print('Privacy Report:')
for k, v in report.items():
    print(f'  {k}: {v}')
"
```

### 6.2 Tính trước privacy budget cho experiment

```bash
cd FedE
python -c "
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from privacy.rdp_accountant import compute_epsilon

sigma = 1.6701
q = 0.6
delta = 1e-5

print('Round | eps_spent | Assessment')
print('-' * 50)
for T in [1, 5, 10, 15, 20, 25]:
    eps, _ = compute_epsilon(T, sigma, q, delta)
    level = 'GOOD' if eps <= 3 else 'MODERATE' if eps <= 8 else 'WEAK'
    print(f'  {T:3d}  |   {eps:6.3f}   | {level}')
"
```

**Output mẫu:**
```
Round | eps_spent | Assessment
--------------------------------------------------
    1  |    0.402   | GOOD
    5  |    2.014   | GOOD
   10  |    4.028   | MODERATE
   15  |    6.042   | MODERATE
   20  |    7.518   | MODERATE
   25  |    8.009   | MODERATE → đúng target!
```

---

## 7. Output & Checkpoints

### 7.1 Cấu trúc output

```
FedE/
├── checkpoints/
│   ├── x-model_2026-03-20_10-00-00_round0.bin   ← Checkpoint mỗi round
│   ├── x-model_2026-03-20_10-05-00_round1.bin
│   └── ...
├── x-model_2026-03-20_11-00-00.bin              ← Model cuối cùng (dùng cái này)
└── num5_alpha05/
    ├── data.json     ← Partition info
    └── *.json        ← Training logs
```

### 7.2 Xem log training

```bash
cd FedE/num5_alpha05
ls -la *.json
cat *.json | python -m json.tool | head -50
```

### 7.3 Load model để verify

```bash
cd FedE
python -c "
import torch
model_path = 'x-model_2026-03-20_11-00-00.bin'   # đổi tên file thực tế
state = torch.load(model_path, map_location='cpu')
print('Keys:', list(state.keys())[:5], '...')
print('Total params:', sum(v.numel() for v in state.values()))
"
```

---

## 8. Convert Model Sang finsaferag

### 8.1 Convert `.bin` → HuggingFace format

```bash
cd FedE
python -c "
import torch
from transformers import BertModel, BertConfig

# 1. Load trained state dict
model_file = 'x-model_2026-03-20_11-00-00.bin'  # đổi tên thực tế
state_dict = torch.load(model_file, map_location='cpu')

# 2. Load base model
base = BertModel.from_pretrained('BAAI/bge-base-en')

# 3. Load weights (loại bỏ prefix 'model.' nếu cần)
clean_state = {}
for k, v in state_dict.items():
    key = k.replace('model.', '', 1) if k.startswith('model.') else k
    clean_state[key] = v

missing, unexpected = base.load_state_dict(clean_state, strict=False)
print(f'Missing: {len(missing)}, Unexpected: {len(unexpected)}')

# 4. Save HuggingFace format
import datetime
out_name = f'x-model_{datetime.datetime.now().strftime(\"%Y-%m-%d_%H-%M-%S\")}_converted'
base.save_pretrained(f'../{out_name}')
print(f'Saved to: ../{out_name}')
"
```

### 8.2 Cập nhật finsaferag config

Sau khi convert, mở `finsaferag/finsaferag/config.toml`:

```toml
[settings]
# Đổi thành tên thư mục vừa tạo:
embeddings = "x-model_2026-03-20_11-00-00_converted"
model_path = "x-model_2026-03-20_11-00-00_converted"
```

### 8.3 Verify model trong finsaferag

```bash
cd finsaferag
python -c "
from finsaferag.embs.embedding import get_embedding_model
emb = get_embedding_model()
test = emb.get_text_embedding('What is 3M revenue in 2019?')
print('Embedding dim:', len(test))
print('Sample values:', test[:5])
"
```

---

## 9. Xử Lý Lỗi Thường Gặp

### Lỗi 1: `FileNotFoundError: new_select_data.json`

```
FileNotFoundError: [Errno 2] No such file or directory: './new_select_data.json'
```

**Nguyên nhân:** Chạy từ sai thư mục, hoặc file chưa tồn tại.

```bash
# Kiểm tra:
ls FedE/new_select_data.json

# Fix: phải chạy từ trong FedE/
cd FedE
python main_dp.py   # ← phải cd vào FedE trước
```

---

### Lỗi 2: `CUDA out of memory`

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Fix:**
```python
# Trong main_dp.py hoặc main.py, giảm batch_size:
'batch_size': 4,    # thay vì 8
# Hoặc dùng CPU:
'gpu': [],
```

---

### Lỗi 3: `ModuleNotFoundError: No module named 'privacy'`

```
ModuleNotFoundError: No module named 'privacy'
```

**Nguyên nhân:** Python path không tìm thấy `FedE/privacy/`.

```bash
# Fix: chạy từ FedE/ (không phải root)
cd FedE
python main_dp.py

# Hoặc set PYTHONPATH:
set PYTHONPATH=d:\Sukem\NCKH\SDA_SecureDataAlliance\FedE
python main_dp.py
```

---

### Lỗi 4: `ModuleNotFoundError: No module named 'flgo'`

```bash
pip install flgo
# Hoặc:
cd FedE && pip install -e .
```

---

### Lỗi 5: Task directory bị corrupt

```
KeyError: 'client_names'
```

```bash
# Xoá task và tạo lại:
cd FedE
rm -rf ./num5_alpha05
python main_dp.py  # tự tạo lại
```

---

### Lỗi 6: `UnicodeEncodeError` trên Windows

```
UnicodeEncodeError: 'charmap' codec can't encode character
```

```bash
# Windows: set encoding
set PYTHONIOENCODING=utf-8
python main_dp.py
```

---

### Lỗi 7: Privacy budget calibration fails

```
ValueError: Cannot achieve ε=8.0 even with σ=200.
```

**Nguyên nhân:** `num_rounds` quá lớn hoặc `sample_rate` quá cao.

```python
# Giảm num_rounds hoặc tăng target_epsilon:
TARGET_EPSILON    = 10.0  # relaxed
NUM_ROUNDS        = 15    # ít rounds hơn
CLIENTS_PER_ROUND = 1     # amplification tối đa (1/5 clients)
```

---

## 10. Bảng Tham Số Đầy Đủ

### Standard FL Parameters

| Tham số | Default | Mô tả |
|---|---|---|
| `num_rounds` | 25 | Số communication rounds T |
| `num_epochs` | 1 | Epochs local mỗi round |
| `batch_size` | 8 | Local batch size |
| `learning_rate` | 0.00001 | Learning rate (lr=1e-5 cho BERT fine-tune) |
| `gpu` | 0 | GPU ID; dùng `[]` cho CPU |
| `num_workers` | 0 | DataLoader workers |

### DP Parameters (NEW — fedrag_dp.py)

| Tham số | Khuyến nghị | Baseline cũ | Mô tả |
|---|---|---|---|
| `dp_enabled` | `True` | `True` | Bật/tắt DP |
| **`target_epsilon`** | **8.0** | N/A | Target ε budget (NEW) |
| **`target_delta`** | **1e-5** | N/A | Target δ (NEW) |
| `dp_clip_norm` | 1.0 | 1.0 | Per-sample clip norm C |
| **`dp_noise_multiplier`** | **auto-calibrated** | **0.1 (SAI)** | Noise multiplier σ |
| `server_clip_norm` | 1.0 | N/A | Server update clip norm (NEW) |
| `server_noise_multiplier` | auto-calibrated | N/A | Server noise (NEW) |
| `dp_clients_per_round` | 3 (của 5) | N/A | Clients/round cho amplification (NEW) |
| `dp_adaptive_clip` | `True` | N/A | Adaptive clipping (NEW) |
| `dp_clip_gamma` | 0.5 | N/A | Target quantile cho adaptive clip (NEW) |

### Quick Reference: Noise Multiplier σ vs ε

| ε target | T=25, q=0.2 | T=25, q=0.4 | T=25, q=0.6 | Privacy Level |
|---|---|---|---|---|
| ε = 1.0 | σ = 2.85 | σ = 4.20 | σ = 6.10 | STRONG |
| ε = 3.0 | σ = 1.20 | σ = 1.80 | σ = 2.58 | GOOD |
| ε = 8.0 | σ = 0.70 | σ = 1.17 | σ = 1.67 | MODERATE |
| ε = 10.0 | σ = 0.60 | σ = 1.00 | σ = 1.43 | WEAK |

> **Lưu ý quan trọng:** σ=0.1 (baseline cũ) → ε≈2505 → KHÔNG CÓ BẤT KỲ PRIVACY GUARANTEE NÀO.

---

## Quick Start Tóm Tắt

```bash
# ── Lần đầu chạy ──────────────────────────────────────────────────────────────
cd d:\Sukem\NCKH\SDA_SecureDataAlliance\FedE

# 1. Kiểm tra data
python -c "import json; d=json.load(open('new_select_data.json')); print(len(d), 'records OK')"

# 2. Chạy training với formal DP (KHUYẾN NGHỊ)
set PYTHONIOENCODING=utf-8
set KMP_DUPLICATE_LIB_OK=TRUE
python main_dp.py

# 3. Sau khi xong, convert model
python -c "
import torch
from transformers import BertModel
import datetime, glob

# Tìm model mới nhất
files = sorted(glob.glob('x-model_*.bin'), key=lambda f: f)
latest = [f for f in files if 'round' not in f][-1]
print('Converting:', latest)
state = torch.load(latest, map_location='cpu')
base = BertModel.from_pretrained('BAAI/bge-base-en')
clean = {k.replace('model.', '', 1): v for k, v in state.items()}
base.load_state_dict(clean, strict=False)
name = 'x-model_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_converted'
base.save_pretrained('../' + name)
print('Saved:', name)
"

# 4. Update finsaferag config.toml với tên model mới
```

---

## 11. Phase Runbook (v3.0)

> **Khi nào dùng v3.0 thay vì v2.0?**
> * Bạn cần **3 lớp bảo vệ cùng lúc** (FL + FHE + DP) trong một lần chạy.
> * Bạn muốn pipeline chạy qua flgo framework (không phải standalone script).
> * Bạn cần truyền tải nhỏ (~1.2 MB/round vs ~437 MB/round).

### 11.1 Bản đồ entrypoint

| File | Phase | Algorithm | Kích hoạt |
|---|---|---|---|
| `main.py` | v2.0 baseline | `fedrag.py` | full BertModel, no DP, no FHE |
| `main_dp.py` | v2.0 DP | `fedrag_dp.py` | full BertModel + user-level DP |
| `main_dp_lora_eps20.py` | v2.0 LoRA+DP | standalone (bypass flgo) | LoRA+DP, không qua flgo |
| **`main_lora.py`** | **v3.0 P1+2+3+5** | `fedrag_lora.py` | LoRA + DP + InfoNCE/KL (+qLoRA Linux) |
| **`main_full.py`** | **v3.0 P1-6** | `fedrag_lora_ckks.py` | tất cả các lớp + FHE thật sự |

### 11.2 Chạy theo từng Phase

**Phase 1 (smoke, no DP) — verify LoRA ship qua flgo:**
```bash
cd FedE
python main_lora.py
# Expect: payload ~1.2 MB / round, 25 rounds complete
```

**Phase 2 (LoRA + DP, regression target):**
```bash
DP_ENABLED=1 python main_lora.py
# Expect: σ=1.2940, final ε ≤ 20, retention ±0.5% so với main_dp_lora_eps20.py
```

**Phase 3 (InfoNCE + KL):** đã tự động ON trong main_lora.py via
`fedrag_core.DEFAULT_TEMPERATURE=0.05, DEFAULT_KD_WEIGHT=1.0`. Không cần env var.

**Phase 4 (true homomorphic CKKS) — Linux only (TenSEAL wheels):**
```bash
# Phải có tenseal: pip install tenseal (Python 3.12 hoặc <)
python main_full.py
# FHE_ENABLED=1 by default. Server không decrypt; final cipher saved to checkpoints/
```

**Phase 5 (qLoRA 4-bit base) — Linux + CUDA + bitsandbytes:**
```bash
USE_QLORA=1 DP_ENABLED=1 python main_lora.py
# Expect: VRAM ~1/3 so với Phase 2; batch_size 16 thay vì 8
```

**Phase 6 (full pipeline) — Linux + CUDA + tenseal + bitsandbytes:**
```bash
USE_QLORA=1 DP_ENABLED=1 FHE_ENABLED=1 python main_full.py
# Expect: 3 lớp bảo vệ active; eps_spent ≤ 20; final cipher in checkpoints/final_cipher.pt
```

### 11.3 Decrypt final cipher (Phase 4/6 only)

Server (Phase 6) chỉ giữ ciphertext sau training. Để evaluate cần một
client (giữ secret key) decrypt:

```bash
python -c "
import torch
from FedE.flgo.algorithm.fedrag_ckks_primitives import _ensure_ctx, _chunked_decrypt
secret_ctx, _ = _ensure_ctx()  # session-shared singleton; works only in same process
blob = torch.load('checkpoints/final_cipher.pt')
state = _chunked_decrypt(blob['cipher'], blob['manifest'], secret_ctx)
print('Decrypted', len(state), 'LoRA tensors')
torch.save(state, 'checkpoints/final_lora_decrypted.pt')
"
```
> **Lưu ý cross-session**: TenSEAL secret key không tự survive across processes
> trong implementation hiện tại. Để eval cipher từ một run khác, cần persist
> secret context kèm `secret_ctx.serialize(save_secret_key=True)` — out-of-scope cho v3.0 PoC.

### 11.4 Evaluate v3.0 checkpoints

`eval_compare.py` đã được nâng cấp để tự động detect 3 format:

```bash
python eval_compare.py
# Tự động xử lý:
#   - x-model_*.bin           (v2.0 full BertModel state)
#   - x-lora_*.bin             (v3.0 LoRA-only state, được merge_and_unload tự động)
#   - x-model_lora_merged_*.bin (v2.0 merged from main_dp_lora_eps20.py)
```

Sửa các đường dẫn `load_model('xxx.bin', 'name')` trong `eval_compare.py:213-221`
để chỉ tới checkpoints bạn vừa train.

### 11.5 Validation tests

```bash
cd FedE
# Phase 1 — LoRA filter (cần torch + transformers + peft)
python tests/test_lora_filter.py
# Expect: trainable=294,912, payload < 2 MB

# Phase 3 — InfoNCE + KL sanity
python tests/test_loss_phase3.py
# Expect: 5/5 checks pass (KL=0 khi student==teacher, vv)

# Phase 4 — CKKS cryptographic correctness (Linux + tenseal)
python tests/test_ckks_correctness.py
# Expect: max abs diff ≤ 1e-3 giữa encrypted-mean và plaintext-mean

# Phase 5 — qLoRA gate
python tests/test_qlora_gate.py
# Expect: gate đúng theo platform; trainable=294,912 dù qLoRA on/off
```

### 11.6 Mapping 3 lớp bảo vệ ↔ Phase

| Lớp | Bảo vệ chống | Implement bởi | Phase |
|---|---|---|---|
| **L1 — không chia sẻ data** | external attacker | flgo FL | 1-6 |
| **L2 — không lộ gradient** | curious server | CKKS encrypt + homomorphic add | 4, 6 |
| **L3 — không reverse được** | post-decrypt attacker | per-sample DP-SGD | 2, 6 |

Phase 6 (`main_full.py`) là cái duy nhất kích hoạt cả 3 lớp đồng thời.
```
