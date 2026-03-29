# Evaluation Report: Differential Privacy in Federated RAG Embedding Training

> **Project:** FinSafeRAG — Privacy-Aware Federated Retrieval-Augmented Generation
> **Branch:** `trang/differential_privacy`
> **Date:** 2026-03-29
> **Server:** Vast.ai — NVIDIA RTX 3060 12GB, 125GB RAM

---

## 1. Experiment Overview

### 1.1 Objective

So sanh hieu suat cua mo hinh embedding (BAAI/bge-base-en) duoc fine-tune trong 4 che do:

1. **Pretrained** — Mo hinh goc, khong fine-tune
2. **Baseline** — Fine-tune FedAvg 25 rounds, DP bat nhung sigma=0.1 (khong co formal guarantee)
3. **DP-FedRAG (eps=8)** — Fine-tune voi formal (epsilon=8.0, delta=1e-5)-DP, sigma=1.67
4. **DP-FedRAG (eps=20)** — Fine-tune voi formal (epsilon=20.0, delta=1e-5)-DP, sigma=0.81

### 1.2 Training Configuration

```mermaid
graph LR
    subgraph "Data Pipeline"
        A[new_select_data.json<br/>20,016 records] --> B[FEDRAG Dataset]
        B --> C[IDPartitioner<br/>5 clients]
    end

    subgraph "FedAvg Training"
        C --> D[Client 0<br/>11 steps]
        C --> E[Client 1<br/>16 steps]
        C --> F[Client 2<br/>22 steps]
        C --> G[Client 3<br/>16 steps]
        C --> H[Client 4<br/>50 steps]
        D & E & F & G & H --> I[Server<br/>Aggregate]
        I -->|25 rounds| D
    end

    subgraph "Output"
        I --> J[x-model.bin<br/>418MB]
    end
```

| Parameter | Baseline | DP (eps=8) | DP (eps=20) |
|---|---|---|---|
| Algorithm | `fedrag.py` | `fedrag_dp.py` | `fedrag_dp.py` |
| Model | BGE-base-en (109M) | BGE-base-en (109M) | BGE-base-en (109M) |
| Rounds (T) | 25 | 25 | 25 |
| Clients (K) | 5 | 5 | 5 |
| Batch size | 8 | 8 | 8 |
| Learning rate | 1e-5 | 1e-5 | 1e-5 |
| **Clip norm (C)** | Fixed 1.0 | Adaptive (gamma=0.5) | Adaptive (gamma=0.5) |
| **Noise multiplier (sigma)** | 0.1 | **1.6701** | **0.8118** |
| **Privacy guarantee** | None (eps~2505) | **(eps=8.0, delta=1e-5)-DP** | **(eps=20.0, delta=1e-5)-DP** |
| **Training time** | 1,023s (~17 min) | 4,284s (~71 min) | 4,309s (~72 min) |

### 1.3 DP Calibration

```mermaid
graph TD
    subgraph "Calibration: eps=8"
        A1["Target: eps=8.0"] --> B1["sigma = 1.6701"]
        B1 --> C1["Verified: eps=8.009"]
    end

    subgraph "Calibration: eps=20"
        A2["Target: eps=20.0"] --> B2["sigma = 0.8118"]
        B2 --> C2["Verified: eps=20.000"]
    end

    D["RDP Accountant<br/>q=0.6, delta=1e-5, T=25"] --> A1
    D --> A2
```

**Sigma Calibration Table (T=25, q=0.6, delta=1e-5):**

| Target eps | Calibrated sigma | Verified eps | Privacy Level |
|---|---|---|---|
| eps=50 | 0.4904 | 50.007 | RELAXED |
| eps=30 | 0.6409 | 30.005 | RELAXED |
| **eps=20** | **0.8118** | **20.000** | **RELAXED** |
| eps=15 | 0.9827 | 15.005 | MODERATE |
| eps=10 | 1.3741 | 9.999 | MODERATE |
| **eps=8** | **1.6701** | **8.009** | **MODERATE** |
| eps=5 | 2.5733 | 4.991 | GOOD |
| eps=3 | 4.1602 | 3.003 | GOOD |

---

## 2. Evaluation Setup

- **Test set:** 200 samples (random seed=42) from `new_select_data.json`
- **Companies:** 3 unique companies in test set
- **Metrics:**
  - **Hit@k:** Exact match in top-k retrieved results
  - **F1@k:** Harmonic mean of Precision@k and Recall@k (company-level relevance)
  - **NDCG@k:** Normalized Discounted Cumulative Gain
  - **MRR:** Mean Reciprocal Rank
  - **Cosine Similarity Gap:** Correct pairs vs incorrect pairs

### Relevance Definition

```mermaid
graph LR
    Q["Query: q_i"] --> R1["r_i exact match = Relevant"]
    Q --> R2["r_j same company = Relevant"]
    Q --> R3["r_k different company = Not Relevant"]

    style R1 fill:#4CAF50,color:#fff
    style R2 fill:#8BC34A,color:#fff
    style R3 fill:#f44336,color:#fff
```

---

## 3. Results

### 3.1 Retrieval Accuracy (Hit@k)

| Metric | Pretrained | Baseline | DP (eps=8) | DP (eps=20) | Retention eps=20 |
|---|---|---|---|---|---|
| **Hit@1** | 84.50% | 84.50% | 4.50% | **60.00%** | **71.0%** |
| **Hit@3** | 97.50% | 97.50% | 8.50% | **75.50%** | **77.4%** |
| **Hit@5** | 98.50% | 98.50% | 11.00% | **82.00%** | **83.2%** |
| **Hit@10** | 99.00% | 99.00% | 15.50% | **86.50%** | **87.4%** |

```mermaid
xychart-beta
    title "Hit@k Comparison (4 Models)"
    x-axis ["Hit@1", "Hit@3", "Hit@5", "Hit@10"]
    y-axis "Accuracy (%)" 0 --> 100
    bar [84.5, 97.5, 98.5, 99.0]
    bar [84.5, 97.5, 98.5, 99.0]
    bar [4.5, 8.5, 11.0, 15.5]
    bar [60.0, 75.5, 82.0, 86.5]
```

### 3.2 F1 Score @ k

| Metric | Pretrained | Baseline | DP (eps=8) | DP (eps=20) | Retention eps=20 |
|---|---|---|---|---|---|
| **F1@1** | 2.80% | 2.80% | 1.02% | **2.13%** | 76.1% |
| **F1@3** | 5.30% | 5.35% | 3.04% | **4.31%** | 80.6% |
| **F1@5** | 7.36% | 7.42% | 5.07% | **6.33%** | **85.4%** |
| **F1@10** | 12.44% | 12.54% | 9.30% | **11.04%** | **88.0%** |

```mermaid
xychart-beta
    title "F1@k Comparison (4 Models)"
    x-axis ["F1@1", "F1@3", "F1@5", "F1@10"]
    y-axis "F1 Score (%)" 0 --> 15
    bar [2.80, 5.30, 7.36, 12.44]
    bar [2.80, 5.35, 7.42, 12.54]
    bar [1.02, 3.04, 5.07, 9.30]
    bar [2.13, 4.31, 6.33, 11.04]
```

### 3.3 NDCG @ k

| Metric | Pretrained | Baseline | DP (eps=8) | DP (eps=20) | Retention eps=20 |
|---|---|---|---|---|---|
| **NDCG@5** | 0.8790 | 0.8787 | 0.7875 | **0.8637** | **98.3%** |
| **NDCG@10** | 0.8605 | 0.8599 | 0.7764 | **0.8467** | **98.5%** |

### 3.4 MRR & Cosine Similarity

| Metric | Pretrained | Baseline | DP (eps=8) | DP (eps=20) |
|---|---|---|---|---|
| **MRR** | 0.9108 | 0.9108 | 0.0917 | **0.6967** |
| **Sim (correct pairs)** | 0.8714 | 0.8690 | 0.8234 | **0.9298** |
| **Sim (incorrect pairs)** | 0.7283 | 0.7238 | 0.8064 | **0.8861** |
| **Sim Gap** | 0.1431 | 0.1452 | 0.0170 | **0.0437** |

---

## 4. Analysis

### 4.1 Architecture Overview

```mermaid
graph TB
    subgraph "Upstream: FedE Training"
        direction TB
        A["Raw Data<br/>20,016 QA pairs"] --> B["Federated Partitioning<br/>5 clients IID"]
        B --> C["FedAvg + DP-SGD<br/>25 rounds"]
        C --> D["Trained Embedding Model<br/>BGE-base-en 109M params"]
    end

    subgraph "DP-SGD Pipeline Per Client"
        direction TB
        E["Forward Pass"] --> F["Backward Pass"]
        F --> G["Per-Sample Gradient Clipping<br/>clip to norm C"]
        G --> H["Gaussian Noise Injection<br/>N of 0 and sigma times C"]
        H --> I["Optimizer Step"]
    end

    subgraph "Server Aggregation"
        direction TB
        J["Receive Client Updates"] --> K["Clip Update Norms"]
        K --> L["Add Server Noise"]
        L --> M["Weighted Average"]
        M --> N["Broadcast to Clients"]
    end

    subgraph "Downstream: FinSafeRAG"
        direction TB
        O["User Query"] --> P["Embed with Trained Model"]
        P --> Q["FAISS Retrieval"]
        Q --> R["LLM Synthesis"]
        R --> S["Privacy Post-processing"]
    end

    D --> O
    C -.-> E
    I -.-> J
    N -.-> E
```

### 4.2 Key Findings

#### Finding 1: Baseline ~ Pretrained (No Improvement)

Baseline fine-tune **gan nhu khong thay doi** so voi pretrained:
- Hit@1: 84.50% vs 84.50% (identical)
- Weight L2 distance: chi 0.112

**Nguyen nhan:** Learning rate qua nho (1e-5) va chi 25 rounds chua du de thay doi model 109M params mot cach dang ke.

#### Finding 2: eps=8 — Severe Utility Degradation

```mermaid
graph LR
    subgraph "Root Cause: eps=8"
        A["sigma=1.67<br/>high noise"] --> B["Gradient norms ~2000-11000<br/>vs clip C ~1-5"]
        B --> C["Signal-to-noise ratio<br/>extremely low"]
        C --> D["Sim Gap: 0.017"]
        D --> E["Hit@1: 4.5%"]
    end
```

- **Hit@1:** 84.5% -> 4.5% (giam 94.7%)
- **MRR:** 0.91 -> 0.09 (giam 89.9%)
- **Sim Gap:** 0.145 -> 0.017 (giam 88.3%)
- **Nguyen nhan:** sigma=1.67 qua lon, noise lan at gradient signal

#### Finding 3: eps=20 — Significant Recovery

```mermaid
graph LR
    subgraph "Improvement: eps=20"
        A["sigma giam 2x<br/>1.67 to 0.81"] --> B["Noise giam 4x<br/>variance = sigma^2"]
        B --> C["Sim Gap tang 2.6x<br/>0.017 to 0.044"]
        C --> D["Hit@1 tang 13x<br/>4.5% to 60%"]
    end
```

- **Hit@1:** 4.5% -> **60.0%** (tang **13.3x**)
- **NDCG@5:** 0.788 -> **0.864** (giu **98.3%** utility)
- **F1@5:** 5.07% -> **6.33%** (giu **85.4%** utility)
- **MRR:** 0.092 -> **0.697** (tang **7.6x**)

#### Finding 4: Privacy-Utility Trade-off Summary

```mermaid
quadrantChart
    title Privacy-Utility Trade-off
    x-axis "Weak Privacy" --> "Strong Privacy"
    y-axis "Low Utility" --> "High Utility"
    quadrant-1 "Ideal Zone"
    quadrant-2 "High Utility No Privacy"
    quadrant-3 "Worst Case"
    quadrant-4 "Strong Privacy Low Utility"
    "Pretrained": [0.05, 0.85]
    "Baseline": [0.08, 0.85]
    "DP eps=20": [0.55, 0.60]
    "DP eps=8": [0.75, 0.05]
```

---

## 5. Training Dynamics

### 5.1 Baseline Training

| Round | Time | Observations |
|---|---|---|
| 1 | 46s | Initial — model loading |
| 2-25 | ~35s/round | Stable convergence |
| **Total** | **1,023s** | 25 rounds completed |

**Final Cosine Similarity Matrix (Round 25):**

```
         q0     q1     q2     q3     q4     q5     q6     q7
r0    [0.754] 0.701  0.694  0.695  0.711  0.713  0.738  0.704
r1     0.768 [0.917] 0.718  0.796  0.758  0.728  0.753  0.744
r2     0.737  0.683 [0.918] 0.699  0.748  0.641  0.778  0.781
r3     0.751  0.761  0.713 [0.860] 0.751  0.698  0.731  0.720
r4     0.734  0.731  0.721  0.755 [0.868] 0.663  0.783  0.752
r5     0.771  0.759  0.717  0.757  0.736 [0.887] 0.765  0.735
r6     0.769  0.769  0.787  0.794  0.795  0.683 [0.938] 0.807
r7     0.727  0.744  0.794  0.727  0.745  0.684  0.807 [0.897]
```

- Diagonal mean: **0.880**, Off-diagonal mean: **0.731**, Gap: **0.149**

### 5.2 DP Training (eps=8, sigma=1.67)

| Round | Time | Clip Norm C | Gradient Norm | eps spent |
|---|---|---|---|---|
| 1 | 172s | 0.95-1.13 | 2,065-2,473 | ~0.40 |
| 5 | 163s | 1.66-2.76 | 3,616-6,020 | ~3.24 |
| 10 | 170s | 2.11-2.66 | 4,603-5,800 | ~4.81 |
| 15 | 170s | 1.97-3.64 | 4,305-7,950 | ~6.05 |
| 20 | 170s | 3.10-4.62 | 6,770-10,100 | ~7.05 |
| 25 | 159s | 1.38-5.21 | 3,015-11,383 | **8.01** |
| **Total** | **4,284s** | | | |

### 5.3 DP Training (eps=20, sigma=0.81)

| Round | Time | Observations |
|---|---|---|
| 1 | 172s | Initial round |
| 2-25 | ~165s/round | Stable |
| **Total** | **4,309s** | 25 rounds completed |

### 5.4 Privacy Budget Over Rounds

```mermaid
xychart-beta
    title "Privacy Budget Consumption Over Rounds"
    x-axis ["R1", "R5", "R10", "R15", "R20", "R25"]
    y-axis "Epsilon Spent" 0 --> 21
    line [0.40, 3.24, 4.81, 6.05, 7.05, 8.01]
    line [1.00, 5.00, 10.00, 14.00, 17.50, 20.00]
```

---

## 6. Full Comparison Table

### 6.1 All Metrics

| Metric | Pretrained | Baseline | DP (eps=8) | DP (eps=20) |
|---|---|---|---|---|
| Hit@1 (%) | 84.50 | 84.50 | 4.50 | **60.00** |
| Hit@3 (%) | 97.50 | 97.50 | 8.50 | **75.50** |
| Hit@5 (%) | 98.50 | 98.50 | 11.00 | **82.00** |
| Hit@10 (%) | 99.00 | 99.00 | 15.50 | **86.50** |
| F1@1 (%) | 2.80 | 2.80 | 1.02 | **2.13** |
| F1@3 (%) | 5.30 | 5.35 | 3.04 | **4.31** |
| F1@5 (%) | 7.36 | 7.42 | 5.07 | **6.33** |
| F1@10 (%) | 12.44 | 12.54 | 9.30 | **11.04** |
| NDCG@5 | 0.879 | 0.879 | 0.788 | **0.864** |
| NDCG@10 | 0.861 | 0.860 | 0.776 | **0.847** |
| MRR | 0.911 | 0.911 | 0.092 | **0.697** |
| Sim Gap | 0.143 | 0.145 | 0.017 | **0.044** |
| Sim (correct) | 0.871 | 0.869 | 0.823 | **0.930** |
| Sim (incorrect) | 0.728 | 0.724 | 0.806 | **0.886** |
| Training Time | N/A | 1,023s | 4,284s | 4,309s |

### 6.2 Utility Retention (vs Baseline)

| Metric | DP eps=8 | DP eps=20 |
|---|---|---|
| **Hit@1** | 5.3% | **71.0%** |
| **Hit@5** | 11.2% | **83.2%** |
| **F1@5** | 68.4% | **85.4%** |
| **F1@10** | 74.2% | **88.0%** |
| **NDCG@5** | 89.6% | **98.3%** |
| **MRR** | 10.1% | **76.5%** |

```mermaid
xychart-beta
    title "Utility Retention: DP vs Baseline (%)"
    x-axis ["Hit@1", "Hit@5", "F1@5", "F1@10", "NDCG@5", "MRR"]
    y-axis "Retention (%)" 0 --> 100
    bar [5.3, 11.2, 68.4, 74.2, 89.6, 10.1]
    bar [71.0, 83.2, 85.4, 88.0, 98.3, 76.5]
```

### 6.3 Privacy-Utility Trade-off Analysis

| | eps=8 (sigma=1.67) | eps=20 (sigma=0.81) |
|---|---|---|
| **Privacy** | Strong formal guarantee | Moderate formal guarantee |
| **Hit@1** | 4.5% (unusable) | 60.0% (acceptable) |
| **NDCG@5** | 0.788 (89.6%) | 0.864 (98.3%) |
| **Sigma reduction** | baseline | **2.06x lower** |
| **Hit@1 improvement** | baseline | **13.3x better** |
| **Practical usability** | Not usable | **Production-viable** |

**Key Insight:** Giam sigma 2x (tu 1.67 xuong 0.81) mang lai cai thien **13x** cho Hit@1 vi noise variance giam theo **sigma^2** (tu 2.79 xuong 0.66 — giam 4.2x).

---

## 7. Conclusions

### 7.1 Summary

| Aspect | eps=8 | eps=20 |
|---|---|---|
| **Privacy guarantee** | (8.0, 1e-5)-DP | (20.0, 1e-5)-DP |
| **Hit@1** | 4.5% (unusable) | **60.0%** (acceptable) |
| **NDCG@5 retention** | 89.6% | **98.3%** |
| **F1@5 retention** | 68.4% | **85.4%** |
| **Training overhead** | 4.2x | 4.2x |
| **Recommendation** | Research only | **Production candidate** |

### 7.2 Key Takeaways

1. **eps=20 la cau hinh khuyen nghi** cho production: giu 98.3% NDCG@5 va 71% Hit@1 voi formal DP guarantee
2. **eps=8 qua strict** cho full-model fine-tuning 109M params: can DP-LoRA hoac model nho hon
3. **Baseline ~ Pretrained**: FedAvg voi lr=1e-5 va 25 rounds chua thay doi model dang ke
4. **Training overhead 4.2x** chap nhan duoc (17min vs 72min tren RTX 3060)

### 7.3 Future Improvements

| Strategy | Expected Impact | Complexity | Priority |
|---|---|---|---|
| **DP-LoRA** (fine-tune 0.5M vs 109M params) | Hit@1 > 80% at eps=8 | Medium | HIGH |
| **Increase rounds** (25 -> 100-200) | Better convergence | Low | MEDIUM |
| **Decrease learning rate** with warmup | Smoother training | Low | MEDIUM |
| **Ghost clipping** (Bu et al., NeurIPS 2022) | 3x faster training | Medium | LOW |
| **Pre-train on public data** then DP fine-tune | Less DP budget needed | High | HIGH |

### 7.4 Recommendation for Paper

Cho paper Q1, khuyen nghi trinh bay:
1. **Baseline** (no DP) lam upper-bound
2. **eps=8** cho thay thach thuc cua DP voi large models
3. **eps=20** cho thay cau hinh thuc te kha thi
4. **Privacy-utility curve** voi nhieu gia tri eps de minh hoa trade-off
5. **So sanh voi SOTA:** C-FedRAG, DP-RAG papers 2024-2025

---

## Appendix A: File Locations

| Item | Path |
|---|---|
| Baseline model (raw) | `FedE/x-model_2026-03-29_04-39-38.bin` |
| DP eps=8 model (raw) | `FedE/x-model_2026-03-29_05-52-40.bin` |
| DP eps=20 model (raw) | `FedE/x-model_2026-03-29_07-56-24.bin` |
| Baseline model (HF) | `x-model_baseline_converted/` |
| Baseline output log | `FedE/logs/baseline_output.log` |
| Baseline error log | `FedE/logs/baseline_error.log` |
| DP eps=8 output log | `FedE/logs/dp_output.log` |
| DP eps=8 error log | `FedE/logs/dp_error.log` |
| DP eps=20 output log | `FedE/logs/dp_eps20_output.log` |
| DP eps=20 error log | `FedE/logs/dp_eps20_error.log` |
| Eval script | `FedE/eval_compare.py` |
| Training guide | `FedE/TRAIN_GUIDE.md` |

## Appendix B: Reproduction

```bash
# 1. Clone and setup
git clone --branch trang/differential_privacy \
  https://github.com/ursuswh-metamorphic/SDA_SecureDataAlliance.git
cd SDA_SecureDataAlliance/FedE
pip install torch torchvision transformers scipy prettytable ujson pyyaml

# 2. Train baseline (no DP)
python main.py

# 3. Train DP eps=8
python main_dp.py

# 4. Train DP eps=20
python main_dp_eps20.py

# 5. Evaluate all models
python eval_compare.py
```

## Appendix C: References

1. Abadi et al., "Deep Learning with Differential Privacy," CCS 2016
2. McMahan et al., "Learning Differentially Private Recurrent Language Models," ICLR 2018
3. Mironov, "Renyi Differential Privacy," CSF 2017
4. Andrew et al., "Differentially Private Learning with Adaptive Clipping," NeurIPS 2021
5. Balle et al., "Privacy Amplification by Subsampling," NeurIPS 2018
6. Gopi et al., "Numerical Composition of Differential Privacy," NeurIPS 2021
7. Bu et al., "Automatic Clipping: DP Deep Learning Made Easier and Stronger," NeurIPS 2023
8. Dong et al., "Gaussian Differential Privacy," JRSS-B 2022
9. Noble et al., "Differentially Private Federated Learning on Heterogeneous Data," AISTATS 2022
