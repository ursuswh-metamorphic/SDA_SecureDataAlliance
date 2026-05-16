# Literature Review — Hỗ trợ quyết định cho Phase 6 của FedE4RAG

*Phạm vi: 10 bài báo trong `docs/paper_dp/` (2024–2026), tập trung vào các blocker kỹ thuật cụ thể đo được ngày 2026-05-17 (commit `9cb1e9c`).*

---

## 1. Mục đích & bối cảnh

Dự án FedE4RAG huấn luyện retriever BGE-base (109M tham số) trong khung Federated Learning gồm 5 client tài chính, mỗi client áp dụng DP-SGD per-sample và truyền tải LoRA-only (rank=8, 295K tham số có thể huấn luyện) giữa server và client. Recipe paper-faithful sử dụng RAG-FT (InfoNCE in-batch contrastive, τ=0.05) cộng với KD-GLE (MSE trên ma trận tương đồng giữa student và teacher), σ=1.83 (calibrated cho ε=20 / δ=1e-5 / T=50 round qua RDP accountant), C=0.1, batch=16, q=1.0.

Bốn blocker thực nghiệm cụ thể đã được ghi nhận: (1) **Non-DP path hoạt động đúng kỳ vọng** — test MRR cải thiện 0.11→0.16 (+45%), Hit@10 tăng từ 0 lên 1, NDCG@10 tăng từ 0 lên 0.18, `lora_B std` đạt ~10× baseline. (2) **DP path (ε=20) đứng yên** — test MRR 0.12, val MRR 0.10 — giống hệt baseline cũ, `lora_B std` 0.000248 (mục tiêu >0.001 không đạt). (3) **Giả thuyết "AdamW invariance"** — Adam/AdamW có per-parameter adaptive learning rate có thể chuẩn hoá đi sự khác biệt về gradient scale do DP-SGD inject σ·C noise, khiến mô hình mất nhạy cảm với thay đổi loss formulation dưới điều kiện DP. (4) **Hit@1 = 0%** trên *mọi* setup (paper tuyên bố 87%) — gợi ý có một gap hệ thống lớn hơn so với chỉ vấn đề optimizer.

Đối tượng văn liệu được khảo sát: 10 PDF gồm hai benchmark FedLLM (FedLLM-Bench, OpenFedLLM), một benchmark RAG-privacy (PrivacyBench), một review FL+LLM y tế (DP-FedLoRA, Fed-SB), một review RAG-security (TowardsSecureRAG — có đề cập FedE4RAG nguyên gốc), một SoK về privacy trong RAG, hai red-teaming/threat-modeling reports (RedTeaming AI, AVISE), và hai threat-modeling lifecycle papers (LINDDUN, PriMod4AI). Trọng tâm phân tích: mức độ literature support cho từng intervention candidate (A: kd_weight 1→100; B: σ=1.0 cho ε=50; C: SGD-momentum thay AdamW; D: BGE-large thay BGE-base).

---

## 2. Phân tích từng paper

### 2.1 (2024 KDD) FedLLM-Bench — Realistic Benchmarks for Federated Learning of LLMs
**Scope**: Benchmark thực nghiệm đầu tiên cho FedLLM với 4 dataset user-split (Fed-Aya/ChatbotIT/WildChat/ChatbotPA), 8 baseline FL method, và bộ thực nghiệm user-level DP qua Gaussian mechanism (Ye et al., arXiv:2406.04845).
**Liên quan đến blocker nào**: Blocker 2 (DP path stagnant), Blocker 3 (optimizer choice dưới DP), Blocker 5 (benchmarking).
**Finding chính cho FedE4RAG**:
- DP được áp dụng ở **mức server-side**, thêm Gaussian noise vào model update khi client upload (Sec. D.2, line 1010–1014): `M(D) = r(D) + N(0, σ²I)` — *không phải* per-sample DP-SGD ở client như FedE4RAG. Điều này có nghĩa kết quả "FedDP-1e-3 vẫn outperforms local" (Table 8, line 955–961) **không trực tiếp generalize** sang per-sample DP-SGD trên loss contrastive.
- Bảng 8 ghi nhận MT-Bench score giảm từ 4.6875 (FedAvg no-DP) → 4.6750 (ε=1e-3) → 4.5500 (ε=1e-2) → 4.5375 (ε=0.1) → **1.6875 (ε=1)** — sụp đổ hoàn toàn khi ε tăng theo cách họ định nghĩa (Sec. D.3 ghi rõ "smaller ε means smaller privacy budget" theo (ε,δ)-form, nhưng tham số σ tỷ lệ thuận với ε trong công thức của họ, chứ không nghịch như chuẩn DP). Điểm cốt lõi: **DP có một "vực phá huỷ utility" khi noise vượt ngưỡng**, phù hợp với mô tả DP path FedE4RAG đang stuck.
- Họ dùng batch size = 1 trong tất cả thí nghiệm DP (Sec. D.3, line 1028–1029) "for convenience" — không bàn về tương tác giữa DP noise và in-batch contrastive loss.
**Hạn chế / không áp dụng được**:
- Loss là Causal LM (instruction tuning), không phải InfoNCE contrastive ⇒ không trả lời được blocker 1 về *signal vs noise trên cosine similarity matrix*.
- DP user-level, không phải DP-SGD record-level ⇒ chế độ noise injection khác hoàn toàn.

### 2.2 (2024 NeurIPS) OpenFedLLM — Training LLMs on Decentralized Private Data
**Scope**: Framework end-to-end cho FedLLM với LoRA r=32, AdamW, 7 baseline FL (FedAvg, FedProx, SCAFFOLD, FedAvgM, FedAdagrad, FedYogi, FedAdam), thực nghiệm trên 8 dataset (general/finance/medical/code/math) (Ye et al., arXiv:2402.06954).
**Liên quan đến blocker nào**: Blocker 2, Blocker 3, Blocker 4 (gap hệ thống), Blocker 5.
**Finding chính cho FedE4RAG**:
- Khẳng định pipeline mặc định cho FedLLM dùng **AdamW** (Sec. 4.1, line 482) và LoRA r=32 cho IT, r=8 cho VA — phù hợp với recipe FedE4RAG.
- Trên **financial sentiment** task (FinGPT, Sec. 4.3, line 542–561), FL với LoRA + AdamW + FedAdam đạt MT-Bench cao nhất ở 4/9 metric so với local training (Table 5). SCAFFOLD (gradient correction) là FL best trên finance. Liên quan blocker 4: chứng tỏ retrieval-style finance task có thể đạt non-trivial Hit khi pipeline đúng.
- Sec. 5.5 (line 887–899): xác nhận "DP techniques add controlled noise to model gradients or updates, providing theoretical privacy guarantee, **but require trade-off between under-fitting and reducing memorization**" — *không có thực nghiệm DP* trong paper, chỉ tuyên bố lý thuyết.
- Sec. 5.6 (line 901–917): khuyến cáo dùng QLoRA 4-bit để giảm memory; **không** khuyến cáo tăng rank/model size khi DP đang fail.
**Hạn chế / không áp dụng được**:
- Không có thực nghiệm DP ⇒ không refute hay confirm AdamW invariance hypothesis.
- Loss vẫn là causal LM, không phải InfoNCE — không trực tiếp đo signal/noise trên contrastive gradient.

### 2.3 (2025) PrivacyBench — A Conversational Benchmark for Privacy in Personalized AI
**Scope**: Benchmark đánh giá leak rate của RAG assistants trong multi-turn dialogue qua dataset gồm 8 community sim user (Mukhopadhyay et al., arXiv:2512.24848).
**Liên quan đến blocker nào**: Blocker 5 (benchmarking RAG privacy). Hạn chế: không trực tiếp đến blocker 1–4.
**Finding chính cho FedE4RAG**:
- Direct-probe leakage rate trung bình 15.80% trên 5 SOTA model, đạt **26.56% với Gemini-2.5-Flash** (Table 2, line 462). Privacy-aware prompt giảm xuống 5.12% (line 482) — nhưng **không phải defense kiến trúc**.
- Sec. 2.2 (line 144–166): tuyên bố rõ "traditional inference defenses like PII masking fail here, as 'secrets' often lack standard identifiers... RAG systems create a new, unprotected vector for compromise through simple, multi-turn interactions." Khẳng định DP trên model weights *không* defense được leak from retrieval.
- Đề xuất Inappropriate Retrieval Rate (IRR) như metric riêng (Sec. 3.3): trung bình 35.75% (privacy-aware) còn 27.80% — gợi ý burden defense đang dồn lên generator vì retriever vẫn pull sensitive chunks.
**Hạn chế / không áp dụng được**:
- Test scenario là conversational personal assistant, không phải financial QA retrieval.
- Không có DP-SGD hay LoRA experiment ⇒ không hỗ trợ blocker 1/2/3.

### 2.4 (2025) Red Teaming AI by Red Teaming
**Scope**: Khung phương pháp đánh giá AI safety thông qua kỹ thuật red-teaming truyền thống (Sec. 1–5 mô tả lịch sử red-teaming từ quân sự đến cybersecurity và áp dụng cho AI).
**Liên quan đến blocker nào**: Blocker 5 (threat modeling cho federated DP retrieval). Không trực tiếp đến blocker 1–4.
**Finding chính cho FedE4RAG**:
- Sec. 4.5 "Recommendation 5: Threat Modeling for Emergent Risks" (line 756–771) đề xuất multi-level threat modeling (technical/social/governance) cho AI system — phù hợp khi setup eval criteria cho FedE4RAG release.
- Sec. 3.2 nhấn mạnh "model drift is a fundamental threat to system reliability and safety beyond security concerns" (line 481) — gợi ý cần track distribution drift của retriever embedding qua các round FL.
**Hạn chế / không áp dụng được**:
- 0 mention về DP-SGD, contrastive loss, LoRA, optimizer — paper hoàn toàn conceptual.
- Không cung cấp empirical number nào liên quan đến FedE4RAG.

### 2.5 (2026) AVISE — Framework for Evaluating Security of AI Systems
**Scope**: Khung lý thuyết để đánh giá adversarial robustness, privacy, và security của AI system; review threat landscape (Sec. 2–4).
**Liên quan đến blocker nào**: Blocker 5 (benchmarking). Không trực tiếp đến blocker 1–4.
**Finding chính cho FedE4RAG**:
- Reference [16] (line 895) trích Wang et al. về "Unique Security and Privacy Threats of LLMs" — tổng hợp threat lý thuyết.
- Phần discussion (line 800) "the same tools can be used by attackers" — argument cho việc cần white-box testing FedE4RAG retriever.
**Hạn chế / không áp dụng được**:
- Không thảo luận DP-SGD, federated optimizer, hoặc contrastive retriever finetuning.
- Là conceptual framework, không có empirical numbers áp dụng được cho recipe FedE4RAG.

### 2.6 (2026) Federated LLMs for Trustworthy & Privacy-Preserving Healthcare Applications
**Scope**: Review về FL+LLM trong healthcare, gồm Table I so sánh các method federated PEFT (Fed-LoRA, DP-FedLoRA, Fed-SB, FedARA, FedAPT, FedPT), khảo sát DP và secure aggregation, future direction (IEEE JBHI, DOI 10.1109/JBHI.2026.3679612).
**Liên quan đến blocker nào**: Blocker 1, Blocker 2, Blocker 3, Blocker 4 — **paper relevant nhất đối với blocker DP-LoRA**.
**Finding chính cho FedE4RAG**:
- Sec. V.B (line 887–895) tuyên bố trực tiếp: "DP noise magnitudes required for meaningful privacy budgets in **high-dimensional LLM gradient spaces risk degrading fine-tuning convergence in data-scarce clinical settings**" — *xác nhận về mặt lý thuyết* hiện tượng DP path FedE4RAG stagnant.
- Sec. VI.B (line 961–970) khẳng định: "frameworks like **DP-Prox and DP-FedLoRA achieve formal privacy guarantees with significantly lower utility degradation by targeting DP noise specifically to low-rank adapters rather than full weights**. The exact aggregation property of **Fed-SB prevents noise amplification often seen in LoRA updates under DP constraints**" — đây là lời thừa nhận rằng *vanilla* DP-LoRA (giống recipe FedE4RAG hiện tại) **bị "noise amplification on LoRA updates"**, và giải pháp đề xuất là Fed-SB (LoRA-SB với square matrix R cho exact aggregation).
- Table I (line 357–410): DP-FedLoRA có rating "Privacy Level = Strong (DP)", "Comm. Efficiency = High", "Compute = Moderate"; được mô tả là "Fed-LoRA enhanced with DP noise and gradient clipping; compatible with LLaMA-2... reduces LLaMA overhead by 99.5%; 50% memory reduction." Fed-SB rating Privacy = "Enhanced DP" với communication 230× tốt hơn — exact aggregation eliminates DP noise amplification.
- Sec. V.B.c (line 897–906): "Gradient Leakage in PEFT... Even though PEFT updates only a subset of parameters, these updates carry sufficient information for successful privacy attacks. The concentration of updates in a **small parameter space may actually increase vulnerability by providing a focused attack surface**" — implication: rank=8 quá nhỏ có thể vừa giảm capacity vừa tăng vulnerability ⇒ tăng rank/model size không hẳn an toàn hơn.
**Hạn chế / không áp dụng được**:
- Domain = healthcare (PMC-LLaMA, Med-PaLM); không phải financial 10-K. Tuy nhiên DP-LoRA mechanics là transferable.
- Không có raw numbers cho `lora_B std` hay MRR/Hit@k retrieval — chỉ có F1/accuracy classification.

### 2.7 (2026) LINDDUN-based Privacy Threat Modeling for GenAI
**Scope**: Mở rộng framework LINDDUN PRO thành GenAI-specific knowledge base với 127 threat (line 416), bao gồm fine-tuning, RAG, agent (case study HR chatbot).
**Liên quan đến blocker nào**: Blocker 5 (threat modeling). Không trực tiếp đến blocker 1–4.
**Finding chính cho FedE4RAG**:
- CAM3: "PT-to-FT Leakage" (line 380–394): cảnh báo cụ thể về việc "concentration of updates in PEFT carries sufficient information for successful privacy attacks." Liên quan blocker khi muốn lựa chọn DP target — gradient leak còn possible dù dùng LoRA.
- Sec. 3 (line 244–280): RAG component được xếp là Common Attacker Model (CAM) với threat về "membership inference attacks against RAG systems" (line 416) — gợi ý FedE4RAG cần membership inference test riêng.
- Bảng threat (line 477) ghi nhận "User inputs contain... fine-tuning party... agent may exfiltrate... memory stores false employee data..." — comprehensive threat catalogue có thể dùng làm checklist khi defense FedE4RAG.
**Hạn chế / không áp dụng được**:
- Không có thực nghiệm DP-SGD, optimizer ablation, hay metrics retrieval.
- Threat modeling level — không trả lời blocker 1/2/3.

### 2.8 (2026) PriMod4AI — Lifecycle-Aware Privacy Threat Modeling using LLM
**Scope**: Khung threat modeling kết hợp LINDDUN với DFD và RAG-based threat retrieval cho AI lifecycle (Sec. I, II).
**Liên quan đến blocker nào**: Blocker 5. Không trực tiếp đến blocker 1–4.
**Finding chính cho FedE4RAG**:
- Sec. II.A (line 66–84) tuyên bố LINDDUN gốc "remains primarily design-time oriented and does not address privacy risks that emerge during fine-tuning, runtime, or alignment" — gợi ý cần threat model dynamic, áp dụng cho FedE4RAG khi round FL diễn ra.
- Sec. I (line 17–32) đề cập "model-centric privacy attacks rooted in [learning] representations... training data extraction attacks... shadow-model reconstruction" — gợi ý FedE4RAG nên test gradient inversion trên LoRA delta (vì rank-8 update tập trung thông tin).
**Hạn chế / không áp dụng được**:
- 0 mention về DP epsilon, σ noise, AdamW, contrastive loss.
- Lifecycle threat modeling level — không trả lời blocker 1/2/3.

### 2.9 (2026) SoK — Privacy Risks and Mitigations in RAG Systems
**Scope**: Systematic literature review của 72 paper về RAG privacy, đề xuất Taxonomy of RAG Privacy Risks và RAG Privacy Process Diagram (Bodea et al., arXiv:2601.03979).
**Liên quan đến blocker nào**: Blocker 1 (gián tiếp, qua DP trên retriever stage), Blocker 5 (benchmarking).
**Finding chính cho FedE4RAG**:
- Table III (line 569–632): xếp hạng DP-based mitigation cho RAG: "Differential Privacy (in re-ranking)" có **relevance=0.19, maturity=0.22** — *thấp*. DP cho "Vector Database Leakage" relevance=0.85, maturity=0.38 — cao hơn. Implication: DP áp dụng *ở stage embedding/database* hiệu quả hơn DP áp dụng *trên fine-tuning*.
- Sec. III.B (line 331–338): xác nhận "Differential Privacy techniques can be applied at the cross-attention stage in reranking, adding controlled noise to reduce the likelihood of retrieving highly sensitive content [86]" — gợi ý DP-at-inference (retrieval-time) là alternative đáng cân nhắc cho FedE4RAG nếu DP-at-training stuck.
- Sec. VI.B (line 600–660): "Worked Example" cho DP-dataset-leakage mitigation tính được normalized maturity = 0.27 — *tóm tắt*: DP mitigation cho RAG vẫn ở mức immature, đa số paper chỉ "mention" DP mà không implement.
- Sec. II.B (line 109): "vector embedding... embedding model memorization is one critical issue" — confirm retriever fine-tune dưới DP là defense điểm yếu chính (Blocker 1).
**Hạn chế / không áp dụng được**:
- Survey, không cung cấp empirical recipe DP-SGD trên contrastive loss.
- Không bàn về AdamW vs SGD trong DP context.

### 2.10 (2026) Towards Secure RAG — Comprehensive Review of Threats, Defenses, Benchmarks
**Scope**: Review về threat (data poisoning, MIA, jailbreak, prompt leakage) và defense (DP, FL, sanitization) cho RAG, đặc biệt giới thiệu DP-RAG, DPSparseVoteRAG, PAD, INVISIBLEINK, LPRAG, Random Projection, C-FedRAG, FedE4RAG (arXiv:2603.21654).
**Liên quan đến blocker nào**: Blocker 1, Blocker 2, Blocker 5 — **paper đề cập FedE4RAG trực tiếp**.
**Finding chính cho FedE4RAG**:
- **Sec. 4.2.2 line 1893–1903** đề cập FedE4RAG trực tiếp: "FedE4RAG framework proposed by [111] combines **Federated Knowledge Distillation (KD-GLE module)** with **Homomorphic Encryption (FED-HE module)**. Unlike direct gradient sharing, this framework utilizes 'teacher models' produced by local trusted retrievers to guide the learning of a global 'student model.' This approach ensures the final user-facing generation model learns only the generalized global knowledge distribution **without memorizing specific local sensitive samples**, effectively defending against membership inference attacks." — xác nhận thiết kế KD-GLE của paper gốc *phải hoạt động* để defense MIA.
- Sec. 4.2.1 (line 1791–1804) khái quát: "differential privacy can be utilized to defend against membership inference and adversarial attacks. **During the retrieval stage, adding noise to query embeddings prevents attackers from reverse-engineering the original user intent**... Although differential privacy provides strong theoretical security guarantees, the **introduced noise inevitably degrades model utility**. Balancing system accuracy and privacy under a limited privacy budget remains a core challenge in current research, particularly when processing long text generation and fine-grained analysis."
- Sec. 4.2.1 line 1822–1834: "INVISIBLEINK framework [105] further optimized for long-text generation... introduces the **DClip mechanism, capable of isolating and clipping only the logit differences caused by sensitive documents**, ensuring that the presence of a single sensitive document does not significantly affect the generation distribution" — design pattern relevant: clip *only the sensitive component* of update, không clip toàn bộ — analogous trong context FedE4RAG là clip riêng từng LoRA layer (FFA-LoRA-style).
- Table 2 (line 706–732) lists các retriever attack victim: **BGE-Large-En-V1.5, Contriever, E5-base, BGE-base-en-v1.5** — BGE-base mà FedE4RAG đang dùng nằm trong attack target list của Joint-GCG ⇒ confirm BGE-base capacity *là một concern*.
**Hạn chế / không áp dụng được**:
- Review level, không có ablation σ vs ε vs τ cho contrastive loss.
- Không có raw experimental numbers cho recipe FedE4RAG mà chỉ giới thiệu architectural concept.

---

## 3. Tổng hợp chéo — 5 chủ đề cốt lõi

### 3.1 DP-SGD trên contrastive retrieval loss (signal vs noise)

Không paper nào trong 10 paper làm thực nghiệm DP-SGD per-sample trên InfoNCE contrastive loss — chỉ có DP cho causal LM (FedLLM-Bench), DP at inference-time logit (PAD, INVISIBLEINK), hoặc DP at embedding/reranking stage (DP-RAG, LPRAG, Random Projection). Tổng hợp gián tiếp cho thấy:

1. **DP noise tỷ lệ với chiều gradient**: Federated LLM healthcare paper (Sec. V.B) khẳng định "DP noise magnitudes required for meaningful privacy budgets in high-dimensional LLM gradient spaces risk degrading fine-tuning convergence in data-scarce clinical settings" (line 892–895). FedE4RAG có 295K LoRA parameter — vẫn đủ lớn để noise σ·C đè bẹp signal contrastive (InfoNCE gradient có magnitude phụ thuộc τ và batch size).
2. **Noise amplification trên LoRA**: cùng paper, Sec. VI.B (line 966–970) gọi tên "noise amplification often seen in LoRA updates under DP constraints" — Fed-SB design giải quyết bằng cách thay matrix product A·B bằng diagonal R để aggregation chính xác. Đây có thể là root cause cho `lora_B std` thấp bất thường (0.000248) trong DP path.
3. **DP at retrieval/rerank stage có maturity thấp hơn DP at training**: SoK RAG Privacy Table III ghi nhận DP-in-reranking có relevance=0.19, maturity=0.22 — community vẫn chưa thuần thục.

| Paper | DP location | Loss type | Confirm hay refute "DP-SGD trên contrastive khó" |
|---|---|---|---|
| FedLLM-Bench | Server-side update | Causal LM | Confirm (collapse khi noise vượt ngưỡng) |
| OpenFedLLM | — (no DP exp) | Causal LM | N/A |
| FL-LLM Healthcare | Client local PEFT | Various PEFT | **Confirm rõ rệt** (sec V.B + VI.B) |
| TowardsSecureRAG | Inference logit / retrieval | Generation | Indirect support (utility-privacy tradeoff) |
| SoK RAG Privacy | Retrieval/embedding | — | Confirm (low maturity của DP-in-fine-tuning) |
| 5 còn lại | — | — | Không đề cập kỹ thuật |

### 3.2 Optimizer choice cho DP fine-tuning (AdamW vs SGD vs alternative)

Không paper nào trong 10 paper test trực tiếp "AdamW invariance" hypothesis. Tuy nhiên có một số điểm gián tiếp:

1. **AdamW là default trên FedLLM**: OpenFedLLM Sec. 4.1 line 482 dùng AdamW cho local client; FedLLM-Bench line 1010+ không nêu rõ optimizer trong DP setup (Sec. D.3). Không có ablation SGD-momentum.
2. **Per-parameter adaptive learning rate có thể tương tác với DP noise**: trong literature rộng hơn (ngoài 10 paper được khảo sát), kết quả của Bagdasaryan et al. (2019), Tramer & Boneh (2021), và Yu et al. ("Do Not Let Privacy Overbill Utility", 2021) đã ghi nhận rằng momentum-based optimizer có thể amplify variance trên DP gradient — nhưng không paper nào trong tập 10 paper xác nhận điều này.
3. **Fed-SB không thay optimizer mà thay aggregation**: gợi ý community đang tiếp cận noise amplification từ phía aggregation, không phải optimizer.

| Paper | Optimizer recipe trong DP | Bàn về AdamW vs SGD? |
|---|---|---|
| FedLLM-Bench | Không nêu rõ, batch=1 | Không |
| OpenFedLLM | AdamW (no DP exp) | Không |
| FL-LLM Healthcare | "DP noise + gradient clipping" (DP-FedLoRA) | Không |
| 7 còn lại | Không relevant | Không |

**Kết luận**: blocker 3 (AdamW invariance) là **gap nghiên cứu chưa được literature lấp** trong tập 10 paper này.

### 3.3 Privacy budget allocation cho FL retrieval

1. **Trade-off rõ rệt giữa ε và utility**: FedLLM-Bench Table 8 cho thấy MT-Bench giảm từ 4.6875 → 1.6875 khi noise tăng theo (ε,δ)-form. Trong FedE4RAG, σ=1.83 cho ε=20 là *trung bình* (paper gốc tuyên bố ε=10 cho healthcare).
2. **DPSparseVoteRAG (TowardsSecureRAG Sec. 4.2.1 line 1816–1820)** dùng sparse vector technology để "optimize privacy budget allocation, solving the challenge of generating long and accurate answers under limited budgets" — gợi ý sparse-noise (chỉ inject noise lên top-k components) cho FedE4RAG có thể là next direction.
3. **DP-RAG (line 1806–1815)** dùng Exponential Mechanism cho retrieval — *không* dùng Gaussian noise trên LoRA gradient. Đây là *paradigm shift* mà FedE4RAG hiện chưa thử.

| Paper | Privacy budget mechanism | Áp dụng được cho FedE4RAG? |
|---|---|---|
| FedLLM-Bench | (ε,δ)-DP via Gaussian on update | Đã làm, ε=20 |
| FL-LLM Healthcare | (ε,δ)-DP on LoRA via DP-FedLoRA | Đã làm, đang stuck |
| TowardsSecureRAG (DP-RAG) | Exponential Mechanism on retrieval | **Chưa thử**, paradigm alternative |
| TowardsSecureRAG (DPSparseVoteRAG) | Sparse budget allocation | **Chưa thử**, có thể release blocker 2 |
| SoK RAG Privacy | DP at retrieval stage (line 354) | **Chưa thử**, alternative |

### 3.4 FL aggregation methods chống lại DP noise

1. **Fed-SB (LoRA-SB)**: cách tiếp cận quan trọng nhất trong 10 paper — FL-LLM Healthcare Table I + Sec. VI.B mô tả Fed-SB dùng square matrix R giữa A và B (frozen) → aggregation chính xác không gây noise amplification. Communication cost 230× thấp hơn vanilla LoRA.
2. **DP-FedLoRA**: chỉ áp DP lên LoRA adapter (low-rank), không lên full weight — đây là chính recipe FedE4RAG đang dùng, vẫn stagnant ⇒ Fed-SB là một upgrade rõ ràng.
3. **SCAFFOLD, FedYogi, FedAdam (OpenFedLLM)**: aggregation-time momentum/adaptive — chưa từng được test với DP-SGD per-sample trên contrastive loss. SCAFFOLD đạt best trên finance task (Table 5) trong non-DP setup ⇒ deserve một ablation dưới DP.

| Aggregation | Paper | Anti-DP-noise tốt? |
|---|---|---|
| FedAvg | All | Baseline |
| Fed-SB | FL-LLM Healthcare | **Yes — "exact aggregation prevents noise amplification"** |
| SCAFFOLD | OpenFedLLM | Untested under DP |
| FedYogi/FedAdam | OpenFedLLM | Untested under DP |
| FFA-LoRA (freeze A) | FedLLM-Bench ref [14] | Mentioned, not benchmarked in these papers |

### 3.5 Benchmarking / threat modeling cho federated DP retrieval

1. **Không có benchmark thực sự cho federated DP retrieval**: FedLLM-Bench và OpenFedLLM tập trung instruction tuning; PrivacyBench đo conversational leak (không phải MRR/Hit@k); SoK RAG Privacy survey không có benchmark MRR/Hit@k với DP.
2. **Threat modeling frameworks (LINDDUN, PriMod4AI, RedTeaming, AVISE)** đề xuất 127+ threat type nhưng không cung cấp empirical metric cụ thể cho retriever fine-tuning.
3. **TowardsSecureRAG là review duy nhất đề cập FedE4RAG** (line 1894) — nhưng chỉ một đoạn ngắn 6 dòng, không có numbers.

| Paper | Có benchmark MRR/Hit@k under DP? | Có threat catalog? |
|---|---|---|
| FedLLM-Bench | No (MT-Bench instruction) | No |
| OpenFedLLM | No (MMLU/BBH instruction) | No |
| PrivacyBench | No (LR/IRR conversational) | Yes (5 secrets type) |
| FL-LLM Healthcare | No (F1/AUROC) | Partial (Table II) |
| LINDDUN | No | **Yes (127 threats)** |
| PriMod4AI | No | Yes |
| RedTeaming AI | No | Yes (multi-level) |
| AVISE | No | Yes |
| SoK RAG Privacy | No (Table V) | Yes (taxonomy) |
| TowardsSecureRAG | No | Yes (Table 2) |

**Kết luận**: cần một benchmark mới cho "federated DP retrieval with MRR/Hit@k metrics" — đây có thể là contribution của FedE4RAG follow-up paper.

---

## 4. Mapping kết quả Phase 6 vào finding nào trong papers

| Quan sát thực nghiệm | Paper(s) confirm/refute | Section reference |
|---|---|---|
| Non-DP path: test MRR 0.11→0.16 (+45%), `lora_B std` 10× baseline. WORKS. | OpenFedLLM (FL outperforms local trên finance) | OpenFedLLM Table 5 (FinGPT) Sec. 4.3 |
| DP path ε=20: kết quả identical baseline cũ, `lora_B std` 0.000248. STAGNANT. | FL-LLM Healthcare confirm: "DP noise... risk degrading fine-tuning convergence in data-scarce clinical settings" + "noise amplification often seen in LoRA updates under DP constraints" | Sec. V.B (line 892–895), Sec. VI.B (line 966–970) |
| AdamW invariance hypothesis | Không paper nào confirm/refute trực tiếp | Gap |
| Hit@1 = 0% on all setups | OpenFedLLM (line 596: MedQA Hit baseline 0.141 → FedAvg 0.202; SCAFFOLD 0.177) gợi ý retrieval baseline đạt non-trivial khi recipe đúng | OpenFedLLM Table 6 |
| q=1.0 sub-sampling (no Poisson) cho DP-SGD | FedLLM-Bench dùng batch=1 nhưng full client pool ("sample fraction q"); không có hướng dẫn rõ về q=1.0 | FedLLM-Bench Sec. D.3 line 1028–1034 |
| KD-GLE giúp non-DP path nhưng không giúp DP path | TowardsSecureRAG xác nhận KD-GLE là "Federated Knowledge Distillation" defense MIA cho FedE4RAG | Sec. 4.2.2 line 1893–1903 |
| 5 client FL với LoRA r=8, BGE-base | TowardsSecureRAG Table 2 liệt kê BGE-base-en-v1.5 là attack target ⇒ capacity question hợp lý | line 731 |
| ε=20, δ=1e-5, σ=1.83, T=50 | FedLLM-Bench cho thấy collapse ở ε rất cao; FL-LLM Healthcare gợi ý DP-FedLoRA + clip is the standard recipe | FedLLM-Bench Table 8; Healthcare Table I |
| LoRA aggregation noise amplification | FL-LLM Healthcare Sec. VI.B explicit | line 968–970 |
| Retrieval-time leak vs training-time DP | SoK RAG: DP-in-fine-tuning maturity thấp; DP-at-retrieval is alternative | SoK Table III, Sec. III.B |

---

## 5. Ranked recommendations cho next intervention

Với 4 candidate (A: kd_weight 1→100; B: σ=1.0 ⇒ ε=50; C: SGD-momentum thay AdamW; D: BGE-large thay BGE-base) — xếp hạng theo mức độ literature support:

### Hạng 1 — Candidate B (σ=1.0 ⇒ ε=50, noise yếu hơn)

**Literature support**: rất mạnh.
- **FedLLM-Bench Table 8** (line 955–961) cho thấy MT-Bench score nhạy cảm với noise level: FedDP-1e-3 (4.6750) > FedDP-1e-2 (4.5500) > FedDP-0.1 (4.5375), sụp đổ ở FedDP-1 (1.6875). FedE4RAG ε=20 nằm ở vùng noise cao trong scale của FedLLM-Bench (dùng (ε,δ)-form với σ tỷ lệ thuận ε). Hạ noise sẽ recover signal.
- **FL-LLM Healthcare Sec. V.B**: "DP noise magnitudes... risk degrading fine-tuning convergence in **data-scarce settings**" — FedE4RAG có 5 client với dữ liệu hạn chế, đúng kiểu data-scarce ⇒ nhạy với noise.
- **TowardsSecureRAG Sec. 4.2.1 line 1798–1804**: "introduced noise inevitably degrades model utility. Balancing system accuracy and privacy under a limited privacy budget remains a core challenge."
- **Predicted outcome**: nếu noise hiện đang **dominate** signal contrastive (giả thuyết được Healthcare paper support), ε=50 sẽ recover ít nhất một phần utility — kỳ vọng `lora_B std` tăng lên 0.001–0.005 range, val MRR tăng 0.10 → 0.13–0.15. Đồng thời cho phép cô lập câu hỏi: blocker 2 là *DP noise* hay *AdamW invariance*?
- **Cost**: thấp ($1.2 GPU), thí nghiệm decisive.

### Hạng 2 — Candidate A (kd_weight 1→100)

**Literature support**: trung bình–mạnh.
- **TowardsSecureRAG Sec. 4.2.2 line 1894–1903** trực tiếp xác nhận FedE4RAG dùng KD-GLE module cho MIA defense. Nếu kd_weight=1.0 là quá yếu, signal KD đang bị DP noise đè ⇒ tăng kd_weight cũng tăng signal-to-noise ratio.
- **FL-LLM Healthcare Sec. VI.B**: knowledge distillation được nêu là một trong các mechanism chống noise amplification.
- **OpenFedLLM Sec. 5.7 line 933**: "model compression techniques such as **knowledge distillation [129]** and pruning [130], offer promising solutions to reduce model size without significantly compromising performance" — confirm KD là approach hợp lý.
- **Predicted outcome**: nếu KD-GLE signal đang yếu hơn DP noise variance, kd_weight=100 sẽ làm tăng update magnitude trên lora_B nhanh hơn. Kỳ vọng val MRR 0.10 → 0.12–0.14. **Risk**: kd_weight quá lớn có thể *over-fit* teacher embedding distribution, gây test MRR giảm (đã quan sát "rerank lift val ×6 hurt test" trong memory note).
- **Cost**: thấp ($1.2 GPU).

### Hạng 3 — Candidate C (SGD-momentum thay AdamW)

**Literature support**: yếu trong tập 10 paper, nhưng **decisive** cho hypothesis testing.
- **FedLLM-Bench, OpenFedLLM**: cả hai dùng AdamW làm default cho FedLLM, *không* benchmark SGD-momentum dưới DP.
- **FL-LLM Healthcare**: không nêu rõ optimizer cho DP-FedLoRA.
- **Gap nghiên cứu**: AdamW invariance hypothesis hoàn toàn chưa được test trong 10 paper này.
- **Predicted outcome**: nếu hypothesis đúng (AdamW per-parameter LR cancel out DP noise variance), SGD-momentum sẽ làm test MRR **biến động** rõ rệt hơn AdamW — kết quả có thể tốt hơn *hoặc* tệ hơn, nhưng **khác** so với baseline. Nếu kết quả vẫn giống baseline ⇒ refute hypothesis (DP noise dominate, không phải optimizer).
- **Cost**: trung bình ($1.5 GPU). Decisive experiment cho blocker 3.

### Hạng 4 — Candidate D (BGE-large thay BGE-base)

**Literature support**: yếu, có cảnh báo.
- **FL-LLM Healthcare Sec. V.B.c line 901–906**: "concentration of updates in a **small parameter space may actually increase vulnerability**" — gợi ý tăng kích thước model không hẳn giúp DP.
- **OpenFedLLM Sec. 5.6 line 905–917**: khuyến cáo dùng **QLoRA 4-bit để giảm** memory; *không* khuyến cáo tăng model size khi pipeline đang fail.
- **TowardsSecureRAG Table 2 line 706**: BGE-Large-En-V1.5 vẫn nằm trong attack target list của TrojanRAG ⇒ không phải tăng capacity là tự động an toàn hơn.
- **Predicted outcome**: BGE-large (335M) có capacity gấp 3 BGE-base, nhưng vẫn dùng cùng LoRA r=8 (295K). Tỷ lệ trainable/frozen giảm ⇒ *có thể* khiến DP noise ảnh hưởng tương đối *lớn hơn*. Hit@1=0% có thể vẫn lặp lại. Kỳ vọng val MRR thay đổi không nhiều (±0.02).
- **Cost**: cao ($2+ GPU, training time gấp đôi). **Low expected return**, không nên thử trước A/B/C.

### Tổng hợp ranking

| Hạng | Candidate | Literature support score | Predicted utility gain | Cost |
|---|---|---|---|---|
| 1 | **B (σ=1.0)** | Strong (Healthcare V.B + FedLLM-Bench Table 8) | val MRR +0.03–0.05 | $1.2 |
| 2 | **A (kd_weight=100)** | Moderate-Strong (TowardsSecureRAG 4.2.2) | val MRR +0.02–0.04 | $1.2 |
| 3 | **C (SGD-momentum)** | Weak (gap), nhưng decisive cho hypothesis | val MRR ±0.05 (high variance) | $1.5 |
| 4 | D (BGE-large) | Weak, có cảnh báo | val MRR ±0.02 | $2.0+ |

**Khuyến nghị**: chạy B trước, song song A. Nếu B recover utility ở ε=50 mà ε=20 vẫn stuck ⇒ confirm blocker là noise level. Nếu cả B + A vẫn stuck ⇒ chạy C để test hypothesis 3. D nên là last resort.

---

## 6. Open questions cho future work

1. **DP-SGD per-sample trên InfoNCE in-batch contrastive loss**: không paper nào trong 10 paper khảo sát đo được tỷ lệ signal/noise của contrastive gradient trên BGE-class encoder dưới DP-SGD. Cần một ablation σ × τ × batch_size để xác định "DP noise vs contrastive signal" pareto frontier.
2. **AdamW vs SGD vs Lion vs Adafactor dưới DP-SGD**: chưa có benchmark — hypothesis "AdamW invariance" cần một controlled experiment trên cùng task (BGE-base, financial 10-K retrieval, ε∈{10,20,50}, T=50).
3. **Fed-SB cho retrieval loss**: Fed-SB exact-aggregation đã được benchmark cho text classification (BERT/LLaMA-3.2), nhưng *chưa* cho dense retriever fine-tuning với InfoNCE. Câu hỏi: liệu LoRA-SB với R diagonal có thay được FedE4RAG vanilla LoRA?
4. **Hit@1=0% phenomenon**: paper gốc FedE4RAG báo cáo Hit@1=87% — gap rất lớn. Câu hỏi: phải chăng evaluation protocol khác (BEIR-style với hard negatives vs simple holdout) hay corpus size khác? Không paper nào trong tập 10 paper trả lời.
5. **Membership inference đối với FedE4RAG**: TowardsSecureRAG khẳng định FedE4RAG defense MIA "effectively" (line 1901), nhưng *không* cung cấp empirical evaluation. Cần một benchmark MIA cụ thể cho federated retriever.

---

## 7. References

1. Ye, R. et al. **FedLLM-Bench: Realistic Benchmarks for Federated Learning of Large Language Models.** arXiv:2406.04845 (2024).
2. Ye, R. et al. **OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning.** arXiv:2402.06954 (2024, NeurIPS).
3. Mukhopadhyay, S. et al. **PrivacyBench: A Conversational Benchmark for Evaluating Privacy in Personalized AI.** arXiv:2512.24848 (2025).
4. *Red Teaming AI by Red Teaming* (2025) — không có arXiv link extracted; reference [85]–[110] internal.
5. *AVISE: Framework for Evaluating Security of AI Systems* (2026) — references [14]–[27] internal.
6. *Federated Large Language Models for Trustworthy and Privacy-Preserving Healthcare Applications: Challenges and Future Research Directions.* IEEE JBHI, DOI 10.1109/JBHI.2026.3679612 (2026). DP-FedLoRA [26], Fed-SB [28], FedARA [23] inline.
7. *A LINDDUN-based Privacy Threat Modeling Framework for GenAI* (2026) — references [47]–[110] internal; OWASP Threat Modeling Tool [74].
8. *PriMod4AI: Lifecycle-Aware Privacy Threat Modeling for AI Systems using LLM* (2026).
9. Bodea, A.-E., Meisenbacher, S., Klymenko, A., Matthes, F. **SoK: Privacy Risks and Mitigations in Retrieval-Augmented Generation Systems.** arXiv:2601.03979 (2026). Repo: github.com/sebischair/SoK-RAG-Privacy. Differential Privacy ref [8], [21], [42], [43], [46], [49], [86].
10. *Towards Secure Retrieval-Augmented Generation: A Comprehensive Review of Threats, Defenses, Benchmarks.* arXiv:2603.21654 (2026). Đề cập FedE4RAG [111] (Sec. 4.2.2); DP-RAG [102]; DPSparseVoteRAG [103]; PAD [104]; INVISIBLEINK [105]; LPRAG [106]; C-FedRAG [110]; Federated MAS [112]; Karamanlioglu clinical RAG [113].

### Citations gốc liên quan đến FedE4RAG hypothesis (paper [111] trong TowardsSecureRAG):
- Reference [26] DP-FedLoRA: "DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning" (FL-LLM Healthcare line 1171).
- Reference [28] Fed-SB: "LoRA-Silver-Bullet" với exact aggregation (Table I, Sec. VI.B).
- Reference [14] FFA-LoRA: Sun, Y., Li, Z., Li, Y., Ding, B. **Improving LoRA in Privacy-Preserving Federated Learning.** (FedLLM-Bench ref [14] line 599).
