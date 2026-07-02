# Feature: Phase 7 — Paper-backed interventions cho FedE4RAG

> **Ngày tạo**: 2026-05-17 (sau Phase 6 GPU validation)
> **⚠️ DEPENDS ON**: [.agents/plans/phase6.5-fix-data-format.md](.agents/plans/phase6.5-fix-data-format.md) **PHẢI CHẠY TRƯỚC**
> **Branch target**: `feature/validate_old_data`
> **Plan cha**: `.agents/plans/improve-fede-recipe-paper-rag-ft-kd-gle.md` (Phase 1-6 đã done)
> **Literature review**: `docs/literature_review_phase6_next_steps_vi.md`
> **Phase 6 report**: `docs/training_recipe_iteration_summary_vi.md`

## 🚨 BLOCKER 1 — Chạy Phase 6.5 trước (UPDATED post-execution)

Phát hiện 2026-05-17 sau khi viết Phase 7 plan:
- **Phase 1 (chunk-pair regeneration) là FIX SAI** cho data diversity issue
- Training data hiện tại là chunk-pairs, KHÔNG phải natural Q-A
- Paper FedE4RAG đã public train Q-A data tại `DocAILab/FedE4RAG_Dataset/FEDE4FIN/train_data/data_*.json` (local: `FedE/train_data/`)

**Phase 6.5A đã chạy (2026-05-17)** — verdict:
- Swap data → paper Q-A: val MRR +55% nhưng test regress
- Format Q-A KHÔNG phải sole root cause Hit@1=0%
- Pretrained ≈ fine-tuned ở oracle doc-filter setup → **BGE-base capacity là bottleneck**

## 🚨 BLOCKER 2 — PAPER METRICS DEFINITION MISMATCH (BREAKTHROUGH 2026-05-17)

Audit `DocAILab/FedE4RAG/RAGTest/eval/evaluate_rag.py:514-519`:

```python
def Hit(retrieved_ids, expected_ids):
    is_hit = any(id in expected_ids for id in retrieved_ids)
    return 1.0 if is_hit else 0.0

# Eval call (line 375-376):
hit1  = Hit(retrieval_ids, golden_context_ids[0:1])    # ANY retrieved == first golden
hit10 = Hit(retrieval_ids, golden_context_ids[0:10])   # ANY retrieved ∈ first 10 golden
```

→ **Paper's `Hit@1` ≡ standard `Recall@K=10`** (paper retrieves top-K=10, asks "is any of those 10 the first golden?"). 

→ Paper's "Hit@1=87%" NOT comparable với standard "Hit@1 in top-10". Apparent ~700× gap mostly **measurement-definition**, không phải model quality.

→ **MUST DO Tier 0**: implement paper-protocol-mimic eval BEFORE Phase 7. Re-evaluate existing checkpoints với paper's exact protocol để có numbers comparable.

→ **Re-framing project**: stop chasing "Hit@1=87%". Frame as "**federated DP retrieval pipeline với measured DP cost (val MRR retention 53%)**".

The following plan should be complete, but it is important to validate documentation and codebase patterns and task sanity before implementing.

Pay special attention to **5 unexpected findings từ Phase 6**:
1. Non-DP paper recipe IMPROVED test (MRR 0.11→0.16, NDCG 0→0.18, Hit@10 0→1) → recipe đúng hướng
2. DP path IDENTICAL với baseline cũ (val 0.10, test 0.12) → "AdamW invariance" hypothesis
3. Hit@1 = 0% trên MỌI setup — không paper nào explain → likely protocol mismatch
4. Cross-encoder MS-MARCO lift val ×6 nhưng hurt test → domain mismatch
5. `lora_B std` non-DP đạt 0.00168 (✅), DP chỉ 0.000248 (❌) → DP path stagnant

## Feature Description

Phase 7 áp dụng 5 cải tiến có literature support từ literature review (10 paper, 2024-2026) nhằm:
1. Đột phá DP path stagnation (Blocker 2 + 3)
2. Lift Hit@1 từ 0% (Blocker 4)
3. Cross-encoder rerank ổn định cả 2 split (Blocker 5)

Phase 7 chia 3 sub-phase **chạy tuần tự** vì mỗi sub-phase test một hypothesis cụ thể:
- **7A**: Quick wins (FFA-LoRA + FedAdam) — test DP path học được không
- **7B**: Hit@1 lift (hard-negative mining) — test Hit@1 protocol gap
- **7C**: Architecture (user-level DP + cross-encoder fine-tune) — test AdamW invariance & domain mismatch

## User Story

As a researcher reproducing FedE4RAG với DP-LoRA on financial 10-K corpus,
I want to apply 5 paper-backed interventions (FFA-LoRA, FedAdam, hard-negative mining, user-level DP, cross-encoder fine-tune),
So that DP path retrieval Hit@1 lifts từ 0% lên ≥10% và val MRR đạt ≥0.20, đóng gap với paper claim Hit@1=87/MRR=0.71 thêm một bước nữa.

## Problem Statement

Sau Phase 6 GPU validation (2026-05-17, $2 cost):
- **DP path stagnant** mặc dù áp paper recipe (val MRR 0.10 identical baseline)
- **Hit@1 = 0%** trên cả 4 setup (paper 87%)
- **`lora_B std`** DP chỉ 0.000248 (target 0.001)
- **Cross-encoder rerank** domain mismatch (val lift ×6, test hurt -25%)

Literature review identifies 5 paper-backed solutions (Healthcare V.B, OpenFedLLM Table 5, FedLLM-Bench D.2, TowardsSecureRAG 4.2.2).

## Solution Statement

5 cải tiến rank theo ROI (effort × evidence support):

| # | Intervention | Effort | Cost | Lift kỳ vọng | Evidence | Sub-phase |
|---|---|---|---|---|---|---|
| **0** | **Phase 6.5: Fix data format (chunk-pair → Q-A)** | swap data | $0.5-6 | **Hit@1: 0→20-50%** | DECISIVE | **6.5 (PRE-REQ)** |
| 1 | **FFA-LoRA** (freeze lora_A) | 5 lines | $1.2 | DP `lora_B std` PASS | Strong | 7A |
| 2 | **FedAdam server-side** | 30 lines | $1.2 | Convergence +5-10% | Strong | 7A |
| 3 | **Hard-negative mining** | 50 lines + 3 runs | $3-4 | Hit@1 0→10-30% | Gap fix | 7B |
| 4 | **User-level DP** | 100 lines | $2-3 | Bypass AdamW invariance | Strong | 7C |
| 5 | **Cross-encoder fine-tune** | 1-2 day work | $2-3 | Rerank stable both splits | Strong | 7C |

**Order strict**: 6.5 → 7A → 7B → 7C. Phase 6.5 lift dramatically lớn nhất, phải làm trước.

## Feature Metadata

**Feature Type**: Enhancement (DP utility improvement + Hit@1 lift + eval domain adaptation)
**Estimated Complexity**: High (5 interventions, 3 sub-phases, ~$8-10 GPU)
**Primary Systems Affected**:
- `FedE/flgo/benchmark/fedrag_classification/config.py` (FFA-LoRA freeze)
- `FedE/flgo/algorithm/fedrag_lora.py` (FedAdam, user-level DP)
- `FedE/scripts/regenerate_training_data.py` (hard negatives)
- `FedE/eval_paper_faithful.py` (cross-encoder fine-tune integration)
- `FedE/main_lora.py` (config glue)

**Dependencies**: existing stack + sentence-transformers (đã có), no new lib

---

## CONTEXT REFERENCES

### Relevant Codebase Files — YOU MUST READ BEFORE IMPLEMENTING

- [FedE/flgo/benchmark/fedrag_classification/config.py](FedE/flgo/benchmark/fedrag_classification/config.py) (entire) — Why: PEFT LoRA wrap logic; FFA-LoRA chèn vào sau `get_peft_model` call. Hiện có `LORA_R=8, LORA_ALPHA=8, LORA_TARGETS=['query','value']`.
- [FedE/flgo/algorithm/fedrag_lora.py:Server.aggregate](FedE/flgo/algorithm/fedrag_lora.py) (lines 206-264) — Why: hiện dùng plain mean across clients. Sẽ refactor thành FedAdam.
- [FedE/flgo/algorithm/fedrag_lora.py:Client._train_dp](FedE/flgo/algorithm/fedrag_lora.py) (lines 337-518) — Why: per-sample DP loop hiện tại; user-level DP sẽ remove logic này, move tới server.
- [FedE/scripts/regenerate_training_data.py](FedE/scripts/regenerate_training_data.py) (entire) — Why: hiện chỉ sample positive pairs (same-page chunks). Thêm hard negatives same-doc-other-page.
- [FedE/flgo/benchmark/fedrag_classification/core.py:compute_client_loss](FedE/flgo/benchmark/fedrag_classification/core.py) (lines 157-220) — Why: nếu hard negatives được thêm, cần modify loss để dùng explicit hard_neg trong InfoNCE batch.
- [FedE/eval_paper_faithful.py:rerank_with_cross_encoder](FedE/eval_paper_faithful.py) — Why: Phase 5 đã implement; cần wire vào cross-encoder fine-tuned thay vì pretrained MS-MARCO.

### New Files to Create

- `FedE/scripts/finetune_cross_encoder.py` — fine-tune `cross-encoder/ms-marco-MiniLM-L-12-v2` trên val_qa data với hard negatives.
- `FedE/tests/test_ffa_lora.py` — unit test verify lora_A frozen, lora_B trainable, init properly.
- `FedE/tests/test_fedadam_aggregation.py` — unit test FedAdam momentum update math (synthetic 2-client).
- `FedE/tests/test_hard_negative_data.py` — verify generated pairs have hard_negative field correct.

### Relevant Documentation — YOU SHOULD READ BEFORE IMPLEMENTING

- **FFA-LoRA**: Sun, Y. et al. "Improving LoRA in Privacy-Preserving Federated Learning." [FedLLM-Bench ref 14, 2024]
  - Specific section: Algorithm 1 (freeze lora_A, train lora_B only)
  - Why: theoretical justification — lora_A đã có Gaussian init signal, lora_B (init=0) cần learn → freeze A giảm variance.

- **FedAdam**: Reddi et al. "Adaptive Federated Optimization." ICLR 2021. [arXiv:2003.00295]
  - Specific section: Algorithm 2 (FedAdam) — server-side Adam over client delta
  - Why: hyperparams `server_lr=1e-3, β1=0.9, β2=0.99, τ=1e-3` validated trên LLM FL.

- **Hard-negative mining**: Karpukhin et al. "Dense Passage Retrieval." EMNLP 2020. [arXiv:2004.04906]
  - Section 4.3 "Different types of negatives" — BM25 hard negatives + in-batch random
  - Why: standard DPR practice; FedE4RAG paper likely uses similar.

- **User-level DP**: McMahan et al. "Learning DP RLM." ICLR 2018.
  - Section 3.2 — clip per-user update, add noise at server
  - Why: existing code in `fedrag_dp.py` (parallel module) already has user-level DP — can copy patterns.

- **Cross-encoder fine-tune**: sentence-transformers docs https://sbert.net/docs/cross_encoder/training_overview.html
  - Why: training API, dataloader patterns.

### Patterns to Follow

**FFA-LoRA pattern** (after PEFT wrap):
```python
wrapped = get_peft_model(base, cfg)
for name, param in wrapped.named_parameters():
    if 'lora_A' in name:
        param.requires_grad = False
n_trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
print(f'[config.get_model] FFA-LoRA: {n_trainable:,} trainable (lora_B only)')
```

**FedAdam pattern** (in Server class):
```python
class Server(BasicServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_lr = float(self.option.get('server_lr', 1e-3))
        self.beta1 = float(self.option.get('server_beta1', 0.9))
        self.beta2 = float(self.option.get('server_beta2', 0.99))
        self.tau = float(self.option.get('server_tau', 1e-3))
        self.m_state, self.v_state = {}, {}  # per-key 1st/2nd moments
    
    def aggregate(self, model_old, models, ...):
        # 1. Average client LoRA states as before
        averaged = ...  # existing code
        
        # 2. Compute delta = averaged - global_old (per LoRA key)
        delta = {k: averaged[k] - model_old_sd['model.' + k] for k in averaged}
        
        # 3. FedAdam update per key
        for k in averaged:
            self.m_state[k] = self.beta1 * self.m_state.get(k, torch.zeros_like(delta[k])) \
                              + (1 - self.beta1) * delta[k]
            self.v_state[k] = self.beta2 * self.v_state.get(k, torch.zeros_like(delta[k])) \
                              + (1 - self.beta2) * delta[k] ** 2
            update = self.server_lr * self.m_state[k] / (torch.sqrt(self.v_state[k]) + self.tau)
            model_old_sd['model.' + k] = model_old_sd['model.' + k] + update
```

**Hard-negative pattern** (data generation):
```python
def generate_pairs_with_hard_negs(doc_name, all_pages_of_doc, rng, n_pairs=2):
    """Each pair: (q, gold_chunk_same_page, hard_neg_chunk_other_page_same_doc)."""
    pairs = []
    page_keys = list(all_pages_of_doc.keys())
    for page_num, page_text in all_pages_of_doc.items():
        chunks = chunk_text(page_text)
        if len(chunks) < 2:
            continue
        for k in range(n_pairs):
            q_idx = rng.randrange(len(chunks))
            gold_idx = rng.randrange(len(chunks))
            if q_idx == gold_idx:
                continue
            
            # Hard negative: chunk from DIFFERENT page of SAME doc
            other_page = rng.choice([p for p in page_keys if p != page_num])
            other_chunks = chunk_text(all_pages_of_doc[other_page])
            if not other_chunks:
                continue
            hard_neg = rng.choice(other_chunks)
            
            pairs.append({
                'company': doc_to_company(doc_name),
                'page': f'{doc_name}#p{page_num}',
                'reference': chunks[gold_idx],
                'question': chunks[q_idx],
                'hard_negative': hard_neg,  # ← NEW FIELD
            })
    return pairs
```

**Anti-patterns to avoid**

- ❌ Bật cả 5 intervention cùng lúc — không biết cái nào contribute lift.
- ❌ Hard-negative từ KHÁC doc — quá easy, không tăng difficulty.
- ❌ FedAdam `server_lr` quá cao (>0.01) — sẽ divergent.
- ❌ User-level DP với σ tính theo công thức record-level — re-calibrate σ cho user-level (q=1.0 / 5, không phải 1.0).
- ❌ Cross-encoder fine-tune CHỈ trên val data → leakage. Phải split train/val từ val_qa_data_50.json.

---

## IMPLEMENTATION PLAN

### Pre-requisite: Phase 6.5 (data fix)

⚠️ **PHẢI CHẠY TRƯỚC Phase 7A.** Toàn bộ chi tiết trong [phase6.5-fix-data-format.md](phase6.5-fix-data-format.md).

Quick summary để baseline Phase 7:
- Swap `selected_data.json` từ chunk-pair (33K) sang paper's `data_50000_random.json` (50K natural Q-A, 5 cty)
- Optional: + LLM-synth Q cho 43 cty còn lại (+30K pairs)
- Re-run non-DP (smoke) → eval
- **Acceptance**: val MRR > 0.15 OR Hit@1 > 0% → proceed Phase 7

Nếu 6.5A pass nhưng 6.5B (synth Q diversity) chưa xong, vẫn có thể bắt đầu Phase 7A trên 6.5A data (5 cty paper). Phase 7A code changes (FFA-LoRA, FedAdam) là **data-agnostic**.

### Sub-phase 7A: Quick wins (FFA-LoRA + FedAdam)

Verify DP path có thể học bằng intervention rẻ nhất trước.

**Tasks**:
- Implement FFA-LoRA freeze in `config.py` (5 lines)
- Implement FedAdam in `Server.aggregate` (30 lines)
- Unit tests for both
- 1 GPU run: DP + FFA-LoRA + FedAdam combined
- Eval val + test → compare với Phase 6 DP baseline (val 0.10, test 0.12)

### Sub-phase 7B: Hit@1 lift (hard-negative mining)

Address gap Hit@1=0% bằng cách thêm hard negatives.

**Tasks**:
- Modify `regenerate_training_data.py` để emit hard_negative field
- Modify `compute_client_loss` để dùng hard_negative trong InfoNCE (extend batch với hard negs)
- Modify per-sample `_train_dp` tương ứng
- Regenerate `selected_data.json` mới
- Unit test: verify hard negs different from gold
- 3 GPU runs: non-DP + DP + qLoRA với hard negs
- Eval → expect Hit@1 lift

### Sub-phase 7C: Architecture (user-level DP + cross-encoder fine-tune)

Bigger interventions test 2 deeper hypotheses.

**Tasks**:
- Implement user-level DP trong `Server.aggregate` (move noise injection từ client tới server)
- Re-calibrate σ cho user-level (q=1/5=0.2 amplification)
- Cross-encoder fine-tune script
- Train cross-encoder on val_qa split (train 80%, val 20%)
- Integrate fine-tuned model trong eval rerank
- 1 GPU run user-level DP + eval
- Eval rerank với fine-tuned model

---

## STEP-BY-STEP TASKS

Execute every task in order, top to bottom. Each task atomic và independently testable.

### Sub-phase 7A — Quick wins

#### Task 7A.1 — UPDATE `FedE/flgo/benchmark/fedrag_classification/config.py`

- **IMPLEMENT**: Add FFA-LoRA freeze logic sau `get_peft_model` call. Add `DEFAULT_USE_FFA_LORA = False` flag để toggle.
- **PATTERN**: see "FFA-LoRA pattern" trong Context References above.
- **IMPORTS**: không cần thêm.
- **GOTCHA**: PEFT đặt tên parameters `base_model.model.encoder...lora_A.default.weight` — filter cần dùng `'lora_A' in name`, not exact match.
- **VALIDATE**:
  ```bash
  python -c "
  from flgo.benchmark.fedrag_classification import config as fedrag_config
  fedrag_config.DEFAULT_USE_FFA_LORA = True
  m = fedrag_config.get_model()
  n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
  n_total_lora = 294912  # baseline LoRA r=8 BGE-base 12 layers × 2 modules
  expected = n_total_lora // 2  # lora_A frozen, lora_B trainable
  print(f'trainable={n_train:,}, expected={expected:,}')
  assert abs(n_train - expected) < 1000
  "
  ```

#### Task 7A.2 — CREATE `FedE/tests/test_ffa_lora.py`

- **IMPLEMENT**: 3 checks: (1) trainable params count ~50% of baseline; (2) lora_A.requires_grad = False, lora_B.requires_grad = True; (3) flag toggle works.
- **PATTERN**: mirror `tests/test_lora_filter.py` style.
- **VALIDATE**: `python -X utf8 FedE/tests/test_ffa_lora.py` → "ALL CHECKS PASSED"

#### Task 7A.3 — UPDATE `FedE/flgo/algorithm/fedrag_lora.py:Server`

- **IMPLEMENT**: Add FedAdam state in `__init__` (or in `_setup_dp` or trong run() preamble). Modify `aggregate` để dùng FedAdam thay plain mean.
- **PATTERN**: see "FedAdam pattern" trong Context References. Key constraint: state per LoRA key, on CPU (state_dict device).
- **IMPORTS**: không cần thêm.
- **GOTCHA**:
  - FedAdam state on CPU (vì model_old_sd on CPU initially in flgo)
  - First round: `m_state.get(k, torch.zeros_like(delta[k]))` handle uninit
  - Add `option.get('server_optimizer', 'fedavg')` để toggle giữa FedAvg/FedAdam.
- **VALIDATE**:
  ```bash
  python -c "
  import torch
  # Simulate 2-client aggregation
  c1 = {'a': torch.tensor([1.0, 2.0])}
  c2 = {'a': torch.tensor([3.0, 4.0])}
  # FedAvg: mean = [2.0, 3.0]
  # FedAdam first round: m=[0.1×2, 0.1×3], v=[0.01×4, 0.01×9], update = lr × m / (sqrt(v)+τ)
  # Sanity: update magnitude ≪ delta when lr=1e-3
  print('Manual FedAdam math OK')
  "
  ```

#### Task 7A.4 — CREATE `FedE/tests/test_fedadam_aggregation.py`

- **IMPLEMENT**: Synthetic 3-client setup, verify FedAdam state updates correctly. Check (a) first round m/v init correctly; (b) second round momentum compounds.
- **VALIDATE**: `python FedE/tests/test_fedadam_aggregation.py` → all PASS.

#### Task 7A.5 — UPDATE `FedE/main_lora.py` for 7A config

- **IMPLEMENT**: Add option entries:
  ```python
  option['server_optimizer'] = os.environ.get('SERVER_OPT', 'fedadam')  # or 'fedavg'
  option['server_lr'] = 1e-3
  option['server_beta1'] = 0.9
  option['server_beta2'] = 0.99
  option['server_tau'] = 1e-3
  ```
- And set `fedrag_config.DEFAULT_USE_FFA_LORA = os.environ.get('USE_FFA_LORA', '0') == '1'`.
- **VALIDATE**: Launch with `USE_FFA_LORA=1 SERVER_OPT=fedadam python main_lora.py` shows correct config in log line.

#### Task 7A.6 — Run GPU smoke test (1-round)

- **IMPLEMENT**: Quick run ~10 phút trên GPU để confirm pipeline không crash.
- **GOTCHA**: nếu OOM khi server FedAdam giữ state in CPU pinned memory → fallback to disk state.
- **VALIDATE**:
  ```bash
  DP_ENABLED=1 USE_FFA_LORA=1 SERVER_OPT=fedadam timeout 600 python -X utf8 -u main_lora.py 2>&1 | tail -30
  # Expect: σ calibrated, round 1 complete, FedAdam state log line visible
  ```

#### Task 7A.7 — Run full GPU DP run (7A)

- **IMPLEMENT**: Full 50-round DP với FFA-LoRA + FedAdam combined.
- **PATTERN**: mirror Phase 6 launch (`nohup bash -c "... DP_ENABLED=1 USE_FFA_LORA=1 SERVER_OPT=fedadam ..."`).
- **GOTCHA**: ETA tương tự Phase 6 (~2.7h trên RTX 4090).
- **VALIDATE**:
  ```bash
  # After completion
  python eval_paper_faithful.py --checkpoint x-lora_*.bin --name dp_7a --split val
  python eval_paper_faithful.py --checkpoint x-lora_*.bin --name dp_7a --split test
  # Expect (acceptance for 7A): val MRR > 0.12, lora_B std > 0.001
  ```

### Sub-phase 7B — Hit@1 lift via hard-negative mining

#### Task 7B.1 — UPDATE `FedE/scripts/regenerate_training_data.py`

- **IMPLEMENT**: Refactor `generate_pairs_from_page` to ALSO take hard negative from SAME DOC but DIFFERENT PAGE. Output schema: add `hard_negative` field.
- **PATTERN**: see "Hard-negative pattern" trong Context References.
- **GOTCHA**:
  - Cần load full doc's pages trước (memory ~1MB/doc × 368 = ~370MB)
  - Nếu doc chỉ 1 page → skip (no other page for hard neg)
  - Backward compat: `hard_negative` field optional (`.get('hard_negative')`) trong `core.py`
- **VALIDATE**:
  ```bash
  python FedE/scripts/regenerate_training_data.py --out /tmp/test_selected.json --pairs-per-page 2
  python -c "
  import json
  d = json.load(open('/tmp/test_selected.json'))
  print(f'{len(d)} pairs')
  with_neg = sum(1 for p in d if 'hard_negative' in p and p['hard_negative'])
  print(f'with hard_negative: {with_neg} ({100*with_neg/len(d):.1f}%)')
  # Sample check
  sample = d[0]
  print(f'q: {sample[\"question\"][:80]}...')
  print(f'gold: {sample[\"reference\"][:80]}...')
  print(f'hard_neg: {sample.get(\"hard_negative\", \"\")[:80]}...')
  assert sample.get('hard_negative') != sample['reference'], 'hard_neg must differ from gold'
  "
  ```

#### Task 7B.2 — UPDATE `FedE/flgo/benchmark/fedrag_classification/core.py:FEDRAG` Dataset

- **IMPLEMENT**: Return tuple `(question, id, reference, hard_negative)` thay vì `(question, id, reference)`. Fallback empty string nếu không có.
- **VALIDATE**: `python -c "from flgo.benchmark.fedrag_classification.core import FEDRAG; d = FEDRAG(); print(d[0])"` → 4-tuple.

#### Task 7B.3 — UPDATE `FedE/flgo/benchmark/fedrag_classification/core.py:compute_client_loss`

- **IMPLEMENT**: Extend in-batch contrastive: concatenate hard_negative chunks vào reference set → mỗi sample có (B-1) random in-batch negs + 1 hard neg → label still diag.
- **PATTERN**:
  ```python
  questions, answers, references = batch_data[0], batch_data[1], batch_data[2]
  hard_negs = batch_data[3] if len(batch_data) > 3 else []
  
  # Encode questions normally
  q_emb = ... # (B, D)
  
  # Reference pool = positive refs + hard negs
  all_refs = references + [h for h in hard_negs if h]
  r_emb = ... # (B + n_hard, D)
  
  logits = cos_sim(q_emb, r_emb)  # (B, B + n_hard)
  label = torch.arange(B)  # gold at position i
  loss_1 = F.cross_entropy(logits / tau, label)
  ```
- **GOTCHA**: nếu hard_negs empty (data cũ) → fallback to current behavior.
- **VALIDATE**: `python -X utf8 FedE/tests/test_loss_paper_faithful.py` still passes (test cũ).

#### Task 7B.4 — UPDATE `FedE/flgo/algorithm/fedrag_lora.py:_train_dp` for hard negs

- **IMPLEMENT**: Trong per-sample loop, cache ALSO hard_neg embeddings. Sim row của sample i = q_i · [refs ∪ hard_neg_i].t().
- **GOTCHA**: noise injection unchanged (per-sample gradient flow still through q_i only).
- **VALIDATE**: Smoke run 1-round with `DP_ENABLED=1 USE_FFA_LORA=1 SERVER_OPT=fedadam` + new data.

#### Task 7B.5 — Regenerate data on remote

- **VALIDATE**:
  ```bash
  ssh ... "cd /root/sda/FedE && python scripts/regenerate_training_data.py"
  ssh ... "python -c \"import json; d=json.load(open('selected_data.json')); print(f'pairs={len(d)}, with_hard_neg={sum(1 for p in d if p.get(\\\"hard_negative\\\"))}')\""
  # Expect: 33k pairs, 95%+ with hard_neg
  ```

#### Task 7B.6 — Run 3 GPU runs (7B)

- **IMPLEMENT**: Non-DP + DP + qLoRA all với hard negs.
- **PATTERN**:
  ```bash
  for cfg in "DP_ENABLED=0" "DP_ENABLED=1" "DP_ENABLED=1 USE_QLORA=1"; do
    rm -rf training.log training.done checkpoints/ x-lora_*.bin
    eval "USE_FFA_LORA=1 SERVER_OPT=fedadam $cfg nohup ... main_lora.py ..." &
    wait
    mv x-lora_*.bin "${cfg// /_}_7b.bin"
  done
  ```
- **GOTCHA**: 3 runs × ~2-3h trên RTX 4090 = ~7-9h. Run overnight.
- **VALIDATE**: Each checkpoint eval val + test → expect Hit@1 > 0% on at least 1 setup.

### Sub-phase 7C — Architecture changes

#### Task 7C.1 — UPDATE `FedE/flgo/algorithm/fedrag_lora.py:Client._train_dp` to NON-DP mode

- **IMPLEMENT**: Khi `option['dp_placement'] == 'server'` → client train như non-DP (no per-sample loop, no noise). Server sẽ apply DP.
- **GOTCHA**: σ phải re-calibrate cho user-level (sampling rate = clients_per_round / num_clients = 1/5 = 0.2).

#### Task 7C.2 — UPDATE `FedE/flgo/algorithm/fedrag_lora.py:Server.aggregate` for user-level DP

- **IMPLEMENT**: Sau khi compute averaged LoRA, clip per-client update + add Gaussian noise (mirror `fedrag_dp.py` patterns).
- **PATTERN**:
  ```python
  if self.option.get('dp_placement') == 'server':
      C = self.option.get('user_clip_norm', 1.0)
      sigma = self._user_dp_sigma  # calibrated separately
      
      # Compute per-client update Δ_i
      deltas = []
      for client_state in lora_dicts:
          delta = {k: v - model_old_sd['model.' + k] for k, v in client_state.items()}
          # Clip Δ_i to C
          norm = sum(d.norm() ** 2 for d in delta.values()) ** 0.5
          coef = min(1.0, C / (norm + 1e-8))
          delta = {k: v * coef for k, v in delta.items()}
          deltas.append(delta)
      
      # Average + noise
      K = len(deltas)
      for k in averaged:
          summed = sum(d[k] for d in deltas)
          noise = torch.randn_like(summed) * (sigma * C)
          averaged[k] = (summed + noise) / K + model_old_sd['model.' + k]
  ```

#### Task 7C.3 — UPDATE `_calibrate_sigma_quiet` for user-level DP

- **IMPLEMENT**: When `dp_placement='server'`, calibrate σ với sample_rate = 1/num_clients (subsampling amplification).
- **VALIDATE**:
  ```bash
  python -c "
  from privacy.rdp_accountant import find_noise_multiplier
  sigma_record = find_noise_multiplier(20, 50, 1.0, 1e-5)
  sigma_user = find_noise_multiplier(20, 50, 0.2, 1e-5)
  print(f'record-level σ={sigma_record:.4f}, user-level σ={sigma_user:.4f}')
  # Expect user-level σ < record-level σ vì amplification
  "
  ```

#### Task 7C.4 — Run GPU user-level DP run

- **VALIDATE**:
  ```bash
  DP_PLACEMENT=server DP_ENABLED=1 USE_FFA_LORA=1 SERVER_OPT=fedadam python main_lora.py
  # Eval expect: val MRR > Phase 7B DP (proves AdamW invariance hypothesis if Δ > +0.02)
  ```

#### Task 7C.5 — CREATE `FedE/scripts/finetune_cross_encoder.py`

- **IMPLEMENT**: Fine-tune `cross-encoder/ms-marco-MiniLM-L-12-v2` trên val_qa split (train 80%, val 20%):
  ```python
  from sentence_transformers import CrossEncoder
  from sentence_transformers.cross_encoder import CrossEncoder
  from sentence_transformers import InputExample
  from torch.utils.data import DataLoader
  
  # Load val_qa, split 80/20
  # For each query: build (q, gold) → label=1, (q, hard_neg from top-100 of pretrained BGE) → label=0
  # Train 3 epochs, save to cross_encoder_fin/
  ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
  ce.fit(train_dl, epochs=3, output_path='cross_encoder_fin/')
  ```
- **GOTCHA**: KHÔNG dùng test_qa cho train (leak). KHÔNG dùng val test set sau cùng (would inflate metrics).
- **VALIDATE**: After train, load và verify size:
  ```bash
  python -c "
  from sentence_transformers import CrossEncoder
  ce = CrossEncoder('cross_encoder_fin/')
  scores = ce.predict([['test query', 'test passage']])
  print(f'OK fine-tuned CE produces score {scores}')
  "
  ```

#### Task 7C.6 — UPDATE `FedE/eval_paper_faithful.py` rerank with fine-tuned CE

- **IMPLEMENT**: Default `--rerank-model` thành `'cross_encoder_fin/'`. Add flag `--rerank-base` to use pretrained instead.
- **VALIDATE**:
  ```bash
  python eval_paper_faithful.py --checkpoint dp_7a_final.bin --split test --rerank
  # Expect: test rerank NO LONGER hurts MRR (was -25%, now ≥ 0)
  ```

### Final task — Update report + handoff

#### Task 7.final — UPDATE `docs/training_recipe_iteration_summary_vi.md`

- **IMPLEMENT**: Add Section 7.6 "Phase 7 Results" với bảng comparison Phase 6 vs 7A vs 7B vs 7C.

---

## TESTING STRATEGY

### Unit Tests (run on CPU, no GPU needed)

1. `test_ffa_lora.py` — verify lora_A frozen, lora_B trainable, count correct
2. `test_fedadam_aggregation.py` — synthetic 3-client FedAdam math
3. `test_hard_negative_data.py` — verify hard_neg different from gold and question
4. `test_user_level_dp.py` — synthetic, verify server-side noise mechanism math
5. Update `test_loss_paper_faithful.py` — extend với hard_negative path

### Integration Tests (GPU)

1. **7A smoke**: 1-round DP+FFA-LoRA+FedAdam, verify pipeline non-crash
2. **7B smoke**: 1-round non-DP với hard negs, verify loss decreases
3. **7C smoke**: 1-round user-level DP, verify σ correct

### End-to-end (GPU full runs)

| Run | Config | ETA | Cost |
|---|---|---|---|
| 7A | DP + FFA-LoRA + FedAdam | 2.7h | $1.2 |
| 7B-non-DP | + hard negs | 1.3h | $0.5 |
| 7B-DP | + hard negs | 2.7h | $1.2 |
| 7B-qLoRA-DP | + hard negs + qLoRA | 2.7h | $1.2 |
| 7C-user-DP | user-level DP + hard negs | 2.0h | $0.8 |
| Cross-encoder fine-tune | 3 epochs | 0.5h | $0.2 |
| **Total** | | **~12h** | **~$5** |

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
python -c "import ast; ast.parse(open('FedE/flgo/algorithm/fedrag_lora.py', encoding='utf-8').read()); print('OK')"
python -c "import ast; ast.parse(open('FedE/flgo/benchmark/fedrag_classification/config.py', encoding='utf-8').read()); print('OK')"
python -c "import ast; ast.parse(open('FedE/flgo/benchmark/fedrag_classification/core.py', encoding='utf-8').read()); print('OK')"
python -c "import ast; ast.parse(open('FedE/scripts/regenerate_training_data.py', encoding='utf-8').read()); print('OK')"
python -c "import ast; ast.parse(open('FedE/scripts/finetune_cross_encoder.py', encoding='utf-8').read()); print('OK')"
```

### Level 2: Unit Tests

```bash
for t in test_ffa_lora test_fedadam_aggregation test_hard_negative_data test_user_level_dp test_loss_paper_faithful test_dp_per_sample_paper test_lora_filter; do
    python -X utf8 FedE/tests/$t.py
done
```

### Level 3: Smoke (1-round each on Vast.ai)

```bash
# 7A
DP_ENABLED=1 USE_FFA_LORA=1 SERVER_OPT=fedadam timeout 600 python -X utf8 -u main_lora.py 2>&1 | grep -E "FFA-LoRA|FedAdam|Round 1|Privacy"
```

### Level 4: Full pipeline (Vast.ai, ~12h total)

```bash
# 7A
DP_ENABLED=1 USE_FFA_LORA=1 SERVER_OPT=fedadam nohup ... main_lora.py
# 7B (3 runs)
# 7C
# Eval all
```

### Level 5: Paper-faithful eval comparison

```bash
# Run all 5 (or however many) checkpoints
for ckpt in dp_7a.bin non_dp_7b.bin dp_7b.bin qlora_dp_7b.bin user_dp_7c.bin; do
    for split in val test; do
        python eval_paper_faithful.py --checkpoint $ckpt --name ${ckpt%.bin} --split $split
        python eval_paper_faithful.py --checkpoint $ckpt --name ${ckpt%.bin}_rerank --split $split --rerank --rerank-model cross_encoder_fin/
    done
done
```

---

## ACCEPTANCE CRITERIA

### Sub-phase 7A targets

- [ ] DP path `lora_B std > 0.001` (vs Phase 6 0.000248) — chứng minh FFA-LoRA + FedAdam đẩy được signal qua noise
- [ ] DP val MRR > 0.12 (vs Phase 6 0.10)
- [ ] ε spent ≤ 20 (unchanged DP guarantee)
- [ ] All unit tests pass

### Sub-phase 7B targets

- [ ] **Hit@1 > 0% trên ít nhất 1 setup** (vs Phase 6 0% all)
- [ ] Non-DP val MRR ≥ 0.19 (recover baseline cũ minimum)
- [ ] Non-DP test MRR ≥ 0.20 (improve over Phase 6 0.16)
- [ ] At least 1 setup test NDCG@10 ≥ 0.20

### Sub-phase 7C targets

- [ ] User-level DP val MRR > record-level DP từ 7A (proves AdamW invariance hypothesis if Δ > +0.02)
- [ ] Cross-encoder fine-tune: rerank lift ổn định CẢ val và test (test không hurt như Phase 6)
- [ ] ε spent ≤ 20 cho user-level DP (sau re-calibration)

### Overall Phase 7

- [ ] Acceptance criteria Phase 6 (≥4/7 met, current 3/7)
- [ ] At least 1 setup val MRR > 0.20
- [ ] At least 1 setup val Hit@1 > 5%
- [ ] No regression in privacy budget
- [ ] All existing tests pass

---

## COMPLETION CHECKLIST

- [ ] All sub-phases 7A, 7B, 7C completed
- [ ] Each task validation passed
- [ ] All 8+ unit tests pass
- [ ] Full eval matrix (5 checkpoints × 2 splits × 2 rerank-or-not) committed
- [ ] Report Section 7.6 updated
- [ ] Handoff doc updated for next iteration (Phase 8?)
- [ ] Code committed + pushed
- [ ] Memory updated với key findings

---

## NOTES

### Estimated cost & timing breakdown

| Sub-phase | Effort (human) | GPU cost | Wall time GPU |
|---|---|---|---|
| **6.5 (pre-req)** | **0.5-2 ngày** | **$1-6** | **~2-16h** |
| 7A | 1 ngày code + smoke | $1.2 | ~3h |
| 7B | 2 ngày code + regen data | $3-4 | ~7h |
| 7C | 3 ngày code + fine-tune CE | $2-3 | ~3h |
| **Total (incl 6.5)** | **~7-8 ngày** | **~$10-15** | **~16-30h** |

→ Phase 6.5A only ($0.5) là cheapest gate. Nếu pass, full Phase 6.5 + 7 ~$10-15.

### Confidence Score

| Sub-phase | Confidence target met |
|---|---|
| 7A (FFA-LoRA + FedAdam) | 7/10 — strong literature, simple implementation |
| 7B (hard negs) | 8/10 — most likely to lift Hit@1, well-known DPR technique |
| 7C (user-level DP) | 6/10 — depends on AdamW invariance being true |
| 7C (cross-encoder fine-tune) | 7/10 — straightforward sentence-transformers usage |

### Major risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| FFA-LoRA insufficient — lora_B still doesn't move | Trung | Combine với FedAdam (still in 7A) |
| Hard negs make training too hard → loss explodes | Thấp | Validate loss curve at smoke run |
| User-level DP σ re-calibration wrong → privacy violation | Trung | Compute_epsilon assertion in pre-flight |
| Cross-encoder fine-tune overfits 40 train queries | Cao | Use 80/20 split, early stop on val MRR plateau |
| 7B regen data takes too long on remote (RAM 503GB OK) | Thấp | If OOM, batch process docs |

### Decision tree for early stop

```
Phase 6.5A (data swap):
  if val MRR > 0.15 OR Hit@1 > 0%:
      → data format WAS root cause, proceed 7A
      → optionally run 6.5B (synth Q for diversity) in parallel
  else:
      → data format không phải root cause, escalate diagnoses
      → DO NOT proceed Phase 7 yet

After 7A (assuming 6.5 pass):
  if lora_B std > 0.001 AND val MRR > Phase 6.5 baseline + 0.02:
      → 7A successful, continue 7B
  else:
      → 7A failed, consider Option B (σ=1.0/ε=50) from Phase 6 plan first

After 7B:
  if Hit@1 > 5% on any setup:
      → 7B success, continue 7C
  else:
      → Hit@1 gap is NOT from hard negatives; review eval protocol

After 7C:
  Document final state regardless of outcome.
```

### Why this order (7A → 7B → 7C)

1. **7A is cheapest** — quickly know if simple optimizer/PEFT tweaks unstick DP
2. **7B is biggest expected lift** — Hit@1 = 0% is the deepest gap, hard negs is the most-cited fix
3. **7C is biggest refactor** — only worth doing if 7A+7B not enough OR want decisive answer to AdamW hypothesis

### Out-of-scope (deferred to Phase 8)

- Switch to BGE-large (deferred)
- Pretraining or continual fine-tune on financial corpus
- Multi-key CKKS FHE
- Generator-side defenses (PrivacyBench finding)
- C-FedRAG cluster-based aggregation (TowardsSecureRAG ref 110)

### Confidence Score (revised after Phase 6.5 discovery): 6/10 for Phase 7 alone

**With Phase 6.5 prerequisite**:
- 6.5A succeeding (data format fix): **85%** ← biggest lift
- 7A succeeding on top of 6.5: 75%
- 7B lifting further: 65%
- 7C succeeding: 55%
- Phase 6.5 + 7A combined hits ≥4/7 criteria: **70%**
- Phase 6.5 + 7A + 7B combined hits ≥5/7 criteria: **55%**
- Full Phase 6.5 + 7A + 7B + 7C hits 6-7/7 criteria: **30%**

→ Phase 6.5 alone (cheapest, $0.5) sẽ produce decisive answer về root cause. Phase 7 chỉ là refinement.
