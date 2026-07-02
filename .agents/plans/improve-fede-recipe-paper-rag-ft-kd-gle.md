# Feature: Apply paper's RAG-FT + KD-GLE recipe + close training gaps

The following plan should be complete, but it is important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to **two unexpected findings from previous validation**:
1. Original `core.py:compute_client_loss` was `loss_1 + 100·MSE(server_logits, local_logits)` — **this IS paper's RAG-FT + KD-GLE recipe** (verified against arXiv:2504.19101v1 §3). Our Phase 3 "improvement" replaced MSE with KL — this DEVIATED from paper.
2. Training data `data_50000_random.json` has only **5 companies** (AES, ACTIVISIONBLIZZARD, BOEING, PEPSICO, PG) but `train_corpus.json` has **368 documents**. Our pretrained-vs-fine-tuned eval gap exists because training data is severely under-diversified relative to eval distribution (43+ companies in val_qa).

## Feature Description

Reproduce paper FedE4RAG's exact retrieval recipe to close the gap with paper's reported scores (Hit@1=87/73), then layer in known recommendations:

1. **Revert Phase 3 deviation**: replace InfoNCE+KL (off-paper) with paper's RAG-FT (InfoNCE) + KD-GLE (MSE on similarity matrix).
2. **Make per-sample DP compatible with paper recipe**: rewrite `_train_dp` so each sample's loss includes in-batch contrastive + teacher MSE while supporting per-sample gradient extraction for clip+noise.
3. **Regenerate diversified training data**: use full `train_corpus.json` (368 docs) to create training Q&A pairs covering all companies, not just 5.
4. **Longer training + paper hyperparams**: 25 → 50 rounds, batch_size=16 (paper's optimal).
5. **Optional Phase 5 enhancement**: cross-encoder re-rank in eval-side for top-100 → top-10.

## User Story

As a researcher reproducing FedE4RAG paper claim "DP-LoRA at ε=20 retains useful retrieval performance",
I want to apply the EXACT paper recipe (RAG-FT InfoNCE + KD-GLE MSE distillation) on training data covering the same diversity as the eval set,
So that my reported Hit@1 / EM / MRR on the paper's val/test splits move from current 0%/6%/0.10 toward paper's 87%/52%/0.71 — proving the pipeline can deliver paper-quality utility under the same DP guarantee.

## Problem Statement

After Phase 1-4 paper-faithful eval (see `docs/paper_faithful_eval_report_vi.md`), 4 root causes for low scores are now identified:

1. **(off-paper)** Loss in DP path is `1 - cos_sim(q, r)` (no negatives, single-sample) — paper uses InfoNCE in-batch.
2. **(off-paper)** Loss in non-DP path is InfoNCE + **KL**(τ²) — paper uses InfoNCE + **MSE** on similarity matrices (we deviated in Phase 3).
3. **(data)** Training data `data_50000_random.json` covers only 5 companies; eval `val_qa_data_50.json` covers 43 → fine-tuned model overfits 5 cos's questions and mismatches eval distribution.
4. **(scale)** Only 25 rounds, batch=8 — paper used 25 rounds at batch=16 explicitly stated optimal.

Combined effect: pipeline reaches Hit@1=0% across all setups; Non-DP LoRA only manages to lift val MRR from 0.10 → 0.19 (rest still 0).

## Solution Statement

Five-phase incremental fix, ordered by expected impact and implementation complexity:

| Phase | Problem addressed | Expected impact | Effort |
|---|---|---|---|
| 1 | (3) data diversity | **Largest** — match training to eval distribution | Med |
| 2 | (2) revert Phase 3 KL → paper MSE | Moderate — restore correct recipe in non-DP path | Low |
| 3 | (1) DP path uses paper's InfoNCE + MSE-KD with per-sample gradient | Large — DP path actually learns from negatives | High |
| 4 | (4) longer training, batch=16, 50 rounds | Moderate — give model time/capacity to converge | Low (config) |
| 5 | (optional) cross-encoder re-rank | Eval-side; can lift Hit@1 if top-100 has gold | Low |

## Feature Metadata

**Feature Type**: Refactor + Enhancement (revert deviation + apply paper recipe + scale)
**Estimated Complexity**: Medium-High (per-sample DP with batch-contrastive is the hardest piece)
**Primary Systems Affected**: `FedE/flgo/algorithm/fedrag_lora.py` (DP loss), `FedE/flgo/benchmark/fedrag_classification/core.py` (revert Phase 3), `FedE/main_lora.py` (hyperparams), training data
**Dependencies**: existing stack (no new libs); optionally `sentence-transformers` for cross-encoder in Phase 5

---

## CONTEXT REFERENCES

### Relevant Codebase Files — IMPORTANT: YOU MUST READ THESE BEFORE IMPLEMENTING!

- `FedE/flgo/algorithm/fedrag_lora.py:_train_dp` (lines ~330-410) — Why: per-sample DP loop currently uses `1 - cos_sim` (no negatives). Will be rewritten in Phase 3 to use full-batch encoded references as negative pool, with per-sample InfoNCE + MSE-KD-GLE.
- `FedE/flgo/algorithm/fedrag_lora.py:_train_plain` (lines ~290-325) — Why: non-DP path; currently uses Phase 3's InfoNCE+KL. Will revert to call `compute_client_loss` (paper's MSE-KD) in Phase 2.
- `FedE/flgo/benchmark/fedrag_classification/core.py:128-201` — Why: this IS paper recipe (loss_1 = CE + 100·MSE = RAG-FT + KD-GLE). Phase 3 replaced MSE with KL. Phase 2 of this plan reverts.
- `FedE/main_dp_lora_eps20.py:111-112` — Why: legacy `loss = 1 - cos_sim` was a workaround for per-sample DP because Phase-3-era core.py loss was degenerate at bs=1. With Phase 2 revert + Phase 3 per-sample InfoNCE, this workaround becomes unnecessary — DO NOT mirror it anymore.
- `FedE/main_lora.py:54-72` — Why: option dict; will update num_rounds=50, batch_size=16, num_steps=50.
- `FedE/eval_paper_faithful.py` (entire) — Why: eval framework already paper-faithful from prior work. Phase 5 may add `--rerank` flag to invoke cross-encoder.
- `FedE/paper_test_data/SCHEMA.md` — Why: documents val_qa/test_qa schema for training-data regeneration in Phase 1 (use same `key_content.question` / `evidence.evidence_page_num` shape).

### New Files to Create

- `FedE/scripts/regenerate_training_data.py` — Pulls (question, golden_page) pairs covering all 368 docs from `train_corpus.json`, writes to `selected_data.json`. Phase 1.
- `FedE/scripts/cross_encoder_rerank.py` (optional, Phase 5) — Re-rank top-100 retrieval with `cross-encoder/ms-marco-MiniLM-L-12-v2`.

### Relevant Documentation — YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [Paper §3 RAG-FT loss](https://arxiv.org/html/2504.19101v1)
  - Specific section: §3.2 RAG-FT formulation, exact loss `−1/N·Σ log(exp(sim(h_i^q, h_i^c)/τ) / Σ_j exp(sim(h_i^q, h_j^c)/τ))`
  - Why: source of truth — paper's RAG-FT IS InfoNCE with in-batch negatives.

- [Paper §3 KD-GLE loss](https://arxiv.org/html/2504.19101v1)
  - Specific section: §3.3 KD-GLE — `1/N · Σ ‖z_l − z_g‖²` MSE on similarity matrices.
  - Why: KD is **MSE on similarity scores**, NOT KL on softmaxed logits. Phase 3 deviated.

- [Paper hyperparameters](https://arxiv.org/html/2504.19101v1)
  - Specific section: §4.1 Implementation: lr=1e-5, batch_size=16 (optimal vs 8/32), 25 rounds (22 typically optimal).
  - Why: ground rules for paper-fidelity.

- [Cross-encoder MS-MARCO](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2) (Phase 5 optional)
  - Why: standard re-ranker for retrieval pipelines; lifts Hit@1 when top-100 already contains gold.

### Patterns to Follow

**Per-sample InfoNCE with cached reference embeddings (Phase 3)**

The hard part: per-sample DP needs per-sample gradient, but InfoNCE wants batch-level negatives. Solution: encode references ONCE per batch (no_grad), then for each query do per-sample backward over `q_i_emb @ ref_embs.T`:

```python
# Outside per-sample loop: encode batch once with grad disabled for refs
with torch.no_grad():
    ref_embs = encode(local_model, references)            # (B, D), no grad
    teacher_q_embs = encode(global_model, questions)       # (B, D), no grad
    teacher_ref_embs = encode(global_model, references)    # (B, D)
    teacher_sim = teacher_q_embs @ teacher_ref_embs.t()    # (B, B), no grad

accumulated = [torch.zeros_like(p.data) for p in trainable_params]

# Per-sample loop
for i in range(B):
    local_model.zero_grad()
    # Only q_i has grad; refs are detached cached tensor
    q_i_emb = encode(local_model, [questions[i]])         # (1, D), with grad
    sim_row = q_i_emb @ ref_embs.t()                       # (1, B)

    # Paper's RAG-FT: InfoNCE
    label_i = torch.tensor([i], device=...)
    loss_rag_ft = F.cross_entropy(sim_row / tau, label_i)

    # Paper's KD-GLE: MSE on the row vs teacher's row
    loss_kd_gle = ((sim_row - teacher_sim[i:i+1]) ** 2).mean()

    loss_i = loss_rag_ft + kd_weight * loss_kd_gle
    loss_i.backward()

    # Per-sample clip
    norm = total_grad_norm(trainable_params)
    coef = min(1.0, clip_norm / (norm + 1e-8))
    for j, p in enumerate(trainable_params):
        if p.grad is not None:
            accumulated[j] += p.grad.detach() * coef

# After batch: noise + average + step
for j, p in enumerate(trainable_params):
    noise = torch.randn_like(accumulated[j]) * (sigma * clip_norm)
    p.grad = (accumulated[j] + noise) / float(B)
clip_grad_norm_(trainable_params, max_norm=1.0)
optimizer.step()
```

This gives: (a) per-sample sensitivity bound preserved (each sample's gradient is independently clipped); (b) **InfoNCE with B-1 in-batch negatives** for each sample, matching paper's recipe; (c) MSE-KD signal from server (matches paper). DP guarantee unchanged.

**Anti-patterns to avoid**

- ❌ `loss = 1 - cos_sim(q, r)` per-sample (current `_train_dp`) — no negatives, no gradient signal beyond "make this pair more similar".
- ❌ KL divergence on softmaxed similarity (Phase 3 core.py) — paper uses raw MSE on similarity scalars/matrices, not soft probability comparisons.
- ❌ Batch-level forward inside per-sample loop (8× full-batch forward) — too slow, OOM-prone. Cache references outside.
- ❌ Treat teacher embeddings as having gradient — must be `torch.no_grad()` (frozen reference).

---

## IMPLEMENTATION PLAN

### Phase 1: Regenerate training data with full diversity

Replace `data_50000_random.json` (5 companies) with newly-generated pairs from `train_corpus.json` (368 docs).

**Tasks:**
- Inspect `train_corpus.json` structure (verified: 368 doc names, each is dict of {page_num: {page_content}})
- Generate (question, golden_page_text) pairs by sampling K random pages per doc and synthesizing a question via simple template OR copy from val_qa for templates.
- Save as `selected_data.json` with the schema `core.py` expects: `[{company, page, index, reference, question}, ...]`.
- Target ~10,000-15,000 pairs (smaller than current 43k but covering 368 docs vs 5 companies).

### Phase 2: Revert Phase 3 KL → paper's MSE-KD (low-risk)

Restore `core.compute_client_loss` to paper-faithful: InfoNCE (CE on cosine logits) + MSE on similarity scores.

**Tasks:**
- Edit `core.py:compute_client_loss` (lines 165-201): replace KL block with MSE on raw `(logits, server_logits)` like the original repo's `nn.MSELoss()`.
- Keep `temperature` parameter for InfoNCE (paper uses it).
- Default `kd_weight = 1.0` (paper doesn't specify, but matched magnitude with InfoNCE without 100× scaling — test both 1.0 and 100.0 in tuning).

### Phase 3: Per-sample DP-SGD with paper's RAG-FT + KD-GLE (highest complexity)

Rewrite `fedrag_lora.Client._train_dp` to use cached batch references + per-sample query encoding + paper's InfoNCE + MSE-KD per query.

**Tasks:**
- Implement `_encode_batch_no_grad(model, texts, tokenizer)` helper.
- Pre-compute teacher and student reference embeddings once per batch (no grad for refs).
- Per-sample loop: forward query with grad, backward `loss_rag_ft + kd_weight * loss_kd_gle`, clip+accumulate.
- After loop: noise injection, gradient normalization, optimizer step.
- σ calibration unchanged (privacy guarantee preserved).

### Phase 4: Paper hyperparams + longer training

Match paper's stated optimal: batch=16, 50 rounds (paper says 22 optimal but more is fine for ε budget).

**Tasks:**
- `main_lora.py`: `batch_size=16`, `num_rounds=50`, `num_steps=50` (kept).
- Recalibrate σ for new num_rounds: `find_noise_multiplier(20.0, 50, 1.0, 1e-5) ≈ ?` (will be larger than 1.2940).
- Reuse `kd_weight=1.0` as new default; keep ability to override via env-var.

### Phase 5: Cross-encoder re-rank in eval (optional, eval-side)

Add `--rerank` flag to `eval_paper_faithful.py`: take top-100 BGE retrieval, re-rank with `cross-encoder/ms-marco-MiniLM-L-12-v2`, recompute Hit@1/Hit@10.

**Tasks:**
- Add `sentence-transformers` to requirements (Linux only, GPU recommended).
- New helper `rerank_with_cross_encoder(queries, top_k_pages, batch_size=16)`.
- Re-emit metrics with `_reranked` suffix.

### Phase 6: Run + Eval + Compare

Full re-run of validation pipeline against paper test set, output updated table.

---

## STEP-BY-STEP TASKS

### Task 1.1 — CREATE `FedE/scripts/regenerate_training_data.py`

- **IMPLEMENT**: Read `train_corpus.json`, sample N pages per doc, synthesize a (company, page, index, reference, question) record per page. The "question" can be auto-generated via simple template like `"What does this excerpt from {doc_name} discuss?"` (paper-style retrieval doesn't need natural questions — it learns content matching), or copied from val_qa templates.
- **PATTERN**: Mirror the schema in current `selected_data.json` records: `{company, page, index, reference, question}`.
- **IMPORTS**:
  ```
  import json, os, random
  ```
- **GOTCHA**: Don't use `test_corpus.json` (eval data) — that's the leak. `train_corpus.json` is the paper-distributed training corpus.
- **RECOMMENDATION**: Initial version = simple template question, K=30 pages per doc → 30 × 368 ≈ 11,000 pairs. Larger if need.
- **VALIDATE**:
  ```
  python FedE/scripts/regenerate_training_data.py
  python -c "import json; d=json.load(open('FedE/selected_data.json')); from collections import Counter; c=Counter(x['company'] for x in d); print(f'{len(d)} pairs, {len(c)} unique companies, top: {c.most_common(5)}')"
  # Expect: ~11,000 pairs, ~368 companies (1 per doc), even distribution
  ```

### Task 2.1 — UPDATE `FedE/flgo/benchmark/fedrag_classification/core.py:compute_client_loss`

- **IMPLEMENT**: Replace the KL block (Phase 3 deviation) with paper's MSE on similarity matrices.
- **PATTERN**: Original repo had:
  ```python
  loss_1 = nn.CrossEntropyLoss()(logits, label)
  loss_2 = nn.MSELoss()(logits, server_logits)
  return loss_1 + 100 * loss_2, loss_1, loss_2
  ```
  Restore this BUT keep temperature scaling on loss_1:
  ```python
  loss_1 = F.cross_entropy(logits / tau, label)               # InfoNCE
  loss_2 = F.mse_loss(logits, server_logits)                  # paper KD-GLE
  return loss_1 + self.kd_weight * loss_2, loss_1, loss_2
  ```
- **GOTCHA**: Default `kd_weight` was 1.0 in our Phase 3 (KL weight). For paper MSE, magnitudes differ. Test 1.0 first; bump to 100.0 if MSE term too small to matter. Add to `option` dict for runtime tuning.
- **VALIDATE**: re-run `tests/test_loss_phase3.py` (will need to update tests for MSE expectations).

### Task 3.1 — UPDATE `FedE/flgo/algorithm/fedrag_lora.py:_train_dp`

- **IMPLEMENT**: Rewrite per-sample DP loop to use batch-cached refs + per-sample query forward + paper's InfoNCE + MSE-KD-GLE.
- **PATTERN**: See "Per-sample InfoNCE with cached reference embeddings" code block in CONTEXT REFERENCES above.
- **IMPORTS** (add):
  ```
  from flgo.benchmark.fedrag_classification.core import cos_sim
  ```
- **GOTCHA**:
  - Refs MUST be encoded with `torch.no_grad()` and detached BEFORE the per-sample loop.
  - Teacher (global model) refs and queries also no_grad.
  - The `q_i_emb @ ref_embs.t()` produces a `(1, B)` row — InfoNCE label is `tensor([i])`.
  - For multi-sample batches (B=16), the per-sample loop runs B times — heavy but tractable on GPU (~5-10s per step at batch=16).
- **VALIDATE**:
  ```
  # Local syntax check
  python -c "import ast; ast.parse(open('FedE/flgo/algorithm/fedrag_lora.py', encoding='utf-8').read()); print('OK')"
  # Then on remote, smoke 1-round run:
  DP_ENABLED=1 timeout 300 python -X utf8 -u main_lora.py 2>&1 | tail -20
  # Expect: σ calibrated, lora_B std starts moving (>0.0001 after round 1)
  ```

### Task 4.1 — UPDATE `FedE/main_lora.py` for paper hyperparams

- **IMPLEMENT**:
  - `'num_rounds': 50` (was 25)
  - `'batch_size': 16` (was 8 — paper's optimal)
  - Add `'kd_weight': 1.0` to option (or 100.0 if Task 2.1 tuning shows MSE term needs upweighting)
- **PATTERN**: edit option dict; preserve env-var DP_ENABLED / USE_QLORA gates.
- **GOTCHA**: σ for 50 rounds will be larger (~1.83-2.0). Print at training start.
- **VALIDATE**: launch with `DP_ENABLED=1 python main_lora.py`; first log line confirms `rounds=50, batch=16, kd_weight=1.0`, σ calibrated.

### Task 5.1 (optional) — ADD `--rerank` flag to `eval_paper_faithful.py`

- **IMPLEMENT**: After top-100 retrieval, re-encode (query, page) pairs with cross-encoder, get scalar score, re-sort.
- **PATTERN**:
  ```python
  from sentence_transformers import CrossEncoder
  reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)
  # For each query: pairs = [(query, page_text) for page in top_100]
  # scores = reranker.predict(pairs, batch_size=32)
  # Sort top_100 by descending scores, recompute Hit/MRR/etc on reranked order
  ```
- **GOTCHA**: Cross-encoder runs at ~50-200 pairs/sec on GPU. 50 queries × 100 pairs = 5,000 evals ≈ 1-2 min. Not bad.

### Task 6.1 — Run end-to-end training on remote (Vast.ai)

- **IMPLEMENT**: Setup, push code, sync data, launch.
- **GOTCHA**: 50 rounds × batch 16 × 50 num_steps × per-sample DP loop = significant compute. Estimate ~2-3h on RTX 6000 Ada. Cost ~$3-5.

### Task 6.2 — Run paper-faithful eval with re-ranking (Phase 5 optional)

- **IMPLEMENT**: Re-run all 4 setups × 2 splits with `--rerank` enabled. Compare against current numbers.
- **GOTCHA**: Re-rank typically lifts MRR/Hit@1 substantially when top-100 has gold (which our EM=6%/4% says is rare — re-rank can only help when there's something to re-rank).

### Task 6.3 — Update report `docs/paper_faithful_eval_report_vi.md`

- **IMPLEMENT**: Add new section "Section 11: Iteration with paper recipe" comparing old vs new numbers.

---

## TESTING STRATEGY

### Unit Tests (Phase 2)
- Update `tests/test_loss_phase3.py` (rename to `test_loss_paper_faithful.py`) to verify:
  - `loss_1 = CE(logits / τ, diag_labels)`
  - `loss_2 = MSE(logits, server_logits)` (NOT KL)
  - On Q=R synthetic: loss_1 → 0, loss_2 → 0
  - On random tensors: loss_1, loss_2 both non-zero, scale roughly comparable.

### Integration Tests (Phase 3)
- 1-round smoke run on Vast.ai: verify lora_B std starts non-zero after round 1 (proves per-sample InfoNCE has gradient signal, unlike old `1 - cos_sim` which gave lora_B=0).
- Verify σ_spent monotonic increase across rounds.

### End-to-end (Phase 6)
- Compare new pipeline numbers against `docs/paper_faithful_eval_report_vi.md`'s baseline. Expected improvements:
  - Val MRR: 0.10 → 0.20-0.40 (DP); 0.19 → 0.40-0.60 (non-DP). Paper claims 0.71 — closing gap.
  - Hit@1: 0% → 5-20% (DP); 10-30% (non-DP).
  - EM: 6% → 20-40%.

### Edge Cases
- Empty batch (B=0): skip step, log warning.
- Per-sample query forward fails (OOM at batch=16 on tight VRAM): fall back to batch=8.
- Teacher model = local model at round 0 (server hasn't aggregated yet) — KD-GLE term effectively zero on first round, fine.

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style
```
python -c "import ast; ast.parse(open('FedE/flgo/algorithm/fedrag_lora.py', encoding='utf-8').read()); print('OK fedrag_lora.py')"
python -c "import ast; ast.parse(open('FedE/flgo/benchmark/fedrag_classification/core.py', encoding='utf-8').read()); print('OK core.py')"
python -c "import ast; ast.parse(open('FedE/scripts/regenerate_training_data.py', encoding='utf-8').read()); print('OK regenerate_training_data.py')"
```

### Level 2: Unit Tests
```
python FedE/tests/test_loss_paper_faithful.py    # updated for MSE
python FedE/tests/test_lora_filter.py             # unchanged
python FedE/tests/test_qlora_gate.py              # unchanged
```

### Level 3: Smoke (1-round on Vast.ai)
```
DP_ENABLED=1 timeout 600 python -X utf8 -u main_lora.py 2>&1 | grep -E "DP calibrated|Round 1|Privacy|σ"
# Expect: σ calibrated for 50 rounds, Round 1 completes, eps_spent visible
```

### Level 4: Full pipeline (Vast.ai, ~3h)
```
DP_ENABLED=1 USE_QLORA=0 nohup ... main_lora.py
# Expect: 50 rounds, eps_spent ≤ 20.0, EXIT=0
```

### Level 5: Paper-faithful eval comparison
```
# Re-run 4 setups × 2 splits, with and without --rerank
python eval_paper_faithful.py --checkpoint <new_dp> --split val --name dp_paper_recipe
# Expect: Hit@1 > 0%, MRR > 0.20 (improvement vs prior 0.10)
```

---

## ACCEPTANCE CRITERIA

- [ ] `selected_data.json` regenerated with ≥300 unique companies (vs current 5)
- [ ] `compute_client_loss` returns `(InfoNCE + kd_weight × MSE, InfoNCE, MSE)` matching paper §3
- [ ] DP per-sample loop computes per-sample gradient over (InfoNCE + MSE-KD) using cached batch refs
- [ ] σ calibrated for new num_rounds (50): printed once at train start, reproducible
- [ ] After full training: lora_B std > 0.001 (vs prior 0.000173) — proves stronger learning signal
- [ ] Paper-faithful eval val MRR: DP-LoRA > 0.20 (vs prior 0.10)
- [ ] Paper-faithful eval val Hit@1: at least one setup > 5% (vs prior 0%)
- [ ] No regression in privacy: eps_spent ≤ 20.0
- [ ] Existing tests still pass (`test_lora_filter`, `test_qlora_gate`, `test_ckks_correctness`)

---

## NOTES

### Why not use sentence-transformers' built-in InfoNCE losses?
`sentence-transformers.losses.MultipleNegativesRankingLoss` IS effectively paper's RAG-FT InfoNCE. We could swap in the library, but it imposes its own training loop. Keep our custom `_train_dp` for full control over per-sample DP logic.

### Why not switch to BGE-large?
Considered but deferred:
- BGE-large = 335M params, ~3× more VRAM than BGE-base (109M). Per-sample DP loop becomes 3× slower.
- LoRA on BGE-large = ~885K trainable (vs 295K for base) — DP noise scales with √n_params, larger model has slightly worse signal/noise.
- Paper used BGE-base (verified). Stay paper-fidelity.

### Why kd_weight=1.0 instead of paper's unspecified weight?
Paper doesn't state weight; original `core.py` had 100× hard-coded. With InfoNCE on logits/τ vs MSE on raw logits, magnitude difference is moderate. 1.0 is the safer starting point for paper recipe; tune if needed.

### Estimated cost
| Phase | Effort | GPU cost |
|---|---|---|
| 1 | local 30 min | $0 |
| 2 | local 15 min | $0 |
| 3 | local 1 h | $0 |
| 4 | local 5 min | $0 |
| 5 | local 30 min (optional) | $0 |
| 6 | Vast.ai 3h training + 30 min eval × 4 setups | $5-7 |
| **Total** | **~5h human work** | **~$5-7** |

### Confidence Score: 7/10

- Phase 1 (data): straightforward, clear schema → 9/10
- Phase 2 (revert): trivial → 10/10
- Phase 3 (per-sample DP-InfoNCE-MSE): novel implementation; risk of subtle bugs in caching grad/no-grad boundaries → 6/10
- Phase 4-6 (run, eval): mechanical → 9/10

Major risk: Phase 3 per-sample loop may have a subtle gradient leak (e.g., refs accidentally tracked through grad graph). Mitigate by `assert ref_embs.requires_grad is False` after caching.

### Out-of-scope (not in this plan)
- Switch to BGE-large (deferred — keep paper fidelity)
- Hard negative mining (paper doesn't use; in-batch random is enough)
- Sub-page chunking (eval-side improvement; would need re-encoded corpus)
- Multi-key CKKS for true multi-tenant FHE (separate concern)
