# Feature: Paper-Faithful Evaluation of FedE Pipeline (FedE4RAG Tables II/III reproduction)

The following plan should be complete, but it is important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to: paper's evaluation uses **page-level retrieval against a separate large corpus** (6,656 / 24,323 pages), NOT diagonal i-i matching against the training set. Our current `eval_phase2.py` does the wrong thing.

## Feature Description

Lấp 4 gap để hoàn thành "khung đánh giá đầu" theo đúng paper *"Privacy-Preserving Federated Embedding Learning for Localized Retrieval-Augmented Generation"* (FedE4RAG, arXiv:2504.19101):

1. **Download paper's exact test data** (`val_qa/data_50.json`, `test_qa/data_100.json`, `test_corpus.json`) from HuggingFace `DocAILab/FedE4RAG_Dataset`.
2. **Implement paper-faithful eval framework** with metrics: Hit@k (k=1,10), EM, MRR, MAP, NDCG. Eval procedure is page-level retrieval against the test corpus, matching against golden `evidence.evidence_page_num`.
3. **Run qLoRA full training** (chưa có run này — Phase 5 chỉ test load).
4. **Re-eval 4 setups** (Pretrained zero-shot, Non-DP LoRA, DP-LoRA ε=20, DP-qLoRA ε=20) trên paper's exact val set + test set, output table format giống Table II/III.

## User Story

As a researcher reproducing the FedE4RAG paper with my own DP+LoRA/qLoRA pipeline,
I want to evaluate my trained checkpoints using the EXACT same test set and metrics as the paper (50/100 queries on 6,656/24,323-page corpus, Hit/EM/MRR/MAP/NDCG),
So that my reported numbers (e.g. "Hit@1=X, MAP=Y") are directly comparable with paper's Table II/III numbers (Hit@1=73/87, MAP=90.22/74.94) and reviewers can verify the privacy-utility tradeoff claim apples-to-apples.

## Problem Statement

Current eval (`eval_phase2.py`):
- Tests on **200 random samples FROM the training data** (`selected_data.json`) — **train/test leak**.
- Diagonal i-i matching (query[i] vs reference[i]) — wrong objective, paper retrieves against a large corpus.
- Missing metrics: **EM**, **MAP** (paper's primary metrics, not in our output).
- Wrong corpus size (200 vs paper's 6,656 / 24,323 pages).

Result: my reported "98.4% Hit@5 retention" cannot be cited alongside paper's numbers because they're not comparable. Reviewers will reject the comparison.

Additionally: Phase 5 qLoRA only verified at the load level (`test_qlora_gate.py` passes — base loads in NF4 4-bit). NEVER ran a full 25-round DP training with qLoRA enabled. Task explicitly asks for "DP-SGD & LoRA/qLoRA" effectiveness — qLoRA half is missing.

## Solution Statement

Build a paper-faithful eval pipeline:

1. Download paper test data → store in `FedE/paper_test_data/`.
2. Write new `eval_paper_faithful.py` that:
   - Loads `test_corpus.json` (large doc corpus).
   - Loads `val_qa/data_50.json` or `test_qa/data_100.json` (queries with golden `evidence.evidence_page_num`).
   - Encodes ALL corpus pages with the model (one-time per checkpoint).
   - For each query, encodes the question, ranks ALL corpus pages by cosine similarity, computes top-k.
   - Matches retrieved page numbers vs `evidence.evidence_page_num` for the query → counts hits.
   - Computes Hit@1, Hit@10, EM, MRR, MAP, NDCG following standard IR formulas.
3. Run qLoRA full training on the 43k financial data.
4. Run paper-faithful eval on 4 checkpoints. Output Markdown table matching paper's Table II/III format.

## Feature Metadata

**Feature Type**: Enhancement (extending existing eval; no algorithm changes)
**Estimated Complexity**: Medium (~3-4 hours: 1h coding + 1h qLoRA training + 1h eval × 4 checkpoints + 30min report)
**Primary Systems Affected**: `FedE/eval_*` (eval scripts), `FedE/paper_test_data/` (new dir), `docs/` (final report), `.agents/validation_artifacts/` (new evaluation outputs)
**Dependencies**:
- `huggingface_hub` (already in requirements.txt)
- `numpy`, `torch`, `transformers`, `peft` (existing)
- No new external library

---

## CONTEXT REFERENCES

### Relevant Codebase Files — IMPORTANT: YOU MUST READ THESE BEFORE IMPLEMENTING!

- `FedE/eval_phase2.py` (entire — 98 lines) — Why: starting point for the new eval; has correct embedding+cosine flow but wrong test-set assumption (diagonal matching). Rewrite, don't edit.
- `FedE/eval_pubmed.py` (entire — 90 lines) — Why: PubMed variant of eval_phase2, same diagonal bug.
- `FedE/eval_compare.py:201-260` — Why: legacy eval has 3-checkpoint comparison structure we'll mirror for the 4-setup output table.
- `FedE/main_lora.py` (entire — 99 lines) — Why: entrypoint shape; we need to launch qLoRA-enabled run by setting `USE_QLORA=1` env-var.
- `FedE/flgo/benchmark/fedrag_classification/config.py:42-93` — Why: `get_model(quantize=True)` is the qLoRA path; verify it works on Vast.ai/RTX 6000 Ada.
- `FedE/flgo/algorithm/fedrag_lora.py` (entire — 460 lines) — Why: nothing to change; just re-runs end-to-end with qLoRA gate flipped on.
- `FedE/main_dp_lora_eps20.py:43-46` — Why: σ calibration confirmed deterministic; `find_noise_multiplier(20.0, 25, 1.0, 1e-5) = 1.2940`.
- `docs/final_validation_report_v2.md` (sections 3, 10) — Why: existing eval numbers we'll supersede with paper-faithful numbers.

### New Files to Create

- `FedE/paper_test_data/` directory containing:
  - `val_qa_data_50.json` (50 queries with golden evidence page numbers)
  - `test_qa_data_100.json` (100 queries)
  - `test_corpus.json` (6,656 pages)
- `FedE/eval_paper_faithful.py` — main eval script (~250 lines)
- `FedE/scripts/download_paper_data.py` — one-shot data downloader (~40 lines)
- `docs/paper_faithful_eval_report.md` — final results table

### Relevant Documentation — YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [arXiv 2504.19101 (FedE4RAG paper, HTML version)](https://arxiv.org/html/2504.19101v1)
  - Specific section: "5. Experimental Results" — Tables II and III with exact numbers.
  - Why: target numbers we're trying to reproduce shape (not values; values WILL differ since we use BGE-base + DP-LoRA, paper uses different setup).

- [Paper retrieval-procedure quote](https://arxiv.org/html/2504.19101v1)
  - Section: Evaluation methodology (paragraph mentions `retrieval-context.ids` and page-number matching).
  - Why: defines correctness — must match against `evidence.evidence_page_num`, not diagonal i-i.

- [HuggingFace dataset DocAILab/FedE4RAG_Dataset](https://huggingface.co/datasets/DocAILab/FedE4RAG_Dataset)
  - Specific section: file viewer + dataset card.
  - Why: source of truth for the test data; dataset card describes schema.

- [HuggingFace `huggingface_hub` Python download API](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.hf_hub_download)
  - Specific: `hf_hub_download(repo_id=..., filename=..., repo_type='dataset')`.
  - Why: programmatic download into `paper_test_data/`.

- [scikit-learn ranking metrics reference](https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-ranking-metrics) (or compute manually — we already have NDCG / MRR helpers)
  - Why: cross-check formula for MAP if we want to use library; we'll likely write our own to keep deps minimal.

### Test Data Schema (from earlier WebFetch of HuggingFace)

```python
# val_qa/data_50.json — list of 50 dicts:
{
    "key_content": {
        "reference": "...short excerpt of evidence...",
        "reference_idx": 5627,
        "question": "What was operating margin in 2018?",
        "answer": "increased $108 million"
    },
    "other_info": {
        "doc_name": "AES_2019_10K",
        "company": "AES",
        "question_type": "factoid",
        "question_reasoning": "lookup",
        "evidence": [
            {
                "evidence_text": "...short excerpt...",
                "doc_name": "AES_2019_10K",
                "evidence_page_num": 82,                     # ← golden answer for retrieval
                "evidence_text_full_page": "...full page..."
            }
        ]
    }
}

# test_corpus.json — dict mapping doc_name → list of pages:
# OR list of pages with {doc_name, page_num, page_text} fields.
# MUST INSPECT after download; schema not certain from web docs.
```

### Patterns to Follow

**Eval skeleton from `eval_phase2.py:14-66`** (reuse this structure for embedding flow):

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('BAAI/bge-base-en')
max_length = tokenizer.model_max_length

def evaluate(model, name):
    model.eval().to(device)
    qe = []  # query embeddings
    re = []  # reference (corpus) embeddings
    bs = 32
    for i in range(0, len(queries), bs):
        inp = tokenizer(queries[i:i+bs], return_tensors='pt', padding=True,
                        truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            qe.append(model(**inp).last_hidden_state.mean(dim=1).cpu())
    # ... same for corpus
    Q = F.normalize(torch.cat(qe), dim=-1)
    R = F.normalize(torch.cat(re), dim=-1)
    sim = (Q @ R.t()).numpy()    # NxC similarity matrix
    # ... rank ALL corpus pages per query
```

**Checkpoint-loading pattern from `eval_phase2.py:78-88`** (3 checkpoint formats):

```python
state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
# Detect format:
if any('lora' in k.lower() for k in state):
    # LoRA-only checkpoint → wrap with PEFT, merge_and_unload
    base = BertModel.from_pretrained('BAAI/bge-base-en')
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=['query','value'],
                     lora_dropout=0.05, bias='none')
    wrapped = get_peft_model(base, cfg)
    wrapped.load_state_dict(state, strict=False)
    model = wrapped.merge_and_unload()
else:
    # Full BertModel state_dict
    model = BertModel.from_pretrained('BAAI/bge-base-en')
    model.load_state_dict(state, strict=False)
```

**qLoRA + DP launch pattern (no precedent in repo — first time)**:

```bash
# On Vast.ai / RTX 6000 Ada (Linux only)
ssh root@<IP> "cd /root/sda/FedE && rm -rf num5_alpha05_lora training.log training.done checkpoints x-lora_*.bin
nohup bash -c 'source /root/sda/venv/bin/activate && export DP_ENABLED=1 USE_QLORA=1 PYTHONUNBUFFERED=1 \
  && python -X utf8 -u main_lora.py > training.log 2>&1; \
  echo EXIT=\$? >> training.log; touch training.done' < /dev/null > /dev/null 2>&1 & disown"
```

Note: `main_lora.py` already has `'batch_size': 16 if USE_QLORA else 8` so qLoRA path uses larger batch. Cap `num_steps=50` is already in option dict (from previous fix).

**MAP / EM formulas (paper-faithful)**:

```python
# EM: did any retrieved page (any rank) match a golden page?
em = 1.0 if set(retrieved_top_n_page_nums) & set(golden_page_nums) else 0.0

# MAP for one query, with R relevant pages and N retrieved:
# precision_at_k_relevant = (count of relevants in top-k) / k
# AP = mean(precision_at_k_relevant for k in positions where retrieved[k] is relevant)
def average_precision(retrieved_pages, golden_pages):
    golden_set = set(golden_pages)
    hits = 0
    score = 0.0
    for k, page in enumerate(retrieved_pages, start=1):
        if page in golden_set:
            hits += 1
            score += hits / k          # precision at this position
    if hits == 0:
        return 0.0
    return score / len(golden_set)     # NOTE: paper might divide by min(R, N); standard MAP divides by R
# MAP = mean(AP for each query)
```

**Anti-patterns to avoid**

- ❌ Diagonal i-i matching (current `eval_phase2.py:48`) — that's not retrieval, it's pair similarity.
- ❌ Encoding the corpus once per query — must encode all 6,656 pages ONCE per model (cache or compute upfront).
- ❌ Using `selected_data.json` as test set — that's the training corpus; results will be inflated.
- ❌ Hard-coding paper's Table II/III numbers as our targets — we WILL get different absolute numbers (different base model + DP). Goal is comparable methodology, not identical numbers.

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation — Download & Understand Paper Test Data

Get the exact files paper used + inspect schema before writing any eval code.

**Tasks:**
- Download `val_qa/data_50.json`, `test_qa/data_100.json`, `test_corpus.json` from `DocAILab/FedE4RAG_Dataset`.
- Print schema + sample records to understand format.
- Document the corpus structure (list of pages? dict?).

### Phase 2: Eval Framework (Local)

Write the paper-faithful eval script. Implement on Windows (no GPU needed for code; can test logic with tiny random tensors).

**Tasks:**
- Implement `compute_metrics(retrieved_pages, golden_pages)` returning Hit@k/EM/MRR/MAP/NDCG.
- Implement `encode_corpus(model, corpus)` returning corpus embeddings tensor (one-time per model).
- Implement `run_eval(model_name, ckpt_path, queries, corpus)` returning metrics dict.
- Implement output: pretty-print + JSON dump matching paper Table format.

### Phase 3: qLoRA Full Training (Vast.ai)

The missing piece — actually run qLoRA end-to-end.

**Tasks:**
- Setup new GPU server (recipe in `setup_gpu_server_recipe.md`).
- Launch `USE_QLORA=1 DP_ENABLED=1 python main_lora.py` on 43k financial data.
- Monitor 25 rounds (~50 min on RTX 6000 Ada with batch=16).
- Save checkpoint as `fin_dp_qlora_run10.bin`.

### Phase 4: Run Paper-Faithful Eval on All 4 Setups

Generate the comparison table.

**Tasks:**
- Eval pretrained BGE-base zero-shot.
- Eval Run #5 non-DP LoRA (`fin_lora_nondp_run5.bin` — need to save from server first; not currently in artifacts).
- Eval Run #4 DP-LoRA ε=20.
- Eval Run #10 DP+qLoRA ε=20 (just trained).
- Output comparison table in Markdown.

### Phase 5: Final Report

Write paper-faithful results into `docs/paper_faithful_eval_report.md` with table format matching paper Table II/III. Update `final_validation_report_v2.md` to cross-reference.

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### Task Format Guidelines
- **CREATE**: New files
- **UPDATE**: Modify existing files
- **DOWNLOAD**: Fetch external files
- **RUN**: Execute on remote (GPU server)

---

### Task 1.1 — CREATE `FedE/scripts/download_paper_data.py`

- **IMPLEMENT**: One-shot Python script that uses `huggingface_hub.hf_hub_download` to fetch:
  - `RAG4FIN/val_qa/data_50.json` → save as `FedE/paper_test_data/val_qa_data_50.json`
  - `RAG4FIN/test_qa/data_100.json` → save as `FedE/paper_test_data/test_qa_data_100.json`
  - `RAG4FIN/test_corpus.json` → save as `FedE/paper_test_data/test_corpus.json`
- **PATTERN**: Follow standard HF download:
  ```
  from huggingface_hub import hf_hub_download
  hf_hub_download(repo_id="DocAILab/FedE4RAG_Dataset",
                  filename="RAG4FIN/val_qa/data_50.json",
                  repo_type="dataset",
                  local_dir="FedE/paper_test_data",
                  local_dir_use_symlinks=False)
  ```
- **IMPORTS**:
  ```
  from huggingface_hub import hf_hub_download
  import os, shutil
  ```
- **GOTCHA**: HF Hub may rate-limit unauthenticated requests. Set `HF_TOKEN` env var if HF user has account. Files won't be huge (under ~50 MB total).
- **VALIDATE**:
  ```
  python FedE/scripts/download_paper_data.py
  ls -la FedE/paper_test_data/
  # Expect 3 files: val_qa_data_50.json, test_qa_data_100.json, test_corpus.json
  ```

---

### Task 1.2 — CREATE `FedE/scripts/inspect_paper_data.py`

- **IMPLEMENT**: Print schema + 1 sample record from each downloaded file. Save the schema observations to `FedE/paper_test_data/SCHEMA.md` for future reference.
- **PATTERN**: Defensive inspection:
  ```
  with open(path) as f: d = json.load(f)
  print(f'  type: {type(d).__name__}')
  if isinstance(d, list):
      print(f'  length: {len(d)}')
      print(f'  first record keys: {list(d[0].keys())}')
      print(f'  first record JSON (truncated):')
      print(json.dumps(d[0], indent=2, ensure_ascii=False)[:1500])
  elif isinstance(d, dict):
      print(f'  top-level keys: {list(d.keys())[:10]}')
      first_k = list(d.keys())[0]
      print(f'  sample value type: {type(d[first_k]).__name__}')
  ```
- **IMPORTS**: `json, os`
- **GOTCHA**: `test_corpus.json` might be a dict mapping `doc_name → page_list` OR a flat list of page records. Don't assume — inspect first, then write `_corpus_iter(corpus_data)` accordingly.
- **VALIDATE**:
  ```
  python FedE/scripts/inspect_paper_data.py
  cat FedE/paper_test_data/SCHEMA.md
  # Expect printed records confirming evidence_page_num field in val_qa
  ```

---

### Task 2.1 — CREATE `FedE/eval_paper_faithful.py` — skeleton + metrics functions

- **IMPLEMENT**: Top-level skeleton with arg parsing (`--checkpoint`, `--name`, `--split` ∈ {val, test}, `--corpus` path), and the metrics functions:
  - `hit_at_k(retrieved_pages, golden_pages, k)` → 0/1
  - `em(retrieved_pages, golden_pages, n=10)` → 0/1 within top-n
  - `mrr_for_query(retrieved_pages, golden_pages)` → reciprocal rank of first hit, 0 if none
  - `average_precision(retrieved_pages, golden_pages)` → MAP component
  - `ndcg_at_k(retrieved_pages, golden_pages, k=10)` → NDCG with binary relevance
- **PATTERN**: Pure functions, no torch needed. Mirror `eval_phase2.py:32-65` style.
- **IMPORTS**:
  ```
  import argparse, json, os
  import numpy as np
  ```
- **GOTCHA**:
  - `evidence_page_num` may be int OR list of ints (multi-page evidence). Handle both: always `set(map(int, x if isinstance(x, list) else [x]))`.
  - For MAP, divide by `len(golden_set)`, NOT by `min(len(golden_set), n_retrieved)` — paper's stated MAP is ~74-90 which suggests standard formula.
  - NDCG@k with binary relevance: `dcg = sum(1/log2(rank+1) for rank, page in enumerate(retrieved[:k], 1) if page in golden_set)`.
- **VALIDATE**:
  ```
  python -c "
  from FedE.eval_paper_faithful import hit_at_k, em, mrr_for_query, average_precision, ndcg_at_k
  # Synthetic: 5 retrieved pages, golden = {3}
  retrieved = [10, 20, 3, 40, 50]
  golden = [3]
  assert hit_at_k(retrieved, golden, 1) == 0
  assert hit_at_k(retrieved, golden, 5) == 1
  assert em(retrieved, golden, n=5) == 1.0
  assert mrr_for_query(retrieved, golden) == 1/3
  assert abs(average_precision(retrieved, golden) - 1/3) < 1e-6  # 1 hit at rank 3
  print('OK metrics')
  "
  ```

---

### Task 2.2 — UPDATE `FedE/eval_paper_faithful.py` — add corpus encoder + retrieval

- **IMPLEMENT**: Add `encode_corpus(model, tokenizer, corpus_pages, batch_size=32)` returning a `(N_pages, dim)` torch tensor. Add `retrieve_top_n(query_emb, corpus_emb, n=100)` returning `(top_n_indices, top_n_scores)` tensors.
- **PATTERN**: Follow `eval_phase2.py:42-50` for encoding loop (batch + mean-pool last hidden state), but loop over corpus pages instead of references. Cosine similarity via `F.normalize` then matmul.
- **IMPORTS**:
  ```
  import torch
  import torch.nn.functional as F
  from transformers import BertTokenizer, BertModel
  ```
- **GOTCHA**:
  - 6,656 pages × 768 dim × float32 = 19.5 MB tensor — fits easily in GPU.
  - Tokenize each page with `truncation=True, max_length=tokenizer.model_max_length` (likely 512). Long pages get truncated to first 512 tokens — accept as paper does.
  - **CRITICAL**: Encode corpus ONCE per model (cache the result), then iterate queries against the cached tensor. Don't re-encode per query.
- **VALIDATE**:
  ```
  python -c "
  import torch
  from FedE.eval_paper_faithful import retrieve_top_n
  q = torch.randn(1, 768); q = q / q.norm(dim=-1, keepdim=True)
  c = torch.randn(100, 768); c = c / c.norm(dim=-1, keepdim=True)
  idx, scores = retrieve_top_n(q, c, n=10)
  assert idx.shape == (1, 10)
  print('OK retrieval shape')
  "
  ```

---

### Task 2.3 — UPDATE `FedE/eval_paper_faithful.py` — wire end-to-end + JSON output

- **IMPLEMENT**: Main function that:
  1. Loads queries from `val_qa_data_50.json` (or test split based on CLI arg).
  2. Loads `test_corpus.json` and creates flat `corpus_pages = [{doc_name, page_num, text}, ...]`.
  3. Loads checkpoint (LoRA-only or full state_dict, autodetect).
  4. Encodes corpus → cached tensor.
  5. For each query: encode question → retrieve top-100 → extract `(doc_name, page_num)` of each retrieved → match against golden `evidence.evidence_page_num`.
  6. Aggregates metrics across all queries.
  7. Pretty-prints + dumps JSON to `eval_output_<setup>_<split>.json`.
- **PATTERN**: Mirror `eval_phase2.py` overall structure but replace diagonal matching with corpus retrieval.
- **IMPORTS** (top of file):
  ```
  import argparse, json, os, sys, time
  import numpy as np
  import torch
  import torch.nn.functional as F
  from transformers import BertModel, BertTokenizer
  from peft import LoraConfig, get_peft_model
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  from flgo.benchmark.fedrag_classification.config import LORA_R, LORA_ALPHA, LORA_TARGETS, LORA_DROPOUT
  ```
- **GOTCHA**:
  - "Match retrieved page vs golden page" means matching on `(doc_name, page_num)` tuple, NOT just page_num (different docs can share page numbers).
  - When golden has multiple evidence entries (`other_info.evidence` is a list), treat ALL of them as relevant pages.
  - Encoder warm-up: first batch on GPU is slow (CUDA kernel compile). Don't time-profile until after warm-up.
- **VALIDATE**:
  ```
  python FedE/eval_paper_faithful.py --checkpoint <pretrained_dummy> --split val --name 'pretrained' \
       --corpus FedE/paper_test_data/test_corpus.json
  # Expect: prints Hit@1, Hit@10, EM, MRR, MAP, NDCG. Writes eval_output_pretrained_val.json.
  ```

---

### Task 3.1 — RUN qLoRA full training on Vast.ai (run #10)

- **IMPLEMENT**: Setup new GPU server per `setup_gpu_server_recipe.md`, then launch:
  ```
  USE_QLORA=1 DP_ENABLED=1 python -X utf8 -u main_lora.py > training_run10.log 2>&1
  ```
- **PATTERN**: nohup-detached launch from `setup_gpu_server_recipe.md` Section 4.
- **IMPORTS**: N/A (shell command).
- **GOTCHA**:
  - Pre-flight: `cp data_50000_random.json selected_data.json` (use 43k financial — same as run #4 baseline for comparison).
  - bitsandbytes needs CUDA + Linux (already verified Phase 5 test_qlora_gate passes).
  - `batch_size: 16 if USE_QLORA else 8` — qLoRA halves VRAM so batch doubles. Per-round time ~similar (more samples per step but smaller forward pass).
  - **σ calibration unchanged**: σ=1.2940 for ε=20, T=25, q=1.0 — no need to recalibrate.
  - **Privacy guarantee identical**: qLoRA only changes how base is stored on GPU; LoRA gradient flow + DP-SGD math unaffected.
- **VALIDATE**:
  ```
  ssh ... "cd /root/sda/FedE && grep -E 'Final privacy|Total Time|EXIT=' training.log | tail -3"
  # Expect: eps_spent=19.99/20, EXIT=0
  ```

---

### Task 3.2 — DOWNLOAD run #10 checkpoint to local

- **IMPLEMENT**: scp from remote.
- **PATTERN**: Same as `setup_gpu_server_recipe.md` Section 7.
- **IMPORTS**: N/A.
- **GOTCHA**: Connection may drop during scp; retry single-file. Use `-o ServerAliveInterval=15`.
- **VALIDATE**:
  ```
  ls -la .agents/validation_artifacts/checkpoints/fin_dp_qlora_run10.bin
  # Expect ~1.17 MB (LoRA-only state)
  ```

---

### Task 3.3 — DOWNLOAD run #5 non-DP baseline checkpoint

- **IMPLEMENT**: We have `fin_dp50_run8.bin` and `fin_dp50_relaxed_run9.bin` locally, but NOT the run #5 non-DP baseline (was overwritten on previous server). Need to re-train.
  - **OPTION A**: Re-train run #5 (~13 min on Vast.ai, $0.35).
  - **OPTION B**: Use existing `pubmed_nondp_run7.bin` as proxy (different data — not apples-to-apples).
  - **RECOMMEND**: Option A (cheaper than re-running DP).
- **PATTERN**:
  ```
  cd /root/sda/FedE && cp data_50000_random.json selected_data.json
  rm -rf num5_alpha05_lora training.log training.done checkpoints x-lora_*.bin
  DP_ENABLED=0 python -X utf8 -u main_lora.py > training_run5_redo.log 2>&1
  ```
- **VALIDATE**: `eps_spent` not present (DP off), trainable LoRA std meaningful.

---

### Task 4.1 — RUN paper-faithful eval on 4 setups (val_qa, 50 queries)

- **IMPLEMENT**: For each of 4 checkpoints, run:
  ```
  python FedE/eval_paper_faithful.py --checkpoint <ckpt> --split val --name <label> --corpus FedE/paper_test_data/test_corpus.json
  ```
  4 setups:
  1. `pretrained` → no checkpoint (use base BGE-base)
  2. `non_dp_lora` → `fin_lora_nondp_run5_redo.bin`
  3. `dp_lora_eps20` → `fin_dp50_run8.bin` (note: this is ε=50, not ε=20 — see GOTCHA)
  4. `dp_qlora_eps20` → `fin_dp_qlora_run10.bin`
- **GOTCHA**:
  - **Run #4 (ε=20) checkpoint NOT in artifacts!** Earlier I overwrote it on remote when switching datasets. Must re-train run #4 OR use run #8/#9 (ε=50, but findings showed numbers are identical due to AdamW invariance — see report Section 10).
  - **Practical decision**: re-run #4 ε=20 to get a clean apples-to-apples checkpoint matching `main_dp_lora_eps20.py` baseline. ETA ~50 min, $1.30.
- **VALIDATE**: 4 JSON files in `eval_output_*_val.json` with all 6 metrics filled.

---

### Task 4.2 — RUN paper-faithful eval on test split (100 queries × 24,323 corpus)

- **IMPLEMENT**: Same as Task 4.1 but `--split test --corpus FedE/paper_test_data/test_corpus.json`.
- **GOTCHA**:
  - Larger corpus = encoding takes ~5-10 min/checkpoint on GPU.
  - 24,323 × 768 × float32 = 71 MB tensor — fits.
  - Total wall-clock per checkpoint ≈ 15 min. 4 checkpoints = 1 hour. Budget $1 for this step.
- **VALIDATE**: 4 JSON files for test split.

---

### Task 5.1 — CREATE `docs/paper_faithful_eval_report.md`

- **IMPLEMENT**: Markdown report with table matching paper Table II/III format. Include:
  - Setup description (model, data, hyperparameters)
  - Comparison table for val (50 queries):
    | Setup | Hit@1 | Hit@10 | EM | MRR | MAP | NDCG |
    |---|---|---|---|---|---|---|
    | Pretrained zero-shot | ... | ... | ... | ... | ... | ... |
    | Non-DP LoRA | ... | ... | ... | ... | ... | ... |
    | DP-LoRA ε=20 | ... | ... | ... | ... | ... | ... |
    | DP+qLoRA ε=20 | ... | ... | ... | ... | ... | ... |
  - Same table for test split (100 queries).
  - Discussion: comparison with paper's Table II numbers (different model + DP, expect lower utility but valid privacy-utility tradeoff demo).
- **PATTERN**: Mirror `docs/final_validation_report_v2.md` style.
- **VALIDATE**: All 6 metric columns filled for all 4 setups across 2 splits.

---

### Task 5.2 — UPDATE `docs/final_validation_report_v2.md` — cross-reference

- **IMPLEMENT**: Add new top-level Section 12 "Paper-faithful evaluation results" pointing to `paper_faithful_eval_report.md`.
- **GOTCHA**: Note that earlier eval (200 random samples) was a methodological prototype; paper-faithful numbers are the canonical ones for citation.
- **VALIDATE**: Section 12 exists, links resolve.

---

### Task 5.3 — Update memory files

- **IMPLEMENT**: Update `phase1_6_validation_results.md` with new Run #10 (qLoRA) + Run #4-redo, and new paper-faithful numbers. Add reference to `paper_faithful_eval_report.md`.
- **VALIDATE**: Memory updated with date + new numbers.

---

## TESTING STRATEGY

### Unit Tests (Task 2.1, 2.2)

- `hit_at_k`, `em`, `mrr_for_query`, `average_precision`, `ndcg_at_k` with synthetic inputs (3-5 cases each).
- `encode_corpus` shape check.
- `retrieve_top_n` shape + sorted-descending invariant.

### Integration Tests (Task 2.3)

- Run end-to-end with pretrained BGE-base on val split (50 queries, 6,656 pages). Should produce sane numbers (Hit@1 likely 30-50%, MRR likely 40-60% — pretrained on financial retrieval is OK but not great).
- Same with one DP-LoRA checkpoint. Numbers should be close to pretrained (privacy-utility tradeoff).

### Edge Cases

- Query with empty `evidence` list → skip query in metrics, log warning.
- Corpus page with empty text → embed as zero vector (cos sim undefined; treat as never-retrieved).
- Multi-page evidence (`evidence_page_num` is a list) → all pages count as relevant.
- `doc_name` mismatch (golden doc not in corpus) → skip query, log warning.

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```
python -c "import ast; ast.parse(open('FedE/scripts/download_paper_data.py', encoding='utf-8').read())"
python -c "import ast; ast.parse(open('FedE/scripts/inspect_paper_data.py', encoding='utf-8').read())"
python -c "import ast; ast.parse(open('FedE/eval_paper_faithful.py', encoding='utf-8').read())"
```

### Level 2: Unit Tests (run from local Windows after downloading test data)

```
python -c "
from FedE.eval_paper_faithful import hit_at_k, em, mrr_for_query, average_precision, ndcg_at_k
# (5 tiny test cases)
"
```

### Level 3: Smoke (5 queries on tiny synthetic corpus)

```
python FedE/eval_paper_faithful.py --checkpoint <pretrained_path> --split val --smoke --name pretrained_smoke
# --smoke flag: only first 5 queries + first 100 corpus pages, just to verify pipeline works
```

### Level 4: Full Eval (Vast.ai, GPU, ~15min/checkpoint)

```
for ckpt in pretrained nondp_run5 dp_run4 dp_qlora_run10; do
    python FedE/eval_paper_faithful.py --checkpoint <ckpt> --split val --name $ckpt
    python FedE/eval_paper_faithful.py --checkpoint <ckpt> --split test --name $ckpt
done
```

### Level 5: Manual Validation

- Compare our `pretrained_val.json` Hit@1 with paper's Table III base BGE row (paper might have similar baseline).
- Verify monotonicity: NDCG@1 ≥ NDCG@10 NEVER true; the OPPOSITE: as k increases, NDCG should USUALLY increase (more chance to hit relevant). Sanity-check this property.
- Verify EM ≥ Hit@1 always (EM allows any rank within top-n; Hit@1 is restricted to rank 1).

---

## ACCEPTANCE CRITERIA

- [ ] `FedE/paper_test_data/` contains 3 files: `val_qa_data_50.json` (50 queries), `test_qa_data_100.json` (100 queries), `test_corpus.json` (≥6,656 page records).
- [ ] `eval_paper_faithful.py` produces all 6 metrics (Hit@1, Hit@10, EM, MRR, MAP, NDCG) for any LoRA-only checkpoint or full BertModel state_dict.
- [ ] Run #10 (qLoRA full training) completes with ε=19.99/20, EXIT=0, checkpoint saved locally.
- [ ] 4 setups × 2 splits = 8 eval JSON files generated, all metrics non-zero (sanity).
- [ ] `docs/paper_faithful_eval_report.md` exists with both tables filled, ≥1 paragraph of analysis.
- [ ] Memory files updated to reference new artifacts.
- [ ] No regression: existing tests (`test_lora_filter`, `test_loss_phase3`, `test_qlora_gate`, `test_ckks_correctness`) still pass.

---

## COMPLETION CHECKLIST

- [ ] Phase 1: paper test data downloaded + schema documented
- [ ] Phase 2: eval framework code-complete + unit-tested locally
- [ ] Phase 3: qLoRA training run #10 complete + checkpoint downloaded
- [ ] Phase 3: run #5 non-DP redo + checkpoint downloaded (gap from earlier validation)
- [ ] Phase 3: run #4 DP ε=20 redo + checkpoint downloaded (gap from earlier validation)
- [ ] Phase 4: 4 setups × 2 splits = 8 evals run on Vast.ai
- [ ] Phase 5: report written, memory updated
- [ ] All validation commands pass
- [ ] PR-ready summary written

---

## NOTES

### Why not use HuggingFace `datasets.load_dataset` library

`datasets.load_dataset('DocAILab/FedE4RAG_Dataset')` would auto-cache and provide a clean Python interface, BUT:
- Adds heavy dependency (`datasets` lib + `pyarrow`) just for 3 JSON files.
- The dataset has multiple subsets (`FEDE4FIN`, `RAG4FIN`); script would need to handle both.
- Direct `hf_hub_download` is simpler + lighter.

### Trade-off: re-run vs. use existing checkpoints

Earlier validation overwrote run #4 (DP ε=20) and run #5 (non-DP) checkpoints when switching datasets. To get clean apples-to-apples, both need redo. Estimated ~$2 in GPU costs. The alternative (using run #8 ε=50 as DP proxy) would be defensible because we showed run #4 and run #8 are bit-identical — but reviewers might question. Safer to redo.

### Confidence Score: 8/10

- Eval framework code (Phase 1+2): straightforward, low risk → 9/10
- qLoRA training (Phase 3): already validated at load-level via `test_qlora_gate.py`, full training is mechanical → 8/10
- Eval re-runs on paper test set (Phase 4): biggest unknown is whether HuggingFace test set actually loads cleanly + has expected schema. If schema differs significantly, may need re-implementation in Task 2.3 → 7/10.
- Final report (Phase 5): mechanical → 10/10.

### Estimated Time + Cost

| Phase | Time | Cost |
|---|---|---|
| 1 (download + inspect) | 15 min local | $0 |
| 2 (eval framework code) | 1.5 h local | $0 |
| 3.1 (qLoRA run) | ~50 min Vast.ai | $1.30 |
| 3.3 (non-DP redo) | ~13 min Vast.ai | $0.35 |
| 3.x (run #4 ε=20 redo) | ~50 min Vast.ai | $1.30 |
| 4 (8 evals) | ~2 h Vast.ai | $3.15 |
| 5 (reports + memory) | 30 min local | $0 |
| **Total** | **~6 h** | **~$6.10** |
