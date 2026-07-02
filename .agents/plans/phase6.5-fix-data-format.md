# Phase 6.5 — Fix data format: chuyển từ chunk-pair sang Q-A (URGENT)

> **Ngày tạo**: 2026-05-17
> **Phase trước**: Phase 6 (chunk-pair regenerate) — Hit@1 = 0% trên MỌI setup
> **Discovery**: Paper FedE4RAG **đã public train Q-A data** tại `DocAILab/FedE4RAG_Dataset/FEDE4FIN/train_data/data_*.json`
> **Local files**: `FedE/train_data/data_{1000,2000,5000,10000,20000,50000}_random.json` (187 MB total, ĐÃ CÓ)
> **Mục đích**: Verify hypothesis "data format mismatch là root cause Hit@1=0%" — rẻ + decisive
> **Cost**: $0.5-1.5 GPU, 2-4 giờ
> **Confidence**: 85% Hit@1 sẽ lift > 5%

## TL;DR

Phase 1 regenerate (chunk-pair) là **fix sai**. Format CORRECT là Q-A natural language (paper's 50K data). Test bằng cách swap data + retrain.

3 sub-step:
- **6.5A**: Swap to paper's data_50000_random.json (5 companies, Q-A đúng) → non-DP run → eval
- **6.5B (conditional)**: Nếu 6.5A lift Hit@1 → expand diversity bằng LLM-synth Q cho 363 doc còn lại
- **6.5C (conditional)**: Re-run đầy đủ non-DP + DP + qLoRA trên data Phase 6.5 cuối cùng

## User Story

Tôi muốn verify nhanh xem **data format** có phải root cause Hit@1=0% không, trước khi commit resources vào Phase 7 (FFA-LoRA / FedAdam / hard negs / user-level DP). Nếu chỉ cần fix data là Hit@1 lift, tất cả Phase 7 interventions có thể downscope.

---

## CONTEXT REFERENCES

### Files cần đọc

- [FedE/train_data/data_50000_random.json](FedE/train_data/data_50000_random.json) — 100MB, 50K natural Q-A pairs, **paper's actual training data**
- [FedE/scripts/regenerate_training_data.py](FedE/scripts/regenerate_training_data.py) — Phase 1's "chunk-pair" generator (incorrect approach to bypass)
- [FedE/flgo/benchmark/fedrag_classification/core.py:FEDRAG.__init__](FedE/flgo/benchmark/fedrag_classification/core.py) (line 60-83) — loads `./selected_data.json`. Schema expected: `{company, page, index, reference, question}` — **identical** với paper's data_50000_random schema. Drop-in compatible.
- [FedE/main_lora.py](FedE/main_lora.py) — entrypoint, không cần modify
- Phase 6 baseline: `docs/training_recipe_iteration_summary_vi.md` Section 7.5

### Schema verification

Paper's `data_50000_random.json` sample (verified 2026-05-17):
```json
{
  "company": "AES",
  "page": "82 | 2019 Annual Report\nOperating Margin...",
  "index": 5627,
  "reference": "Consolidated Operating Margin — Operating margin increased $108 million...",
  "question": "What were the main factors contributing to the increase in consolidated operating margin in 2018 compared to 2017?"
}
```

`core.py:FEDRAG.__getitem__` expects `(question, id, reference)`:
```python
for entry in data:
    self.questions.append(entry['question'])     # ← natural Q (đúng format!)
    self.id.append(entry['company'])              # company as ID
    self.ref.append(entry['reference'])            # gold passage
```

→ **Drop-in compatible**, không cần code change. Chỉ cần đổi `selected_data.json` content.

### Train/Eval leak check

| Source | Companies | Count | Overlap với eval? |
|---|---|---|---|
| Paper train data_50000_random | PEPSICO, PG, BOEING, ACTIVISION, AES | 50K | ❌ Different questions (verified) |
| FedE/paper_test_data/val_qa_data_50.json | 24 companies | 50 | reference (eval only) |
| FedE/paper_test_data/test_qa_data_100.json | 30 companies | 100 | reference (eval only) |

Companies overlap: PEPSICO + BOEING xuất hiện cả trong train + eval. BUT **questions là KHÁC** → không bị data leak ở question level. Đây là setup paper sử dụng.

---

## IMPLEMENTATION PLAN

### Sub-phase 6.5A — Quick verify (CRITICAL, $0.5)

**Mục đích**: Chỉ test hypothesis. Run 1 non-DP training trên paper's 50K data, eval val+test.

**Acceptance**:
- ✅ Pass: val MRR > 0.15 (vs Phase 6 0.11) OR Hit@1 > 0%
- ❌ Fail: same Phase 6 → data format không phải root cause, hypothesis wrong

### Sub-phase 6.5B — Expand diversity (conditional on 6.5A pass)

Generate synthetic Q-A pairs cho 363 docs khác (ngoài 5 cty paper train):
- Cho mỗi chunk gold trong page → LLM sinh 1 natural question
- Tool: Phi-3-mini-128k (local, free trên RTX 4090) hoặc OpenAI GPT-3.5-turbo
- Output: ~30K synthetic Q-A pairs từ 43 cty còn lại
- Merge với paper's 50K → **80K total covering all 43+ companies**

**Acceptance**:
- ✅ Pass: val MRR > 0.30 (vs 6.5A baseline)
- ❌ Fail: diversity không help, stick với paper's 5-cty data

### Sub-phase 6.5C — Final 3-setup runs (conditional on 6.5B)

Re-run đầy đủ non-DP + DP + qLoRA với data Phase 6.5B cuối cùng → eval matrix.

---

## STEP-BY-STEP TASKS

### Sub-phase 6.5A

#### Task 6.5A.1 — UPLOAD paper's data lên remote

- **IMPLEMENT**:
  ```bash
  ssh ... "cd /root/sda/FedE && ls -lh train_data/ 2>&1 || mkdir -p train_data"
  scp -i ~/.ssh/id_ed25519 -P <PORT> FedE/train_data/data_50000_random.json root@<IP>:/root/sda/FedE/train_data/
  # OR (cheaper): download from HF on remote
  ssh ... "cd /root/sda/FedE && source ../venv/bin/activate && python3 -c \"
  from huggingface_hub import hf_hub_download
  import shutil, os
  os.makedirs('train_data', exist_ok=True)
  p = hf_hub_download('DocAILab/FedE4RAG_Dataset', 'FEDE4FIN/train_data/data_50000_random.json', repo_type='dataset')
  shutil.copy(p, 'train_data/data_50000_random.json')
  print('Downloaded:', os.path.getsize('train_data/data_50000_random.json'))
  \""
  ```
- **GOTCHA**: scp 100MB qua mạng có thể chậm. Download trực tiếp từ HF nhanh hơn (server có bandwidth tốt).
- **VALIDATE**: `ssh ... "ls -lh /root/sda/FedE/train_data/data_50000_random.json"` → 100 MB

#### Task 6.5A.2 — Swap selected_data.json

- **IMPLEMENT**:
  ```bash
  ssh ... "cd /root/sda/FedE && cp train_data/data_50000_random.json selected_data.json"
  ```
- **GOTCHA**: backup old chunk-pair data trước khi overwrite, in case need diff:
  ```bash
  ssh ... "cd /root/sda/FedE && mv selected_data.json selected_data_chunk_pair_phase6.json && cp train_data/data_50000_random.json selected_data.json"
  ```
- **VALIDATE**:
  ```bash
  ssh ... "cd /root/sda/FedE && python3 -X utf8 -c \"
  import json
  d = json.load(open('selected_data.json'))
  print(f'Entries: {len(d):,}')
  print(f'Sample Q: {d[0][\\\"question\\\"][:150]}')
  print(f'Sample R: {d[0][\\\"reference\\\"][:150]}')
  from collections import Counter
  c = Counter(e['company'] for e in d)
  print(f'Companies: {len(c)}, top: {c.most_common(3)}')
  \""
  ```
- Expect: 50,000 entries, natural language Q (not chunk text), 5 companies (PEPSICO/PG/BOEING/ACTIVISION/AES).

#### Task 6.5A.3 — Re-generate task partition

- **IMPLEMENT**:
  ```bash
  ssh ... "cd /root/sda/FedE && rm -rf num5_alpha05_lora training.log training.done checkpoints/ x-lora_*.bin"
  ```
- flgo sẽ re-generate task khi training launch (đọc selected_data.json mới).
- **VALIDATE**: directory cleaned.

#### Task 6.5A.4 — Launch non-DP training (50 rounds, batch=16)

- **IMPLEMENT**:
  ```bash
  ssh ... "cd /root/sda/FedE && nohup bash -c 'source /root/sda/venv/bin/activate && export DP_ENABLED=0 PYTHONUNBUFFERED=1 && python -X utf8 -u main_lora.py > training.log 2>&1; echo EXIT=\$? >> training.log; touch training.done' < /dev/null > /dev/null 2>&1 & disown"
  ```
- **GOTCHA**: Paper recipe Phase 1-5 đã active (commit 9cb1e9c), không cần re-deploy code. Chỉ swap data.
- **ETA**: ~77 min trên RTX 4090 (same as Phase 6 non-DP — same hyperparams, just different data).
- **VALIDATE**:
  ```bash
  ssh ... "cd /root/sda/FedE && grep -E 'main_lora.*DP_ENABLED|Round [0-9]+|client loss' training.log | tail -10"
  ```
- Expect: DP_ENABLED=False, rounds counting, loss values decreasing.

#### Task 6.5A.5 — Eval val + test

- **IMPLEMENT**:
  ```bash
  ssh ... "cd /root/sda/FedE && source ../venv/bin/activate
  FINAL=\$(ls -t x-lora_*.bin | head -1)
  mv \$FINAL non_dp_paper_data_final.bin
  for split in val test; do
    python3 -X utf8 eval_paper_faithful.py --checkpoint non_dp_paper_data_final.bin --name non_dp_paper_data --split \$split 2>&1 | grep -E 'Results|Hit@|EM|MRR|MAP|NDCG'
  done"
  ```
- **ACCEPTANCE check**:
  - **Pass scenario**: val MRR > 0.15 OR Hit@1 > 0% → format IS root cause, proceed 6.5B
  - **Fail scenario**: val MRR ≤ 0.11 (same Phase 6) → format không phải root cause, escalate diagnosis

#### Task 6.5A.6 — Decision gate

Dựa trên 6.5A.5 metrics:
- **Pass**: write 1-page comparison note (Phase 6 vs 6.5A), proceed 6.5B
- **Fail**: write postmortem, escalate to alternative root cause (BGE-base capacity? eval corpus protocol? hardware?)

---

### Sub-phase 6.5B — Expand diversity (conditional)

#### Task 6.5B.1 — CREATE `FedE/scripts/synth_qa_from_chunks.py`

- **IMPLEMENT**: Generate natural Q from chunks using local Phi-3-mini-128k-instruct.
- **DEPS**: `pip install transformers accelerate` (already installed).
- **PATTERN**:
  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM
  import torch, json

  tok = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-128k-instruct')
  llm = AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-128k-instruct',
                                              torch_dtype=torch.bfloat16, device_map='cuda')
  
  def generate_question(chunk: str, company: str) -> str:
      prompt = f"""<|user|>Given this excerpt from {company}'s 10-K SEC filing, write ONE natural question that a financial analyst would ask, which has this excerpt as the answer. The question should be specific (mention {company} and a date/figure/topic) but not require additional context.

Excerpt: {chunk[:1500]}

Question:<|end|>
<|assistant|>"""
      inputs = tok(prompt, return_tensors='pt').to('cuda')
      out = llm.generate(**inputs, max_new_tokens=80, temperature=0.7, do_sample=True, pad_token_id=tok.eos_token_id)
      q = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
      return q.split('\n')[0]  # take first line
  ```
- **GOTCHA**:
  - Filter low-quality Q (length < 20 chars, contains "I don't know", duplicates)
  - Run on RTX 4090 ~150 tokens/sec → 30K Q × 80 tokens = 2.4M tokens ≈ 4.4 hours
  - Estimate cost: $1.8 GPU
- **VALIDATE**:
  ```bash
  ssh ... "cd /root/sda/FedE && python3 scripts/synth_qa_from_chunks.py --max-docs 5 --out /tmp/synth_test.json"
  ssh ... "python3 -c \"
  import json
  d = json.load(open('/tmp/synth_test.json'))
  print(f'{len(d)} pairs')
  for s in d[:3]:
      print(f'Q: {s[\\\"question\\\"][:120]}')
      print(f'R: {s[\\\"reference\\\"][:120]}')
      print()
  \""
  ```
- Expect: 3 sample Q look natural, specific, mention company.

#### Task 6.5B.2 — Run full synth Q generation

- **IMPLEMENT**:
  ```bash
  ssh ... "cd /root/sda/FedE && nohup python3 scripts/synth_qa_from_chunks.py \
      --skip-companies PEPSICO,PG,BOEING,ACTIVISIONBLIZZARD,AES \
      --max-per-doc 100 \
      --out train_data/synth_qa_other43cty.json > synth.log 2>&1 & disown"
  ```
- **ETA**: ~4-5h on RTX 4090
- **VALIDATE**:
  ```bash
  ssh ... "python3 -c \"
  import json
  d = json.load(open('train_data/synth_qa_other43cty.json'))
  print(f'Total synth Q: {len(d):,}')
  from collections import Counter
  c = Counter(e['company'] for e in d)
  print(f'Companies: {len(c)}')
  print(f'Top 10: {c.most_common(10)}')
  \""
  ```
- Expect: ~25-30K synth pairs across 43+ companies (excluding paper's 5).

#### Task 6.5B.3 — Merge with paper's 50K

- **IMPLEMENT**:
  ```bash
  ssh ... "cd /root/sda/FedE && python3 -c \"
  import json
  paper = json.load(open('train_data/data_50000_random.json'))
  synth = json.load(open('train_data/synth_qa_other43cty.json'))
  combined = paper + synth
  with open('selected_data.json', 'w') as f:
      json.dump(combined, f)
  print(f'Combined: {len(combined):,} pairs')
  from collections import Counter
  c = Counter(e['company'] for e in combined)
  print(f'Companies: {len(c)}')
  \""
  ```
- Expect: 75-80K pairs across 48+ companies.

---

### Sub-phase 6.5C — Full 3-setup re-run (conditional)

#### Task 6.5C.1 — Run non-DP + DP + qLoRA với combined data

- **IMPLEMENT**: Mirror Phase 6 protocol, chỉ data đã đổi:
  ```bash
  ssh ... "cd /root/sda/FedE
  # Backup previous
  cp selected_data.json selected_data_6.5B.json
  
  # 3 sequential training runs
  for cfg in 'DP_ENABLED=0' 'DP_ENABLED=1' 'DP_ENABLED=1 USE_QLORA=1'; do
    rm -rf num5_alpha05_lora training.log training.done checkpoints/ x-lora_*.bin
    suffix=\$(echo \$cfg | tr ' =' '__')
    eval \"\$cfg nohup bash -c 'source /root/sda/venv/bin/activate && export PYTHONUNBUFFERED=1 && python -X utf8 -u main_lora.py > training_\${suffix}.log 2>&1; touch training.done' & disown\"
    while ! test -f training.done; do sleep 60; done
    mv training.done training_\${suffix}.done
    mv \$(ls -t x-lora_*.bin | head -1) ckpt_6.5C_\${suffix}.bin
  done"
  ```
- **ETA**: ~7-9h (non-DP 1.3h + DP 2.7h + qLoRA-DP 2.7h)
- **Cost**: ~$3-4

#### Task 6.5C.2 — Eval all checkpoints

- **IMPLEMENT**:
  ```bash
  ssh ... "cd /root/sda/FedE && source ../venv/bin/activate
  for ckpt in ckpt_6.5C_*.bin; do
    for split in val test; do
      python3 -X utf8 eval_paper_faithful.py --checkpoint \$ckpt --name \${ckpt%.bin} --split \$split 2>&1 | grep -E 'Results|Hit@|EM|MRR|MAP|NDCG'
    done
  done"
  ```

#### Task 6.5C.3 — Final report

- **IMPLEMENT**: Update `docs/training_recipe_iteration_summary_vi.md` Section 7.6 "Phase 6.5 Results".

---

## VALIDATION COMMANDS

### Level 1: Pre-flight (local)

```bash
# Verify paper data exists locally
python -X utf8 -c "
import json
d = json.load(open('FedE/train_data/data_50000_random.json'))
assert len(d) == 50000
assert 'question' in d[0]
assert 'reference' in d[0]
print('OK paper data:', len(d), 'pairs')
print('Sample Q:', d[0]['question'][:100])
"
```

### Level 2: Remote setup check

```bash
ssh ... "cd /root/sda/FedE && ls -lh selected_data.json train_data/data_50000_random.json"
```

### Level 3: Training smoke (1 round, ~5 min)

```bash
ssh ... "cd /root/sda/FedE && timeout 300 python -X utf8 -u main_lora.py 2>&1 | grep -E 'DP_ENABLED|Round 1|client loss'"
```

### Level 4: Full eval comparison

```bash
ssh ... "python3 eval_paper_faithful.py --checkpoint non_dp_paper_data_final.bin --split val"
ssh ... "python3 eval_paper_faithful.py --checkpoint non_dp_paper_data_final.bin --split test"
```

---

## ACCEPTANCE CRITERIA

### Sub-phase 6.5A (the critical test)

- [ ] Non-DP training completes (50 rounds, EXIT=0)
- [ ] `lora_B std` > 0.001 (consistent với Phase 6 non-DP)
- [ ] **EITHER**: val MRR > 0.15 (>+36% vs Phase 6 0.11)
- [ ] **OR**: val Hit@1 > 0% (paper claim 87%)
- [ ] **OR**: test MRR > 0.20 (>+25% vs Phase 6 0.16)

→ Một trong 3 đủ để confirm hypothesis "data format là root cause".

### Sub-phase 6.5B (if 6.5A pass)

- [ ] Synth Q generation completes, ≥25K pairs
- [ ] Q quality spot-check: 10 random Q look natural + specific
- [ ] Combined data covers ≥40 unique companies
- [ ] Re-training với combined data → val MRR > 6.5A baseline

### Sub-phase 6.5C (if 6.5B pass)

- [ ] 3 setups complete (non-DP, DP, qLoRA)
- [ ] ε spent ≤ 20 cho DP runs
- [ ] At least 1 setup: val Hit@1 > 5%
- [ ] At least 1 setup: val MRR > 0.30

---

## COST & TIMING

| Sub-phase | Wall time | GPU cost | Cumulative |
|---|---|---|---|
| 6.5A (verify) | ~2h (setup + train + eval) | $0.5-0.8 | $0.8 |
| 6.5B (synth Q) | ~5h (LLM-synth) + verify | $2.0 | $2.8 |
| 6.5C (3 runs full) | ~9h | $3-4 | $5.8-6.8 |
| **Total Phase 6.5** | **~16h GPU** | **~$6-7** | |

→ Rẻ hơn Phase 7 ($8) và có upside lift dramatically lớn hơn.

## DECISION TREE

```
6.5A run:
  ├── PASS (val MRR > 0.15 OR Hit@1 > 0%)
  │   └── Continue 6.5B (expand diversity)
  │       └── 6.5B PASS (lift further) → 6.5C (full re-run)
  │           └── 6.5C → may meet 4-5/7 acceptance criteria
  │       └── 6.5B FAIL → stop, use 6.5A data, downscope Phase 7
  └── FAIL (no lift)
      └── data format không phải root cause
          └── alternative diagnoses:
              - BGE-base capacity (try BGE-large)
              - eval protocol mismatch (audit eval code)
              - corpus size 30K too large (try sub-corpus eval)
              - paper's actual training has more docs / pre-trained
              → write postmortem, plan alternative experiment
```

---

## RISKS

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Paper's 50K data overlap với eval (data leak) | Thấp | Cao | Verified 0 question overlap (Phase 6.5 pre-flight) |
| 5 companies thiếu diversity → underfit | Trung | Trung | Phase 6.5B sẽ extend diversity nếu cần |
| Phi-3 synth Q quality kém | Trung | Trung | 6.5B.1 spot-check 10 sample trước run full; alternative: GPT-3.5-turbo API |
| HF download chậm | Thấp | Thấp | Có local copy fallback (scp) |
| Hit@1 vẫn 0% sau 6.5A → hypothesis sai | Trung | Cao | 6.5A.6 decision gate → escalate alternative diagnoses |

## CONFIDENCE SCORE: 8/10

- 6.5A succeeding (verify format): **85%** (strong logic chain: chunk-pair train → can't do Q-retrieval)
- 6.5B further lift: **65%** (depends on synth Q quality)
- 6.5C meeting full criteria: **55%**

## OUT-OF-SCOPE

- BGE-large upgrade (defer Phase 7+)
- FFA-LoRA, FedAdam, hard-negative mining (Phase 7)
- Cross-encoder fine-tune (Phase 7C)
- All Phase 7 interventions are deferred conditionally — re-rank by ROI sau khi Phase 6.5 results in.

---

**Tóm tắt action plan khi user say "tôi đã restart server, làm Phase 6.5A"**:
1. Connect SSH (recipe có sẵn)
2. Download/upload paper's data_50000_random.json
3. `cp` swap selected_data.json
4. `rm -rf num5_alpha05_lora` để re-gen task
5. Launch non-DP training, ETA 77 min
6. Eval val + test
7. Report: pass or fail acceptance → decide 6.5B or stop
