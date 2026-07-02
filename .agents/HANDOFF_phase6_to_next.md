# Handoff — Phase 6 + 6.5 + 7 → Publication

> **Ngày update**: 2026-05-19 sau Phase 7BC (final phase)
> **Status**: ALL EXPERIMENTS COMPLETE. Total project cost ~$19-21 GPU. Server shutdown.
> **6 publishable findings ready** — see [phase7_dp_helps_finding.md](../memory/phase7_dp_helps_finding.md)
> **Latest commit**: `718bbcf` on `feature/validate_old_data`
> **Final project headline**: *"Privacy-Preserving Federated Retrieval — When DP HELPS by Preventing Overfitting"*

## Trạng thái commits

GitHub branch `feature/validate_old_data`:

```
fcc9ec3  Add training recipe iteration report (Phase 1-6 results)         ← LATEST
9cb1e9c  Apply paper-faithful RAG-FT + KD-GLE recipe (Phases 1-5)
9adedd5  Add paper-faithful eval orchestration + final reports
f691387  Add paper-faithful evaluation framework (Phase 1+2)
1799cd6  Apply remote-validated changes: num_steps cap, _train_plain AdamW, eval scripts
60e1b80  DP per-sample path: override flgo's default SGD with AdamW
```

## Số liệu key (Phase 6, RTX 4090, 2026-05-16)

| Setup | Val MRR | Test MRR | Test Hit@10 | Test NDCG@10 | lora_B std |
|---|---|---|---|---|---|
| Old non-DP (KL) | 0.19 | 0.11 | 0 | 0 | — |
| **NEW non-DP paper (MSE)** | 0.11 | **0.16** ✅ | **1** ✅ | **0.18** ✅ | **0.00168** ✅ |
| Old DP ε=20 | 0.10 | 0.12 | 0 | 0 | 0.000173 |
| **NEW DP paper** | 0.10 | 0.12 | 0 | 0 | 0.000248 |

→ Non-DP recipe works, DP stagnant (AdamW invariance).

## Artifacts đã lưu local (KHÔNG trong git)

```
.agents/validation_artifacts_phase6/
├── checkpoints/
│   ├── non_dp_paper_final.bin     (1.2 MB)
│   └── dp_lora_paper_final.bin    (1.2 MB)
├── eval_outputs/                   (10 JSON files: pre + rerank × 3 setups × 2 splits)
└── logs/
    └── dp_training.log             (DP run; non-DP log đã rm trước khi launch DP — TODO)
```

## File state local

- `docs/training_recipe_iteration_summary_vi.md` — comprehensive report cho meeting (committed `fcc9ec3`)
- `.agents/HANDOFF_phase6_to_next.md` — file này
- Memory: `paper_recipe_phase6_results.md` (đã sync)

## TODO khi restart server

### Option A — kd_weight sweep (recommended single intervention)

```bash
ssh -i ~/.ssh/id_ed25519 root@<NEW_IP>
# 1. Setup (theo setup_gpu_server_recipe.md)
git clone -b feature/validate_old_data --depth 1 https://github.com/ursuswh-metamorphic/SDA_SecureDataAlliance.git sda
cd sda && python3 -m venv venv && source venv/bin/activate
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121  # CRITICAL: CUDA 12.1 build
pip install --quiet transformers peft huggingface_hub tenseal numpy bitsandbytes requests scipy matplotlib prettytable ujson pyyaml pynvml sentence-transformers

# 2. Paper test data + regenerate training data
cd FedE && python3 scripts/download_paper_data.py
# (upload train_corpus.json via scp hoặc download separately)
python3 scripts/regenerate_training_data.py

# 3. Override kd_weight=100 trong main_lora.py (TẠI LINE 34):
#   fedrag_core.DEFAULT_KD_WEIGHT = 100.0    ← thay vì 1.0

# 4. Launch DP-only run
DP_ENABLED=1 nohup python -X utf8 -u main_lora.py > training.log 2>&1 & disown

# 5. Eval sau 3h
python3 eval_paper_faithful.py --checkpoint x-lora_*.bin --name dp_kw100 --split val
python3 eval_paper_faithful.py --checkpoint x-lora_*.bin --name dp_kw100 --split test
```

ETA: 3h training + 30 min eval = **~$1.5 trên RTX 4090**.

### Option B — σ=1.0 (ε=50, looser DP budget)

Cùng setup nhưng:
- `target_epsilon: 50.0` trong `main_lora.py`
- σ sẽ calibrate xuống ~1.0 → noise yếu hơn → signal có thể thắng

Test giả thuyết: nếu noise giảm thì AdamW invariance có còn không.

### Option C — SGD-momentum thay AdamW (DP only)

Edit `FedE/flgo/algorithm/fedrag_lora.py:_train_dp`:
```python
optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9)
# thay vì AdamW
```

Test trực tiếp giả thuyết AdamW invariance.

### Option D — BGE-large (paper deferred)

Edit `FedE/flgo/benchmark/fedrag_classification/config.py`:
```python
MODEL_NAME = 'BAAI/bge-large-en'  # thay 'BAAI/bge-base-en'
```

Cost: VRAM tăng 3× (đủ trên 4090 24GB), training time tăng 3× (~9h DP).

## Recommended priority (REVISED after Phase 6.5A breakthrough)

**TIER 0 (highest priority, $0.2, 1 day code)**:
- **Implement paper-protocol-mimic eval** — re-chunk corpus theo paper's `chunk_size=1024 sentence-split`, retrieve top-K=10 chunks, match `chunk_id ∈ key_content.reference_idx`, compute Hit@k với paper's `any retrieved == first golden` formula.
- Đây là điều kiện cần để compare numbers apples-to-apples với paper's 87%.
- Cost: $0.2 GPU (1 eval run với new protocol on existing 2 checkpoints).

**TIER 1 (rerun trên paper-protocol eval)**:
- Re-eval Phase 6 checkpoints (`non_dp_paper_final.bin`, `dp_lora_paper_final.bin`, `non_dp_paper_qa_final.bin`) với paper-protocol → compare với paper's 87%.

**TIER 2 (only if needed)**:
1. **Option A** (kd_weight=100): cheapest DP intervention. Bắt đầu nếu DP path vẫn ≥30% gap với non-DP.
2. **Option C** (SGD-momentum): test AdamW invariance directly.
3. **Option B** (ε=50): test noise barrier.
4. **Option D** (BGE-large): chỉ làm nếu paper-protocol vẫn show large gap.

## Phase 6.5A artifacts (đã download local)

```
.agents/validation_artifacts_phase6.5/
├── checkpoints/non_dp_paper_qa_final.bin  (1.2 MB, paper Q-A trained)
├── eval_outputs/eval_output_non_dp_paper_qa_{val,test}.json
└── logs/non_dp_training_phase6.5A.log
```

## Key numbers Phase 6.5A

| Setup | Val MRR | Val Hit@10 | Test MRR | Test Hit@10 |
|---|---|---|---|---|
| Phase 6 chunk-pair | 0.11 | 0 | 0.16 | 1 |
| **Phase 6.5A paper Q-A** | 0.17 (+55%) | 0 | 0.12 (-25%) | 0 |
| Oracle doc-filter (~100p) | 3.78 | 6.1 | 2.87 | 7.1 |
| Pretrained + oracle | 2.14 | 6.1 | 3.04 | 5.1 |

→ Trade-off: paper Q-A help val, hurt test. Pretrained ≈ fine-tuned ở oracle → BGE-base capacity bottleneck.

→ Paper's `Hit@1=87%` ≡ standard `Recall@K=10` ≠ standard Hit@1. Apparent gap mostly measurement-definition.

## Sửa lỗi process trong session tới

1. **Backup logs trước cleanup**: trước khi launch trainings sequentially, đặt log per-run riêng:
   ```bash
   mv training.log training_non_dp.log  # thay vì rm
   ```
2. **Track lora_B std mỗi vài rounds**, không chỉ cuối — sớm detect AdamW invariance.
3. **Cross-encoder re-rank chỉ dùng diagnostic, không quyết định**: domain mismatch trên test → results misleading.

## Setup tip — CUDA mismatch trên Vast.ai

Vast.ai instances thường có driver CUDA 12.x. Default pip install torch lấy CUDA 13.0 build → MISMATCH → `cuda_available=False`.

**Fix**: `pip install torch --index-url https://download.pytorch.org/whl/cu121` (force CUDA 12.1 build).

## Server cuối session

- Vast.ai instance `213.181.123.26:30657` (RTX 4090) đã tắt sau Phase 6.
- Branch `feature/validate_old_data` đã sync `fcc9ec3`.
- 2 checkpoints + 10 eval JSONs + 1 log đã backup về `.agents/validation_artifacts_phase6/`.

## Confidence next session

- Recipe works ở non-DP: 9/10 (confirmed)
- kd_weight=100 sẽ help DP: 5/10 (cứ thử, không biết chắc)
- Tổng project trajectory: 7/10 — có lift trên 1 path (non-DP test), cần tiếp tục tune DP path để match paper claim.

---

**Khi user nói "tôi đã restart server, IP là X, làm tiếp Option A"**, mình sẽ:
1. SSH + setup (recipe ở trên)
2. Override kd_weight=100 trong main_lora.py
3. Launch DP-only
4. Eval + compare với Phase 6 DP baseline (val 0.10, test 0.12)
5. Update report nếu có lift
