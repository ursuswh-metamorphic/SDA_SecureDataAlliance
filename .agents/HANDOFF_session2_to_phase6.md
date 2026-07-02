# Handoff — Session 2 → Phase 6 Restart

> **Ngày**: 2026-05-09
> **Status**: Phase 1-4 paper-faithful eval xong; user tắt server; sẽ quay lại làm Phase 6 sau.

## Trạng thái hiện tại — đã commit + push

### GitHub branch `feature/validate_old_data`

```
9adedd5  Add paper-faithful eval orchestration + final reports                ← LATEST
f691387  Add paper-faithful evaluation framework (Phase 1+2)
1799cd6  Apply remote-validated changes: num_steps cap, _train_plain AdamW, eval scripts
60e1b80  DP per-sample path: override flgo's default SGD with AdamW
8c80ebb  DP per-sample path: replace InfoNCE+KL with cos_sim loss
5bd67c9  Fix sigma calibration cache key collision with flgo's option default
e2cf342  Fix 4 runtime issues found by GPU validation on RTX 6000 Ada
8b71805  Add unified FL+FHE+LoRA+DP+InfoNCE/KL+qLoRA pipeline (Phases 1-6)
```

### Files committed lên GitHub session này

| File | Mục đích |
|---|---|
| `FedE/scripts/run_all_evals.sh` | Eval orchestration (4 setups × 2 splits) |
| `docs/paper_faithful_eval_report.md` | Final report (English) |
| `docs/paper_faithful_eval_report_vi.md` | Final report (Vietnamese, có giải thích chi tiết từng metric) |
| `FedE/eval_paper_faithful.py` | Paper-faithful eval framework (committed earlier f691387) |
| `FedE/scripts/download_paper_data.py` | HF data fetcher (earlier) |
| `FedE/scripts/inspect_paper_data.py` | Schema inspector (earlier) |

### Local artifacts (không trong git, giữ tại `.agents/`)

```
.agents/
├── HANDOFF_session2_to_phase6.md           # File này
├── plans/
│   ├── complete-fede4rag-paper-faithful-evaluation.md   # Plan Phase 1-5 (Phase 1-4 done)
│   └── improve-fede-recipe-paper-rag-ft-kd-gle.md       # Plan Phase 6 (chưa execute) ← cần cho restart
├── validation_artifacts/
│   ├── checkpoints/        # 7 LoRA checkpoints (~5 MB total)
│   ├── eval_outputs/       # 8 paper-faithful eval JSON results
│   └── logs/               # 7 logs (5 training + eval orchestration)
├── remote_snapshot/        # Remote file snapshots dùng cho diff
└── scripts/                # SSH/poll helper scripts (download, inspect, eval polling)
```

## Khi restart server cho Phase 6 — hỏi tôi gì cũng được

Tôi đã lưu vào memory:
- `setup_gpu_server_recipe.md` — cập nhật Phase 6 quick-start (Section "Phase 6 quick-start" cuối file)
- `paper_faithful_eval_session2.md` — progress + numbers + 4 root causes
- `phase1_6_validation_results.md` — validation cũ (Phase 1-6 v1)
- `fede_pipeline_pitfalls.md` — 10 bugs đã catch
- `MEMORY.md` — index

→ Khi user nói "tôi đã restart server, IP là X, làm tiếp Phase 6", tôi sẽ:
1. Đọc memory
2. Check `.agents/plans/improve-fede-recipe-paper-rag-ft-kd-gle.md`
3. Setup server theo recipe (cùng commands như cũ)
4. Execute Phase 1 (regenerate training data) → Phase 2 (revert KL→MSE) → Phase 3 (per-sample InfoNCE) → ...

## Các quyết định chính cần nhớ cho Phase 6

| Quyết định | Lý do |
|---|---|
| **Revert Phase 3 KL→MSE** trong `core.py` | Paper §3 dùng MSE on similarity matrices, ta đã đi lệch |
| **Regenerate `selected_data.json`** từ 368-doc `train_corpus.json` | Hiện tại chỉ có 5 companies, eval 43+ → mismatch |
| **Per-sample DP InfoNCE**: cache batch refs outside | Cách duy nhất để có B-1 negatives mỗi query mà preserve per-sample DP |
| **Hyperparams**: 25→50 rounds, batch 8→16 | Paper claim batch=16 optimal |
| **Giữ BGE-base** (không switch BGE-large) | Paper-fidelity, tránh tăng VRAM 3× |
| **Không hard negative mining** | Paper không dùng |
| **Cross-encoder re-rank**: optional Phase 5 | Eval-side improvement, không bắt buộc |

## 4 root causes (theo thứ tự fix Phase 6)

1. **Data diversity**: 5 → 368 docs (Phase 6.1)
2. **Loss off-paper**: KL → MSE (Phase 6.2)
3. **DP per-sample loss yếu**: `1-cos_sim` → InfoNCE+MSE-KD (Phase 6.3)
4. **Hyperparams**: 25/8 → 50/16 (Phase 6.4)

## Acceptance criteria sau Phase 6

| Metric | Hiện tại | Target | Paper |
|---|---|---|---|
| Val MRR (DP) | 0.10 | **>0.20** | 0.71 |
| Val Hit@1 (best setup) | 0% | **>5%** | 87% |
| Val EM | 6% | **>20%** | 52% |
| lora_B std (DP) | 0.000173 | **>0.001** | — |
| Existing tests pass | ✅ | ✅ | — |

## Cost ước tính Phase 6

~$5-7 GPU + ~5h human work. Tổng cumulative đến nay ~$15-17 (Phase 1-6 v1 cũ + paper-faithful Phase 1-4 mới).

## TODOs khi user nói "restart cho Phase 6"

- [ ] User cung cấp IP server mới
- [ ] Setup server theo recipe (clone, venv, deps)
- [ ] Upload paper_test_data/ (138 MB) HOẶC chạy `download_paper_data.py` trên remote
- [ ] Execute Phase 1: `regenerate_training_data.py` (chưa code, theo plan task 1.1)
- [ ] Execute Phase 2: edit `core.py:compute_client_loss` (KL → MSE)
- [ ] Execute Phase 3: rewrite `fedrag_lora.py:_train_dp` với per-sample InfoNCE+MSE
- [ ] Execute Phase 4: update `main_lora.py` (rounds=50, batch=16)
- [ ] Run training trên Vast.ai (~3h)
- [ ] Run `bash scripts/run_all_evals.sh` (~30 min)
- [ ] So sánh với current numbers trong `paper_faithful_eval_report_vi.md`
- [ ] Update report với Phase 6 results

## Server state cuối session 2

Server `178.128.239.48` đã idle. User sẽ tắt qua DigitalOcean console. Branch `feature/validate_old_data` đã sync với commit `9adedd5`.

## Tự tin (confidence)

- Pipeline đúng kỹ thuật: ✅ verified
- Memory recipe + plan đủ chi tiết để restart: ✅
- Phase 6 expected improvement: 7/10 (Phase 3 implementation novel, có rủi ro)

---

**Nếu có gì lạ khi restart, hỏi tôi để debug. Memory đã đủ context để pick up không break.**
