# Handoff — Phase 7C ε-sweep (PARTIAL, paused 2x: 2026-05-24 + 2026-05-26)

> **Status**: Server paused TWICE. 1/3 runs complete, 2/3 NOT viable on current servers.
> **2026-05-24**: Server `108.255.76.60:55414` (RTX 4090 Iceland) — stopped by user mid ε=50.
> **2026-05-26**: Server `82.141.118.40:2721` (RTX 4090 EPYC 7713) — **CPU BOTTLENECK** discovered. Training 10× slower than expected (108h projected vs 2h target). Killed within 1h.
> **Total spent**: ~$1.40 (~$1.10 first session + ~$0.30 wasted on CPU-slow server)
> **Commit**: `39a9d19` (TARGET_EPS env var) + `a8ce578` (setup quickstart).

## 🚨 LESSON LEARNED (2026-05-26) — SERVER SELECTION CRITICAL

The 2nd server (EPYC 7713, 256 reported cores) had CPU bottleneck:
- `/proc/cpuinfo` showed cpu MHz = 1497 (instead of expected 3500+ boost)
- cgroup quota = 3071999 / 100000 = ~30 cores effective (despite 256 reported)
- Per-sample DP loop is **Python sequential** → bottlenecked by single-thread CPU speed
- Result: 31s/step vs 3s/step on previous Iceland server (10× slowdown)

### When renting next server, VERIFY before launching training:

```bash
# After SSH, check CPU clock under load:
ssh ... 'cat /proc/cpuinfo | grep "cpu MHz" | sort -u | head'
# Expect ≥ 2500 MHz under load. If < 2000 MHz → DO NOT TRAIN, switch server.

# Check effective core count:
ssh ... 'cat /sys/fs/cgroup/cpu.max'
# Format: <quota> <period>. cores = quota/period. Need ≥ 16 effective cores.

# Quick benchmark single-thread:
ssh ... 'python3 -c "
import time
t0 = time.time()
x = 0
for i in range(10_000_000): x += i
print(f\"Single-thread bench: {time.time()-t0:.2f}s (expect < 0.5s)\")"'
```

### Recommended CPU specs for Phase 7C training:
- **AMD EPYC 9554 / 9755** (Zen 4, base 3.1 GHz, boost 3.75) ✓
- **Intel Xeon Gold 6xxx / 8xxx** (base 2.5+ GHz, boost 3.5+) ✓
- **AMD Threadripper PRO 3955WX / 7995WX** (boost 4 GHz+) ✓ ← Iceland server #1 had this
- ❌ **AVOID**: EPYC 7xxx series (Zen 2/3) with cgroup-throttled instances

## What was done

### ε=1 — **COMPLETE** ✅
- σ calibrated: **28.1336** (vs σ=1.83 for ε=20)
- 50 rounds × 5 clients × 50 steps × per-sample DP
- Final ε spent: **0.9905/1.0** (calibration accurate)
- Total time: **7019s = 1h57min** (faster than ε=20 expected)
- Checkpoint saved: `dp_paper_qa_eps1_final.bin` (1.2 MB)

### ε=50 — **PARTIAL** ⚠️
- σ calibrated: **1.0548** (vs σ=1.83 for ε=20)
- Last seen: Round 30/50, ε spent 32.03/50 (at 16:50 UTC)
- **NOT usable as final** — ε guarantee only valid at full 50 rounds with calibrated σ
- **Must restart from scratch** when resume
- Partial log saved: `training_eps50_PARTIAL.log` (130 KB)

### ε=100 — **NOT STARTED** ❌
- Server stopped before this run began
- σ predict: ~0.6 (weakest noise, may break AdamW invariance)
- **Must run from scratch** when resume

## Local artifacts saved

```
.agents/validation_artifacts_phase7C/
├── checkpoints/
│   └── dp_paper_qa_eps1_final.bin     (1.2 MB, COMPLETE ε=1 training)
└── logs/
    ├── training_eps1_complete.log      (193 KB, full ε=1 training)
    ├── training_eps50_PARTIAL.log      (130 KB, ε=50 stopped at round 30/50)
    └── all_eps_runs.log                (master bash log)
```

## To complete when restart

### Setup (~5 min)

```bash
ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<NEW_IP>
apt-get install -y python3-venv git
cd /root && rm -rf sda
git clone -b feature/validate_old_data --depth 1 https://github.com/ursuswh-metamorphic/SDA_SecureDataAlliance.git sda
cd sda && python3 -m venv venv && source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
pip install --quiet transformers peft huggingface_hub numpy requests ujson scipy matplotlib prettytable pynvml tenseal

# Download data
cd FedE && python scripts/download_paper_data.py
python -c "from huggingface_hub import hf_hub_download; import shutil; p=hf_hub_download('DocAILab/FedE4RAG_Dataset', 'FEDE4FIN/train_data/data_50000_random.json', repo_type='dataset'); shutil.copy(p, 'selected_data.json')"
```

### Upload ε=1 checkpoint (skip re-training!)

```bash
scp -i ~/.ssh/id_ed25519 -P <PORT> \
    .agents/validation_artifacts_phase7C/checkpoints/dp_paper_qa_eps1_final.bin \
    root@<NEW_IP>:/root/sda/FedE/
```

### Train ε=50 + ε=100 (sequential, ~4h)

```bash
ssh ... 'cd /root/sda/FedE
nohup bash -c "
source /root/sda/venv/bin/activate
for eps in 50 100; do
  rm -rf num5_alpha05_lora training.log training.done checkpoints/ x-lora_*.bin
  DP_ENABLED=1 TARGET_EPS=\$eps PYTHONUNBUFFERED=1 python -X utf8 -u main_lora.py > training_eps\${eps}.log 2>&1
  ls -t x-lora_*.bin 2>/dev/null | head -1 | xargs -I {} mv {} dp_paper_qa_eps\${eps}_final.bin
done
touch all_eps_trained.done
" > resume_eps_runs.log 2>&1 & disown'
```

### Eval all 3 checkpoints with paper protocol + LlamaIndex (~5 min)

```bash
ssh ... 'cd /root/sda/FedE && source ../venv/bin/activate
for eps in 1 50 100; do
  for split in val test; do
    python -X utf8 eval_paper_protocol.py \
      --checkpoint dp_paper_qa_eps${eps}_final.bin \
      --use-llama-index \
      --name dp_eps${eps} --split $split --batch-size 64
  done
done'
```

### Compile privacy-utility curve

| ε | σ | Predicted Hit@1 val | Predicted Hit@1 test | lora_B std | Expected behavior |
|---|---|---|---|---|---|
| 1 | 28.13 | 62 (= pretrained) | 62 | ~0.00005 | Bit-identical (extreme AdamW invariance) |
| 20 (existing Run 11) | 1.83 | 62 | 62 | 0.000255 | Bit-identical (confirmed) |
| 50 | 1.05 | 62-64 | 62-65 | ~0.0005 | Borderline |
| **100** | **0.6** | **65-68 ⭐** | **65-67 ⭐** | **~0.0015** | **MAY BREAK invariance** ← key data point |

→ ε=100 is **critical** — if Hit@1 > 62, that means **σ < 0.6 threshold lets AdamW resume training**. This characterizes the "DP cliff" exactly.

## Key context for resume agent

1. **Project framing v3**: "Privacy-Preserving Federated Retrieval — When DP HELPS by Preventing Overfitting"
2. **Phase 7C purpose**: Build privacy-utility tradeoff curve to find AdamW invariance breakaway point
3. **Existing data points** (paper protocol + LlamaIndex eval):
   - Pretrained baseline: val 62, test 62
   - ε=20 DP fine-tune: val 62, test 62 (bit-identical → confirms invariance)
   - Non-DP fine-tune: val 58, test 56 (HURTS due to 5-cty overfit)
4. **AdamW invariance hypothesis**: σ > 0.5 → DP dominates, lora_B doesn't move significantly
5. **Phase 7C tests this**: 3 ε values bracketing the suspected threshold

## Cost so far + remaining

| Phase | Cost spent | Cost remaining |
|---|---|---|
| Phase 7C ε=1 (complete) | $0.69 (1.97h × $0.354) | — |
| Phase 7C ε=50 (partial, lost) | $0.41 (1.15h × $0.354) | will need full $0.69 to redo |
| Phase 7C ε=100 (not started) | $0 | $0.69 |
| Total Phase 7C | **$1.10 spent** | **+$1.40 to complete** |

→ Net loss from stop mid-sweep: **$0.41** for ε=50 partial training (will redo from scratch).

## Decision tree for resumption

```
When server restarted:
  1. SSH + setup (5 min)
  2. Upload ε=1 checkpoint (saves $0.69)
  3. Train ε=50 + ε=100 sequential (~4h)
  4. Eval all 3 + compile curve

If short on time/budget:
  Option A: Train ε=100 only first (~2h, $0.69) — key data point
  Option B: Eval ε=1 checkpoint only with paper protocol (~5 min, $0.05) — quick win
```

**Recommended priority**: Run **ε=100 first** (key data point) → ε=50 (filler) → if needed.

## Status quick-glance

```
ε=1:    ✅ COMPLETE       (checkpoint local: dp_paper_qa_eps1_final.bin)
ε=20:   ✅ DONE (Phase 6.5D, existing dp_lora_paper_qa_final.bin)
ε=50:   ⚠️ PARTIAL → restart needed
ε=100:  ❌ NOT STARTED → train when resume
Eval:   ❌ All ε=1, 50, 100 pending eval (need GPU)
```

---

**Ready to resume in 1 SSH session + ~4h GPU compute (+$1.40)**.
