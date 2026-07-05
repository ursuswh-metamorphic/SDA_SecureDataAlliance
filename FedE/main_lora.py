"""
main_lora.py — Phase 1 + 2 + 3 + 5 + paper-faithful Phase 6 entrypoint.

Runs the FedRAG pipeline through the flgo framework using fedrag_lora.
Defaults follow the FedE4RAG paper (arXiv:2504.19101) §4.1:
    rounds=50, batch_size=16, lr=1e-5, num_clients=5, clip_norm=0.1.
σ for ε=20 at q=1.0, δ=1e-5 over 50 rounds is auto-calibrated (≈1.83).

Loss path uses paper §3 RAG-FT (InfoNCE with τ=0.05) + KD-GLE (MSE on
similarity matrices), not the legacy KL deviation. Both DP and non-DP
paths share the same loss formulation.

Env-var toggles:
    DP_ENABLED=1    -> per-sample DP (paper-faithful InfoNCE + MSE-KD-GLE).
    USE_QLORA=1     -> Phase 5 4-bit base (Linux/Vast.ai only; auto-disabled
                       on Windows or when bitsandbytes is missing).

Phase 4-FHE and Phase 6 (full pipeline) live in main_full.py.
"""
import os
import platform

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import flgo
import flgo.algorithm.fedrag_lora as fedrag_lora
# Phase 3: override loss hyperparameters BEFORE flgo.init() instantiates
# the TaskCalculator (which reads DEFAULT_TEMPERATURE / DEFAULT_KD_WEIGHT
# at __init__).
from flgo.benchmark.fedrag_classification import core as fedrag_core
from flgo.benchmark.fedrag_classification import config as fedrag_config
fedrag_core.DEFAULT_TEMPERATURE = float(os.environ.get('TEMPERATURE', '0.05'))
fedrag_core.DEFAULT_KD_WEIGHT = float(os.environ.get('KD_WEIGHT', '1.0'))  # KD_WEIGHT=0 ablates KD-GLE

# ── Phase 5: qLoRA gate ──────────────────────────────────────────────────────
# Reads env var USE_QLORA. Auto-disabled on non-Linux or when bitsandbytes
# is missing. Must be flipped BEFORE flgo.init() because get_model() consumes
# the DEFAULT_USE_QLORA flag at fedllm.Model() instantiation time.
_USE_QLORA_REQUESTED = os.environ.get('USE_QLORA', '0') == '1'
if _USE_QLORA_REQUESTED:
    if platform.system() != 'Linux':
        print(f'[main_lora] USE_QLORA=1 requested but platform is {platform.system()}; '
              f'forcing use_qlora=False (bitsandbytes is Linux/CUDA only).')
        USE_QLORA = False
    elif not fedrag_config._QLORA_AVAILABLE:
        print(f'[main_lora] USE_QLORA=1 requested but bitsandbytes/PEFT kbit '
              f'helpers unavailable; forcing use_qlora=False.')
        USE_QLORA = False
    else:
        USE_QLORA = True
else:
    USE_QLORA = False
fedrag_config.DEFAULT_USE_QLORA = USE_QLORA

# ── Ablation: full fine-tune gate (Khung 1 cell D/D') ────────────────────────
# USE_LORA=0 → train ALL of BGE-base (no PEFT). qLoRA is meaningless without
# LoRA, so it is forced off. Must be set BEFORE flgo.init() reads get_model().
USE_LORA = os.environ.get('USE_LORA', '1') == '1'
if not USE_LORA:
    print('[main_lora] USE_LORA=0 → FULL fine-tune ablation (no PEFT); qLoRA forced off.')
    USE_QLORA = False
    fedrag_config.DEFAULT_USE_QLORA = False
fedrag_config.DEFAULT_USE_LORA = USE_LORA

# Use a separate task path so the existing main.py baseline ('./num5_alpha05')
# stays untouched and re-runnable.
task = './num5_alpha05_lora'
config = {
    'benchmark': {'name': 'flgo.benchmark.fedrag_classification'},
    'partitioner': {'name': 'IIDPartitioner', 'para': {'num_clients': 5}},
}
if not os.path.exists(task):
    flgo.gen_task(config, task_path=task)

# ── Toggle DP via env var DP_ENABLED=1 (or edit dp_enabled below) ────────────
DP_ENABLED = os.environ.get('DP_ENABLED', '0') == '1'
# Override target epsilon via env var (Phase 7C — privacy-utility tradeoff curve)
# Common values: 0.1 (very strong), 1 (strong/deployment), 5 (moderate-strong),
#                10 (moderate), 20 (default, Phase 6/6.5 baseline), 50, 100 (weak)
TARGET_EPS = float(os.environ.get('TARGET_EPS', '20.0'))
if TARGET_EPS != 20.0:
    print(f'[main_lora] ⚠ TARGET_EPS overridden to {TARGET_EPS} (default 20.0)')

option = {
    # FL hyperparameters — paper §4.1 defaults
    'num_rounds': 50,         # was 25; paper says 22 typically optimal but ε=20 budget allows 50
    'num_epochs': 1,
    'gpu': 0,
    'batch_size': 16,         # paper's stated optimal (vs 8/32). Per-sample DP loop runs B times so this scales encoding cost ~2× vs old batch=8 — tractable on RTX 6000 Ada (48GB) for BGE-base
    'learning_rate': 1e-5,
    'num_clients': 5,
    'num_steps': 50,          # cap per-round steps (mirrors main_dp_lora_eps20.py:87 — needed when client dataset is large, e.g. 33k records / 5 clients)
    'use_qlora': USE_QLORA,   # logged for traceability; actual gate is config.DEFAULT_USE_QLORA

    # ── Phase 2 DP knobs (mirror main_dp_lora_eps20.py:29-38) ─────────────
    'dp_enabled': DP_ENABLED,
    'target_epsilon': TARGET_EPS,
    'target_delta': 1e-5,
    'dp_clip_norm': 0.1,
    'dp_clients_per_round': 5,   # all 5 clients per round → q=1.0 (no amplification)
    # 'dp_noise_multiplier' is computed by Server._setup_dp() via find_noise_multiplier.

    # ── Phase 3 loss knobs (logged here for traceability; actually plumbed
    # via fedrag_core.DEFAULT_TEMPERATURE / DEFAULT_KD_WEIGHT above) ─────
    'temperature': fedrag_core.DEFAULT_TEMPERATURE,
    'kd_weight': fedrag_core.DEFAULT_KD_WEIGHT,
}

print(f'[main_lora] DP_ENABLED={DP_ENABLED}, USE_QLORA={USE_QLORA}, '
      f'eps={option["target_epsilon"]}, '
      f'rounds={option["num_rounds"]}, clients={option["num_clients"]}, '
      f'batch={option["batch_size"]}, num_steps={option["num_steps"]}, '
      f'clip={option["dp_clip_norm"]}, '
      f'tau={option["temperature"]}, kd_weight={option["kd_weight"]}')
if DP_ENABLED:
    print(f'[main_lora] DP: σ will be calibrated for {option["num_rounds"]} rounds, '
          f'q={option["dp_clients_per_round"]}/{option["num_clients"]}=1.0, '
          f'δ={option["target_delta"]}; expected σ≈1.83 for ε=20 over 50 rounds.')

runner = flgo.init(task=task, algorithm=fedrag_lora, option=option)
runner.run()
