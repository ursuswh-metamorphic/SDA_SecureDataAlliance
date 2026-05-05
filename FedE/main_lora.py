"""
main_lora.py — Phase 1 + 2 + 3 + 5 entrypoint.

Runs the FedRAG pipeline through the flgo framework using fedrag_lora.
Defaults reproduce the validated standalone DP-LoRA recipe at
main_dp_lora_eps20.py (eps=20, 25 rounds, 5 clients, clip_norm=0.1).

Env-var toggles:
    DP_ENABLED=1    -> Phase 2 per-sample DP (must match 98.4% retention).
    USE_QLORA=1     -> Phase 5 4-bit base (Linux/Vast.ai only; auto-disabled
                       on Windows or when bitsandbytes is missing).

Phase 4 (FHE) and Phase 6 (full pipeline) live in main_full.py.
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
fedrag_core.DEFAULT_TEMPERATURE = 0.05
fedrag_core.DEFAULT_KD_WEIGHT = 1.0

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

option = {
    # FL hyperparameters
    'num_rounds': 25,
    'num_epochs': 1,
    'gpu': 0,
    # Phase 5: qLoRA frees ~2/3 of VRAM, so we can double batch size from 8 → 16
    # when 4-bit is active. Falls back to 8 on the non-quantized path.
    'batch_size': 16 if USE_QLORA else 8,
    'learning_rate': 1e-5,
    'num_clients': 5,
    'num_steps': 50,          # cap per-round steps (mirrors main_dp_lora_eps20.py:87 — needed when client dataset is large, e.g. 43k records / 5 clients)
    'use_qlora': USE_QLORA,   # logged for traceability; actual gate is config.DEFAULT_USE_QLORA

    # ── Phase 2 DP knobs (mirror main_dp_lora_eps20.py:29-38) ─────────────
    'dp_enabled': DP_ENABLED,
    'target_epsilon': 20.0,
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
      f'batch={option["batch_size"]}, clip={option["dp_clip_norm"]}, '
      f'tau={option["temperature"]}, kd_weight={option["kd_weight"]}')

runner = flgo.init(task=task, algorithm=fedrag_lora, option=option)
runner.run()
