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
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# This pipeline is PyTorch-only. Avoid importing an unrelated TensorFlow
# installation (and its binary NumPy ABI) through transformers.
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import flgo
import flgo.algorithm.fedrag_lora as fedrag_lora
import flgo.algorithm.fedrag_fedprox as fedrag_fedprox
import flgo.algorithm.fedrag_client_dp as fedrag_client_dp
# Phase 3: override loss hyperparameters BEFORE flgo.init() instantiates
# the TaskCalculator (which reads DEFAULT_TEMPERATURE / DEFAULT_KD_WEIGHT
# at __init__).
from flgo.benchmark.fedrag_classification import core as fedrag_core
from flgo.benchmark.fedrag_classification import config as fedrag_config
from lora_hparams import resolve_lora_hparams
MODEL_NAME = os.environ.get('MODEL_NAME', 'BAAI/bge-base-en-v1.5')
POOLING = os.environ.get('POOLING', 'cls')
if POOLING not in {'cls', 'masked_mean'}:
    raise ValueError(f'POOLING must be cls or masked_mean, got {POOLING}')
fedrag_core.DEFAULT_TEMPERATURE = float(os.environ.get('TEMPERATURE', '0.05'))
fedrag_core.DEFAULT_KD_WEIGHT = float(os.environ.get('KD_WEIGHT', '1.0'))  # KD_WEIGHT=0 ablates KD-GLE
fedrag_core.DEFAULT_MODEL_NAME = MODEL_NAME
fedrag_core.DEFAULT_POOLING = POOLING
fedrag_config.DEFAULT_MODEL_NAME = MODEL_NAME
LORA_HPARAMS = resolve_lora_hparams(os.environ)
fedrag_config.LORA_R = LORA_HPARAMS['rank']
fedrag_config.LORA_ALPHA = LORA_HPARAMS['alpha']
fedrag_config.LORA_DROPOUT = LORA_HPARAMS['dropout']

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

# Use a versioned task path so legacy IID tasks stay untouched and re-runnable.
FEDE_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CLIENTS = int(os.environ.get('NUM_CLIENTS', '5'))
if NUM_CLIENTS <= 0:
    raise ValueError('NUM_CLIENTS must be positive')
PARTITIONER_NAME = os.environ.get('PARTITIONER', 'company_lpt').strip().lower()
if PARTITIONER_NAME not in {'company_lpt', 'paper_five'}:
    raise ValueError(
        'PARTITIONER must be company_lpt or paper_five, '
        f'got {PARTITIONER_NAME}'
    )
if PARTITIONER_NAME == 'paper_five' and NUM_CLIENTS != 5:
    raise ValueError('PARTITIONER=paper_five requires NUM_CLIENTS=5')
task = os.environ.get(
    'TASK_PATH',
    os.path.join(
        FEDE_DIR,
        'num1_central_clean' if NUM_CLIENTS == 1
        else (
            'num5_paper_five'
            if PARTITIONER_NAME == 'paper_five'
            else f'num{NUM_CLIENTS}_company_clean'
        ),
    ),
)
train_data_arg = os.environ.get('TRAIN_DATA', 'selected_data_clean.json')
train_data = (
    train_data_arg if os.path.isabs(train_data_arg)
    else os.path.join(FEDE_DIR, train_data_arg)
)
if not os.path.exists(train_data):
    raise FileNotFoundError(
        f'Clean training data is required but missing: {train_data}. '
        f'Run FedE/tools/clean_training_data.py first.'
    )
DATA_SEED = int(os.environ.get('DATA_SEED', '13'))
GPU_OPTION = os.environ.get('GPU', 'auto').strip().lower()
if GPU_OPTION == 'cpu' or not torch.cuda.is_available():
    RUNTIME_GPU = []
elif GPU_OPTION == 'auto':
    RUNTIME_GPU = 0
else:
    RUNTIME_GPU = int(GPU_OPTION)
config = {
    'benchmark': {'name': 'flgo.benchmark.fedrag_classification'},
    'partitioner': {
        'name': (
            'PaperFiveCompanyPartitioner'
            if PARTITIONER_NAME == 'paper_five'
            else 'CompanyPartitioner'
        ),
        'para': {'num_clients': NUM_CLIENTS},
    },
}
if not os.path.exists(task):
    flgo.gen_task(
        config, task_path=task, rawdata_path=train_data_arg, seed=DATA_SEED
    )
if os.environ.get('PREPARE_TASK_ONLY', '0') == '1':
    print(f'[main_lora] Company-partitioned task ready: {task}')
    raise SystemExit(0)

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
    'num_rounds': int(os.environ.get('NUM_ROUNDS', '50')),
    'num_epochs': 1,
    'gpu': RUNTIME_GPU,
    'batch_size': int(os.environ.get('BATCH_SIZE', '16')),
    'learning_rate': LORA_HPARAMS['learning_rate'],
    'num_clients': NUM_CLIENTS,
    'num_steps': int(os.environ.get('NUM_STEPS', '50')),
    'seed': int(os.environ.get('SEED', '13')),
    'dataseed': DATA_SEED,
    'use_qlora': USE_QLORA,   # logged for traceability; actual gate is config.DEFAULT_USE_QLORA

    # ── Phase 2 DP knobs (mirror main_dp_lora_eps20.py:29-38) ─────────────
    'dp_enabled': DP_ENABLED,
    'target_epsilon': TARGET_EPS,
    'target_delta': 1e-5,
    'dp_clip_norm': 0.1,
    'dp_clients_per_round': int(
        os.environ.get('DP_CLIENTS_PER_ROUND', str(NUM_CLIENTS))
    ),
    # 'dp_noise_multiplier' is computed by Server._setup_dp() via find_noise_multiplier.

    # ── Phase 3 loss knobs (logged here for traceability; actually plumbed
    # via fedrag_core.DEFAULT_TEMPERATURE / DEFAULT_KD_WEIGHT above) ─────
    'temperature': fedrag_core.DEFAULT_TEMPERATURE,
    'kd_weight': fedrag_core.DEFAULT_KD_WEIGHT,
    'pooling': POOLING,
    'lora_r': LORA_HPARAMS['rank'],
    'lora_alpha': LORA_HPARAMS['alpha'],
    'lora_dropout': LORA_HPARAMS['dropout'],
    'fedprox_mu': float(os.environ.get('FEDPROX_MU', '0.01')),
    'client_dp_mode': os.environ.get('CLIENT_DP_MODE', 'noise0').lower(),
    'client_dp_clip_norm': float(
        os.environ.get('CLIENT_DP_CLIP_NORM', '1.0')
    ),
    'client_dp_target_epsilon': float(
        os.environ.get('CLIENT_DP_TARGET_EPS', str(TARGET_EPS))
    ),
    'client_dp_target_delta': float(
        os.environ.get('CLIENT_DP_TARGET_DELTA', '1e-5')
    ),
    'client_dp_clients_per_round': int(
        os.environ.get('CLIENT_DP_CLIENTS_PER_ROUND', str(NUM_CLIENTS))
    ),
    'client_dp_noise_multiplier': (
        float(os.environ['CLIENT_DP_NOISE_MULTIPLIER'])
        if 'CLIENT_DP_NOISE_MULTIPLIER' in os.environ
        else None
    ),
}

ALGORITHM_NAME = os.environ.get('ALGORITHM', 'fedavg').strip().lower()
if ALGORITHM_NAME == 'fedavg':
    algorithm = fedrag_lora
elif ALGORITHM_NAME == 'fedprox':
    algorithm = fedrag_fedprox
elif ALGORITHM_NAME == 'client_dp':
    algorithm = fedrag_client_dp
else:
    raise ValueError(
        f'ALGORITHM must be fedavg, fedprox, or client_dp, got {ALGORITHM_NAME}'
    )
if algorithm is fedrag_fedprox and DP_ENABLED:
    raise ValueError('FedProx+DP is outside the B4 non-DP baseline protocol')
if algorithm is fedrag_client_dp:
    if DP_ENABLED:
        raise ValueError(
            'client_dp uses server-side client-level DP; set DP_ENABLED=0 '
            'to prevent record-level DP-SGD from being composed accidentally'
        )
    if not USE_LORA:
        raise ValueError('client_dp currently requires USE_LORA=1')

print(f'[main_lora] algorithm={ALGORITHM_NAME}, '
      f'partitioner={PARTITIONER_NAME}, '
      f'fedprox_mu={option["fedprox_mu"]}, '
      f'DP_ENABLED={DP_ENABLED}, USE_QLORA={USE_QLORA}, '
      f'device={"cpu" if RUNTIME_GPU == [] else f"cuda:{RUNTIME_GPU}"}, '
      f'model={MODEL_NAME}, pooling={POOLING}, '
      f'eps={option["target_epsilon"]}, '
      f'rounds={option["num_rounds"]}, clients={option["num_clients"]}, '
      f'batch={option["batch_size"]}, num_steps={option["num_steps"]}, '
      f'lr={option["learning_rate"]}, lora_r={option["lora_r"]}, '
      f'lora_alpha={option["lora_alpha"]}, '
      f'lora_dropout={option["lora_dropout"]}, '
      f'client_dp_mode={option["client_dp_mode"]}, '
      f'client_dp_clip={option["client_dp_clip_norm"]}, '
      f'clip={option["dp_clip_norm"]}, '
      f'tau={option["temperature"]}, kd_weight={option["kd_weight"]}')
if DP_ENABLED:
    print(f'[main_lora] DP: σ will be calibrated for {option["num_rounds"]} rounds, '
          f'q={option["dp_clients_per_round"]}/{option["num_clients"]}=1.0, '
          f'δ={option["target_delta"]}; expected σ≈1.83 for ε=20 over 50 rounds.')

runner = flgo.init(task=task, algorithm=algorithm, option=option)
runner.run()
