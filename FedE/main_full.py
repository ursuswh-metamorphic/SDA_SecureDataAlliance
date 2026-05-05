"""
main_full.py — Phase 6 single entrypoint: full upstream pipeline.

Wires every privacy/efficiency layer into ONE flgo run:
  * Phase 1   LoRA-only state transport          (fedrag_lora_ckks inherits)
  * Phase 2   per-sample DP-SGD on LoRA params   (option['dp_enabled']=True)
  * Phase 3   InfoNCE + KL distillation          (fedrag_core defaults flipped)
  * Phase 4   true homomorphic CKKS aggregation  (algorithm=fedrag_lora_ckks)
  * Phase 5   4-bit qLoRA base                   (Linux only, auto-gated)

Env-var toggles:
    DP_ENABLED=1     -> per-sample DP (default: 1 here, since the whole point
                        of the full pipeline is the 3-layer protection).
    USE_QLORA=1      -> 4-bit base (auto-disabled on Windows / no bitsandbytes).
    FHE_ENABLED=0    -> drop back to plain LoRA aggregation (debug).

Three baselines remain runnable side-by-side:
    main.py              -> FedRAG baseline (no DP, no LoRA, no FHE)
    main_dp.py           -> User-level DP on full model (no LoRA, no FHE)
    main_dp_lora_eps20.py-> Standalone LoRA+DP (bypasses flgo, no FHE)
    main_lora.py         -> Phase 1+2+3+5 only (no FHE)
    main_full.py         -> THIS (Phase 1+2+3+4+5 = full pipeline)

Acceptance target: end-of-run ε ≤ 20, retention within 2 % of main_lora.py
on PubMed (decrypt the final cipher with the secret-key client to evaluate).
"""
import os
import platform

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import flgo

# ── Algorithm selection ─────────────────────────────────────────────────────
# Default to the FHE-enabled fedrag_lora_ckks. FHE_ENABLED=0 is a debug knob
# that drops back to fedrag_lora (Phase 1+2+3 only) without re-running on a
# different file.
FHE_ENABLED = os.environ.get('FHE_ENABLED', '1') == '1'
if FHE_ENABLED:
    import flgo.algorithm.fedrag_lora_ckks as algorithm
    _alg_name = 'fedrag_lora_ckks'
else:
    import flgo.algorithm.fedrag_lora as algorithm
    _alg_name = 'fedrag_lora'

# ── Phase 3 loss hyperparameters ────────────────────────────────────────────
from flgo.benchmark.fedrag_classification import core as fedrag_core
from flgo.benchmark.fedrag_classification import config as fedrag_config
fedrag_core.DEFAULT_TEMPERATURE = 0.05
fedrag_core.DEFAULT_KD_WEIGHT = 1.0

# ── Phase 5 qLoRA gate ──────────────────────────────────────────────────────
_USE_QLORA_REQUESTED = os.environ.get('USE_QLORA', '1') == '1'  # default on for Phase 6
if _USE_QLORA_REQUESTED:
    if platform.system() != 'Linux':
        print(f'[main_full] USE_QLORA=1 requested but platform is {platform.system()}; '
              f'forcing use_qlora=False (bitsandbytes is Linux/CUDA only).')
        USE_QLORA = False
    elif not fedrag_config._QLORA_AVAILABLE:
        print(f'[main_full] USE_QLORA=1 requested but bitsandbytes/PEFT kbit '
              f'helpers unavailable; forcing use_qlora=False.')
        USE_QLORA = False
    else:
        USE_QLORA = True
else:
    USE_QLORA = False
fedrag_config.DEFAULT_USE_QLORA = USE_QLORA

# ── Task setup ──────────────────────────────────────────────────────────────
# Use a separate task path so other entrypoints' tasks stay untouched.
task = './num5_alpha05_full'
task_config = {
    'benchmark': {'name': 'flgo.benchmark.fedrag_classification'},
    'partitioner': {'name': 'IIDPartitioner', 'para': {'num_clients': 5}},
}
if not os.path.exists(task):
    flgo.gen_task(task_config, task_path=task)

# ── DP setup (default ON for full pipeline) ────────────────────────────────
DP_ENABLED = os.environ.get('DP_ENABLED', '1') == '1'

option = {
    # FL
    'num_rounds': 25,
    'num_epochs': 1,
    'gpu': 0,
    'batch_size': 16 if USE_QLORA else 8,
    'learning_rate': 1e-5,
    'num_clients': 5,
    'use_qlora': USE_QLORA,

    # Phase 2: DP
    'dp_enabled': DP_ENABLED,
    'target_epsilon': 20.0,
    'target_delta': 1e-5,
    'dp_clip_norm': 0.1,
    'dp_clients_per_round': 5,

    # Phase 3: loss (logged for traceability)
    'temperature': fedrag_core.DEFAULT_TEMPERATURE,
    'kd_weight': fedrag_core.DEFAULT_KD_WEIGHT,

    # Phase 4: FHE flag (the algorithm class is the actual gate)
    'fhe_enabled': FHE_ENABLED,
}

print('=' * 68)
print(f'[main_full] Phase 6 full upstream pipeline')
print(f'  algorithm   = flgo.algorithm.{_alg_name}')
print(f'  DP          = {DP_ENABLED} (eps={option["target_epsilon"]}, '
      f'delta={option["target_delta"]}, clip={option["dp_clip_norm"]})')
print(f'  qLoRA       = {USE_QLORA}')
print(f'  FHE         = {FHE_ENABLED}')
print(f'  rounds      = {option["num_rounds"]}, '
      f'clients={option["num_clients"]}, batch={option["batch_size"]}')
print(f'  loss        = InfoNCE(tau={option["temperature"]}) '
      f'+ {option["kd_weight"]}*KL_div')
print(f'  task path   = {task}')
print('=' * 68)

runner = flgo.init(task=task, algorithm=algorithm, option=option)
runner.run()

# ── Phase 4 post-run: persist final encrypted cipher ──────────────────────
# When FHE is on, the server's plaintext self.model is stale (never updated).
# The real trained state lives only in self._last_aggregated_cipher. Save it
# to disk so a secret-key holder can decrypt offline for evaluation.
if FHE_ENABLED:
    final_cipher_path = os.path.join('checkpoints', 'final_cipher.pt')
    try:
        runner.server.save_final_cipher(final_cipher_path)
    except AttributeError:
        # runner exposes server differently across flgo versions; try the
        # most common attribute names.
        srv = getattr(runner, 'server', None) or getattr(runner, 'sv', None)
        if srv is not None and hasattr(srv, 'save_final_cipher'):
            srv.save_final_cipher(final_cipher_path)
        else:
            print('[main_full] Warning: could not locate runner.server; '
                  'skipping final_cipher dump.')
