"""
DP-FedRAG Training Entry Point
================================
Demonstrates the improved DP-FedRAG algorithm with:
  - Formal (ε, δ)-DP guarantee via RDP accounting
  - Per-sample gradient clipping at clients
  - User-level DP aggregation at server
  - Adaptive clip norm (Andrew et al., NeurIPS 2021)
  - Privacy amplification by client subsampling

Usage:
    python main_dp.py

Configuration guide:
    target_epsilon:           Set the desired ε budget (lower = stronger privacy)
    target_delta:             Set δ << 1/N (N = total users/clients)
    server_noise_multiplier:  σ ≥ 0.8 for formal guarantee (σ=1.1 → ε≈5 for 25 rounds)
    server_clip_norm:         C = 1.0 is a safe default; adaptive clipping will tune it
    dp_clients_per_round:     Use < total_clients for amplification  (e.g., 3 out of 5)
    dp_noise_multiplier:      σ for client per-sample DP-SGD (same as server if desired)

Privacy calibration:
    The code automatically calibrates σ to match target_epsilon.
    Alternatively, use privacy/rdp_accountant.py::find_noise_multiplier() manually.

References:
    [1] Abadi et al. (CCS 2016) — DP-SGD foundation
    [2] McMahan et al. (ICLR 2018) — User-level DP FedAvg
    [3] Mironov (CSF 2017) — RDP accounting
    [4] Andrew et al. (NeurIPS 2021) — Adaptive clipping
    [5] Balle et al. (NeurIPS 2018) — Privacy amplification by subsampling
"""

import os
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import flgo
import flgo.algorithm.fedrag_dp as fedrag_dp  # Our improved DP algorithm

# ── Pre-flight: Print calibrated σ for given ε target ─────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from privacy.rdp_accountant import compute_epsilon, find_noise_multiplier

TARGET_EPSILON    = 8.0    # Desired total privacy budget ε
TARGET_DELTA      = 1e-5   # δ << 1/N_clients
NUM_ROUNDS        = 25     # T — total communication rounds
NUM_CLIENTS       = 5      # K — total clients
CLIENTS_PER_ROUND = 3      # C — clients sampled per round (q = C/K = 0.6 amplification)
BATCH_SIZE        = 8      # Local batch size

# NOTE on privacy amplification:
#   q = CLIENTS_PER_ROUND / NUM_CLIENTS = 3/5 = 0.6 (60% per round)
#   Lower q → stronger amplification → lower ε at same σ
#   Recommendation: q ≤ 0.3 for meaningful amplification (e.g., 1-2 clients out of 5)
#   Trade-off: fewer clients → slower convergence

sampling_rate = CLIENTS_PER_ROUND / NUM_CLIENTS  # q = 3/5 = 0.6

# Calibrate σ automatically
try:
    calibrated_sigma = find_noise_multiplier(
        target_epsilon=TARGET_EPSILON,
        num_steps=NUM_ROUNDS,
        sample_rate=sampling_rate,
        delta=TARGET_DELTA,
    )
    print(f"\n{'='*60}")
    print(f"[DP Calibration]")
    print(f"  Target ε     = {TARGET_EPSILON}")
    print(f"  Target δ     = {TARGET_DELTA}")
    print(f"  Num rounds   = {NUM_ROUNDS}")
    print(f"  Sampling q   = {sampling_rate:.3f} ({CLIENTS_PER_ROUND}/{NUM_CLIENTS} clients)")
    print(f"  → Calibrated σ = {calibrated_sigma:.4f}")
    print(f"{'='*60}\n")
    SERVER_SIGMA = calibrated_sigma
except Exception as e:
    print(f"[WARNING] Calibration failed ({e}). Using σ=1.1 as default.")
    SERVER_SIGMA = 1.1

# Verify: what ε does σ=SERVER_SIGMA actually give?
eps_check, best_alpha = compute_epsilon(
    num_steps=NUM_ROUNDS,
    noise_multiplier=SERVER_SIGMA,
    sample_rate=sampling_rate,
    delta=TARGET_DELTA,
)
print(f"[Verification] σ={SERVER_SIGMA:.4f} → ε={eps_check:.4f} at α={best_alpha:.1f}")
print(f"               δ={TARGET_DELTA}, rounds={NUM_ROUNDS}, q={sampling_rate:.3f}\n")

# ── Task setup ─────────────────────────────────────────────────────────────────
task = './num5_alpha05'
config = {
    'benchmark': {'name': 'flgo.benchmark.fedrag_classification'},
    'partitioner': {'name': 'IIDPartitioner', 'para': {'num_clients': NUM_CLIENTS}},
}
if not os.path.exists(task):
    flgo.gen_task(config, task_path=task)

# ── Run DP-FedRAG ───────────────────────────────────────────────────────────────
dp_runner = flgo.init(
    task=task,
    algorithm=fedrag_dp,
    option={
        # ── Standard FL options ────────────────────────────────────────────
        'num_rounds':   NUM_ROUNDS,
        'num_epochs':   1,
        'gpu':          0,
        'batch_size':   BATCH_SIZE,
        'learning_rate': 0.00001,

        # ── DP: Master switch ──────────────────────────────────────────────
        'dp_enabled': True,

        # ── DP: Privacy budget ─────────────────────────────────────────────
        'target_epsilon': TARGET_EPSILON,      # Overall ε budget
        'target_delta':   TARGET_DELTA,        # δ (must be << 1/N)

        # ── DP: Server-side (User-Level DP) ───────────────────────────────
        # McMahan et al. ICLR 2018: clip full client updates, add server noise
        'server_clip_norm':          1.0,       # C_server: initial clip norm
        'server_noise_multiplier':   SERVER_SIGMA,  # σ_server (auto-calibrated)
        'dp_clients_per_round':      CLIENTS_PER_ROUND,  # for amplification

        # ── DP: Client-side (Per-Sample DP-SGD) ───────────────────────────
        # Abadi et al. CCS 2016: per-sample clip + noise
        'dp_clip_norm':          1.0,       # C_client: initial per-sample clip norm
        'dp_noise_multiplier':   SERVER_SIGMA,  # σ_client (same budget)

        # ── DP: Adaptive clipping (Andrew et al. NeurIPS 2021) ────────────
        'dp_adaptive_clip': True,           # Enable adaptive clip norm
        'dp_clip_gamma':    0.5,            # Target: median of update norms (γ=0.5)
    }
)

print("\n[DP-FedRAG] Starting training with formal privacy guarantee...")
dp_runner.run()
print("\n[DP-FedRAG] Training complete. Check logs for final privacy report.")
