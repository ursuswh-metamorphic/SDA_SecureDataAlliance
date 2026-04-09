import os, sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import flgo
import flgo.algorithm.fedrag_dp as fedrag_dp

sys.path.insert(0, os.path.dirname(__file__))
from privacy.rdp_accountant import compute_epsilon, find_noise_multiplier

TARGET_EPSILON    = 20.0
TARGET_DELTA      = 1e-5
NUM_ROUNDS        = 25
NUM_CLIENTS       = 5
CLIENTS_PER_ROUND = 3
BATCH_SIZE        = 8

sampling_rate = CLIENTS_PER_ROUND / NUM_CLIENTS

calibrated_sigma = find_noise_multiplier(TARGET_EPSILON, NUM_ROUNDS, sampling_rate, TARGET_DELTA)
print(f'Calibrated sigma = {calibrated_sigma:.4f} for eps={TARGET_EPSILON}')

eps_check, best_alpha = compute_epsilon(NUM_ROUNDS, calibrated_sigma, sampling_rate, TARGET_DELTA)
print(f'Verification: sigma={calibrated_sigma:.4f} -> eps={eps_check:.4f}')

task = './num5_alpha05'
config = {
    'benchmark': {'name': 'flgo.benchmark.fedrag_classification'},
    'partitioner': {'name': 'IIDPartitioner', 'para': {'num_clients': NUM_CLIENTS}},
}
if not os.path.exists(task):
    flgo.gen_task(config, task_path=task)

dp_runner = flgo.init(
    task=task, algorithm=fedrag_dp,
    option={
        'num_rounds': NUM_ROUNDS, 'num_epochs': 1, 'gpu': 0,
        'batch_size': BATCH_SIZE, 'learning_rate': 0.00001,
        'dp_enabled': True,
        'target_epsilon': TARGET_EPSILON, 'target_delta': TARGET_DELTA,
        'server_clip_norm': 1.0, 'server_noise_multiplier': calibrated_sigma,
        'dp_clients_per_round': CLIENTS_PER_ROUND,
        'dp_clip_norm': 1.0, 'dp_noise_multiplier': calibrated_sigma,
        'dp_adaptive_clip': True, 'dp_clip_gamma': 0.5,
    }
)
print(f'[DP-FedRAG eps={TARGET_EPSILON}] Starting training...')
dp_runner.run()
print(f'[DP-FedRAG eps={TARGET_EPSILON}] Training complete.')
