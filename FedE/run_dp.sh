#!/bin/sh
# DP-FedRAG Training with formal (ε,δ)-DP guarantee
# Usage: bash run_dp.sh

main_program="main_dp.py"
conda_env="fedrag"
log_folder="logs"
dataset="law_dp"
current_time=$(date +'%Y-%m-%d_%H-%M-%S')
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $conda_env
mkdir -p $log_folder

error_log="$log_folder/${current_time}_${dataset}_error.log"
output_log="$log_folder/${current_time}_${dataset}_output.log"

echo "=== DP-FedRAG Training ==="
echo "Output log: $output_log"
echo "Error  log: $error_log"
echo "=========================="

python $main_program > $output_log 2> $error_log
echo "Training finished. Check logs."
