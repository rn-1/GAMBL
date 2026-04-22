#!/bin/bash
#SBATCH --job-name=grok_sweep
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep.out
#SBATCH --error=logs/sweep.err
#SBATCH --array=0-2

# Usage:
#   sbatch slurm_sweep.sh <sweep_config.yaml> [--parallel N] [extra args...]
#
# Examples:
#   sbatch slurm_sweep.sh configs/sweep_weight_decay.yaml
#   sbatch slurm_sweep.sh configs/sweep_architecture.yaml --parallel 2
#   sbatch slurm_sweep.sh configs/sweep_train_fraction.yaml --dry-run
#
# Note: With a single GPU, --parallel > 1 will share the GPU across jobs.
#       Leave it at 1 (default) for sequential runs to avoid contention.

set -euo pipefail

export OMP_NUM_THREADS=8

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nanopore

cd /scratch1/rnene/csci567

sweep_configs=("sweep_train_fraction.yaml" "sweep_weight_decay.yaml" "sweep_architecture.yaml")

# mkdir -p logs

# if [[ $# -lt 1 ]]; then
#     echo "ERROR: No sweep config provided."
#     echo "Usage: sbatch slurm_sweep.sh <sweep_config.yaml> [--parallel N] [extra args...]"
#     exit 1
# fi

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "Config: ${sweep_configs[$SLURM_ARRAY_TASK_ID]}"

python run_experiment.py configs/${sweep_configs[$SLURM_ARRAY_TASK_ID]}

echo
echo "Finished:  $(date)"
