#!/bin/bash
#SBATCH --job-name=grok_sweep
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=5-6

# Array index → sweep config:
#   0: sweep_train_fraction          (train_fraction on mod97,    18 jobs)
#   1: sweep_weight_decay            (weight_decay on mod97,      15 jobs)
#   2: sweep_architecture            (transformer vs MLP,          6 jobs)
#   3: sweep_model_size              (width × depth,              36 jobs)
#   4: sweep_regularization          (weight_decay × dropout,     36 jobs)
#   5: sweep_analogy_train_fraction  (train_fraction on analogy,  18 jobs)
#   6: sweep_analogy_weight_decay    (weight_decay on analogy,    15 jobs)
#
# Usage:
#   sbatch slurm_sweep.sh               # run all seven sweeps in parallel
#   sbatch --array=5-6 slurm_sweep.sh   # analogy sweeps only
#   sbatch --array=5   slurm_sweep.sh   # analogy train_fraction only
#   sbatch --array=6   slurm_sweep.sh   # analogy weight_decay only
#
# Pass extra args to run_experiment.py, e.g.:
#   sbatch slurm_sweep.sh --dry-run
#   sbatch slurm_sweep.sh --parallel 2

set -euo pipefail

export OMP_NUM_THREADS=8

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nanopore

cd /scratch1/rnene/csci567

sweep_configs=(
    "sweep_train_fraction.yaml"
    "sweep_weight_decay.yaml"
    "sweep_architecture.yaml"
    "sweep_model_size.yaml"
    "sweep_regularization.yaml"
    "sweep_analogy_train_fraction.yaml"
    "sweep_analogy_weight_decay.yaml"
)

# mkdir -p logs

# if [[ $# -lt 1 ]]; then
#     echo "ERROR: No sweep config provided."
#     echo "Usage: sbatch slurm_sweep.sh <sweep_config.yaml> [--parallel N] [extra args...]"
#     exit 1
# fi

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "Config:    ${sweep_configs[$SLURM_ARRAY_TASK_ID]}"

python run_experiment.py "configs/${sweep_configs[$SLURM_ARRAY_TASK_ID]}" "$@"

echo
echo "Finished:  $(date)"
