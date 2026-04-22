#!/bin/bash
#SBATCH --job-name=grok_train
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/train.out
#SBATCH --error=logs/train.err

# Usage:
#   sbatch slurm_train.sh [train.py args...]
#
# Examples:
#   sbatch slurm_train.sh --model transformer --n_steps 100000
#   sbatch slurm_train.sh --model mlp --weight_decay 0.0 --seed 1

set -euo pipefail

export OMP_NUM_THREADS=8

# Activate conda environment

eval "$(conda shell.bash hook)"

conda activate nanopore

# mkdir -p logs

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "Args:      $*"
echo

cd /scratch1/rnene/csci567/

python train.py "$@"

echo
echo "Finished:  $(date)"
