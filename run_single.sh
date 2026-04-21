#!/bin/bash
#SBATCH --job-name=gambl_single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/home1/plwang/GAMBL/logs/single_%j.out
#SBATCH --error=/home1/plwang/GAMBL/logs/single_%j.err

# Canonical grokking reproduction (Power et al. 2022 settings).
# dropout=0.0 is critical — dropout prevents the clean memorization phase
# that grokking requires.
#
# Usage:
#   sbatch run_single.sh                      # canonical transformer baseline
#   sbatch run_single.sh --model mlp          # MLP instead
#   sbatch run_single.sh --weight_decay 0.0   # no weight decay (control)

set -euo pipefail

PROJECT=/home1/plwang/GAMBL
RESULTS=/scratch1/plwang/gambl_results

cd "$PROJECT"
mkdir -p logs "$RESULTS"

source /spack/conda/miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate /scratch1/plwang/envs/gambl

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "Args:     $@"
echo "Started:  $(date)"

python train.py \
    --model transformer \
    --n_steps 100000 \
    --dropout 0.0 \
    --results_dir "$RESULTS" \
    --device auto \
    "$@"

echo "Finished: $(date)"
