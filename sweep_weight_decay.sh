#!/bin/bash
#SBATCH --job-name=gambl_wd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-14
#SBATCH --output=/home1/plwang/GAMBL/logs/wd_%A_%a.out
#SBATCH --error=/home1/plwang/GAMBL/logs/wd_%A_%a.err

# Sweep: weight_decay in [0.0, 0.1, 0.5, 1.0, 5.0] x seed in [0, 1, 2]
# = 15 jobs total.
# dropout=0.0 throughout — required for clean grokking.
# Expected result: only wd >= 0.5 should show grokking; wd=0.0 should not.

set -euo pipefail

PROJECT=/home1/plwang/GAMBL
RESULTS=/scratch1/plwang/gambl_results

cd "$PROJECT"
mkdir -p logs "$RESULTS"

source /spack/conda/miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate /scratch1/plwang/envs/gambl

WD_VALUES=(0.0 0.1 0.5 1.0 5.0)
SEEDS=(0 1 2)

WD_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

WD=${WD_VALUES[$WD_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Job ID:         $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:           $SLURMD_NODENAME"
echo "GPU:            $CUDA_VISIBLE_DEVICES"
echo "weight_decay:   $WD"
echo "seed:           $SEED"
echo "Started:        $(date)"

python train.py \
    --model transformer \
    --weight_decay "$WD" \
    --seed "$SEED" \
    --dropout 0.0 \
    --n_steps 100000 \
    --results_dir "$RESULTS" \
    --device auto

echo "Finished: $(date)"
