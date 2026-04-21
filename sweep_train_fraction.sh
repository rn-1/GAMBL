#!/bin/bash
#SBATCH --job-name=gambl_frac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-17
#SBATCH --output=/home1/plwang/GAMBL/logs/frac_%A_%a.out
#SBATCH --error=/home1/plwang/GAMBL/logs/frac_%A_%a.err

# Sweep: train_fraction in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8] x seed in [0, 1, 2]
# = 18 jobs total.
# dropout=0.0 throughout.
# Expected: lower fractions → earlier/sharper grokking (less data = harder to memorize).

set -euo pipefail

PROJECT=/home1/plwang/GAMBL
RESULTS=/scratch1/plwang/gambl_results

cd "$PROJECT"
mkdir -p logs "$RESULTS"

source /spack/conda/miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate /scratch1/plwang/envs/gambl

FRACS=(0.2 0.3 0.4 0.5 0.6 0.8)
SEEDS=(0 1 2)

FRAC_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

FRAC=${FRACS[$FRAC_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Job ID:         $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:           $SLURMD_NODENAME"
echo "GPU:            $CUDA_VISIBLE_DEVICES"
echo "train_fraction: $FRAC"
echo "seed:           $SEED"
echo "Started:        $(date)"

python train.py \
    --model transformer \
    --train_fraction "$FRAC" \
    --seed "$SEED" \
    --dropout 0.0 \
    --n_steps 100000 \
    --results_dir "$RESULTS" \
    --device auto

echo "Finished: $(date)"
