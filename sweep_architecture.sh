#!/bin/bash
#SBATCH --job-name=gambl_arch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-5
#SBATCH --output=/home1/plwang/GAMBL/logs/arch_%A_%a.out
#SBATCH --error=/home1/plwang/GAMBL/logs/arch_%A_%a.err

# Sweep: model in [transformer, mlp] x seed in [0, 1, 2]
# = 6 jobs total.
# dropout=0.0 throughout for fair comparison.
# Key question: does MLP also grokk, or is it transformer-specific?

set -euo pipefail

PROJECT=/home1/plwang/GAMBL
RESULTS=/scratch1/plwang/gambl_results

cd "$PROJECT"
mkdir -p logs "$RESULTS"

source /spack/conda/miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate /scratch1/plwang/envs/gambl

MODELS=(transformer mlp)
SEEDS=(0 1 2)

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Job ID:   $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "model:    $MODEL"
echo "seed:     $SEED"
echo "Started:  $(date)"

python train.py \
    --model "$MODEL" \
    --seed "$SEED" \
    --dropout 0.0 \
    --n_steps 100000 \
    --results_dir "$RESULTS" \
    --device auto

echo "Finished: $(date)"
