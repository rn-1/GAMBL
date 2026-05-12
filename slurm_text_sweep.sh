#!/bin/bash
#SBATCH --job-name=grok_text
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/text_%A_%a.out
#SBATCH --error=logs/text_%A_%a.err
#SBATCH --array=0-2

# Run text-dataset grokking sweeps.
# Array index → sweep config:
#   0: sweep_text_weight_decay   (weight_decay on RTE,  12 jobs)
#   1: sweep_text_train_fraction (train_fraction on RTE, 15 jobs)
#   2: sweep_text_datasets       (4 datasets compared,  12 jobs)
#
# Usage:
#   sbatch --array=0-2 slurm_text_sweep.sh
#
#   # Single sweep only:
#   sbatch --array=0   slurm_text_sweep.sh   # weight decay
#   sbatch --array=1   slurm_text_sweep.sh   # train fraction
#   sbatch --array=2   slurm_text_sweep.sh   # dataset comparison
#
#   # Dry run to preview commands:
#   sbatch --array=0 slurm_text_sweep.sh --dry-run

set -euo pipefail

export OMP_NUM_THREADS=8
# Allow HuggingFace datasets/models to cache inside the project scratch dir
export HF_HOME="${SLURM_SUBMIT_DIR:-/scratch1/rnene/csci567}/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nanopore

cd /scratch1/rnene/csci567
mkdir -p logs "$HF_HOME"

# Serialize dataset downloads across array tasks so they don't race on the
# shared cache.  The first task to acquire the lock downloads everything;
# subsequent tasks wait, then skip via the sentinel file.
_SENTINEL="$HF_HOME/.datasets_ready"
_LOCKFILE="$HF_HOME/.download.lock"
if [ ! -f "$_SENTINEL" ]; then
    echo "Acquiring download lock..."
    (
        flock -x 200
        if [ ! -f "$_SENTINEL" ]; then
            echo "Downloading HuggingFace datasets and tokenizer..."
            python download_text_datasets.py && touch "$_SENTINEL"
        fi
    ) 200>"$_LOCKFILE"
fi

sweep_configs=(
    "sweep_text_weight_decay.yaml"
    "sweep_text_train_fraction.yaml"
    "sweep_text_datasets.yaml"
)

config="${sweep_configs[$SLURM_ARRAY_TASK_ID]}"

echo "Job ID:    $SLURM_JOB_ID"
echo "Array idx: $SLURM_ARRAY_TASK_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "Config:    $config"
echo

python run_experiment.py "configs/${config}" "$@"

echo
echo "Finished:  $(date)"
