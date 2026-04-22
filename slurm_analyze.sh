#!/bin/bash
#SBATCH --job-name=grok_analyze
#SBATCH --partition=nlp_hiprio
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --output=logs/analyze_%a.out
#SBATCH --error=logs/analyze_%a.err

# Usage:
#   # Analyze all experiments (one array task per experiment):
#   N=$(ls -d results/*/ 2>/dev/null | wc -l); sbatch --array=0-$((N-1)) slurm_analyze.sh
#
#   # Analyze a filtered subset (e.g. only transformer runs):
#   N=$(ls -d results/transformer_*/ 2>/dev/null | wc -l); \
#     sbatch --array=0-$((N-1)) slurm_analyze.sh --pattern "transformer_*"
#
#   # After all per-experiment jobs finish, run sweep summaries:
#   sbatch --array=0-2 slurm_analyze.sh --mode sweep
#
# Modes:
#   per_exp  (default): plot grokking_curve + loss_curve for one experiment per task
#   sweep:              print sweep summary for each sweep config group

set -euo pipefail

export OMP_NUM_THREADS=2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nanopore

cd /scratch1/rnene/csci567
mkdir -p logs figures

# Parse optional args
MODE="per_exp"
PATTERN="*"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)   MODE="$2";    shift 2 ;;
        --pattern) PATTERN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Task ID:   ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:      ${SLURMD_NODENAME:-localhost}"
echo "Started:   $(date)"
echo "Mode:      $MODE"
echo

if [[ "$MODE" == "sweep" ]]; then
    # One task per sweep groupby dimension
    sweep_groups=("weight_decay" "train_fraction" "model")
    groupby="${sweep_groups[${SLURM_ARRAY_TASK_ID:-0}]}"

    case "$groupby" in
        weight_decay)    pat="transformer_mod97_plus_wd*_frac0.5_*" ;;
        train_fraction)  pat="transformer_mod97_plus_wd1.0_frac*_*" ;;
        model)           pat="*_mod97_plus_wd1.0_frac0.5_*" ;;
    esac

    echo "Sweep summary: groupby=$groupby  pattern=$pat"
    python analyze.py \
        --plot sweep_summary \
        --pattern "$pat" \
        --groupby "$groupby" \
        --save

else
    # per_exp mode: collect matching dirs, pick one by array index
    mapfile -t exps < <(
        find results -maxdepth 1 -mindepth 1 -type d \
            -name "$PATTERN" \
            | xargs -I{} basename {} \
            | sort
    )

    n_exps=${#exps[@]}
    if [[ $n_exps -eq 0 ]]; then
        echo "No experiments found matching pattern '$PATTERN' in results/"
        exit 1
    fi

    idx="${SLURM_ARRAY_TASK_ID:-0}"
    if [[ $idx -ge $n_exps ]]; then
        echo "Array task $idx >= number of experiments ($n_exps); nothing to do."
        exit 0
    fi

    exp="${exps[$idx]}"
    echo "Experiment ($((idx+1))/$n_exps): $exp"

    python analyze.py --plot grokking_curve --exp "$exp" --save
    python analyze.py --plot loss_curve     --exp "$exp" --save
fi

echo
echo "Finished:  $(date)"
