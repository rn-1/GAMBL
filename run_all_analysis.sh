#!/bin/bash

export N=$(ls -d results/*/ | wc -l)

sbatch --array=0-$((N-1)) slurm_analyze.sh

