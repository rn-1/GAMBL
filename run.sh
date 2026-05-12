#!/bin/bash
#SBATCH --job-name=analogy
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --output=analogy.out   # stdout
#SBATCH --error=analogy.err    # stderr



cd $HOME/cs567/GAMBL
lira
python train.py --dataset csv --csv_path data/questions-words.csv --model transformer_lm --train_fraction 0.1 --batch_size 128
