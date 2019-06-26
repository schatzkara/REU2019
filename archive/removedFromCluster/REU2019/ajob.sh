#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -p gpu 
#SBATCH -n 1 
#SBATCH -c 4 
#SBATCH --mem-per-cpu=8G
#SBATCH -o logs/output_%j.out

module load anaconda3 tensorflow

python trainModel.py
