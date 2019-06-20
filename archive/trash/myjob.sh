#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -p gpu 
#SBATCH -n 1 
#SBATCH -c 4 
#SBATCH -w c3-1
#SBATCH --mem-per-cpu=8G
#SBATCH -o logs/output_%j.out

module load anaconda3 tensorflow

python train1616.py
