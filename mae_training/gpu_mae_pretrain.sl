#!/bin/bash

#SBATCH --job-name mae_pretraining
#SBATCH --time=2-00:00:00
#SBATCH --mem 100G
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=64
#SBATCH -o logs/mae-vit-large_pretrain_hephaestus_minicubes.out

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate torch2
nvidia-smi -L
python3 mae_pretrain.py
conda deactivate