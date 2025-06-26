#!/bin/bash

#SBATCH --job-name mae_pretraining
#SBATCH --time=4-00:00:00
#SBATCH --mem 100G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -o logs/mae_pretrain_hephaestus_minicubes.out

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate torch2
python3 mae_pretrain.py
conda deactivate