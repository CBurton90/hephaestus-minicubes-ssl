#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=64
#SBATCH --error=/share/home/conradb/git/hephaestus-minicubes-ssl/mae_training/submitit_logs/e2e-finetune/%j_0_log.err
#SBATCH --gres=gpu:3
#SBATCH --job-name=mae-e2e-finetune
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=logs/test.out
#SBATCH --signal=USR2@90
#SBATCH --time=2880
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /share/home/conradb/git/hephaestus-minicubes-ssl/mae_training/submitit_logs/e2e-finetune/%j_%t_log.out --error /share/home/conradb/git/hephaestus-minicubes-ssl/mae_training/submitit_logs/e2e-finetune/%j_%t_log.err /home/conradb/.conda/envs/torch2/bin/python3 -u -m submitit.core._submit /share/home/conradb/git/hephaestus-minicubes-ssl/mae_training/submitit_logs/e2e-finetune
