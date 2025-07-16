import argparse
import submitit
import os
import sys

def my_job(arg):
    # local imports
    sys.path.insert(0, '/share/home/conradb/git/hephaestus-minicubes-ssl/')
    import mae_training.mae_finetune as mae_finetune

    # Activate conda environment inside the function (only works in bash shells)
    os.system('module load Miniconda3 && eval "$(conda shell.bash hook)" && conda activate torch2')
    if arg == 'linprobe':
        mae_finetune.finetune('../configs/linprobe_config.toml', True)
    elif arg == 'e2e-finetune':
        mae_finetune.finetune('../configs/e2e-finetune_config.toml', True)
    else:
        print('Finetuning stage is not set')
    os.system('conda deactivate')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, help="Enter either lin-probe or e2e-finetune")
    args = parser.parse_args()

    folder = "submitit_logs/"+args.stage
    job_name = "mae-"+args.stage

    # Setup SLURM executor
    executor = submitit.AutoExecutor(folder=folder)
    executor.update_parameters(
        slurm_job_name=job_name,
        timeout_min=2880,  # 2 days
        mem_gb=100,
        slurm_additional_parameters={
            "gres": "gpu:3",
            "output": "logs/test.out"
            },
        cpus_per_task=64
    )

    # Submit job with arguments
    job = executor.submit(my_job, args.stage)
    print("Submitted job ID:", job.job_id)

if __name__ == "__main__":
    main()






