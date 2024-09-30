#!/usr/bin/bash
#SBATCH --job-name gpt2_train
#SBATCH --account OPEN-29-45
#SBATCH --partition qgpu
#SBATCH --gpus 7
#SBATCH --nodes 1
#SBATCH --time=2-00:00:00 # 2 days, 0 hours, 0 minutes, 0 seconds 

# 200 seconds before training ends send SIGHUP signal for the lightning module to save the model and resubmit the job
#SBATCH --signal=SIGHUP@200

ml purge
ml load Python/3.11.5-GCCcore-13.2.0
. ./venv/bin/activate

cd learn_lightning
nvidia-smi
srun python3 gpt2_train.py

# ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9
