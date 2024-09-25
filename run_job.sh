#!/usr/bin/bash
#SBATCH --job-name gpt2_train
#SBATCH --account OPEN-29-45
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --time=7-00:00:00 # 7 days, 0 hours, 0 minutes, 0 seconds 
#SBATCH --output=gpt2_train_%j.out
#SBATCH --error=gpt2_train_%j.err

ml purge
ml load Python/3.11.5-GCCcore-13.2.0
. ./venv/bin/activate

cd learn_lightning
srun python3 gpt2_train.py

