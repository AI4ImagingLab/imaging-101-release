#!/bin/bash
#SBATCH --job-name=ssnp_regen
#SBATCH --account=bghj-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bghj/schen47/inverse-101/results/ssnp_regen.log

PY=/projects/bgkc/schen47/conda_envs/fpm_inr/bin/python
TASK=/work/hdd/bghj/schen47/inverse-101/tasks/SSNP_ODT
echo "host=$(hostname) start=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
cd "$TASK"
PYTHONPATH="$TASK" PYTHONUNBUFFERED=1 "$PY" -u main.py
echo "exit=$? end=$(date '+%F %T')"
