#!/bin/bash
#SBATCH --job-name=dd_main
#SBATCH --account=bgkc-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bghj/schen47/inverse-101/tasks/differentiable_deflectometry/_main_run.log

set -uo pipefail
TASK=/work/hdd/bghj/schen47/inverse-101/tasks/differentiable_deflectometry
PY=/sw/user/python/miniforge3-pytorch-24.11.3/bin/python
cd "$TASK"
echo "host=$(hostname) start=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
PYTHONPATH="$TASK" PYTHONUNBUFFERED=1 "$PY" -u main.py
echo "exit=$? end=$(date '+%F %T')"
