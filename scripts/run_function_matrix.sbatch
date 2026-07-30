#!/bin/bash
#SBATCH --job-name=ds_fn_matrix
#SBATCH --account=bgkc-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --gpus=1
#SBATCH --time=18:00:00
#SBATCH --output=/work/hdd/bghj/schen47/inverse-101/results/function_mode/_matrix_logs/_slurm_%j.log

set -uo pipefail
REPO=/work/hdd/bghj/schen47/inverse-101
cd "$REPO"

# Build a NODE-LOCAL, job-private harness venv so the interpreter can't be
# clobbered by anything touching the shared .venv_harness (which broke the
# first run mid-flight). sys.executable + per-task `python -m venv` resolve here.
BASEPY=/sw/user/python/miniforge3-pytorch-24.11.3/bin/python
VENV="${SLURM_TMPDIR:-/tmp}/.venv_job_${SLURM_JOB_ID}"
"$BASEPY" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install -q requests pyyaml
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"

# Fresh full re-run; resumable on today's results (skips the 51 already done
# today), old dated runs preserved.
export FORCE=1
export RUN_DATE="$(date +%Y%m%d)"
export MODEL="Vendor3/DeepSeek-V4-Pro"
export MAX_CONCURRENCY=6

echo "host=$(hostname) start=$(date '+%F %T') RUN_DATE=$RUN_DATE VENV=$VENV"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
"$PY" -c 'import sys,platform;print("python",platform.machine(),platform.python_version(),sys.executable)'

"$PY" scripts/run_function_matrix.py
rc=$?
echo "matrix exit=$rc end=$(date '+%F %T')"
rm -rf "$VENV"
exit $rc
