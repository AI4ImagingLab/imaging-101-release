#!/bin/bash
#SBATCH --job-name=cl_fn_matrix
#SBATCH --account=bgkc-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --gpus=1
#SBATCH --time=18:00:00
#SBATCH --output=/work/hdd/bghj/schen47/inverse-101/results/function_mode/_matrix_logs/_slurm_claude_%j.log

set -uo pipefail
REPO=/work/hdd/bghj/schen47/inverse-101
cd "$REPO"

# Node-local, job-private harness venv (immune to the shared-symlink breakage
# that killed the first DeepSeek run). sys.executable + per-task `python -m venv`
# both resolve to this interpreter.
BASEPY=/sw/user/python/miniforge3-pytorch-24.11.3/bin/python
VENV="${SLURM_TMPDIR:-/tmp}/.venv_job_${SLURM_JOB_ID}"
"$BASEPY" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install -q requests pyyaml
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"

# Fresh full re-run for Claude-4.6-opus; resumable on today's results, old
# dated runs (e.g. 20260615) preserved. max_iterations = 10 (per request).
export FORCE=1
export RUN_DATE="$(date +%Y%m%d)"
export MODEL="Vendor2/Claude-4.6-opus"
export MAX_ITER=10
export MAX_CONCURRENCY=6

echo "host=$(hostname) start=$(date '+%F %T') RUN_DATE=$RUN_DATE MODEL=$MODEL MAX_ITER=$MAX_ITER VENV=$VENV"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
"$PY" -c 'import sys,platform;print("python",platform.machine(),platform.python_version(),sys.executable)'

"$PY" scripts/run_function_matrix.py
rc=$?
echo "matrix exit=$rc end=$(date '+%F %T')"
rm -rf "$VENV"
exit $rc
