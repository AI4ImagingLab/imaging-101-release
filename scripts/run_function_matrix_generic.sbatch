#!/bin/bash
#SBATCH --account=bgkc-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180g
#SBATCH --gpus=1
#SBATCH --time=18:00:00
#SBATCH --output=/work/hdd/bghj/schen47/inverse-101/results/function_mode/_matrix_logs/_slurm_%x_%j.log

# Parametrized full function-matrix run. Submit with:
#   sbatch -J <tag> --export=ALL,MODEL=<m>,OUTROOT=<dir>,MAXIT=<n> run_function_matrix_generic.sbatch
set -uo pipefail
REPO=/work/hdd/bghj/schen47/inverse-101
cd "$REPO"
mkdir -p "$OUTROOT/function_mode/_matrix_logs"

# Node-local, job-private harness venv (immune to shared-symlink breakage).
BASEPY=/sw/user/python/miniforge3-pytorch-24.11.3/bin/python
VENV="${SLURM_TMPDIR:-/tmp}/.venv_job_${SLURM_JOB_ID}"
"$BASEPY" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install -q requests pyyaml
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"

export OUTPUT="$OUTROOT"
export FORCE=1
export RUN_DATE="$(date +%Y%m%d)"
export MAX_ITER="${MAXIT:-10}"
export MAX_CONCURRENCY=6
# MODEL comes from --export

echo "host=$(hostname) start=$(date '+%F %T') MODEL=$MODEL OUTPUT=$OUTPUT MAX_ITER=$MAX_ITER VENV=$VENV"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
"$PY" -c 'import sys,platform;print("python",platform.machine(),platform.python_version(),sys.executable)'

"$PY" scripts/run_function_matrix.py
rc=$?
echo "matrix exit=$rc end=$(date '+%F %T')"
rm -rf "$VENV"
exit $rc
