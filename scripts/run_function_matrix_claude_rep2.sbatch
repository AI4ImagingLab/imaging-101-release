#!/bin/bash
#SBATCH --job-name=cl_fn_rep2
#SBATCH --account=bgkc-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180g
#SBATCH --gpus=1
#SBATCH --time=18:00:00
#SBATCH --output=/work/hdd/bghj/schen47/inverse-101/results_claude_rep2/function_mode/_matrix_logs/_slurm_%j.log

set -uo pipefail
REPO=/work/hdd/bghj/schen47/inverse-101
cd "$REPO"
mkdir -p results_claude_rep2/function_mode/_matrix_logs

# Node-local, job-private harness venv (immune to shared-symlink breakage).
BASEPY=/sw/user/python/miniforge3-pytorch-24.11.3/bin/python
VENV="${SLURM_TMPDIR:-/tmp}/.venv_job_${SLURM_JOB_ID}"
"$BASEPY" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install -q requests pyyaml
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"

# SECOND Claude run (reproducibility check). Isolated output root so the first
# run under results/ is never touched. Same settings as run 1: iter=10.
# RESUME-SAFE: FORCE=0 -> skip any pair already present in results_claude_rep2
# (the 57 from the OOM-killed attempt), midnight-safe. Lower concurrency + more
# memory to avoid the OOM-kill that stopped the first attempt.
export OUTPUT=results_claude_rep2
export FORCE=0
export MODEL="Vendor2/Claude-4.6-opus"
export MAX_ITER=10
export MAX_CONCURRENCY=4

echo "host=$(hostname) start=$(date '+%F %T') OUTPUT=$OUTPUT MODEL=$MODEL MAX_ITER=$MAX_ITER VENV=$VENV"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
"$PY" -c 'import sys,platform;print("python",platform.machine(),platform.python_version(),sys.executable)'

"$PY" scripts/run_function_matrix.py
rc=$?
echo "matrix exit=$rc end=$(date '+%F %T')"
rm -rf "$VENV"
exit $rc
