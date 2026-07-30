#!/bin/bash
#SBATCH --job-name=ds_fn_smoke
#SBATCH --account=bgkc-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/work/hdd/bghj/schen47/inverse-101/results/function_mode/_smoke.log

set -uo pipefail
REPO=/work/hdd/bghj/schen47/inverse-101
PY=$REPO/.venv_harness/bin/python
export PATH="$REPO/.venv_harness/bin:$PATH"
# keep BLAS single-threaded (matches run_function_matrix.py)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$REPO"

echo "host=$(hostname) start=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
echo "python: $($PY -c 'import platform;print(platform.machine(),platform.python_version())')"

echo "--- [1/3] compute-node internet: DeepSeek API endpoint ---"
curl -s -o /dev/null -w "gpugeek /models HTTP %{http_code}\n" --max-time 25 \
  -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.gpugeek.com/v1/models

echo "--- [2/3] pip reachability (PyPI) ---"
$PY -m pip download --no-deps --dest /tmp/pipsmoke_$$ numpy 2>&1 | tail -2
rm -rf /tmp/pipsmoke_$$

echo "--- [3/3] one real function-mode eval (ct_fan_beam / preprocessing) ---"
timeout 1500 $PY -m evaluation_harness run \
  --task ct_fan_beam --mechanism function --target-function preprocessing \
  --model "Vendor3/DeepSeek-V4-Pro" \
  --base-url "https://api.gpugeek.com/v1" \
  --api-key "YOUR_API_KEY" \
  --max-iterations 8 --timeout 600 --output results -v
echo "eval exit=$?"
echo "end=$(date '+%F %T')"
