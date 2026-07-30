#!/bin/bash
#SBATCH --job-name=e2e_glm
#SBATCH --account=bghj-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --array=0-5%6
#SBATCH --output=/work/hdd/bghj/schen47/inverse-101/results/e2e_glm_sample/logs/slurm_%a.log

# End-to-end sample eval of Vendor2/GLM-5.1 on 6 representative tasks, max-iter=5.
REPO=/work/hdd/bghj/schen47/inverse-101
PY=/projects/bgkc/schen47/conda_envs/fpm_inr/bin/python   # stable driver (numpy+requests)
export BASE_URL="https://api.gpugeek.com/v1"
export API_KEY="YOUR_API_KEY"
MODEL="Vendor2/GLM-5.1"
OUT=$REPO/results/e2e_glm_sample

# mix of easy 2D + harder, incl. 2 DeepSeek comparison points (weather, eht_original)
TASKS=( "ct_fan_beam" "mri_sense" "pet_mlem" "eht_black_hole_original" \
        "weather_radar_data_assimilation" "fourier_ptychography" )

T=${TASKS[$SLURM_ARRAY_TASK_ID]}
echo "host=$(hostname) task=$T start=$(date '+%F %T')"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
cd "$REPO"
timeout 5000 "$PY" -u -m evaluation_harness run \
  --task "$T" --mechanism end2end --level L1 --framework react \
  --model "$MODEL" --base-url "$BASE_URL" --api-key "$API_KEY" \
  --max-iterations 5 --timeout 600 --output "$OUT" \
  > "$OUT/logs/${T}.log" 2>&1
echo "task=$T exit=$? end=$(date '+%F %T')"
