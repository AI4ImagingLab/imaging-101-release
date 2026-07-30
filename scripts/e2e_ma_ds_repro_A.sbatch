#!/bin/bash
#SBATCH --job-name=ma_repA
#SBATCH --account=bgkc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/work/hdd/bghj/schen47/inverse-101/results/e2e_ma_ds_repro_A/logs/slurm_%a.log
REPO=/work/hdd/bghj/schen47/inverse-101
PY=/projects/bgkc/schen47/conda_envs/fpm_inr/bin/python
export BASE_URL="https://api.gpugeek.com/v1"
export API_KEY="YOUR_API_KEY"
MODEL="Vendor3/DeepSeek-V4-Pro"
OUT=$REPO/results/e2e_ma_ds_repro_A
TASKS=( "ct_fan_beam" "mri_sense" "pet_mlem" "eht_black_hole_original" "weather_radar_data_assimilation" "fourier_ptychography" )
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
echo "host=$(hostname) task=$T repro=A start=$(date '+%F %T')"
cd "$REPO"
timeout 7000 "$PY" -u -m evaluation_harness run \
  --task "$T" --mechanism end2end --level L1 --framework multi_agent \
  --model "$MODEL" --base-url "$BASE_URL" --api-key "$API_KEY" \
  --max-iterations 5 --timeout 600 --output "$OUT" \
  > "$OUT/logs/${T}.log" 2>&1
echo "task=$T exit=$? end=$(date '+%F %T')"
