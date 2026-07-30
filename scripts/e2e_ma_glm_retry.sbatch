#!/bin/bash
#SBATCH --job-name=glm_retry
#SBATCH --account=bgkc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-27%3
#SBATCH --output=/work/hdd/bghj/schen47/inverse-101/results/e2e_ma_glm_full/logs/retry_%a.log
REPO=/work/hdd/bghj/schen47/inverse-101
PY=/projects/bgkc/schen47/conda_envs/fpm_inr/bin/python
export BASE_URL="https://api.gpugeek.com/v1"
export API_KEY="YOUR_API_KEY"
MODEL="Vendor2/GLM-5.1"
OUT=$REPO/results/e2e_ma_glm_full
TASKS=( "SSNP_ODT" "confocal-nlos-fk" "ct_fan_beam" "eht_black_hole_feature_extraction_dynamic" "eht_black_hole_tomography" "eit_conductivity_reconstruction" "fourier_ptychography" "fpm_inr_reconstruction" "hessian_sim" "lensless_imaging" "light_field_microscope" "mcr_hyperspectral" "microscope_denoising" "mri_dynamic_dce" "mri_grappa" "mri_l1_wavelet" "mri_noncartesian_cs" "mri_pnp_admm" "mri_tv" "mri_varnet" "pet_mlem" "photoacoustic_tomography" "reflection_ODT" "seismic_FWI_original" "seismic_lsrtm_original" "usct_FWI" "xray_laminography_tike" "xray_tooth_gridrec"  )
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
echo "host=$(hostname) task=$T model=GLM max_tokens=8192 start=$(date '+%F %T')"
cd "$REPO"
timeout 7000 "$PY" -u -m evaluation_harness run \
  --task "$T" --mechanism end2end --level L1 --framework multi_agent \
  --model "$MODEL" --base-url "$BASE_URL" --api-key "$API_KEY" \
  --max-iterations 5 --max-tokens 8192 --timeout 600 --output "$OUT" \
  > "$OUT/logs/${T}.log" 2>&1
echo "task=$T exit=$? end=$(date '+%F %T')"
