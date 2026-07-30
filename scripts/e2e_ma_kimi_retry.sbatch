#!/bin/bash
#SBATCH --job-name=kimi_retry
#SBATCH --account=bgkc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-40%4
#SBATCH --output=/work/hdd/bghj/schen47/inverse-101/results/e2e_ma_kimi_full/logs/retry_%a.log
REPO=/work/hdd/bghj/schen47/inverse-101
PY=/projects/bgkc/schen47/conda_envs/fpm_inr/bin/python
export BASE_URL="https://api.gpugeek.com/v1"
export API_KEY="YOUR_API_KEY"
MODEL="Vendor2/Kimi-k2.6"
OUT=$REPO/results/e2e_ma_kimi_full
TASKS=( "SSNP_ODT" "cars_spectroscopy" "confocal-nlos-fk" "conventional_ptychography" "ct_sparse_view" "differentiable_deflectometry" "eht_black_hole_UQ" "eht_black_hole_dynamic" "eht_black_hole_feature_extraction_dynamic" "eht_black_hole_original" "eht_black_hole_tomography" "electron_ptychography" "era5_tensorvar" "fourier_ptychography" "fpm_inr_reconstruction" "lensless_imaging" "mri_grappa" "mri_l1_wavelet" "mri_noncartesian_cs" "mri_pnp_admm" "mri_tv" "mri_varnet" "photoacoustic_tomography" "plane_wave_ultrasound" "pnp_mri_reconstruction" "raman_cell_phenotyping" "reflection_ODT" "s2ism" "seismic_FWI_original" "seismic_lsrtm_original" "seismic_traveltime_tomography" "shack-hartmann" "shapelet_source_reconstruction" "single_molecule_light_field" "spectral_snapshot_compressive_imaging" "ultrasound_sos_tomography" "usct_FWI" "weather_radar_data_assimilation" "xray_laminography_tike" "xray_ptychography_tike" "xray_tooth_gridrec"  )
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
echo "host=$(hostname) task=$T model=Kimi max_tokens=8192 start=$(date '+%F %T')"
cd "$REPO"
timeout 7000 "$PY" -u -m evaluation_harness run \
  --task "$T" --mechanism end2end --level L1 --framework multi_agent \
  --model "$MODEL" --base-url "$BASE_URL" --api-key "$API_KEY" \
  --max-iterations 5 --max-tokens 8192 --timeout 600 --output "$OUT" \
  > "$OUT/logs/${T}.log" 2>&1
echo "task=$T exit=$? end=$(date '+%F %T')"
