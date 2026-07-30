#!/bin/bash
#SBATCH --job-name=ma_ds_full
#SBATCH --account=bgkc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-50%8
#SBATCH --output=/work/hdd/bghj/schen47/inverse-101/results/e2e_ma_deepseek_full/logs/slurm_%a.log
REPO=/work/hdd/bghj/schen47/inverse-101
PY=/projects/bgkc/schen47/conda_envs/fpm_inr/bin/python
export BASE_URL="https://api.gpugeek.com/v1"
export API_KEY="YOUR_API_KEY"
MODEL="Vendor3/DeepSeek-V4-Pro"
OUT=$REPO/results/e2e_ma_deepseek_full
TASKS=( "SSNP_ODT" "cars_spectroscopy" "confocal-nlos-fk" "conventional_ptychography" "ct_dual_energy" "ct_poisson_lowdose" "ct_sparse_view" "differentiable_deflectometry" "diffusion_mri_dti" "eht_black_hole_UQ" "eht_black_hole_dynamic" "eht_black_hole_feature_extraction_dynamic" "eht_black_hole_tomography" "eit_conductivity_reconstruction" "electron_ptychography" "era5_tensorvar" "exoplanet_imaging" "fpm_inr_reconstruction" "hessian_sim" "insar_phase_unwrapping" "lensless_imaging" "light_field_microscope" "lucky_imaging" "mcr_hyperspectral" "microscope_denoising" "mri_dynamic_dce" "mri_grappa" "mri_l1_wavelet" "mri_noncartesian_cs" "mri_pnp_admm" "mri_t2_mapping" "mri_tv" "mri_varnet" "photoacoustic_tomography" "plane_wave_ultrasound" "pnp_mri_reconstruction" "raman_cell_phenotyping" "reflection_ODT" "s2ism" "seismic_FWI_original" "seismic_lsrtm_original" "seismic_traveltime_tomography" "shack-hartmann" "shapelet_source_reconstruction" "single_molecule_light_field" "spectral_snapshot_compressive_imaging" "ultrasound_sos_tomography" "usct_FWI" "xray_laminography_tike" "xray_ptychography_tike" "xray_tooth_gridrec"  )
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
echo "host=$(hostname) task=$T fw=multi_agent acct=bgkc start=$(date '+%F %T')"
cd "$REPO"
timeout 7000 "$PY" -u -m evaluation_harness run \
  --task "$T" --mechanism end2end --level L1 --framework multi_agent \
  --model "$MODEL" --base-url "$BASE_URL" --api-key "$API_KEY" \
  --max-iterations 5 --timeout 600 --output "$OUT" \
  > "$OUT/logs/${T}.log" 2>&1
echo "task=$T exit=$? end=$(date '+%F %T')"
