#!/usr/bin/env bash
# Batch planning runs for the missing model/task pairs.
# - skips the LLM-as-judge scoring (--no-plan-score); only saves approach.md/design.md
# - idempotent: skips a pair whose approach.md AND design.md already exist
# - bounded concurrency
set -u
cd /work/hdd/bghj/schen47/inverse-101

BASE_URL="https://api.gpugeek.com/v1"
API_KEY="YOUR_API_KEY"
OUTPUT="results/planning"
LOGDIR="$OUTPUT/_batch_logs"
SUMMARY="$LOGDIR/_summary.txt"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
mkdir -p "$LOGDIR"
: > "$SUMMARY"

# "model|task" pairs
PAIRS=(
  "Vendor2/Claude-4.6-opus|ct_sparse_view"
  "Vendor2/Claude-4.6-opus|differentiable_deflectometry"
  "Vendor2/Claude-4.6-opus|eht_black_hole_feature_extraction_dynamic"
  "Vendor2/Claude-4.6-opus|hessian_sim"
  "Vendor2/Claude-4.6-opus|pnp_mri_reconstruction"
  "Vendor2/Claude-4.6-opus|single_molecule_light_field"
  "Vendor2/Claude-4.6-opus|xray_ptychography_tike"
  "Vendor3/DeepSeek-V4-Pro|cars_spectroscopy"
  "Vendor3/DeepSeek-V4-Pro|fpm_inr_reconstruction"
  "Vendor3/DeepSeek-V4-Pro|mri_noncartesian_cs"
  "Vendor3/DeepSeek-V4-Pro|seismic_lsrtm_original"
  "Vendor3/DeepSeek-V4-Pro|ultrasound_sos_tomography"
  "Vendor2/GPT-5.5|mri_l1_wavelet"
  "Vendor2/GPT-5.5|shack-hartmann"
  "Vendor2/GPT-5.5|usct_FWI"
  "Vendor2/Kimi-k2.6|eht_black_hole_original"
  "Vendor2/Kimi-k2.6|mri_noncartesian_cs"
  "Vendor3/qwen3-max|eht_black_hole_tomography"
  "Vendor3/qwen3-max|mri_varnet"
  "Vendor2/Gemini-3.5-flash|confocal-nlos-fk"
  "Vendor2/Gemini-3.5-flash|diffusion_mri_dti"
  "Vendor2/Gemini-3.5-flash|eht_black_hole_original"
  "Vendor2/Gemini-3.5-flash|era5_tensorvar"
  "Vendor2/Gemini-3.5-flash|fpm_inr_reconstruction"
  "Vendor2/Gemini-3.5-flash|microscope_denoising"
  "Vendor2/Gemini-3.5-flash|mri_tv"
  "Vendor2/Gemini-3.5-flash|seismic_FWI_original"
  "Vendor2/Gemini-3.5-flash|usct_FWI"
  "Vendor2/Gemini-3.5-flash|weather_radar_data_assimilation"
  "Vendor2/Gemini-3.5-flash|xray_laminography_tike"
  "Vendor2/Gemini-3.5-flash|xray_ptychography_tike"
)

run_one() {
  local model="$1" task="$2"
  local safe="${model//\//_}"
  local plan_dir="$OUTPUT/$safe/$task"
  local log="$LOGDIR/${task}__${safe}.log"

  # idempotent skip
  if [ -s "$plan_dir/approach.md" ] && [ -s "$plan_dir/design.md" ]; then
    echo "SKIP (exists) :: $model :: $task" >> "$SUMMARY"
    return 0
  fi

  python -m evaluation_harness run \
    --task "$task" --mechanism planning --model "$model" \
    --base-url "$BASE_URL" --api-key "$API_KEY" \
    --output "$OUTPUT" --no-plan-score > "$log" 2>&1
  local rc=$?

  local status
  if [ $rc -ne 0 ]; then
    status="FAIL(rc=$rc)"
  elif [ -s "$plan_dir/approach.md" ] && [ -s "$plan_dir/design.md" ]; then
    status="PASS"
  else
    local got=""
    [ -s "$plan_dir/approach.md" ] && got="${got}approach "
    [ -s "$plan_dir/design.md" ] && got="${got}design "
    status="INCOMPLETE(${got:-none})"
  fi
  echo "$status :: $model :: $task" >> "$SUMMARY"
}

echo "Batch start $(date) — ${#PAIRS[@]} pairs, max concurrency $MAX_CONCURRENCY"
for pair in "${PAIRS[@]}"; do
  model="${pair%%|*}"; task="${pair##*|}"
  # throttle
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENCY" ]; do wait -n; done
  echo "  launch -> $model :: $task"
  run_one "$model" "$task" &
done
wait

echo "Batch done $(date)"
echo "================ SUMMARY ================"
sort "$SUMMARY"
echo "----------------------------------------"
echo "PASS:       $(grep -c '^PASS' "$SUMMARY")"
echo "SKIP:       $(grep -c '^SKIP' "$SUMMARY")"
echo "INCOMPLETE: $(grep -c '^INCOMPLETE' "$SUMMARY")"
echo "FAIL:       $(grep -c '^FAIL' "$SUMMARY")"
