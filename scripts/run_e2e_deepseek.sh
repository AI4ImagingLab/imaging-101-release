#!/bin/bash
# End-to-end eval of Vendor3/DeepSeek-V4-Pro on 5 stochastically-selected tasks.
# Intended to run on gpua099 (A100). Each task runs in the ReAct framework, L1.
set -uo pipefail

cd /work/hdd/bghj/schen47/inverse-101

PY=/work/hdd/bghj/schen47/inverse-101/.venv_harness/bin/python
export BASE_URL="https://api.gpugeek.com/v1"
export API_KEY="YOUR_API_KEY"
MODEL="Vendor3/DeepSeek-V4-Pro"

TASKS=(cars_spectroscopy ct_dual_energy eht_black_hole_UQ eht_black_hole_feature_extraction_dynamic xray_ptychography_tike)

LOGDIR=/work/hdd/bghj/schen47/inverse-101/results/e2e_deepseek_logs
mkdir -p "$LOGDIR"

echo "host=$(hostname) start=$(date '+%F %T')" | tee "$LOGDIR/_run.meta"

for T in "${TASKS[@]}"; do
  echo "=================================================="
  echo "$(date '+%F %T') START $T"
  echo "=================================================="
  timeout 3000 "$PY" -m evaluation_harness run \
    --task "$T" \
    --mechanism end2end \
    --level L1 \
    --framework react \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --max-iterations 20 \
    --timeout 600 \
    --output results \
    -v \
    > "$LOGDIR/${T}.log" 2>&1
  echo "$(date '+%F %T') DONE $T exit=$?"
done

echo "$(date '+%F %T') ALL DONE" | tee -a "$LOGDIR/_run.meta"
