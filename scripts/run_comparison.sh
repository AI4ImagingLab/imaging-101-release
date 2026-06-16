#!/bin/bash
# Run framework comparison: ReAct vs Multi-Agent Pipeline
#
# Usage:
#   bash run_comparison.sh [task] [model] [base_url] [api_key]
#
# Examples:
#   bash run_comparison.sh eht_black_hole_original cds/Claude-4.6-opus
#   bash run_comparison.sh eht_black_hole_original cds/Claude-4.6-opus https://api.example.com/v1 $MY_API_KEY

set -euo pipefail
cd "$(dirname "$0")"

TASK="${1:-eht_black_hole_original}"
MODEL="${2:-cds/Claude-4.6-opus}"
BASE_URL="${3:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
API_KEY="${4:-${OPENAI_API_KEY:-}}"
MAX_ITER="${MAX_ITERATIONS:-10}"
TIMEOUT="${TIMEOUT:-600}"

if [ -z "$API_KEY" ]; then
    echo "Error: API key not provided. Set OPENAI_API_KEY or pass as argument."
    exit 1
fi

echo "============================================"
echo "  Framework Comparison"
echo "  Task:       $TASK"
echo "  Model:      $MODEL"
echo "  Base URL:   $BASE_URL"
echo "  Max Iters:  $MAX_ITER"
echo "  Timeout:    ${TIMEOUT}s"
echo "============================================"

# Run comparison
python compare_frameworks.py \
    --task "$TASK" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --max-iterations "$MAX_ITER" \
    --timeout "$TIMEOUT" \
    --output results \
    -v

echo ""
echo "Done. Results in results/"
