#!/bin/bash
# Run function-level evaluations for eht_black_hole_original
# Only function mode, 4 modules: preprocessing, physics_model, solvers, visualization

set -uo pipefail

API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
MODEL="${MODEL:-gpt-4o}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

if [ -z "$API_KEY" ]; then
    echo "ERROR: set API_KEY or OPENAI_API_KEY before running this script." >&2
    exit 1
fi

TASK="eht_black_hole_original"
MODE="function"
MAX_ITER=20
TIMEOUT=600

MODULES=("preprocessing" "physics_model" "solvers" "visualization")

echo "============================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting function-level evaluations"
echo "Task: $TASK"
echo "Modules: ${MODULES[*]}"
echo "============================================="

for module in "${MODULES[@]}"; do
    echo ""
    echo "============================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') Running: $TASK / function / $module"
    echo "============================================="
    
    python -m evaluation_harness run \
        --task "$TASK" \
        --mode "$MODE" \
        --target-function "$module" \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --api-key "$API_KEY" \
        --max-iterations "$MAX_ITER" \
        --timeout "$TIMEOUT" \
        --output results \
        -v \
        2>&1 | tee "logs_function_eval_${module}.log"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') Finished: $module (exit code: $?)"
done

echo ""
echo "============================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') All function-level evaluations complete"
echo "============================================="

# Show results
echo ""
echo "New result files:"
ls -lt results/ | head -10
