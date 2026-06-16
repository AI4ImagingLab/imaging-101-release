#!/bin/bash
# Run end-to-end evaluations for eht_black_hole_original
# Compare ReAct vs Multi-Agent frameworks using gemini-3-pro-preview

set -uo pipefail

API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
MODEL="${MODEL:-gemini-2.5-pro}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

if [ -z "$API_KEY" ]; then
    echo "ERROR: set API_KEY or OPENAI_API_KEY before running this script." >&2
    exit 1
fi

TASK="eht_black_hole_original"
MODE="end_to_end"

# Fair comparison parameters
# Multi-agent uses ~95 LLM calls per run (Planner+Critic+Architect+Coder+Judge),
# spread across its internal "iterations" (pipeline cycles).
# ReAct uses exactly 1 LLM call per iteration.
# To ensure a fair comparison, we give ReAct 100 iterations (about multi-agent's 95 calls)
# and match the timeout to multi-agent's wall time (~5500s).
REACT_MAX_ITER=100
MULTI_MAX_ITER=10     # multi-agent: 10 pipeline cycles, about 95-130 LLM calls
TIMEOUT=7200          # 2 hours shared timeout

echo "============================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting end-to-end evaluations"
echo "Task:      $TASK"
echo "Model:     $MODEL"
echo "ReAct:     $REACT_MAX_ITER iterations, ${TIMEOUT}s timeout"
echo "Multi-Agent: $MULTI_MAX_ITER pipeline cycles, ${TIMEOUT}s timeout"
echo "============================================="

# --- Framework 1: ReAct ---
echo ""
echo "============================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') Running: ReAct framework"
echo "============================================="

python -m evaluation_harness run \
    --task "$TASK" \
    --mode "$MODE" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --max-iterations "$REACT_MAX_ITER" \
    --timeout "$TIMEOUT" \
    --framework react \
    --output results \
    -v \
    2>&1 | tee "logs_e2e_gemini_react.log"

REACT_EXIT=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') ReAct finished (exit code: $REACT_EXIT)"

# --- Framework 2: Multi-Agent ---
echo ""
echo "============================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') Running: Multi-Agent framework"
echo "============================================="

python -m evaluation_harness run \
    --task "$TASK" \
    --mode "$MODE" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --max-iterations "$MULTI_MAX_ITER" \
    --timeout "$TIMEOUT" \
    --framework multi_agent \
    --output results \
    -v \
    2>&1 | tee "logs_e2e_gemini_multi_agent.log"

MULTI_EXIT=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') Multi-Agent finished (exit code: $MULTI_EXIT)"

# --- Summary ---
echo ""
echo "============================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') All end-to-end evaluations complete"
echo "============================================="

echo ""
echo "Result files:"
ls -lt results/*gemini* 2>/dev/null | head -10

echo ""
echo "Comparison with baseline (Claude-4.6-opus ReAct):"
echo "  Baseline: NRMSE=1.6258, NCC=0.0727"
echo ""
echo "New results:"
for f in results/*gemini*end_to_end*; do
    if [ -f "$f" ]; then
        echo "  $(basename "$f"):"
        python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
    qm = d.get('quality_metrics', {})
    print(f'    Framework:  {d.get(\"framework\", \"react\")}')
    print(f'    NRMSE:      {qm.get(\"nrmse\", \"N/A\")}')
    print(f'    NCC:        {qm.get(\"ncc\", \"N/A\")}')
    print(f'    PSNR:       {qm.get(\"psnr\", \"N/A\")}')
    print(f'    SSIM:       {qm.get(\"ssim\", \"N/A\")}')
    print(f'    Iterations: {d.get(\"iterations\", \"N/A\")}')
    print(f'    LLM Calls:  {d.get(\"llm_calls\", \"N/A\")}')
    print(f'    Time:       {d.get(\"wall_time_seconds\", \"N/A\")}s')
    print(f'    Tokens:     {d.get(\"total_tokens\", \"N/A\")}')
"
    fi
done
