#!/usr/bin/env bash
# =================================================================
# Prepare all Claude Code (tmp_L1) sandboxes for 17 tasks
# Creates: tmp_L1/<task_name>/ with .prompt.md ready to paste
# =================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LEVEL="${1:-L1}"
OUT_BASE="${REPO_ROOT}/tmp_${LEVEL}"

ALL_TASKS=(
    conventional_ptychography
)

echo "========================================================"
echo "  Preparing Claude Code sandboxes"
echo "  Level: $LEVEL"
echo "  Output: $OUT_BASE/"
echo "  Tasks: ${#ALL_TASKS[@]}"
echo "========================================================"

mkdir -p "$OUT_BASE"

COUNT=0
for task in "${ALL_TASKS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/${#ALL_TASKS[@]}] Preparing $task..."
    
    python -m evaluation_harness prepare \
        --task "$task" \
        --level "$LEVEL" \
        --workspace-dir "$OUT_BASE/$task" \
        2>/dev/null || echo "  ⚠️  Failed to prepare $task"
done

echo ""
echo "========================================================"
echo "  DONE: $COUNT sandboxes prepared in $OUT_BASE/"
echo ""
echo "  For each task:"
echo "    1. Open the folder in Claude Code"
echo "    2. Paste .prompt.md into the agent"  
echo "    3. Let it produce output/reconstruction.npy"
echo "    4. Collect results with:"
echo "       python -m evaluation_harness collect \\"
echo "           --task <task_name> \\"
echo "           --workspace-dir $OUT_BASE/<task_name> \\"
echo "           --level $LEVEL --agent-name claude_code"
echo "========================================================"
