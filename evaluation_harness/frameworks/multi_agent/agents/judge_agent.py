"""Judge agent: diagnoses failure root causes and assigns tickets.

Adapted from agentic_pipeline_dev for the imaging-101 benchmark.
Follows a 4-step diagnostic protocol to identify whether failures
should be routed to the Planner, Architect, or Coder.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from .base import BaseAgent

log = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    """Diagnoses failures and routes tickets to the appropriate agent."""

    def _build_system_prompt(self) -> str:
        return """\
You are the Chief Auditor of a Computational Imaging Solver System.
Your Mission: Diagnose the ROOT CAUSE of failure with surgical precision.

### DIAGNOSTIC PROTOCOL (Follow STRICTLY in order):

#### STEP 1: CHECK SYNTAX & IMPORTS
- Errors: SyntaxError, IndentationError, ImportError, NameError, AttributeError
- VERDICT → "Coder"

#### STEP 2: CHECK INTERFACE CONTRACT (Architect's Responsibility)
- Errors: Shape mismatch in function signatures, missing required parameters
- VERDICT → "Architect"

#### STEP 3: CHECK IMPLEMENTATION FIDELITY (Coder's Responsibility)
- Compare code against Planner's mathematical plan
- Mismatch examples: wrong formula, wrong sign, wrong hyperparameters
- VERDICT → "Coder"

#### STEP 4: CHECK ALGORITHM CORRECTNESS (Planner's Responsibility)
- Code runs without errors AND matches plan exactly BUT metrics are poor
- Wrong algorithm choice, missing regularization, incorrect convergence
- VERDICT → "Planner"

### OUTPUT FORMAT (STRICT JSON):
{
  "status": "FAIL",
  "ticket_assigned_to": "Planner" | "Architect" | "Coder",
  "analysis": "Step-by-step reasoning following the 4-step protocol",
  "evidence": "Exact line from logs/code showing the failure",
  "fix_target": "Specific file or function to fix",
  "feedback": "Actionable instruction for the assigned agent"
}

### CRITICAL RULES:
1. NEVER assign to Planner if implementation deviates from plan (that's Coder's fault).
2. ALWAYS verify implementation fidelity BEFORE blaming Planner.
3. If the SAME error repeats multiple iterations, suggest a DIFFERENT approach.
4. **MANDATORY ROUTING RULE — READ CAREFULLY**:
   Runtime errors are the Coder's fault. This includes ALL of these:
   KeyError, TypeError, ValueError, IndexError, AttributeError, NameError,
   ZeroDivisionError, FileNotFoundError, RuntimeError, StopIteration.
   For ANY of these errors → ticket_assigned_to = "Coder".
   The ONLY reason to assign to "Architect" is if a function's SIGNATURE
   (the `def` line itself) is MISSING a parameter. A wrong variable name,
   wrong dictionary key, wrong array index inside a function body is ALWAYS
   the Coder's fault, NEVER the Architect's.
   **If in doubt, assign to Coder.** Architect rewrites ALL files from scratch
   which is extremely expensive. Coder patches only the broken file.
5. ALWAYS specify "fix_target" — the exact filename (e.g., "src/solvers.py") where
   the error originates in the traceback. Look at the LAST "File" in the traceback.
6. PREFER "Coder" over "Architect". Reassigning to Architect causes a FULL
   skeleton regeneration, wasting time. If the error is in the code body
   (not the function interface), assign to Coder so only the broken file is fixed.
7. When assigning to Coder, set "fix_target" to the SPECIFIC file that caused
   the error, not all files. This enables targeted patching.
8. NEVER assign to "Architect" for: KeyError, TypeError, ValueError, IndexError.
   These are ALWAYS code-body bugs that the Coder must fix.
"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        prompt = f"### TASK\n{context.get('task_desc', 'N/A')}\n\n"
        prompt += f"### EXECUTION LOGS (Most Recent)\n{context.get('logs', 'No logs')}\n\n"
        prompt += f"### METRICS\nCurrent: {context.get('metrics', 'N/A')}\n\n"
        prompt += f"### PLAN\n{context.get('plan', 'N/A')}\n\n"

        if context.get("current_code"):
            code = context["current_code"]
            if len(code) > 6000:
                # Smart truncation: identify the error-relevant file from logs
                # and show it in full; show only signatures for other files
                error_file = self._identify_error_file(
                    context.get("logs", ""), code
                )
                code = error_file
            prompt += f"### CODE (Current Implementation)\n```python\n{code}\n```\n\n"

        if context.get("failure_history"):
            prompt += f"### PAST FAILURES\n{context['failure_history']}\n\n"

        prompt += "### YOUR ANALYSIS\nDiagnose using the 4-step protocol. Output STRICT JSON."
        return prompt

    @staticmethod
    def _identify_error_file(logs: str, all_code: str) -> str:
        """Extract the error-relevant file in full; show signatures for others.

        Parses traceback to find which file caused the error, shows that
        file completely, and shows only import/def/class lines for the rest.
        """
        # Split concatenated code into per-file blocks
        file_blocks = {}
        current_fname = None
        current_lines: list = []
        for line in all_code.split("\n"):
            if line.startswith("# === ") and line.endswith(" ==="):
                if current_fname:
                    file_blocks[current_fname] = "\n".join(current_lines)
                current_fname = line[6:-4].strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_fname:
            file_blocks[current_fname] = "\n".join(current_lines)

        if not file_blocks:
            # Fallback: return last 6000 chars
            return all_code[-6000:]

        # Find error file from traceback
        error_fname = None
        for line in logs.split("\n"):
            match = re.search(r'File "([^"]+\.py)"', line)
            if match:
                fpath = match.group(1)
                for fname in file_blocks:
                    if fpath.endswith(fname):
                        error_fname = fname

        # Build output: error file in full, others as signatures
        parts = []
        for fname in sorted(file_blocks):
            if fname == error_fname:
                parts.append(f"# === {fname} === (FULL — error source)")
                parts.append(file_blocks[fname])
            else:
                parts.append(f"# === {fname} === (signatures only)")
                for line in file_blocks[fname].split("\n"):
                    stripped = line.strip()
                    if (stripped.startswith(("import ", "from ", "def ", "class ", "@"))
                            or stripped == ""):
                        parts.append(line)
            parts.append("")

        result = "\n".join(parts)
        # If still too long, truncate non-error files further
        if len(result) > 10000:
            parts = []
            for fname in sorted(file_blocks):
                if fname == error_fname:
                    parts.append(f"# === {fname} === (FULL — error source)")
                    parts.append(file_blocks[fname])
                else:
                    parts.append(f"# === {fname} === (omitted)")
                parts.append("")
            result = "\n".join(parts)
        return result

    def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate judgment and parse as structured dict.

        Returns a dict with at minimum: ticket_assigned_to, analysis, feedback.
        Falls back to routing to Coder if parsing fails.
        """
        max_retries = 2
        for attempt in range(max_retries):
            raw = super().generate(context)
            # Try to extract JSON
            json_match = re.search(r"\{[\s\S]*\}", raw)
            candidate = json_match.group(0) if json_match else raw.strip()
            try:
                result = json.loads(candidate)
                # Validate required fields
                if "ticket_assigned_to" not in result:
                    result["ticket_assigned_to"] = "Coder"
                if "analysis" not in result:
                    result["analysis"] = raw[:500]
                if "feedback" not in result:
                    result["feedback"] = result.get("analysis", "Fix the code.")
                result.setdefault("fix_target", None)
                result.setdefault("evidence", "")
                return result
            except Exception as e:
                log.warning("Judge JSON parse failed (attempt %d): %s", attempt + 1, e)

        # Fallback
        return {
            "status": "FAIL",
            "ticket_assigned_to": "Coder",
            "analysis": f"Judge output unparseable. Raw: {raw[:300]}",
            "evidence": "",
            "fix_target": None,
            "feedback": "Fix the code based on the execution logs.",
        }
