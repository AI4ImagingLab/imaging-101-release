"""Planner and Critic agents for the multi-agent pipeline.

Adapted from agentic_pipeline_dev to work within the imaging-101 benchmark.
The Planner produces algorithmic plans; the Critic verifies them with
structured JSON output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from .base import BaseAgent

log = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """Produces a mathematical and algorithmic plan from the task description."""

    def _build_system_prompt(self) -> str:
        return """\
You are a Principal Scientist in Computational Imaging and Inverse Problems.
Your Goal: Formulate a rigorous mathematical and algorithmic plan to solve
the user's imaging reconstruction task.

### Guidelines:
1. **Mathematical Modeling**: Explicitly define the Forward Model y = A(x) + n.
   What is A? What is the noise n?
2. **Method Selection**: Choose the most robust solver.
   - For linear problems: Consider ADMM, Primal-Dual, FISTA, or Conjugate Gradient.
   - For non-linear problems: Consider classical optimization (scipy L-BFGS-B,
     scipy.optimize.minimize) with explicit gradient computation.
   - Prefer stability and standard implementations over experimental papers.
3. **Dimensionality Awareness**: Check input/output shapes. Ensure operations
   preserve or transform dimensions correctly.
4. **Self-Contained**: The solution must use the data files available in data/.
   No external data files are available beyond what the task provides.
5. **Simplicity First**: Prefer well-understood classical algorithms that are
   straightforward to implement. Avoid complex deep learning unless needed.
   Design a FLAT code structure: put all solver logic (objective, gradient,
   regularization) in ONE file (src/solvers.py) rather than splitting across
   many helper modules. Fewer files = fewer cross-import bugs.
6. **CRITICAL DEPENDENCY CONSTRAINT**: Check the "Available Python Packages"
   section in the data inventory. You MUST ONLY use those packages plus the
   Python standard library. In most cases this means numpy, scipy, and
   matplotlib ONLY. Do NOT plan solutions using jax, torch, tensorflow,
   autograd, or any deep-learning framework unless explicitly listed.
   Use scipy.optimize.minimize (L-BFGS-B) for optimization, NOT autodiff.
7. **Use Pre-Computed Data**: If the data files contain pre-computed
   intermediate results (e.g., processed observations, derived features), USE
   them directly. Do NOT recompute from raw measurements unless necessary.
8. **Optimizer Constraints**: When using scipy.optimize.minimize:
   - L-BFGS-B only supports simple box bounds (`bounds=`), NOT general
     constraints. Do NOT pass `constraints=` to L-BFGS-B.
   - For equality constraints (e.g., sum(x) = constant), enforce them via
     projection/normalization AFTER each optimizer step, or include them as
     a penalty term in the objective function.
   - Always verify the optimizer converges properly (check number of
     iterations, final gradient norm, function value decrease).
9. **Multi-Round Optimization Strategy**: Inverse problems rarely converge
   well in a single optimization pass. Plan a COARSE-TO-FINE strategy:
   - Round 1: Strong regularization (large λ), fewer iterations → rough shape
   - Round 2: Medium regularization, warm-start from Round 1 → refined features
   - Round 3: Weak regularization, warm-start from Round 2 → fine detail
   Each round should use 100–500 iterations with scipy L-BFGS-B.
   Total across all rounds: at least 300–1000 iterations.
   ALWAYS warm-start (initialize from previous round's output).
10. **Regularization is Mandatory**: Every inverse problem is ill-posed.
    ALWAYS include at least one regularizer in the objective function:
    - Total Variation (TV) for piecewise-smooth images
    - L2 (Tikhonov) for smooth images
    - Entropy for positive-valued distributions
    The regularization weight λ should be tunable and DECREASE across rounds.
11. **Convergence Monitoring**: Print the objective value every 50 iterations.
    After optimization, print: final cost, gradient norm, number of iterations,
    and whether the optimizer converged. This is essential for diagnosing issues.
12. **Gradient Strategy — Prefer Numerical Over Analytical**: Analytical
    gradients are error-prone and a common source of bugs. Plan accordingly:
    - PREFERRED: Use scipy.optimize.minimize with ONLY the objective function
      (no jac= argument). L-BFGS-B will compute gradients numerically.
      This is slower but much more reliable.
    - ALTERNATIVE: If speed is critical, implement analytical gradients BUT
      include a runtime verification against finite differences. If the check
      fails, fall back to numerical gradients (remove jac= argument).
    - The objective function should be a SINGLE function that takes a flat
      vector x and returns a scalar cost. Keep it simple — no callbacks,
      no closure tricks, no complex class hierarchies.
    - ALWAYS print the objective value at the start and end of optimization
      so convergence can be verified from the logs.
13. **Data Term Selection — Use Highest Fidelity Available**: When the data
    inventory shows multiple versions of the same measurement (e.g., raw vs.
    calibrated, noisy vs. denoised, direct vs. derived), ALWAYS prefer the
    highest-fidelity/most-direct version as the PRIMARY data term:
    - Direct measurements > derived/closure quantities (more information)
    - Calibrated > uncalibrated/corrupted data (fewer systematic errors)
    - Use derived quantities (e.g., closure phases) as SECONDARY or
      regularization terms, not as the sole data fidelity term.
    - In the chi-squared data fidelity term, use the BEST available data.
    This is the single biggest factor in reconstruction quality.

### Output Format (Markdown):
1. **[Problem Formulation]**: Math equation and variable definitions.
2. **[Proposed Strategy]**: Name of the algorithm/architecture.
3. **[Step-by-Step Plan]**:
   - Step 1: Data Preprocessing
   - Step 2: Forward Operator Implementation
   - Step 3: Solver/Network Architecture Details
   - Step 4: Loss Function & Optimizer
4. **[Hyperparameters]**: ALL numerical parameters with EXACT values.
5. **[File Structure]**: Which src/*.py files to create and what each contains.
6. **[Sign Convention]**: For each update rule, explicitly state the sign.
"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        prompt = f"### Task Description\n{context['task_desc']}\n"

        if context.get("data_spec"):
            prompt += f"\n### Data Specification\n{context['data_spec']}\n"

        if context.get("data_inventory"):
            prompt += f"\n### Available Data Files\n{context['data_inventory']}\n"

        if context.get("requirements"):
            prompt += (
                f"\n### Available Python Packages (STRICT — no others available)\n"
                f"{context['requirements']}\n"
                "⚠️ You MUST NOT plan solutions using jax, torch, tensorflow, or "
                "any package not listed above. Use scipy.optimize for optimization.\n"
            )

        if context.get("feedback"):
            fb = context["feedback"]
            if isinstance(fb, dict):
                fb_str = fb.get("analysis") or fb.get("feedback") or str(fb)
            else:
                fb_str = str(fb)
            prompt += (
                f"\n### ⚠️ CRITICAL FEEDBACK (Previous Plan Failed)\n"
                f"Review and REVISE your plan:\n\"{fb_str}\"\n"
                "Do not repeat the same mistake. Adjust the math or algorithm.\n"
            )

        if context.get("failure_history"):
            prompt += f"\n### Past Failures\n{context['failure_history']}\n"

        prompt += (
            "\nProduce the [Problem Formulation], [Proposed Strategy], "
            "[Step-by-Step Plan], [Hyperparameters], and [File Structure] now."
        )
        return prompt


class CriticAgent(BaseAgent):
    """Evaluates a plan with structured JSON output (PASS/REJECT)."""

    def _build_system_prompt(self) -> str:
        return """\
You are a Senior Technical Reviewer for Inverse Problems in Computational Imaging.
Your sole responsibility: Critically evaluate algorithmic plans BEFORE coding begins.

### STRICT OUTPUT REQUIREMENTS
1. Output ONLY a valid JSON object with EXACTLY these fields:
   {
     "decision": "PASS" | "REJECT",
     "reason": "Concise technical justification (max 100 chars)",
     "suggestion": "Actionable improvement suggestion (if REJECTED, else empty string)"
   }
2. NO MARKDOWN, NO PREFIXES, NO EXPLANATIONS — ONLY RAW JSON.

### Evaluation Checklist (Reject ONLY if a CRITICAL item fails)
CRITICAL (reject if missing):
✅ Forward model A(x) defined mathematically?
✅ At least one regularizer with explicit weight λ?
✅ Optimizer specified (e.g., L-BFGS-B) with iteration count ≥ 100?
✅ Output format: 2-D numpy array saved as reconstruction.npy?

DESIRABLE (note in reason but do NOT reject):
○ Multi-round / annealing strategy present?
○ Epsilon guards on divisions/logs mentioned?
○ All hyperparameter values explicitly specified?
○ Numerical stability precautions described?

IMPORTANT: Be LENIENT. Most plans are good enough to try. Only REJECT if the
plan is fundamentally flawed (missing forward model, no regularization, no
optimizer, or uses unavailable packages). Minor omissions can be handled by
the Coder. When in doubt, PASS.
"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        return (
            f"### TASK DESCRIPTION\n{context.get('task_desc', 'N/A')}\n\n"
            f"### PROPOSED PLAN BY PLANNER\n{context.get('plan', 'NO PLAN PROVIDED')}\n\n"
            "### YOUR MISSION\n"
            "Review against the STRICT checklist above. Output ONLY valid JSON."
        )

    def generate(self, context: Dict[str, Any]) -> str:
        """Generate with built-in JSON validation and retry."""
        max_retries = 3
        for attempt in range(max_retries):
            raw = super().generate(context)
            json_match = re.search(r"\{[\s\S]*\}", raw)
            candidate = json_match.group(0) if json_match else raw.strip()
            try:
                parsed = json.loads(candidate)
                if "decision" not in parsed:
                    raise ValueError("Missing 'decision'")
                if parsed["decision"] not in ("PASS", "REJECT"):
                    raise ValueError(f"Invalid decision: {parsed['decision']}")
                parsed.setdefault("reason", "")
                parsed.setdefault("suggestion", "")
                return json.dumps({
                    "decision": parsed["decision"],
                    "reason": str(parsed["reason"])[:150],
                    "suggestion": str(parsed.get("suggestion", ""))[:200],
                }, ensure_ascii=False)
            except Exception as e:
                log.warning("Critic JSON validation failed (attempt %d): %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    context = dict(context)
                    context["feedback"] = (
                        f"PREVIOUS ATTEMPT FAILED: {e}. "
                        "Output MUST be valid JSON with fields: decision, reason, suggestion."
                    )

        fallback = {
            "decision": "REJECT",
            "reason": f"LLM output parsing failed after {max_retries} attempts",
            "suggestion": "Planner must regenerate plan with explicit mathematical definitions",
        }
        return json.dumps(fallback, ensure_ascii=False)
