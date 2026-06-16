"""Coder agent: implements code modules based on plan + architecture.

Adapted from agentic_pipeline_dev for the imaging-101 benchmark.
Instead of a single solver.py, the Coder writes complete files
(src/preprocessing.py, src/physics_model.py, etc.) that the sandbox
runner executes.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from .base import BaseAgent


class CoderAgent(BaseAgent):
    """Implements complete Python files based on plan and architecture."""

    def _build_system_prompt(self) -> str:
        return """\
You are a Senior Python Developer in Scientific Computing.
Your Goal: Implement a SPECIFIC Python file based on the plan and architecture provided.

### Critical Rules:
1. Output: Return ONLY the complete Python file content — NO explanations, NO markdown fences.
2. MANDATORY: Write the FULL implementation logic. NO `pass`, NO `TODO`, NO placeholders.
3. Data Flow:
   - Load data from data/ directory (numpy .npy or .npz files)
   - Save final reconstruction to output/reconstruction.npy as a 2-D numpy array
4. Type Safety: Explicitly handle dtype conversions (e.g., .astype(np.float64))
5. SELF-CONTAINED: Only import packages from requirements.txt and standard library.
   Do NOT import jax, torch, tensorflow, or any package not in requirements.txt.
   Do NOT import from local project files unless they are in src/.
6. API COMPATIBILITY: Use only well-documented, standard function parameters.
   Wrap risky API calls in try/except with a fallback.
7. SAFE DATA LOADING: Use np.load(..., allow_pickle=True) and check ndim
   before calling .item().
8. Sign and hyperparameter correctness: Follow the plan EXACTLY.
   If plan says "x_new = x_old - tau * grad", use subtraction, not addition.
9. PYTHON 3.9 COMPATIBILITY: Use `Optional[X]` from typing, NOT `X | None`.
   Use `Union[X, Y]` from typing, NOT `X | Y`. Use `list[X]` or `List[X]`.
10. DATA KEY NAMES: Use the EXACT key names shown in the data inventory.
    Do NOT guess key names — always check the data inventory for exact names.
11. IMPORT CONSISTENCY: Only import names (classes, functions) from src/ modules
    that are listed in the "ALL MODULE INTERFACES" section below. Do NOT invent
    imports from modules or symbols that do not exist in the architecture.
12. SCIPY OPTIMIZER CONSTRAINTS: scipy.optimize.minimize with method='L-BFGS-B'
    only supports box bounds via `bounds=`. Do NOT pass `constraints=` to L-BFGS-B
    (it will be silently ignored). Enforce equality constraints via projection
    or penalty terms in the objective instead.

### Numerical Computing Safety (MANDATORY for all scientific code):
13. ARRAY INDEXING SAFETY: Before indexing with computed indices, always verify
    bounds: `idx = np.clip(idx, 0, arr.shape[axis] - 1)`. Never assume array
    lengths match — use `min(len(a), len(b))` when iterating over paired arrays.
14. NUMERICAL STABILITY: Add epsilon to all denominators: `x / (y + 1e-10)`.
    Use `np.log(np.maximum(x, 1e-30))` instead of bare `np.log(x)`.
    Use `np.exp(np.clip(x, -500, 500))` to prevent overflow.
    Cast to float64 before accumulating sums: `total = np.float64(0.0)`.
15. GRADIENT VERIFICATION: If you implement an analytical gradient function,
    ADD A RUNTIME CHECK wrapped in try/except that compares your gradient
    to a finite-difference approximation. Print the result but DO NOT crash
    if it fails:
    ```
    try:
        from scipy.optimize import approx_fprime
        grad_num = approx_fprime(x0, objective_only, 1e-7)
        grad_ana = gradient_func(x0)
        max_err = np.max(np.abs(grad_ana - grad_num))
        rel_err = max_err / (np.max(np.abs(grad_num)) + 1e-10)
        print(f"Gradient check: max|ana-num|={max_err:.2e}, relative={rel_err:.2e}")
    except Exception as e:
        print(f"Gradient check skipped due to error: {e}")
    ```
    This check should print a diagnostic but NEVER crash the program.
16. SHAPE ASSERTIONS: At key pipeline stages (after loading data, after forward
    model, after solver output), add assertions:
    `assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"`
17. OUTPUT VALIDATION: Before saving reconstruction.npy, verify:
    `assert result.ndim == 2, f"Expected 2D, got {result.ndim}D"`
    `assert np.all(np.isfinite(result)), "Result contains NaN/Inf"`
    `assert np.std(result) > 1e-15, "Result is constant (solver did not converge)"`
18. OPTIMIZER CONVERGENCE: When using scipy.optimize.minimize, ALWAYS:
    - Set `options={'maxiter': N, 'disp': True}` with N >= 200
    - Print the final cost and gradient norm after optimization
    - Check `result.success` and print a warning if False
19. GRADIENT STRATEGY: Prefer calling scipy.optimize.minimize WITHOUT the
    jac= argument (let scipy compute gradients numerically via finite
    differences). This is slower but much more reliable than hand-coded
    analytical gradients. Only pass jac= if the gradient has been verified
    against finite differences and found to be correct.
    Example:
    ```
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'disp': True})
    ```
    Do NOT pass jac= unless you are certain the gradient is correct.
20. DATA FIDELITY TERM: When loading data, check if calibrated/direct
    measurements are available alongside derived/corrupted ones. If the
    data inventory shows a key like '*_cal' or '*_calibrated' alongside
    '*_corrupt' or '*_raw', prefer the calibrated version in the main
    chi-squared data fidelity term. Direct measurement data (e.g.,
    visibilities, projections) gives more information than derived
    quantities (e.g., closure phases, amplitudes-only) and leads to
    significantly better reconstruction quality.
"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        target = context.get("target_file", "unknown file")
        plan = context.get("plan", "No plan provided")
        skeleton = context.get("skeleton", "No skeleton provided")
        full_architecture = context.get("full_architecture", "")
        current_code = context.get("current_code", "")
        feedback = context.get("feedback", "")
        data_inventory = context.get("data_inventory", "")
        requirements = context.get("requirements", "")

        prompt = f"### TARGET FILE: {target}\n\n"
        prompt += f"### IMPLEMENTATION PLAN\n{plan}\n\n"
        prompt += f"### CODE ARCHITECTURE (Skeleton for this file)\n{skeleton}\n\n"

        if full_architecture:
            prompt += (
                f"### ALL MODULE INTERFACES (imports & signatures across ALL files)\n"
                f"{full_architecture}\n"
                "⚠️ ONLY import names that actually exist in the above interfaces. "
                "Do NOT invent imports from modules or classes not shown here.\n\n"
            )

        if requirements:
            prompt += (
                f"### AVAILABLE PACKAGES (STRICT — no others installed)\n{requirements}\n"
                "⚠️ ONLY use these packages. Do NOT import jax, torch, tensorflow.\n\n"
            )

        if data_inventory:
            prompt += f"### AVAILABLE DATA FILES\n{data_inventory}\n\n"

        if current_code:
            prompt += f"### CURRENT CODE (to fix/improve)\n```python\n{current_code}\n```\n\n"

        if feedback:
            fb_str = feedback
            if isinstance(feedback, dict):
                fb_str = feedback.get("analysis") or feedback.get("feedback") or str(feedback)
            prompt += f"### FEEDBACK TO ADDRESS\n{fb_str}\n\n"

        prompt += (
            "Output ONLY the complete Python file content for the target file.\n"
            "CRITICAL: Write FULL implementation. No placeholders. No pass statements.\n"
            "Ensure the code is syntactically correct Python."
        )
        return prompt

    @staticmethod
    def strip_markdown(code: str) -> str:
        """Remove markdown code fences from LLM output."""
        # Try to extract from ```python ... ``` blocks
        matches = re.findall(r"```(?:python)?\n(.*?)\n```", code, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
        # Fallback: strip simple fences
        code = re.sub(r"^```\w*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)
        return code.strip()

    @staticmethod
    def extract_python(text: str) -> str:
        """Extract Python code from LLM response, handling markdown wrapping."""
        code = text.strip()
        matches = re.findall(r"```python\s*\n(.*?)\n?\s*```", text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
        matches = re.findall(r"```\s*\n(.*?)\n?\s*```", text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
        # Heuristic: find first line starting with import/from/class/def
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ", "class ", "def ", "@", "#")):
                return "\n".join(lines[i:]).strip()
        return code
