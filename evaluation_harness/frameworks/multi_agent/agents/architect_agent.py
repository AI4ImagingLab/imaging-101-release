"""Architect agent: designs the code skeleton / class structure.

Adapted from agentic_pipeline_dev for the imaging-101 benchmark.
Instead of a single InverseSolver class, the Architect designs the
multi-file src/ layout matching imaging-101's task structure.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent


class ArchitectAgent(BaseAgent):
    """Designs the code architecture (file structure + function signatures)."""

    def _build_system_prompt(self) -> str:
        return """\
You are a Senior Software Architect for Computational Imaging pipelines.
Your Goal: Design the Python code architecture (skeleton) for the Planner's algorithm.
You DO NOT write the implementation logic — only the structure and interfaces.

### Output Rules:
1. Design a modular file structure under src/ with:
   - src/__init__.py (empty)
   - src/preprocessing.py — data loading and preprocessing
   - src/physics_model.py — forward model implementation
   - src/solvers.py — inverse solver(s)
   - src/visualization.py — plotting utilities (optional)
   - main.py — entry point that orchestrates the full pipeline
2. For each file, provide the complete skeleton with:
   - All imports
   - Class/function signatures with full type hints
   - Docstrings explaining purpose and parameters
   - `pass` or `# TODO: Implement` placeholders for bodies
3. main.py MUST:
   - Load data from data/ directory
   - Run the reconstruction pipeline
   - Save output to output/reconstruction.npy as a 2-D numpy array
4. IMPORT CONSISTENCY (CRITICAL):
   - Every `from src.X import Y` must reference a file `src/X.py` that YOU
     provide in your output AND class/function Y must appear in that file's skeleton.
   - Do NOT create imports to modules outside of your file list.
   - Keep the design FLAT: prefer fewer files with more functions over many
     files with thin interfaces. If in doubt, put helper classes/functions
     in the SAME file that uses them rather than creating a separate module.
4. DEPENDENCY CONSTRAINT: ONLY use packages listed in requirements.txt
   (typically numpy, scipy, matplotlib). Do NOT import jax, torch,
   tensorflow, or any unlisted package.
5. Do NOT nest class definitions inside other classes.
6. Use `Optional[X]` from typing instead of `X | None` for Python 3.9 compat.

### Output Format:
Provide each file as a clearly labeled code block:

```python
# FILE: src/preprocessing.py
import numpy as np
...
```

```python
# FILE: src/physics_model.py
...
```

```python
# FILE: main.py
...
```
"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        prompt = f"### TASK\n{context['task_desc']}\n\n"
        prompt += f"### PLAN TO IMPLEMENT\n{context['plan']}\n"

        if context.get("data_inventory"):
            prompt += f"\n### AVAILABLE DATA FILES\n{context['data_inventory']}\n"

        if context.get("requirements"):
            prompt += (
                f"\n### AVAILABLE PACKAGES (STRICT — no others)\n"
                f"{context['requirements']}\n"
                "ONLY import these packages. Use `from typing import Optional` "
                "instead of `X | None` for Python 3.9 compatibility.\n"
            )

        if context.get("previous_skeleton"):
            prompt += (
                f"\n### PREVIOUS ARCHITECTURE (Reference)\n"
                f"{context['previous_skeleton']}\n"
                "\nPRESERVE correct interfaces. ONLY modify parts flagged by feedback.\n"
            )

        if context.get("feedback"):
            fb = context["feedback"]
            if isinstance(fb, dict):
                fb_str = fb.get("analysis") or fb.get("feedback") or str(fb)
            else:
                fb_str = str(fb)
            prompt += f"\n### FEEDBACK\n{fb_str}\n"

        prompt += (
            "\nDesign the complete code architecture now. "
            "Output skeleton files with function signatures, docstrings, and pass placeholders."
        )
        return prompt
