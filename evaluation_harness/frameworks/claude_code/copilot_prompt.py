"""Prompt generation for third-party agent evaluation.

Generates carefully crafted prompts for black-box coding agents (Copilot,
Claude Code, Cursor, etc.).  The prompts:

1. Describe the task clearly with all necessary context.
2. Specify exact deliverables (output/reconstruction.npy).
3. Explain the self-evaluation mechanism (self_eval.py).
4. Set constraints to prevent cheating.
5. Encourage iterative improvement using self_eval.py.

Three prompt variants for L1/L2/L3 levels:
- L1: Task description only — agent must plan and implement from scratch.
- L2: + approach.md — agent is given the algorithmic approach.
- L3: + approach.md + design.md — agent is given both approach and code design.
"""

from __future__ import annotations


def generate_agent_prompt(
    task_name: str,
    readme: str,
    meta_data: str,
    requirements: str,
    level: str = "L1",
    approach: str | None = None,
    design: str | None = None,
) -> str:
    """Generate the prompt to paste into a third-party coding agent.

    Parameters
    ----------
    task_name : str
        Name of the task (e.g. "eht_black_hole_original").
    readme : str
        Task README.md content — the problem description.
    meta_data : str
        Data specification (JSON metadata).
    requirements : str
        requirements.txt content.
    level : str
        "L1", "L2", or "L3" — determines how much guidance is given.
    approach : str or None
        Content of plan/approach.md (for L2/L3).
    design : str or None
        Content of plan/design.md (for L3).

    Returns
    -------
    str
        The full prompt text to paste into the agent.
    """
    if level == "L1":
        return _prompt_L1(task_name, readme, meta_data, requirements)
    elif level == "L2":
        return _prompt_L2(task_name, readme, meta_data, requirements, approach or "")
    elif level == "L3":
        return _prompt_L3(task_name, readme, meta_data, requirements, approach or "", design or "")
    else:
        raise ValueError(f"Unknown level: {level}")


def generate_instruction_file(task_name: str, level: str) -> str:
    """Generate the INSTRUCTIONS.md file placed in the sandbox root.

    This is a shorter, file-based version of the prompt that agents
    can read if they automatically scan the workspace.
    """
    level_desc = {
        "L1": "You are given only the task description and data. Plan your approach and implement from scratch.",
        "L2": "You are given the task description, data, and a solution approach (plan/approach.md). Design the code architecture and implement.",
        "L3": "You are given the task description, data, a solution approach (plan/approach.md), and code design (plan/design.md). Implement the code following the design.",
    }

    return f"""\
# {task_name} — Computational Imaging Reconstruction Task

## Objective
Implement a computational imaging reconstruction pipeline that produces
`output/reconstruction.npy` — a 2D numpy array representing the reconstructed image.

## Difficulty Level: {level}
{level_desc.get(level, "")}

## Available Files
- `README.md` — Problem description with physics, data format, and method hints
- `data/` — Observation data and metadata
- `requirements.txt` — Available Python packages
{('- `plan/approach.md` — Solution approach (algorithmic strategy)' + chr(10)) if level in ('L2', 'L3') else ''}\
{('- `plan/design.md` — Code design (file structure + function signatures)' + chr(10)) if level == 'L3' else ''}\
- `self_eval.py` — Self-evaluation script (DO NOT MODIFY)

## Deliverables
1. `src/` directory with your implementation modules
2. `main.py` entry point that runs the full pipeline
3. `output/reconstruction.npy` — the reconstructed 2D image

## Self-Evaluation
After producing your reconstruction, run:
```bash
python self_eval.py              # Print quality metrics
python self_eval.py --visualize  # Also generate comparison figures
```
Use the metrics to iteratively improve your reconstruction.

## Rules
- ONLY use packages listed in `requirements.txt`
- Do NOT modify `self_eval.py` or any read-only files
- Do NOT attempt to extract the ground truth from `self_eval.py`
- Your final output MUST be `output/reconstruction.npy`
"""


# ---------------------------------------------------------------------------
# Level-specific prompt templates
# ---------------------------------------------------------------------------

_COMMON_HEADER = """\
You are a computational imaging expert. Your task is to implement a full
reconstruction pipeline that processes observation data and produces a
reconstructed image.

"""

_COMMON_RULES = """\

## ⚠️ IMPORTANT RULES

1. **Output**: Your pipeline MUST produce `output/reconstruction.npy` — a 2D
   numpy array (N×N) representing the reconstructed image. This is the ONLY
   deliverable that will be evaluated.

2. **Packages**: ONLY use packages listed in `requirements.txt`. Available:
   {requirements}
   Do NOT use jax, torch, tensorflow, or any unlisted package.

3. **Self-evaluation**: After producing output/reconstruction.npy, run:
   ```
   python self_eval.py
   ```
   This prints quality metrics (NRMSE, NCC, PSNR, SSIM). Use these to
   iteratively improve your reconstruction. You can also run:
   ```
   python self_eval.py --visualize
   ```
   to generate comparison figures in output/.

4. **Anti-cheat**: Do NOT modify `self_eval.py`. Do NOT attempt to extract
   or reverse-engineer the ground truth data embedded in it. Your task is to
   reconstruct the image from the observation data, not to decode the answer.

5. **Iterative improvement**: You are encouraged to:
   - Run your pipeline → check self_eval.py metrics → improve → repeat
   - Target: NRMSE < 0.5, NCC > 0.8, SSIM > 0.5

6. **File organization**: Follow standard structure:
   - `src/__init__.py` — empty module marker
   - `src/preprocessing.py` — data loading and preprocessing
   - `src/physics_model.py` — forward model implementation
   - `src/solvers.py` — inverse solver / optimization
   - `src/visualization.py` — plotting utilities
   - `main.py` — entry point that orchestrates the pipeline
"""

_COMMON_WORKFLOW = """\

## Recommended Workflow

1. Read `README.md` to understand the physics and problem formulation.
2. Explore `data/` to understand the data format and contents.
3. Read `requirements.txt` to know available packages.
{extra_steps}\
4. Implement the pipeline step by step (preprocessing → forward model → solver).
5. Write `main.py` to run the full pipeline.
6. Run `python main.py` and verify `output/reconstruction.npy` is created.
7. Run `python self_eval.py` to check quality metrics.
8. Iterate to improve reconstruction quality.
9. Run `python self_eval.py --visualize` for visual comparison.
"""


def _prompt_L1(task_name: str, readme: str, meta_data: str, requirements: str) -> str:
    """L1: Task description only — agent plans from scratch."""
    _common_workflow_no_extra = _COMMON_WORKFLOW.format(extra_steps="")
    return f"""\
{_COMMON_HEADER}
# Task: {task_name} (Level 1 — Plan + Implement from Scratch)

You are given ONLY the problem description and observation data. You must:
1. Analyze the problem and plan your approach
2. Design the code architecture
3. Implement the full reconstruction pipeline
4. Produce `output/reconstruction.npy`

## Problem Description

{readme}

## Data Specification (meta_data)

```json
{meta_data}
```
{_COMMON_RULES.format(requirements=requirements)}
{_common_workflow_no_extra}

## Start Now

Begin by reading the README and exploring the data directory to understand
the problem. Then plan your approach, implement the pipeline, and run it.
Focus on producing a working reconstruction as quickly as possible, then
iterate to improve quality using self_eval.py.
"""


def _prompt_L2(task_name: str, readme: str, meta_data: str, requirements: str,
               approach: str) -> str:
    """L2: Task description + approach — agent designs and implements."""
    _common_workflow_l2 = _COMMON_WORKFLOW.format(
        extra_steps="   Read `plan/approach.md` and follow the approach exactly.\n"
    )
    return f"""\
{_COMMON_HEADER}
# Task: {task_name} (Level 2 — Design + Implement with Given Approach)

You are given the problem description and a solution approach. You must:
1. Read the given approach in `plan/approach.md`
2. Design the code architecture (write `plan/design.md`)
3. Implement the full reconstruction pipeline following the approach
4. Produce `output/reconstruction.npy`

## Problem Description

{readme}

## Data Specification (meta_data)

```json
{meta_data}
```

## Solution Approach (GIVEN — follow this)

The file `plan/approach.md` contains the algorithmic strategy. Read it carefully.
Here is its content for reference:

{approach}
{_COMMON_RULES.format(requirements=requirements)}
{_common_workflow_l2}

## Start Now

Read `plan/approach.md`, then design the code architecture and implement it.
Follow the given approach — do not change the algorithm or methodology.
"""


def _prompt_L3(task_name: str, readme: str, meta_data: str, requirements: str,
               approach: str, design: str) -> str:
    """L3: Task description + approach + design — agent only implements."""
    _common_workflow_l3 = _COMMON_WORKFLOW.format(
        extra_steps="   Read `plan/approach.md` and `plan/design.md`. Follow the design exactly.\n"
    )
    return f"""\
{_COMMON_HEADER}
# Task: {task_name} (Level 3 — Implement with Given Approach + Design)

You are given the problem description, solution approach, AND code design.
You ONLY need to implement the code following the given specifications:
1. Read `plan/approach.md` (algorithmic strategy)
2. Read `plan/design.md` (code architecture and function signatures)
3. Implement ALL source modules as specified
4. Write `main.py` entry point
5. Produce `output/reconstruction.npy`

## Problem Description

{readme}

## Data Specification (meta_data)

```json
{meta_data}
```

## Solution Approach (GIVEN)

File: `plan/approach.md`

{approach}

## Code Design (GIVEN — follow this exactly)

File: `plan/design.md`

{design}
{_COMMON_RULES.format(requirements=requirements)}
{_common_workflow_l3}

## Start Now

The approach and code design are fully specified. Your job is to implement
the code faithfully following plan/design.md. Start with src/__init__.py,
then implement each module as specified, and finally write main.py.
"""
