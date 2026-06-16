"""Prompt templates for each evaluation mode."""

# ---------------------------------------------------------------------------
# Compact context summary prompt
# ---------------------------------------------------------------------------

COMPACT_SUMMARY_PROMPT = """\
You are a technical assistant. Your job is to produce a concise, structured \
summary of a coding agent's conversation history. This summary will REPLACE \
the original messages in the agent's context window, so it must preserve ALL \
critical information needed to continue working effectively.

Analyze the conversation and produce a summary with EXACTLY these sections:

## Current State
- What files exist and their purpose (list every file written)
- What is the current implementation approach

## Key Findings
- Data format discoveries (shapes, dtypes, file contents explored)
- Important parameter values or constants discovered

## Error History
- Each distinct error encountered, what caused it, and whether it was resolved
- Pattern of recurring issues (if any)

## What Works
- Which parts of the code are verified working
- Successful test results or command outputs

## What Remains
- Outstanding bugs or failures not yet fixed
- Next logical step to take

RULES:
- Be CONCISE but COMPLETE — every file path, error message, and key number matters.
- Do NOT include actual file contents (too long) — just describe what each file does.
- Do NOT include generic advice — only facts from the actual conversation.
- Total length MUST be under 3000 characters.
"""


# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a computational imaging expert implementing a reconstruction pipeline.
You work inside a Linux container at /workspace.
Available files: data/ (observations), README.md (problem description).

You can take these actions:

1) WRITE_FILE — create or overwrite a file
   Action: WRITE_FILE
   Path: <file path relative to /workspace>
   Content:
   <file content — include the ENTIRE file, not just a snippet>
   END_CONTENT

2) RUN — execute a shell command
   Action: RUN
   Command: <shell command>

3) READ_FILE — read a file
   Action: READ_FILE
   Path: <file path>

4) DONE — signal that you have finished
   Action: DONE

Respond in this exact format every turn:
Thought: <your reasoning>
Action: <one of WRITE_FILE | RUN | READ_FILE | DONE>
<action arguments as shown above>

Important rules:
- Write ONE action per response. Wait for the Observation before continuing.
- Always write complete files, never partial patches.
- After writing code, run it to verify correctness.
- If errors occur, read the error, fix the code, and re-run.
- Do NOT use invented actions like CHECK, TEST, or ANALYZE. To inspect data, write a script and RUN it.
"""

# System prompt variant for function mode (mentions evaluation/tests)
SYSTEM_PROMPT_FUNCTION = """\
You are a computational imaging expert implementing a reconstruction pipeline.
You work inside a Linux container at /workspace.
Available files: data/ (observations), README.md (problem description), \
evaluation/ (fixtures and tests).

You can take these actions:

1) WRITE_FILE — create or overwrite a file
   Action: WRITE_FILE
   Path: <file path relative to /workspace>
   Content:
   <file content — include the ENTIRE file, not just a snippet>
   END_CONTENT

2) RUN — execute a shell command
   Action: RUN
   Command: <shell command>

3) READ_FILE — read a file
   Action: READ_FILE
   Path: <file path>

4) DONE — signal that you have finished
   Action: DONE

Respond in this exact format every turn:
Thought: <your reasoning>
Action: <one of WRITE_FILE | RUN | READ_FILE | DONE>
<action arguments as shown above>

Important rules:
- Write ONE action per response. Wait for the Observation before continuing.
- Always write complete files, never partial patches.
- After writing a source file, run its unit tests to verify.
- If tests fail, read the error, fix the code, and re-run.
- Do NOT use invented actions like CHECK, TEST, or ANALYZE. To inspect data, write a script and RUN it.
"""


# ---------------------------------------------------------------------------
# Plan-generation mode
# ---------------------------------------------------------------------------

def plan_approach_prompt(readme: str, meta_data: str) -> str:
    return f"""\
Read the problem description and data specification below, then write a \
solution approach document to plan/approach.md.

The document should include:
1. Problem statement (what we recover, from what measurements)
2. Mathematical formulation (forward model equation)
3. Solution strategy (step-by-step algorithmic approach)
4. Expected results (which methods, expected quality)

== Problem Description ==
{readme}

== Data Specification (meta_data) ==
{meta_data}

Begin by reading any additional data files if needed, then write plan/approach.md.
"""


def plan_design_prompt(readme: str, approach: str) -> str:
    return f"""\
Based on the problem description and the solution approach below, write a \
code design document to plan/design.md.

The document should include:
1. File structure (which src/*.py files)
2. Function signatures with full type annotations and docstrings
3. Class definitions where appropriate
4. Data flow diagram (text-based)

== Problem Description ==
{readme}

== Solution Approach ==
{approach}

Write plan/design.md with complete function signatures for every function \
that needs to be implemented.
"""


# ---------------------------------------------------------------------------
# Function-level mode
# ---------------------------------------------------------------------------

def function_prompt(
    readme: str,
    approach: str,
    design: str,
    target_function: str,
    test_file_content: str,
) -> str:
    return f"""\
Implement the function specified below. Context is provided so you understand \
the overall pipeline, but you only need to implement the target function.

== Problem Description ==
{readme}

== Solution Approach ==
{approach}

== Code Design (full signatures) ==
{design}

== Target Function ==
{target_function}

== Unit Tests (your implementation will be tested against these) ==
{test_file_content}

Steps:
1. Read any relevant data/fixture files to understand inputs and outputs.
2. Write the source file containing the target function.
3. Run the unit tests to verify.
4. Fix any failures, then signal DONE.
"""


# ---------------------------------------------------------------------------
# End-to-end mode
# ---------------------------------------------------------------------------

def end_to_end_plan_prompt(readme: str, meta_data: str) -> str:
    """Phase 1 of end-to-end: generate approach + design."""
    return f"""\
You will implement a full computational imaging pipeline from scratch.

First, create the solution plan:
1. Write plan/approach.md — solution strategy with math formulation
2. Write plan/design.md — code architecture with all function signatures

== Problem Description ==
{readme}

== Data Specification ==
{meta_data}

IMPORTANT CONSTRAINTS:
- Start by running `ls data/` and reading data files to understand the data.
- Read `requirements.txt` to know which packages are available.
- ONLY use packages listed in requirements.txt (typically numpy, scipy, matplotlib).
  Do NOT plan to use jax, torch, tensorflow, or any unlisted package.
- Prefer classical optimization (scipy.optimize.minimize with L-BFGS-B) over autodiff.
- If data contains pre-computed intermediate results, use them directly.
- Use `Optional[X]` from typing, not `X | None` (Python 3.9 compatibility).

Start by reading any data files to understand the problem, then write both \
plan documents. Signal DONE when both files are written.
"""


def end_to_end_impl_prompt(approach: str, design: str) -> str:
    """Phase 2 of end-to-end: implement all source files.

    The agent is NOT given any test cases.  It must freely implement the full
    reconstruction pipeline and produce ``output/reconstruction.npy``.
    """
    return f"""\
Implement the full reconstruction pipeline following the plan below.

== Solution Approach ==
{approach}

== Code Design ==
{design}

CRITICAL INSTRUCTIONS — you have limited iterations, so write code immediately:
1. First, explore the data directory: ls data/ — to understand available files.
   Also read requirements.txt to know available packages.
2. Write src/__init__.py (empty module marker).
3. Implement the source modules as described in your code design.
   You are free to organize your code however you see fit.
   After writing each module, run a quick sanity check if appropriate
   (e.g. python -c "from src.preprocessing import ...; print('OK')").
4. Write main.py — the entry point that runs the full reconstruction pipeline.
5. Run: python main.py
   The pipeline MUST produce output/reconstruction.npy containing the
   reconstructed image as a 2-D numpy array.

IMPORTANT CONSTRAINTS:
- ONLY use packages from requirements.txt. Do NOT import jax, torch, tensorflow.
- Use `Optional[X]` from typing, NOT `X | None` (Python 3.9 compatibility).
- Use the EXACT data key names from the .npz files. Do NOT guess key names.
- Do NOT spend time exploring or analyzing. Start writing code IMMEDIATELY.
- Use WRITE_FILE to create complete source files.
- Each iteration is precious — always include a WRITE_FILE action, not just
  RUN commands for exploration.

Your reconstruction quality will be evaluated by comparing
output/reconstruction.npy against the ground truth using NRMSE and NCC.
Focus on producing the best possible reconstruction.

Signal DONE after main.py runs successfully and output/reconstruction.npy
is saved, or you cannot improve further.
"""


# ---------------------------------------------------------------------------
# End-to-end Level 2: task description + approach given
# ---------------------------------------------------------------------------

def end_to_end_L2_plan_prompt(readme: str, meta_data: str, approach: str) -> str:
    """L2 planning phase: approach is given, agent only writes design.md."""
    return f"""\
You will implement a full computational imaging pipeline.

The solution approach has already been determined (see below). Your task is
to design the code architecture: write plan/design.md with file structure,
class definitions, and function signatures.

Also save the provided approach to plan/approach.md so it is available
for the implementation phase.

== Problem Description ==
{readme}

== Data Specification ==
{meta_data}

== Solution Approach (GIVEN — follow this) ==
{approach}

IMPORTANT CONSTRAINTS:
- Start by running `ls data/` and reading data files to understand the data.
- Read `requirements.txt` to know which packages are available.
- ONLY use packages listed in requirements.txt (typically numpy, scipy, matplotlib).
  Do NOT plan to use jax, torch, tensorflow, or any unlisted package.
- Prefer classical optimization (scipy.optimize.minimize with L-BFGS-B) over autodiff.
- If data contains pre-computed intermediate results, use them directly.
- Use `Optional[X]` from typing, not `X | None` (Python 3.9 compatibility).
- Follow the approach exactly — do not change the algorithm or methodology.

Steps:
1. Write plan/approach.md (copy the given approach above).
2. Write plan/design.md (code architecture with full function signatures,
   following the given approach).

Signal DONE when both files are written.
"""


def end_to_end_L2_impl_prompt(approach: str, design: str) -> str:
    """L2 implementation phase — same as standard since approach+design exist."""
    return end_to_end_impl_prompt(approach, design)


# ---------------------------------------------------------------------------
# End-to-end Level 3: task description + approach + design given
# ---------------------------------------------------------------------------

def end_to_end_L3_impl_prompt(
    readme: str, meta_data: str, approach: str, design: str,
) -> str:
    """L3: approach and design are both given. Agent only implements code.

    This skips the planning phase entirely. The agent receives the full
    problem description, approach, and design, then directly implements
    the source files and runs main.py.
    """
    return f"""\
Implement the full reconstruction pipeline following the plan and design below.

Both the solution approach and code design have been provided for you.
Your ONLY job is to implement the code and produce output/reconstruction.npy.

== Problem Description ==
{readme}

== Data Specification ==
{meta_data}

== Solution Approach ==
{approach}

== Code Design (function signatures and file structure) ==
{design}

CRITICAL INSTRUCTIONS — you have limited iterations, so write code immediately:
1. First, explore the data directory: ls data/ — to understand available files.
   Also read requirements.txt to know available packages.
2. Write src/__init__.py (empty module marker).
3. Implement ALL source modules exactly as specified in the code design.
   Follow the function signatures precisely.
   After writing each module, run a quick sanity check if appropriate
   (e.g. python -c "from src.preprocessing import ...; print('OK')").
4. Write main.py — the entry point that runs the full reconstruction pipeline.
5. Run: python main.py
   The pipeline MUST produce output/reconstruction.npy containing the
   reconstructed image as a 2-D numpy array.

IMPORTANT CONSTRAINTS:
- ONLY use packages from requirements.txt. Do NOT import jax, torch, tensorflow.
- Use `Optional[X]` from typing, NOT `X | None` (Python 3.9 compatibility).
- Use the EXACT data key names from the .npz files. Do NOT guess key names.
- Follow the code design's function signatures and file structure EXACTLY.
- Do NOT spend time exploring or analyzing. Start writing code IMMEDIATELY.
- Use WRITE_FILE to create complete source files.
- Each iteration is precious — always include a WRITE_FILE action, not just
  RUN commands for exploration.

Your reconstruction quality will be evaluated by comparing
output/reconstruction.npy against the ground truth using NRMSE and NCC.
Focus on producing the best possible reconstruction.

Signal DONE after main.py runs successfully and output/reconstruction.npy
is saved, or you cannot improve further.
"""
