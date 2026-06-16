"""Claude Code / Third-party agent (copilot) framework.

Prepares a sandbox workspace + prompt for black-box agents like
Claude Code, GitHub Copilot, Cursor, Windsurf.

Workflow:
  1. `prepare` — builds sandbox with visible files + self_eval.py
  2. Operator pastes .prompt.md into agent
  3. Agent works autonomously → produces output/reconstruction.npy
  4. `collect` — scores results

Used by: `--framework copilot` or `prepare`/`collect` subcommands

Key files:
  - copilot_runner.py  — Sandbox preparation & result collection
  - copilot_prompt.py  — Prompt generation
  - copilot_scorer.py  — Results scoring
"""

from .copilot_runner import (
    prepare_copilot_evaluation,
    collect_copilot_results,
)
from .copilot_scorer import score_copilot_results
from .copilot_prompt import (
    generate_agent_prompt,
    generate_instruction_file,
)

__all__ = [
    "prepare_copilot_evaluation",
    "collect_copilot_results",
    "score_copilot_results",
    "generate_agent_prompt",
    "generate_instruction_file",
]
