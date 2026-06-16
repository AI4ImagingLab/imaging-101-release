"""Framework registry for imaging-101 evaluation harness.

Each sub-package implements a distinct agent framework:
  - react/        Single-agent ReAct (Thought→Action→Observation) loop
  - multi_agent/  Multi-agent pipeline (Plan→Architect→Code→Judge)
  - claude_code/  Third-party agent evaluation (Claude Code, Copilot, Cursor…)
  - deepcode/     DeepCode (HKUDS) autonomous multi-agent coding framework
"""

FRAMEWORKS = ("react", "multi_agent", "claude_code", "deepcode")
