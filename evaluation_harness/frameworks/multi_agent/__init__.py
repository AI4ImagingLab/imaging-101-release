"""Multi-agent pipeline framework.

Plan → Critic → Architect → Coder → Execute → Judge pipeline.

Used by: `--framework multi_agent`

Key files:
  - multi_agent.py — MultiAgentPipeline orchestrator
  - agents/        — Specialized agent implementations
    - planner_agent.py
    - architect_agent.py
    - coder_agent.py
    - judge_agent.py
"""

from .multi_agent import MultiAgentPipeline

__all__ = ["MultiAgentPipeline"]
