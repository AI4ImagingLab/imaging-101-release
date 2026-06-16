"""ReAct (single-agent) framework.

This framework wraps the ReAct loop:
  Thought → Action → Observation → repeat

Used by: `--framework react` (default)

Key files:
  - agent.py   — Agent class with ReAct loop
  - prompts.py — System/task prompt templates
"""

from .agent import Agent, AgentResult

__all__ = ["Agent", "AgentResult"]
