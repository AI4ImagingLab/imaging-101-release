"""Configuration dataclasses for the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """OpenAI-compatible LLM endpoint configuration."""

    model: str  # e.g. "gpt-4o", "deepseek-chat"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 32768


# Valid end-to-end difficulty levels
E2E_LEVELS = ("L1", "L2", "L3")


@dataclass
class TaskConfig:
    """Which task to evaluate and in what mode."""

    task_name: str  # e.g. "eht_black_hole_original"
    task_dir: Path = field(default_factory=Path)  # resolved at runtime
    mode: str = "end_to_end"  # "plan" | "function" | "end_to_end"
    target_function: str | None = None  # for function mode, e.g. "preprocessing.load_observation"
    level: str = "L1"  # end-to-end difficulty level:
    #   "L1" — task description only (README.md)
    #   "L2" — task description + approach (README.md + plan/approach.md)
    #   "L3" — task description + approach + design (README.md + plan/approach.md + plan/design.md)


@dataclass
class RunConfig:
    """Top-level run configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    max_iterations: int = 20
    docker_image: str = "imaging101-sandbox"
    timeout_seconds: int = 600
    output_dir: Path = field(default_factory=lambda: Path("results"))
    log_file: Path | None = None  # Path to save detailed interaction logs
    framework: str = "react"  # "react" | "multi_agent" | "copilot" | "deepcode"
    skip_plan_score: bool = False  # plan mode: skip LLM-as-judge scoring, only emit the plan docs
