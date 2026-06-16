"""Configuration for DeepCode framework integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DeepCodeConfig:
    """Configuration for running DeepCode on imaging-101 tasks.

    Attributes
    ----------
    deepcode_path : Path or None
        Path to cloned DeepCode repo (for Docker/source mode).
        If None, assumes `deepcode` CLI is available via pip.
    mode : str
        "cli" (headless), "web" (launches UI), or "docker".
    llm_provider : str
        "google", "anthropic", or "openai" — maps to DeepCode's
        `mcp_agent.config.yaml` → `llm_provider`.
    model : str
        Model name to use (e.g. "gemini-2.5-pro", "claude-sonnet-4-20250514").
    api_key : str
        API key for the chosen provider.
    base_url : str or None
        Custom base URL (e.g. for OpenRouter).
    workspace_dir : Path or None
        Custom workspace directory for DeepCode to work in.
    timeout_seconds : int
        Max time to wait for DeepCode to finish.
    """

    deepcode_path: Path | None = None
    mode: str = "cli"  # "cli" | "web" | "docker"
    llm_provider: str = "google"
    model: str = "gemini-2.5-pro"
    api_key: str = ""
    base_url: str | None = None
    workspace_dir: Path | None = None
    timeout_seconds: int = 3600  # DeepCode can take a while

    # MCP server toggles
    enable_brave_search: bool = False
    enable_github_downloader: bool = False
    enable_code_reference_indexer: bool = True

    def to_secrets_yaml(self) -> str:
        """Generate mcp_agent.secrets.yaml content for this configuration."""
        provider_block = ""
        if self.llm_provider == "openai":
            provider_block = f"""\
openai:
  api_key: "{self.api_key}"
  base_url: "{self.base_url or 'https://api.openai.com/v1'}"
"""
        elif self.llm_provider == "anthropic":
            provider_block = f"""\
anthropic:
  api_key: "{self.api_key}"
"""
        elif self.llm_provider == "google":
            provider_block = f"""\
google:
  api_key: "{self.api_key}"
"""

        return provider_block

    def to_config_yaml_patch(self) -> dict:
        """Return key-value overrides for mcp_agent.config.yaml."""
        return {
            "llm_provider": self.llm_provider,
        }
