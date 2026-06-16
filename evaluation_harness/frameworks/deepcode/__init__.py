"""DeepCode (HKUDS) framework integration.

Integrates the HKUDS/DeepCode autonomous multi-agent coding framework
into the imaging-101 evaluation harness.

DeepCode can be run in two modes:
  1. CLI mode — `deepcode --cli` with a prompt file
  2. Web UI mode — `deepcode` launches a web interface

For evaluation, we use CLI mode with automated prompt injection.

Installation:
  pip install deepcode-hku

Repository: https://github.com/HKUDS/DeepCode
"""

from .runner import DeepCodeRunner
from .config import DeepCodeConfig

__all__ = ["DeepCodeRunner", "DeepCodeConfig"]
