"""imaging-101 evaluation harness."""

from .config import LLMConfig, RunConfig, TaskConfig

def __getattr__(name):
    # Public export and Judge collection require no LLM/network client packages.
    if name == "BenchmarkRunner":
        from .runner import BenchmarkRunner
        return BenchmarkRunner
    raise AttributeError(name)

__all__ = ["BenchmarkRunner", "RunConfig", "LLMConfig", "TaskConfig"]
