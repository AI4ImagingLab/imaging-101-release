"""Base agent class adapted for the imaging-101 evaluation harness.

Uses the harness's LLMClient (requests-based, OpenAI-compatible) instead of
the OpenAI SDK directly.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any, Dict, Optional

from ..llm_client import LLMClient

log = logging.getLogger(__name__)


def _is_generation_complete(finish_reason: str | None, content: str) -> bool:
    """Check if LLM generation is complete or needs continuation."""
    if finish_reason == "length":
        return False
    if finish_reason == "content_filter":
        raise RuntimeError("Generation blocked by content filter.")
    if finish_reason is None:
        return not (content and content.strip())
    if content and content.strip():
        return True
    return True


class BaseAgent(abc.ABC):
    """Base class for all multi-agent pipeline agents.

    Each agent has a system prompt and a user prompt builder.
    Calls go through LLMClient (the harness's lightweight requests wrapper).
    Supports automatic continuation if the response is truncated (finish_reason=length).
    """

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.3,
        max_tokens: int = 32768,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = self._build_system_prompt()

    @abc.abstractmethod
    def _build_system_prompt(self) -> str:
        """Return the system prompt defining this agent's role."""
        ...

    @abc.abstractmethod
    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        """Build the user prompt from the given context dict."""
        ...

    def call_llm(
        self,
        user_prompt: str,
        max_loops: int = 3,
    ) -> str:
        """Call the LLM with automatic continuation for truncated responses.

        Parameters
        ----------
        user_prompt : str
            The user message to send.
        max_loops : int
            Maximum continuation attempts if truncated.

        Returns
        -------
        str
            Full assembled response text.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        full_content = ""
        for loop_i in range(max_loops):
            text, usage = self.client.chat(messages)
            full_content += text

            # Check for truncation: if the response seems cut off mid-sentence
            # and is very long, we can do a continuation.  However, the harness's
            # LLMClient doesn't expose finish_reason directly.  We approximate:
            # if the response is very close to max_tokens * 4 chars (rough proxy)
            # AND doesn't end with a "complete" marker, try a continuation.
            char_limit = self.max_tokens * 3  # rough proxy
            if len(text) >= char_limit and not text.rstrip().endswith(("```", "}", ".")):
                log.info("[%s] Response may be truncated (%d chars). Continuation %d/%d",
                         self.__class__.__name__, len(text), loop_i + 1, max_loops)
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        "CONTINUATION REQUEST: Continue EXACTLY from where you left off.\n"
                        "DO NOT repeat any previous content. Output ONLY the continuation."
                    ),
                })
            else:
                break

        return full_content

    def generate(self, context: Dict[str, Any]) -> str:
        """Standard entry point: build prompt from context and call LLM."""
        user_prompt = self._build_user_prompt(context)
        return self.call_llm(user_prompt)
