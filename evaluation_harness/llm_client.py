"""Thin OpenAI-compatible chat completions client using requests."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from .config import LLMConfig

log = logging.getLogger(__name__)


class LLMClient:
    """Calls any OpenAI-compatible chat/completions endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._call_count = 0

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, str]],
        stop: list[str] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Send a chat completion request.

        Returns
        -------
        (response_text, usage_dict)
            usage_dict has keys ``prompt_tokens`` and ``completion_tokens``.
        """
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
        }
        # kimi-k2.5 only accepts temperature=1; omit it entirely for that model
        if "kimi" not in self.config.model.lower():
            body["temperature"] = self.config.temperature
        if stop:
            body["stop"] = stop

        # Retry with exponential backoff (handles transient errors, rate limits, etc.)
        # Generous retry budget: serial execution means we can afford to wait.
        max_retries = 12
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=600)
                if resp.status_code != 200:
                    log.error("LLM API error %d: %s", resp.status_code, resp.text[:500])
                # Retry on transient server/rate-limit errors (including 400 from gateway overload)
                retryable = resp.status_code in (400, 429, 500, 502, 503, 504)
                # Also retry 404 (sometimes gateway flakes) but not on last attempt
                retryable = retryable or (resp.status_code == 404 and attempt < max_retries - 1)
                if retryable:
                    wait = min(10 * (2 ** attempt), 300)  # 10, 20, 40, 80, 160, 300, 300…
                    log.warning("Retryable HTTP %d, waiting %d s… (attempt %d/%d)",
                                resp.status_code, wait, attempt + 1, max_retries)
                    time.sleep(wait)
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"LLM request failed after {max_retries} retries: HTTP {resp.status_code}\n"
                            f"Last response: {resp.text[:500]}"
                        )
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.Timeout as exc:
                if attempt < max_retries - 1:
                    wait = min(30 * (2 ** attempt), 300)
                    log.warning("LLM request timed out, waiting %d s… (attempt %d/%d)",
                                wait, attempt + 1, max_retries)
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"LLM request timed out after {max_retries} retries") from exc
            except (requests.RequestException, ValueError) as exc:
                if attempt < max_retries - 1:
                    wait = min(10 * (2 ** attempt), 300)
                    log.warning("LLM request failed (%s), waiting %d s… (attempt %d/%d)",
                                exc, wait, attempt + 1, max_retries)
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"LLM request failed after {max_retries} retries: {exc}") from exc

        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        usage_out = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
        self._total_usage["prompt_tokens"] += usage_out["prompt_tokens"]
        self._total_usage["completion_tokens"] += usage_out["completion_tokens"]
        self._call_count += 1

        log.debug(
            "LLM call: +%d prompt, +%d completion tokens",
            usage_out["prompt_tokens"],
            usage_out["completion_tokens"],
        )
        return text, usage_out

    # ------------------------------------------------------------------
    @property
    def total_usage(self) -> dict[str, int]:
        """Accumulated token counts across all calls."""
        return dict(self._total_usage)

    @property
    def call_count(self) -> int:
        """Total number of successful LLM API calls."""
        return self._call_count
