"""ReAct agent loop: Thought → Action → Observation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from .docker_runner import DockerRunner
from .llm_client import LLMClient
from .prompts import COMPACT_SUMMARY_PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT_FUNCTION

log = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Outcome of a single agent run."""

    messages: list[dict[str, str]] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    commands_run: list[dict] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = "max_iterations"  # "done" | "max_iterations" | "error"


class Agent:
    """Minimal ReAct agent that drives an LLM to write & test code."""

    def __init__(
        self,
        client: LLMClient,
        runner: DockerRunner,
        max_iterations: int = 10,
        max_context_messages: int = 10,
        mode: str = "function",
        log_file: Path | None = None,
    ) -> None:
        self.client = client
        self.runner = runner
        self.max_iterations = max_iterations
        self.max_context_messages = max_context_messages
        self.mode = mode
        self.log_file = log_file

        if self.log_file:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_file, "w", encoding="utf-8") as f:
                    f.write(f"# Agent Interaction Log\n\n")
                    f.write(f"**Date**: {datetime.now().isoformat()}\n")
                    f.write(f"**Mode**: {mode}\n")
                    f.write("---\n\n")
            except Exception as e:
                log.warning(f"Failed to initialize log file {self.log_file}: {e}")

    def _log_step(
        self,
        iteration: int,
        inputs: list[dict[str, str]],
        response: str,
        actions: list[tuple[str, str]],
        observations: list[str],
    ) -> None:
        """Append a single interaction step to the log file."""
        if not self.log_file:
            return

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"## Iteration {iteration}\n\n")

                # 1. Inputs (Full Context Window)
                f.write("### Context Window (Inputs)\n")
                for msg in inputs:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    
                    # Truncate very long inputs for readability (e.g. huge file reads)
                    display_content = content
                    if len(content) > 10000:
                        display_content = content[:5000] + "\n\n... [truncated 10000+ chars] ...\n\n" + content[-5000:]

                    f.write(f"#### {role}\n")
                    f.write(f"```text\n{display_content}\n```\n\n")

                # 2. Response
                f.write(f"### Model Response\n")
                f.write(f"```text\n{response}\n```\n\n")

                # 3. Execution
                f.write(f"### Execution\n")
                if not actions:
                    f.write("_No valid actions parsed._\n\n")
                
                for i, ((act_type, act_args), obs) in enumerate(zip(actions, observations)):
                    f.write(f"**Action {i+1}:** `{act_type}`\n")
                    # If args are multiline (like file content), block quote it
                    if "\n" in str(act_args):
                        f.write(f"```text\n{act_args}\n```\n")
                    else:
                        f.write(f"> {act_args}\n")
                    
                    # Observation
                    trunc_obs = obs
                    if len(obs) > 5000:
                        trunc_obs = obs[:2500] + "\n... [truncated] ...\n" + obs[-2500:]
                    f.write(f"**Observation:**\n```text\n{trunc_obs}\n```\n\n")
                
                f.write("---\n\n")

        except Exception as e:
            log.warning(f"Failed to write to log file: {e}")

    def _log_compact(self, iteration: int, messages: list[dict[str, str]]) -> None:
        """Log a compaction event to the interaction log file."""
        if not self.log_file:
            return
        try:
            # Find the compact summary message (should be at index 2)
            summary_content = ""
            for msg in messages:
                if msg.get("content", "").startswith("[Compact Summary"):
                    summary_content = msg["content"]
                    break
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"## 🗜️ Context Compaction (before iteration {iteration})\n\n")
                f.write(f"Messages after compaction: {len(messages)}\n")
                f.write(f"Total chars: {sum(len(m['content']) for m in messages)}\n\n")
                if summary_content:
                    f.write(f"### Compact Summary\n")
                    f.write(f"```text\n{summary_content[:3000]}\n```\n\n")
                f.write("---\n\n")
        except Exception as e:
            log.warning(f"Failed to log compact event: {e}")

    # ------------------------------------------------------------------
    def run(self, user_message: str) -> AgentResult:
        """Execute the ReAct loop until DONE or iteration limit."""
        result = AgentResult()
        # Use mode-appropriate system prompt:
        # - function mode mentions evaluation/tests
        # - end_to_end / plan modes use the generic prompt (no tests)
        sys_prompt = SYSTEM_PROMPT_FUNCTION if self.mode == "function" else SYSTEM_PROMPT
        messages: list[dict[str, str]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_message},
        ]

        for i in range(1, self.max_iterations + 1):
            result.iterations = i
            log.info("── Iteration %d/%d ──", i, self.max_iterations)

            # Compact history if the conversation has grown too large.
            # This replaces the old middle messages with an LLM-generated
            # summary, preserving key context while freeing token budget.
            if self._should_compact(messages):
                log.info("Triggering context compaction at iteration %d", i)
                messages = self._compact_history(messages)
                self._log_compact(i, messages)

            # Apply sliding window to keep conversation within context limits
            windowed = self._apply_sliding_window(messages)
            response_text, _ = self.client.chat(windowed)
            messages.append({"role": "assistant", "content": response_text})

            # Log first 500 chars and last 500 chars of response for debugging
            if len(response_text) > 1200:
                log.debug("Response (first 500): %s", response_text[:500])
                log.debug("Response (last 500): %s", response_text[-500:])
            else:
                log.debug("Response: %s", response_text)

            # Parse and execute actions from the response.
            actions = self._parse_all_actions(response_text)
            if not actions:
                log.info("Action: FORMAT_ERROR (no actions found)")
                observation = (
                    "[Format Error] Could not parse your action.\n"
                    "Please use exactly one of: WRITE_FILE, RUN, READ_FILE, or DONE."
                )
                messages.append({"role": "user", "content": f"Observation:\n{observation}"})
                continue

            # Check if DONE appears after simulated Observation blocks.
            # If so, it's a hallucinated DONE — strip it and execute real actions.
            has_simulated_done = False
            if any(a[0] == "DONE" for a in actions):
                simulated_obs_count = len(
                    re.findall(r"^Observation:", response_text, re.MULTILINE))
                if simulated_obs_count > 0:
                    log.warning(
                        "Stripping simulated DONE (found %d fake Observation "
                        "blocks in model response)", simulated_obs_count)
                    actions = [(t, a) for t, a in actions if t != "DONE"]
                    has_simulated_done = True

            done = False
            observations = []
            for action_type, action_args in actions:
                log.info("Action: %s", action_type)
                if action_type == "DONE":
                    done = True
                    break
                observation = self._execute_action(action_type, action_args, result)
                observations.append(observation)

            # -- Log the step --
            self._log_step(i, windowed, response_text, actions, observations)

            if done:
                # Mode-specific DONE gating:
                # - function: require at least one pytest run
                # - end_to_end: require main.py to have been executed
                # - plan: no gating (just write plan files)
                if self.mode == "end_to_end":
                    # Check if output/reconstruction.npy exists in the sandbox
                    # rather than checking for a specific filename like "main.py".
                    # The agent may name its entry point differently (e.g. solution.py).
                    _, rc = self.runner.exec("test -f output/reconstruction.npy")
                    output_exists = (rc == 0)
                    if not output_exists and result.iterations > 1:
                        log.warning(
                            "DONE signaled but output/reconstruction.npy not found — "
                            "forcing continuation (iter %d)", i)
                        observation = (
                            "[System] output/reconstruction.npy was not found.\n"
                            "You must run your pipeline to produce "
                            "output/reconstruction.npy before signaling DONE.\n"
                            "Run your entry point (e.g. python main.py) and ensure "
                            "it saves the reconstructed image to output/reconstruction.npy.\n"
                            "Then signal DONE when the output is saved."
                        )
                        messages.append({"role": "user",
                                         "content": f"Observation:\n{observation}"})
                        continue
                    if output_exists:
                        # Sanity-check: validate output shape and basic stats
                        sanity_result = self._sanity_check_output()
                        if sanity_result:
                            log.warning("DONE accepted but sanity check has warnings: %s", sanity_result)
                            # Still accept DONE — warnings are informational only
                elif self.mode == "function":
                    pytest_run = any(
                        "pytest" in c.get("cmd", "") for c in result.commands_run
                    )
                    if not pytest_run and result.iterations > 1:
                        log.warning(
                            "DONE signaled without running pytest — "
                            "forcing continuation (iter %d)", i)
                        observation = (
                            "[System] You must run pytest on your implementation "
                            "before signaling DONE.\n"
                            "Run: python -m pytest evaluation/tests/ -v --tb=short\n"
                            "Then fix any failures and signal DONE when ready."
                        )
                        messages.append({"role": "user",
                                         "content": f"Observation:\n{observation}"})
                        continue
                # plan mode: no gating — accept DONE immediately
                result.stopped_reason = "done"
                break

            # Format observations for next prompt
            obs_text = ""
            for i, obs in enumerate(observations):
                obs_text += f"Observation {i+1}: {obs}\n"
            
            messages.append({"role": "user", "content": obs_text})
        
        return result

    # ------------------------------------------------------------------
    # Sanity-check output
    # ------------------------------------------------------------------
    def _sanity_check_output(self) -> str | None:
        """Validate output/reconstruction.npy shape and basic statistics.

        Returns a warning string if issues are found, or None if OK.
        """
        try:
            import numpy as np
            out_path = self.runner.workspace / "output" / "reconstruction.npy"
            if not out_path.exists():
                return "output/reconstruction.npy not found"
            arr = np.load(str(out_path))
            warnings = []
            if arr.ndim != 2:
                warnings.append(f"expected 2D array, got {arr.ndim}D (shape={arr.shape})")
            if arr.size == 0:
                warnings.append("array is empty")
            if np.all(arr == 0):
                warnings.append("array is all zeros")
            if np.any(np.isnan(arr)):
                warnings.append(f"array contains {np.isnan(arr).sum()} NaN values")
            if np.any(np.isinf(arr)):
                warnings.append(f"array contains {np.isinf(arr).sum()} Inf values")
            return "; ".join(warnings) if warnings else None
        except Exception as exc:
            return f"sanity check error: {exc}"

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_all_actions(text: str) -> list[tuple[str, dict]]:
        """Extract ALL (action_type, args_dict) from the LLM response.

        Models sometimes simulate the entire multi-turn loop in a single
        response. This method extracts every action and returns them in
        order so they can be executed sequentially.
        """
        actions = []
        # Find all "Action:" markers — allow mid-line matches for models that
        # don't always put "Action:" on its own line (e.g., Gemini).
        action_matches = list(re.finditer(r"(?:^|(?<=\n)|(?<=\.))\ ?Action:\s*(\S+)", text, re.MULTILINE))
        if not action_matches:
            # Fallback: try anywhere in text
            action_matches = list(re.finditer(r"Action:\s*(\S+)", text))
        if not action_matches:
            return []

        for idx, action_match in enumerate(action_matches):
            action_type = action_match.group(1).upper()

            # Determine the text region for this action's arguments
            start = action_match.end()
            if idx + 1 < len(action_matches):
                # Find the start of simulated "Observation:" or "Thought:" before next action
                end = action_matches[idx + 1].start()
                rest = text[start:end]
            else:
                rest = text[start:]

            if action_type == "DONE":
                actions.append(("DONE", {}))
                break  # DONE terminates the list

            if action_type == "WRITE_FILE":
                path_m = re.search(r"^Path:\s*(.+)", rest, re.MULTILINE)
                if not path_m:
                    actions.append(("FORMAT_ERROR", {"reason": "WRITE_FILE missing Path:"}))
                    continue
                path = path_m.group(1).strip()

                content_m = re.search(r"^Content:\s*\n?", rest, re.MULTILINE)
                if not content_m:
                    actions.append(("FORMAT_ERROR", {"reason": "WRITE_FILE missing Content:"}))
                    continue
                content_start = content_m.end()

                end_m = re.search(r"^END_CONTENT\s*$", rest[content_start:], re.MULTILINE)
                if end_m:
                    content = rest[content_start : content_start + end_m.start()]
                else:
                    # If no END_CONTENT, take until next Thought/Action/Observation
                    next_marker = re.search(
                        r"^(Thought:|Action:|Observation:)",
                        rest[content_start:], re.MULTILINE
                    )
                    if next_marker:
                        content = rest[content_start : content_start + next_marker.start()]
                    else:
                        content = rest[content_start:]

                # Strip optional code fences
                content = re.sub(r"^```\w*\n?", "", content)
                content = re.sub(r"\n?```\s*$", "", content)

                actions.append(("WRITE_FILE", {"path": path, "content": content}))

            elif action_type == "RUN":
                cmd_m = re.search(r"^Command:\s*(.+)", rest, re.MULTILINE)
                if not cmd_m:
                    for line in rest.splitlines():
                        line = line.strip()
                        if line:
                            actions.append(("RUN", {"command": line}))
                            break
                    else:
                        actions.append(("FORMAT_ERROR", {"reason": "RUN missing Command:"}))
                else:
                    actions.append(("RUN", {"command": cmd_m.group(1).strip()}))

            elif action_type == "READ_FILE":
                path_m = re.search(r"^Path:\s*(.+)", rest, re.MULTILINE)
                if not path_m:
                    actions.append(("FORMAT_ERROR", {"reason": "READ_FILE missing Path:"}))
                else:
                    actions.append(("READ_FILE", {"path": path_m.group(1).strip()}))

            else:
                actions.append(("FORMAT_ERROR", {"reason": f"Unknown action: {action_type}"}))

        return actions

    @staticmethod
    def _parse_action(text: str) -> tuple[str, dict]:
        """Extract first (action_type, args_dict) from the LLM response.
        
        Kept for backward compatibility. Uses _parse_all_actions internally.
        """
        actions = Agent._parse_all_actions(text)
        if not actions:
            return "FORMAT_ERROR", {}
        return actions[0]

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def _execute_action(
        self,
        action_type: str,
        args: dict,
        result: AgentResult,
    ) -> str:
        if action_type == "FORMAT_ERROR":
            reason = args.get("reason", "Could not parse your action.")
            
            # Specific feedback for the common "CHECK" hallucination
            if "Unknown action: CHECK" in reason:
                return (
                    f"[Format Error] {reason}\n"
                    "CHECK is not a valid action. To verify data or code, you must "
                    "write a Python script (WRITE_FILE) and then execute it (RUN).\n"
                    "Valid actions: WRITE_FILE, RUN, READ_FILE, DONE."
                )

            return (
                f"[Format Error] {reason}\n"
                "Please use exactly one of: WRITE_FILE, RUN, READ_FILE, or DONE."
            )

        if action_type == "WRITE_FILE":
            path = args["path"]
            self.runner.write_file(path, args["content"])
            if path not in result.files_written:
                result.files_written.append(path)
            return f"File written: {path}"

        if action_type == "RUN":
            cmd = args["command"]
            output, rc = self.runner.exec(cmd)
            result.commands_run.append({"cmd": cmd, "exit_code": rc, "output": output})
            status = "OK" if rc == 0 else f"EXIT CODE {rc}"
            # Truncate long command output (e.g. verbose pytest)
            output = self._truncate_text(output, max_chars=8000, keep_start=3500, keep_end=3500)
            return f"[{status}]\n{output}"

        if action_type == "READ_FILE":
            content = self.runner.read_file(args["path"])
            return self._truncate_text(content, max_chars=8000, keep_start=3500, keep_end=3500)

        return f"[Error] Unhandled action type: {action_type}"

    # ------------------------------------------------------------------
    # Context management helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _truncate_text(
        text: str,
        max_chars: int = 12000,
        keep_start: int = 5000,
        keep_end: int = 5000,
    ) -> str:
        """Truncate *text* if it exceeds *max_chars*, keeping head and tail."""
        if len(text) <= max_chars:
            return text
        omitted = len(text) - keep_start - keep_end
        return (
            f"{text[:keep_start]}\n\n"
            f"[...truncated {omitted} characters...]\n\n"
            f"{text[-keep_end:]}"
        )

    def _apply_sliding_window(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Return a windowed copy of *messages* that fits the context budget.

        Always keeps:
          - messages[0]  (system prompt)
          - messages[1]  (initial user prompt)

        If the history has been compacted, messages[2] is the compact summary
        and is also always kept.  Then keeps the most recent messages.

        Finally, enforces a total character budget by truncating the largest
        messages (assistant WRITE_FILE responses) if the total exceeds the cap.
        """
        MAX_TOTAL_CHARS = 90_000  # ~22K tokens, safe for 128K context

        # Determine how many head messages to always keep:
        # 2 (system + initial user) or 3 (+ compact summary)
        head_count = 2
        if len(messages) > 2 and messages[2].get("role") == "user" and \
                messages[2].get("content", "").startswith("[Compact Summary"):
            head_count = 3

        if len(messages) <= self.max_context_messages:
            windowed = list(messages)
        else:
            head = messages[:head_count]
            recent_count = self.max_context_messages - head_count
            tail = messages[-recent_count:]
            dropped = len(messages) - head_count - recent_count
            dropped_iters = dropped // 2
            summary = {
                "role": "user",
                "content": (
                    f"[Earlier conversation with {dropped_iters} iterations "
                    f"({dropped} messages) omitted to fit context window. "
                    f"Refer to the Compact Summary above for accumulated context.]"
                ),
            }
            windowed = head + [summary] + tail
            log.debug(
                "Sliding window: kept %d head + 1 summary + %d recent "
                "(dropped %d messages)",
                head_count, len(tail), dropped,
            )

        # Enforce total character budget by truncating the largest messages
        total = sum(len(m["content"]) for m in windowed)
        while total > MAX_TOTAL_CHARS and len(windowed) > 3:
            # Find the largest non-head message
            max_idx, max_len = -1, 0
            for idx in range(head_count, len(windowed)):
                clen = len(windowed[idx]["content"])
                if clen > max_len:
                    max_idx, max_len = idx, clen
            if max_idx < 0 or max_len <= 2000:
                break  # nothing left to trim
            msg = windowed[max_idx]
            trimmed = self._truncate_text(msg["content"], max_chars=3000,
                                          keep_start=1500, keep_end=1200)
            windowed[max_idx] = {"role": msg["role"], "content": trimmed}
            total = sum(len(m["content"]) for m in windowed)
            log.debug(
                "Budget trim: msg[%d] %d→%d chars, total now %d",
                max_idx, max_len, len(trimmed), total,
            )

        return windowed

    # ------------------------------------------------------------------
    # Context compaction
    # ------------------------------------------------------------------
    def _should_compact(self, messages: list[dict[str, str]]) -> bool:
        """Decide whether the conversation needs compaction.

        Triggers when EITHER:
        - Total character count exceeds the compact threshold (60K chars), OR
        - Number of messages exceeds 2× max_context_messages

        Never triggers if the conversation is still short (< 8 messages) or
        if a compact summary already exists and was generated recently
        (within the last max_context_messages messages).
        """
        COMPACT_CHAR_THRESHOLD = 60_000  # trigger before the 90K hard cap
        MIN_MESSAGES_BEFORE_COMPACT = 8  # don't compact tiny conversations

        if len(messages) < MIN_MESSAGES_BEFORE_COMPACT:
            return False

        # Check if we already have a recent compact summary
        # (it would be at messages[2] after a prior compaction)
        if len(messages) > 2 and messages[2].get("role") == "user" and \
                messages[2].get("content", "").startswith("[Compact Summary"):
            # How many messages since the last compact?
            messages_since_compact = len(messages) - 3
            # Only re-compact if enough new messages have accumulated
            if messages_since_compact < self.max_context_messages:
                return False

        total_chars = sum(len(m["content"]) for m in messages)
        too_many_chars = total_chars > COMPACT_CHAR_THRESHOLD
        too_many_messages = len(messages) > 2 * self.max_context_messages

        return too_many_chars or too_many_messages

    def _compact_history(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Compact the conversation by summarizing old messages with an LLM call.

        Preserves:
          - messages[0]  (system prompt)
          - messages[1]  (initial user prompt)
          - Most recent ``max_context_messages // 2`` messages (assistant + user pairs)

        The dropped middle section is summarized by an LLM call into a single
        compact summary message inserted at position [2].

        Returns the compacted messages list (mutates nothing).
        """
        # Determine how many recent messages to keep verbatim
        keep_recent = max(self.max_context_messages // 2, 4)
        # Ensure we keep an even number (assistant+user pairs)
        if keep_recent % 2 != 0:
            keep_recent += 1

        head = messages[:2]  # system + initial user
        tail = messages[-keep_recent:]
        middle = messages[2:-keep_recent] if keep_recent < len(messages) - 2 else []

        if not middle:
            log.debug("Compact: nothing to compact (no middle section)")
            return list(messages)

        # Build a condensed representation of the middle section for the summarizer
        middle_text = self._format_history_for_summary(middle)

        log.info(
            "Compacting history: %d middle messages (%d chars) → LLM summary",
            len(middle), len(middle_text),
        )

        # Call LLM to produce the summary
        summary_messages = [
            {"role": "system", "content": COMPACT_SUMMARY_PROMPT},
            {"role": "user", "content": (
                "Summarize the following agent conversation history. "
                "This agent is working on a computational imaging reconstruction task.\n\n"
                f"--- CONVERSATION HISTORY ({len(middle)} messages) ---\n\n"
                f"{middle_text}"
            )},
        ]

        try:
            summary_text, _ = self.client.chat(summary_messages)
            # Ensure summary isn't too long
            if len(summary_text) > 4000:
                summary_text = summary_text[:4000] + "\n\n[Summary truncated]"
        except Exception as exc:
            log.warning("Compact LLM call failed: %s — falling back to basic summary", exc)
            summary_text = self._build_basic_summary(middle)

        compact_msg = {
            "role": "user",
            "content": (
                f"[Compact Summary — iterations 1-{len(middle) // 2} "
                f"({len(middle)} messages compacted)]\n\n{summary_text}"
            ),
        }

        compacted = head + [compact_msg] + tail
        total_before = sum(len(m["content"]) for m in messages)
        total_after = sum(len(m["content"]) for m in compacted)
        log.info(
            "Compact done: %d→%d messages, %d→%d chars (%.0f%% reduction)",
            len(messages), len(compacted),
            total_before, total_after,
            (1 - total_after / max(total_before, 1)) * 100,
        )

        return compacted

    @staticmethod
    def _format_history_for_summary(middle: list[dict[str, str]]) -> str:
        """Format the middle messages into a condensed text for the summarizer.

        Extracts the key information (actions, observations, errors) without
        including full file contents.
        """
        MAX_SUMMARY_INPUT = 30_000  # cap input to summarizer
        parts = []
        current_len = 0

        for msg in middle:
            role = msg["role"]
            content = msg["content"]

            if role == "assistant":
                # Extract just the Thought + Action lines, skip file contents
                thought_m = re.search(r"Thought:(.+?)(?=Action:)", content, re.DOTALL)
                thought = thought_m.group(1).strip()[:500] if thought_m else ""

                # Extract action types
                action_types = re.findall(r"Action:\s*(\S+)", content)

                # For WRITE_FILE, just note the path (not content)
                paths = re.findall(r"Path:\s*(.+)", content)
                path_info = f" → files: {', '.join(paths)}" if paths else ""

                # For RUN, note the command
                cmds = re.findall(r"Command:\s*(.+)", content)
                cmd_info = f" → commands: {'; '.join(cmds)}" if cmds else ""

                entry = (
                    f"[Assistant] Thought: {thought[:300]}\n"
                    f"  Actions: {', '.join(action_types)}{path_info}{cmd_info}"
                )
            else:
                # User messages are observations — keep them more intact
                # but truncate very long outputs
                if len(content) > 1500:
                    # Keep first 500 + last 500 chars of observations
                    entry = f"[Observation] {content[:700]}...[truncated]...{content[-500:]}"
                else:
                    entry = f"[Observation] {content}"

            if current_len + len(entry) > MAX_SUMMARY_INPUT:
                parts.append(f"[...{len(middle) - len(parts)} remaining messages truncated...]")
                break
            parts.append(entry)
            current_len += len(entry)

        return "\n\n".join(parts)

    @staticmethod
    def _build_basic_summary(middle: list[dict[str, str]]) -> str:
        """Fallback summary when the LLM compact call fails.

        Extracts files written, commands run, and error snippets.
        """
        files = []
        commands = []
        errors = []

        for msg in middle:
            content = msg["content"]
            # Extract file paths from WRITE_FILE actions
            for m in re.finditer(r"Path:\s*(.+)", content):
                path = m.group(1).strip()
                if path not in files:
                    files.append(path)
            # Extract commands
            for m in re.finditer(r"Command:\s*(.+)", content):
                commands.append(m.group(1).strip())
            # Extract error lines
            for m in re.finditer(r"(Error|EXIT CODE \d+|Traceback|FAILED).*", content):
                err = m.group(0)[:200]
                if err not in errors:
                    errors.append(err)

        summary = "## Current State\n"
        summary += f"Files written: {', '.join(files) if files else 'none'}\n\n"
        summary += "## Commands Run\n"
        for cmd in commands[-10:]:  # last 10 commands
            summary += f"- {cmd}\n"
        summary += "\n## Errors Encountered\n"
        for err in errors[-5:]:  # last 5 distinct errors
            summary += f"- {err}\n"

        return summary
