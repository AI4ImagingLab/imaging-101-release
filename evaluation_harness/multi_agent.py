"""Multi-agent pipeline: Plan → Architect → Code → Execute → Judge.

This module implements the multi-agent pipeline from agentic_pipeline_dev,
adapted for the imaging-101 benchmark evaluation harness. It uses the same
sandbox runner interface (exec/write_file/read_file) and LLMClient as the
existing ReAct agent, enabling fair comparison between the two frameworks.

The pipeline stages:
1. **Planner** — analyzes the task and produces an algorithmic plan
2. **Critic** — reviews the plan (PASS/REJECT loop)
3. **Architect** — designs the code skeleton (file structure + signatures)
4. **Coder** — implements each file based on plan + skeleton
5. **Execution** — runs main.py in the sandbox
6. **Judge** — diagnoses failures, routes tickets back to appropriate stage

Returns an AgentResult compatible with the existing Scorer.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .agent import AgentResult
from .agents import (
    ArchitectAgent,
    CoderAgent,
    CriticAgent,
    JudgeAgent,
    PlannerAgent,
)
from .llm_client import LLMClient

log = logging.getLogger(__name__)


class MultiAgentPipeline:
    """Orchestrates the Plan→Architect→Code→Execute→Judge pipeline.

    Interacts with the sandbox runner using the same interface as the ReAct
    agent (exec, write_file, read_file), so scoring is identical.
    """

    def __init__(
        self,
        client: LLMClient,
        runner: Any,  # DockerRunner or LocalRunner
        max_iterations: int = 5,
        log_file: Any = None,
    ) -> None:
        self.client = client
        self.runner = runner
        self.max_iterations = max_iterations
        self.log_file = log_file

        # Initialize agents
        self.planner = PlannerAgent(client)
        self.critic = CriticAgent(client)
        self.architect = ArchitectAgent(client)
        self.coder = CoderAgent(client)
        self.judge = JudgeAgent(client)

        # Pipeline state
        self.current_plan = ""
        self.current_skeleton = ""
        self.current_files: Dict[str, str] = {}  # filename -> code content
        self.failure_history: List[Dict] = []
        self.data_inventory = ""
        self.requirements = ""
        self.last_judge_feedback: Optional[Dict] = None  # Carry error context across stages

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def run(self, task_desc: str, data_spec: str = "",
            requirements: str = "",
            level: str = "L1",
            given_approach: Optional[str] = None,
            given_design: Optional[str] = None) -> AgentResult:
        """Execute the full multi-agent pipeline.

        Parameters
        ----------
        task_desc : str
            README.md content describing the imaging task.
        data_spec : str
            meta_data content describing data format.
        requirements : str
            requirements.txt content listing available packages.
        level : str
            Difficulty level — "L1" (task desc only), "L2" (+approach),
            "L3" (+approach+design).
        given_approach : str or None
            Pre-supplied approach.md content (for L2/L3).
        given_design : str or None
            Pre-supplied design.md content (for L3).

        Returns
        -------
        AgentResult
            Compatible with the existing Scorer for evaluation.
        """
        result = AgentResult()
        t0 = time.time()

        # Store requirements and level for use in agent prompts
        self.requirements = requirements
        self.level = level

        # For L2/L3: pre-seed the plan from the given approach/design
        if given_approach and level in ("L2", "L3"):
            self.current_plan = given_approach
            self.runner.write_file("plan/approach.md", given_approach)
            result.files_written.append("plan/approach.md")
            log.info("[Level %s] Using pre-supplied approach.md", level)

        if given_design and level == "L3":
            self.current_skeleton = given_design
            # Parse file blocks from the given design
            files = self._parse_file_blocks(given_design)
            if files:
                self.current_files = files
                log.info("[Level L3] Parsed %d file skeletons from given design", len(files))
            self.runner.write_file("plan/design.md", given_design)
            result.files_written.append("plan/design.md")
            log.info("[Level L3] Using pre-supplied design.md")

        # Phase 0: Explore data directory
        self._explore_data(result)
        self._log_to_file(f"# Multi-Agent Pipeline Log (Level: {level})\n")

        try:
            self._run_pipeline_loop(task_desc, data_spec, result)
        except Exception as exc:
            log.error("Multi-agent pipeline crashed: %s", exc, exc_info=True)
            self._log_to_file(f"\n## ❌ CRASH: {exc}\n")
            if not result.stopped_reason:
                result.stopped_reason = f"error: {type(exc).__name__}: {str(exc)[:200]}"
            if result.iterations == 0:
                result.iterations = len(self.failure_history) + 1

        return result

    def _run_pipeline_loop(self, task_desc: str, data_spec: str,
                           result: AgentResult) -> None:
        """Inner loop — separated so that run() can catch exceptions."""
        feedback = None

        # Determine starting ticket based on level
        level = getattr(self, "level", "L1")
        if level == "L3":
            # L3: approach + design both given → start at Coder
            ticket = "Coder"
            log.info("[Level L3] Skipping Planner and Architect — starting at Coder")
        elif level == "L2":
            # L2: approach given → start at Architect (to produce design)
            ticket = "Architect"
            log.info("[Level L2] Skipping Planner — starting at Architect")
        else:
            # L1: start from scratch
            ticket = "Planner"

        for iteration in range(1, self.max_iterations + 1):
            log.info("═" * 20 + f" Iteration {iteration} (Ticket: {ticket}) " + "═" * 20)
            self._log_to_file(f"\n## Iteration {iteration} — Ticket: {ticket}\n")

            # ─── Stage 1: Planning ───
            if ticket == "Planner":
                log.info("[Agent] Planner...")
                self._log_to_file("### Planner\n")

                plan_ctx = {
                    "task_desc": task_desc,
                    "data_spec": data_spec,
                    "data_inventory": self.data_inventory,
                    "requirements": self.requirements,
                    "feedback": feedback,
                    "failure_history": self._format_failure_history(),
                }
                self.current_plan = self.planner.generate(plan_ctx)
                self._log_to_file(f"**Plan:**\n{self.current_plan[:2000]}...\n")

                # Critic loop
                critic_passed = False
                for critic_round in range(3):
                    critic_resp_str = self.critic.generate({
                        "task_desc": task_desc,
                        "plan": self.current_plan,
                    })
                    try:
                        critic_resp = json.loads(critic_resp_str)
                        if critic_resp["decision"] == "PASS":
                            critic_passed = True
                            log.info("[Critic] PASS")
                            break
                        else:
                            log.info("[Critic] REJECT: %s", critic_resp.get("reason", ""))
                            feedback_str = f"Critic rejected: {critic_resp['reason']}"
                            if critic_resp.get("suggestion"):
                                feedback_str += f" | Fix: {critic_resp['suggestion']}"
                            plan_ctx["feedback"] = feedback_str
                            self.current_plan = self.planner.generate(plan_ctx)
                    except Exception as e:
                        log.warning("Critic parse error: %s", e)
                        break

                if not critic_passed:
                    log.warning("Critic rejected plan after max retries. Proceeding anyway.")

                # Write plan files to sandbox
                self.runner.write_file("plan/approach.md", self.current_plan)
                result.files_written.append("plan/approach.md")

                ticket = "Architect"

            # ─── Stage 2: Architecture ───
            if ticket == "Architect":
                log.info("[Agent] Architect...")
                self._log_to_file("### Architect\n")

                arch_ctx = {
                    "task_desc": task_desc,
                    "plan": self.current_plan,
                    "data_inventory": self.data_inventory,
                    "requirements": self.requirements,
                    "previous_skeleton": self.current_skeleton if self.current_skeleton else None,
                    "feedback": feedback if isinstance(feedback, dict) and feedback.get("ticket_assigned_to") == "Architect" else None,
                }

                for attempt in range(3):
                    arch_resp = self.architect.generate(arch_ctx)
                    self.current_skeleton = arch_resp

                    # Parse file blocks from architect's response
                    files = self._parse_file_blocks(arch_resp)
                    if files:
                        log.info("[Architect] Produced %d file skeletons", len(files))
                        self._log_to_file(f"Architect produced {len(files)} files: {list(files.keys())}\n")
                        break
                    else:
                        log.warning("[Architect] No file blocks parsed (attempt %d)", attempt + 1)
                        arch_ctx["feedback"] = (
                            "Your response did not contain parseable file blocks. "
                            "Output each file as:\n```python\n# FILE: <path>\n...\n```"
                        )
                else:
                    log.error("Architect failed to produce valid skeletons after 3 attempts")
                    # Create a minimal skeleton
                    files = self._create_minimal_skeleton(task_desc, self.current_plan)

                # Store skeleton files
                self.current_files = files

                # Write design.md
                design_doc = "# Code Design\n\n"
                for fname, code in files.items():
                    design_doc += f"## {fname}\n```python\n{code}\n```\n\n"
                self.runner.write_file("plan/design.md", design_doc)
                result.files_written.append("plan/design.md")

                ticket = "Coder"

            # ─── Stage 3: Coding ───
            if ticket == "Coder":
                log.info("[Agent] Coder...")
                self._log_to_file("### Coder\n")

                # Determine which files to implement
                if isinstance(feedback, dict) and feedback.get("ticket_assigned_to") == "Coder":
                    fix_target = feedback.get("fix_target")
                    if fix_target and fix_target in self.current_files:
                        # Check for repeated errors: if the same file was the
                        # fix_target in the previous iteration too, expand scope
                        # to include related files (the bug may be elsewhere)
                        repeated = (
                            len(self.failure_history) >= 2
                            and self.failure_history[-1].get("fix_target") == fix_target
                            and self.failure_history[-2].get("fix_target") == fix_target
                        )
                        if repeated:
                            log.info("[Coder] Same file targeted 2+ times. "
                                     "Expanding to all src/ files.")
                            # Also tell the Coder about the repeated failure
                            if isinstance(feedback, dict):
                                feedback["feedback"] = (
                                    (feedback.get("feedback", "") or "") +
                                    "\n\nIMPORTANT: The same error has occurred "
                                    "MULTIPLE TIMES after patching this file. "
                                    "The root cause may be in a DIFFERENT file "
                                    "(e.g., a class definition in another module). "
                                    "Check ALL files for consistency."
                                )
                            files_to_code = dict(self.current_files)
                        else:
                            files_to_code = {fix_target: self.current_files[fix_target]}
                    else:
                        files_to_code = dict(self.current_files)
                else:
                    files_to_code = dict(self.current_files)

                # Build full architecture overview so Coder can see all
                # module interfaces (prevents cross-module import mismatches)
                full_architecture = self._build_full_architecture_summary(
                    self.current_files
                )

                for fname, skeleton_code in files_to_code.items():
                    log.info("[Coder] Implementing %s", fname)
                    self._log_to_file(f"Coding: {fname}\n")

                    # Determine feedback for this file:
                    # 1. Direct Coder ticket with matching fix_target
                    # 2. If coming via Architect, inject the original error
                    #    context so the Coder knows what went wrong
                    file_feedback = None
                    if isinstance(feedback, dict) and feedback.get("fix_target") == fname:
                        file_feedback = feedback
                    elif self.last_judge_feedback and isinstance(feedback, dict) and feedback.get("ticket_assigned_to") == "Architect":
                        # Architect redesigned the skeleton due to an error.
                        # Pass the error context to the Coder so it can avoid
                        # repeating the same mistake in the implementation.
                        jf = self.last_judge_feedback
                        file_feedback = {
                            "analysis": f"Previous iteration failed. Root cause: {jf.get('analysis', 'N/A')[:300]}",
                            "feedback": f"Avoid this error: {jf.get('feedback', 'N/A')[:200]}",
                        }

                    coder_ctx = {
                        "target_file": fname,
                        "plan": self.current_plan,
                        "skeleton": skeleton_code,
                        "full_architecture": full_architecture,
                        "current_code": self.current_files.get(fname, ""),
                        "data_inventory": self.data_inventory,
                        "requirements": self.requirements,
                        "feedback": file_feedback,
                    }

                    # When doing a targeted fix (single file), also provide
                    # full code of other src/ modules so the Coder can see
                    # class definitions, data structures, etc. it depends on
                    if len(files_to_code) == 1:
                        related_code = []
                        for other_f, other_code in self.current_files.items():
                            if other_f != fname and other_f.startswith("src/") and other_f != "src/__init__.py":
                                related_code.append(
                                    f"# === {other_f} (current implementation) ===\n{other_code}"
                                )
                        if related_code:
                            # Limit total related code to avoid exceeding context
                            related_str = "\n\n".join(related_code)
                            if len(related_str) > 12000:
                                related_str = related_str[:12000] + "\n... (truncated)"
                            coder_ctx["full_architecture"] = (
                                full_architecture +
                                "\n\n### FULL CODE OF OTHER MODULES (for reference):\n" +
                                related_str
                            )

                    raw_code = self.coder.generate(coder_ctx)
                    clean_code = CoderAgent.extract_python(raw_code)

                    # Syntax check
                    syntax_ok = self._check_syntax(clean_code)
                    if not syntax_ok:
                        log.warning("[Coder] Syntax error in %s, retrying...", fname)
                        coder_ctx["feedback"] = {
                            "analysis": "Previous code had syntax errors. Fix them.",
                            "feedback": "Write syntactically correct Python code.",
                        }
                        coder_ctx["current_code"] = clean_code
                        raw_code = self.coder.generate(coder_ctx)
                        clean_code = CoderAgent.extract_python(raw_code)

                    self.current_files[fname] = clean_code

                    # Write to sandbox
                    self.runner.write_file(fname, clean_code)
                    if fname not in result.files_written:
                        result.files_written.append(fname)

                # Ensure src/__init__.py exists
                self.runner.write_file("src/__init__.py", "")
                if "src/__init__.py" not in result.files_written:
                    result.files_written.append("src/__init__.py")

                # ─── Stage 3.5: Pre-Execution Smoke Test ───
                smoke_error = self._run_smoke_test()
                if smoke_error:
                    log.warning("[Smoke Test] Failed: %s", smoke_error[:200])
                    self._log_to_file(
                        f"### Smoke Test FAILED\n```\n{smoke_error[:1000]}\n```\n"
                    )
                    # Identify the failing file from the traceback
                    fix_target = self._identify_error_file(smoke_error)
                    feedback = {
                        "ticket_assigned_to": "Coder",
                        "fix_target": fix_target,
                        "analysis": f"IMPORT/SMOKE TEST FAILURE (caught before running main.py):\n{smoke_error[-1000:]}",
                        "feedback": (
                            "Fix the error above. This was caught by importing "
                            "and calling functions with minimal data BEFORE running "
                            "the full pipeline. The fix is usually a simple bug: "
                            "wrong key name, missing import, wrong function signature, "
                            "or type mismatch."
                        ),
                    }
                    self.failure_history.append({
                        "iteration": iteration,
                        "ticket_assigned_to": "Coder",
                        "fix_target": fix_target,
                        "analysis": f"Smoke test failed: {smoke_error[:200]}",
                        "evidence": smoke_error[:500],
                        "feedback": "Fix import/call error found in smoke test",
                    })
                    self.last_judge_feedback = feedback
                    ticket = "Coder"
                    continue

                ticket = "Execution"

            # ─── Stage 4: Execution ───
            if ticket == "Execution":
                log.info("[System] Executing main.py...")
                self._log_to_file("### Execution\n")

                # Find the entry point
                entry = "main.py"
                if "main.py" not in self.current_files:
                    # Try to find an entry point
                    for f in self.current_files:
                        if "main" in f.lower():
                            entry = f
                            break

                output, rc = self.runner.exec(f"python {entry}")
                result.commands_run.append({
                    "cmd": f"python {entry}",
                    "exit_code": rc,
                    "output": output[:3000],
                })
                success = rc == 0

                self._log_to_file(f"Exit code: {rc}\n```\n{output[:2000]}\n```\n")

                # Check if reconstruction output exists
                check_output, check_rc = self.runner.exec("test -f output/reconstruction.npy")
                output_exists = check_rc == 0

                if success and output_exists:
                    # Validate reconstruction quality before declaring success
                    validation_issues = self._validate_reconstruction(output)
                    if validation_issues:
                        log.warning("Pipeline produced output but validation failed: %s",
                                    validation_issues)
                        self._log_to_file(
                            f"⚠️ Output exists but validation failed: {validation_issues}\n"
                        )
                        # Treat as a failure — send to Judge with validation info
                        output = (output or "") + f"\n\nVALIDATION WARNING: {validation_issues}"
                    else:
                        # Check optimizer convergence from execution output
                        convergence_issue = self._check_optimizer_convergence(output)
                        if convergence_issue and iteration < self.max_iterations:
                            log.info("Pipeline produced output but optimizer may not have converged well: %s",
                                     convergence_issue)
                            self._log_to_file(
                                f"⚠️ Output produced but quality concern: {convergence_issue}\n"
                                "Routing directly to Coder to fix solver.\n"
                            )
                            # Route directly to Coder with targeted feedback
                            # Skip Judge — this is a known solver/gradient issue
                            feedback = {
                                "ticket_assigned_to": "Coder",
                                "fix_target": "src/solvers.py",
                                "analysis": (
                                    f"OPTIMIZER CONVERGENCE FAILURE: {convergence_issue}\n\n"
                                    "The optimizer finished but the solution is poor. "
                                    "Common causes:\n"
                                    "1. Gradient function returns zeros or near-zeros "
                                    "(most common — verify gradient with finite differences)\n"
                                    "2. All variables stuck at lower bound (0) — initial "
                                    "guess too small or regularization pushes everything to 0\n"
                                    "3. Regularization weight λ is too large relative to "
                                    "data fidelity term\n"
                                    "4. Missing factor in gradient (e.g., forgot conjugate "
                                    "or factor of 2)"
                                ),
                                "feedback": (
                                    "FIX THE GRADIENT in src/solvers.py. Specifically:\n"
                                    "1. Add finite-difference gradient verification at startup: "
                                    "compute grad_numerical = approx_fprime(x0, objective, 1e-7) "
                                    "and print max|grad_analytical - grad_numerical|. If large, "
                                    "the analytical gradient is WRONG.\n"
                                    "2. Ensure initial guess is non-trivial (e.g., uniform "
                                    "positive values, not zeros).\n"
                                    "3. Print the gradient norm at the FIRST iteration to "
                                    "verify it is non-zero and reasonable.\n"
                                    "4. Check that regularization weight is not too large "
                                    "(start with λ=0.1 or smaller)."
                                ),
                            }
                            self.last_judge_feedback = feedback
                            self.failure_history.append({
                                "iteration": iteration,
                                "ticket_assigned_to": "Coder",
                                "fix_target": "src/solvers.py",
                                "analysis": f"Optimizer convergence failure: {convergence_issue}",
                                "evidence": convergence_issue,
                                "feedback": "Fix gradient computation in solvers.py",
                            })
                            ticket = "Coder"
                            continue
                        else:
                            log.info("✅ Pipeline succeeded — reconstruction.npy produced")
                            self._log_to_file("✅ **SUCCESS** — output/reconstruction.npy exists\n")
                            result.stopped_reason = "done"
                            result.iterations = iteration
                            return result

                # Execution failed or no output — invoke Judge
                log.warning("Pipeline failed (rc=%d, output_exists=%s)", rc, output_exists)

                # Quick-fix attempt for common errors
                if not success and self._attempt_quick_fix(output, result):
                    # Re-run after quick fix
                    output2, rc2 = self.runner.exec(f"python {entry}")
                    result.commands_run.append({
                        "cmd": f"python {entry} (quick-fix retry)",
                        "exit_code": rc2,
                        "output": output2[:3000],
                    })
                    check2, check2_rc = self.runner.exec("test -f output/reconstruction.npy")
                    if rc2 == 0 and check2_rc == 0:
                        validation_issues = self._validate_reconstruction(output2)
                        if not validation_issues:
                            log.info("✅ Quick-fix succeeded!")
                            result.stopped_reason = "done"
                            result.iterations = iteration
                            return result
                        output2 = (output2 or "") + f"\n\nVALIDATION WARNING: {validation_issues}"
                    output = output2  # Use updated output for Judge

                # ─── Stage 5: Judge ───
                log.info("[Agent] Judge analyzing failure...")
                self._log_to_file("### Judge\n")

                # Collect code context for judge
                all_code = ""
                for fname, code in self.current_files.items():
                    all_code += f"# === {fname} ===\n{code}\n\n"

                judge_ctx = {
                    "task_desc": task_desc,
                    "logs": output[-2000:] if output else "No output",
                    "metrics": None,
                    "plan": self.current_plan,
                    "current_code": all_code,
                    "failure_history": self._format_failure_history(),
                }

                judgment = self.judge.generate(judge_ctx)
                log.info("[Judge] Ticket: %s | Analysis: %s",
                         judgment.get("ticket_assigned_to"),
                         judgment.get("analysis", "")[:200])
                self._log_to_file(
                    f"Ticket → **{judgment.get('ticket_assigned_to')}**\n"
                    f"Analysis: {judgment.get('analysis', '')[:300]}\n"
                )

                # Record failure
                self.failure_history.append({
                    "iteration": iteration,
                    "ticket_assigned_to": judgment.get("ticket_assigned_to", "Coder"),
                    "fix_target": judgment.get("fix_target"),
                    "analysis": judgment.get("analysis", ""),
                    "evidence": judgment.get("evidence", ""),
                    "feedback": judgment.get("feedback", ""),
                })

                # Stuck detection: same agent assigned 3+ consecutive times
                if len(self.failure_history) >= 3:
                    recent = [h["ticket_assigned_to"] for h in self.failure_history[-3:]]
                    if len(set(recent)) == 1:
                        stuck_agent = recent[0]
                        if stuck_agent in ("Coder", "Architect"):
                            log.warning("Stuck detection: %s assigned 3x. Escalating to Planner.", stuck_agent)
                            judgment["ticket_assigned_to"] = "Planner"
                            judgment["feedback"] = (
                                f"The {stuck_agent} has failed 3 times with the current approach. "
                                "Propose a COMPLETELY DIFFERENT and SIMPLER algorithm."
                            )

                ticket = judgment.get("ticket_assigned_to", "Coder")

                # ── Level-aware routing constraints ──
                # L3: approach + design are fixed — never route to Planner or Architect
                # L2: approach is fixed — never route to Planner
                level = getattr(self, "level", "L1")
                if level == "L3" and ticket in ("Planner", "Architect"):
                    log.info("Level L3: overriding ticket %s → Coder (approach+design are fixed)", ticket)
                    ticket = "Coder"
                    judgment["ticket_assigned_to"] = "Coder"
                    judgment["feedback"] = (
                        (judgment.get("feedback", "") or "") +
                        "\nNOTE: The approach and design are FIXED (Level 3 evaluation). "
                        "You must fix the implementation within the given design."
                    )
                elif level == "L2" and ticket == "Planner":
                    log.info("Level L2: overriding ticket Planner → Architect (approach is fixed)", ticket)
                    ticket = "Architect"
                    judgment["ticket_assigned_to"] = "Architect"
                    judgment["feedback"] = (
                        (judgment.get("feedback", "") or "") +
                        "\nNOTE: The approach is FIXED (Level 2 evaluation). "
                        "Redesign the code architecture within the given approach."
                    )

                # Safety override: prevent routing runtime errors to Architect
                # Architect rewrites ALL files from scratch, which is very expensive.
                # KeyError, TypeError, ValueError, IndexError etc. are code-body bugs
                # that the Coder should fix in the specific file.
                if ticket == "Architect" and output:
                    runtime_errors = ["KeyError", "TypeError", "ValueError", "IndexError",
                                      "AttributeError", "NameError", "ZeroDivisionError",
                                      "FileNotFoundError", "RuntimeError"]
                    for err_type in runtime_errors:
                        if err_type in output:
                            log.info("Override: Judge routed to Architect for %s, "
                                     "redirecting to Coder (runtime error).", err_type)
                            ticket = "Coder"
                            judgment["ticket_assigned_to"] = "Coder"
                            break

                feedback = judgment
                # Preserve judge analysis so Coder sees error context
                # even when routed through Architect first
                self.last_judge_feedback = judgment

        # Exhausted iterations
        log.warning("Multi-agent pipeline exhausted %d iterations", self.max_iterations)
        result.stopped_reason = "max_iterations"
        result.iterations = self.max_iterations
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_smoke_test(self) -> str:
        """Run a lightweight smoke test on the generated code.

        Imports all src/ modules and tries to call their main functions
        with tiny dummy data. This catches import errors, KeyErrors,
        TypeErrors, and other bugs BEFORE running the full pipeline.

        Returns empty string if OK, or the error output if something fails.
        """
        # Build a smoke-test script that imports all modules
        src_modules = [
            f for f in self.current_files
            if f.startswith("src/") and f.endswith(".py") and f != "src/__init__.py"
        ]
        if not src_modules:
            return ""

        # Generate import lines
        import_lines = []
        for mod_path in src_modules:
            # src/preprocessing.py -> src.preprocessing
            mod_name = mod_path.replace("/", ".").replace(".py", "")
            import_lines.append(f"import {mod_name}")

        smoke_script = (
            "import sys, traceback\n"
            "print('=== Smoke Test: Importing modules ===')\n"
            "try:\n"
            + "\n".join(f"    {line}" for line in import_lines) + "\n"
            "    print('All imports OK')\n"
            "except Exception as e:\n"
            "    traceback.print_exc()\n"
            "    sys.exit(1)\n"
            "\n"
            "# Quick syntax/import check for main.py\n"
            "print('=== Smoke Test: Checking main.py imports ===')\n"
            "try:\n"
            "    import importlib.util\n"
            "    spec = importlib.util.spec_from_file_location('main', 'main.py')\n"
            "    mod = importlib.util.module_from_spec(spec)\n"
            "    # Don't execute, just compile to check for syntax errors\n"
            "    import ast\n"
            "    with open('main.py') as f:\n"
            "        ast.parse(f.read())\n"
            "    print('main.py syntax OK')\n"
            "except Exception as e:\n"
            "    traceback.print_exc()\n"
            "    sys.exit(1)\n"
            "\n"
            "print('=== Smoke Test PASSED ===')\n"
        )

        self.runner.write_file("_smoke_test.py", smoke_script)
        output, rc = self.runner.exec("python _smoke_test.py")

        if rc != 0:
            return output or "Smoke test failed with no output"
        return ""

    def _identify_error_file(self, error_output: str) -> str:
        """Extract the source file that caused an error from a traceback.

        Returns the file path (e.g., 'src/preprocessing.py') or a default.
        """
        if not error_output:
            return "main.py"

        # Look for 'File "xxx.py"' in traceback, prefer src/ files
        import re as _re
        matches = _re.findall(r'File "([^"]+\.py)"', error_output)
        for fpath in reversed(matches):  # Last match is usually the root cause
            for fname in self.current_files:
                if fpath.endswith(fname):
                    return fname
        # Default to main.py
        return "main.py" if "main.py" in self.current_files else next(
            (f for f in self.current_files if f.endswith(".py")), "main.py"
        )

    def _explore_data(self, result: AgentResult) -> None:
        """Explore the data/ directory to build a rich inventory.

        This provides agents with:
        - File listing with sizes
        - NPY/NPZ array shapes, dtypes, and ALL key names
        - Sample values for small arrays (to understand data semantics)
        - meta_data JSON content (imaging parameters)
        - requirements.txt content (available packages)
        """
        output, rc = self.runner.exec("ls -la data/ 2>/dev/null || echo 'No data directory'")
        self.data_inventory = f"Data directory contents:\n{output}\n"

        # Read meta_data if present (critical imaging parameters)
        meta_output, meta_rc = self.runner.exec(
            "cat data/meta_data 2>/dev/null || echo '[No meta_data file]'"
        )
        if meta_rc == 0 and meta_output and not meta_output.startswith("[No"):
            self.data_inventory += f"\nImaging Parameters (data/meta_data):\n{meta_output}\n"

        # Read requirements.txt if present (available packages)
        req_output, req_rc = self.runner.exec(
            "cat requirements.txt 2>/dev/null || echo '[No requirements.txt]'"
        )
        if req_rc == 0 and req_output and not req_output.startswith("[No"):
            self.data_inventory += (
                f"\nAvailable Python Packages (requirements.txt):\n{req_output}\n"
                "IMPORTANT: ONLY these packages (plus standard library) are available. "
                "Do NOT use jax, torch, tensorflow, or any package not listed above.\n"
            )

        # Deep data inspection: shapes, dtypes, key names, sample values
        shape_script = (
            "import numpy as np, os, json, sys\n"
            "info = {}\n"
            "for f in sorted(os.listdir('data')):\n"
            "    path = os.path.join('data', f)\n"
            "    if f.endswith('.npy'):\n"
            "        arr = np.load(path, allow_pickle=True)\n"
            "        entry = {'shape': list(arr.shape), 'dtype': str(arr.dtype)}\n"
            "        if arr.size <= 10:\n"
            "            entry['values'] = arr.tolist()\n"
            "        elif arr.ndim >= 1:\n"
            "            entry['first_3'] = arr.flat[:3].tolist()\n"
            "            entry['min'] = float(np.nanmin(arr))\n"
            "            entry['max'] = float(np.nanmax(arr))\n"
            "        info[f] = entry\n"
            "    elif f.endswith('.npz'):\n"
            "        npz = np.load(path, allow_pickle=True)\n"
            "        file_info = {}\n"
            "        for k in sorted(npz.files):\n"
            "            arr = npz[k]\n"
            "            entry = {'shape': list(arr.shape), 'dtype': str(arr.dtype)}\n"
            "            if arr.size <= 10:\n"
            "                entry['values'] = arr.tolist()\n"
            "            elif arr.ndim >= 1 and np.issubdtype(arr.dtype, np.number):\n"
            "                entry['first_3'] = arr.flat[:3].tolist()\n"
            "                entry['min'] = float(np.nanmin(arr))\n"
            "                entry['max'] = float(np.nanmax(arr))\n"
            "            file_info[k] = entry\n"
            "        info[f] = file_info\n"
            "    elif f.endswith('.json'):\n"
            "        try:\n"
            "            with open(path) as jf:\n"
            "                info[f] = json.load(jf)\n"
            "        except: pass\n"
            "print(json.dumps(info, indent=2))\n"
        )
        self.runner.write_file("_explore_data.py", shape_script)
        shape_output, _ = self.runner.exec("python _explore_data.py")
        if shape_output:
            self.data_inventory += f"\nDetailed Data Inventory:\n{shape_output}\n"

        # Get Python version info for agents
        py_ver_output, _ = self.runner.exec("python --version 2>&1")
        if py_ver_output:
            self.data_inventory += f"\nPython Version: {py_ver_output.strip()}\n"

        result.commands_run.append({
            "cmd": "explore data/",
            "exit_code": 0,
            "output": self.data_inventory[:2000],
        })

    def _parse_file_blocks(self, response: str) -> Dict[str, str]:
        """Parse file blocks from architect's response.

        Expected format:
        ```python
        # FILE: src/preprocessing.py
        import ...
        ```
        """
        files = {}

        # Pattern 1: ```python\n# FILE: <path>\n...\n```
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", response, re.DOTALL)
        for block in blocks:
            # Look for FILE comment
            file_match = re.match(r"#\s*FILE:\s*(.+)", block.strip())
            if file_match:
                fname = file_match.group(1).strip()
                code = "\n".join(block.strip().split("\n")[1:])  # Skip FILE comment
                files[fname] = code

        # Pattern 2: Just look for "## filename.py" headers followed by code blocks
        if not files:
            sections = re.split(r"#{1,3}\s+(`?[\w/]+\.py`?)", response)
            for i in range(1, len(sections) - 1, 2):
                fname = sections[i].strip("`").strip()
                content = sections[i + 1]
                code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", content, re.DOTALL)
                if code_match:
                    files[fname] = code_match.group(1).strip()

        # Ensure we have main.py and at least one src file
        if files and "main.py" not in files:
            # Check for any file with "main" in it
            for f in list(files.keys()):
                if "main" in f.lower() and not f.startswith("src/"):
                    files["main.py"] = files.pop(f)
                    break

        return files

    def _create_minimal_skeleton(self, task_desc: str, plan: str) -> Dict[str, str]:
        """Create a minimal file structure if architect fails."""
        return {
            "src/__init__.py": "",
            "src/preprocessing.py": (
                "import numpy as np\n\n"
                "def load_data(data_dir: str = 'data') -> dict:\n"
                "    \"\"\"Load observation data.\"\"\"\n"
                "    pass\n"
            ),
            "src/solvers.py": (
                "import numpy as np\n\n"
                "def reconstruct(observations: np.ndarray, **kwargs) -> np.ndarray:\n"
                "    \"\"\"Run reconstruction algorithm.\"\"\"\n"
                "    pass\n"
            ),
            "main.py": (
                "import numpy as np\n"
                "import os\n"
                "from src.preprocessing import load_data\n"
                "from src.solvers import reconstruct\n\n"
                "def main():\n"
                "    data = load_data()\n"
                "    result = reconstruct(data)\n"
                "    os.makedirs('output', exist_ok=True)\n"
                "    np.save('output/reconstruction.npy', result)\n\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            ),
        }

    def _check_syntax(self, code: str) -> bool:
        """Check if code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _attempt_quick_fix(self, error_output: str, result: AgentResult) -> bool:
        """Attempt quick fixes for common runtime errors.

        Returns True if a fix was applied, False otherwise.
        Only patches the specific file identified as causing the error,
        not all files.
        """
        if not error_output:
            return False

        patterns = [
            ("No module named", "Fix import or remove the unavailable dependency. "
             "ONLY use packages from requirements.txt."),
            ("cannot import name", "Fix import path"),
            ("unexpected keyword argument", "Remove unsupported kwarg"),
            ("can only convert an array of size 1", "Fix .item() usage"),
            ("FileNotFoundError", "Fix file path"),
            ("unsupported operand type(s) for |", "Use Optional[X] instead of X | None"),
            ("KeyError:", "Check data key names against data inventory"),
        ]

        matched_hint = None
        for pattern, hint in patterns:
            if pattern in error_output:
                matched_hint = hint
                break

        if not matched_hint:
            return False

        log.info("[Quick-Fix] Detected fixable error. Attempting targeted patch...")

        # Identify which file caused the error from the traceback
        target_file = None
        for line in error_output.split("\n"):
            # Look for 'File "xxx.py"' in traceback
            match = re.search(r'File "([^"]+\.py)"', line)
            if match:
                fpath = match.group(1)
                # Map absolute path back to relative
                for fname in self.current_files:
                    if fpath.endswith(fname):
                        target_file = fname
                        break

        # If we couldn't identify the file, pick the most likely one
        if target_file is None:
            # Default to main.py since it's the entry point
            target_file = "main.py" if "main.py" in self.current_files else next(
                (f for f in self.current_files if f.endswith(".py")), None
            )

        if target_file is None:
            return False

        log.info("[Quick-Fix] Targeting: %s", target_file)
        code = self.current_files.get(target_file, "")
        full_architecture = self._build_full_architecture_summary(
            self.current_files
        )
        fix_ctx = {
            "target_file": target_file,
            "plan": self.current_plan,
            "skeleton": code,
            "full_architecture": full_architecture,
            "current_code": code,
            "data_inventory": self.data_inventory,
            "requirements": self.requirements,
            "feedback": {
                "analysis": f"RUNTIME ERROR:\n{error_output[-800:]}",
                "feedback": f"Fix: {matched_hint}. Keep the same algorithm but fix the bug.",
            },
        }
        raw_code = self.coder.generate(fix_ctx)
        clean_code = CoderAgent.extract_python(raw_code)
        if self._check_syntax(clean_code):
            self.current_files[target_file] = clean_code
            self.runner.write_file(target_file, clean_code)
            if target_file not in result.files_written:
                result.files_written.append(target_file)
            return True

        return False

    def _format_failure_history(self) -> str:
        """Format failure history for injection into agent prompts."""
        if not self.failure_history:
            return ""
        lines = ["### Past Failures (Avoid Repeating)\n"]
        for h in self.failure_history[-3:]:
            lines.append(
                f"- Iter {h.get('iteration', '?')}: "
                f"{h.get('ticket_assigned_to', '?')} → {h.get('fix_target', 'N/A')}\n"
                f"  Cause: {h.get('analysis', 'N/A')[:100]}\n"
            )
        return "\n".join(lines)

    def _check_optimizer_convergence(self, exec_output: str) -> str:
        """Check execution output for signs of poor optimizer convergence.

        Returns empty string if convergence looks OK, or a description
        of the concern.
        """
        if not exec_output:
            return ""

        lower_output = exec_output.lower()

        # Signs of poor convergence
        concern_patterns = [
            ("abnormal_termination_in_lnsrch", "L-BFGS-B line search failed (ABNORMAL_TERMINATION)"),
            ("maximum number of function evaluations", "Optimizer hit max function evaluations without converging"),
            ("stopped the iterations because", "Optimizer stopped early"),
            ("lbfgs-b: there are no bounds", "L-BFGS-B ignoring bounds"),
        ]

        for pattern, desc in concern_patterns:
            if pattern in lower_output:
                return desc

        import re as _re

        # Check for L-BFGS-B Fortran summary line format:
        # " N    Tit     Tnf  Tnint  Skip  Nact     Projg        F"
        # " 4096      2     34  11149     0  4096   0.000D+00   3.246D+03"
        # Parse Tit (total iterations) from the summary table
        tit_matches = _re.findall(
            r"^\s*\d+\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\d+",
            exec_output, _re.MULTILINE,
        )
        if tit_matches:
            # Check all optimization rounds
            low_iter_rounds = []
            for i, m in enumerate(tit_matches):
                tit = int(m)
                if tit <= 5:
                    low_iter_rounds.append((i + 1, tit))
            if low_iter_rounds:
                desc = "; ".join(f"Round {r}: {t} iters" for r, t in low_iter_rounds)
                return (
                    f"Optimizer converged suspiciously fast ({desc}). "
                    "This usually means the gradient is wrong (returning zero), "
                    "all variables are stuck at bounds, or the regularization "
                    "is too strong. The reconstruction is likely poor."
                )

        # Check for "variables are exactly at the bounds" — sign of trivial solution
        bounds_matches = _re.findall(
            r"(\d+)\s+variables are exactly at the bounds", exec_output
        )
        if bounds_matches:
            max_at_bounds = max(int(m) for m in bounds_matches)
            # If more than half the variables are at bounds, it's suspicious
            n_match = _re.search(r"N\s*=\s*(\d+)", exec_output)
            if n_match:
                n_vars = int(n_match.group(1))
                if max_at_bounds > n_vars * 0.5:
                    return (
                        f"{max_at_bounds}/{n_vars} variables stuck at bounds. "
                        "The solver is returning a near-trivial solution. "
                        "Check that the gradient is correct and the bounds "
                        "are not too restrictive."
                    )

        # Also check scipy Python-format output: "nit: 5"
        nit_match = _re.search(r"(?:nit|number of iterations)[:\s]*(\d+)", lower_output)
        if nit_match:
            nit = int(nit_match.group(1))
            if nit <= 5:
                return f"Optimizer converged in only {nit} iterations — likely not a good solution"

        return ""

    def _build_full_architecture_summary(self, files: Dict[str, str]) -> str:
        """Build a compact summary of ALL file skeletons/interfaces.

        This gives the Coder visibility into every module's public API,
        preventing cross-module import mismatches (e.g., importing a class
        from a module that doesn't define it).
        """
        lines = []
        for fname in sorted(files):
            if fname == "src/__init__.py":
                continue
            code = files[fname]
            # Extract just the imports and function/class signatures
            sig_lines = []
            for line in code.split("\n"):
                stripped = line.strip()
                if (stripped.startswith(("import ", "from "))
                        or stripped.startswith(("def ", "class "))
                        or stripped.startswith("@")):
                    sig_lines.append(line)
            if sig_lines:
                lines.append(f"# --- {fname} ---")
                lines.extend(sig_lines)
                lines.append("")
        return "\n".join(lines) if lines else ""

    def _validate_reconstruction(self, exec_output: str) -> str:
        """Validate that reconstruction.npy is a meaningful result.

        Checks:
        1. The file is a valid 2-D numeric numpy array (not None/object)
        2. The array has reasonable values (not all zeros, not trivially small)
        3. The execution output doesn't contain warnings about silent failures

        Returns empty string if valid, or a description of the issue.
        """
        # Check 1: Validate the array itself
        validate_script = (
            "import numpy as np, json, sys\n"
            "try:\n"
            "    arr = np.load('output/reconstruction.npy', allow_pickle=True)\n"
            "    if arr.dtype == object:\n"
            "        print(json.dumps({'error': 'Array is object dtype (likely None or non-numeric)'}))\n"
            "    elif arr.ndim != 2:\n"
            "        print(json.dumps({'error': f'Array is {arr.ndim}-D, expected 2-D'}))\n"
            "    elif arr.size == 0:\n"
            "        print(json.dumps({'error': 'Array is empty'}))\n"
            "    elif np.all(arr == 0) or np.all(np.isnan(arr)):\n"
            "        print(json.dumps({'error': 'Array is all zeros or all NaN'}))\n"
            "    elif np.std(arr) < 1e-20:\n"
            "        print(json.dumps({'error': f'Array is essentially constant (std={float(np.std(arr)):.2e})'}))\n"
            "    else:\n"
            "        print(json.dumps({'ok': True, 'shape': list(arr.shape), 'min': float(arr.min()), 'max': float(arr.max()), 'std': float(np.std(arr))}))\n"
            "except Exception as e:\n"
            "    print(json.dumps({'error': str(e)}))\n"
        )
        self.runner.write_file("_validate_recon.py", validate_script)
        val_output, val_rc = self.runner.exec("python _validate_recon.py")

        try:
            val_result = json.loads(val_output.strip().splitlines()[-1])
            if "error" in val_result:
                return val_result["error"]
        except Exception:
            return f"Could not validate reconstruction: {val_output[:200]}"

        # Check 2: Look for warning patterns in execution output that indicate
        # silent failures (optimizer not converging, constraints ignored, etc.)
        warning_patterns = [
            "cannot handle constraints",
            "ABNORMAL_TERMINATION_IN_LNSRCH",
            "Maximum number of function evaluations has been exceeded",
            "NaN encountered",
            "overflow encountered",
            "invalid value encountered",
        ]
        found_warnings = []
        if exec_output:
            for pattern in warning_patterns:
                if pattern.lower() in exec_output.lower():
                    found_warnings.append(pattern)

        if found_warnings:
            return (
                f"Execution warnings suggest silent failure: {', '.join(found_warnings)}. "
                "The optimizer may not have converged properly."
            )

        return ""

    def _log_to_file(self, text: str) -> None:
        """Append text to the log file."""
        if not self.log_file:
            return
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass
