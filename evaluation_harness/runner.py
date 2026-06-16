"""Top-level benchmark runner: orchestrates agent, Docker/local, and scoring."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from .frameworks.react.agent import Agent, AgentResult
from .config import RunConfig
from .llm_client import LLMClient
from .frameworks.multi_agent.multi_agent import MultiAgentPipeline
from .frameworks.react.prompts import (
    end_to_end_impl_prompt,
    end_to_end_plan_prompt,
    end_to_end_L2_plan_prompt,
    end_to_end_L2_impl_prompt,
    end_to_end_L3_impl_prompt,
    function_prompt,
    plan_approach_prompt,
    plan_design_prompt,
)
from .scorer import EvalResult, Scorer

log = logging.getLogger(__name__)


def _docker_available() -> bool:
    """Check if Docker is available on this system."""
    return shutil.which("docker") is not None


class BenchmarkRunner:
    """Runs a single benchmark evaluation."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.client = LLMClient(config.llm)

        if _docker_available():
            from .docker_runner import DockerRunner
            self.runner = DockerRunner(
                image=config.docker_image,
                task_dir=config.task.task_dir,
                timeout=config.timeout_seconds,
            )
        else:
            log.warning("Docker not available — using local runner")
            from .local_runner import LocalRunner
            self.runner = LocalRunner(
                image=config.docker_image,
                task_dir=config.task.task_dir,
                timeout=config.timeout_seconds,
            )

    # ------------------------------------------------------------------
    # Which files the agent is allowed to see per mode
    # (matches the Evaluation Modes table in CLAUDE.md)
    VISIBLE_PATHS = {
        "plan": ["README.md", "data"],
        "function": ["README.md", "plan", "evaluation", "data"],
        "end_to_end": ["README.md", "data", "requirements.txt"],
    }

    def _get_visible_paths(self) -> list[str]:
        """Return visible paths considering mode and e2e level.

        For end-to-end mode:
          L1: README.md + data + requirements.txt  (agent plans from scratch)
          L2: README.md + data + requirements.txt + plan/approach.md  (approach given)
          L3: README.md + data + requirements.txt + plan/  (approach+design given)
        """
        mode = self.config.task.mode
        if mode != "end_to_end":
            return self.VISIBLE_PATHS.get(mode, [])

        level = self.config.task.level
        base = ["README.md", "data", "requirements.txt"]

        if level == "L2":
            # L2: also seed approach.md into workspace
            return base + ["plan/approach.md"]
        elif level == "L3":
            # L3: seed both approach.md and design.md
            return base + ["plan"]
        else:
            # L1 (default): no plan files
            return base

    # ------------------------------------------------------------------
    def run(self) -> EvalResult:
        """Execute the benchmark and return scored results."""
        self._validate()
        visible = self._get_visible_paths()
        self.runner.start(visible_paths=visible)
        t0 = time.time()

        try:
            mode = self.config.task.mode
            if mode == "plan":
                agent_result = self._run_plan_mode()
            elif mode == "function":
                agent_result = self._run_function_mode()
            elif mode == "end_to_end":
                agent_result = self._run_end_to_end_mode()
            else:
                raise ValueError(f"Unknown mode: {mode}")

            wall_time = time.time() - t0

            scorer = Scorer(self.runner, self.config, llm_client=self.client)
            result = scorer.score(agent_result, self.client.total_usage, wall_time,
                                  llm_calls=self.client.call_count)
            scorer.save(result, self.config.output_dir)
            return result
        finally:
            self.runner.stop()

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------
    def _run_plan_mode(self) -> AgentResult:
        """Generate approach.md and design.md."""
        readme = self._read_host_file("README.md")
        meta_data = self._read_host_file("data/meta_data")

        agent = Agent(self.client, self.runner, self.config.max_iterations,
                      mode="plan", log_file=self.config.log_file)

        # Phase 1: approach
        log.info("Plan mode — generating approach.md")
        prompt1 = plan_approach_prompt(readme, meta_data)
        result1 = agent.run(prompt1)

        # Phase 2: design (read the generated approach from container)
        approach = self.runner.read_file("plan/approach.md")
        log.info("Plan mode — generating design.md")
        prompt2 = plan_design_prompt(readme, approach)
        result2 = agent.run(prompt2)

        # Merge results
        result2.files_written = result1.files_written + result2.files_written
        result2.iterations = result1.iterations + result2.iterations
        return result2

    def _run_function_mode(self) -> AgentResult:
        """Implement a single target function."""
        target = self.config.task.target_function
        if not target:
            raise ValueError("function mode requires --target-function")

        readme = self._read_host_file("README.md")
        approach = self._read_host_file("plan/approach.md")
        design = self._read_host_file("plan/design.md")

        # Determine test file
        module = target.split(".")[0]
        test_path = f"evaluation/tests/test_{module}.py"
        test_content = self._read_host_file(test_path)

        # --- FIX: Copy reference implementations of non-target modules ---
        # Some test files have cross-module dependencies (e.g. test_physics_model
        # imports from src.preprocessing). In function mode the agent only
        # implements the target module, so we seed the workspace with the
        # reference implementations for all *other* modules so that cross-module
        # imports succeed.  The target module itself is NOT copied — the agent
        # must create it from scratch.
        self._seed_dependency_modules(module)

        prompt = function_prompt(readme, approach, design, target, test_content)
        agent = Agent(self.client, self.runner, self.config.max_iterations,
                      mode="function", log_file=self.config.log_file)
        return agent.run(prompt)

    def _seed_dependency_modules(self, target_module: str) -> None:
        """Copy reference src/ modules (except the target) into the sandbox.

        This ensures that test files with cross-module imports (e.g.
        test_physics_model.py importing src.preprocessing) can run without
        error, isolating failures to the target module's implementation.
        """
        src_dir = self.config.task.task_dir / "src"
        if not src_dir.is_dir():
            log.warning("No src/ directory found in task — skipping dependency seeding")
            return

        # Ensure src/__init__.py exists in the sandbox
        self.runner.write_file("src/__init__.py", "")

        for py_file in sorted(src_dir.glob("*.py")):
            mod_name = py_file.stem  # e.g. "preprocessing"
            if mod_name == "__init__":
                continue
            if mod_name == target_module:
                # Skip the target module — the agent must implement it
                log.info("Skipping target module src/%s.py (agent must implement)", mod_name)
                continue
            content = py_file.read_text(encoding="utf-8")
            self.runner.write_file(f"src/{py_file.name}", content)
            log.info("Seeded dependency: src/%s.py", mod_name)

    def _run_end_to_end_mode(self) -> AgentResult:
        """Full pipeline: plan → implement → run.

        In end-to-end mode the agent is NOT given any test cases or evaluation
        code.  It must freely design and implement the full reconstruction
        pipeline, then run ``main.py`` to produce ``output/reconstruction.npy``.
        Scoring is based solely on reconstruction quality (NRMSE / NCC) against
        the reference ground truth — no unit-test pass rate is computed.
        """
        # Check framework choice
        if self.config.framework == "multi_agent":
            return self._run_end_to_end_multi_agent()
        return self._run_end_to_end_react()

    def _run_end_to_end_react(self) -> AgentResult:
        """End-to-end using the ReAct single-agent framework.

        Behavior depends on ``self.config.task.level``:

        * **L1** (task description only): Phase 1 plans from scratch,
          Phase 2 implements.
        * **L2** (+ approach): Phase 1 only writes design.md using the
          given approach, Phase 2 implements.
        * **L3** (+ approach + design): No planning phase — agent directly
          implements using the provided approach and design.
        """
        readme = self._read_host_file("README.md")
        meta_data = self._read_host_file("data/meta_data")
        level = self.config.task.level

        if level == "L3":
            # ── L3: Skip planning entirely — approach + design already given ──
            approach = self._read_host_file("plan/approach.md")
            design = self._read_host_file("plan/design.md")

            # Seed plan/ files into sandbox
            self.runner.write_file("plan/approach.md", approach)
            self.runner.write_file("plan/design.md", design)

            impl_agent = Agent(self.client, self.runner, self.config.max_iterations,
                               mode="end_to_end", log_file=self.config.log_file)
            log.info("End-to-end [L3] — implementation only (approach+design given)")
            impl_prompt = end_to_end_L3_impl_prompt(readme, meta_data, approach, design)
            return impl_agent.run(impl_prompt)

        elif level == "L2":
            # ── L2: Approach given, agent writes design then implements ──
            approach = self._read_host_file("plan/approach.md")

            plan_agent = Agent(self.client, self.runner, self.config.max_iterations,
                               mode="plan", log_file=self.config.log_file)
            log.info("End-to-end [L2] — Phase 1: design only (approach given)")
            plan_prompt = end_to_end_L2_plan_prompt(readme, meta_data, approach)
            plan_result = plan_agent.run(plan_prompt)

            # Phase 2: implementation
            impl_agent = Agent(self.client, self.runner, self.config.max_iterations,
                               mode="end_to_end", log_file=self.config.log_file)
            approach_sandbox = self.runner.read_file("plan/approach.md")
            design_sandbox = self.runner.read_file("plan/design.md")
            log.info("End-to-end [L2] — Phase 2: implementation")
            impl_prompt = end_to_end_L2_impl_prompt(approach_sandbox, design_sandbox)
            impl_result = impl_agent.run(impl_prompt)

            impl_result.files_written = plan_result.files_written + impl_result.files_written
            impl_result.iterations = plan_result.iterations + impl_result.iterations
            return impl_result

        else:
            # ── L1 (default): Plan from scratch ──
            plan_agent = Agent(self.client, self.runner, self.config.max_iterations,
                               mode="plan", log_file=self.config.log_file)
            log.info("End-to-end [L1] — Phase 1: planning from scratch")
            plan_prompt = end_to_end_plan_prompt(readme, meta_data)
            plan_result = plan_agent.run(plan_prompt)

            # Phase 2: implementation
            impl_agent = Agent(self.client, self.runner, self.config.max_iterations,
                               mode="end_to_end", log_file=self.config.log_file)
            approach = self.runner.read_file("plan/approach.md")
            design = self.runner.read_file("plan/design.md")
            log.info("End-to-end [L1] — Phase 2: implementation")
            impl_prompt = end_to_end_impl_prompt(approach, design)
            impl_result = impl_agent.run(impl_prompt)

            impl_result.files_written = plan_result.files_written + impl_result.files_written
            impl_result.iterations = plan_result.iterations + impl_result.iterations
            return impl_result

    def _run_end_to_end_multi_agent(self) -> AgentResult:
        """End-to-end using the multi-agent (Plan→Architect→Code→Judge) pipeline.

        For L2/L3 levels, pre-supplied approach/design are passed so the
        pipeline can skip or constrain the corresponding stages.
        """
        readme = self._read_host_file("README.md")
        meta_data = self._read_host_file("data/meta_data")
        requirements = self._read_host_file("requirements.txt")
        level = self.config.task.level

        # Load approach/design from host for L2/L3
        given_approach = None
        given_design = None
        if level in ("L2", "L3"):
            given_approach = self._read_host_file("plan/approach.md")
            if given_approach.startswith("[File not found"):
                log.warning("L2/L3 requested but approach.md not found — falling back to L1")
                given_approach = None
                level = "L1"
        if level == "L3":
            given_design = self._read_host_file("plan/design.md")
            if given_design.startswith("[File not found"):
                log.warning("L3 requested but design.md not found — falling back to L2")
                given_design = None
                level = "L2"

        log.info("End-to-end [multi_agent, %s] — running pipeline", level)
        pipeline = MultiAgentPipeline(
            client=self.client,
            runner=self.runner,
            max_iterations=self.config.max_iterations,
            log_file=self.config.log_file,
        )
        return pipeline.run(
            task_desc=readme,
            data_spec=meta_data,
            requirements=requirements,
            level=level,
            given_approach=given_approach,
            given_design=given_design,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        task_dir = self.config.task.task_dir
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")
        if not (task_dir / "README.md").exists():
            raise FileNotFoundError(f"Missing README.md in {task_dir}")

    def _read_host_file(self, rel_path: str) -> str:
        """Read a file from the task directory on the host."""
        p = self.config.task.task_dir / rel_path
        if not p.exists():
            return f"[File not found: {rel_path}]"
        return p.read_text(encoding="utf-8")
