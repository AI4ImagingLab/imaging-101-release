"""Scoring: run tests, collect metrics, save results."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import RunConfig
from .docker_runner import DockerRunner
if TYPE_CHECKING:
    from .agent import AgentResult
    from .llm_client import LLMClient

log = logging.getLogger(__name__)


def _is_valid_doc(content: str | None) -> bool:
    """True if `content` is real plan text, not an empty/placeholder read result.

    `runner.read_file` returns strings like "[Error reading <path>]: ..." or
    "[File not found ...]" when the agent never produced the file.
    """
    if not content or not content.strip():
        return False
    head = content.lstrip()[:40].lower()
    return not (head.startswith("[error reading") or head.startswith("[file not found"))


@dataclass
class EvalResult:
    """Structured evaluation output."""

    task_name: str = ""
    mode: str = ""
    model: str = ""
    framework: str = "react"  # "react" | "multi_agent"
    level: str = "L1"  # end-to-end difficulty: "L1" | "L2" | "L3"
    timestamp: str = ""
    # Tests
    tests_total: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    test_pass_rate: float = 0.0
    test_details: list[dict] = field(default_factory=list)
    # Quality (end-to-end only)
    quality_metrics: dict | None = None
    # Plan evaluation (plan mode only)
    plan_scores: dict | None = None
    # Generated plan documents (plan mode only)
    generated_approach: str | None = None
    generated_design: str | None = None
    # Cost
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    iterations: int = 0
    llm_calls: int = 0  # Total individual LLM API calls (ReAct: =iterations, Multi-Agent: >>iterations)
    # Agent
    stopped_reason: str = ""
    files_created: list[str] = field(default_factory=list)
    # Visualization (end-to-end only)
    visualization_paths: dict[str, str] = field(default_factory=dict)


class Scorer:
    """Runs tests inside the container and assembles an EvalResult."""

    def __init__(self, runner: DockerRunner, config: RunConfig,
                 llm_client: LLMClient | None = None) -> None:
        self.runner = runner
        self.config = config
        self.llm_client = llm_client
        # Captured generated plan documents (plan mode only)
        self._generated_approach: str | None = None
        self._generated_design: str | None = None
        # Raw pytest output from the last _run_tests call (function mode)
        self._last_pytest_output: str | None = None

    # ------------------------------------------------------------------
    def score(
        self,
        agent_result: AgentResult,
        llm_usage: dict[str, int],
        wall_time: float,
        llm_calls: int = 0,
    ) -> EvalResult:
        result = EvalResult(
            task_name=self.config.task.task_name,
            mode=self.config.task.mode,
            model=self.config.llm.model,
            framework=self.config.framework,
            level=self.config.task.level,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_tokens=llm_usage.get("prompt_tokens", 0),
            completion_tokens=llm_usage.get("completion_tokens", 0),
            total_tokens=llm_usage.get("prompt_tokens", 0) + llm_usage.get("completion_tokens", 0),
            wall_time_seconds=round(wall_time, 2),
            iterations=agent_result.iterations,
            llm_calls=llm_calls,
            stopped_reason=agent_result.stopped_reason,
            files_created=agent_result.files_written,
        )

        # Run tests (function mode only — end-to-end uses quality metrics)
        if self.config.task.mode == "function":
            total, passed, failed, details = self._run_tests()
            result.tests_total = total
            result.tests_passed = passed
            result.tests_failed = failed
            result.test_pass_rate = passed / total if total > 0 else 0.0
            result.test_details = details

        # Plan mode: always capture the generated plan docs (survives sandbox cleanup)
        if self.config.task.mode == "plan":
            self._generated_approach = self.runner.read_file("plan/approach.md")
            self._generated_design = self.runner.read_file("plan/design.md")
            result.generated_approach = self._generated_approach
            result.generated_design = self._generated_design
            # LLM-as-judge rubric scoring — optional (skipped with --no-plan-score)
            if self.llm_client and not self.config.skip_plan_score:
                result.plan_scores = self._evaluate_plan()

        # Quality metrics for end-to-end (sole evaluation criterion)
        if self.config.task.mode == "end_to_end":
            result.quality_metrics = self._compute_quality_metrics()
            # Visualization is handled by a dedicated downstream agent,
            # not the end-to-end evaluation pipeline.
            # result.visualization_paths = self._generate_visualizations(
            #     result.quality_metrics, result
            # )

        return result

    # ------------------------------------------------------------------
    def _run_tests(self) -> tuple[int, int, int, list[dict]]:
        """Run pytest and parse the summary."""
        # -ra emits the "short test summary info" section (FAILED <nodeid> - <msg>)
        # that _run_tests parses into per-test `message` fields for error analysis.
        if self.config.task.mode == "function" and self.config.task.target_function:
            module = self.config.task.target_function.split(".")[0]
            test_cmd = f"python -m pytest evaluation/tests/test_{module}.py -v -ra --tb=short --no-header"
        else:
            test_cmd = "python -m pytest evaluation/tests/ -v -ra --tb=short --no-header"

        output, _ = self.runner.exec(test_cmd)
        log.info("Test output:\n%s", output)
        # Retain raw output so _save_function_mode can persist it for debugging
        self._last_pytest_output = output

        # Parse the "short test summary info" section for per-test error one-liners,
        # e.g. "FAILED tests/test_x.py::TestA::test_b - AssertionError: not close".
        messages: dict[str, str] = {}
        for m in re.finditer(r"^(?:FAILED|ERROR)\s+(\S+::\S+)\s+-\s+(.*)$", output, re.MULTILINE):
            messages[m.group(1)] = m.group(2).strip()

        # Parse per-test results (lines like "test_foo.py::TestBar::test_baz PASSED")
        details: list[dict] = []
        for m in re.finditer(r"(\S+::\S+)\s+(PASSED|FAILED|ERROR)", output):
            nodeid, status = m.group(1), m.group(2)
            entry = {"test": nodeid, "status": status}
            if status in ("FAILED", "ERROR") and nodeid in messages:
                entry["message"] = messages[nodeid]
            details.append(entry)

        # Parse summary line: "N passed", "M failed"
        passed = 0
        failed = 0
        pm = re.search(r"(\d+)\s+passed", output)
        fm = re.search(r"(\d+)\s+failed", output)
        em = re.search(r"(\d+)\s+error", output)
        if pm:
            passed = int(pm.group(1))
        if fm:
            failed = int(fm.group(1))
        if em:
            failed += int(em.group(1))
        total = passed + failed

        # Fallback: count from per-test details if summary was truncated
        if total == 0 and details:
            passed = sum(1 for d in details if d["status"] == "PASSED")
            failed = sum(1 for d in details if d["status"] in ("FAILED", "ERROR"))
            total = passed + failed

        return total, passed, failed, details

    # ------------------------------------------------------------------
    def _compute_quality_metrics(self) -> dict | None:
        """Compute end-to-end quality metrics (canonical generic design).

        The agent emits a single ``output/reconstruction.npy``; the scorer
        auto-discovers the task's reference array (``.npy`` or ``.npz``, with
        the right ``.npz`` key chosen by shape + canonical name), reduces both
        to comparable numeric arrays (squeeze / magnitude), and computes one
        metric set (range-normalized NRMSE, cosine NCC, MSE, PSNR, SSIM) plus
        the pass/fail verdict from ``evaluation/metrics.json`` boundaries. No
        per-task recipes or task-native output formats are required of the agent.
        """
        from . import reference_scoring

        from .benchmark import uses_npz_contract, score_submission
        if uses_npz_contract(self.config.task.task_dir):
            workspace = getattr(self.runner, "workspace", None)
            if workspace is not None and Path(workspace).is_dir():
                return score_submission(self.config.task.task_dir,
                                        Path(workspace) / "output/reconstruction.npz")
            # Copy only the inert submitted NPZ out of the container; the
            # trusted Judge and hidden references never enter the container.
            import subprocess
            import tempfile
            with tempfile.TemporaryDirectory(prefix="imaging101-submission-") as temp:
                prediction = Path(temp) / "reconstruction.npz"
                subprocess.run(["docker", "cp", f"{self.runner.container}:/workspace/output/reconstruction.npz",
                                str(prediction)], capture_output=True, check=False)
                return score_submission(self.config.task.task_dir, prediction)

        workspace = Path(self.runner.container) if getattr(self.runner, "container", None) else None
        if workspace and workspace.is_dir():
            metrics = reference_scoring.score_reconstruction(self.config.task.task_dir, workspace)
            if isinstance(metrics, dict):
                metrics.setdefault("scorer", "generic")
            return metrics
        # No filesystem workspace (e.g. Docker runner) — fall back to the
        # in-sandbox metric script in _compute_quality_metrics_generic.
        metrics = self._compute_quality_metrics_generic()
        if isinstance(metrics, dict):
            metrics.setdefault("scorer", "generic")
        return metrics

    # ------------------------------------------------------------------
    def _score_task_aware(self, task_name: str, workspace: Path) -> dict | None:
        """Score with the registered task-aware recipe, in-process.

        The harness environment carries every dependency the recipes need
        (numpy/scipy/scikit-image/h5py/opencv), and each recipe loads the
        task's own ``data/*.npz`` + ``src`` helpers directly, so we call the
        scorer in-process rather than shelling into the task venv. Reads the
        agent's reconstruction from the sandbox ``workspace`` (and
        ``workspace/output``). Returns the metrics dict (which may contain
        ``{"error": ...}`` if the agent produced no usable output), or ``None``
        on an unexpected failure so the caller falls back to the generic scorer.
        """
        from . import task_scoring

        try:
            metrics = task_scoring.score_task(task_name, self.config.task.task_dir, workspace)
        except Exception as exc:
            log.warning("Task-aware scoring raised for %s: %s", task_name, exc)
            return None
        if isinstance(metrics, dict):
            metrics.setdefault("scorer", "task_aware")
            log.info("Task-aware metrics for %s: %s", task_name, metrics)
        return metrics

    # ------------------------------------------------------------------
    def _compute_quality_metrics_generic(self) -> dict | None:
        """Compare reconstruction output against ground truth (generic 2-D).

        In end-to-end mode the evaluation/ directory is NOT copied into the
        sandbox, so we copy the ground truth file from the host task directory
        into the sandbox before running the comparison script.
        """
        import shutil as _shutil

        # Ensure the ground truth file is available in the sandbox
        # Check multiple possible locations for ground truth
        gt_candidates = [
            self.config.task.task_dir / "evaluation" / "reference_outputs" / "ground_truth.npy",
            self.config.task.task_dir / "data" / "ground_truth.npy",
        ]
        gt_host = None
        for candidate in gt_candidates:
            if candidate.exists():
                gt_host = candidate
                break
        if gt_host is None:
            log.warning("Ground truth file not found in any location: %s", gt_candidates)
            return {"error": "ground_truth.npy not found in task directory"}

        # Copy ground truth into the sandbox workspace
        workspace = Path(self.runner.container) if hasattr(self.runner, 'container') else None
        if workspace and workspace.is_dir():
            gt_dst = workspace / "evaluation" / "reference_outputs" / "ground_truth.npy"
            gt_dst.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(gt_host, gt_dst)
        else:
            # Docker runner — use exec to copy
            pass

        snippet = """\
import numpy as np, json, sys, os
out_path = "output/reconstruction.npy"
gt_path = "evaluation/reference_outputs/ground_truth.npy"
if not os.path.exists(out_path):
    print(json.dumps({"error": "output/reconstruction.npy not found"}))
    sys.exit(0)
out = np.load(out_path, allow_pickle=True)
if out.dtype == object:
    print(json.dumps({"error": "reconstruction is not a valid numeric array"}))
    sys.exit(0)
out = out.astype(np.float64)
if out.ndim != 2:
    print(json.dumps({"error": f"reconstruction has wrong dimensions: {out.ndim}"}))
    sys.exit(0)
gt = np.load(gt_path).astype(np.float64)
# Flux-normalize
out = out * (gt.sum() / (out.sum() + 1e-30))
# --- NRMSE ---
nrmse = float(np.linalg.norm(out - gt) / (np.linalg.norm(gt) + 1e-30))
# --- NCC ---
ncc = float(np.sum(out * gt) / (np.linalg.norm(out) * np.linalg.norm(gt) + 1e-30))
# --- MSE ---
mse = float(np.mean((out - gt) ** 2))
# --- PSNR ---
max_val = float(gt.max())
if mse > 0:
    psnr = float(20 * np.log10(max_val / np.sqrt(mse)))
else:
    psnr = float("inf")
# --- SSIM (simplified, no skimage dependency) ---
def _ssim_2d(a, b, data_range):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a2, sig_b2 = a.var(), b.var()
    sig_ab = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(num / den)
ssim = _ssim_2d(out, gt, data_range=max_val)
print(json.dumps({"nrmse": round(nrmse, 4), "ncc": round(ncc, 4),
                   "mse": round(mse, 8), "psnr": round(psnr, 2), "ssim": round(ssim, 4)}))
"""
        output, rc = self.runner.exec(f"python -c '{snippet}'")
        try:
            return json.loads(output.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            log.warning("Could not parse quality metrics: %s", output)
            return None

    # ------------------------------------------------------------------
    def _generate_visualizations(
        self, quality_metrics: dict | None, result: EvalResult
    ) -> dict[str, str]:
        """Generate evaluation figures and persist reconstruction.npy.

        Copies the agent's ``output/reconstruction.npy`` from the sandbox
        into the results directory (so it survives sandbox cleanup), then
        calls the visualizer to produce comparison figures.

        Returns dict mapping figure name → absolute path string.
        """
        import shutil as _shutil

        from .visualizer import generate_eval_figures

        if quality_metrics is None or "error" in quality_metrics:
            return {}

        # --- Determine output directory for this run's artifacts ---
        safe_model = result.model.replace("/", "_").replace("\\", "_")
        run_id = f"{result.task_name}_{result.framework}_{safe_model}"
        fig_dir = self.config.output_dir / "figures" / run_id
        fig_dir.mkdir(parents=True, exist_ok=True)

        # --- Copy reconstruction.npy from sandbox to results ---
        workspace = Path(self.runner.container) if hasattr(self.runner, "container") else None
        recon_src = workspace / "output" / "reconstruction.npy" if workspace else None
        recon_dst = fig_dir / "reconstruction.npy"

        if recon_src and recon_src.exists():
            _shutil.copy2(recon_src, recon_dst)
            log.info("Saved reconstruction to %s", recon_dst)
        else:
            log.warning("reconstruction.npy not found in sandbox — cannot generate figures")
            return {}

        # --- Load arrays ---
        gt_path = self.config.task.task_dir / "evaluation" / "reference_outputs" / "ground_truth.npy"
        if not gt_path.exists():
            gt_path = self.config.task.task_dir / "data" / "ground_truth.npy"
        if not gt_path.exists():
            log.warning("ground_truth.npy not found — cannot generate figures")
            return {}

        try:
            recon = np.load(str(recon_dst), allow_pickle=True)
            if recon.dtype == object or recon.ndim != 2:
                log.warning("reconstruction is not a valid 2D numeric array — cannot generate figures")
                return {}
            gt = np.load(str(gt_path))
        except Exception as e:
            log.warning("Failed to load arrays for visualization: %s", e)
            return {}

        # Also save ground truth copy for reference
        gt_dst = fig_dir / "ground_truth.npy"
        if not gt_dst.exists():
            _shutil.copy2(gt_path, gt_dst)

        # --- Generate figures ---
        run_label = f"{result.framework}_{safe_model}"
        paths = generate_eval_figures(
            reconstruction=recon,
            ground_truth=gt,
            metrics=quality_metrics,
            output_dir=fig_dir,
            run_label=run_label,
            task_name=result.task_name,
        )

        return paths

    # ------------------------------------------------------------------
    def _evaluate_plan(self) -> dict | None:
        """Evaluate generated plan using ELO-inspired pairwise + rubric scoring.

        Compares the generated plan against the reference plan (golden standard)
        shipped with the task, following the inverse_planning_eval methodology:
        1. Pairwise comparison with position swapping (3 rounds)
        2. Rubric scoring (6 dimensions, weighted average)
        3. Combined score: 0.5 * pairwise_win_rate + 0.5 * rubric_normalized
        """
        from .plan_scorer import evaluate_plan
        from dataclasses import asdict as _asdict

        readme = (self.config.task.task_dir / "README.md").read_text(encoding="utf-8")

        # Reuse the already-captured plan files, or read them if not yet captured
        generated_approach = self._generated_approach
        generated_design = self._generated_design
        if generated_approach is None:
            generated_approach = self.runner.read_file("plan/approach.md")
            self._generated_approach = generated_approach
        if generated_design is None:
            generated_design = self.runner.read_file("plan/design.md")
            self._generated_design = generated_design

        if not generated_approach or not generated_design:
            log.warning("Plan files not found in container")
            return {"error": "plan files not generated"}

        # Read reference (golden) plan from the task directory
        ref_approach_path = self.config.task.task_dir / "plan" / "approach.md"
        ref_design_path = self.config.task.task_dir / "plan" / "design.md"
        reference_approach = ref_approach_path.read_text(encoding="utf-8") if ref_approach_path.exists() else ""
        reference_design = ref_design_path.read_text(encoding="utf-8") if ref_design_path.exists() else ""

        log.info("Evaluating plan quality (pairwise + rubric)...")
        score = evaluate_plan(
            self.llm_client,
            readme,
            generated_approach,
            generated_design,
            reference_approach=reference_approach,
            reference_design=reference_design,
            n_pairwise_rounds=3,
        )
        return _asdict(score)

    # ------------------------------------------------------------------
    def save(self, result: EvalResult, output_dir: Path) -> Path:
        """Save evaluation result.

        For function mode, saves into a structured directory:
            results/function_mode/{task}/{model_date}/{module}/result.json
            results/function_mode/{task}/{model_date}/{module}/src/{module}.py
        For other modes, saves flat JSON files in output_dir.
        """
        if result.mode == "function" and self.config.task.target_function:
            return self._save_function_mode(result, output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize model name for filename (replace / with _)
        safe_model = result.model.replace("/", "_").replace("\\", "_")
        # Include level in filename for end-to-end mode
        level_suffix = f"_{result.level}" if result.mode == "end_to_end" else ""
        name = f"{result.task_name}_{result.mode}{level_suffix}_{result.framework}_{safe_model}_{ts}.json"
        path = output_dir / name
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        log.info("Results saved to %s", path)

        # Persist generated plan documents so they survive sandbox cleanup.
        # Layout: results/planning/<model_name>/<task_name>/{approach,design}.md
        # Only write real content — skip empty or "[Error reading ...]" / "[File not
        # found ...]" placeholders the runner returns when the agent never produced
        # the file (so a missing .md correctly signals an incomplete run).
        if result.mode == "plan":
            approach_ok = _is_valid_doc(result.generated_approach)
            design_ok = _is_valid_doc(result.generated_design)
            if approach_ok or design_ok:
                plan_dir = output_dir / safe_model / result.task_name
                plan_dir.mkdir(parents=True, exist_ok=True)
                if approach_ok:
                    (plan_dir / "approach.md").write_text(result.generated_approach, encoding="utf-8")
                if design_ok:
                    (plan_dir / "design.md").write_text(result.generated_design, encoding="utf-8")
                log.info("Generated plan documents saved to %s (approach=%s design=%s)",
                         plan_dir, approach_ok, design_ok)
            if not (approach_ok and design_ok):
                log.warning("Incomplete plan for %s/%s — approach=%s design=%s",
                            safe_model, result.task_name, approach_ok, design_ok)

        return path

    def _save_function_mode(self, result: EvalResult, output_dir: Path) -> Path:
        """Save function-mode result into structured directory layout."""
        module = self.config.task.target_function.split(".")[0]
        safe_model = result.model.replace("/", "_").replace("\\", "_")
        date_str = datetime.now().strftime("%Y%m%d")

        # results/function_mode/{task}/{model_date}/{module}/
        mod_dir = output_dir / "function_mode" / result.task_name / f"{safe_model}_{date_str}" / module
        mod_dir.mkdir(parents=True, exist_ok=True)

        # Save result.json
        result_path = mod_dir / "result.json"
        result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        log.info("Function-mode result saved to %s", result_path)

        # Persist the raw pytest output (with tracebacks) for debugging
        if self._last_pytest_output:
            (mod_dir / "pytest_output.txt").write_text(self._last_pytest_output, encoding="utf-8")
            log.info("pytest output saved to %s", mod_dir / "pytest_output.txt")

        # Copy model-generated target module source from sandbox
        workspace = Path(self.runner.container) if hasattr(self.runner, 'container') and self.runner.container else None
        if workspace:
            src_file = workspace / "src" / f"{module}.py"
            if src_file.exists():
                dst_src = mod_dir / "src"
                dst_src.mkdir(exist_ok=True)
                import shutil
                shutil.copy2(src_file, dst_src / f"{module}.py")
                log.info("Archived model code: %s -> %s", src_file, dst_src / f"{module}.py")

        return result_path
