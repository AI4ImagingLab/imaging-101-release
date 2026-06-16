"""DeepCode runner — drives the HKUDS/DeepCode framework for imaging-101 tasks.

Two execution paths:
  1. CLI mode  — spawns `deepcode --cli` as a subprocess
  2. Sandbox mode — reuses the copilot_runner prepare/collect workflow
     (same anti-cheat sandbox, but DeepCode is the agent)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class DeepCodeRunner:
    """Run a DeepCode evaluation on an imaging-101 task.

    Parameters
    ----------
    task_name : str
        e.g. "eht_black_hole_original"
    task_dir : Path
        Path to tasks/<task_name>/
    level : str
        "L1", "L2", or "L3"
    config : DeepCodeConfig or None
        DeepCode-specific settings.  If None, uses defaults.
    """

    def __init__(
        self,
        task_name: str,
        task_dir: Path,
        level: str = "L1",
        config: Any = None,
    ) -> None:
        self.task_name = task_name
        self.task_dir = task_dir
        self.level = level
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_workspace(self, workspace_dir: Path | None = None) -> Dict[str, Any]:
        """Prepare a sandbox workspace for DeepCode.

        Reuses the same copilot_runner sandbox logic (anti-cheat, obfuscated
        ground truth, self_eval.py) but generates a DeepCode-friendly prompt.

        Returns dict with workspace_path, prompt_file, etc.
        """
        from evaluation_harness.frameworks.claude_code.copilot_runner import prepare_copilot_evaluation

        result = prepare_copilot_evaluation(
            task_name=self.task_name,
            task_dir=self.task_dir,
            level=self.level,
            workspace_dir=workspace_dir,
        )

        # Also write a DeepCode-specific prompt
        workspace = Path(result["workspace_path"])
        deepcode_prompt = self._generate_deepcode_prompt(workspace)
        prompt_path = workspace / "deepcode_prompt.txt"
        prompt_path.write_text(deepcode_prompt, encoding="utf-8")
        result["deepcode_prompt_file"] = str(prompt_path)

        log.info("DeepCode workspace prepared at %s", workspace)
        return result

    def run_cli(self, workspace_dir: Path, timeout: int = 3600) -> Dict[str, Any]:
        """Run DeepCode CLI on a prepared workspace.

        Requires `deepcode` to be installed (pip install deepcode-hku).

        Parameters
        ----------
        workspace_dir : Path
            Path to the prepared sandbox workspace.
        timeout : int
            Max seconds to wait.

        Returns
        -------
        dict with status, duration, stdout, stderr
        """
        if not shutil.which("deepcode"):
            raise RuntimeError(
                "DeepCode CLI not found. Install with: pip install deepcode-hku\n"
                "Or clone from: https://github.com/HKUDS/DeepCode"
            )

        prompt_file = workspace_dir / "deepcode_prompt.txt"
        if not prompt_file.exists():
            prompt_file = workspace_dir / ".prompt.md"

        prompt_text = prompt_file.read_text(encoding="utf-8")

        cmd = [
            "deepcode", "--cli",
            "--workspace", str(workspace_dir),
        ]

        # Check if Docker is available; fall back to --local mode if not
        if not shutil.which("docker"):
            log.warning("Docker not found, using --local mode for DeepCode")
            cmd = [
                "deepcode", "--local",
                "--workspace", str(workspace_dir),
            ]

        log.info("Launching DeepCode CLI: %s", " ".join(cmd))
        t0 = time.time()

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            proc = subprocess.run(
                cmd,
                input=prompt_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workspace_dir),
                env=env,
            )
            duration = time.time() - t0
            return {
                "status": "completed" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "duration_seconds": duration,
                "stdout": proc.stdout[-5000:] if len(proc.stdout) > 5000 else proc.stdout,
                "stderr": proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "returncode": -1,
                "duration_seconds": timeout,
                "stdout": "",
                "stderr": f"DeepCode timed out after {timeout}s",
            }

    def collect_and_score(
        self,
        workspace_dir: Path,
        output_dir: Path = Path("results"),
    ) -> Any:
        """Collect and score results from a completed DeepCode run.

        Uses the same scoring infrastructure as copilot framework.
        """
        from evaluation_harness.frameworks.claude_code.copilot_runner import collect_copilot_results
        from evaluation_harness.frameworks.claude_code.copilot_scorer import score_copilot_results

        raw = collect_copilot_results(workspace_dir, self.task_dir, self.level)

        eval_result = score_copilot_results(
            workspace_path=workspace_dir,
            task_dir=self.task_dir,
            task_name=self.task_name,
            level=self.level,
            agent_name="deepcode",
            framework="deepcode",
        )

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self.task_name}_end_to_end_{self.level}_deepcode_{ts}.json"
        result_path = output_dir / name

        import dataclasses
        with open(result_path, "w") as f:
            json.dump(dataclasses.asdict(eval_result), f, indent=2, default=str)

        log.info("Results saved to %s", result_path)
        return eval_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_deepcode_prompt(self, workspace: Path) -> str:
        """Generate a DeepCode-optimized prompt for the task."""
        readme_path = workspace / "README.md"
        readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

        reqs_path = workspace / "requirements.txt"
        reqs = reqs_path.read_text(encoding="utf-8") if reqs_path.exists() else ""

        approach = ""
        design = ""
        if self.level in ("L2", "L3"):
            ap = workspace / "plan" / "approach.md"
            if ap.exists():
                approach = ap.read_text(encoding="utf-8")
        if self.level == "L3":
            dp = workspace / "plan" / "design.md"
            if dp.exists():
                design = dp.read_text(encoding="utf-8")

        sections = [
            "# Computational Imaging Reconstruction Task",
            "",
            "## Task Description",
            readme,
            "",
            "## Requirements",
            f"```\n{reqs}\n```",
            "",
        ]

        if approach:
            sections += [
                "## Algorithmic Approach",
                approach,
                "",
            ]
        if design:
            sections += [
                "## Code Design",
                design,
                "",
            ]

        sections += [
            "## Deliverables",
            "1. Create `src/` directory with implementation modules",
            "2. Create `main.py` as the entry point",
            "3. Run `main.py` to produce `output/reconstruction.npy` (2D numpy array)",
            "",
            "## Constraints",
            "- DO NOT modify `self_eval.py` — it is read-only",
            "- DO NOT try to read or reverse-engineer the obfuscated ground truth",
            "- Use only packages listed in `requirements.txt`",
            "- The reconstruction must be a 2D numpy array saved as .npy",
            "",
            "## Self-Evaluation",
            "After producing your reconstruction, run:",
            "```bash",
            "python self_eval.py",
            "```",
            "This prints NRMSE, NCC, PSNR, SSIM metrics.",
            "Iterate to improve your metrics.",
        ]

        return "\n".join(sections)
