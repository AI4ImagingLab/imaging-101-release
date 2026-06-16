"""Third-party agent evaluation: sandbox preparation and result collection.

This module handles the "copilot" framework — evaluating black-box coding
agents (GitHub Copilot, Claude Code, Cursor, Windsurf, etc.) that we cannot
control programmatically.  The workflow is:

1. **Prepare** a sandbox workspace with visible files + a self-eval harness.
2. **Generate** a prompt that instructs the agent what to do.
3. The human operator pastes the prompt into the agent and lets it work.
4. **Collect** results from the sandbox after the agent finishes.
5. **Score** the results using the same Scorer infrastructure.

Anti-cheat design:
- evaluation/reference_outputs/ is NEVER copied to the sandbox.
- evaluation/tests/ and evaluation/fixtures/ are NEVER copied.
- src/ (reference implementation) is NEVER copied.
- A self-evaluation script (self_eval.py) is placed in the sandbox.  It
  loads the ground truth from an OBFUSCATED binary blob (compressed +
  base64-encoded + XOR-scrambled) that the agent cannot trivially inspect.
  The script prints only aggregate metrics (NRMSE, NCC, PSNR, SSIM) — it
  never exposes the raw ground truth array.
- The self-eval script is marked read-only and includes a stern warning.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import shutil
import zlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# XOR key for obfuscating ground truth blob
_OBFUSCATION_KEY = b"imaging101-benchmark-gt-v1"

# Files/dirs the agent is NEVER allowed to see
_FORBIDDEN = {
    "evaluation/reference_outputs",
    "evaluation/tests",
    "evaluation/fixtures",
    "src",            # reference implementation
    "notebooks",      # may contain answers
    "output",         # pre-computed outputs
}


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    """XOR data with a repeating key."""
    key_len = len(key)
    return bytes(b ^ key[i % key_len] for i, b in enumerate(data))


def _obfuscate_array(arr: np.ndarray) -> str:
    """Serialize a numpy array into an obfuscated base64 string.

    Pipeline: npy_bytes → zlib compress → XOR scramble → base64 encode.
    The agent would have to reverse all three steps AND know the XOR key
    to recover the ground truth.  The self_eval.py script in the sandbox
    embeds the key and reversal logic, but it only prints metrics, never
    the raw array.
    """
    import io
    buf = io.BytesIO()
    np.save(buf, arr)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)
    scrambled = _xor_bytes(compressed, _OBFUSCATION_KEY)
    encoded = base64.b64encode(scrambled).decode("ascii")
    return encoded


def _generate_self_eval_script(gt_blob: str, gt_shape: tuple, gt_hash: str) -> str:
    """Generate the self-evaluation Python script placed in the sandbox.

    This script:
    - Loads the reconstruction from output/reconstruction.npy
    - Decodes the obfuscated ground truth (embedded as a string literal)
    - Computes quality metrics (NRMSE, NCC, PSNR, SSIM)
    - Prints ONLY the metrics — NEVER the ground truth data
    - Optionally generates comparison visualizations with the GT blurred
      (so the agent can see spatial structure but not exact values)
    """
    return f'''\
#!/usr/bin/env python3
"""Self-evaluation script — DO NOT MODIFY.

This script evaluates your reconstruction against the ground truth.
It prints quality metrics and optionally generates comparison figures.

Usage:
    python self_eval.py                     # print metrics
    python self_eval.py --visualize         # also generate figures

WARNING: The ground truth is embedded in obfuscated form.  Any attempt
to extract, decode, or reverse-engineer the ground truth data constitutes
cheating and will invalidate your evaluation results.
"""

import sys
import os
import json
import base64
import zlib
import hashlib
import io

import numpy as np

# ---------------------------------------------------------------------------
# Obfuscated ground truth (DO NOT MODIFY)
# ---------------------------------------------------------------------------
_GT_BLOB = """{gt_blob}"""
_GT_SHAPE = {gt_shape}
_GT_HASH = "{gt_hash}"
_KEY = b"imaging101-benchmark-gt-v1"


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    key_len = len(key)
    return bytes(b ^ key[i % key_len] for i, b in enumerate(data))


def _load_gt() -> np.ndarray:
    """Decode the ground truth array (internal use only)."""
    decoded = base64.b64decode(_GT_BLOB)
    unscrambled = _xor_bytes(decoded, _KEY)
    decompressed = zlib.decompress(unscrambled)
    buf = io.BytesIO(decompressed)
    arr = np.load(buf)
    # Integrity check
    h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    assert h == _GT_HASH, "Ground truth integrity check failed — file may be corrupted"
    return arr


def _ssim_2d(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    """Simplified structural similarity index (no skimage needed)."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a2, sig_b2 = a.var(), b.var()
    sig_ab = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(num / den)


def compute_metrics(recon_path: str = "output/reconstruction.npy") -> dict:
    """Compute quality metrics for the reconstruction."""
    if not os.path.exists(recon_path):
        return {{"error": f"Reconstruction not found: {{recon_path}}"}}

    try:
        out = np.load(recon_path, allow_pickle=True)
    except Exception as e:
        return {{"error": f"Failed to load reconstruction: {{e}}"}}

    if out.dtype == object:
        return {{"error": "Reconstruction is not a valid numeric array"}}

    out = out.astype(np.float64)
    if out.ndim != 2:
        return {{"error": f"Reconstruction has wrong dimensions: {{out.ndim}} (expected 2)"}}

    gt = _load_gt().astype(np.float64)

    # Check shape compatibility
    if out.shape != gt.shape:
        return {{
            "error": f"Shape mismatch: reconstruction {{out.shape}} vs expected {{gt.shape}}",
            "expected_shape": list(gt.shape),
        }}

    # Flux-normalize
    out = out * (gt.sum() / (out.sum() + 1e-30))

    # NRMSE
    nrmse = float(np.linalg.norm(out - gt) / (np.linalg.norm(gt) + 1e-30))
    # NCC (Normalized Cross-Correlation)
    ncc = float(np.sum(out * gt) / (np.linalg.norm(out) * np.linalg.norm(gt) + 1e-30))
    # MSE
    mse = float(np.mean((out - gt) ** 2))
    # PSNR
    max_val = float(gt.max())
    psnr = float(20 * np.log10(max_val / np.sqrt(mse))) if mse > 0 else float("inf")
    # SSIM
    ssim = _ssim_2d(out, gt, data_range=max_val)

    return {{
        "nrmse": round(nrmse, 6),
        "ncc": round(ncc, 6),
        "mse": round(mse, 10),
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 6),
        "recon_shape": list(out.shape),
        "recon_min": round(float(out.min()), 6),
        "recon_max": round(float(out.max()), 6),
        "recon_sum": round(float(out.sum()), 6),
    }}


def generate_visualization(recon_path: str = "output/reconstruction.npy",
                           out_dir: str = "output") -> None:
    """Generate comparison figures (reconstruction vs blurred GT reference).

    The ground truth is intentionally blurred in the visualization to prevent
    the agent from pixel-copying it.  The visualization only shows spatial
    structure, not exact values.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    os.makedirs(out_dir, exist_ok=True)

    gt = _load_gt().astype(np.float64)
    out = np.load(recon_path, allow_pickle=True).astype(np.float64)
    out = out * (gt.sum() / (out.sum() + 1e-30))

    # Blur the GT so agent cannot pixel-copy
    gt_blurred = gaussian_filter(gt, sigma=3.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(out, cmap="afmhot", origin="lower")
    axes[0].set_title("Your Reconstruction")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(gt_blurred, cmap="afmhot", origin="lower")
    axes[1].set_title("Reference (blurred)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    diff = np.abs(out - gt)
    im2 = axes[2].imshow(diff, cmap="hot", origin="lower")
    axes[2].set_title("|Difference| (exact)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.suptitle("Reconstruction Evaluation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "self_eval_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {{out_dir}}/self_eval_comparison.png")


if __name__ == "__main__":
    visualize = "--visualize" in sys.argv or "-v" in sys.argv

    metrics = compute_metrics()
    print("=" * 60)
    print("SELF-EVALUATION RESULTS")
    print("=" * 60)

    if "error" in metrics:
        print(f"ERROR: {{metrics['error']}}")
        if "expected_shape" in metrics:
            print(f"Expected shape: {{metrics['expected_shape']}}")
        sys.exit(1)

    print(f"  NRMSE : {{metrics['nrmse']:.6f}}  (lower is better, target < 0.5)")
    print(f"  NCC   : {{metrics['ncc']:.6f}}  (higher is better, target > 0.8)")
    print(f"  PSNR  : {{metrics['psnr']:.2f}} dB  (higher is better)")
    print(f"  SSIM  : {{metrics['ssim']:.6f}}  (higher is better, target > 0.5)")
    print(f"  MSE   : {{metrics['mse']:.2e}}")
    print("-" * 60)
    print(f"  Shape : {{metrics['recon_shape']}}")
    print(f"  Range : [{{metrics['recon_min']:.6f}}, {{metrics['recon_max']:.6f}}]")
    print(f"  Sum   : {{metrics['recon_sum']:.6f}}")
    print("=" * 60)

    # Save metrics to JSON
    os.makedirs("output", exist_ok=True)
    with open("output/self_eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to output/self_eval_metrics.json")

    if visualize:
        try:
            generate_visualization()
        except Exception as e:
            print(f"Warning: could not generate visualization: {{e}}")
'''


# ---------------------------------------------------------------------------
# Sandbox workspace preparation
# ---------------------------------------------------------------------------
class CopilotSandbox:
    """Prepares and manages a sandbox workspace for third-party agent evaluation.

    The sandbox contains only the files the agent is allowed to see, plus
    a self-evaluation script with obfuscated ground truth.
    """

    def __init__(
        self,
        task_dir: Path,
        workspace_dir: Optional[Path] = None,
        level: str = "L1",
    ) -> None:
        self.task_dir = task_dir.resolve()
        self.level = level
        self.task_name = task_dir.name

        if workspace_dir is None:
            # Auto-generate workspace path
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = Path.home() / "copilot_workspaces"
            self.workspace = base / f"{self.task_name}_{level}_{ts}"
        else:
            self.workspace = workspace_dir.resolve()

    def prepare(self) -> Path:
        """Create the sandbox workspace and return its path.

        Returns the absolute path to the workspace directory that should
        be opened in the third-party agent's IDE.
        """
        log.info("Preparing copilot sandbox at %s", self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        # 1. Copy visible files based on level
        self._copy_visible_files()

        # 2. Generate and place self-evaluation script
        self._place_self_eval_script()

        # 3. Create agent instruction file (INSTRUCTIONS.md)
        # This is redundant with the prompt but helps agents that read files
        self._place_instructions_file()

        # 4. Set permissions
        self._set_permissions()

        log.info("Sandbox ready at %s", self.workspace)
        return self.workspace

    def collect_results(self) -> dict:
        """Collect results from the sandbox after agent has finished.

        Returns a dict with:
        - has_reconstruction: bool
        - reconstruction_path: str or None
        - metrics_path: str or None
        - self_eval_metrics: dict or None
        - files_created: list of relative paths
        - src_files: dict of filename -> content
        """
        results: dict = {
            "has_reconstruction": False,
            "reconstruction_path": None,
            "metrics_path": None,
            "self_eval_metrics": None,
            "files_created": [],
            "src_files": {},
        }

        # Check for reconstruction
        recon_path = self.workspace / "output" / "reconstruction.npy"
        if recon_path.exists():
            results["has_reconstruction"] = True
            results["reconstruction_path"] = str(recon_path)

        # Check for self-eval metrics
        metrics_path = self.workspace / "output" / "self_eval_metrics.json"
        if metrics_path.exists():
            results["metrics_path"] = str(metrics_path)
            try:
                results["self_eval_metrics"] = json.loads(metrics_path.read_text())
            except (json.JSONDecodeError, Exception):
                pass

        # Collect all files created/modified by the agent
        for root, dirs, files in os.walk(self.workspace):
            # Skip hidden dirs and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in files:
                full = Path(root) / f
                rel = full.relative_to(self.workspace)
                results["files_created"].append(str(rel))

                # Collect src/ contents
                if str(rel).startswith("src/") and f.endswith(".py"):
                    try:
                        results["src_files"][str(rel)] = full.read_text()
                    except Exception:
                        pass

        return results

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------
    def _copy_visible_files(self) -> None:
        """Copy only the files the agent is allowed to see."""
        # Always visible
        always_visible = ["README.md", "requirements.txt", "data"]

        # Level-dependent
        if self.level in ("L2", "L3"):
            # L2+: approach is given
            approach_src = self.task_dir / "plan" / "approach.md"
            if approach_src.exists():
                dst = self.workspace / "plan" / "approach.md"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(approach_src, dst)

        if self.level == "L3":
            # L3: design is also given
            design_src = self.task_dir / "plan" / "design.md"
            if design_src.exists():
                dst = self.workspace / "plan" / "design.md"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(design_src, dst)

        for item in always_visible:
            src = self.task_dir / item
            dst = self.workspace / item
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            else:
                log.warning("Visible item not found: %s", src)

        # Verify no forbidden files leaked
        for forbidden in _FORBIDDEN:
            forbidden_path = self.workspace / forbidden
            if forbidden_path.exists():
                log.error("FORBIDDEN path found in sandbox: %s — removing!", forbidden_path)
                if forbidden_path.is_dir():
                    shutil.rmtree(forbidden_path)
                else:
                    forbidden_path.unlink()

    def _place_self_eval_script(self) -> None:
        """Generate self_eval.py with obfuscated ground truth."""
        gt_path = self.task_dir / "evaluation" / "reference_outputs" / "ground_truth.npy"
        if not gt_path.exists():
            log.warning("Ground truth not found: %s — self-eval will not work", gt_path)
            # Place a stub self_eval.py
            script = '''\
#!/usr/bin/env python3
"""Self-evaluation script — ground truth not available for this task."""
print("ERROR: Ground truth not available. Cannot compute metrics.")
'''
            (self.workspace / "self_eval.py").write_text(script, encoding="utf-8")
            return

        gt = np.load(str(gt_path))
        gt_blob = _obfuscate_array(gt)
        gt_hash = hashlib.sha256(gt.tobytes()).hexdigest()[:16]

        script = _generate_self_eval_script(gt_blob, gt.shape, gt_hash)
        script_path = self.workspace / "self_eval.py"
        script_path.write_text(script, encoding="utf-8")
        log.info("Placed self_eval.py with obfuscated GT (%d bytes blob)", len(gt_blob))

    def _place_instructions_file(self) -> None:
        """Place an INSTRUCTIONS.md file in the sandbox root.

        This helps agents that automatically read markdown files at startup.
        """
        from .copilot_prompt import generate_instruction_file
        content = generate_instruction_file(self.task_name, self.level)
        (self.workspace / "INSTRUCTIONS.md").write_text(content, encoding="utf-8")

    def _set_permissions(self) -> None:
        """Set file permissions to protect evaluation infrastructure.

        Makes self_eval.py read-only so agents don't accidentally modify it.
        """
        self_eval = self.workspace / "self_eval.py"
        if self_eval.exists():
            os.chmod(self_eval, 0o444)  # read-only for all

        instructions = self.workspace / "INSTRUCTIONS.md"
        if instructions.exists():
            os.chmod(instructions, 0o444)


# ---------------------------------------------------------------------------
# High-level orchestration functions
# ---------------------------------------------------------------------------
def prepare_copilot_evaluation(
    task_name: str,
    task_dir: Path,
    level: str = "L1",
    workspace_dir: Optional[Path] = None,
) -> dict:
    """Prepare everything needed for a third-party agent evaluation.

    Returns a dict with:
    - workspace_path: str — path for the agent to open
    - prompt: str — prompt to paste into the agent
    - level: str
    - task_name: str
    """
    from .copilot_prompt import generate_agent_prompt

    sandbox = CopilotSandbox(task_dir, workspace_dir, level)
    workspace_path = sandbox.prepare()

    # Generate the prompt
    readme = (task_dir / "README.md").read_text(encoding="utf-8")
    meta_data_path = task_dir / "data" / "meta_data"
    meta_data = meta_data_path.read_text(encoding="utf-8") if meta_data_path.exists() else ""
    requirements = (task_dir / "requirements.txt").read_text(encoding="utf-8") if (task_dir / "requirements.txt").exists() else ""

    approach = None
    design = None
    if level in ("L2", "L3"):
        approach_path = task_dir / "plan" / "approach.md"
        approach = approach_path.read_text(encoding="utf-8") if approach_path.exists() else None
    if level == "L3":
        design_path = task_dir / "plan" / "design.md"
        design = design_path.read_text(encoding="utf-8") if design_path.exists() else None

    prompt = generate_agent_prompt(
        task_name=task_name,
        readme=readme,
        meta_data=meta_data,
        requirements=requirements,
        level=level,
        approach=approach,
        design=design,
    )

    # Save the prompt to a file in the workspace
    prompt_path = workspace_path / ".prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    os.chmod(prompt_path, 0o444)

    return {
        "workspace_path": str(workspace_path),
        "prompt": prompt,
        "prompt_file": str(prompt_path),
        "level": level,
        "task_name": task_name,
    }


def collect_copilot_results(
    workspace_path: Path,
    task_dir: Path,
    level: str = "L1",
) -> dict:
    """Collect and score results from a completed copilot evaluation.

    Returns a dict with metrics and metadata.
    """
    sandbox = CopilotSandbox(task_dir, workspace_path, level)
    sandbox.workspace = workspace_path  # Use existing workspace
    return sandbox.collect_results()
