"""Demonstrate the end2end scorer's ground-truth handling on two tasks.

Exercises the REAL Scorer._compute_quality_metrics() code path (same function
the harness calls at the end of an end2end run), with a synthetic
output/reconstruction.npy so we don't have to wait for the LLM agent loop.

  correct task: eht_black_hole_UQ  -> has evaluation/reference_outputs/ground_truth.npy
  wrong  task:  cars_spectroscopy  -> ships only data/ground_truth.npz
"""
import logging, tempfile, shutil
from pathlib import Path
import numpy as np

from evaluation_harness.config import RunConfig, TaskConfig, LLMConfig
from evaluation_harness.local_runner import LocalRunner
from evaluation_harness.scorer import Scorer
from evaluation_harness.agent import AgentResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

REPO = Path("/work/hdd/bghj/schen47/inverse-101")
HARNESS_VENV_BIN = str(REPO / ".venv_harness" / "bin")  # python here has numpy


def score_task(task_name: str, make_recon):
    print("\n" + "=" * 70)
    print(f"TASK: {task_name}")
    print("=" * 70)
    task_dir = REPO / "tasks" / task_name

    # show what ground-truth files actually exist for this task
    npy1 = task_dir / "evaluation" / "reference_outputs" / "ground_truth.npy"
    npy2 = task_dir / "data" / "ground_truth.npy"
    print(f"  scorer looks for: evaluation/reference_outputs/ground_truth.npy -> exists={npy1.exists()}")
    print(f"  scorer looks for: data/ground_truth.npy                        -> exists={npy2.exists()}")
    npzs = list((task_dir / "data").glob("*.npz"))
    print(f"  what the task actually ships in data/: {[p.name for p in npzs]}")

    # Build a runner whose workspace we populate by hand (skip the heavy venv install).
    runner = LocalRunner(image="x", task_dir=task_dir, timeout=300)
    runner.workspace = Path(tempfile.mkdtemp(prefix=f"demo-{task_name}-"))
    runner.container = str(runner.workspace)
    runner._venv_bin = HARNESS_VENV_BIN  # so the metric snippet's `python` has numpy

    # Write a synthetic output/reconstruction.npy (what the agent would produce)
    out_dir = runner.workspace / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "reconstruction.npy", make_recon())

    cfg = RunConfig(
        llm=LLMConfig(model="Vendor3/DeepSeek-V4-Pro"),
        task=TaskConfig(task_name=task_name, task_dir=task_dir, mode="end_to_end", level="L1"),
    )
    scorer = Scorer(runner, cfg, llm_client=None)
    print("  --- calling Scorer._compute_quality_metrics() ---")
    metrics = scorer._compute_quality_metrics()
    print(f"  >>> quality_metrics = {metrics}")
    shutil.rmtree(runner.workspace, ignore_errors=True)
    return metrics


# CORRECT task: produce a recon matching the gt shape (perfect copy -> NRMSE~0, NCC~1)
def recon_uq():
    gt = np.load(REPO / "tasks/eht_black_hole_UQ/evaluation/reference_outputs/ground_truth.npy")
    return gt.astype(np.float64) + 0.01 * np.abs(gt).max() * np.random.RandomState(0).randn(*gt.shape)

# WRONG task: any 2D array; scorer errors out before it is even loaded
def recon_cars():
    return np.linspace(0, 1, 200).reshape(10, 20)


if __name__ == "__main__":
    m_ok = score_task("eht_black_hole_UQ", recon_uq)
    m_bad = score_task("cars_spectroscopy", recon_cars)
    print("\n" + "#" * 70)
    print("SUMMARY")
    print(f"  eht_black_hole_UQ (.npy present): {m_ok}")
    print(f"  cars_spectroscopy (.npz only):    {m_bad}")
    print("#" * 70)
