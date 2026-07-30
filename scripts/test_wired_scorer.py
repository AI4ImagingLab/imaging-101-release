"""Exercise the wired Scorer._compute_quality_metrics on all 5 tasks.

Seeds a sandbox workspace's output/ with each task's shipped reference
reconstruction, then calls the REAL Scorer (the harness end-to-end scoring
path). Confirms:
  - the 4 registered tasks route to the task-aware scorer and reproduce baseline
  - eht_black_hole_UQ (unregistered) falls back to the generic 2-D scorer
"""
import shutil, tempfile
from pathlib import Path

from evaluation_harness.config import RunConfig, TaskConfig, LLMConfig
from evaluation_harness.local_runner import LocalRunner
from evaluation_harness.scorer import Scorer

REPO = Path("/work/hdd/bghj/schen47/inverse-101")
HARNESS_VENV_BIN = str(REPO / ".venv_harness" / "bin")  # numpy available here

# task -> list of (reference file in evaluation/reference_outputs, dest name in output/)
SEED = {
    "cars_spectroscopy": [("reconstruction.npz", "reconstruction.npz")],
    "ct_dual_energy": [("reference_reconstruction.npz", "reconstruction.npz")],
    "eht_black_hole_feature_extraction_dynamic": [
        ("all_params.npy", "all_params.npy"),
        ("all_weights.npy", "all_weights.npy"),
    ],
    "xray_ptychography_tike": [("reconstructed_object.npy", "reconstructed_object.npy")],
    # unregistered -> generic scorer; generic needs a 2-D output/reconstruction.npy
    "eht_black_hole_UQ": [("ground_truth.npy", "reconstruction.npy")],
}


def run(task):
    task_dir = REPO / "tasks" / task
    ws = Path(tempfile.mkdtemp(prefix=f"wired-{task}-"))
    (ws / "output").mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in SEED[task]:
        shutil.copy2(task_dir / "evaluation" / "reference_outputs" / src_name,
                     ws / "output" / dst_name)

    runner = LocalRunner(image="x", task_dir=task_dir, timeout=300)
    runner.workspace = ws
    runner.container = str(ws)
    runner._venv_bin = HARNESS_VENV_BIN

    cfg = RunConfig(
        llm=LLMConfig(model="Vendor3/DeepSeek-V4-Pro"),
        task=TaskConfig(task_name=task, task_dir=task_dir, mode="end_to_end", level="L1"),
    )
    metrics = Scorer(runner, cfg)._compute_quality_metrics()
    shutil.rmtree(ws, ignore_errors=True)
    return metrics


print(f"{'TASK':<45}{'SCORER':<11} METRICS")
print("-" * 110)
for t in ["cars_spectroscopy", "ct_dual_energy",
          "eht_black_hole_feature_extraction_dynamic",
          "xray_ptychography_tike", "eht_black_hole_UQ"]:
    m = run(t)
    scorer = (m or {}).get("scorer", "?")
    body = {k: v for k, v in (m or {}).items() if k != "scorer"}
    print(f"{t:<45}{scorer:<11} {body}")
