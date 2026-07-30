"""Validate the task-aware scorer reproduces each task's metrics.json baseline.

For each supported task it scores the shipped reference reconstruction
(evaluation/reference_outputs/) and checks the computed metrics match
evaluation/metrics.json["baseline"][0].
"""
import json
from pathlib import Path

from evaluation_harness import task_scoring as ts

REPO = Path(__file__).resolve().parent.parent
TASKS = [
    "cars_spectroscopy",
    "ct_dual_energy",
    "eht_black_hole_feature_extraction_dynamic",
    "xray_ptychography_tike",
]

all_ok = True
for t in TASKS:
    print("=" * 78)
    print(f"TASK: {t}")
    res = ts.validate_against_baseline(t, REPO / "tasks" / t)
    if "error" in res:
        print(f"  ERROR: {res['error']}"); all_ok = False; continue
    print(f"  reproduces baseline: {res['match']}")
    for metric, d in res["diffs"].items():
        flag = "ok" if d["abs_diff"] <= max(abs(d["baseline"]) * 0.02, 0.05) else "MISMATCH"
        print(f"    {metric:<22} computed={d['computed']:<12} "
              f"baseline={d['baseline']:<12} |diff|={d['abs_diff']:<10} [{flag}]")
    # also show the live pass/fail grading on the reference reconstruction
    graded = ts.score_task(t, REPO / "tasks" / t,
                           REPO / "tasks" / t / "evaluation" / "reference_outputs")
    print(f"  graded: passed={graded.get('passed')}  boundaries={graded.get('boundaries')}")
    all_ok = all_ok and res["match"]

print("=" * 78)
print("ALL BASELINES REPRODUCED" if all_ok else "SOME MISMATCHES — see above")
