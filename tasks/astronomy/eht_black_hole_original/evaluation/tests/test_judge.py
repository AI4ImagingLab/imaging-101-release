import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


TASK_DIR = Path(__file__).resolve().parents[2]
JUDGE = TASK_DIR / "evaluation" / "judge.py"


def _judge(tmp_path, image=None, key="reconstruction", missing=False):
    prediction = tmp_path / "prediction.npz"
    if not missing:
        np.savez(prediction, **{key: image})
    report = tmp_path / "nested" / "report.json"
    process = subprocess.run(
        [sys.executable, str(JUDGE), "--prediction", str(prediction), "--report", str(report)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    return process.returncode, json.loads(report.read_text(encoding="utf-8"))


def test_reference_passes(tmp_path):
    with np.load(TASK_DIR / "evaluation" / "ground_truth.npz", allow_pickle=False) as data:
        image = data["reconstruction"]
    code, report = _judge(tmp_path, image)
    assert code == 0 and report["passed"]


def test_zero_image_fails(tmp_path):
    code, report = _judge(tmp_path, np.zeros((1, 64, 64), dtype=np.float64))
    assert code == 1 and report["status"] == "fail"


@pytest.mark.parametrize("image,key,missing", [
    (None, "reconstruction", True),
    (np.zeros((1, 64, 64)), "wrong", False),
    (np.zeros((64, 64)), "reconstruction", False),
    (np.zeros((1, 64, 64), dtype=np.float32), "reconstruction", False),
    (np.full((1, 64, 64), np.nan), "reconstruction", False),
    (np.full((1, 64, 64), np.inf), "reconstruction", False),
])
def test_invalid_submission_errors(tmp_path, image, key, missing):
    code, report = _judge(tmp_path, image, key, missing)
    assert code == 2 and report["status"] == "error"
    assert report["metrics"] == {} and report["errors"]
