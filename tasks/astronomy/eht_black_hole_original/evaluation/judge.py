"""Validate and score a batch-first EHT image submission."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


EVALUATION_DIR = Path(__file__).resolve().parent
KEY = "reconstruction"
SHAPE = (1, 64, 64)
DTYPE = np.dtype("float64")


def _read_image(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"{label} file missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if KEY not in archive.files:
            raise ValueError(f"{label} missing required key: {KEY}")
        image = archive[KEY]
    if image.shape != SHAPE:
        raise ValueError(f"{label} {KEY} shape must be {SHAPE}, got {image.shape}")
    if image.dtype != DTYPE:
        raise ValueError(f"{label} {KEY} dtype must be {DTYPE}, got {image.dtype}")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{label} {KEY} contains NaN or infinity")
    return image


def evaluate(prediction):
    config = json.loads((EVALUATION_DIR / "metrics.json").read_text(encoding="utf-8"))
    thresholds = {
        "ncc": {"comparison": ">=", "boundary": config["ncc_boundary"]},
        "nrmse": {"comparison": "<=", "boundary": config["nrmse_boundary"]},
    }
    if not all(np.isfinite(item["boundary"]) for item in thresholds.values()):
        raise ValueError("Metric boundaries must be finite")
    reference_file = EVALUATION_DIR / "ground_truth.npz"
    if not reference_file.is_file():
        reference_file = EVALUATION_DIR / "baseline_reference.npz"
    reference = _read_image(reference_file, "reference")
    estimate = _read_image(prediction, "prediction")
    ref_norm = np.linalg.norm(reference.ravel())
    est_norm = np.linalg.norm(estimate.ravel())
    data_range = np.ptp(reference)
    if ref_norm == 0 or data_range == 0:
        raise ValueError("Reference is degenerate; task-specific metrics are required")
    ncc = float(np.dot(estimate.ravel(), reference.ravel()) / (est_norm * ref_norm)) if est_norm else 0.0
    nrmse = float(np.sqrt(np.mean((estimate - reference) ** 2)) / data_range)
    metrics = {"ncc": ncc, "nrmse": nrmse}
    passed = ncc >= thresholds["ncc"]["boundary"] and nrmse <= thresholds["nrmse"]["boundary"]
    return {"status": "pass" if passed else "fail", "passed": passed,
            "metrics": metrics, "thresholds": thresholds, "errors": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(args.prediction)
        code = 0 if report["passed"] else 1
    except Exception as exc:
        report = {"status": "error", "passed": False, "metrics": {},
                  "thresholds": {}, "errors": [f"{type(exc).__name__}: {exc}"]}
        code = 2
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"Could not write report: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
