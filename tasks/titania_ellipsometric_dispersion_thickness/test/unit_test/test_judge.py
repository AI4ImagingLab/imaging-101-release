"""Independent numeric cases and real standalone CLI contract regressions."""
from pathlib import Path
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
JUDGE = ROOT / "evaluation/judge.py"


@pytest.fixture(scope="module")
def judge():
    spec = importlib.util.spec_from_file_location("independent_optical_judge", JUDGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reference():
    with np.load(ROOT / "evaluation/baseline_reference.npz", allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


@pytest.fixture
def assets(tmp_path):
    task = tmp_path / "hidden_task"
    evaluation = task / "evaluation"
    evaluation.mkdir(parents=True)
    (task / "data").mkdir()
    for name in ("metrics.json", "baseline_reference.npz"):
        shutil.copy2(ROOT / "evaluation" / name, evaluation / name)
    shutil.copy2(ROOT / "data/raw_data.npz", task / "data/raw_data.npz")
    return evaluation


def save(path, output):
    np.savez_compressed(path, **output)
    return path


def strict_json(text):
    def reject_constant(value):
        raise AssertionError("Nonstandard JSON numeric constant: " + value)
    return json.loads(text, parse_constant=reject_constant)


def test_actual_fresh_submission_passes_all_seven_fields(judge):
    report = judge.evaluate(ROOT / "output/reconstruction.npz")
    assert report["status"] == "pass" and report["passed"]
    assert set(report["metrics"]) == set(judge.METRICS)
    assert set(report["thresholds"]) == set(judge.METRICS)
    assert all(item["comparison"] == "<=" for item in report["thresholds"].values())
    assert np.isfinite(list(report["metrics"].values())).all()


def test_independent_judge_full_forward_matches_frozen_original(judge, reference):
    raw = judge.read_observations(ROOT / "data/raw_data.npz")
    with np.load(ROOT / "test/fixtures/output_upstream_fit.npz", allow_pickle=False) as source:
        rho = source["rho_fitted"]
    actual = judge.forward_rho(reference["parameters"][0], raw)
    np.testing.assert_allclose(actual, rho, rtol=1e-12, atol=1e-13)


def test_known_metric_normalization_and_source_coefficient_coordinates(judge, reference):
    raw = judge.read_observations(ROOT / "data/raw_data.npz")
    candidate = {k: v.copy() for k, v in reference.items()}
    candidate["parameters"][0] += [0.01, 2.0, 3.0, 0.1]
    candidate["reconstruction"] += 1e-5
    values = judge.measure(candidate, reference, raw)
    np.testing.assert_allclose([values[n] for n in judge.PARAMETER_METRICS], [.01, 2, 3, .1], rtol=0, atol=1e-13)
    expected = 1e-5 / np.ptp(reference["reconstruction"])
    assert values["nrmse"] == pytest.approx(expected, rel=1e-9)
    # Independently stated source numerical convention, not bare B/C values.
    wavelength = raw["wavelength_nm"][0]
    parameters = reference["parameters"][0]
    expected_index = parameters[0] + (100 * parameters[1]) / wavelength**2 + (1e7 * parameters[2]) / wavelength**4
    np.testing.assert_allclose(judge.refractive_index(parameters, wavelength), expected_index, rtol=0, atol=0)


@pytest.mark.parametrize("parameter", range(4))
def test_each_unknown_has_its_own_decisive_boundary(judge, reference, tmp_path, parameter):
    limits = judge.read_limits(ROOT / "evaluation/metrics.json")
    key = judge.PARAMETER_METRICS[parameter]
    reference["parameters"][0, parameter] += 2 * limits[key]
    raw = judge.read_observations(ROOT / "data/raw_data.npz")
    reference["reconstruction"][0] = judge.refractive_index(reference["parameters"][0], raw["wavelength_nm"][0])
    report = judge.evaluate(save(tmp_path / "one_parameter.npz", reference))
    assert report["status"] == "fail" and not report["passed"]
    assert report["metrics"][key] > report["thresholds"][key]["boundary"]


def test_index_self_consistency_is_independent_of_reference_agreement(judge, reference, tmp_path):
    reference["reconstruction"] += 5e-10
    report = judge.evaluate(save(tmp_path / "inconsistent.npz", reference))
    assert report["status"] == "fail"
    assert report["metrics"]["nrmse"] < report["thresholds"]["nrmse"]["boundary"]
    assert report["metrics"]["index_consistency_absolute_error"] > 1e-10


def test_observation_forward_gate_is_independent_of_reference_identity(judge, assets, tmp_path):
    raw = judge.read_observations(ROOT / "data/raw_data.npz")
    parameters = np.array([[1.8, 50., 10., 60.]])
    poor = dict(parameters=parameters, reconstruction=judge.refractive_index(parameters[0], raw["wavelength_nm"][0])[None])
    save(assets / "baseline_reference.npz", poor)
    report = judge.evaluate(save(tmp_path / "poor_reference_identical.npz", poor), assets)
    assert report["status"] == "fail"
    assert report["metrics"]["rho_relative_rms"] > .02
    assert all(report["metrics"][k] == 0 for k in judge.METRICS if k != "rho_relative_rms")


@pytest.mark.parametrize("mode", [
    "missing_key", "extra_key", "parameter_shape", "index_shape", "float32", "integer",
    "complex_parameters", "complex_index", "nan_parameters", "inf_parameters",
    "nan_index", "inf_index", "object", "nonpositive_thickness", "nonpositive_index",
])
def test_malformed_submission_has_empty_metrics(judge, reference, tmp_path, mode):
    if mode == "missing_key": reference.pop("parameters")
    elif mode == "extra_key": reference["extra"] = np.array([1.])
    elif mode == "parameter_shape": reference["parameters"] = reference["parameters"][0]
    elif mode == "index_shape": reference["reconstruction"] = reference["reconstruction"].T
    elif mode == "float32": reference["parameters"] = reference["parameters"].astype(np.float32)
    elif mode == "integer": reference["reconstruction"] = reference["reconstruction"].astype(np.int64)
    elif mode == "complex_parameters": reference["parameters"] = reference["parameters"].astype(np.complex128)
    elif mode == "complex_index": reference["reconstruction"] = reference["reconstruction"].astype(np.complex128)
    elif mode == "nan_parameters": reference["parameters"][0, 0] = np.nan
    elif mode == "inf_parameters": reference["parameters"][0, 3] = np.inf
    elif mode == "nan_index": reference["reconstruction"][0, 17] = np.nan
    elif mode == "inf_index": reference["reconstruction"][0, 11] = -np.inf
    elif mode == "object": reference["parameters"] = reference["parameters"].astype(object)
    elif mode == "nonpositive_thickness": reference["parameters"][0, 3] = 0
    elif mode == "nonpositive_index": reference["reconstruction"][0, 0] = 0
    report = judge.evaluate(save(tmp_path / "malformed.npz", reference))
    assert report["status"] == "error" and not report["passed"]
    assert report["metrics"] == {} and report["errors"]
    strict_json(json.dumps(report, allow_nan=False))


@pytest.mark.parametrize("value", [None, True, "1", 0., -1., np.nan, np.inf, -np.inf])
def test_undefined_metric_configuration_is_an_evaluator_error(judge, assets, reference, tmp_path, value):
    path = assets / "metrics.json"
    config = json.loads(path.read_text())
    config["nrmse_boundary"] = value
    path.write_text(json.dumps(config))
    report = judge.evaluate(save(tmp_path / "valid.npz", reference), assets)
    assert report["status"] == "error" and report["metrics"] == {}


def test_real_truth_precedes_method_baseline(judge, reference, assets, tmp_path):
    raw = judge.read_observations(ROOT / "data/raw_data.npz")
    prediction = save(tmp_path / "original.npz", reference)
    reference["parameters"][0, 0] += .001
    reference["reconstruction"][0] = judge.refractive_index(reference["parameters"][0], raw["wavelength_nm"][0])
    save(assets / "ground_truth.npz", reference)
    report = judge.evaluate(prediction, assets)
    assert report["status"] == "fail"
    assert report["metrics"]["n0_absolute_error"] == pytest.approx(.001)


@pytest.mark.parametrize("kind", ["invalid_file", "directory", "dangling_symlink"])
def test_invalid_available_truth_never_falls_back(judge, reference, assets, tmp_path, kind):
    truth = assets / "ground_truth.npz"
    if kind == "invalid_file": truth.write_bytes(b"not an npz file")
    elif kind == "directory": truth.mkdir()
    else:
        target = assets / "unavailable_actual_truth.npz"
        try:
            truth.symlink_to(target)
        except OSError:
            if os.name != "nt":
                raise
            # Windows file symlinks need a privilege that may be unavailable.
            # A native dangling junction still occupies the truth directory
            # entry and must receive the same strict no-fallback treatment.
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(truth), str(target)], capture_output=True)
            assert result.returncode == 0, result.stderr
        assert os.path.lexists(truth) and not truth.exists()
    report = judge.evaluate(save(tmp_path / "valid.npz", reference), assets)
    assert report["status"] == "error" and report["metrics"] == {}
    assert report["errors"]
    if kind == "dangling_symlink":
        shutil.copy2(JUDGE, assets / "judge.py")
        report_path = tmp_path / "broken_truth_report.json"
        result = subprocess.run([sys.executable, "-X", "utf8", str(assets / "judge.py"), "--prediction", str(tmp_path / "valid.npz"), "--report", str(report_path)], cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
        actual = strict_json(report_path.read_text())
        assert result.returncode == 2 and actual["status"] == "error" and actual["metrics"] == {}


@pytest.mark.parametrize("asset", ["metrics.json", "baseline_reference.npz", "raw_data.npz"])
def test_missing_hidden_or_public_evaluator_asset(judge, reference, assets, tmp_path, asset):
    path = assets.parent / "data/raw_data.npz" if asset == "raw_data.npz" else assets / asset
    path.unlink()
    report = judge.evaluate(save(tmp_path / "valid.npz", reference), assets)
    assert report["status"] == "error" and report["metrics"] == {}


def test_undefined_zero_range_reference_rejected(judge, assets, reference, tmp_path):
    reference["parameters"][0, :3] = [2., 0., 0.]
    reference["reconstruction"][:] = 2.
    save(assets / "baseline_reference.npz", reference)
    report = judge.evaluate(save(tmp_path / "valid.npz", reference), assets)
    assert report["status"] == "error" and report["metrics"] == {}
    assert "dynamic range" in report["errors"][0]


def test_standalone_foreign_cwd_pass_and_missing_prediction(tmp_path, reference):
    prediction = save(tmp_path / "valid.npz", reference)
    report_path = tmp_path / "nested/report.json"
    # A submitted source package that raises must never be imported by Judge.
    (tmp_path / "src").mkdir()
    (tmp_path / "src/__init__.py").write_text("raise RuntimeError('submitted solver executed')")
    command = [sys.executable, "-X", "utf8", str(JUDGE), "--prediction", str(prediction), "--report", str(report_path)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert strict_json(report_path.read_text())["status"] == "pass"
    prediction.unlink()
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    report = strict_json(report_path.read_text())
    assert result.returncode == 2 and report["status"] == "error" and report["metrics"] == {}


@pytest.mark.parametrize("field", ["parameters", "reconstruction"])
def test_actual_cli_extreme_finite_values_write_valid_error_json(tmp_path, reference, field):
    if field == "parameters": reference[field][0, 1] = 1e308
    else: reference[field][:] = 1e308
    prediction = save(tmp_path / "extreme.npz", reference)
    report_path = tmp_path / "extreme_report.json"
    result = subprocess.run([sys.executable, "-X", "utf8", str(JUDGE), "--prediction", str(prediction), "--report", str(report_path)], cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    report = strict_json(report_path.read_text())
    assert result.returncode == 2 and report["status"] == "error"
    assert report["metrics"] == {} and report["errors"] and not report["passed"]


def test_actual_cli_valid_poor_output_returns_one(tmp_path, reference):
    reference["parameters"][0, 3] += 1.
    prediction = save(tmp_path / "poor.npz", reference)
    report_path = tmp_path / "poor_report.json"
    result = subprocess.run([sys.executable, "-X", "utf8", str(JUDGE), "--prediction", str(prediction), "--report", str(report_path)], cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    report = strict_json(report_path.read_text())
    assert result.returncode == 1 and report["status"] == "fail" and not report["passed"]
    assert len(report["metrics"]) == 7


@pytest.mark.parametrize("kind", ["directory", "file_parent"])
def test_actual_cli_report_write_failure_returns_two(tmp_path, reference, kind):
    prediction = save(tmp_path / "valid.npz", reference)
    blocker = tmp_path / "blocker"
    if kind == "directory": blocker.mkdir(); report_path = blocker
    else: blocker.write_text("regular file"); report_path = blocker / "report.json"
    result = subprocess.run([sys.executable, "-X", "utf8", str(JUDGE), "--prediction", str(prediction), "--report", str(report_path)], cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 2 and "Report-write failure" in result.stderr
