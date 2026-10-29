"""Standalone evaluator for the declared four-parameter optical inverse.

Uses only NumPy and the standard library. No solver, upstream library, submitted
code, or checkpoint is imported. The Fresnel recursion is independent of the
reference implementation's transfer-matrix multiplication.
"""
from pathlib import Path
import argparse
import json
import os
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT_SHAPES = {"parameters": (1, 4), "reconstruction": (1, 926)}
PARAMETER_METRICS = (
    "n0_absolute_error", "n1_absolute_error", "n2_absolute_error",
    "thickness_nm_absolute_error",
)
METRICS = PARAMETER_METRICS + (
    "nrmse", "index_consistency_absolute_error", "rho_relative_rms",
)


def _archive(path, shapes, complex_keys=()):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Expected an existing regular NPZ file: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("Expected an NPZ archive")
        if len(archive.files) != len(shapes) or set(archive.files) != set(shapes):
            raise ValueError("NPZ must contain exactly the declared keys: " + ", ".join(shapes))
        result = {}
        for key, shape in shapes.items():
            value = archive[key]
            if value.shape != shape:
                raise ValueError(f"Invalid shape for {key}: expected {shape}, got {value.shape}")
            kind, size = ("c", 16) if key in complex_keys else ("f", 8)
            if value.dtype.kind != kind or value.dtype.itemsize != size:
                raise ValueError(f"{key} requires {'complex128' if kind == 'c' else 'real float64'}")
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite values in {key}")
            result[key] = value
    return result


def read_output(path):
    result = _archive(path, OUTPUT_SHAPES)
    if result["parameters"][0, 3] <= 0:
        raise ValueError("The submitted film thickness must be positive, in nm")
    if np.any(result["reconstruction"] <= 0):
        raise ValueError("The submitted refractive index must be positive")
    return result


def read_observations(path):
    shapes = {key: (1, 926) for key in (
        "wavelength_nm", "psi_deg", "delta_deg", "silicon_index",
    )}
    raw = _archive(path, shapes, complex_keys=("silicon_index",))
    wavelength = raw["wavelength_nm"][0]
    if np.any(np.diff(wavelength) <= 0) or wavelength[0] < 400 or wavelength[-1] > 800:
        raise ValueError("Invalid complete ordered author 400--800 nm acquisition")
    if np.any(raw["psi_deg"] < 0) or np.any(raw["psi_deg"] > 90):
        raise ValueError("Invalid observed Psi in degrees")
    if np.any(raw["silicon_index"].real <= 0) or np.any(raw["silicon_index"].imag < 0):
        raise ValueError("Invalid passive n+i*k silicon calibration")
    return raw


def read_limits(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    limits = {}
    for name in METRICS:
        value = config.get(name + "_boundary")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Undefined numeric boundary: " + name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Boundary must be finite and positive: " + name)
        limits[name] = float(value)
    return limits


def refractive_index(parameters, wavelength_nm):
    """Exact declared source-scaled transparent Cauchy formula."""
    n0, n1, n2, _ = parameters
    return n0 + 100.0 * n1 / wavelength_nm**2 + 1.0e7 * n2 / wavelength_nm**4


def forward_rho(parameters, raw):
    """Coherent Fresnel recursion: air / titania / fixed silica / silicon.

    All lengths are nm. Passive optical index is n+i*k; the independently
    composed reflection recursion uses exp(+2i*kz*d). The model angle is the
    author's fixed nominal 70 degrees, distinct from measured metadata70.06.
    """
    wavelength = raw["wavelength_nm"][0]
    titania = refractive_index(parameters, wavelength)
    if np.any(titania <= 0):
        raise ValueError("Parameters imply a nonpositive optical index")
    silica = 1.452 + 3600.0 / wavelength**2
    indices = np.asarray([
        np.ones_like(wavelength), titania, silica, raw["silicon_index"][0],
    ], dtype=np.complex128)
    transverse = np.sin(np.deg2rad(70.0))
    kz = np.sqrt(indices**2 - transverse**2 + 0j)
    reverse = (kz.imag < 0) | ((np.abs(kz.imag) < 1e-15) & (kz.real < 0))
    kz = np.where(reverse, -kz, kz)
    rs = (kz[:-1] - kz[1:]) / (kz[:-1] + kz[1:])
    epsilon = indices**2
    rp = (epsilon[1:] * kz[:-1] - epsilon[:-1] * kz[1:]) / (
        epsilon[1:] * kz[:-1] + epsilon[:-1] * kz[1:]
    )
    silica_phase = np.exp(4j * np.pi * kz[2] * 276.36 / wavelength)
    titania_phase = np.exp(4j * np.pi * kz[1] * parameters[3] / wavelength)

    def reflection(coefficients):
        rear = (coefficients[1] + coefficients[2] * silica_phase) / (
            1 + coefficients[1] * coefficients[2] * silica_phase
        )
        return (coefficients[0] + rear * titania_phase) / (
            1 + coefficients[0] * rear * titania_phase
        )

    result = reflection(rp) / reflection(rs)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite independent Fresnel prediction")
    return result


def measure(prediction, reference, raw):
    """All four unknowns, complete index field, consistency and observation fit."""
    parameters = prediction["parameters"][0]
    delta = np.abs(parameters - reference["parameters"][0])
    metrics = {name: float(delta[i]) for i, name in enumerate(PARAMETER_METRICS)}
    reference_index = reference["reconstruction"][0]
    dynamic_range = float(np.ptp(reference_index))
    if not np.isfinite(dynamic_range) or dynamic_range <= 0:
        raise ValueError("Reference index has undefined zero/nonfinite dynamic range")
    field_error = prediction["reconstruction"][0] - reference_index
    metrics["nrmse"] = float(np.sqrt(np.mean(field_error**2)) / dynamic_range)
    calculated_index = refractive_index(parameters, raw["wavelength_nm"][0])
    metrics["index_consistency_absolute_error"] = float(np.max(np.abs(
        prediction["reconstruction"][0] - calculated_index
    )))
    observed = np.tan(np.deg2rad(raw["psi_deg"][0])) * np.exp(
        -1j * np.deg2rad(raw["delta_deg"][0])
    )
    denominator = float(np.linalg.norm(observed))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Undefined observed complex-rho norm")
    metrics["rho_relative_rms"] = float(
        np.linalg.norm(forward_rho(parameters, raw) - observed) / denominator
    )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite derived evaluation metric")
    return metrics


def evaluate(prediction_path, evaluation_directory=HERE):
    report = dict(status="error", passed=False, metrics={}, thresholds={}, errors=[])
    try:
        directory = Path(evaluation_directory)
        limits = read_limits(directory / "metrics.json")
        report["thresholds"] = {
            name: dict(comparison="<=", boundary=value) for name, value in limits.items()
        }
        truth = directory / "ground_truth.npz"
        # A directory or dangling symlink is an available invalid truth asset,
        # never permission to silently fall back to the method reference.
        reference_path = truth if os.path.lexists(truth) else directory / "baseline_reference.npz"
        reference = read_output(reference_path)
        raw = read_observations(directory.parent / "data/raw_data.npz")
        prediction = read_output(prediction_path)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            reference_consistency = np.max(np.abs(
                reference["reconstruction"][0] - refractive_index(
                    reference["parameters"][0], raw["wavelength_nm"][0]
                )
            ))
            if not np.isfinite(reference_consistency) or reference_consistency > limits["index_consistency_absolute_error"]:
                raise ValueError("Reference violates the declared Cauchy representation")
            values = measure(prediction, reference, raw)
        passed = all(values[name] <= limits[name] for name in METRICS)
        report.update(status="pass" if passed else "fail", passed=bool(passed), metrics=values)
    except Exception as error:
        report.update(status="error", passed=False, metrics={}, errors=[type(error).__name__ + ": " + str(error)])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args(argv)
    report = evaluate(arguments.prediction)
    try:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception as error:
        print("Report-write failure: " + str(error), file=sys.stderr)
        return 2
    return {"pass": 0, "fail": 1, "error": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
