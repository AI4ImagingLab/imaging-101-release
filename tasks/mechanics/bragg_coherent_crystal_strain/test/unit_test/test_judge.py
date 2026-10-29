"""Independently known metric and standalone malformed-submission contracts."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

TASK = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("independent_bragg_judge", TASK/"evaluation/judge.py")
judge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(judge)


def test_fixed_reference_phase_mask_and_zero_phase_penalty():
    reference = np.ones((7, 8, 9), dtype=complex)
    candidate = reference.copy()
    candidate[:3] *= np.exp(.2j)
    phase, strain = judge.phase_strain_errors(candidate, reference)
    assert phase == pytest.approx(.2*np.sqrt(3/7))
    assert strain > 0.
    candidate[0] = 0.
    phase, _ = judge.phase_strain_errors(candidate, reference)
    assert phase == pytest.approx(np.sqrt((np.pi**2+2*.2**2)/7))
    phase, strain = judge.phase_strain_errors(np.zeros_like(reference), reference)
    assert phase == pytest.approx(np.pi)
    # A constant zero field has zero phase gradient; complex/phase metrics reject it.
    assert strain == 0.


def test_independent_quadratic_phase_strain_sign_scale_and_mean():
    shape = (7, 8, 9)
    coordinates = np.indices(shape)
    phase = .001*coordinates[0]**2-.0007*coordinates[1]**2+.0003*coordinates[2]**2
    obj = np.exp(1j*phase)
    _, interior = judge.reference_masks(obj)
    # Reciprocal/direct duality gives inv(D).T = B diag(N)/(2 pi).
    delta, gamma, step = np.deg2rad([28.488, .0478, .002])
    c = 2*np.pi*9/1.2398419843320026
    q = c*np.array([np.cos(gamma)*np.sin(delta), np.sin(gamma),
                    np.cos(delta)*np.cos(gamma)-1])
    b = c*np.column_stack((step*np.array([1-np.cos(delta)*np.cos(gamma), 0,
                                          np.sin(delta)*np.cos(gamma)]),
                           55000/2.8e9*np.array([np.sin(delta)*np.sin(gamma),
                                               -np.cos(gamma), np.cos(delta)*np.sin(gamma)]),
                           -55000/2.8e9*np.array([-np.cos(delta), 0, np.sin(delta)])))
    coefficients = (q@b)*np.asarray(shape)/(2*np.pi*np.dot(q,q))
    expected = sum(coefficients[j]*factor*coordinates[j][interior]
                   for j, factor in enumerate((.002, -.0014, .0006)))
    expected -= expected.mean()
    actual = judge.relative_projected_strain(obj, interior)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-18)
    assert np.linalg.norm(actual) > 0.
    _, error = judge.phase_strain_errors(np.conj(obj), obj)
    assert error == pytest.approx(2*np.sqrt(np.mean(expected**2)), rel=1e-12)


def test_phase_strain_legal_gauges_and_invalid_reference_support():
    rng = np.random.default_rng(192)
    reference = (1+.1*rng.random((7, 8, 9)))*np.exp(.03j*rng.normal(size=(7, 8, 9)))
    candidate = np.roll(np.conj(np.flip(reference)), (1, 2, -3), axis=(0,1,2))*np.exp(.9j)
    aligned = judge.align_to_reference(reference, candidate)
    phase, strain = judge.phase_strain_errors(aligned, reference)
    assert phase < 1e-14
    assert strain < 1e-16
    sparse = np.zeros_like(reference)
    sparse[3,4,4] = 1.
    with pytest.raises(ValueError, match="eroded"):
        judge.reference_masks(sparse)


@pytest.mark.parametrize("twin", [False, True])
def test_declared_complex_symmetries(twin):
    rng = np.random.default_rng(724)
    reference = rng.normal(size=(5, 6, 7)) + 1j*rng.normal(size=(5, 6, 7))
    candidate = np.conj(np.flip(reference)) if twin else reference.copy()
    candidate = np.roll(candidate, (2, -1, 3), axis=(0, 1, 2))*np.exp(.73j)
    aligned = judge.align_to_reference(reference, candidate)
    assert judge.relative_l2(aligned, reference) < 1e-14
    assert judge.relative_l2(.8*aligned, reference) == pytest.approx(.2)


def test_known_count_scale_residual():
    obj = np.zeros((5, 6, 7), dtype=complex)
    obj[2, 3, 3] = 2.
    counts = np.full(obj.shape, 4, dtype=np.int32)
    assert judge.measured_consistency(obj, np.ones((1, 1, 1)), counts) < 1e-28
    assert judge.measured_consistency(.5*obj, np.ones((1, 1, 1)), counts) == pytest.approx(.25)
    with pytest.raises(ValueError, match="zero squared norm"):
        judge.relative_l2(obj, np.zeros_like(obj))


def test_raw_consistency_preserves_every_original_plane_after_padding():
    from scipy import fft
    counts = np.arange(1,61,dtype=np.int32).reshape(3,4,5)
    padding = ((0,1),(1,1),(1,2))
    embedded = np.pad(counts,padding)
    peak = np.asarray(np.unravel_index(embedded.argmax(),embedded.shape))
    shift = np.asarray(embedded.shape)//2-peak
    wave = np.sqrt(np.roll(embedded,shift,axis=(0,1,2)).astype(np.float64))
    obj = fft.fftshift(fft.fftn(fft.ifftshift(wave),norm='forward'))
    assert judge.measured_consistency(obj,np.ones((1,1,1)),counts) < 1e-14
    changed = counts.copy()
    changed[0,0,0] += 1
    assert judge.measured_consistency(obj,np.ones((1,1,1)),changed) > 1e-5


def threshold_case():
    rules = {}
    for position in (1, 2):
        for name in judge.METRIC_NAMES:
            boundary = .9 if name == "complex_cosine" else .1
            if name == "kernel_mass_error":
                boundary = 1e-10
            if name == "kernel_negative_mass":
                boundary = 0.
            rules[f"p{position}_{name}"] = {
                "comparison": ">=" if name == "complex_cosine" else "<=", "boundary": boundary}
    return rules


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -1., 1.])
def test_invalid_threshold_boundaries_rejected(bad):
    rules = threshold_case()
    rules["p1_raw_amplitude_residual"]["boundary"] = bad
    with pytest.raises(ValueError):
        judge.validate_thresholds(rules)


def test_invalid_threshold_direction_and_names():
    rules = threshold_case()
    judge.validate_thresholds(rules)
    rules["p1_raw_amplitude_residual"]["comparison"] = ">="
    with pytest.raises(ValueError, match="direction"):
        judge.validate_thresholds(rules)
    with pytest.raises(ValueError, match="names"):
        judge.validate_thresholds({})


def metric_specification_case():
    rules = threshold_case()
    return {"thresholds": rules,
            **{f"{name}_boundary": rule["boundary"] for name, rule in rules.items()}}


def test_flat_metric_boundaries_match_every_nested_rule():
    specification = metric_specification_case()
    assert judge.validate_metric_specification(specification) == specification["thresholds"]


@pytest.mark.parametrize("case", ["mismatch", "missing", "extra", "bool", "nan", "inf"])
def test_conflicting_or_invalid_flat_boundaries_rejected(case):
    specification = metric_specification_case()
    key = "p1_raw_amplitude_residual_boundary"
    if case == "mismatch":
        specification[key] = .2
    elif case == "missing":
        del specification[key]
    elif case == "extra":
        specification["unknown_metric_boundary"] = .2
    elif case == "bool":
        specification["p1_kernel_negative_mass_boundary"] = False
    else:
        specification[key] = float(case)
    with pytest.raises(ValueError, match="[Ff]lat"):
        judge.validate_metric_specification(specification)


def test_metric_specification_requires_explicit_rules():
    with pytest.raises(ValueError, match="specification"):
        judge.validate_metric_specification([])


@pytest.mark.parametrize("bad", ["zero_density", "zero_kernel", "negative_kernel", "negative_counts", "zero_counts"])
def test_degenerate_hidden_assets_rejected(bad):
    reference = {"reconstruction": np.ones((1, 2, 3, 4, 5), dtype=complex),
                 "coherence": np.ones((1, 2, 1, 1, 1))}
    counts = np.ones((2, 3, 4, 5), dtype=np.int32)
    if bad == "zero_density":
        reference["reconstruction"][:] = 0.
    elif bad == "zero_kernel":
        reference["coherence"][:] = 0.
    elif bad == "negative_kernel":
        reference["coherence"][:] = -1.
    elif bad == "negative_counts":
        counts[0,0,0,0] = -1
    else:
        counts[:] = 0
    with pytest.raises(ValueError):
        judge.validate_reference(reference, counts)


@pytest.mark.parametrize("case", ["missing", "key", "shape", "dtype", "nan", "inf"])
def test_invalid_submissions_report_error(tmp_path, case):
    prediction = tmp_path/"prediction.npz"
    if case == "key":
        np.savez_compressed(prediction, wrong=np.zeros(1))
    elif case == "shape":
        np.savez_compressed(prediction, reconstruction=np.zeros((1,), dtype=np.complex128))
    elif case == "dtype":
        np.savez_compressed(prediction, reconstruction=np.zeros((1,2,126,256,256), dtype=np.float32))
    elif case in ("nan", "inf"):
        obj = np.zeros((1,2,126,256,256), dtype=np.complex128)
        obj[0,0,0,0,0] = np.nan if case == "nan" else np.inf
        np.savez_compressed(prediction, reconstruction=obj)
    report = tmp_path/"nested/report.json"
    process = subprocess.run([sys.executable, str(TASK/"evaluation/judge.py"),
                              "--prediction", str(prediction), "--report", str(report)],
                             cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 2
    result = json.loads(report.read_text(encoding="utf8"))
    assert result["status"] == "error" and result["passed"] is False
    assert result["metrics"] == {} and result["errors"]


def test_missing_hidden_reference_is_evaluator_error(tmp_path, monkeypatch):
    contract = {"reconstruction": ((1,2,3,4,5), np.dtype("complex128")),
                "coherence": ((1,2,1,1,1), np.dtype("float64"))}
    prediction = tmp_path/"prediction.npz"
    np.savez_compressed(prediction, reconstruction=np.ones((1,2,3,4,5), dtype=np.complex128),
                        coherence=np.ones((1,2,1,1,1), dtype=np.float64))
    monkeypatch.setattr(judge, "CONTRACT", contract)
    monkeypatch.setattr(judge, "HERE", tmp_path)
    result, code = judge.evaluate(prediction)
    assert code == 2 and result["status"] == "error"
    assert result["metrics"] == {} and result["passed"] is False
    assert "baseline_reference.npz" in result["errors"][0]


def test_report_write_failure_returns_two(tmp_path, capsys):
    report_directory = tmp_path/"report.json"
    report_directory.mkdir()
    code = judge.main(["--prediction", str(tmp_path/"absent.npz"),
                       "--report", str(report_directory)])
    assert code == 2
    assert "Cannot write Judge report:" in capsys.readouterr().err
