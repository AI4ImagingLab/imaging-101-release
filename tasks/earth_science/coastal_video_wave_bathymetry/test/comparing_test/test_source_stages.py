"""Independent original fixtures and analytic physical checks; no upstream imports."""
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import brentq
from src.preprocessing import load_data, recordings, window_specs, G
from src.physics_model import (
    analytic_signal, dmd, kappa_to_mu, dispersion_wavenumber,
    gamma_from_bed, ransac_trial_count,
)
from src.solvers import fit_phase_plane

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

def test_original_hilbert_and_all_native_dmd_windows():
    count = {"oblique": 0, "planview": 0}
    with np.load(FIXTURES / "upstream_stages.npz", allow_pickle=False) as f:
        for record in recordings(load_data()):
            cached = {}
            for spec in window_specs(record):
                start, nh = spec["start"], spec["nh"]
                tag = f"{record['label']}_{start}_{int(spec['seconds'])}"
                if start not in cached:
                    x = record["intensity"][:, start:start + spec["nmax"] + 2 * nh]
                    cached[start] = analytic_signal(x)
                    rows = f[f"input_probe_rows_{record['label']}_{start}"]
                    np.testing.assert_allclose(
                        cached[start][rows], f[f"output_analytic_{record['label']}_{start}"],
                        rtol=0, atol=1e-12)
                x = cached[start][:, nh:-nh][:, :spec["columns"]]
                result = dmd(x, rank=6, dt=record["dt"])
                expected = f["output_period_" + tag]
                phases = f["output_phase_" + tag]
                amplitudes = f["output_amplitude_" + tag]
                assert result["phases"].shape == phases.shape
                assert result["amplitudes"].shape == amplitudes.shape
                assert np.all(result["amplitudes"] >= 0)
                expected_frequency = np.nan_to_num(2 * np.pi / expected)
                measured_frequency = np.nan_to_num(2 * np.pi / result["periods"])
                np.testing.assert_allclose(np.sort(measured_frequency),
                                           np.sort(expected_frequency), atol=1e-10, rtol=1e-8)
                for j, period in enumerate(expected):
                    if not 3.5 <= period <= 15:
                        continue
                    pos = int(np.argmin(abs(result["periods"] - period)))
                    assert abs(result["periods"][pos] - period) / period < 1e-8
                    weight = amplitudes[j] ** 2
                    delta = result["phases"][pos] - phases[j]
                    origin = np.angle(np.sum(weight * np.exp(1j * delta)))
                    residual = np.angle(np.exp(1j * (delta - origin)))
                    assert np.sqrt(np.sum(weight * residual ** 2) / weight.sum()) < 1e-6
                    assert np.linalg.norm(result["amplitudes"][pos] - amplitudes[j]) / np.linalg.norm(amplitudes[j]) < 1e-6
                    count[record["label"]] += 1
    assert count == {"oblique": 12, "planview": 8}

def test_original_dispersion_approximation_and_independent_exact_physics():
    with np.load(FIXTURES / "upstream_stages.npz", allow_pickle=False) as f:
        np.testing.assert_allclose(kappa_to_mu(f["input_kappa"]), f["output_mu"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            dispersion_wavenumber(f["input_omega"], f["input_depth"]),
            f["output_wavenumber"], rtol=1e-12, atol=0)
        kappa = f["input_kappa"]
        exact = np.array([brentq(lambda mu: mu * np.tanh(mu) - v,
                                1e-14, max(2., 2 * v), xtol=1e-14) for v in kappa])
        mu = kappa_to_mu(kappa)
        assert np.max(abs(mu - exact) / exact) < .0005
        # Logarithmic sensitivity of mu*tanh(mu) is between 1 and 2.
        assert np.max(abs(mu * np.tanh(mu) - kappa) / kappa) < 2 * .0005 + .0005 ** 2
        assert abs(kappa_to_mu(1e-8) / np.sqrt(1e-8) - 1) < 1e-7
        assert abs(kappa_to_mu(100.) / 100. - 1) < 1e-12

def test_analytic_signal_and_exact_travelling_mode():
    time = np.arange(128)
    phase = 2 * np.pi * 8 * time / 128
    np.testing.assert_allclose(analytic_signal(np.cos(phase)),
                               np.exp(1j * phase), atol=2e-14)
    x = np.linspace(-4, 4, 31)
    t = np.arange(80) * .2
    values = np.exp(1j * (.12 * x[:, None] + 2 * np.pi / 6 * t))
    result = dmd(values, rank=1, dt=.2)
    assert abs(result["periods"][0] - 6) < 1e-12
    relative_phase = np.unwrap(result["phases"][0])
    assert np.max(abs(np.diff(relative_phase) / np.diff(x) - .12)) < 1e-12
    # Uniform shifts of water and bed preserve the physical depth/dispersion response.
    omega = np.array([.7, 1., 1.4])
    np.testing.assert_allclose(gamma_from_bed(omega, .177, -3.),
                               gamma_from_bed(omega, .227, -2.95), atol=1e-14)

@pytest.mark.parametrize("case", range(4))
def test_original_ransac_distributions_with_independent_sampling(case):
    with np.load(FIXTURES / "upstream_ransac_statistics.npz", allow_pickle=False) as f:
        x, y, phase = [f[f"input_{key}_{case}"] for key in ("x", "y", "phase")]
        expected = f[f"output_coefficients_{case}"]
        expected_counts = f[f"output_inlier_count_{case}"]
    measured, counts = [], []
    for repeat in range(64):
        coeff, inliers, info = fit_phase_plane(
            x, y, phase, np.random.default_rng(27000 + 1000 * case + repeat))
        measured.append(coeff)
        counts.append(len(inliers))
        assert info["trials"] <= 200
    measured = np.array(measured)
    # Five-standard-error comparison, not an exact seeded-output comparison.
    tolerance = 5 * np.sqrt(expected.var(0, ddof=1) / len(expected) +
                            measured.var(0, ddof=1) / len(measured)) + 1e-9
    assert np.all(abs(measured.mean(0) - expected.mean(0)) <= tolerance)
    count_tolerance = 5 * np.sqrt(expected_counts.var(ddof=1) / 64 +
                                  np.var(counts, ddof=1) / 64) + .1
    assert abs(np.mean(counts) - expected_counts.mean()) <= count_tolerance

def test_plane_physics_outliers_and_unidentifiable_geometry():
    rng = np.random.default_rng(1234)
    x, y = rng.uniform(-10, 10, (2, 100))
    phase = .12 * x - .08 * y + .03 + rng.normal(0, .003, 100)
    phase[:20] += 1.5
    coeff, inliers, info = fit_phase_plane(x, y, phase, np.random.default_rng(5678))
    np.testing.assert_allclose(coeff, [.12, -.08, .03], atol=.002)
    assert len(inliers) == 80
    with pytest.raises(ValueError):
        fit_phase_plane(np.arange(8.), np.zeros(8), np.ones(8),
                        np.random.default_rng(0))
    expected = int(np.ceil(np.log(1e-9) / np.log(1 - .5 ** 3)))
    assert abs(ransac_trial_count(.5, 1 - 1e-9) - expected) <= 1
