"""Frozen independent upstream stages; no oracle imports in these tests."""
from pathlib import Path
import numpy as np
from src import physics_model as physics
from src.preprocessing import reciprocal_geometry
from src.solvers import reconstruction_step

FIXTURE = Path(__file__).parents[1] / "fixtures/upstream_stages.npz"


def test_fourier_and_modulus_parity():
    with np.load(FIXTURE, allow_pickle=False) as f:
        wave = physics.reciprocal_wavefield(f["input_object"])
        np.testing.assert_allclose(wave, f["output_reciprocal"], rtol=1e-12, atol=1e-12)
        value, error = physics.amplitude_projection(wave, f["input_amplitude"])
        np.testing.assert_allclose(value, f["output_modulus"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(error, f["output_error"], rtol=1e-12)


def test_er_hio_parity():
    with np.load(FIXTURE, allow_pickle=False) as f:
        for hio, key in [(False, "output_er"), (True, "output_hio")]:
            value, _ = reconstruction_step(f["input_object"], f["input_support"],
                                           f["input_amplitude"], hio=hio)
            np.testing.assert_allclose(value, f[key], rtol=1e-12, atol=1e-12)


def test_partial_coherence_forward_parity():
    with np.load(FIXTURE, allow_pickle=False) as f:
        value = np.sqrt(physics.linear_convolution(abs(f["output_reciprocal"]) ** 2,
                                                   f["input_kernel"]))
        np.testing.assert_allclose(value, f["output_pc_amplitude"], rtol=1e-12, atol=1e-12)


def test_corrected_partial_coherence_update_parity():
    with np.load(FIXTURE, allow_pickle=False) as f:
        shape = f["input_kernel"].shape
        previous = physics.crop_center(f["input_previous"], shape)
        measured = physics.crop_center(f["input_amplitude"], shape)
        value = physics.update_coherence(abs(f["output_reciprocal"]), previous,
                                         measured, f["input_kernel"])
        np.testing.assert_allclose(value, f["output_pc_kernel"], rtol=1e-12, atol=1e-14)


def test_support_parity():
    with np.load(FIXTURE, allow_pickle=False) as f:
        np.testing.assert_array_equal(physics.shrink_support(f["input_object"], 3.),
                                      f["output_shrink_support"])


def test_original_beamline_geometry_parity():
    with np.load(FIXTURE, allow_pickle=False) as f:
        basis, q = reciprocal_geometry()
        np.testing.assert_allclose(basis, f["output_basis"], rtol=1e-13, atol=1e-18)
        np.testing.assert_allclose(q, f["output_q"], rtol=1e-13, atol=1e-14)


def test_even16_partial_coherence_parity():
    with np.load(FIXTURE.parent / "corrected_even16.npz", allow_pickle=False) as f:
        previous = physics.crop_center(f["input_previous"], (16, 16, 16))
        measured = physics.crop_center(f["input_amplitude"], (16, 16, 16))
        result = physics.update_coherence(f["input_wave_amplitude"], previous,
                                          measured, f["input_kernel"], iterations=20)
        np.testing.assert_allclose(result, f["output_pc_kernel"], rtol=2e-12, atol=1e-14)
        forward = np.sqrt(physics.linear_convolution(f["input_wave_amplitude"] ** 2,
                                                      f["input_kernel"]))
        np.testing.assert_allclose(forward, f["output_pc_amplitude"], rtol=1e-12, atol=1e-13)
