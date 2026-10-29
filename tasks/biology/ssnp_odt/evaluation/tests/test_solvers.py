"""Unit tests for SSNP-ODT solvers."""

import numpy as np
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.physics_model import SSNPConfig, SSNPForwardModel
from src.solvers import SSNPReconstructor, tv_3d


METADATA = {
    "volume_shape": [8, 16, 16],
    "res_um": [0.1, 0.1, 0.1],
    "wavelength_um": 0.5,
    "n0": 1.0,
    "NA": 0.65,
    "n_angles": 4,
}


@pytest.fixture
def config():
    return SSNPConfig.from_metadata(METADATA)


@pytest.fixture
def model(config):
    return SSNPForwardModel(config, device="cpu")


@pytest.fixture
def dummy_measurements(config):
    n_angles = config.n_angles
    ny, nx = config.volume_shape[1], config.volume_shape[2]
    real = torch.randn(n_angles, ny, nx, dtype=torch.float64)
    imag = torch.randn(n_angles, ny, nx, dtype=torch.float64)
    return torch.complex(real, imag)


class TestTV3D:
    def test_constant_volume_minimal(self):
        # Huber-smoothed TV on a constant volume equals N_voxels * sqrt(epsilon).
        # For (4,8,8) with epsilon=1e-8: TV = 256 * 1e-4 ≈ 0.0256.
        v = torch.ones(4, 8, 8, dtype=torch.float64)
        n_voxels = 4 * 8 * 8
        expected = n_voxels * (1e-8 ** 0.5)
        tv = tv_3d(v)
        assert abs(tv.item() - expected) < 1e-6

    def test_nonnegative(self):
        v = torch.randn(4, 8, 8, dtype=torch.float64)
        assert tv_3d(v).item() >= 0.0

    def test_scalar_output(self):
        v = torch.randn(4, 8, 8, dtype=torch.float64)
        assert tv_3d(v).ndim == 0

    def test_differentiable(self):
        v = torch.randn(4, 8, 8, dtype=torch.float64, requires_grad=True)
        tv_3d(v).backward()
        assert v.grad is not None


class TestSSNPReconstructor:
    def test_output_shape(self, model, config, dummy_measurements):
        rec = SSNPReconstructor(n_iter=2, lr=10.0, device="cpu")
        dn, losses = rec.reconstruct(dummy_measurements, model)
        assert dn.shape == tuple(config.volume_shape)

    def test_loss_history_length(self, model, config, dummy_measurements):
        rec = SSNPReconstructor(n_iter=3, lr=10.0, device="cpu")
        _, losses = rec.reconstruct(dummy_measurements, model)
        assert len(losses) == 3

    def test_loss_decreases(self, model, config, dummy_measurements):
        # lr=0.1 is conservative enough to guarantee descent for any random init.
        rec = SSNPReconstructor(n_iter=5, lr=0.1, positivity=False, device="cpu")
        _, losses = rec.reconstruct(dummy_measurements, model)
        assert losses[-1] < losses[0]

    def test_positivity_constraint(self, model, config, dummy_measurements):
        rec = SSNPReconstructor(n_iter=3, lr=10.0, positivity=True, device="cpu")
        dn, _ = rec.reconstruct(dummy_measurements, model)
        assert np.all(dn >= -1e-12)

    def test_output_dtype(self, model, config, dummy_measurements):
        rec = SSNPReconstructor(n_iter=2, lr=10.0, device="cpu")
        dn, _ = rec.reconstruct(dummy_measurements, model)
        assert dn.dtype == np.float64

    def test_zero_measurements_zero_recon(self, model, config):
        nz, ny, nx = config.volume_shape
        n_angles = config.n_angles
        zero_meas = torch.zeros(n_angles, ny, nx, dtype=torch.complex128)
        rec = SSNPReconstructor(n_iter=3, lr=10.0, positivity=True, device="cpu")
        dn, losses = rec.reconstruct(zero_meas, model)
        assert np.all(dn >= -1e-12)
