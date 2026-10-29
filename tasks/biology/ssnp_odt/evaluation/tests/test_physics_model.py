"""Unit tests for SSNP-ODT physics model."""

import numpy as np
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.physics_model import SSNPConfig, SSNPForwardModel


METADATA = {
    "volume_shape": [16, 32, 32],
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
def zero_volume(config):
    nz, ny, nx = config.volume_shape
    return torch.zeros(nz, ny, nx, dtype=torch.float64)


class TestSSNPConfig:
    def test_from_metadata_res_normalisation(self, config):
        n0, wl = METADATA["n0"], METADATA["wavelength_um"]
        res_um = METADATA["res_um"]
        expected = tuple(r * n0 / wl for r in res_um)
        assert np.allclose(config.res, expected)

    def test_from_metadata_shape(self, config):
        assert config.volume_shape == (16, 32, 32)
        assert config.n_angles == 4


class TestPrecomputedQuantities:
    def test_kz_shape(self, model):
        assert model.kz.shape == (32, 32)

    def test_kz_positive(self, model):
        assert (model.kz > 0).all()

    def test_evanescent_mask_range(self, model):
        assert model.eva_mask.min() >= 0.0
        assert model.eva_mask.max() <= 1.0 + 1e-6

    def test_pupil_shape_and_dtype(self, model):
        assert model.pupil.shape == (32, 32)
        assert model.pupil.is_complex()


class TestIncidentField:
    def test_shape(self, model):
        u, ud = model._make_incident_field(0)
        assert u.shape == (32, 32)
        assert ud.shape == (32, 32)

    def test_unit_amplitude(self, model):
        u, _ = model._make_incident_field(0)
        assert torch.allclose(torch.abs(u), torch.ones(32, 32, dtype=torch.float64),
                              atol=1e-10)

    def test_angles_differ(self, model):
        u0, _ = model._make_incident_field(0)
        u1, _ = model._make_incident_field(1)
        assert not torch.allclose(u0, u1)


class TestForwardModel:
    def test_forward_single_shape(self, model, zero_volume):
        E = model.forward_single(zero_volume, 0)
        assert E.shape == (32, 32)

    def test_forward_single_complex(self, model, zero_volume):
        E = model.forward_single(zero_volume, 0)
        assert E.is_complex()

    def test_forward_shape(self, model, zero_volume):
        fields = model.forward(zero_volume)
        assert fields.shape == (4, 32, 32)
        assert fields.is_complex()

    def test_forward_autograd(self, model, config):
        nz, ny, nx = config.volume_shape
        dn = torch.zeros(nz, ny, nx, dtype=torch.float64, requires_grad=True)
        E = model.forward_single(dn, 0)
        loss = torch.sum(torch.abs(E) ** 2)
        loss.backward()
        assert dn.grad is not None
        assert dn.grad.dtype == torch.float64

    def test_nonzero_dn_changes_field(self, model, config, zero_volume):
        E_zero = model.forward_single(zero_volume, 0)
        nz, ny, nx = config.volume_shape
        dn_nonzero = torch.ones(nz, ny, nx, dtype=torch.float64) * 0.005
        E_nonzero = model.forward_single(dn_nonzero, 0)
        assert not torch.allclose(E_zero, E_nonzero)


class TestPropagation:
    def test_shape_preserved(self, model, config):
        ny, nx = config.volume_shape[1], config.volume_shape[2]
        u = torch.ones(ny, nx, dtype=torch.complex128)
        ud = torch.zeros(ny, nx, dtype=torch.complex128)
        u_new, ud_new = model._apply_propagation(u, ud, dz=1.0)
        assert u_new.shape == (ny, nx)
        assert ud_new.shape == (ny, nx)

    def test_zero_dz_applies_evanescent_mask(self, model, config):
        # At dz=0: cos(kz*0)=1, sin(kz*0)=0, so the operator applies only the
        # evanescent mask (not the full identity — attenuates high-frequency modes).
        ny, nx = config.volume_shape[1], config.volume_shape[2]
        u = torch.ones(ny, nx, dtype=torch.complex128)
        ud = torch.zeros(ny, nx, dtype=torch.complex128)
        u_new, _ = model._apply_propagation(u, ud, dz=0.0)
        # Output amplitude should be <= input amplitude (mask in [0,1])
        assert (torch.abs(u_new) <= torch.abs(u) + 1e-10).all()


class TestScattering:
    def test_u_unchanged(self, model, config):
        ny, nx = config.volume_shape[1], config.volume_shape[2]
        u = torch.randn(ny, nx, dtype=torch.complex128)
        ud = torch.randn(ny, nx, dtype=torch.complex128)
        dn_slice = torch.ones(ny, nx, dtype=torch.float64) * 0.01
        u_new, _ = model._apply_scattering(u, ud, dn_slice)
        assert torch.allclose(u_new, u)

    def test_zero_dn_ud_unchanged(self, model, config):
        ny, nx = config.volume_shape[1], config.volume_shape[2]
        u = torch.randn(ny, nx, dtype=torch.complex128)
        ud = torch.randn(ny, nx, dtype=torch.complex128)
        dn_zero = torch.zeros(ny, nx, dtype=torch.float64)
        _, ud_new = model._apply_scattering(u, ud, dn_zero)
        assert torch.allclose(ud_new, ud)
