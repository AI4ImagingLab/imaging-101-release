"""Unit tests for SSNP-ODT preprocessing."""

import numpy as np
import json
import os
import pytest
import tempfile

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.preprocessing import load_measurements, load_metadata, prepare_data


METADATA = {
    "volume_shape": [8, 16, 16],
    "res_um": [0.1, 0.1, 0.1],
    "wavelength_um": 0.5,
    "n0": 1.0,
    "NA": 0.65,
    "n_angles": 4,
    "ri_contrast_scale": 0.01,
    "tiff_scale": 1.0,
}


@pytest.fixture
def tmp_data_dir():
    with tempfile.TemporaryDirectory() as d:
        # Write metadata
        with open(os.path.join(d, "meta_data.json"), "w") as f:
            json.dump(METADATA, f)

        # Write complex measurements
        meas = (np.random.randn(1, 4, 16, 16) +
                1j * np.random.randn(1, 4, 16, 16)).astype(np.complex128)
        np.savez(os.path.join(d, "raw_data.npz"), measurements=meas)

        yield d


class TestLoadMetadata:
    def test_required_keys(self, tmp_data_dir):
        meta = load_metadata(tmp_data_dir)
        for key in ["volume_shape", "res_um", "wavelength_um", "n0", "NA", "n_angles"]:
            assert key in meta

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_metadata("/nonexistent_dir")


class TestLoadMeasurements:
    def test_shape(self, tmp_data_dir):
        meas = load_measurements(tmp_data_dir)
        assert meas.shape == (4, 16, 16)

    def test_dtype_complex(self, tmp_data_dir):
        meas = load_measurements(tmp_data_dir)
        assert np.iscomplexobj(meas)
        assert meas.dtype == np.complex128

    def test_has_nonzero_imaginary(self, tmp_data_dir):
        meas = load_measurements(tmp_data_dir)
        assert np.any(meas.imag != 0)


class TestPrepareData:
    def test_returns_tuple(self, tmp_data_dir):
        result = prepare_data(tmp_data_dir)
        assert len(result) == 2

    def test_measurements_complex(self, tmp_data_dir):
        meas, meta = prepare_data(tmp_data_dir)
        assert np.iscomplexobj(meas)

    def test_metadata_dict(self, tmp_data_dir):
        meas, meta = prepare_data(tmp_data_dir)
        assert isinstance(meta, dict)
