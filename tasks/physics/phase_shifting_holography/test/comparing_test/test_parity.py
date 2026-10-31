from pathlib import Path
import numpy as np
from src.preprocessing import load_observations, center_frames
from src.physics_model import centered_spectrum
from src.solvers import find_carrier, reconstruct

ROOT=Path(__file__).resolve().parents[2]

def test_preprocessing_and_spectrum():
    h=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        centered=center_frames(h)
        np.testing.assert_allclose(centered[0],f['input_centered'],rtol=1e-11,atol=1e-10)
        np.testing.assert_allclose(centered_spectrum(centered[0]),f['output_spectrum'],rtol=1e-11,atol=1e-7)

def test_upstream_carrier_and_pipeline():
    h=load_observations(ROOT/'data/raw_data.npz')
    centered=center_frames(h)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(find_carrier(centered[0]),f['output_peak'])
        field,info=reconstruct(h)
        np.testing.assert_array_equal(info['carrier'],f['output_selected'])
        assert info['score']==int(f['output_score'])
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        np.testing.assert_allclose(np.stack([field.real,field.imag])[None],f['reconstruction'],rtol=1e-10,atol=1e-9)
