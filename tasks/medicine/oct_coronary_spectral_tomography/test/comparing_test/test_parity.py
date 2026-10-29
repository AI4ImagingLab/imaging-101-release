from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
from src.physics_model import forward
ROOT=Path(__file__).resolve().parents[2]
def test_original_full_pipeline_and_stages():
    result,stage=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f: np.testing.assert_allclose(result,f['reconstruction'],rtol=2e-5,atol=2e-5)
    with np.load(ROOT/'test/fixtures/oracle_stages.npz') as f:
        np.testing.assert_array_equal(stage['indices'],f['output_chirp'])
        np.testing.assert_allclose(stage['spectrum'],f['output_spectrum'],rtol=2e-6,atol=0.1)
        np.testing.assert_allclose(stage['dispersion'],f['output_dispersion'],rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(stage['window'],f['output_window'])
def test_fourier_observation_roundtrip():
    result,stage=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    spectrum=forward(result[0])
    target=stage['spectrum']*stage['window'][None]
    # Float32 FFT roundtrip is normwise stable; endpoints have exact zero
    # apodization, so per-pixel relative errors are undefined there.
    assert np.linalg.norm(spectrum[:,:800]-target)/np.linalg.norm(target)<5e-7
    assert np.linalg.norm(spectrum[:,800:])/np.linalg.norm(spectrum)<1e-6
