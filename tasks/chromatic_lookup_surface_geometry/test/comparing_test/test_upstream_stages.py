from pathlib import Path
import numpy as np
from src.preprocessing import load_observations, normalize_observations
from src.physics_model import calibration_rays

ROOT=Path(__file__).resolve().parents[2]

def test_original_complete_preprocessing():
    raw=load_observations(ROOT/'data/raw_data.npz')
    q,mask=normalize_observations(raw['pattern'][0],raw['white'][0])
    with np.load(ROOT/'test/fixtures/upstream_primary_stages.npz',allow_pickle=False) as f:
        assert np.array_equal(mask,f['mask'])
        assert np.array_equal(q,f['normalized'])
    assert [int(m.sum()) for m in mask]==[142286,142389,142450]

def test_original_full_blas_calibration_rays():
    raw=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_calibration_rays.npz',allow_pickle=False) as f:
        color,depth=calibration_rays(raw,f['rows'])
        np.testing.assert_allclose(color,f['lut'],rtol=5e-7,atol=2e-6)
        np.testing.assert_allclose(depth,f['depth'],rtol=0,atol=2e-4)
