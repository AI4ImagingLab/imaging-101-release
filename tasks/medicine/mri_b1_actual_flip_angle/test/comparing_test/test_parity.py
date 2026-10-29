from pathlib import Path
import numpy as np
from src.physics_model import invert_afi
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_upstream_scalar_inverse():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as f:
        np.testing.assert_allclose(invert_afi(f['input_signal']),f['output_field'],rtol=1e-12,atol=1e-13)
def test_full_native_volume():
    result=reconstruct(ROOT/'data/raw_data.npz')['reconstruction']
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:
        np.testing.assert_allclose(result,f['reconstruction'],rtol=1e-12,atol=1e-13)
    assert result.shape==(1,88,128,32)
