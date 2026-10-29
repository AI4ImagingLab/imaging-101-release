from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,optical_density
from src.solvers import reconstruct
from src.physics_model import forward
ROOT=Path(__file__).resolve().parents[2]

def test_full_original_stages_and_forward():
    obs=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        od=optical_density(obs['intensity']); c=reconstruct(obs['intensity'],obs['extinction'],obs['distances'])
        np.testing.assert_allclose(od,z['output_od'],rtol=1e-12,atol=1e-16)
        np.testing.assert_allclose(c,z['output_hemoglobin'],rtol=1e-12,atol=1e-16)
    np.testing.assert_allclose(forward(c,obs['extinction'],obs['distances']),od,rtol=1e-12,atol=1e-15)
    assert c.shape==(1,68,2,4860) and c.dtype==np.float64
