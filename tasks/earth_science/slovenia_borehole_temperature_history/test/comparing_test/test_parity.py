from pathlib import Path
import numpy as np
from src.preprocessing import load_data,profiles,detrend
from src.physics_model import heat_matrix
from src.solvers import reconstruct,truncated_inverse

ROOT=Path(__file__).resolve().parents[2]

def test_upstream_stages():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as f:
        for i,(z,temp) in enumerate(profiles(data)):
            a,c=detrend(z,temp);g=heat_matrix(z)
            np.testing.assert_allclose(g,f[f'matrix_{i}'],atol=1e-13,rtol=0)
            np.testing.assert_allclose(a,f[f'anomalies_{i}'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(c,f[f'coefficients_{i}'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(truncated_inverse(g,a),f[f'histories_{i}'],atol=1e-9,rtol=0)
            np.testing.assert_allclose(g@f[f'histories_{i}'],f[f'forward_{i}'][:,1:],atol=1e-9,rtol=0)

def test_full_pipeline():
    fresh=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz') as a:
        np.testing.assert_allclose(fresh,a['reconstruction'][0],atol=1e-9,rtol=0)
