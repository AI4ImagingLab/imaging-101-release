from pathlib import Path
import numpy as np
from src.physics_model import parallel_ray
from src.preprocessing import load_data
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_original_ray_matrix():
    f=np.load(ROOT/'test/fixtures/upstream_operator.npz')
    a=parallel_ray(128,1.,f['param_angles'],128,1.)
    np.testing.assert_array_equal(a.row,f['output_rows']);np.testing.assert_array_equal(a.col,f['output_cols']);np.testing.assert_array_equal(a.data,f['output_vals'])
def test_original_full_pipeline():
    d,angles=load_data(ROOT/'data/raw_data.npz');x,h=reconstruct(d,angles)
    ref=np.load(ROOT/'test/fixtures/upstream_full.npz')['output_reconstruction'][0]
    np.testing.assert_allclose(x,ref,rtol=2e-6,atol=.0005)
