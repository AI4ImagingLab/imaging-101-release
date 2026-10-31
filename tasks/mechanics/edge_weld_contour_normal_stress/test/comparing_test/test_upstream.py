from pathlib import Path
import numpy as np
from src.preprocessing import load_raw,fit_surface,boundary_displacements

ROOT=Path(__file__).resolve().parents[2]

def test_original_surface_coefficients():
    raw=load_raw(ROOT/'data/raw_data.npz')
    tck=fit_surface(raw)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(tck[0],f['tx'])
        np.testing.assert_array_equal(tck[1],f['ty'])
        np.testing.assert_allclose(tck[2],f['coefficients'],rtol=1e-10,atol=1e-14)

def test_original_serialized_boundary():
    raw=load_raw(ROOT/'data/raw_data.npz')
    boundary=boundary_displacements(raw,fit_surface(raw))
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        np.testing.assert_allclose(boundary,f['boundary_displacements'],rtol=0,atol=1e-12)
