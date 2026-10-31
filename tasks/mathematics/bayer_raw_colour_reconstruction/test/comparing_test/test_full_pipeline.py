from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,normalize_mosaic
from src.solvers import reconstruct,bilinear_comparison
T=Path(__file__).resolve().parents[2]

def test_native_full_original_parity_and_physics():
    a,b,_=load_observations(T/'data/raw_data.npz');m=normalize_mosaic(a,b)
    fresh=reconstruct(m)
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(fresh[None],z['reconstruction'],rtol=1e-13,atol=1e-14)
    assert fresh.shape==(2128,2840,3) and fresh.dtype==np.float64
    np.testing.assert_array_equal(fresh[0::2,0::2,0],m[0::2,0::2])
    np.testing.assert_array_equal(fresh[1::2,1::2,2],m[1::2,1::2])
    np.testing.assert_array_equal(fresh[0::2,1::2,1],m[0::2,1::2])
    np.testing.assert_array_equal(fresh[1::2,0::2,1],m[1::2,0::2])
    del fresh
    with np.load(T/'notebooks/reference_outputs/upstream_bilinear.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(bilinear_comparison(m)[None],z['reconstruction'],rtol=1e-13,atol=1e-14)
