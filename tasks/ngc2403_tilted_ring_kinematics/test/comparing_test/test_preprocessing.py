from pathlib import Path
import numpy as np
from src import preprocessing as prep
T=Path(__file__).resolve().parents[2]

def test_original_input_only_search_mask():
    function=getattr(prep,'search_mask',None)
    assert callable(function),'input-only SEARCH is not implemented'
    with np.load(T/'data/raw_data.npz',allow_pickle=False) as z: cube=z['spectral_cube'][0]
    with np.load(T/'test/fixtures/preprocessing.npz',allow_pickle=False) as z: reference=z['output_mask']
    actual,stats=function(cube)
    np.testing.assert_array_equal(actual,reference)

def test_original_float32_first_moment():
    function=getattr(prep,'first_moment',None)
    assert callable(function),'first moment is not implemented'
    with np.load(T/'data/raw_data.npz',allow_pickle=False) as z: cube=z['spectral_cube'][0]
    with np.load(T/'test/fixtures/preprocessing.npz',allow_pickle=False) as z:
        actual=function(cube,z['output_mask'])
        np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(z['output_moment']))
        np.testing.assert_allclose(actual,z['output_moment'],rtol=0,atol=5e-5,equal_nan=True)
