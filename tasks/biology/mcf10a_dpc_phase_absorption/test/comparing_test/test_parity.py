from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_original_normalization_sources_wotf_and_full_tikhonov():
    result,stage=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:
        for k in result:np.testing.assert_allclose(result[k],f[k],rtol=1e-10,atol=1e-12)
    with np.load(ROOT/'test/fixtures/oracle_stages.npz') as f:
        for key in ['normalized','source','pupil','Hu','Hp','fx','fy']:np.testing.assert_array_equal(stage[key],f['output_'+key])
def test_original_tv_comparison():
    result,_=reconstruct(load_data(ROOT/'data/raw_data.npz'),method='TV')
    with np.load(ROOT/'notebooks/reference_outputs/upstream_tv.npz') as f:
        for k in result:np.testing.assert_allclose(result[k],f[k],rtol=1e-8,atol=1e-10)
