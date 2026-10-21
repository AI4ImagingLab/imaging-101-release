from pathlib import Path
import numpy as np
from src.preprocessing import load_data,as_matrix
from src.physics_model import shrink,singular_threshold
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_original_input_svt_shrink_and_dual_updates():
    d=as_matrix(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(d,f['input_matrix'])
        low=singular_threshold(d,float(f['threshold']))
        np.testing.assert_allclose(low,f['output_low_rank'],rtol=1e-8,atol=1e-9)
        mu=1/float(f['threshold']);lam=1/np.sqrt(max(d.shape))
        sparse=shrink(d-low,lam/mu)
        np.testing.assert_allclose(sparse,f['output_sparse'],rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(mu*(d-low-sparse),f['output_dual'],rtol=1e-8,atol=1e-9)

def test_full_native_original_components_and_history():
    result,diag=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        for key in f.files:np.testing.assert_allclose(result[key],f[key],rtol=1e-8,atol=1e-9)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        assert diag['iterations']==len(f['output_history'])
        np.testing.assert_allclose(diag['history'][:,:2],f['output_history'][:,:2],rtol=1e-7,atol=1e-9)
    assert diag['relative_primal']<=1e-7
