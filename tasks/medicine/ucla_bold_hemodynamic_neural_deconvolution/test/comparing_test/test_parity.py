from pathlib import Path
import numpy as np
from src.preprocessing import load_data,preprocess
from src.physics_model import canonical_basis
from src.solvers import reconstruct
R=Path(__file__).resolve().parents[2];F=np.load(R/'test/fixtures/upstream_stages.npz',allow_pickle=False)
def test_preprocessing_and_basis():
    data=load_data(R/'data/raw_data.npz');first,filtered=preprocess(data['bold'])
    np.testing.assert_allclose(first,F['output_filter_0'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(filtered,F['output_filter_1'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(canonical_basis(),F['output_basis'],rtol=1e-12,atol=1e-12)
def test_complete_all400roi_original_path():
    result=reconstruct(load_data(R/'data/raw_data.npz'))
    np.testing.assert_allclose(result['hrf'][0],F['output_hrf'].T,rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(result['neural_drive'][0],F['output_neural_drive'].T,rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(result['coefficients'],F['output_beta'],rtol=1e-9,atol=1e-9)
    np.testing.assert_array_equal(result['events'],F['output_events'])
