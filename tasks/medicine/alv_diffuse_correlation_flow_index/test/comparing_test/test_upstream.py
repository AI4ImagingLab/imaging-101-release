from pathlib import Path
import numpy as np
from src.preprocessing import load_data,prepare
from src.physics_model import correlation
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def fixture():return np.load(ROOT/'test/fixtures/original_stages.npz')
def test_original_preprocessing():
    f=fixture();weighted,mask=prepare(load_data(ROOT/'data/raw_data.npz'))
    np.testing.assert_allclose(weighted,f['output_weighted'],rtol=1e-14,atol=1e-14)
    np.testing.assert_array_equal(mask,f['output_mask'])
def test_original_forward():
    f=fixture()
    for p,y in zip(f['input_case_parameters'],f['output_case_forward']):
        np.testing.assert_allclose(correlation(f['input_case_tau'].ravel(),*p),y,rtol=1e-13,atol=1e-13)
def test_complete_original_inverse():
    f=fixture();r=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    assert np.max(np.abs(r['beta'][0]-f['output_beta'].ravel()))<=1e-4
    assert np.max(np.abs(r['flow_index'][0]/f['output_flow_index'].ravel()-1))<=.002
    assert np.max(np.abs(r['predicted_g2'][0]-f['output_prediction']))<=3e-4
