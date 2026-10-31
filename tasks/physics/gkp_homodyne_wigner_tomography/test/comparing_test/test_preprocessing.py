from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,calibrate
ROOT=Path(__file__).resolve().parents[2]

def test_complete_bound_ADC_phase_and_confidence_calibration_against_original():
    data=load_observations();fresh=calibrate(data)
    with np.load(ROOT/'test/fixtures/upstream_preprocessing.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(fresh['phase_fit_param'],z['phase_fit_param'],rtol=1e-10,atol=1e-12)
        np.testing.assert_array_equal(fresh['cutoff'],z['cutoff'])
        np.testing.assert_array_equal(fresh['accepted_indices'],z['idx'])
        ids=z['raw_sample_indices']
        np.testing.assert_array_equal(fresh['q'][ids],z['q_sample'])
        np.testing.assert_allclose(fresh['phi'][ids],z['phi_sample'],rtol=1e-10,atol=1e-12)
        ids=z['sample_indices']
        np.testing.assert_array_equal(fresh['q_accepted'][ids],z['q_d_sample'])
        np.testing.assert_allclose(fresh['phi_accepted'][ids],z['phi_d_sample'],rtol=1e-10,atol=1e-12)
        assert len(fresh['q_accepted'])==1881028
        assert np.any(abs(fresh['q_accepted'])>7), 'No silent Wigner-window truncation of measurements'
