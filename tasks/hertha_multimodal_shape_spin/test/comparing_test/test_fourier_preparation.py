from pathlib import Path
import numpy as np
from src.preprocessing import load_observations

def test_all_fourier_observations_and_psfs_against_original_capture():
    root=Path(__file__).resolve().parents[2];data=load_observations(root/'data/raw_data.npz')
    with np.load(root/'test/fixtures/original_fourier_observations.npz',allow_pickle=False) as ref:
        for j in range(2):
            expected=ref['AO_real'+str(j)].ravel()+1j*ref['AO_imag'+str(j)].ravel()
            np.testing.assert_allclose(data['ao_fourier'][j],expected,atol=2e-15,rtol=2e-14)
            expected=ref['AO_psf_real'+str(j)].ravel()+1j*ref['AO_psf_imag'+str(j)].ravel()
            np.testing.assert_allclose(data['ao_psf_fourier'][j],expected,atol=2e-15,rtol=2e-14)
            np.testing.assert_allclose(data['ao_frequency'][:,0],ref['AO_freqx'+str(j)].ravel(),atol=1e-14,rtol=3e-16)
            np.testing.assert_allclose(data['ao_frequency'][:,1],ref['AO_freqy'+str(j)].ravel(),atol=1e-14,rtol=3e-16)
