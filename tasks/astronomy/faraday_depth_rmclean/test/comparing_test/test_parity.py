from pathlib import Path
import numpy as np
from src.preprocessing import prepare
from src.physics_model import dirty_spectrum,rotation_measure_response
from src.solvers import reconstruct

FIXTURE=Path(__file__).resolve().parents[1]/'fixtures/upstream_stages.npz'

def test_preprocessing_and_original_transform():
    with np.load(FIXTURE,allow_pickle=False) as f:
        data=prepare(f['input_frequency'],f['input_polarization'],f['input_noise'])
        for key,reference in [('lambda2','output_lambda2'),('weights','output_weights'),('lambda0','output_lambda0')]:
            np.testing.assert_array_equal(data[key],f[reference])
        dirty=dirty_spectrum(data['polarization'],data['lambda2'],f['config_phi'],data['weights'],data['lambda0'])
        np.testing.assert_allclose(dirty,f['output_dirty'],rtol=1e-9,atol=2e-11)
        response,phi2,width=rotation_measure_response(data['lambda2'],f['config_phi'],data['weights'],data['lambda0'])
        np.testing.assert_array_equal(phi2,f['output_phi2'])
        np.testing.assert_allclose(response,f['output_rmsf'],rtol=1e-12,atol=1e-13)
        assert width==float(f['output_fwhm'])

def test_full_original_pipeline_and_complete_update_path():
    with np.load(FIXTURE,allow_pickle=False) as f:
        output,details=reconstruct(f['input_frequency'],f['input_polarization'],f['input_noise'])
        for key,reference in [('reconstruction','output_clean'),('components','output_components')]:
            np.testing.assert_allclose(output[key][0],f[reference],rtol=1e-9,atol=2e-11)
        np.testing.assert_allclose(details['residual'],f['output_residual'],rtol=1e-9,atol=2e-11)
        assert details['iterations']==int(f['output_iterations'])==50
