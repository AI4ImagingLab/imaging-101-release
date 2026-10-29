from pathlib import Path
import numpy as np
from src.solvers import reconstruct,_fit_circle

def test_full_original_stages():
    with np.load(Path(__file__).parents[1]/'fixtures/upstream_stages.npz',allow_pickle=False) as a:
        fit,response,normalized,cal=reconstruct(a['input_frequency'],a['input_scattering'])
        np.testing.assert_allclose(cal,a['output_calibration'],rtol=1e-8,atol=1e-14)
        np.testing.assert_allclose(normalized,a['output_normalized'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(response,a['output_response'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(_fit_circle(normalized),a['output_circle'],rtol=1e-8,atol=1e-10)
        assert abs(1/fit['Ql']-1/fit['Qc_dia_corr']-1/fit['Qi_dia_corr'])<1e-18
