from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]

def test_complete_upstream_pipeline_and_stages():
    raw=load_observations(ROOT/'data/raw_data.npz');result,diagnostics=reconstruct(raw)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        for i,d in enumerate(diagnostics):
            for local,original in [('x','x_Arai'),('y','y_Arai'),('vectors','zdata'),('slope','specimen_b'),('sigma_slope','specimen_b_sigma'),('beta','specimen_b_beta'),('frac','FRAC'),('mad','MAD_Anc'),('free_mad','MAD_Free'),('dang','DANG'),('direction','best_fit_vector_Anc'),('intercept','specimen_YT'),('field','B_anc'),('scat_low','scat_low'),('scat_high','scat_high')]:
                np.testing.assert_allclose(d[local],z[f'{original}_{i}'],rtol=1e-9,atol=1e-11)
            assert d['scat']==bool(z[f'SCAT_{i}'])
            np.testing.assert_allclose(d['temperature']+273,z[f't_Arai_{i}'])
            for kind,original in [('ptrm','ptrm_checks'),('tail','tail_check')]:
                check=d['checks'][kind]
                for j,axis in enumerate(['x','y']):np.testing.assert_allclose(check['points'][:,j],z[f'{axis}_{kind}_check_{i}'],rtol=1e-9,atol=1e-11)
                np.testing.assert_allclose(check['temperature']+273,z[f'{original}_temperatures_{i}'])
                key='ptrm_checks_starting_temperatures' if kind=='ptrm' else 'tail_checks_starting_temperatures'
                np.testing.assert_allclose(check['starting']+273,z[f'{key}_{i}'])
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:np.testing.assert_allclose(result,z['reconstruction'],rtol=1e-9,atol=1e-11)

def test_standard_entry_point():
    from src.main import run
    result,_=run(ROOT)
    with np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as z:np.testing.assert_array_equal(result,z['reconstruction'])
