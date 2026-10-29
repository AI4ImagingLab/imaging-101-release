from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import ewlc_marko_siggia_force,ewlc_marko_siggia_force_jac
from src.solvers import reconstruct
R=Path(__file__).resolve().parents[2]
F=np.load(R/'test/fixtures/original_stages.npz',allow_pickle=False)
def test_exact_public_observations_and_selection():
    d=load_data(R/'data/raw_data.npz')
    for name in ['control','reca']:
        for key in ['force','distance']:
            np.testing.assert_array_equal(d[key+'_'+name],F['input_all_'+key+'_'+name])
        np.testing.assert_array_equal(d['mask_'+name],F['config_mask_'+name])
def test_original_forward_and_jacobian():
    p=F['output_parameters']
    for name,lc in [('control',p[1]),('reca',p[4])]:
        d=F['input_distance_'+name]
        np.testing.assert_allclose(ewlc_marko_siggia_force(d,p[0],lc,p[2],p[3]),F['output_force_'+name],rtol=1e-12,atol=1e-11)
        np.testing.assert_allclose(ewlc_marko_siggia_force_jac(d,p[0],lc,p[2],p[3]),F['output_jac_'+name],rtol=1e-12,atol=1e-11)
def test_full_two_condition_inverse():
    fit=reconstruct(load_data(R/'data/raw_data.npz'))
    np.testing.assert_allclose(fit.x,F['output_parameters'][[0,1,2,4]],rtol=1e-6,atol=1e-7)
