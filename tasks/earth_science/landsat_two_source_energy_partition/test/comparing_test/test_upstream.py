from pathlib import Path
import numpy as np
from src import physics_model as p
from src.preprocessing import load_observations,prepare_forcing
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_all_original_preprocessing_stages():
    fresh=prepare_forcing(load_observations(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/upstream_preprocessing.npz',allow_pickle=False) as z:
        for key in z.files:np.testing.assert_allclose(fresh[key],z[key],rtol=1e-6,atol=1e-5)
def test_original_pure_stages():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        t,pressure,ea,u,l=[z[k] for k in ['temperature','pressure','vapor_pressure','ustar','stability_length']]
        results={'rho':p.met_calc_rho(pressure,ea,t),'cp':p.met_calc_c_p(pressure,ea),'lambda':p.met_calc_lambda(t),'slope':p.met_calc_delta_vapor_pressure(t),'vapor':p.met_calc_vapor_pressure(t),'psi_h':p.MO_calc_Psi_H(z['psi_input']),'psi_m':p.MO_calc_Psi_M(z['psi_input']),'friction':p.MO_calc_u_star(np.array([1.,2.,3.,4.]),10.,l,.195,.0375),'L_recomputed':p.MO_calc_L(u,t,z['rho'],z['cp'],np.array([50.,100.,150.,200.]),np.array([0.,100.,200.,300.])),'view_fraction':p.TSEB_calc_F_theta_campbell(np.array([0.,20.,40.,60.]),np.array([.1,1.,2.,3.]))}
        for key,value in results.items():np.testing.assert_allclose(value,z[key],rtol=1e-6,atol=1e-5)
def test_full_image_original_and_physical_consistency():
    from evaluation.judge import evaluate
    result,diagnostics=reconstruct(ROOT/'data/raw_data.npz')
    assert evaluate(result)['passed'],evaluate(result)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as ref:
        for key in result:
            tolerance=.001 if 'temperature' in key else .1
            assert np.max(abs(result[key]-ref[key]))<=tolerance
    flags,counts=np.unique(diagnostics['pt_flag'],return_counts=True)
    np.testing.assert_array_equal(flags,[0,3,5]);np.testing.assert_array_equal(counts,[14195,47348,49386])
    assert np.all(diagnostics['soil_flag']==10) and len(diagnostics['soil_flag'])==6
    assert np.all(diagnostics['pt_L']<0) and np.all(diagnostics['soil_L']<0)
    assert np.all(diagnostics['pt_LEc']>=0) and np.all(diagnostics['pt_LEs']>=0)
