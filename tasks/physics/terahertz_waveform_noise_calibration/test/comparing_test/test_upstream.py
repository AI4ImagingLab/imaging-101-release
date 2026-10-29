from pathlib import Path
import numpy as np
from src.preprocessing import load_data,DT_PS
from src.physics_model import initialize,objective,gradient,decode_outputs,curvature
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_initialization_and_frozen_objective_gradient():
    x=load_data(ROOT/'data/raw_data.npz');u=np.load(ROOT/'test/fixtures/upstream_air_scans.npz');theta,s,d=initialize(x,DT_PS)
    np.testing.assert_allclose(theta,u['initial_theta'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(d['sigma_est'],[u['initial_sigma_alpha0'],u['initial_sigma_beta0'],u['initial_sigma_tau0']],rtol=1e-12)
    for local,key in [(s.mu,'delta_mu'),(s.a,'delta_a'),(s.eta,'eta')]:np.testing.assert_allclose(local,u['initial_scale_'+key],rtol=1e-12)
    np.testing.assert_array_equal(s.logv,[u['initial_scale_logv_alpha'],u['initial_scale_logv_beta'],u['initial_scale_logv_tau']])
    np.testing.assert_allclose(d['mu0'],u['initial_mu0']);np.testing.assert_array_equal(d['a0'],u['initial_a0']);np.testing.assert_array_equal(d['eta0'],u['initial_eta0'])
    assert not any(bool(u[k]) for k in u.files if k.startswith('initial_fix_') or k=='initial_est_mu');assert u['initial_workers']==1
    np.testing.assert_allclose(objective(theta,x,s),u['initial_nll'],rtol=1e-12)
    np.testing.assert_allclose(gradient(theta,x,s),u['initial_gradient'],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(objective(u['output_final_theta'],x,s),u['output_fval'],rtol=1e-12)
    np.testing.assert_allclose(gradient(u['output_final_theta'],x,s),u['output_final_gradient'],rtol=1e-6,atol=1e-7)

def test_full_pipeline_parity_and_stopping_state():
    x=load_data(ROOT/'data/raw_data.npz');out,d=reconstruct(x);ref=np.load(ROOT/'evaluation/baseline_reference.npz');up=np.load(ROOT/'test/fixtures/upstream_air_scans.npz')
    for k,v in ref.items():np.testing.assert_allclose(out[k],v,rtol=1e-6,atol=1e-7,err_msg=k)
    assert not d['success'];assert 'precision loss' in d['message'];assert d['gradient_inf_per_observation']<2e-9
    np.testing.assert_allclose(d['nll'],up['output_fval'],rtol=1e-12)
    # Frozen physical quantities and separately captured original forward/noise model are all consumed.
    for k,outkey in [('output_mu','reconstruction'),('output_a','amplitude'),('output_eta','delay'),('output_noise','noise')]:np.testing.assert_allclose(out[outkey][0],up[k],rtol=1e-6,atol=1e-7)
    extra=np.load(ROOT/'test/fixtures/extra_upstream.npz')
    for k in extra.files:np.testing.assert_allclose(out[k][0],extra[k],rtol=1e-6,atol=1e-7)

def test_curvature_against_independent_original_hessian_and_marginals():
    x=load_data(ROOT/'data/raw_data.npz');u=np.load(ROOT/'test/fixtures/upstream_air_scans.npz');diag=np.load(ROOT/'test/fixtures/diagnostic_stages.npz');theta,s,_=initialize(x,DT_PS)
    c=curvature(u['output_final_theta'],x,s,DT_PS)
    np.testing.assert_allclose(c['hessian'],diag['hessian'],rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(c['hess_inv'],u['output_hess_inv'],rtol=1e-5,atol=1e-9)
    for k in ['err_mu','err_a','err_eta']:np.testing.assert_allclose(c[k],u['output_'+k],rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(gradient(u['output_final_theta'],x,s),diag['final_gradient'],atol=1e-7)
    np.testing.assert_allclose(diag['newton_theta'],u['output_final_theta']-diag['newton_step'],rtol=1e-12)
    assert np.linalg.eigvalsh((c['hessian']+c['hessian'].T)/2).min()>0
    assert abs(gradient(diag['newton_theta'],x,s)).max()<1e-8
