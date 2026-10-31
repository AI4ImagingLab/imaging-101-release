from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from src.physics_model import rf_factors,trajectory,residual
from src.preprocessing import load_data,prepare
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_exact_rf_preprocessing_and_stage_fixture():
    s,f=load_data(ROOT/'data/raw_data.npz');_,flips=prepare(s,f)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(flips,z['input_flips']);ss,zz=rf_factors(flips)
        np.testing.assert_array_equal(ss,z['output_signal_scale']);np.testing.assert_array_equal(zz,z['output_longitudinal_scale'])
        for i in range(5):
            state,u=trajectory(z[f'input_params_{i}'],z[f'input_pyr_{i}'],zz)
            np.testing.assert_allclose(state,z[f'output_longitudinal_{i}'],rtol=2e-13,atol=2e-9);np.testing.assert_allclose(u,z[f'output_forcing_{i}'],rtol=2e-13,atol=2e-9)
            mz=np.zeros_like(state);mz[0]=z[f'input_pyr_{i}'];mz[1]=z[f'input_lactate_{i}']/ss[1]
            np.testing.assert_allclose(residual(z[f'input_params_{i}'],mz,zz,ss),z[f'output_residual_{i}'],rtol=2e-12,atol=1e-9)
def test_constant_substrate_analytic_ode():
    times=np.arange(25)*2.;p=np.full(25,10.);state,u=trajectory([.012,4],p,np.ones((4,25)))
    expected=4*np.exp(-times/25)+.012*10*25*(1-np.exp(-times/25))
    np.testing.assert_allclose(state[1],expected,rtol=2e-14,atol=1e-13)
    np.testing.assert_allclose(u[:-1],(.012+1/30)*10,rtol=2e-14)
def test_zero_exchange_rf_depletion_and_no_signal_rank_failure():
    _,z=rf_factors(np.full((4,25),.2));state,u=trajectory([0,4],np.arange(25,dtype=float),z)
    np.testing.assert_allclose(state[1],4*(np.cos(.2)*np.exp(-2/25))**np.arange(25),rtol=1e-13)
    a,_=trajectory([.01,0],np.zeros(25),z);b,_=trajectory([.8,0],np.zeros(25),z);np.testing.assert_array_equal(a,b)
def test_independent_continuous_integration_and_rf_grouping():
    ss,z=rf_factors(np.full((4,25),.2));p=np.random.default_rng(384).uniform(1,10,25);state,u=trajectory([.013,3.],p,z)
    for t in (0,1,14,23):
        initial=[p[t]*z[0,t],state[1,t]*z[1,t]]
        sol=solve_ivp(lambda _,v:[u[t]-(1/30+.013)*v[0],.013*v[0]-v[1]/25],(0,2),initial,rtol=1e-11,atol=1e-13)
        np.testing.assert_allclose(sol.y[:,-1],state[:2,t+1],rtol=1e-10,atol=1e-10)
    angles=np.array([[.1,.2,.3,.4]]);s,z=rf_factors(angles,2)
    np.testing.assert_allclose(s,[[np.sin(.1)/2+np.cos(.1)*np.sin(.2)/2,np.sin(.3)/2+np.cos(.3)*np.sin(.4)/2]])
    np.testing.assert_allclose(z,[[np.cos(.1)*np.cos(.2),np.cos(.3)*np.cos(.4)]])
def test_full_original_parity_and_real_observation_quality():
    s,f=load_data(ROOT/'data/raw_data.npz');fields,d=reconstruct(s,f)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as ref:
        for k,a in fields.items():
            r=ref[k];dr=np.ptp(r);assert np.sqrt(np.mean((a-r)**2))/dr<=1e-6;assert abs(a-r).max()/dr<=1e-5
    y=s[1];assert np.linalg.norm(d['fitted_lactate']-y)/np.linalg.norm(y)<.5
    assert np.linalg.norm(d['fitted_lactate']-y)<np.linalg.norm(y-y.mean(-1)[...,None])
    assert np.all(d['status']>0);assert all(np.isfinite(a).all() and np.all(a>=0) for a in fields.values())
