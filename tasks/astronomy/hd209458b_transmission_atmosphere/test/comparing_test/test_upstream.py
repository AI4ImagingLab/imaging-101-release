from pathlib import Path
import numpy as np
import pytest
from scipy.interpolate import UnivariateSpline
from src.preprocessing import load_observations,SPECIES,bracket_grid
from src.physics_model import TransmissionModel,integrate_radii,REF_PRESSURE,N_PROFILE,shell_paths
from src.solvers import (initialize_walkers,stretch_chain,log_probability,GUESS,
                         INITIAL_LOW,INITIAL_HIGH,LOW,HIGH)
T=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def case():
    model=TransmissionModel(load_observations(T/'data/raw_data.npz'))
    with np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:fixture={k:z[k] for k in z.files}
    return model,fixture

def test_all_frozen_physics_stages(case):
    model,f=case;input_fiducial_parameters=f['best'];p=input_fiducial_parameters
    depth,info=model.forward(p,full_output=True);mapped={}
    for key in ['absorption_coeff_atm','tau_los','stellar_spectrum','radii','P_profile','T_profile','mu_profile','unbinned_depths','unbinned_wavelengths']:
        mapped['compute_depths_0_2_'+key]=info[key]
    for i,s in enumerate(SPECIES):mapped['compute_depths_0_2_atm_abundances_'+s]=info['atm_abundances'][i]
    wave=np.array([np.mean(model.wave[m]) for m in model.bin_masks])
    for key in ['_get_binned_corrected_depths_0_0','compute_depths_0_0','band_wavelength']:mapped[key]=wave
    for key in ['_get_binned_corrected_depths_0_1','compute_depths_0_1','depth']:mapped[key]=depth
    mapped['_get_binned_corrected_depths_0_2']=info['stellar_spectrum']
    mapped['_solve_0_0']=info['radii'];mapped['_solve_0_1']=-np.diff(info['radii']);mapped['get_dl_0']=shell_paths(info['radii']);mapped['get_line_of_sight_tau_0']=info['tau_los']
    pressure=np.logspace(np.log10(model.pressure[0]),np.log10(model.pressure[-1]),N_PROFILE)
    ts=UnivariateSpline(np.log(pressure),np.full(N_PROFILE,p[3]),s=0);ms=UnivariateSpline(np.log(pressure),info['mu_profile'],s=0)
    mapped['_get_radii_0']=integrate_radii(np.log(np.append(REF_PRESSURE,pressure[pressure>REF_PRESSURE])),p[1],p[2],ts,ms)
    mapped['_get_radii_1']=integrate_radii(np.log(np.append(REF_PRESSURE,pressure[pressure<=REF_PRESSURE][::-1])),p[1],p[2],ts,ms)
    a=model.abundances(p[5]);tc=bracket_grid(info['T_profile'],model.temperature);pc=bracket_grid(info['P_profile'],model.pressure,10**p[6]);_,components=model.absorption(a,tc,pc,10**p[4])
    for stage,key in [('_get_gas_absorption_0','gas'),('_get_scattering_absorption_0','rayleigh'),('_get_collisional_absorption_0','collision')]:mapped[stage]=components[key]
    mapped['lnprob']=np.array(log_probability(p,model));mapped['chi2']=np.array(np.sum(((depth-model.raw['transit_depth'])/model.raw['uncertainty'])**2));mapped['best']=p
    for key,value in mapped.items():np.testing.assert_allclose(value,f[key],rtol=1e-9,atol=1e-12,err_msg=key)
    # Source-named case fields preserve provenance. Nonrandom physics outputs
    # are compared strictly here; sampler outputs have statistical consumers.
    assert set(f)==set(mapped)|{'initial','first_two_chain','first_two_logp'}

def test_uniform_initialization_and_two_transition_statistics(case):
    model,f=case
    config_guess=GUESS
    output_initial_samples=f['initial']
    output_two_step_chain=f['first_two_chain']
    output_two_step_logp=f['first_two_logp']
    # A different seed is a debugging convenience, never an equality oracle.
    initial,_=initialize_walkers(seed=19371,walkers=2001)
    assert initial.shape==(2001,8) and initial.dtype==np.float64 and np.isfinite(initial).all()
    assert output_initial_samples.shape==(50,8) and output_initial_samples.dtype==np.float64
    # The first point is explicitly assigned, not drawn from the RNG.
    np.testing.assert_array_equal(initial[0],config_guess)
    np.testing.assert_array_equal(output_initial_samples[0],config_guess)
    scale=INITIAL_HIGH-INITIAL_LOW
    new_uniform=(initial[1:]-INITIAL_LOW)/scale
    original_uniform=(output_initial_samples[1:]-INITIAL_LOW)/scale
    assert ((new_uniform>=0)&(new_uniform<=1)).all()
    assert ((original_uniform>=0)&(original_uniform<=1)).all()
    n=len(new_uniform);nref=len(original_uniform)
    variance=1/12
    # For U(0,1): E[X]=1/2, Var[X]=1/12, fourth central moment=1/80.
    # Six standard errors per coordinate are conservative distribution checks;
    # the reference49draws are never treated as exact expected coordinates.
    mean_se=np.sqrt(variance/n)
    def variance_of_sample_variance(size):
        return (1/80-((size-3)/(size-1))*variance**2)/size
    variance_se=np.sqrt(variance_of_sample_variance(n))
    np.testing.assert_allclose(new_uniform.mean(0),.5,rtol=0,atol=6*mean_se)
    np.testing.assert_allclose(new_uniform.var(0,ddof=1),variance,rtol=0,atol=6*variance_se)
    combined_mean_se=np.sqrt(variance*(1/n+1/nref))
    combined_variance_se=np.sqrt(variance_of_sample_variance(n)+variance_of_sample_variance(nref))
    np.testing.assert_allclose(new_uniform.mean(0),original_uniform.mean(0),rtol=0,atol=6*combined_mean_se)
    np.testing.assert_allclose(new_uniform.var(0,ddof=1),original_uniform.var(0,ddof=1),rtol=0,atol=6*combined_variance_se)
    chain,logp,acceptance=stretch_chain(model,steps=2,seed=8179)
    # Two transitions are a short transition contract, not a converged posterior
    # or an IID sample. Do not apply an IID posterior-moment error formula here.
    for states,probability in [(chain,logp),(output_two_step_chain,output_two_step_logp)]:
        assert states.shape==(50,2,8) and states.dtype==np.float64
        assert probability.shape==(50,2) and probability.dtype==np.float64
        assert np.isfinite(states).all() and not np.isnan(probability).any()
        assert not np.isposinf(probability).any()
        finite=np.isfinite(probability)
        assert finite.mean()>=.5  # Broad guard against a wholly invalid sampler.
        assert (states[:,:,:2]>0).all()
        assert (states[:,:,2:]>=LOW[2:]).all() and (states[:,:,2:]<=HIGH[2:]).all()
        assert (states[finite][:,2:]>LOW[2:]).all() and (states[finite][:,2:]<HIGH[2:]).all()
        normalized=(states.reshape(-1,8)-INITIAL_LOW)/scale
        assert np.isfinite(normalized.mean(0)).all() and (normalized.var(0)>0).all()
    assert acceptance.shape==(50,) and acceptance.dtype==np.float64
    assert np.isfinite(acceptance).all() and ((acceptance>=0)&(acceptance<=1)).all()
    assert 0<acceptance.mean()<1
    np.testing.assert_allclose(2*acceptance,np.rint(2*acceptance),rtol=0,atol=1e-14)

def test_full_pipeline_artifact():
    # Audit a separately executed complete CLI run; this test does not claim
    # a new 24-minute sampling run. Frozen stages independently recompute above.
    for filename in ['fresh_measurements.npz','notebook_measurements.npz']:
        path=T/'notebooks/reference_outputs'/filename
        assert path.exists(),'Complete both mandatory full CLI and active notebook first'
        with np.load(path,allow_pickle=False) as z:
            assert z['chain'].shape==(50,1000,8) and z['log_probability'].shape==(50,1000)
            # These are deterministic reductions of this same already-computed
            # artifact, not comparisons of stochastic samples between seeds/runs.
            np.testing.assert_array_equal(z['reconstruction'][0],np.median(z['chain'].reshape(-1,8),axis=0))
            np.testing.assert_array_equal(z['best_fit'][0],z['chain'].reshape(-1,8)[np.argmax(z['log_probability'].reshape(-1))])
            assert z['runtime_seconds']>0 and ((z['acceptance_fraction']>=0)&(z['acceptance_fraction']<=1)).all()
