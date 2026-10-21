from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.physics_model import *
F=Path(__file__).resolve().parents[1]/'fixtures/upstream.npz'

def test_exact_observation_and_hankel_parity():
    d=load_observations();q,H,H0=hankel(d)
    with np.load(F,allow_pickle=False) as f:
        for key,old in [('spin_echo_length','input_z'),('depolarisation','input_depolarisation'),('standard_error','input_error'),('wavelength','input_wavelength'),('length_standard_error','input_z_error')]:np.testing.assert_array_equal(d[key],f[old])
        np.testing.assert_array_equal(q,f['input_q'])
        np.testing.assert_allclose(H0,f['output_H0'],rtol=1e-13,atol=0)
        np.testing.assert_allclose(H[f['input_H_indices']],f['output_H_samples'],rtol=1e-13,atol=1e-25)
        # Independent dimensional check: both the physical angle limit and the
        # upstream's inconsistent angle/Q comparison leave this bound grid unmasked.
        assert np.arcsin(q.max()*d['wavelength'].max()/(2*np.pi))<THETA_MAX

def test_original_c_kernel_and_full_forward():
    d=load_observations();q,H,H0=hankel(d)
    with np.load(F,allow_pickle=False) as f:
        np.testing.assert_allclose(sphere_intensity(f['output_parameters'],q),f['output_iq'],rtol=3e-10,atol=1e-10)
        np.testing.assert_allclose(forward(INITIAL,q,H,H0),f['output_initial'],rtol=2e-11,atol=1e-13)
        np.testing.assert_allclose(forward(f['output_parameters'],q,H,H0),f['output_prediction'],rtol=2e-11,atol=1e-13)
        np.testing.assert_allclose(residuals(INITIAL,d,q,H,H0),f['output_initial_residual'],rtol=1e-10,atol=1e-10)

def test_zero_q_guinier_and_total_scattering():
    p=np.array([10000.,.08]);v=4*np.pi*p[0]**3/3
    iq0=p[1]*(1-p[1])*(1.41-2.70)**2*1e-4*v
    np.testing.assert_allclose(sphere_intensity(p,np.array([0.])),iq0,rtol=2e-14)
    tiny=np.array([1e-8]);ratio=sphere_intensity(p,tiny)[0]/iq0
    np.testing.assert_allclose(ratio,1-(tiny[0]*p[0])**2/5,rtol=1e-13)
    d=load_observations();q,H,H0=hankel(d)
    # Analytic integral0..infinity q*I(q)/(2pi)=1.5R*contrast²*scale.
    numeric=np.dot(H0,sphere_intensity(p,q))
    analytic=1.5*p[0]*(1.41-2.70)**2*1e-4*p[1]*(1-p[1])
    np.testing.assert_allclose(numeric,analytic,rtol=.002)

def test_autocovariance_support_and_phase_exchange():
    p=np.array([10300.,.08]);r=np.array([0.,p[0],2*p[0],3*p[0]])
    np.testing.assert_allclose(autocovariance(p,r),p[1]*(1-p[1])*np.array([1.,5/16,0,0]),atol=1e-15)
    p2=p.copy();p2[1]=1-p[1]
    np.testing.assert_allclose(autocovariance(p),autocovariance(p2),rtol=2e-14)
    assert np.all(autocovariance(p)>=0)

def test_full_pipeline_upstream():
    from src.solvers import reconstruct
    x,info=reconstruct(load_observations())
    with np.load(F,allow_pickle=False) as f:
        assert np.max(np.abs(x['parameters'][0]-f['output_parameters'])/PARAMETER_SCALES)<1e-5
    assert info['success'] and info['chisqr']<40.4

def test_original_uniform_prior_and_bound_penalty():
    d=load_observations();q,H,H0=hankel(d)
    with np.load(F,allow_pickle=False) as f:
        np.testing.assert_array_equal(f['output_prior_residual'],np.zeros(NPRIOR))
    np.testing.assert_allclose(bounded_residuals(INITIAL,d,q,H,H0),np.r_[residuals(INITIAL,d,q,H,H0),np.zeros(13)])
    outside=INITIAL.copy();outside[0]=50
    got=bounded_residuals(outside,d,q,H,H0)
    inside=outside.copy();inside[0]=100
    plain=residuals(inside,d,q,H,H0)
    assert got.shape==(53,) and np.max(np.abs(got[:40]-plain))>18000
    np.testing.assert_array_equal(got[40:],np.zeros(13))
