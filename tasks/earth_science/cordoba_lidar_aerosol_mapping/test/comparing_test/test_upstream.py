from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import load_data
from src.physics_model import reference_integral,forward_signal,REFERENCE,MOLECULAR_LIDAR_RATIO
from src.solvers import reconstruct,fernald,source_smooth
ROOT=Path(__file__).resolve().parents[2]

def test_all_original_stages_and_complete_field():
    out,stages=reconstruct(load_data())
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as z:
        for key in stages:
            actual=stages[key]
            if key not in ('beta_unsmoothed','alpha_unsmoothed'):actual=actual[...,53:]
            expected=z[key]
            if key=='integral_molecular':actual=np.broadcast_to(actual,expected.shape)
            assert_allclose(actual,expected,rtol=0,atol=1e-10*max(np.max(abs(expected)),1e-30),err_msg=key)
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:
        for key in out:assert_allclose(out[key],z[key],rtol=0,atol=1e-7*np.ptp(z[key]))
    assert out['reconstruction'].shape==(1,66,759)
    assert_allclose(out['extinction'],50*out['reconstruction'],rtol=2e-15,atol=1e-19)

def test_independent_constant_and_linear_quadrature():
    y=np.full(20,3.);result=reference_integral(y,reference=10,start=0,step=.25)
    assert_allclose(result,(np.arange(20)-10)*.75,atol=1e-15)
    y=np.arange(20)*.25
    assert_allclose(reference_integral(y,10,0,.25),.5*(y*y-y[10]**2),atol=1e-15)

def test_analytic_molecular_only_limit_and_gain_invariance():
    n=4096;r=np.arange(n)*7.5;bm=np.full(n,1e-6);am=bm*MOLECULAR_LIDAR_RATIO
    # Exact Beer-Lambert molecular-only return, known boundary aerosol zero.
    signal=np.exp(-2*am*(r-r[REFERENCE]))[None]
    beta,alpha,_=fernald(signal,bm,am,smooth=False)
    assert_allclose(beta[:,54:813],0,atol=6e-14)
    scaled=fernald(signal*137,bm,am,smooth=False)[0]
    assert_allclose(beta[:,54:813],scaled[:,54:813],atol=1e-20)
    forward=forward_signal(np.zeros((1,n)),bm,am)
    assert_allclose(forward,signal,rtol=2e-14)

def test_source_alias_smoothing_is_not_symmetric_convolution():
    actual=source_smooth(np.arange(5.,dtype=float)[None],width=3)
    assert_allclose(actual[0],[.5,7/6,37/18,163/54,379/108],rtol=1e-14)
