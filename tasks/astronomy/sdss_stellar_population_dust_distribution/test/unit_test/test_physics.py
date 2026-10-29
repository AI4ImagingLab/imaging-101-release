from pathlib import Path
import sys,importlib.util,json,shutil,subprocess
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.preprocessing import load_data,normalize_models
from src.physics_model import highpass,calzetti,forward,adjoint,smooth_curve
from src.solvers import reconstruct,fit_population,sigma_clip
spec=importlib.util.spec_from_file_location('taskjudge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_normalization_and_linear_adjoint():
    rng=np.random.default_rng(91);models=rng.uniform(1,3,(4,120));flux=rng.uniform(2,3,120);a=rng.uniform(.8,1.2,120)
    norm=normalize_models(flux,models*a);np.testing.assert_allclose(np.median(norm,axis=1),np.median(flux))
    w=rng.normal(size=4);r=rng.normal(size=120)
    assert abs(np.dot(forward(w,a,models,flux),r)-np.dot(w,adjoint(r,norm)))<1e-10
    np.testing.assert_allclose(forward(w*2,a,models,flux),2*forward(w,a,models,flux))
def test_highpass_constant_and_scaling():
    x=2+np.sin(np.arange(200)/3)*.1
    np.testing.assert_allclose(highpass(np.full(200,3.)),3.,atol=1e-14)
    np.testing.assert_allclose(highpass(x*4),4*highpass(x),atol=1e-13)
def test_calzetti_monotonic_and_reference():
    wave=np.array([4000.,5500.,7000.]);np.testing.assert_array_equal(calzetti(0,wave),np.ones(3))
    assert np.all(calzetti(.2,wave)<calzetti(.1,wave));assert np.all(np.diff(calzetti(.2,wave))>0)
    l=.55;k=2.659*(-2.156+1.509/l-.198/l**2+.011/l**3)+4.05
    assert calzetti(.2,wave)[1]==pytest.approx(10**(-.08*k))
def test_running_median_and_sigmaclip_known_case():
    np.testing.assert_array_equal(smooth_curve(np.arange(5.),np.array([1.,9.,2.,3.,4.]),1.1),[5.,2.,3.,3.,3.5])
    clipped,lo,hi=sigma_clip(np.r_[np.tile([-1.,0.,1.],20),100.]);assert len(clipped)==60 and lo<0<hi
def test_small_two_component_recovery_and_bic():
    x=np.linspace(0,2*np.pi,100);models=np.array([2+.3*np.sin(x),2+.3*np.cos(x)])
    result=fit_population(models.mean(axis=0),np.full(100,.01),models)
    np.testing.assert_allclose(result['best_weights'],[.5,.5],atol=1e-14)
    assert result['chis'].min()<1e-20;assert result['history'][-1,1]==result['history'][-2,1]
def test_public_acquisition_and_support_mask():
    d=load_data(ROOT/'data/raw_data.npz');assert d['templates'].shape==(198,2308)
    assert len(d['observed_flux'])==3835 and len(d['context_flux'])==2327
    assert d['support_mask'].sum()==2308
    np.testing.assert_array_equal(d['context_wavelength'][d['support_mask']],d['wavelength'])
    np.testing.assert_array_equal(d['context_flux'][d['support_mask']],d['flux'])
    assert np.min(d['templates'])>0 and np.all(d['uncertainty']>0)
    assert d['wavelength'].min()>3750 and d['wavelength'].max()<7400

def test_display_metric_and_marginal_known_values():
    from src.visualization import residual_statistic,marginal
    assert residual_statistic(np.array([3.,5.]),{'flux':np.array([1.,1.]),'uncertainty':np.array([2.,2.])})==2.5
    nodes,weights=marginal(np.array([.1,.2,.7]),np.array([1.,2.,1.]));np.testing.assert_array_equal(nodes,[1.,2.]);np.testing.assert_allclose(weights,[.8,.2])

def test_readme_declared_constants_and_keys():
    text=(ROOT/'README.md').read_text(encoding='utf8');d=load_data(ROOT/'data/raw_data.npz')
    for key in d:assert '| '+key+' |' in text
    for value in ['2308','198','3835','2327','3750','7400','0.02127545326948166','135.89956665039062','0.0938291996717453']:assert value in text
    assert d['wavelength'].max()==np.float32(7397.02685546875)
