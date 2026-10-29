import numpy as np
import pytest
from src.preprocessing import subtract_background,gaussian_half_kernel,gaussian_blur,registration_shift,translate_zero,preprocess_sample
from src.physics_model import donor_acquisition,corrected_efficiency
from src.solvers import estimate_corrections,_display_factor

def test_background_units_and_native_clamp():
    a=np.full((512,512),25,np.float32);a[100,100]=50;a[101,101]=10
    x,b=subtract_background(a)
    assert b==25 and x[100,100]==25 and x[101,101]==0

def test_gaussian_constant_impulse_and_boundary():
    k,t=gaussian_half_kernel()
    assert len(k)==5 and abs(float(k[0]+2*k[1:].sum())-1)<2e-7
    a=np.ones((33,33),np.float32)*7
    np.testing.assert_allclose(gaussian_blur(a),7,atol=2e-6,rtol=0)
    a=np.zeros((33,33),np.float32);a[16,16]=1
    x=gaussian_blur(a)
    np.testing.assert_allclose(x[16,12:21],k[0]*np.r_[k[:0:-1],k],atol=1e-8,rtol=0)
    assert np.max(abs(x-x[::-1,::-1]))==0

def test_integer_registration_sign_and_zero_boundary():
    a=np.zeros((32,32));a[10,8]=5;a[11,10]=3
    b=translate_zero(a,(3,-5))
    assert registration_shift(a,b)==(-3,5)
    np.testing.assert_array_equal(translate_zero(b,(-3,5)),a)
    assert not translate_zero(a,(32,0)).any()

def test_forward_inverse_and_brightness_scale():
    e=np.linspace(0,.8,101);d=1000+np.arange(101);a=np.full(101,300.)
    factors=[1.04,.02,.03,.12]
    db,da,ab,aa=donor_acquisition(e,d,a,factors)
    got=corrected_efficiency(db,da,ab,factors)
    np.testing.assert_allclose(got,e,atol=3e-7,rtol=0)
    np.testing.assert_allclose(corrected_efficiency(db*3,da*3,ab*3,factors),got,atol=3e-7,rtol=0)
    np.testing.assert_allclose(aa/ab,.12)

def test_no_transfer_physics_and_signed_noise_retained():
    db,da,ab,_=donor_acquisition(np.zeros(3),np.array([500,800,900]),np.ones(3)*100,[1,.02,.01,.1])
    np.testing.assert_allclose(corrected_efficiency(db,da,ab,[1,.02,.01,.1]),0,atol=2e-7)
    assert corrected_efficiency(np.array([120]),np.array([100]),np.array([0]),[1,0,0,0])[0]<0

def test_factor_rounding_matches_control_convention():
    assert _display_factor(.9939639486321431)==np.float32(.994)
    assert _display_factor(.01595887909330368)==np.float32(.016)
    d=np.ones((2,2,8,8),np.float32)*200;a=d.copy();t=d.copy()
    d[1,0]=250;a[0,0]=4;a[1,0]=6;t[1,1]=10
    factors,masks=estimate_corrections(d,a,t)
    np.testing.assert_array_equal(factors,np.array([.8,.02,.03,.05],np.float32))
    assert all(x.all() for x in masks)

@pytest.mark.parametrize('bad',[np.full((2,2,32,32),np.nan),np.zeros((2,32,32))])
def test_preprocess_rejects_invalid_observation(bad):
    with pytest.raises(ValueError):preprocess_sample(bad)

def test_empty_control_support_rejected():
    with pytest.raises(ValueError):estimate_corrections(*(np.zeros((2,2,8,8),np.float32) for _ in range(3)))

def test_visual_metric_definitions_independent_known_case():
    from src.visualization import comparison_metrics
    x=np.array([1.,2.,3.]);y=np.array([1.,2.,4.])
    m=comparison_metrics(x,y,np.ones(3,bool))
    assert abs(m['ncc']-17/(np.sqrt(14)*np.sqrt(21)))<1e-14
    assert abs(m['dynamic_range_nrmse']-np.sqrt(1/3)/3)<1e-14
    assert m['max_abs']==1
