import numpy as np
import pytest
from src.physics_model import fluorescence_response,invert_response,normalize_ratio,TOP,BOTTOM,EPS
from src.preprocessing import calibration_levels,valid_pixels
from src.solvers import invert_image
from src.visualization import agreement


def test_independent_acid_base_response_and_inverse():
    ph=np.array([4.5,5.,5.5,6.,6.5,7.,7.5])
    # Independent equilibrium odds: acid/base = 10^(h*(pH50-pH)).
    acid_to_base=np.exp(np.log(10)*1.2133*(6.2110-ph))
    expected=(-.0295*acid_to_base+1.1171)/(1+acid_to_base)
    np.testing.assert_allclose(fluorescence_response(ph),expected,atol=1e-15)
    inverse,clipped=invert_response(expected)
    np.testing.assert_allclose(inverse,ph,atol=2e-14);assert not clipped.any()
    assert np.all(np.diff(inverse)>0)


def test_source_saturation_is_not_display_range_clamping():
    ph,clipped=invert_response([-100.,BOTTOM,0.,1.,TOP,100.])
    assert np.isfinite(ph).all() and ph[0]==ph[1] and ph[-1]==ph[-2]
    assert ph[0]<5 and ph[-1]>7
    np.testing.assert_allclose(fluorescence_response(ph),np.clip([-100.,BOTTOM,0.,1.,TOP,100.],BOTTOM+EPS,TOP-EPS),atol=1e-15)


def test_mask_uses_only_input_availability_and_preserves_negative_ratios():
    image=np.array([[0,np.nan,-1],[.4,1.,5.]],dtype=np.float32)
    ph,mask,clipped=invert_image(image,.3,1.8)
    np.testing.assert_array_equal(mask,[[False,False,True],[True,True,True]])
    assert np.all(ph[~mask]==0) and ph[0,2]>0 and clipped[0,2]


def test_independent_calibration_equal_root_weight_not_pooled_pixels():
    low=np.array([[[1.,np.nan]],[[3.,3.]]]);high=low+4
    lower,upper,a,b=calibration_levels(low,high)
    assert lower==2 and upper==6
    assert lower!=np.nanmean(low)
    with pytest.raises(ValueError):normalize_ratio([1],2,2)
    with pytest.raises(ValueError):calibration_levels(low,low)


def test_known_metrics_and_degenerate_reference():
    metric=agreement(np.array([1.,2.,3.]),np.array([1.,2.,4.]),np.ones(3,bool))
    assert metric['ncc']==pytest.approx(17/np.sqrt(14*21))
    assert metric['nrmse']==pytest.approx(np.sqrt(1/3)/3)
    with pytest.raises(ValueError):agreement(np.ones(3),np.ones(3),np.ones(3,bool))
