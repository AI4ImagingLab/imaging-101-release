from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from src.preprocessing import time_axis,interleave
from src.physics_model import components,forward,physical_jacobian
from src.solvers import INITIAL_VALUES,FREE_INDICES,expand_parameters,internal_values,external_values
from src.visualization import amplitude_metrics,observation_metrics

ROOT=Path(__file__).resolve().parents[2]


def test_independent_initial_forward_and_jacobian_fixtures():
    with np.load(ROOT/'test/fixtures/initial_stage.npz',allow_pickle=False) as ref:
        parameters=expand_parameters(INITIAL_VALUES[FREE_INDICES])
        np.testing.assert_array_equal(time_axis(1024),ref['input_time'])
        np.testing.assert_allclose(interleave(forward(parameters,ref['input_time'])),ref['output_initial_fid'],rtol=2e-13,atol=2e-13)
        np.testing.assert_allclose(physical_jacobian(parameters,ref['input_time']),ref['output_initial_jacobian'],rtol=2e-13,atol=2e-13)
        np.testing.assert_allclose(components(parameters,ref['input_time']),ref['output_initial_components'],rtol=2e-13,atol=2e-13)


def test_model_derivatives_by_finite_differences():
    p=np.array([[1.2,70.,50.,.2,0.],[.4,-30.,15.,-.5,0.]])
    time=np.linspace(.0003,.04,40);jac=physical_jacobian(p,time)
    for peak in range(2):
        for column in range(4):
            h=1e-6*max(1,abs(p[peak,column]));plus=p.copy();minus=p.copy()
            plus[peak,column]+=h;minus[peak,column]-=h
            numeric=(interleave(forward(plus,time))-interleave(forward(minus,time)))/(2*h)
            np.testing.assert_allclose(jac[:,4*peak+column],numeric,rtol=3e-6,atol=1e-9)


def test_physical_decay_frequency_phase_and_lorentzian_width():
    p=np.array([[2.,0.,np.pi*10,0.,0.]])
    time=np.array([0.,.01,.02]);np.testing.assert_allclose(forward(p,time),2*np.exp(-np.pi*10*time))
    p[0,1]=25.;p[0,3]=.3
    expected=2*np.exp(-np.pi*10*time)*np.exp(1j*(.3+2*np.pi*25*time))
    np.testing.assert_allclose(forward(p,time),expected,rtol=1e-14,atol=1e-14)
    # Independent continuous Fourier integral: real absorption is half-height
    # at ±FWHM/2 for the exp(-pi*FWHM*t) decay convention.
    center=quad(lambda t:np.exp(-np.pi*10*t),0,np.inf)[0]
    half=quad(lambda t:np.exp(-np.pi*10*t)*np.cos(2*np.pi*5*t),0,np.inf)[0]
    assert half/center==pytest.approx(.5,rel=1e-10)


def test_multiplet_constraints_and_bounded_parameter_roundtrip():
    free=INITIAL_VALUES[FREE_INDICES].copy();free[3]=.3
    p=expand_parameters(free)
    assert np.all(p[:,3]==.3) and np.all(p[:,4]==0)
    assert p[1,0]==p[0,0]/2 and p[2,0]==p[0,0]/2
    assert p[1,1]==p[0,1]-15 and p[2,1]==p[0,1]+15
    assert p[4,1]==p[3,1]-16 and p[6,1]==p[5,1]-16
    np.testing.assert_allclose(external_values(internal_values(free)),free,rtol=1e-13,atol=2e-13)


def test_metrics_keep_original_normalization_separate():
    r=np.array([1+1j,2-1j,-1+3j]);m=observation_metrics(r,2*r)
    assert m['relative_l2']==pytest.approx(1)
    assert m['upstream_centered_relative_squared_error']==pytest.approx(np.sum(abs(r)**2)/np.sum(abs(r-r.mean())**2))
    a=np.arange(1.,17.);result=amplitude_metrics(2*a,a)
    assert result['ncc']==pytest.approx(1)
    assert result['nrmse']==pytest.approx(np.sqrt(np.mean(a*a))/np.ptp(a))
    with pytest.raises(ValueError):amplitude_metrics(a,np.ones(16))
