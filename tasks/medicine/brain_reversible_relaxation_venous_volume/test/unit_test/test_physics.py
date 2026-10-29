from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import TAU,log_measurements,load_observations,observation_mask,SPACING_MM
from src.physics_model import design_matrix,forward_log,oxygen_diagnostics,GAMMA,B0,DELTA_CHI,HCT,KAPPA
from src.solvers import fit_signals
from src.visualization import rmse
ROOT=Path(__file__).resolve().parents[2]

def test_analytic_inverse_and_normal_equation():
    # Independent closed form: slope of log(signal) at positive long tau;
    # intercept minus measured log spin-echo gives DBV.
    signal=np.full(24,100.)
    t=TAU[TAU>.015];signal[TAU>.015]=100*np.exp(.06-4*t)
    p,r=fit_signals(signal)
    np.testing.assert_allclose(p,[.06,4,np.log(100)],atol=2e-13)
    assert r<1e-14
    X,idx=design_matrix();np.testing.assert_allclose(X.T@(X@p-np.log(signal[idx])),0,atol=1e-12)

def test_intensity_invariance_and_time_units():
    signal=np.full(24,80.);signal[TAU>.015]=80*np.exp(.04-3*TAU[TAU>.015])
    p,_=fit_signals(signal);q,_=fit_signals(signal*7)
    np.testing.assert_allclose(q[:2],p[:2],atol=1e-12)
    assert abs(q[2]-p[2]-np.log(7))<1e-12
    p_ms,_=fit_signals(signal,TAU*1000,cutoff=15)
    np.testing.assert_allclose(p_ms,[p[0],p[1]/1000,p[2]],atol=1e-12)

def test_forward_and_oxygen_units():
    p=np.array([.04,3.,np.log(100)])
    pred=forward_log(p)
    assert pred[0]==p[2]
    assert abs(pred[-1]-(np.log(100)+.04-.064*3))<1e-14
    oef,dhb=oxygen_diagnostics(3.,.04)
    assert abs(oef-3/(.04*2.675e4*4/3*np.pi*.264e-6*3e4*.34))<1e-14
    assert abs(dhb/oef-HCT/KAPPA)<1e-12
    assert np.isinf(oxygen_diagnostics(3.,0.)[0])

def test_constants_and_readme():
    assert SPACING_MM==(3.75,3.75,5.)
    assert (GAMMA,B0,DELTA_CHI,HCT,KAPPA)==(26750.,30000.,.264e-6,.34,.03)
    readme=(ROOT/'README.md').read_text(encoding='utf8')
    for literal in ['3.75','5.0','74','2004','1.21','0.702','0.511','-28','64']:assert literal in readme

def test_preprocessing_and_input_errors(tmp_path):
    np.testing.assert_array_equal(log_measurements([0,1,np.e]),[0,0,1])
    with pytest.raises(ValueError):log_measurements([-1])
    with pytest.raises(ValueError):fit_signals(np.ones(4))
    with pytest.raises(ValueError):design_matrix(np.array([1.,2.]))
    p=tmp_path/'bad.npz';np.savez(p,measurements=np.zeros(24))
    with pytest.raises(ValueError):load_observations(p)

def test_mask_independent_intensity_scaling():
    v=load_observations(ROOT/'data/raw_data.npz')
    np.testing.assert_array_equal(observation_mask(v),observation_mask(v*4))

def test_metric_known_case():
    assert rmse([0,0],[3,4])==np.sqrt(12.5)
