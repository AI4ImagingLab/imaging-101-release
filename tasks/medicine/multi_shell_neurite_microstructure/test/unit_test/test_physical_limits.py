from pathlib import Path
import numpy as np
import pytest
from src.physics_model import spherical_mean,predicted_signal,transform_parameters,signal_objective
from src.preprocessing import load_data
from src.solvers import fit_signals,_sort_three,_sort_last
ROOT=Path(__file__).resolve().parents[2]

def test_b0_isotropic_and_compartment_limits():
    b=np.array([0.,200.,1000.,3000.]);D=.0015
    assert np.allclose(spherical_mean(b,D,D),np.exp(-b*D))
    assert np.all(spherical_mean(0,[0,D],[0,0])==1)
    assert np.allclose(predicted_signal(np.array([[0.,D]]),b,np.array([250.])),250*np.exp(-b*D))
    assert np.allclose(predicted_signal(np.array([[1.,D]]),b,np.array([250.])),250*spherical_mean(b,D,0))

def test_logit_transform_and_overflow_guard():
    x=transform_parameters(np.array([[0.,0],[1000,1000],[-1000,-1000]]))
    assert np.isfinite(x).all()
    assert np.allclose(x,[[.5,.001525],[1,.00305],[0,0]])

def test_explicit_source_tie_behavior():
    values=np.ones((1,3));indices=np.array([[0,1,2]])
    assert np.array_equal(_sort_three(values,indices),[[2,1,0]])
    assert np.array_equal(_sort_last(values,indices),indices)

def test_directional_objective_equals_weighted_mean_plus_constant():
    y=np.array([[100.,70,60,50,40],[200,100,120,80,70.]])
    b=np.array([0,1000,1000,2000,2000]);z=np.array([[.2,-.1],[-.3,.5]])
    means=np.stack([y[:,0],y[:,1:3].mean(1),y[:,3:5].mean(1)],axis=1)
    observed=signal_objective(z,y,b)
    mean_loss=2*signal_objective(z,means,np.array([0,1000,2000]))
    constant=((y[:,1:3]-means[:,1,None])**2).sum(1)+((y[:,3:5]-means[:,2,None])**2).sum(1)
    np.testing.assert_allclose(observed,mean_loss+constant,rtol=1e-14,atol=1e-10)

def test_batch_members_independent_and_real_bound_data():
    with np.load(ROOT/'test/fixtures/original_stages.npz') as f:y=f['observed'][:4];b=f['bvalues']
    together,diag=fit_signals(y,b)
    individual=np.concatenate([fit_signals(y[k:k+1],b)[0] for k in range(4)])
    np.testing.assert_array_equal(together,individual)
    assert diag['converged'].all()
    data=load_data(ROOT/'data/raw_data.npz')
    assert data['observation_mask'].sum()==44427
    assert np.array_equal(np.unique(data['bvalues'],return_counts=True)[1],np.r_[1,np.full(15,33)])

def test_invalid_signal_rejected():
    with pytest.raises(ValueError):fit_signals(np.array([[1.,np.nan]]),np.array([0,1000]))

def test_isotropic_null_recovers_known_noiseless_diffusivity():
    from src.solvers import fit_isotropic_signals
    b=np.arange(0,3001,200,dtype=float);d=np.array([.0004,.0013,.0028]);s0=np.array([100.,200.,300.])
    y=s0[:,None]*np.exp(-d[:,None]*b)
    fitted,objective=fit_isotropic_signals(y,b)
    np.testing.assert_allclose(fitted,d,atol=1e-11,rtol=0)
    assert np.all(objective<1e-12)
