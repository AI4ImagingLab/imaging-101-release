from pathlib import Path
import numpy as np
import pytest
from src.physics_model import modal_model,matern52,augmented_model,hellinger_squared
from src.preprocessing import load_observations,observation_indices
from src.solvers import kalman_rts
ROOT=Path(__file__).resolve().parents[2]

def test_mass_normal_modes_positive_energy_and_sensor_units():
    d=load_observations(ROOT/'data/raw_data.npz');m=modal_model(d['mass'],d['stiffness'],d['damping'],d['sensor_matrix'],.01)
    for k in ['mass','stiffness','damping']:
        np.testing.assert_allclose(d[k],d[k].T,atol=1e-12);assert np.linalg.eigvalsh(d[k]).min()>0
    P=m['Phi_r'];np.testing.assert_allclose(P.T@d['mass']@P,np.eye(3),atol=1e-13)
    np.testing.assert_allclose(d['stiffness']@P,d['mass']@P@m['Omega2_r'],atol=1e-10)
    assert np.flatnonzero(d['sensor_matrix'].sum(0)).tolist()==[0,2,4,6,8]
    assert np.array_equal(observation_indices(d['observation_time'],d['reconstruction_time']),np.arange(0,100001,5))

def test_matern_sde_stationary_covariance_independent_closed_form():
    alpha=2.;length=.7;k=matern52([alpha,length],.01);rate=np.sqrt(5)/length
    expected=alpha**2*np.array([[1,0,-rate**2/3],[0,rate**2/3,0],[-rate**2/3,0,rate**4]])
    np.testing.assert_allclose(k['P'],expected,rtol=1e-12,atol=1e-10)
    residual=k['F_c']@k['P']+k['P']@k['F_c'].T+k['L_c']*k['q']@k['L_c'].T
    np.testing.assert_allclose(residual,0,atol=1e-9)
    assert np.linalg.eigvalsh((k['Q']+k['Q'].T)/2).min()>-1e-10

def test_augmented_lyapunov_and_observation_adjoint():
    d=load_observations(ROOT/'data/raw_data.npz');m=modal_model(d['mass'],d['stiffness'],d['damping'],d['sensor_matrix'],.01);a=augmented_model(m,[12.,2.5])
    r=a['F_ac']@a['P_0']+a['P_0']@a['F_ac'].T+a['Q_ac'];assert np.linalg.norm(r)/np.linalg.norm(a['Q_ac'])<1e-10
    rng=np.random.default_rng(9);x=rng.normal(size=15);y=rng.normal(size=5)
    assert (a['H_ad']@x)@y==pytest.approx(x@(a['H_ad'].T@y),abs=1e-14)

def test_hellinger_quadrature_independent_analytic_limit():
    r=np.array([1.,2.]);s=np.array([1.2,.7]);exact=np.mean(1-np.sqrt(2*r*s/(r*r+s*s)))
    assert hellinger_squared(r,s)==pytest.approx(exact,abs=1e-6)

def test_scalar_kf_rts_against_joint_gaussian_conditioning():
    C=np.array([[1.,1.,1.],[1.,1.1,1.1],[1.,1.1,1.2]])
    cross=C[:,[1,2]];S=C[np.ix_([1,2],[1,2])]+np.eye(2)*.2
    mean=cross@np.linalg.inv(S)@np.array([1.,2.]);cov=C-cross@np.linalg.inv(S)@cross.T
    result,info=kalman_rts(np.ones((1,1)),np.array([[.1]]),np.ones((1,1)),np.array([[.2]]),np.array([[1.],[2.]]),np.array([1,2]),3,np.ones((1,1)),{'x':np.ones((1,1))})
    np.testing.assert_allclose(result[:,0],mean,atol=1e-14)
    np.testing.assert_allclose(info['variance']['x'][:,0],np.diag(cov),atol=1e-14)
