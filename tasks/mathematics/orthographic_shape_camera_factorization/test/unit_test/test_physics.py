import numpy as np
import pytest
from pathlib import Path
from src.physics_model import symmetric_products,metric_system,forward,joint_alignment
from src.preprocessing import centre_and_scale,load_observations
from src.solvers import reconstruct
from src.visualization import metrics

def test_symmetric_metric_equations():
    C=np.array([[4.,1.,2.],[1.,3.,.4],[2.,.4,5.]])
    v=C[np.triu_indices(3)];a=np.array([.3,.5,.2]);b=np.array([.2,-.3,.7])
    assert symmetric_products(a,b)@v==pytest.approx(a@C@b)

def test_rejected_upper_cholesky_orientation_independently():
    C=np.array([[4.,1.,2.],[1.,3.,.4],[2.,.4,5.]])
    upper=np.linalg.cholesky(C).T
    np.testing.assert_allclose(upper.T@upper,C,atol=1e-14)
    assert np.linalg.norm(upper@upper.T-C)>1

def test_projection_joint_reflection_gauge_and_units():
    rng=np.random.default_rng(7);shape=rng.normal(size=(8,3));cam=rng.normal(size=(5,2,3));translation=rng.normal(size=(5,2))
    Q=np.linalg.qr(rng.normal(size=(3,3)))[0];Q[:,0]*=-1
    np.testing.assert_allclose(forward(shape@Q,cam@Q,translation),forward(shape,cam,translation),atol=1e-14)
    a,c,_=joint_alignment(shape@Q,cam@Q,shape);np.testing.assert_allclose(a,shape,atol=1e-14);np.testing.assert_allclose(c,cam,atol=1e-14)
    # Scaling only shape is physically observable for unit camera rows.
    assert np.linalg.norm(forward(shape*2,cam,translation)-forward(shape,cam,translation))>1

def test_rank3_optimum_and_metric_psd():
    root=Path(__file__).resolve().parents[2];tracks=load_observations(root/'data/raw_data.npz');s,c,t,d=reconstruct(tracks)
    assert np.linalg.eigvalsh(d['C']).min()>0
    residual=forward(s,c,t)-tracks
    assert np.sqrt(np.mean(residual**2))==pytest.approx(np.sqrt(np.sum(d['singular_values'][3:]**2)/tracks.size)*d['scale'],rel=1e-12)
    assert np.linalg.norm(s.mean(0))<1e-9
    norms=np.linalg.norm(c,axis=2);assert np.max(np.abs(norms-1))<.013
    assert np.max(np.abs(np.sum(c[:,0]*c[:,1],axis=1)))<.016

def test_forward_shape_adjoint_and_gradient():
    rng=np.random.default_rng(6);S=rng.normal(size=(7,3));C=rng.normal(size=(4,2,3));v=rng.normal(size=(4,7,2));d=rng.normal(size=S.shape)
    F=forward(S,C,np.zeros((4,2)));adj=np.einsum('fkc,fpk->pc',C,v)
    assert np.sum(F*v)==pytest.approx(np.sum(S*adj),abs=1e-13)
    h=1e-5;loss=lambda x:np.sum(forward(x,C,np.zeros((4,2)))**2)/2
    grad=np.einsum('fkc,fpk->pc',C,F)
    assert (loss(S+h*d)-loss(S-h*d))/(2*h)==pytest.approx(np.sum(grad*d),rel=1e-9)

def test_metrics_known_numbers():
    m=metrics(np.array([2.,4.]),np.array([1.,2.]));assert m['ncc']==pytest.approx(1);assert m['nrmse']==pytest.approx(np.sqrt(2.5))
