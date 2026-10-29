from pathlib import Path
import numpy as np
import pytest
from src.physics_model import features,law,weak_operator,frame_system,ALPHAS
from src.preprocessing import load_data,element_deformation
from src.solvers import positive_quadratic,solve,objective
ROOT=Path(__file__).resolve().parents[2]
def test_feature_dimension_and_nearzero_convention():
    assert len(ALPHAS)==150 and ALPHAS[np.argmin(abs(ALPHAS))]!=0
    Q,G=features(np.eye(2)[None]);assert Q.shape==(1,171) and G.shape==(1,171,4)
    np.testing.assert_allclose(Q,0,atol=1e-14)
def test_analytic_jacobian_central_difference():
    F=np.array([[[1.2,.1],[.05,.9]],[[.9,-.2],[.1,1.1]]]);Q,G=features(F)
    for k in range(4):
        e=np.zeros(4);e[k]=1e-6;e=e.reshape(2,2)
        fd=(features(F+e)[0]-features(F-e)[0])/2e-6
        np.testing.assert_allclose(fd,G[:,:,k],rtol=3e-6,atol=3e-7)
def test_objectivity_and_stress_covariance():
    F=np.array([[[1.2,.1],[.05,.9]]]);R=np.array([[.8,-.6],[.6,.8]])
    theta=np.zeros(171);theta[[5,20,79]]=[.001,.1,.4]
    W,P=law(F,theta);W2,P2=law(R@F,theta)
    np.testing.assert_allclose(W2,W,rtol=1e-12);np.testing.assert_allclose(P2,R@P,rtol=1e-12,atol=1e-12)
def test_nonpositive_determinant_rejected():
    with pytest.raises(ValueError):features(np.array([[[1,0],[0,-1]]]))
def test_identity_displacement_and_partition_unity():
    d=load_data(ROOT/'data/raw_data.npz')
    np.testing.assert_allclose(d['shape_gradients'].sum(1),0,atol=2e-12)
    np.testing.assert_allclose(element_deformation(d['triangles'],d['shape_gradients'],np.zeros((383,2))),np.tile(np.eye(2),(696,1,1)),atol=0)
    assert not d['boundary_masks'][2:].any() and not d['reactions'][:,2:].any()
def test_quadratic_active_set_reactivates():
    M=np.array([[2.,.5],[.5,1.]]);x,_=positive_quadratic(M,np.array([1.,2.]))
    np.testing.assert_allclose(x,[0,2],atol=1e-13)
    assert np.min(M@x-[1,2])>=-1e-12
def test_cost_normalization():
    theta=np.array([2.,3.]);H=np.eye(2)*2;b=np.array([1.,2.]);C=20.
    o=objective(theta,H,b,C,penalty=4)
    assert o['weak_sse']==25 and o['penalty']==20 and o['total']==45
def test_plane_stress_neohookean_pressure_elimination():
    F=np.array([[[1.4,.1],[.03,.8]]]);theta=np.zeros(171);theta[0]=.5
    J=np.linalg.det(F)[0];cof=J*np.linalg.inv(F[0]).T
    W,P=law(F,theta)
    np.testing.assert_allclose(P[0],F[0]-cof/J**3,rtol=1e-13)
    F3=np.zeros((3,3));F3[:2,:2]=F[0];F3[2,2]=1/J
    pressure=F3[2,2]**2
    fullP=F3-pressure*np.linalg.inv(F3).T
    assert abs(fullP[2,2])<1e-14
    np.testing.assert_allclose(fullP[:2,:2],P[0],rtol=1e-13)
def test_frame_normal_factor_two():
    rng=np.random.RandomState(42);A=rng.normal(size=(3,2,171));free=np.ones((3,2),bool);free[0,0]=False
    masks=np.zeros((1,3,2),bool);masks[0,0,0]=True;r=np.array([2.])
    H,b,C=frame_system(A,free,masks,r);theta=rng.normal(size=171)
    cost=np.sum((A[free]@theta)**2)+20*(A[0,0]@theta-2)**2
    np.testing.assert_allclose(.5*theta@H@theta-b@theta+C,cost,rtol=1e-13)
def test_public_table_and_no_privileged_arrays():
    import re
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        text=(ROOT/'README.md').read_text(encoding='utf8')
        assert all(f'| {k} |' in text for k in z.files)
        assert all(z[k].shape[0]==1 for k in z.files)
        assert set(z.files)=={'coordinates','triangles','displacements','reactions','boundary_masks','free_dofs','load_steps','shape_gradients','element_areas','query_gradients'}
        assert not re.search(r'github|githubusercontent|codeload|git[.]io',text,re.I)
