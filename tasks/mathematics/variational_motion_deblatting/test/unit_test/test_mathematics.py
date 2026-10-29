import numpy as np
import pytest
from src.physics_model import createDerivatives0, psfshift, psfshift_idx, proj2simplex, project2pyramid, project2pyramid_exact, lp_proximal_mapping, forward
from src.preprocessing import line_aa, load_data


def test_zero_boundary_differences_independently():
    x=np.arange(12.).reshape(3,4)
    dx,dy=createDerivatives0(x.shape)
    padded=np.pad(x,1)
    expected_x=np.zeros((4,5)); expected_y=np.zeros((4,5))
    expected_x[:3,:]=np.diff(padded[1:-1,:],axis=1)
    expected_y[:,:4]=np.diff(padded[:,1:-1],axis=0)
    np.testing.assert_array_equal((dx@x.flatten('F')).reshape(4,5,order='F'),expected_x)
    np.testing.assert_array_equal((dy@x.flatten('F')).reshape(4,5,order='F'),expected_y)


def test_discrete_adjoint_and_positive_normal():
    rng=np.random.default_rng(52);x=rng.normal(size=30);y=rng.normal(size=42)
    dx,dy=createDerivatives0((5,6))
    for d in (dx,dy):
        assert np.dot(d@x,y)==pytest.approx(np.dot(x,d.T@y),abs=1e-12)
        assert x@(d.T@d@x)>=0


def test_simplex_kkt():
    rng=np.random.default_rng(93); x=rng.normal(size=100)
    z=proj2simplex(x)
    assert z.min()>=0
    assert z.sum()==pytest.approx(1.)
    tau=(x-z)[z>0]
    np.testing.assert_allclose(tau,tau[0],atol=1e-14)
    assert np.all(x[z==0]<=tau[0]+1e-14)


def test_exact_pyramid_projection_is_feasible():
    m,f=project2pyramid_exact(np.array([-.8,.2,2.]),np.array([[.9,-.7,.4],[1.1,.8,.7],[2.,3.,-1.]]),1.)
    assert m.min()>=0 and m.max()<=1
    assert f.min()>=0
    assert np.max(f-m[:,None])<1e-8


def test_upstream_projection_bug_is_explicitly_preserved_for_parity_only():
    m,f=project2pyramid(np.array([-.8,.2,2.]),np.array([[.9,-.7,.4],[1.1,.8,.7],[2.,3.,-1.]]),1.)
    assert np.max(f-m[:,None])==pytest.approx(.0068359375)
    exact_m,exact_f=project2pyramid_exact(np.array([0.]),np.array([[1.,0.,0.]]),1.)
    np.testing.assert_allclose(exact_m,[.5])
    np.testing.assert_allclose(exact_f,[[.5,0,0]])


def test_exact_projection_matches_independent_constrained_optimizer():
    from scipy.optimize import minimize
    rng=np.random.default_rng(25)
    for _ in range(50):
        x=rng.normal(size=4)
        m,f=project2pyramid_exact(x[:1],x[None,1:],1.)
        result=minimize(lambda y:.5*np.sum((y-x)**2),np.clip(x,0,1),
            jac=lambda y:y-x,bounds=[(0,1)]+[(0,None)]*3,
            constraints=[{'type':'ineq','fun':lambda y:y[0]-y[1:],
              'jac':lambda y:np.c_[np.ones(3),-np.eye(3)]}],method='SLSQP',
            options={'ftol':1e-12,'maxiter':100})
        assert result.success
        np.testing.assert_allclose(np.r_[m,f.ravel()],result.x,atol=1e-10,rtol=1e-10)


def test_projection_kkt_normal_cone():
    rng=np.random.default_rng(3)
    for _ in range(30):
        source=rng.normal(size=4)
        m,f=project2pyramid_exact(source[:1],source[None,1:],1.)
        projected=np.r_[m,f.ravel()]
        # A Euclidean projection has nonpositive residual inner product with
        # every feasible direction from the projected point.
        for _ in range(30):
            opacity=rng.random(); feasible=np.r_[opacity,rng.random(3)*opacity]
            assert np.dot(source-projected,feasible-projected)<=1e-12


def test_shift_index_and_embedding_match():
    x=np.arange(60.).reshape(4,5,3)
    z=psfshift(x,(9,11));idx=psfshift_idx(x.shape,(9,11,3))
    np.testing.assert_array_equal(z[idx],x.flatten('F'))


def test_soft_threshold_vector_formula():
    norm=np.array([0.,.1,1.,3.])
    np.testing.assert_allclose(lp_proximal_mapping(norm,.5,1),[0,0,.5,5/6])


def test_line_horizontal_and_single_pixel():
    rr,cc,value=line_aa(2,1,2,4)
    keep=value>0
    np.testing.assert_array_equal(rr[keep],np.full(4,2))
    np.testing.assert_array_equal(cc[keep],[1,2,3,4])
    np.testing.assert_allclose(value[keep],1)
    rr,cc,value=line_aa(3,2,3,2)
    assert list(zip(rr,cc,value))==[(3,2,1.)]


def test_forward_identity_and_zero_foreground():
    rng=np.random.default_rng(5);b=rng.random((8,10,3))
    h=np.zeros((8,10));h[0,0]=1
    f=np.zeros((3,3,3));m=np.zeros((3,3))
    np.testing.assert_allclose(forward(b,h,f,m),b,atol=1e-14)
    f[1,1,:]=.4;m[1,1]=.7
    expected=b.copy();expected[0,0,:]=.4+.3*b[0,0,:]
    np.testing.assert_allclose(forward(b,h,f,m),expected,atol=1e-14)


def test_public_input_contract():
    i,b=load_data()
    assert i.shape==(169,137,3) and b.shape==i.shape
    assert 0<=i.min()<=i.max()<=1


def test_even_object_origin_matches_public_metadata():
    from pathlib import Path
    from src.preprocessing import OBJECT_SHAPE
    impulse=np.zeros(OBJECT_SHAPE);impulse[18,18]=1
    embedded=psfshift(impulse,(169,137))
    assert embedded[0,0]==1 and embedded.sum()==1
    text=(Path(__file__).resolve().parents[2]/'README.md').read_text()
    assert '| Compact-array origin | index (18,18)' in text
    assert '| Object support | 36 by 36 | pixel |' in text
