from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.physics_model import matting_laplacian,mixing_model
from src.solvers import conjugate_gradient
from src.preprocessing import load_data,partition_trimap


def test_mixing_and_gauge_nonidentifiability():
    a=np.array([[0.,.25,1.]]);f=np.full((1,3,3),.8);b=np.full((1,3,3),.2)
    expected=np.repeat(np.array([[[.2],[.35],[.8]]]),3,axis=2)
    np.testing.assert_allclose(mixing_model(a,f,b),expected,atol=1e-15)
    # A uniform observed RGB cannot determine opacity without independent priors.
    np.testing.assert_array_equal(mixing_model(a,f,f),mixing_model(1-a,f,f))


def test_independent_schur_elimination_and_positive_semidefiniteness():
    rng=np.random.default_rng(118);image=rng.uniform(size=(3,3,3));epsilon=1e-3
    design=np.c_[image.reshape(-1,3),np.ones(9)]
    penalty=np.diag([epsilon,epsilon,epsilon,0.])
    independent=np.eye(9)-design@np.linalg.solve(design.T@design+penalty,design.T)
    actual=matting_laplacian(image,epsilon=epsilon).toarray()
    np.testing.assert_allclose(actual,independent,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(actual,actual.T,atol=2e-14)
    assert np.linalg.eigvalsh(actual).min()>-1e-12
    np.testing.assert_allclose(actual@np.ones(9),0,atol=1e-12)


def test_quadratic_gradient_independent_finite_differences():
    rng=np.random.default_rng(120);image=rng.uniform(size=(6,7,3));L=matting_laplacian(image,epsilon=1e-4)
    a=rng.normal(size=42);direction=rng.normal(size=42);step=1e-5
    numerical=((a+step*direction)@(L@(a+step*direction))-(a-step*direction)@(L@(a-step*direction)))/(2*step)
    np.testing.assert_allclose(numerical,direction@(2*(L@a)),rtol=1e-8,atol=1e-8)


def test_cg_known_linear_system_and_zero_rhs():
    A=csr_matrix([[3.,1.],[1.,2.]]);x,h,_=conjugate_gradient(A,np.array([4.,3.]))
    np.testing.assert_allclose(x,[1,1],atol=1e-14);assert len(h)==2
    x,h,_=conjugate_gradient(A,np.zeros(2));assert np.all(x==0) and h.shape==(0,3)


def test_observation_metadata_and_authored_trimap():
    image,trimap=load_data();assert image.shape==(440,680,3) and trimap.shape==(440,680)
    fg,bg,known,unknown=partition_trimap(trimap)
    assert unknown.sum()==38666 and np.all((fg|bg)==known) and np.all(known!=unknown)
    # Gray .4 is an independent unknown label, not an opacity measurement.
    assert np.all(trimap.ravel()[unknown]==.4)
    readme=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    assert '| Native rows | 440 |' in readme and '| Native columns | 680 |' in readme


@pytest.mark.parametrize('problem',['small','channels','dtype','nan','epsilon','radius'])
def test_invalid_physics_inputs(problem):
    a=np.ones((4,5,3),dtype='float64');kw={}
    if problem=='small':a=a[:2]
    if problem=='channels':a=a[:,:,:2]
    if problem=='dtype':a=a.astype('float32')
    if problem=='nan':a[0,0,0]=np.nan
    if problem=='epsilon':kw['epsilon']=0
    if problem=='radius':kw['radius']=2
    with pytest.raises(ValueError):matting_laplacian(a,**kw)
