import numpy as np
from src.physics_model import forward,adjoint,poisson_prox,poisson_loss
from src.preprocessing import load_data,detector_units,window,PIXEL_PITCH_M
from src.solvers import solve_window,group_shrink
from src.visualization import agreement

def test_adjoint():
    _,m=load_data();rng=np.random.default_rng(42);x=rng.normal(size=(8,13));z=rng.normal(size=(8,24))
    np.testing.assert_allclose(np.sum(forward(x,m)*z),np.sum(x*adjoint(z,m)),rtol=1e-12)

def test_poisson_prox_stationarity_and_gradient():
    z=np.array([.4,2.,5.]);y=np.array([1.,4.,2.]);mu=.7;v=poisson_prox(z,y,mu)
    np.testing.assert_allclose(mu*(v-z)+1-y/v,0,atol=1e-12)
    h=1e-6;direction=np.array([.1,-.3,.2]);numeric=(poisson_loss(v+h*direction,y)-poisson_loss(v-h*direction,y))/(2*h)
    np.testing.assert_allclose(numeric,np.dot(1-y/v,direction),atol=1e-8)

def test_units_boundaries_and_zero():
    y,m=load_data();t=detector_units(y);assert t.dtype==np.float64
    np.testing.assert_array_equal(t,y/255.)
    np.testing.assert_array_equal(window(t,14,14)[:,4],t[14,14])
    assert PIXEL_PITCH_M==9.9521491860627815e-8
    u,a,d=solve_window(np.zeros((24,9)),m);assert not u.any() and not a.any()
    x=np.array([[3.,4.]]);np.testing.assert_allclose(group_shrink(x,1.),x*.96)

def test_known_metrics():
    x=np.array([0.,1.,2.]);r=np.array([0.,2.,4.]);n,e=agreement(x,r)
    assert abs(n-1)<1e-14
    np.testing.assert_allclose(e,np.sqrt(5/3)/4)

def test_calibration_rank_and_readme():
    _,m=load_data();assert np.linalg.matrix_rank(m)==13;assert np.all(m>0)
    from pathlib import Path
    text=Path('README.md').read_text(encoding='utf-8');assert '9.9521491860627815e-8' in text and '255' in text
