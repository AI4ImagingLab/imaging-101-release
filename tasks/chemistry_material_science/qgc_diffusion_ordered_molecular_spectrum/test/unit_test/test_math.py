from pathlib import Path
import numpy as np
from scipy.special import lambertw
from src.physics_model import kernel,forward
from src.solvers import approximate_lambert_exponential,entropy_l1_prox,ball_projection

def test_lambert_equation_and_source_asymptotic():
    c=np.linspace(-50,49,500)
    w=approximate_lambert_exponential(c)
    np.testing.assert_allclose(w*np.exp(w),np.exp(c),rtol=2e-14,atol=1e-25)
    c=np.geomspace(50,10000,1000);w=approximate_lambert_exponential(c)
    # Stable equation w+log(w)=c, avoiding exponential overflow.
    assert np.max(abs((w+np.log(w)-c)/c))<.003

def test_entropy_prox_optimality_and_soft_threshold():
    z=np.array([[-4.,-.5,0.,.7,2.,8.]])
    result=entropy_l1_prox(z,.1,1.)
    np.testing.assert_allclose(result-z+.9+.1*(np.log(result)+1),0,atol=.001)
    np.testing.assert_array_equal(entropy_l1_prox(z,0),np.sign(z)*np.maximum(abs(z)-1,0))

def test_ball_and_positive_attenuation():
    y=np.array([[1.,2.],[3.,4.]])
    projected=ball_projection(y+3,y,np.array([1.,2.]))
    np.testing.assert_allclose(np.linalg.norm(projected-y,axis=0),[1.,2.])
    np.testing.assert_array_equal(ball_projection(y,y,np.array([1.,2.])),y)
    b=np.array([0.,1e9,2e9]);K=kernel(b)
    np.testing.assert_array_equal(K[0],np.ones(256));assert np.all(np.diff(K,axis=0)<0)
    x=np.zeros((1,256));x[0,30]=2
    np.testing.assert_allclose(forward(x,b)[0],2*K[:,30])
