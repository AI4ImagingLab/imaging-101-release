import numpy as np
from scipy.optimize import brentq
from src.physics_model import contact_force,weights,RADIUS
from src.preprocessing import contact_index,preprocess

def test_contact_and_scaling():
    x=np.linspace(1e-6,-8e-6,200); f=contact_force(x,100,0)
    assert np.all(f[x>=0]==0); assert np.all(np.diff(f)>=0)
    np.testing.assert_allclose(contact_force(x,200,0),2*f,rtol=1e-14)
    np.testing.assert_array_equal(weights(np.array([-3e-6,0,1e-6]),0),[1,0,.5])

def test_independent_exact_spherical_law():
    R=RADIUS; E=100.; nu=.5
    for delta in R*np.array([.001,.01,.1,.25,.5]):
        a=brentq(lambda a:a/2*np.log((R+a)/(R-a))-delta,1e-15,R*(1-1e-12),xtol=1e-18)
        exact=E/(1-nu**2)*((R*R+a*a)/2*np.log((R+a)/(R-a))-a*R)
        np.testing.assert_allclose(contact_force(np.array([-delta]),E,0)[0],exact,rtol=1e-4,atol=1e-20)

def test_baseline_and_tip():
    f=np.r_[np.zeros(20),np.arange(10.)]; h=np.arange(30.)
    assert contact_index(f)==21
    x,y,o=preprocess(f,h,2.)
    np.testing.assert_array_equal(y,f);np.testing.assert_array_equal(x+o,h+f/2)
