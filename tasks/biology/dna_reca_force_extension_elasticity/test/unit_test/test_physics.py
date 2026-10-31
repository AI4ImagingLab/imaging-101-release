import numpy as np
from src.physics_model import ewlc_marko_siggia_force as force,ewlc_marko_siggia_force_jac as jac,ewlc_marko_siggia_force_derivative as stiffness
def test_zero_extension_and_positive_stiffness():
    d=np.linspace(0,2.8,50);q=force(d,60.,2.75,1000.,4.11)
    assert abs(q[0])<1e-9
    assert np.all(q[1:]>0) and np.all(stiffness(d,60.,2.75,1000.,4.11)>0)
def test_independent_implicit_equation():
    d=np.linspace(.01,2.8,70);q=force(d,60.,2.75,1000.,4.11)
    u=d/2.75-q/1000
    expected=.25/(1-u)**2-.25+u
    np.testing.assert_allclose(q*60/4.11,expected,rtol=2e-8,atol=2e-8)
def test_jacobian_finite_difference():
    x=np.array([1.6,2.3,2.6,2.72]);p=np.array([60.,2.75,1000.,4.11]);j=np.array(jac(x,*p))
    for k in range(4):
        h=1e-5*abs(p[k]);a=p.copy();b=p.copy();a[k]+=h;b[k]-=h
        fd=(force(x,*a)-force(x,*b))/(2*h)
        np.testing.assert_allclose(j[k],fd,rtol=2e-4,atol=3e-7)
