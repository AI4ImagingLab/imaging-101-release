import numpy as np
from scipy.integrate import quad
from src.physics_model import lorentzian,source_quadrature,transmission
from src.solvers import elimination,source_norm,jacobian,configuration,outputs

def test_line_normalization_empty_absorber_and_additive_depth():
    assert abs(quad(lambda v:lorentzian(.3,v),-np.inf,np.inf)[0]-1)<1e-10
    p,_,_=configuration();v=np.linspace(-4,4,100);p[8::9]=0
    np.testing.assert_allclose(transmission(v,p),1,atol=2e-14,rtol=0)
    p[8]=.1;p[17:26]=p[8:17]
    separate=transmission(v,p);p[8]=.2;p[17]=0
    np.testing.assert_allclose(transmission(v,p),separate,atol=2e-14,rtol=0)

def test_thin_absorber_continuous_convolution_and_symmetry():
    p,_,_=configuration(count=1);p[8]=1e-7;p[9]=0
    v=np.array([-1.3,-.7,0.,.7,1.3]);prediction=transmission(v,p)
    np.testing.assert_allclose(prediction,prediction[::-1],atol=2e-14,rtol=0)
    def continuous(value):
        depth=lambda z:np.pi*.098/4*p[8]*(lorentzian(p[11],z+p[10])+lorentzian(p[11],z-p[10]))
        return quad(lambda u:lorentzian(.098,u)*np.exp(-depth(value+u)),-np.inf,np.inf,epsabs=1e-12)[0]
    np.testing.assert_allclose(prediction,[continuous(x) for x in v],atol=1e-9,rtol=0)

def test_source_linear_solve_norm_and_jacobian():
    H=np.array([[4.,1.],[1.,3.]]);d=np.array([.1,.2]);g=np.array([2.,3.])
    np.testing.assert_allclose(elimination(H,d,g),np.linalg.solve(H+np.diag(d),g),rtol=1e-14)
    assert source_norm(np.array([3.,4.]))==2.5
    v=np.array([1.,2.,3.]);p=np.array([2.,3.]);fun=lambda x,z:z[0]+z[1]*x
    J,active=jacobian(fun,p,v,[],fun(v,p));np.testing.assert_allclose(J,np.stack([np.ones(3),v]),rtol=1e-9)
    np.testing.assert_array_equal(active,[0,1])

def test_canonical_component_permutation():
    p,_,_=configuration();q=p.copy();q[8:]=p[8:].reshape(4,9)[[3,1,0,2]].ravel();v=np.linspace(-4,4,512)[None]
    a=outputs([p],v);b=outputs([q],v)
    for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-14)
