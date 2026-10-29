import numpy as np
from src.physics_model import magnification,magnification_derivative
from src.preprocessing import flux_from_magnitude,magnitude_from_flux,load_data
from src.solvers import fit_fluxes,INITIAL
def test_known_lens_and_symmetry():
    np.testing.assert_allclose(magnification(np.array([0.]),[0,1,1]),3/np.sqrt(5),rtol=1e-15)
    assert abs(magnification(np.array([1e6]),[0,1,1])[0]-1)<1e-10
    t=np.linspace(-20,20,21);np.testing.assert_allclose(magnification(t,[0,.5,10]),magnification(-t,[0,-.5,10]),rtol=1e-15)
def test_derivative():
    u=np.array([.1,.5,1,2,10]);h=1e-5
    a=lambda u:(u*u+2)/(u*np.sqrt(u*u+4))
    np.testing.assert_allclose((a(u+h)-a(u-h))/(2*h),magnification_derivative(u),rtol=2e-8,atol=1e-10)
def test_magnitude_roundtrip():
    m=np.array([10.,13.9,22.,25.]);f,e=flux_from_magnitude(m,np.ones(4)*.0068)
    np.testing.assert_allclose(magnitude_from_flux(f),m,atol=1e-14)
    np.testing.assert_allclose(e/f,.0068*np.log(10)*.4)
def test_weighted_projection_orthogonality():
    d=load_data();f,e=flux_from_magnitude(d['magnitude'],d['magnitude_error']);a=magnification(d['time'],INITIAL);c,model=fit_fluxes(a,f,e)
    normal=np.array([a,np.ones_like(a)])@((f-model)/e**2)
    assert np.linalg.norm(normal)<1e-9
