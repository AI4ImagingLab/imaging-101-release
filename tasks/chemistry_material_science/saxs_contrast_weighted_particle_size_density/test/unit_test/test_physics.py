from pathlib import Path
import sys,numpy as np
from scipy.special import spherical_jn
from scipy.integrate import quad
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.physics_model import sphere_amplitude,build_kernel,forward,weighted_objective
from src.preprocessing import radius_grid,bin_widths,estimate_background
from src.solvers import interior_point_gradient
from src.visualization import density_metrics,size_integrals
def test_sphere_independent_special_function():
    x=np.r_[0.,np.logspace(-8,4,200)];expected=np.ones_like(x);expected[1:]=3*spherical_jn(1,x[1:])/x[1:]
    np.testing.assert_allclose(sphere_amplitude(x),expected,rtol=2e-9,atol=1e-12)
def test_sphere_independent_realspace_volume_integral():
    for x in [0.,.001,.3,2.,6.]:
        amplitude=3*quad(lambda t:t*t*np.sinc(x*t/np.pi),0,1,epsabs=1e-12)[0]
        np.testing.assert_allclose(sphere_amplitude(np.array(x)),amplitude,rtol=1e-8,atol=1e-12)
def test_monodisperse_units_zeroq_and_guinier():
    r=52.;contrast=7.;fraction=.003;Vcm=4*np.pi/3*(r*1e-8)**3
    expected=contrast*1e20*fraction*Vcm
    measured=build_kernel(np.array([0.]),np.array([r]))[0,0]*contrast*fraction
    np.testing.assert_allclose(measured,expected,rtol=1e-15)
    q=1e-6;ratio=build_kernel(np.array([q]),np.array([r]))[0,0]/build_kernel(np.array([0.]),np.array([r]))[0,0]
    np.testing.assert_allclose((1-ratio)/q**2, r*r/5,rtol=1e-6)
def test_porod_invariant_independent_quadrature():
    # Integral x²F(x)² dx tends to3pi/2. Oscillatory finite-tail error is boundedO(1/L).
    edges=np.arange(0,2000*np.pi,np.pi)
    integral=sum(quad(lambda x:x*x*float(sphere_amplitude(np.array(x)))**2,a,b,epsabs=1e-10)[0] for a,b in zip(edges[:-1],edges[1:]))
    assert abs(integral-1.5*np.pi)<.001
def test_bin_mass_integral_endpoint_convention():
    r=np.array([1.,2.,4.]);w=bin_widths(r);np.testing.assert_array_equal(w,[1.,1.5,2.])
    quantities=size_integrals(np.ones(3),r,w)
    assert quantities['bin_sum_contrast']==4.5 and quantities['source_trapezoid_contrast']==3.
    assert bin_widths(np.array([1.]))[0]==1
def test_background_from_observations_and_linear_forward():
    q=np.array([.1,.5,.7,1.]);I=np.array([9.,2.,4.,6.]);assert estimate_background(q,I)==4.
    r=radius_grid(7);w=bin_widths(r);a=np.arange(7.)
    np.testing.assert_allclose(forward(3*a,2,q,r,w),3*forward(a,0,q,r,w)+2)
def test_ipg_step_quadratic_and_positivity():
    A=np.eye(3);b=np.array([2.,3.,4.]);fit=interior_point_gradient(A,b,np.ones(3)*.01)
    np.testing.assert_allclose(fit['bin_mass'],b,rtol=1e-12)
    assert (fit['bin_mass']>0).all() and fit['iterations']==1
    assert weighted_objective(fit['bin_mass'],A,b,np.ones(3))<1e-25
def test_known_metrics():
    assert density_metrics([1,2,4],[1,2,4])==dict(ncc=1.,nrmse=0.)
    m=density_metrics([2,4,8],[1,2,4]);assert abs(m['ncc']-1)<1e-15
    assert abs(m['nrmse']-np.sqrt(7)/3)<1e-15
