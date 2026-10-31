import numpy as np
from scipy.special import spherical_jn
from src.physics_model import lorentzian,kernel_basis,predict,LOWER,UPPER
from src.solvers import profile_energy

def test_lorentzian_width_and_integral():
    g=.1;e=np.linspace(-100,100,100001)
    assert np.isclose(lorentzian(g,g)/lorentzian(0.,g),.5)
    assert abs(np.trapezoid(lorentzian(e,g),e)-1)<.001

def test_truncated_rotational_mass_is_not_unit_normalized():
    x=np.array([1.5,3.,4.4]);mass=sum((2*l+1)*spherical_jn(l,x)**2 for l in range(3))
    assert np.all(mass>0) and np.all(mass<1) and mass[-1]<.5
    np.testing.assert_allclose(sum((2*l+1)*spherical_jn(l,x)**2 for l in range(40)),1,rtol=1e-14)

def test_profile_known_interior_and_box_boundary():
    H=np.array([[1.,2.,4.,8.],[2.,3.,5.,9.]])
    for a,b in [(.3,np.array([.2,.4])),(1/67.5,np.array([.001,10.]))]:
        y=a*H+b[:,None];value,N,bfit,pred=profile_energy(H,y,np.ones_like(y))
        assert value<1e-25
        np.testing.assert_allclose(pred,y,atol=1e-13)

def test_profile_constant_degenerate_basis_remains_valid():
    H=np.ones((2,5));y=np.full_like(H,.5)
    value,N,b,pred=profile_energy(H,y,np.ones_like(H))
    assert value<1e-25 and .0225<=N<=67.5 and np.all((b>=.001)&(b<=10))

def test_energy_axis_affine_offset_cancels_only_in_differences():
    edges=np.linspace(-2,2,321);true=(edges[:-1]+edges[1:])/2;wrong=edges[:-1]+edges[1:]/2
    np.testing.assert_allclose(wrong,1.5*true-(edges[1]-edges[0])/4,atol=1e-15)
    np.testing.assert_allclose(wrong[:,None]-wrong[None,:],1.5*(true[:,None]-true[None,:]),atol=2e-15)
