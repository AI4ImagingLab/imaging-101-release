import numpy as np
import pytest
from src.physics_model import (mapping_derivatives, fluence_interpolator,
 transport_state, magnetic_from_velocity, magnetic_from_mapping, mapping_from_magnetic,
 M_PROTON_G, C_CMS, ESU)
from src.solvers import reconstruct, linear_poisson

def grid(nx=17,ny=14):
    return np.meshgrid(np.linspace(-.09,.09,nx),np.linspace(-.08,.08,ny),indexing='ij')

def test_identity_hessian_normal_boundary_and_uniform_zero_field():
    x,y=grid(); f=np.ones_like(x)*713
    result=reconstruct(x,y,f,max_steps=3)
    assert result['iterations']==1
    np.testing.assert_allclose(result['reconstruction'],0,atol=2e-10)
    np.testing.assert_allclose(result['det'],1,atol=2e-14)
    assert np.array_equal(result['phix'][[0,-1],:],x[[0,-1],:])
    assert np.array_equal(result['phiy'][:,[0,-1]],y[:,[0,-1]])

def test_quadratic_hessian_and_mixed_derivative_inside():
    x,y=grid();phi=.5*(1.2*x*x+.8*y*y)+.14*x*y
    gx,gy,hxx,hyy,hxy=mapping_derivatives(phi,x,y)
    np.testing.assert_allclose(hxx[2:-2,2:-2],1.2,atol=2e-13)
    np.testing.assert_allclose(hyy[2:-2,2:-2],.8,atol=2e-13)
    np.testing.assert_allclose(hxy[2:-2,2:-2],.14,atol=2e-13)
    assert not hxy[[0,-1],:].any() and not hxy[:,[0,-1]].any()

def test_bilinear_exactness_and_exterior_mean():
    x,y=grid();f=700+2*x+3*y+4*x*y;interp=fluence_interpolator(x,y,f)
    a,b=.021,-.016
    np.testing.assert_allclose(interp((a,b)),700+2*a+3*b+4*a*b,atol=1e-12)
    assert interp((2,3))==np.mean(f)

def test_lorentz_cross_product_sign_and_cgs_scale():
    impulse=np.array([[[2e7]],[[3e7]]])
    expected=np.array([[[-M_PROTON_G*C_CMS/ESU*3e7]],[[M_PROTON_G*C_CMS/ESU*2e7]]])
    assert np.array_equal(magnetic_from_velocity(impulse),expected)
    # Laboratory -Z beam cross (+Bx,+By)=(+By,-Bx).
    assert expected[0,0,0]<0 and expected[1,0,0]>0

@pytest.mark.parametrize('component',[0,1])
def test_positive_field_particle_trajectory_in_independent_world_basis(component):
    from scipy.integrate import solve_ivp
    speed=5.24e9
    field=np.zeros(3);field[component]=100.
    initial=np.array([0.,0.,-speed])
    result=solve_ivp(lambda t,v:ESU/(M_PROTON_G*C_CMS)*np.cross(v,field),
                     (0,.2/speed),initial,rtol=1e-11,atol=1e-9)
    impulse=result.y[:2,-1]-initial[:2]
    assert impulse[1]<0 if component==0 else impulse[0]>0
    estimated=magnetic_from_velocity(impulse[:,None,None])
    np.testing.assert_allclose(estimated[component,0,0],20.,rtol=3e-10)

def test_lorentz_round_trip_and_units():
    x,y=grid();b=np.stack([100*np.sin(x*4),-200*np.cos(y*3)])
    mx,my=mapping_from_magnetic(x,y,b)
    np.testing.assert_allclose(magnetic_from_mapping(x,y,mx,my),b,rtol=2e-12,atol=1e-9)

def test_fluence_intensity_scale_invariance():
    x,y=grid();f=700+80*np.exp(-((x-.02)**2+(y+.01)**2)/.001)
    a=reconstruct(x,y,f,max_steps=9);b=reconstruct(x,y,3.7*f,max_steps=9)
    np.testing.assert_allclose(a['phin'],b['phin'],rtol=1e-12,atol=2e-18)

def test_linear_uniform_case_is_zero():
    x,y=grid()
    np.testing.assert_array_equal(linear_poisson(x,y,np.ones_like(x)),np.zeros((2,*x.shape)))
