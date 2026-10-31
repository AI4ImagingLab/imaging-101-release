import numpy as np
from src.physics_model import forward,adjoint,difference,difference_adjoint,chronological_cube
from src.solvers import tv_denoise

def test_explicit_known_coded_exposure_and_adjoint():
    from pathlib import Path
    from src.physics_model import SUBFRAMES,BURSTS
    text=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    assert f'| Compression per exposure | {SUBFRAMES} |' in text
    assert f'| Recorded exposures | {BURSTS} |' in text
    mask=np.array([[[1.,2.],[0.,1.]],[[.5,0.],[2.,1.]]]);x=np.array([[[2.,1.],[3.,4.]],[[4.,5.],[6.,7.]]]);y=np.array([[3.,2.],[1.,-1.]])
    np.testing.assert_array_equal(forward(x,mask),[[4.,2.],[12.,11.]])
    assert np.vdot(forward(x,mask),y)==np.vdot(x,adjoint(y,mask))
    np.testing.assert_allclose(forward(adjoint(y,mask),mask),np.sum(mask*mask,axis=0)*y)

def test_finite_boundary_difference_transposes_and_variation_gradient():
    rng=np.random.default_rng(2);x=rng.normal(size=(3,4,5));direction=rng.normal(size=x.shape)
    for axis in [1,2]:
        d=difference(x,axis);z=rng.normal(size=d.shape)
        np.testing.assert_allclose(np.vdot(d,z),np.vdot(x,difference_adjoint(z,axis)),rtol=1e-14)
        gradient=difference_adjoint(d,axis);h=1e-5
        finite=(np.sum(difference(x+h*direction,axis)**2)-np.sum(difference(x-h*direction,axis)**2))/(4*h)
        np.testing.assert_allclose(finite,np.sum(gradient*direction),rtol=1e-9)

def test_tv_constant_identity_and_predual_return_convention():
    x=np.ones((2,4,5))*3
    np.testing.assert_array_equal(tv_denoise(x),x)
    impulse=np.zeros_like(x);impulse[:,1,2]=1
    np.testing.assert_array_equal(tv_denoise(impulse,iterations=1),impulse)
    assert np.linalg.norm(tv_denoise(impulse)-impulse)>0

def test_alternating_mask_sweep_is_an_exact_involution():
    a=np.arange(14*2*3).reshape(14,2,3)
    np.testing.assert_array_equal(chronological_cube(a,0),a)
    np.testing.assert_array_equal(chronological_cube(a,1),a[::-1])
    np.testing.assert_array_equal(chronological_cube(chronological_cube(a,1),1),a)
