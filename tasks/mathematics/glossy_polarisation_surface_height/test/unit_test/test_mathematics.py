import numpy as np
import pytest
from src.preprocessing import decompose
from src.physics_model import diffuse_zenith,assemble_height_system
from src.visualization import metrics


def test_known_sinusoid_and_angles():
    a=np.arange(36)*np.pi/36
    phase=.42;mean=.7;rho=.12
    observations=mean*(1+rho*np.cos(2*(a-phase)))
    r,p,i=decompose(observations[None,None],a,np.ones((1,1),dtype=bool))
    np.testing.assert_allclose([r.item(),p.item(),i.item()],[rho,phase,mean],atol=2e-15)


def test_independent_diffuse_fresnel_round_trip():
    theta=np.linspace(0,1.45,30);n=1.5
    rho=(n-1/n)**2*np.sin(theta)**2/(2+2*n*n-(n+1/n)**2*np.sin(theta)**2+4*np.cos(theta)*np.sqrt(n*n-np.sin(theta)**2))
    np.testing.assert_allclose(diffuse_zenith(rho),theta,atol=3e-14)


def test_flat_surface_boundary_and_adjoint():
    mask=np.ones((9,7),dtype=bool);mask[0,0]=False
    zero=np.zeros(mask.shape);light=np.array([.1,.05,.3])
    C,d=assemble_height_system(zero,np.full(mask.shape,.3),zero,light,mask,np.zeros_like(mask))
    np.testing.assert_allclose(d,0,atol=1e-15)
    rng=np.random.default_rng(3);x=rng.normal(size=C.shape[1]);y=rng.normal(size=C.shape[0])
    np.testing.assert_allclose((C@x)@y,x@(C.T@y),atol=3e-14)
    direction=rng.normal(size=x.shape);eps=1e-6
    loss=lambda z: .5*np.linalg.norm(C@z-d)**2
    finite=(loss(x+eps*direction)-loss(x-eps*direction))/(2*eps)
    exact=direction@(C.T@(C@x-d))
    np.testing.assert_allclose(finite,exact,rtol=1e-8,atol=1e-8)


def test_metrics_known_case_and_degenerate():
    b=np.array([0.,1.,2.]);a=np.array([0.,2.,2.]);m=np.ones(3,dtype=bool)
    r=metrics(a,b,m)
    np.testing.assert_allclose(r['ncc'],6/np.sqrt(40))
    np.testing.assert_allclose(r['nrmse'],np.sqrt(1/3)/2)
    with pytest.raises(ValueError):metrics(a,np.ones(3),m)


def test_affine_derivative_stencils_and_axis_signs():
    mask=np.ones((9,7),dtype=bool);mask[0,0]=False
    zero=np.zeros(mask.shape);light=np.array([.1,.05,.3])
    C,d=assemble_height_system(zero,np.full(mask.shape,.3),zero,light,mask,np.zeros_like(mask))
    row,col=np.indices(mask.shape);height=3*col-2*row
    projected=C@height[mask]
    expected=[]
    padded=np.pad(mask,1)
    for r,c in np.argwhere(padded):
        if padded[r,c-1] and padded[r,c+1] and padded[r-1,c] and padded[r+1,c]:expected.append(0.)
        expected.extend([3.,-.4])
    expected.append(float(height[mask][0]))
    np.testing.assert_allclose(projected,expected,atol=5e-15)


def test_published_halfway_approximation_scale_dependence_is_explicit():
    # Independent optics check of a documented approximation, not a truth claim.
    direction=np.array([2.,0.,7.]);view=np.array([0.,0.,1.])
    scaled=.04152569*direction
    physical=(direction/np.linalg.norm(direction)+view)
    published=scaled+view
    physical_slope=-physical[0]/physical[2]
    published_slope=-published[0]/published[2]
    assert abs(physical_slope-published_slope)>.07
    normalized_again=scaled/np.linalg.norm(scaled)+view
    np.testing.assert_allclose(physical_slope,-normalized_again[0]/normalized_again[2],atol=1e-15)


def test_readme_calibration_and_no_backend_side_effects():
    from pathlib import Path
    from src.physics_model import REFRACTIVE_INDEX,LIGHT_DIRECTION
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before
    text=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf-8')
    assert '| Refractive index | 1.5 |' in text
    assert '| Light direction | (2, 0, 7) |' in text
    assert REFRACTIVE_INDEX==1.5 and np.array_equal(LIGHT_DIRECTION,[2,0,7])
