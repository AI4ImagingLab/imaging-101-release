import numpy as np
from numpy.testing import assert_allclose
from src.physics_model import harmonic_basis,temporal_basis,observables,observable_jacobian
from src.preprocessing import geodetic_to_geocentric
from src.solvers import damping_matrices,spline_overlap

def test_axial_dipole_and_radial_power():
    theta=np.array([.4,.8,1.1]); r=np.full(3,6371.2)
    loc=np.column_stack((theta,np.zeros(3),r,np.ones(3),np.zeros(3)))
    b=harmonic_basis(loc); g=-30000.
    assert_allclose(b[:,0,0]*g,-g*np.sin(theta),rtol=1e-13)
    assert_allclose(b[:,2,0]*g,-2*g*np.cos(theta),rtol=1e-13)
    loc[:,2]*=2
    assert_allclose(harmonic_basis(loc)[:,:,0],b[:,:,0]/8,rtol=1e-13)

def test_ellipsoid_independent_cartesian_geometry():
    lat=np.deg2rad(np.array([-70.,-30.,0.,45.,80.]))
    theta,r,cd,sd=geodetic_to_geocentric(lat)
    a,b=6378137.,6356752.; e=1-(b/a)**2; n=a/np.sqrt(1-e*np.sin(lat)**2)
    x=n*np.cos(lat); z=n*(1-e)*np.sin(lat)
    assert_allclose(r,np.hypot(x,z)/1000,rtol=1e-14)
    assert_allclose(np.pi/2-theta,np.arctan2(z,x),atol=1e-14)
    assert_allclose(cd*cd+sd*sd,1,atol=1e-14)

def test_frechet_independent_finite_difference():
    theta,r,cd,sd=geodetic_to_geocentric(np.deg2rad(np.array([20.,40.,60.])))
    coords=np.column_stack((theta,[.2,.5,1.],r,cd,sd)); b=harmonic_basis(coords); t=temporal_basis([0.,500.,1000.])
    c=np.zeros((43,15)); c[:,0]=-30000; c[:,1]=-1500
    kinds=np.array([4,5,6]); p=observables(c,b,t); j=observable_jacobian(b,p,kinds)
    for k in range(15):
        delta=np.zeros_like(c); delta[:,k]=.01
        fd=(observables(c+delta,b,t)-observables(c-delta,b,t))/(.02)
        assert_allclose(fd[np.arange(3),kinds],j[:,k],rtol=3e-6,atol=3e-9)

def test_spline_partition_and_damping_psd():
    assert_allclose(temporal_basis(np.linspace(-2000,2000,31)).sum(axis=1),1,atol=1e-15)
    s,t=damping_matrices(); rng=np.random.default_rng(7)
    for _ in range(4):
        x=rng.normal(size=len(s)); assert x@s@x>=0; assert x@t@x>=0
    assert spline_overlap(1,9,43,100,0)==0

def test_readme_constants_and_contract():
    from pathlib import Path
    from src.physics_model import REFERENCE_RADIUS_KM,EPOCHS,KNOTS
    text=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    assert f'| Reference radius | {REFERENCE_RADIUS_KM} |' in text
    assert '| Ellipsoid major semiaxis | 6378137 |' in text
    assert '| Ellipsoid minor semiaxis | 6356752 |' in text
    assert '(1, 5, 3, 36, 72)' in text and '(1, 43, 15)' in text
    _,r,_,_=geodetic_to_geocentric(np.array([0.]))
    assert r[0]==6378.137
    assert_allclose(EPOCHS,[-1500,-500,500,1000,1700],rtol=0,atol=0)
    assert_allclose(KNOTS,np.arange(-2300,2400,100),rtol=0,atol=0)
