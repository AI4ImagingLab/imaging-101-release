import numpy as np
from scipy.integrate import quad
from src.physics_model import blocked,eccentric_anomaly,orbit,P,T0,disk_mean
from src.solvers import ensemble,valid

def test_kepler_equation_and_circular_limit():
    m=np.linspace(0,2*np.pi,201,endpoint=False)
    for e in [0,.2,.7,.95]:
        E=eccentric_anomaly(m,e)
        assert max(abs(E-e*np.sin(E)-m))<1e-12
    np.testing.assert_allclose(eccentric_anomaly(m,0),m,atol=1e-14)

def test_extreme_eccentricity_monotone_root_robustness():
    m=np.r_[np.geomspace(1e-15,.1,200),np.linspace(.1,6.28,2000)]
    for e in [.999,.999999,.999999999]:
        E=eccentric_anomaly(m,e)
        assert np.max(abs(E-e*np.sin(E)-m))<1e-12
        assert np.all(np.diff(E)>=0)

def test_kepler_endpoints_and_periodicity():
    phase=np.array([0.,.001,.7,2.,5.9])
    for e in [0.,.3,.999999]:
        base=eccentric_anomaly(phase,e)
        assert abs(eccentric_anomaly(0.,e))<1e-14
        assert abs(eccentric_anomaly(2*np.pi,e))<1e-14
        for k in [-3,-1,1,4]:
            got=eccentric_anomaly(phase+2*np.pi*k,e)
            # At wrapped endpoint, equivalent zero and 2pi are the same root.
            difference=np.arctan2(np.sin(got-base),np.cos(got-base))
            assert max(abs(difference))<1e-9

def test_uniform_disk_exact_overlap_and_bounds():
    R=1.;r=.15;d=np.linspace(0,1.3,201);got=blocked(R,r,d,(0.,0.))
    expected=np.zeros(len(d));expected[d<=R-r]=r*r
    mask=(d>R-r)&(d<R+r);x=d[mask]
    expected[mask]=(np.arccos((x*x+1-r*r)/(2*x))+r*r*np.arccos((x*x+r*r-1)/(2*x*r))-.5*np.sqrt((-x+1+r)*(x+1-r)*(x-1+r)*(x+1+r)))/np.pi
    np.testing.assert_allclose(got,expected,rtol=0,atol=1e-14)
    assert np.all((got>=0)&(got<=1))
    assert blocked(.1,1.,np.array([0.]),(.3,.2))[0]==1

def test_limb_darkening_independent_radial_integral():
    # Broad selected grid approximation, checked against adaptive continuum
    # integration rather than claiming the coarse quadrature is exact physics.
    R=1.;r=.15;d=.9;u=.3532;v=.2606
    def fun(rho):
        theta=2*np.arccos(np.clip((rho*rho+d*d-r*r)/(2*rho*d),-1,1)) if rho else 0
        mu=np.sqrt(max(0,1-rho*rho));return rho*theta*(1-u*(1-mu)-v*(1-mu)**2)
    value=quad(fun,0,1,points=[d-r],epsabs=1e-12)[0]/(np.pi*(1-u/3-v/6))
    assert abs(blocked(R,r,np.array([d]),(u,v))[0]-value)<4e-5
    assert abs(disk_mean(1,(u,v))-(1-u/3-v/6))<.0007

def test_velocity_scale_and_inclination_mirror_convention():
    p=np.array([.04,.006,90,0,0,.3,12.]);times=T0+np.linspace(0,P,501)
    rv=orbit(times,p,True)[2]
    assert abs((rv.max()-rv.min())/2-12*.019771142/.019765685498705825)<.001
    assert valid(p) and not valid(p+np.array([0,0,1,0,0,0,0]))

def test_affine_sampler_gaussian_statistics():
    center=np.array([.05,.005,89,-.4,.1,.3,15]);scale=np.full(7,1e-4)
    logp=lambda x:-.5*np.sum(((x-center)/scale)**2)
    chain,acceptance=ensemble(None,center,np.diag(scale**2),np.random.default_rng(),walkers=60,updates=700,probability=logp)
    z=(chain[:,350:]-center)/scale
    assert chain.shape==(60,700,7) and np.isfinite(chain).all()
    assert np.max(abs(z.mean(axis=(0,1))))<.16
    assert np.all((z.var(axis=(0,1))>.70)&(z.var(axis=(0,1))<1.3))
    assert .2<acceptance.mean()<.7
