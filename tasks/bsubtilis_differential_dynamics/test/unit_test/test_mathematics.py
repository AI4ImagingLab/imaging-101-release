import numpy as np
from src.preprocessing import image_structure,radial_average,resample_loglog,logspace_int,PIXEL_SIZE_UM,FRAME_RATE_HZ
from src.physics_model import structure_model,model_jacobian,intermediate_scattering
from src.solvers import fit_curve,fit_speed

def test_finite_overlap_structure_independent_time_differences():
    rng=np.random.default_rng(36);movie=rng.normal(size=(19,12,14))
    packed=image_structure(movie,chunk_size=7)
    spectrum=np.fft.fftshift(np.fft.rfft2(movie)/np.sqrt(168),axes=-2)
    for lag in (1,2,7,18):np.testing.assert_allclose(packed[lag-1],np.mean(abs(spectrum[lag:]-spectrum[:-lag])**2,axis=0),atol=1e-14)
    np.testing.assert_allclose(packed[-2],np.mean(abs(spectrum)**2,axis=0),atol=1e-14)
    np.testing.assert_allclose(packed[-1],np.var(spectrum,axis=0),atol=1e-14)

def test_constant_movie_zero_dynamics_and_plane_wave_scale():
    x=np.arange(16);field=np.cos(2*np.pi*x/16)[None,:]*np.ones((12,1))
    movie=np.repeat(field[None],20,axis=0);p=image_structure(movie)
    np.testing.assert_allclose(p[:-2],0,atol=1e-12);np.testing.assert_allclose(p[-1],0,atol=1e-12)
    assert np.isclose(p[-2,6,1],12*16/4)

def test_model_jacobian_and_limits():
    t=np.geomspace(.001,2,70);p=np.array([24.,3.,7.,1.4]);jac=model_jacobian(t,p)
    for j in range(4):
        d=np.zeros(4);d[j]=1e-5;np.testing.assert_allclose(jac[:,j],(structure_model(t,p+d)-structure_model(t,p-d))/2e-5,rtol=2e-7,atol=2e-8)
    assert structure_model(np.array([0.]),p)[0]==3
    assert np.isclose(structure_model(np.array([1e6]),p)[0],27)
    assert np.all(np.diff(intermediate_scattering(t,p[2],p[3]))<0)

def test_known_curve_inverse_and_speed():
    t=np.geomspace(.001,2,100);p=np.array([100.,2.,12.,1.35])
    fit,_,_=fit_curve(t,structure_model(t,p),[95.,1.,10.,1.2]);np.testing.assert_allclose(fit,p,rtol=1e-8)
    q=np.linspace(.75,4,39);assert abs(fit_speed(q,17*q)-17)<1e-8

def test_metadata_and_integer_lags():
    assert PIXEL_SIZE_UM==75.59/128 and FRAME_RATE_HZ==100
    lag=logspace_int(999);assert len(lag)==200 and np.all(np.diff(lag)>0) and lag[0]==1


def test_isotropic_distribution_gaussian_analytic_limit():
    from src.physics_model import isotropic_speed_density
    n=np.array([0.,.5,1.,2.,4.])
    expected=n*n/(2*np.sqrt(np.pi))*np.exp(-n*n/4)
    np.testing.assert_allclose(isotropic_speed_density(n,beta=2.),expected,atol=1e-10)
