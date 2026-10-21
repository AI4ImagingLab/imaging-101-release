import numpy as np
from scipy.optimize._numdiff import approx_derivative
from src.physics_model import lorentz,spectral_model,VIPA,setup,vipa_objective,arc_coordinates,fit_circle
from src.preprocessing import extract_spectrum,interpolate_time

def test_lorentz_half_maximum_and_quartic_gradient():
    p=np.array([2.,.5,7.,1.]);x=np.array([2.,1.75,2.25]);np.testing.assert_allclose(spectral_model(x,p),[8,4.5,4.5])
    y=np.array([8.3,4.1,4.8]);z=spectral_model(x,p);h=p[1]/2;delta=x-p[0];den=delta**2+h**2
    J=np.column_stack([p[2]*h*h*2*delta/den**2,p[2]*h*delta**2/den**2,h*h/den,np.ones(3)])
    analytic=2*J.T@(z-y)**3
    numeric=approx_derivative(lambda q:np.array([.5*np.sum((spectral_model(x,q)-y)**4)]),p).ravel()
    np.testing.assert_allclose(analytic,numeric,rtol=1e-6,atol=1e-7)

def test_independent_circle_and_bilinear_image():
    theta=np.linspace(.2,1.2,6);points=np.array([30+20*np.cos(theta),40+20*np.sin(theta)]).T
    center,r=fit_circle([tuple(p) for p in points]);np.testing.assert_allclose(center,[30,40],atol=1e-6);np.testing.assert_allclose(r,20,atol=1e-6)
    row,col=np.mgrid[:20,:20];image=3*row+2*col+7
    arc=np.array([[[2.2,3.4],[2.8,3.6]],[[9.1,7.3],[9.7,7.5]]])
    np.testing.assert_allclose(extract_spectrum(image,arc),np.mean(3*arc[:,:,0]+2*arc[:,:,1]+7,axis=1))
    values=np.array([[[1.,2.]],[[3.,6.]]]);np.testing.assert_allclose(interpolate_time([0,2],values,[-1,1,3]),[[[1,2]],[[2,4]],[[3,6]]])

def test_vipa_frequency_derivative_and_dimensions():
    p=np.array([2.602634e-15,-2.08e-22,-3.6e-25,15e9]);x=np.array([100.,200.,400.])
    analytic=-(p[1]+2*p[2]*x)/(p[0]+p[1]*x+p[2]*x*x)**2
    numeric=(VIPA(x+.01,p)-VIPA(x-.01,p))/.02
    np.testing.assert_allclose(analytic,numeric,rtol=1e-6);assert (analytic>0).all()
    s=setup();assert s.f0==299792458/780.24e-9;assert np.array_equal(s.calibration.shifts,[0,3.78e9,5.066e9,-5.066e9,-3.78e9,0])


def test_bound_calibration_monotonicity_and_independent_standard_residual():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    original=np.load(root/'test/fixtures/upstream_stages.npz',allow_pickle=False)
    instrument=setup()
    for key in ['1','2']:
        axis=original['calibration_frequency_'+key]
        assert (np.diff(axis,axis=-1)>0).all()
        params=original['vipa_'+key];peaks=original['calibration_peak_positions_'+key]
        for p,position in zip(params,peaks):
            actual=VIPA(position,p)-instrument.f0
            expected=instrument.calibration.shifts+p[3]*instrument.calibration.orders
            assert abs(actual-expected).max()<50e6
            assert 14e9<p[3]<16e9
