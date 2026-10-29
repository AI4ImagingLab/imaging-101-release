import numpy as np
import pytest
from src.physics_model import reference_wave,quadrature,interference,phase_score,centered_spectrum
from src.preprocessing import center_frames
from src.solvers import reconstruct,find_carrier

def test_quadrature_cancels_dc_and_twin():
    rng=np.random.default_rng(37)
    obj=rng.normal(size=(17,20))+1j*rng.normal(size=(17,20))
    ref=np.exp(1j*rng.normal(size=obj.shape))
    h=interference(obj,ref)
    np.testing.assert_allclose(quadrature(h,np.ones(obj.shape)),4*obj*np.conj(ref),rtol=1e-12,atol=1e-12)

def test_digital_carrier_sign_units_and_norm():
    shape=(17,20); fx=3.4; fy=4.7
    r=reference_wave(shape,fx,fy)
    x,y=np.meshgrid(np.arange(20)-10,np.arange(17)-8.5)
    expected=np.exp(2j*np.pi*((10-fx)*x/20+(8.5-fy)*y/17))
    np.testing.assert_allclose(r,expected,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(abs(r),1,atol=2e-16)
    np.testing.assert_allclose(r,reference_wave(shape,fx,fy,wavelength=532e-9,dx=3e-6,dy=4e-6),atol=1e-13)

def test_centering_and_amplitude_identity():
    rng=np.random.default_rng(41); h=rng.normal(size=(4,13,12))
    c=center_frames(h)
    np.testing.assert_allclose(c.mean(axis=(1,2)),0,atol=1e-15)
    result=quadrature(c,reference_wave(c.shape[1:],5.2,4.9))
    expected=np.hypot(c[0]-c[2],c[1]-c[3])
    np.testing.assert_allclose(abs(result),expected,atol=3e-15)

def test_known_phase_score_and_fft_parseval():
    field=np.exp(1j*np.array([[-2.,-1.,0.,1.,2.]]))
    assert phase_score(field)==1
    assert phase_score(np.ones((2,2)))==0
    data=np.arange(35).reshape(5,7)
    np.testing.assert_allclose(np.sum(abs(centered_spectrum(data))**2),data.size*np.sum(data**2),rtol=1e-14)

@pytest.mark.parametrize('data',[np.zeros((3,4,4)),np.full((4,4,4),np.nan)])
def test_bad_frames(data):
    with pytest.raises(ValueError): center_frames(data)

def test_bad_search():
    with pytest.raises(ValueError): reconstruct(np.zeros((4,4,4)),s=2,steps=1)
