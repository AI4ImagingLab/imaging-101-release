import numpy as np
from src.physics_model import depth_transform,windowed_spectrum,calibration
from src.preprocessing import subtract_background
def test_single_spectral_tone_depth_and_parseval():
    n=np.arange(800);s=np.exp(2j*np.pi*100*n/2048)[None,:,None]
    y=depth_transform(s,np.ones((800,1)))
    assert np.argmax(abs(y[0,:,0]))==2047-(100+1024)%2048
    assert np.isclose(np.linalg.norm(y)**2,np.linalg.norm(s)**2*2048e-12,rtol=1e-6)
    np.testing.assert_allclose(windowed_spectrum(y)[:,:800],s,rtol=2e-6,atol=2e-6)
def test_background_convention_and_linearity():
    x=np.full((2,1600,4),10,dtype=np.uint16);b=np.full((2,1600),2.5)
    assert np.all(subtract_background(x,b)==7)
    assert subtract_background(x,b).dtype==np.int64
def test_calibration_endpoints():
    indices,d,w=calibration(np.linspace(0,1,800),np.r_[np.ones(800),np.zeros(800)])
    assert indices[0]==0 and indices[-1]==3198
    np.testing.assert_array_equal(d,np.ones(800))
    assert w[0]==0 and w[-1]==0
