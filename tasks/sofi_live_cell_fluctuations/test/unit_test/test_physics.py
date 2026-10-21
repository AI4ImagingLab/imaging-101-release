import numpy as np
from src.preprocessing import temporal_mean
from src.physics_model import moments,cumulants,fluorescence_image

def test_exact_symmetric_bernoulli_cumulants():
    x=np.array([0,2,0,2],dtype=np.uint16)[:,None,None]
    c=cumulants(moments(x,temporal_mean(x)))
    assert c[2].item()==1 and c[4].item()==-2

def test_independent_emitters_cumulants_add():
    blink=np.array([[0,0],[0,1],[1,0],[1,1]])
    psfs=np.array([[[2.]],[[4.]]]);x=fluorescence_image(psfs,blink)
    c=cumulants(moments(x,temporal_mean(x)))
    assert c[2].item()==5 and c[4].item()==-34

def test_camera_offset_invariance():
    x=np.arange(40,dtype=np.uint16).reshape(10,2,2)
    a=cumulants(moments(x,temporal_mean(x)));b=cumulants(moments(x+100,temporal_mean(x+100)))
    for k in a:np.testing.assert_array_equal(a[k],b[k])

def test_gaussian_psf_power_narrows_by_sqrt_order():
    x=np.linspace(-10,10,10001);h=np.exp(-x*x/2)
    sigma=lambda p:np.sqrt(np.sum(x*x*p)/np.sum(p))
    assert abs(sigma(h**4)-sigma(h)/2)<1e-10
