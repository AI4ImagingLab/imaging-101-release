import numpy as np
from src.physics_model import slab_transfer,slab_inverse,C
from src.preprocessing import common_window
from src.solvers import transform
def test_analytic_slab_roundtrip_and_units():
    f=np.linspace(.2e12,2.5e12,31);n=np.full(31,1.5);alpha=np.linspace(0,25,31);d=.002
    a,p=slab_transfer(n,alpha,f,d);ni,ai=slab_inverse(a,p,f,d)
    np.testing.assert_allclose(ni,n,rtol=1e-12);np.testing.assert_allclose(ai,alpha,atol=1e-12)
    assert abs(a[0]-.96)<1e-14;assert abs(p[0]-2*np.pi*f[0]*.5*d/C)<1e-14
def test_window_preserves_relative_pulse_delay():
    t=np.tile(np.arange(1024)*.05,(3,1));e=np.array([np.exp(-((t[0]-x)/.3)**2)for x in(22,21,20)])
    w=common_window(t,e,5)
    for i in range(3):assert abs(w[i,1,np.argmax(w[i,0])]-(22-i))<1e-12
def test_fft_delay_phase_and_numerical_frequency_grid():
    time=np.arange(2048)*.05;ref=np.exp(-((time-30)/.3)**2);sample=np.exp(-((time-32)/.3)**2)
    a,p,f=transform(np.array([sample,time]),np.array([ref,time]))
    np.testing.assert_allclose(a,1,atol=1e-9);np.testing.assert_allclose(p,2*np.pi*f*2e-12,atol=1e-9)
