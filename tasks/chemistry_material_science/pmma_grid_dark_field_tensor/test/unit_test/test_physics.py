from pathlib import Path
import numpy as np
from src.physics_model import moments_from_coefficients,mode_ratios,covariance_eigenvalues
ROOT=Path(__file__).resolve().parents[2]
def test_independent_gaussian_fourier_blur_inverse():
    n=128;p=16.;y,x=np.mgrid[:n,:n];q=2*np.pi/p
    reference=10+np.cos(q*x)+2*np.cos(-q*y)+.8*np.cos(q*(x+y))+.6*np.cos(q*(x-y))
    ky,kx=np.meshgrid(-2*np.pi*np.fft.fftfreq(n),2*np.pi*np.fft.fftfreq(n),indexing='ij');cxx,cyy,cxy=4.,2.,1.;trans=.7
    blurred=np.fft.ifft2(np.fft.fft2(reference)*np.exp(-.5*(cxx*kx*kx+cyy*ky*ky+2*cxy*kx*ky))).real*trans
    r=np.fft.fft2(reference);s=np.fft.fft2(blurred);f=n//int(p);positions=[(0,0),(0,f),(-f,0),(f,f),(-f,f)];ratios=np.array([abs(s[a])/abs(r[a]) for a in positions]);grid=np.ones((13,13,5));sample=grid*ratios
    c,t,_=moments_from_coefficients(grid,sample,period=p)
    np.testing.assert_allclose(c[:,0,0],[4.,2.,1.],atol=1e-12);np.testing.assert_allclose(t,.7,atol=1e-14)
    np.testing.assert_allclose(mode_ratios(c,t,p)[:,0,0],ratios,atol=1e-14)
def test_zero_blur_and_signed_estimate():
    grid=np.ones((13,13,5));c,t,_=moments_from_coefficients(grid,grid*.8);np.testing.assert_allclose(c,0,atol=1e-14);np.testing.assert_allclose(t,.8)
    values=covariance_eigenvalues(np.array([[[-.2]],[[.5]],[[0.]]]))
    np.testing.assert_allclose(values[:,0,0],[-.2,.5]);assert values[0,0,0]<0

def test_public_constants_and_keys():
    from src.preprocessing import load_data,PERIOD,KERNEL,AVERAGE
    a,b=load_data(ROOT/'data/raw_data.npz');assert a.shape==(468,468) and a.dtype==np.float32
    assert PERIOD==12.68550106609808 and KERNEL==7 and AVERAGE==13
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:assert set(z.files)=={'grid','sample','flat'}
