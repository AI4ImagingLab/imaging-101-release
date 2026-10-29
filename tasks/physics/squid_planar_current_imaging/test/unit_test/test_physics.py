import numpy as np
from src.physics_model import frequency_grid,psf_transfer,hanning,invert_spectrum,forward_spectrum,MU0
def test_exact_odd_frequency_bins_delta_origin_and_even_PSF():
    n=9;x=np.arange(n);wave=np.exp(2j*np.pi*2*x/n);peak=np.argmax(abs(np.fft.fftshift(np.fft.fft(wave))))
    kx,ky,k=frequency_grid((n,n),1.,1.);assert kx[0,peak]==4*np.pi/n
    old=np.linspace(-np.pi,np.pi,n,endpoint=False);np.testing.assert_allclose(old[peak]-kx[0,peak],-np.pi/n,atol=1e-14)
    delta=np.zeros((n,n));delta[n//2,n//2]=1;np.testing.assert_allclose(psf_transfer(delta),1,atol=1e-15)
    q=np.arange(n)-n//2;gauss=np.exp(-(q[:,None]**2+q[None,:]**2)/3);assert np.max(abs(psf_transfer(gauss).imag))<1e-14
def test_analytic_fourier_mode_signed_transfer_and_reconstruction():
    shape=(15,17);dx=.7;dy=.9;X,Y=np.meshgrid(np.arange(17)*dx,np.arange(15)*dy)
    a=2*np.pi*2/(17*dx);b=2*np.pi/(15*dy);K=np.hypot(a,b);theta=a*X+b*Y
    expected=np.stack([-b*np.sin(theta),a*np.sin(theta)])
    # Standard right-hand +Bz is +MU0*K/2*exp(-Kh)*cos(theta).
    # Source signed flux convention is its negative, not a fitted sign.
    flux=-MU0*K/2*np.exp(-K*.8)*np.cos(theta)
    kx,ky,k=frequency_grid(shape,dx,dy);F=np.fft.fftshift(np.fft.fft2(flux));H=np.ones(shape);P=np.ones(shape,dtype=complex)
    jk,_=invert_spectrum(F,P,kx,ky,k,H,.8);actual=np.fft.ifft2(np.fft.ifftshift(jk,axes=(-2,-1)),axes=(-2,-1))
    np.testing.assert_allclose(actual.real,expected,rtol=2e-12,atol=2e-12);assert np.max(abs(actual.imag))<1e-12
    np.testing.assert_allclose(forward_spectrum(jk,P,kx,ky,k,.8),F,rtol=1e-10,atol=1e-16)
def test_zero_mode_divergence_and_filter_attenuation():
    kx,ky,k=frequency_grid((9,11),1.,1.);H=hanning(k,1.);P=np.ones(k.shape,dtype=complex);F=np.ones(k.shape,dtype=complex)
    jk,factor=invert_spectrum(F,P,kx,ky,k,H,1.);assert np.all(jk[:,k==0]==0) and factor[k==0]==0
    np.testing.assert_allclose(kx*jk[0]+ky*jk[1],0,atol=1e-8)
    assert H[k==0]==1 and np.all(H[k>np.sqrt(2)]==0)
    z=np.zeros_like(jk);np.testing.assert_array_equal(forward_spectrum(z,P,kx,ky,k,1.),0)
def test_even_and_odd_centered_PSF_shift_distinction():
    for n in [8,9]:
        p=np.zeros((n,n));p[n//2,n//2]=1;np.testing.assert_array_equal(psf_transfer(p),np.ones((n,n)))
        legacy=np.fft.fft2(np.fft.fftshift(p))
        assert (np.max(abs(legacy-1))<1e-14)==(n%2==0)
