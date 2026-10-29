"""Rotation and convolution operators for a plane-wave receiver response."""
import numpy as np
from scipy.fftpack import fft, ifft, next_fast_len
from .preprocessing import BACK_AZIMUTH, SAMPLING_RATE
def rotate_zne(waveforms, back_azimuth=BACK_AZIMUTH):
    """ZNE to ZRT; R points away from source and T follows ObsPy handedness."""
    z,n,e=np.asarray(waveforms); b=np.deg2rad(back_azimuth)
    return np.stack((z,-e*np.sin(b)-n*np.cos(b),-e*np.cos(b)+n*np.sin(b)))
def filters(nfft, gauss=.5, shift=30.):
    """Return Gaussian low pass times delay, frequency array in Hz."""
    f=np.fft.fftfreq(nfft,1/SAMPLING_RATE)
    return np.exp(np.maximum(-.5*(f/gauss)**2,-700))*np.exp(-2j*np.pi*f*shift), f
def convolve(source, response, nfft=None):
    """Discrete circular acquisition y[n]=sum s[m]h[n-m]; no dt factor."""
    if nfft is None:nfft=next_fast_len(len(source))
    return ifft(fft(source,nfft)*fft(response,nfft)).real
def adjoint(source, residual, nfft=None):
    """Euclidean adjoint of the same circular convolution."""
    if nfft is None:nfft=next_fast_len(len(source))
    return ifft(np.conj(fft(source,nfft))*fft(residual,nfft)).real
def reconvolution(zrt, receiver, norm, gauss=.5):
    """Predict Gaussian-smoothed response, undoing 30s delay and normalization.
    The declared 4801 samples are zero padded to the native 4860 FFT grid.
    Return predicted and observed smoothed radial/transverse velocities (m/s).
    """
    n=zrt.shape[-1]; nf=next_fast_len(n); f=np.fft.fftfreq(nf,1/SAMPLING_RATE)
    pred=ifft(fft(receiver,nf,axis=-1)*fft(zrt[0],nf)*np.exp(2j*np.pi*f*30.)/norm,axis=-1).real[...,:n]
    obs=ifft(fft(zrt[1:],nf,axis=-1)*np.exp(-.5*(f/gauss)**2),axis=-1).real[...,:n]
    return pred,obs
