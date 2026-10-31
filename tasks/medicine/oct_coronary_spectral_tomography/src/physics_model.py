"""Local calibrated Fourier acquisition and depth operators (MIT upstream adapted)."""
import numpy as np
NUM_SAMPLES=1600
NUM_DEPTH=2048
ZOOM=2
CARRIER_INDEX=400
REVERSE_INDEX=1024
FLIP_DEPTH=True
SCALE=1e-6
def calibration(chirp,dispersion):
    x=np.arange(0,1,1/len(chirp)); target=np.arange(0,1,1/800)
    values=np.interp(target,x,chirp); values[0]=0; values[-1]=1
    values=np.floor(values*3200); values[0]=1; values[-1]=3199
    indices=np.searchsorted(np.arange(1,3201),values)
    half=len(dispersion)//2
    complex_dispersion=dispersion[:half]+1j*dispersion[half:]
    correction=np.interp(target,np.arange(0,1,1/half),complex_dispersion)
    return indices,correction,np.hanning(800)[:,None]
def analytic_spectrum(corrected,indices,dispersion):
    # Raw ADC (C,N,A); detector-channel C; spectral sample N; A-line A.
    transformed=np.fft.fft(corrected,axis=1)
    padded=np.zeros((corrected.shape[0],3200,corrected.shape[2]),dtype=np.complex64)
    padded[:,:800]=transformed[:,:800]
    padded=np.roll(padded,-CARRIER_INDEX,axis=1)
    zoomed=3200*np.fft.ifft(padded,axis=1)
    return zoomed[:,indices]*dispersion[None,:,None]
def depth_transform(spectrum,window):
    depth=np.fft.fft(spectrum*window[None],n=NUM_DEPTH,axis=1)*SCALE
    depth=np.roll(depth,REVERSE_INDEX,axis=1)
    return np.flip(depth,axis=1).astype(np.complex64)
def windowed_spectrum(depth):
    unflipped=np.flip(depth,axis=1)
    centered=np.roll(unflipped,-REVERSE_INDEX,axis=1)
    return np.fft.ifft(centered/SCALE,axis=1)
def forward(reflectivity):
    """Depth profile to the calibrated zero-padded spectral representation."""
    return windowed_spectrum(reflectivity)
