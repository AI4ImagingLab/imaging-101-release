"""Crocker–Grier bandpass, extracted from sdt-python (BSD-3-Clause).
Copyright2020 Lukas Schrangl; adapted2026 to explicit local arrays.
"""
import numpy as np
from scipy.signal import fftconvolve

FEATURE_RADIUS=3
NOISE_RADIUS=1
SIGNAL_THRESHOLD=300.
MASS_THRESHOLD=5000.

def load_observations(path):
    with np.load(path,allow_pickle=False) as z: frames=z['frames'].copy()
    if frames.shape!=(1,10,128,128) or frames.dtype!=np.uint16:
        raise ValueError('Expected complete native (1,10,128,128) uint16 frames')
    return frames

def bandpass_filter(image, feature_radius=3, noise_radius=1, nonneg=True):
    """Gaussian-minus-box valid convolution; zero padded margin, ADU output."""
    w=max(feature_radius,2*noise_radius)
    gaussian_1d=np.exp(-(np.arange(-w,w+1)/(2*noise_radius))**2)
    B=np.sum(gaussian_1d)**2
    K0=np.sum(gaussian_1d**2)**2/B-B/(2*w+1)**2
    kernel=(np.outer(gaussian_1d,gaussian_1d)/B-1/(2*w+1)**2)/K0
    filtered=fftconvolve(image,kernel,'valid')
    ret=np.zeros_like(image,dtype=float)
    ret[w:-w,w:-w]=np.clip(filtered,0,np.inf) if nonneg else filtered
    return ret
