"""Correlation acquisition model; adapted from OpenPIV (GPL-3.0-or-later)."""
import numpy as np
from scipy.fft import rfft2, irfft2
from .preprocessing import normalize

def correlate(a,b):
    """Normalized circular cross-correlation, positive shift from A toward B."""
    a,b=normalize(a),normalize(b)
    c=irfft2(np.conj(rfft2(a,axes=(-2,-1)))*rfft2(b,axes=(-2,-1)),axes=(-2,-1)).real
    return np.fft.fftshift(c,axes=(-2,-1))/(c.shape[-2]*c.shape[-1])

def field_coordinates(shape=(369,511),search=64,overlap=16):
    """Actual window centers in image pixels; x right, y down, no recentering."""
    ys=np.arange(0,shape[0]-search+1,search-overlap)+search/2
    xs=np.arange(0,shape[1]-search+1,search-overlap)+search/2
    return np.meshgrid(xs,ys)
