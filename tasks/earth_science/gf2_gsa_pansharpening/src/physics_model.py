"""Explicit SFIM resolution-matching operators; registered pixel geometry.
Adapted from Anjing Guo's toolbox, MIT and SFIM Apache-2.0; see NOTICE.
"""
import numpy as np
from scipy.ndimage import correlate1d
from scipy.signal import convolve2d
RATIO=4
def interpolate23(image,ratio=RATIO):
    """23-tap periodic dyadic interpolation (H,W,C), first insertion at odd indices."""
    if ratio<2 or ratio&(ratio-1):raise ValueError('ratio must be a power of two >=2')
    core=2*np.array([.5,.305334091185,0,-.072698593239,0,.021809577942,0,-.005192756653,0,.000807762146,0,-.000060081482])
    coeff=np.r_[core[::-1][:-1],core]
    current=image.transpose(2,0,1)
    for stage in range(int(np.log2(ratio))):
        up=np.zeros((current.shape[0],current.shape[1]*2,current.shape[2]*2))
        offset=1 if stage==0 else 0;up[:,offset::2,offset::2]=current
        current=correlate1d(correlate1d(up,coeff,axis=2,mode='wrap'),coeff,axis=1,mode='wrap')
    return current.transpose(1,2,0)
def lowpass(image,width=5):
    """Unit-sum width-by-width box blur, periodic boundary; float64 array."""
    return convolve2d(image,np.ones((width,width))/(width*width),mode='same',boundary='wrap')
def degrade(image):
    """Diagnostic box-blurred decimation, not a calibrated native sensor PSF."""
    return np.stack([lowpass(image[...,c])[::4,::4] for c in range(image.shape[-1])],-1)
