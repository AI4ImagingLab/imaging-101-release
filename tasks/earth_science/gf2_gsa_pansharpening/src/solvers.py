"""Explicit adaptive Gram-Schmidt fusion and retained SFIM failure comparison."""
import numpy as np
from .physics_model import interpolate23,lowpass
def solve_sfim(data,match_statistics=True,detail_gain=1.0):
    """Return float64 normalized radiometry (256,256,4), clipped to [0,1].
    PAN is histogram-matched to each interpolated band (sample standard deviation ddof=1).
    detail_gain=0 is interpolation-only; 1 is the unchanged selected SFIM method.
    """
    up=interpolate23(data['multispectral']);pan=np.tile(data['panchromatic'],(1,1,up.shape[-1]))
    if match_statistics:
        pan=(pan-pan.mean((0,1)))*(up.std((0,1),ddof=1)/pan.std((0,1),ddof=1))+up.mean((0,1))
    out=np.zeros_like(up)
    for c in range(up.shape[-1]):
        ratio=pan[...,c]/(lowpass(pan[...,c])+1e-8)
        out[...,c]=up[...,c]*(1+detail_gain*(ratio-1))
    return np.clip(out,0,1)

def downsample_pan(image):
    """Original effective OpenCV linear resize, factor4, half-pixel centers.
    Source passes INTER_CUBIC as dst positional argument; actual interpolation is linear.
    Difference-form lerp preserves its float32 arithmetic exactly on this geometry.
    """
    top=image[1::4,1::4]+(image[1::4,2::4]-image[1::4,1::4])*.5
    bottom=image[2::4,1::4]+(image[2::4,2::4]-image[2::4,1::4])*.5
    return top+(bottom-top)*.5
def regression_intensity(low,centered,pan_low):
    """Fit observed low-resolution regression and return centered intensity plus coefficients."""
    design=np.concatenate((low,np.ones((*low.shape[:2],1))),axis=-1)
    alpha=np.linalg.lstsq(design.reshape(-1,5),pan_low.reshape(-1,1),rcond=None)[0]
    intensity=np.concatenate((centered,np.ones((*centered.shape[:2],1))),axis=-1)@alpha
    return intensity-intensity.mean(),alpha
def detail_gains(intensity,centered):
    """Return four covariance/variance gains, preserving source ddof1/ddof0 convention."""
    variance=np.var(intensity)
    if not np.isfinite(variance) or variance==0:
        raise ValueError('GSA is undefined for zero-variance synthetic intensity')
    gains=[]
    for c in range(4):
        covariance=np.cov(intensity.ravel(),centered[...,c].ravel(),ddof=1)
        gains.append(covariance[0,1]/variance)
    return np.array(gains)
def solve(data,detail_gain=1.0,adaptive=True):
    """Adaptive Gram-Schmidt regression fusion, normalized float64(H,W,4).
    detail_gain1 is originalGSA;0 is interpolation. adaptiveFalse is equal-band-intensity ablation.
    Returns clipped image; no hidden calibration or target is used.
    """
    hs=data['multispectral'];pan=data['panchromatic']
    if np.all(np.ptp(hs,axis=(0,1))==0) or np.ptp(pan)==0:
        raise ValueError('GSA is undefined for zero-variance native observations')
    up=interpolate23(hs)
    means=up.mean((0,1));centered=up-means
    low=hs-hs.mean((0,1));hr=pan-pan.mean()
    pan_low=downsample_pan(hr[...,0])[...,None]
    if adaptive:
        intensity,_=regression_intensity(low,centered,pan_low)
    else:
        intensity=centered.mean(-1,keepdims=True)
        intensity=intensity-intensity.mean()
    gains=detail_gains(intensity,centered)
    detail=hr-intensity
    fused=centered+detail_gain*detail*np.array(gains)[None,None,:]
    fused=fused-fused.mean((0,1))+means
    return np.clip(fused,0,1)
def reconstruct(data):
    """PublishedGSA uint8 normalized radiometry scaled255; (1,256,256,4)."""
    return {'reconstruction':np.uint8(solve(data)*255)[None]}
