"""Explicit RGGB sampling and general Bayer masks; no domain APIs.

Adapted from Colour Developers BSD-3-Clause reference (LICENSE).
"""
import numpy as np
PATTERN='RGGB'
WHITE_LEVEL=1023
SHAPE=(2128,2840)

def cfa_masks(shape,pattern=PATTERN):
    """Return R/G/B bool masks for a height,width grid with top-left pattern."""
    if pattern not in ('RGGB','BGGR','GRBG','GBRG'):raise ValueError('Invalid Bayer pattern')
    if len(shape)!=2 or min(shape)<2:raise ValueError('Expected 2D spatial grid')
    masks={k:np.zeros(shape,dtype=bool) for k in 'RGB'}
    for i,c in enumerate(pattern):masks[c][i//2::2,i%2::2]=True
    return tuple(masks[c] for c in 'RGB')

def sample_rgb(image,pattern=PATTERN):
    """Map unbatched H,W,3 linear camera RGB to unbatched H,W CFA values."""
    a=np.asarray(image)
    if a.ndim!=3 or a.shape[-1]!=3:raise ValueError('Expected H,W,3 camera RGB')
    return sum(a[...,i]*m for i,m in enumerate(cfa_masks(a.shape[:2],pattern)))

def sampling_adjoint(mosaic,pattern=PATTERN):
    """Insert a scalar CFA image into its three measured colour channels."""
    a=np.asarray(mosaic)
    return np.stack([a*m for m in cfa_masks(a.shape,pattern)],axis=-1)
