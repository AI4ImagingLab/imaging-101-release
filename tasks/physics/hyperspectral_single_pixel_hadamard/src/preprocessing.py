"""Validate original detector readings and undo complementary pattern splitting.

Adapted from SPAS, Guilherme Beneti Martins et al.; LGPL-3.0 upstream.
See LICENSE and NOTICE for source and data attribution.
"""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SIDE=64
SPECTRAL_CHANNELS=2048
INTEGRATION_MS=1.0
PATTERN_PERIOD_US=1400
NOMINAL_PIXEL_UM=182.4

def load_observations(path=ROOT/'data/raw_data.npz'):
    """Return copied unbatched arrays, preserving float64 counts and wavelengths."""
    specifications={'measurements':((1,8192,2048),np.dtype('float64')),
                    'pattern_id':((1,8192),np.dtype('int64')),
                    'wavelength_nm':((1,2048),np.dtype('float64'))}
    result={}
    with np.load(path,allow_pickle=False) as f:
        for key,(shape,dtype) in specifications.items():
            a=f[key]
            if a.shape!=shape or a.dtype!=dtype or not np.isfinite(a).all():raise ValueError('Invalid '+key)
            result[key]=a[0].copy()
    ids=result['pattern_id']
    if not np.array_equal(np.sort(ids),np.arange(8192)):raise ValueError('Pattern identifiers must be a permutation of0..8191')
    if np.any(ids[::2]%2) or not np.array_equal(ids[1::2],ids[::2]+1):raise ValueError('Expected adjacent positive/negative pairs')
    if np.any(np.diff(result['wavelength_nm'])<=0):raise ValueError('Wavelengths must increase')
    return result

def differential_coefficients(measurements,pattern_id):
    """Return Walsh-indexed coefficients (4096, channels), counts."""
    raw=np.asarray(measurements);ids=np.asarray(pattern_id)
    if raw.ndim!=2 or raw.shape[0]!=len(ids) or len(ids)%2:raise ValueError('Invalid complementary arrays')
    out=np.zeros((4096,raw.shape[1]),dtype=np.float64)
    out[ids[::2]//2]=raw[::2]-raw[1::2]
    return out

def spectral_bins(measurements,wavelength_nm,edges_nm):
    """Sum detector channels in left-closed/right-open wavelength bands."""
    values=np.asarray(measurements);wave=np.asarray(wavelength_nm)
    edges=np.asarray(edges_nm,dtype=float)
    if np.any(np.diff(edges)<=0):raise ValueError('Band edges must increase')
    masks=[(wave>=a)&(wave<b) for a,b in zip(edges[:-1],edges[1:])]
    if any(not m.any() for m in masks):raise ValueError('Empty wavelength band')
    return np.stack([values[...,m].sum(axis=-1) for m in masks],axis=-1)
