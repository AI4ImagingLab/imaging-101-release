"""Explicit NOAA observation conventions; no interpolation of missing data."""
import numpy as np
N_OBSERVATIONS=16822
CONSTITUENTS=('M2','S2','N2','K1','O1','M4','MS4','MN4')
def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False)as f:a={k:f[k].copy()for k in ('time_seconds','speed','direction_deg','frequency_cph')}
    for k,v in a.items():
        shape=(1,8)if k=='frequency_cph'else(1,N_OBSERVATIONS)
        if v.shape!=shape or v.dtype.kind!='f'or not np.isfinite(v).all():raise ValueError('Invalid '+k)
    if not(np.diff(a['time_seconds'][0])>0).all():raise ValueError('Time must strictly increase')
    if (a['speed']<0).any()or(a['direction_deg']<0).any()or(a['direction_deg']>360).any():raise ValueError('Invalid measured polar velocity')
    return{k:v[0]for k,v in a.items()}
def polar_to_velocity(speed,direction_deg):
    """Current direction is clockwise from true north, TOWARD flow; cm/s."""
    angle=np.deg2rad(direction_deg)
    return np.array([speed*np.sin(angle),speed*np.cos(angle)])
def gregorian_days(time_seconds):
    """Match original absolute Gregorian float64 time normalization exactly.
    2025-01-01 UTC has Unix seconds1735689600 and Python ordinal739252.
    """
    return(np.asarray(time_seconds,dtype=np.float64)+1735689600.)/86400+719163.
