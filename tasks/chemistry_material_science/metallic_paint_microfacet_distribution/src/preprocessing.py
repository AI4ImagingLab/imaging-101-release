"""MERL encoded reflectance lookup; conventions from Dupuy's public-domain dj_brdf."""
import numpy as np
from pathlib import Path
RGB_SCALE=np.array([1.,1.15,1.66])/1500
LUMINANCE=np.array([.2126,.7152,.0722])
def load_data(path):
    with np.load(Path(path),allow_pickle=False) as f: a=f['brdf']
    if a.shape!=(1,3,90,90,180) or a.dtype!=np.float64 or not np.isfinite(a).all():
        raise ValueError('Expected finite float64 encoded BRDF (1,3,90,90,180)')
    return a[0]
def backscatter(brdf):
    """Return requested angles (89,) and measured RGB (89,3), in rad and sr^-1."""
    theta=(np.arange(89)/89*np.sqrt(np.pi/2))**2
    # The original direction constructor snaps vectors near the pole to zero.
    half=np.where(np.cos(theta)>.99999,0.,np.arccos(np.cos(theta)))
    index=np.clip(np.sqrt(half/(np.pi/2)*90*90).astype(int),0,89)
    rgb=brdf[:,index,0,0].T*RGB_SCALE
    rgb[np.any(rgb<0,axis=1)]=0
    return theta,rgb
