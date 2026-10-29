"""Explicit normal-gravity, height and nuisance-trend treatment."""
from pathlib import Path
import numpy as np
EARTH_RADIUS=6371000.
GRAVITATIONAL_CONSTANT=6.67430e-11
def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as f:data={k:f[k][0].copy() for k in f.files}
    for value in data.values():
        if not np.all(np.isfinite(value)):raise ValueError('Nonfinite observations')
    return data
def prepare_observations(data):
    """Return locations m, upward simple-Bouguer residual mGal, and stages."""
    latitude=data['latitude'];longitude=data['longitude'];h=data['elevation']
    x=EARTH_RADIUS*np.cos(np.deg2rad(19.4))*np.deg2rad(longitude+155.275)
    y=EARTH_RADIUS*np.deg2rad(latitude-19.4)
    s2=np.sin(np.deg2rad(latitude))**2
    normal=978032.53359*(1+.00193185265241*s2)/np.sqrt(1-.00669437999013*s2)
    bouguer=data['observed_gravity']-normal+.3086*h-2*np.pi*GRAVITATIONAL_CONSTANT*2670.*h*1e5
    trend=np.c_[np.ones(len(x)),x/10000.,y/10000.]
    anomaly=-project_trend(bouguer,trend)
    return np.c_[x,y,h],anomaly,dict(normal_gravity=normal,simple_bouguer=bouguer,trend=trend)
def project_trend(values,trend):
    """Apply the same orthogonal nuisance-plane removal to data or a matrix."""
    return values-trend@np.linalg.lstsq(trend,values,rcond=None)[0]
