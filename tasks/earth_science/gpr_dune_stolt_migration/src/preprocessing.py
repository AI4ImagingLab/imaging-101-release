"""Explicit acquisition/time calibration and background removal."""
import numpy as np

TIME_ZERO_NS=5.0
TRACE_PITCH_M=.02
VELOCITY_M_S=1.4e8
HORIZONTAL_TAPER=100
VERTICAL_TAPER=40

def load_data(path='data/raw_data.npz'):
    """Return all measured radar samples, time ns, and position m."""
    with np.load(path,allow_pickle=False) as d:
        data,time,x=(d[k][0] for k in ('radar','time_ns','position_m'))
    if data.shape!=(512,1040) or time.shape!=(512,) or x.shape!=(1040,):raise ValueError('Invalid acquisition shape')
    if any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in (data,time,x)):raise ValueError('Expected finite float64 acquisition')
    return data,time,x

def load_topography(path='data/raw_data.npz'):
    """Return independently surveyed station positions/elevations, metres."""
    with np.load(path,allow_pickle=False) as d:
        station,elevation=(d[k][0] for k in ('topo_position_m','surface_elevation_m'))
    if station.shape!=(36,) or elevation.shape!=(36,) or np.any(np.diff(station)<=0):raise ValueError('Invalid survey geometry')
    if any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in (station,elevation)):raise ValueError('Invalid topography calibration')
    return station,elevation

def prepare(data,time_ns,time_zero_ns=TIME_ZERO_NS):
    """Uniform continuous time-zero resampling, trace DC and mean-trace removal.

    Input time x trace samples; return shifted/dewow/final arrays (458,1040)
    for the bound observation. Resampling interval equals original interval.
    """
    dt=time_ns[1]-time_ns[0]
    n=int(np.floor((time_ns[-1]-time_zero_ns)/dt))+1
    target=time_zero_ns+np.arange(n)*dt
    left=np.searchsorted(time_ns,target,side='right')-1;left=np.clip(left,0,len(time_ns)-2)
    fraction=(target-time_ns[left])/(time_ns[left+1]-time_ns[left])
    shifted=data[left]+fraction[:,None]*(data[left+1]-data[left])
    dewow=shifted-shifted.mean(axis=0,keepdims=True)
    preprocessed=dewow-dewow.mean(axis=1,keepdims=True)
    return shifted,dewow,preprocessed

def taper(shape,horizontal=HORIZONTAL_TAPER,vertical=VERTICAL_TAPER):
    """Separable linear edge taper identical to selected ImpDAR convention."""
    n,nx=shape
    h=np.minimum(np.arange(nx),np.arange(nx)[::-1])/horizontal
    v=np.minimum(np.arange(n),np.arange(n)[::-1])/vertical
    return np.minimum(v,1)[:,None]*np.minimum(h,1)[None,:]
