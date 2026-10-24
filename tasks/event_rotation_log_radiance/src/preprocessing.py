"""Acquisition-only clock and camera calibration, without a target image."""
from pathlib import Path
import numpy as np
HEIGHT=180
WIDTH=240
def load_observations(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as z:
        return z['events'][0].copy(),z['angular_velocity'][0].copy(),z['intrinsics'][0].copy()
def normalize_clock(events):
    e=np.array(events,dtype=np.float64,copy=True)
    duration=float(e[-1,0]-e[0,0])
    if duration<=0:raise ValueError('Events need an increasing nonzero acquisition window')
    e[:,0]=(e[:,0]-e[0,0])/duration
    return e,duration
def rotational_displacement(angular_velocity,intrinsics,duration):
    """Pinhole rotational flow in pixel/window, retaining original float32 grid arithmetic."""
    fx,fy,px,py=np.asarray([intrinsics[0,0],intrinsics[1,1],intrinsics[0,2],intrinsics[1,2]],dtype=np.float32)
    yy,xx=np.mgrid[:HEIGHT,:WIDTH].astype(np.float32)
    v,u=yy-py,xx-px
    basis_y=np.stack((v*v/fy+fy,-u*v/fx,-fy*u/fx),axis=-1)
    basis_x=np.stack((u*v/fy,-u*u/fx-fx,fx*v/fy),axis=-1)
    omega=np.asarray(angular_velocity,dtype=np.float64)
    return np.stack((np.inner(basis_x,omega),np.inner(basis_y,omega)))*duration
