"""Metrics and physically labeled spatial sections; no backend changes."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import VOXEL_MIN_MM,VOXEL_WIDTH_MM

def metrics(a,r):
    """Uncentered cosine NCC and reference-range-normalized RMS error."""
    a=np.asarray(a,dtype=np.float64).ravel();r=np.asarray(r,dtype=np.float64).ravel()
    span=np.ptp(r);nr=np.linalg.norm(r);na=np.linalg.norm(a)
    if span<=0 or nr<=0:raise ValueError('Degenerate reference')
    return {'ncc':float(np.dot(a,r)/(na*nr)) if na else 0.,'nrmse':float(np.sqrt(np.mean((a-r)**2))/span)}

def slice_panel(ax,image,z_index,title,vmax=None,cmap='viridis',signed=False):
    """XY section at one declared z voxel; inputs rad, display mrad; return image."""
    section=image[:,:,z_index].T*1000
    if signed:
        lim=float(np.max(abs(section))) if vmax is None else vmax
        im=ax.imshow(section,origin='lower',extent=[-180,180,-400,400],cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='equal')
    else:
        im=ax.imshow(section,origin='lower',extent=[-180,180,-400,400],cmap=cmap,vmin=0,vmax=vmax,aspect='equal')
    z=VOXEL_MIN_MM[2]+VOXEL_WIDTH_MM*(z_index+.5)
    ax.set(xlabel='x / mm',ylabel='y / mm',title=f'{title}\nz = {z:g} mm')
    return im
