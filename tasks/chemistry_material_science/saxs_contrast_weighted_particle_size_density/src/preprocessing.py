"""Observed pinhole SAXS arrays and public-data background inference."""
import numpy as np
def load_observations(path):
    with np.load(path,allow_pickle=False) as z:
        return tuple(np.asarray(z[k][0],dtype=np.float64) for k in ('scattering_vector','intensity','intensity_uncertainty'))
def estimate_background(q,intensity,q_min=0.5,q_max=1.0):
    keep=(q>=q_min)&(q<=q_max)&np.isfinite(q)&np.isfinite(intensity)&(q>0)
    if not np.any(keep):raise ValueError('No high-q background observations')
    return float(np.mean(intensity[keep]))
def radius_grid(n_bins=500):
    return np.logspace(np.log10(10.),np.log10(30000.),n_bins)
def bin_widths(radius):
    r=np.asarray(radius,dtype=float);w=np.empty_like(r)
    if len(r)==1:return np.ones(1)
    w[0]=r[1]-r[0];w[-1]=r[-1]-r[-2];w[1:-1]=(r[2:]-r[:-2])/2
    return w
