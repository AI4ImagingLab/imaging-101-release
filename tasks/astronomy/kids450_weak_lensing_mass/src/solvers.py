"""Local KiDS ensemble calibration, catalogue gridding, KS93 and Gaussian filtering."""
import numpy as np
from scipy.ndimage import gaussian_filter
from .physics_model import inverse
GRID_SIZE=64
SMOOTHING_PIXELS=2.0
RA_CENTRE=129.0
DEC_CENTRE=-0.5
def grid_catalogue(data,calibrate=True):
    """Weighted ensemble-corrected shear maps; empty pixels are zero.
    Inputs: positions(N,2) degrees; ellipticity(N,2), weight(N,), multiplicative_bias(N,).
    Returns gamma1,gamma2,counts,weight_sum arrays (64,64), north then east.
    """
    ra,dec=data['positions'].T;w=data['weight'];e=data['ellipticity'];m=data['multiplicative_bias']
    c=np.cos(np.deg2rad(DEC_CENTRE));x=(ra-RA_CENTRE)*c;y=dec-DEC_CENTRE
    bounds=[[-.5*c,.5*c],[-.5,.5]]
    def hist(v=None):return np.histogram2d(x,y,bins=GRID_SIZE,range=bounds,weights=v)[0].T
    counts=hist();sw=hist(w);den=hist(w*(1+m)) if calibrate else sw
    g=[np.divide(hist(w*e[:,j]),den,out=np.zeros_like(den),where=den>0) for j in range(2)]
    return g[0],g[1],counts,sw
def solve(data,sigma=SMOOTHING_PIXELS,calibrate=True):
    """Return unbatched E/B maps after periodic Gaussian sigma (pixels)."""
    g1,g2,_,_=grid_catalogue(data,calibrate)
    e,b=inverse(g1,g2)
    if sigma>0:e,b=gaussian_filter(e,sigma,mode='wrap'),gaussian_filter(b,sigma,mode='wrap')
    return e,b
def reconstruct(data):
    """Return required dimensionless float64 arrays (1,64,64)."""
    e,b=solve(data)
    return {'reconstruction':e[None],'bmode':b[None]}
