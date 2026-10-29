"""Explicit Gaussian peak fit and confidence, adapted from OpenPIV, GPL-3.0-or-later."""
import numpy as np
from .preprocessing import prepare
from .physics_model import correlate

def subpixel_peak(c):
    """Return row/column peak, mutating c by upstream epsilon for valid peaks."""
    i,j=np.unravel_index(np.argmax(c),c.shape)
    if i in (0,c.shape[0]-1) or j in (0,c.shape[1]-1):
        return np.array([np.nan,np.nan])
    c+=1e-7
    center,lo,hi,left,right=c[i,j],c[i-1,j],c[i+1,j],c[i,j-1],c[i,j+1]
    if min(center,lo,hi,left,right)<0:
        dy=(lo-hi)/(2*lo-4*center+2*hi)
        dx=(left-right)/(2*left-4*center+2*right)
    else:
        def offset(a,b):
            den=2*np.log(a)-4*np.log(center)+2*np.log(b)
            return (np.log(a)-np.log(b))/den if den!=0 else 0.
        dy,dx=offset(lo,hi),offset(left,right)
    return np.array([i+dy,j+dx])

def peak_confidence(c,width=2):
    """Peak-to-second-peak ratio with upstream edge rejection; dimensionless."""
    i,j=np.unravel_index(np.argmax(c),c.shape);p=c[i,j]
    if p<1e-3 or i in (0,c.shape[0]-1) or j in (0,c.shape[1]-1):return 0.
    other=c.copy();other[max(0,i-width):i+width+1,max(0,j-width):j+width+1]=-np.inf
    ii,jj=np.unravel_index(np.argmax(other),other.shape);q=other[ii,jj]
    if q==0 or ((ii in (0,c.shape[0]-1) or jj in (0,c.shape[1]-1)) and q>0.5*p):return 0.
    return p/q

def solve(a,b,window=32,search=64,overlap=16,dt=1.):
    """Return velocity (1,7,10,2), confidence (1,7,10); dt is frame intervals."""
    if dt<=0:raise ValueError('dt must be positive')
    aa,bb=prepare(a,b,window,search,overlap);c=correlate(aa,bb)
    shape=((a.shape[0]-search)//(search-overlap)+1,(a.shape[1]-search)//(search-overlap)+1)
    peaks=np.array([subpixel_peak(ci) for ci in c])-search//2
    velocity=peaks[:,::-1].reshape(*shape,2)/dt
    confidence=np.array([peak_confidence(ci) for ci in c]).reshape(shape)
    if not np.isfinite(velocity).all():raise ValueError('Invalid edge/degenerate correlation peak')
    return velocity[None],confidence[None]
