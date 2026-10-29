"""Euler location and local full-vector magnetic-moment inversion.

Copyright (c) 2023 Gelson Ferreira de Souza-Junior, Leonardo Uieda.
Adapted under the MIT license; see NOTICE.
"""
import numpy as np
from scipy.linalg import solve
from .physics_model import dipole_jacobian
from .preprocessing import prepare,detect_windows,HEIGHT,CONTINUATION

def euler(field,dx,dy,dz,x,y,z=HEIGHT+CONTINUATION):
    """Solve homogeneity SI=3 normal equations; position um and base nT."""
    xx,yy=np.meshgrid(x,y)
    g=np.column_stack((dx.ravel(),dy.ravel(),dz.ravel(),np.full(field.size,3.)))
    h=(xx*dx+yy*dy+z*dz+3*field).ravel()
    p=np.linalg.solve(g.T@g,g.T@h)
    return p[:3],p[3]

def invert_moment(field,x,y,position,z=HEIGHT+CONTINUATION):
    """Return moment A m^2, covariance, R2 and SNR dB for one window."""
    xx,yy=np.meshgrid(x,y)
    a=dipole_jacobian(xx,yy,z,position);d=field.ravel()/1e9
    h=a.T@a
    m=solve(h,a.T@d,assume_a='pos')
    residual=d-a@m;sse=np.sum(residual**2)
    covariance=sse/(d.size-3)*np.linalg.inv(h)
    r2=1-sse/np.linalg.norm(d-d.mean())**2
    snr=10*np.log10(np.var(d,ddof=1)/np.var(residual,ddof=1))
    return m,covariance,r2,snr

def reconstruct(field,x,y):
    """Run full paper real-data benchmark; reject above-surface sources.

    Return position (K,3), moment (K,3), base (K,), score (K,2), windows
    (K,4), covariance (K,3,3), prepared field tuple, detection windows.
    """
    prep=prepare(field,x,y);f,dx,dy,dz,_,stretch=prep
    windows=detect_windows(stretch,x,y)
    positions=[];moments=[];bases=[];scores=[];valid=[];cov=[]
    for w in windows:
        ix=(x>=w[0])&(x<=w[1]);iy=(y>=w[2])&(y<=w[3]);sel=np.ix_(iy,ix)
        p,b=euler(f[sel],dx[sel],dy[sel],dz[sel],x[ix],y[iy])
        m,c,r,s=invert_moment(f[sel]-b,x[ix],y[iy],p)
        if p[2]>0:continue
        positions.append(p);moments.append(m);bases.append(b);scores.append([r,s]);valid.append(w);cov.append(c)
    return tuple(np.asarray(v) for v in (positions,moments,bases,scores,valid,cov))+(prep,windows)
