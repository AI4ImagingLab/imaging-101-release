"""Explicit block-DCT acquisition and joint total variation.

Adapted from Victor VdE jpeg2png v1.01; GPL-3.0-or-later.
"""
import numpy as np
from scipy.fft import dctn,idctn
def blocks(u):
    c,h,w=u.shape;return u.reshape(c,h//8,8,w//8,8).transpose(0,1,3,2,4)
def image(b):
    c,h,w,_,_=b.shape;return b.transpose(0,1,3,2,4).reshape(c,h*8,w*8)
def transform(u):return dctn(blocks(u),axes=(-2,-1),norm='ortho',workers=1)
def inverse(b):return image(idctn(b,axes=(-2,-1),norm='ortho',workers=1))
def tv_gradient(u):
    dx=np.zeros_like(u);dy=dx.copy();dx[:,:,:-1]=np.diff(u,axis=2);dy[:,:-1]=np.diff(u,axis=1)
    norm=np.zeros_like(u[0])
    for c in range(3):norm+=dx[c]**2;norm+=dy[c]**2
    np.sqrt(norm,out=norm);norm[norm==0]=1
    ax=dx/norm/np.sqrt(np.float32(3));ay=dy/norm/np.sqrt(np.float32(3))
    g=-ax-ay;g[:,:,1:]+=ax[:,:,:-1];g[:,1:]+=ay[:,:-1]
    return g
def objective(u,coeff,q):
    dx=np.zeros_like(u);dy=dx.copy();dx[:,:,:-1]=np.diff(u,axis=2);dy[:,:-1]=np.diff(u,axis=1)
    tv=np.sqrt(np.sum(dx.astype(float)**2+dy.astype(float)**2,axis=0)).sum()/np.sqrt(3)
    fidelity=.001*2*255*np.sqrt(2)/2*np.sum(((transform(u)-coeff*q[:,None,None])/q[:,None,None]).astype(float)**2)
    return float(tv+fidelity)
