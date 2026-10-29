"""Adaptive GPA and weighted Poisson integration, explicit ten-step PCG."""
import numpy as np
from scipy.fft import dctn,idctn
from .physics_model import adaptive_lockin,lockin,weighted_projection,wrap_phase
from .preprocessing import prepare,NM_PER_PIXEL

def divergence(dx,dy):
    return np.diff(dx,axis=1,prepend=0,append=0)+np.diff(dy,axis=0,prepend=0,append=0)

def integrate_gradient(dx,dy,weight,kmax=10):
    dx=wrap_phase(dx);dy=wrap_phase(dy);ww=weight**2
    wx=np.minimum(ww[:,:-1],ww[:,1:]);wy=np.minimum(ww[:-1],ww[1:])
    r=divergence(wx*dx,wy*dy);norm0=np.linalg.norm(r);phi=np.zeros_like(weight)
    n,m=weight.shape
    if n!=m:raise ValueError('This validated upstream discretization is bound to square images')
    i,j=np.ogrid[:n,:m];scale=2*(np.cos(np.pi*i/m)+np.cos(np.pi*j/n)-2);scale[0,0]=1
    previous=None
    for iteration in range(kmax):
        if np.all(r==0):break
        z=idctn(dctn(r)/scale);rz=np.tensordot(r,z)
        direction=z if previous is None else z+(rz/previous)*direction
        q=divergence(wx*np.diff(direction,axis=1),wy*np.diff(direction,axis=0))
        alpha=rz/np.tensordot(direction,q);phi+=alpha*direction;r-=alpha*q;previous=rz
        if np.linalg.norm(r)<1e-9*norm0:break
    return phi

def reconstruct(data,sigma=None,adaptive=True,kmax=10):
    image=prepare(data['image']);ks=data['reference_vectors'];norms=np.linalg.norm(ks,axis=1)
    if sigma is None:sigma=int(np.ceil(1/norms.min()))
    kw=norms.mean()/2.5;kstep=kw/3
    if adaptive:signals=np.stack([adaptive_lockin(image,k,sigma,kw,kstep)[0] for k in ks])
    else:signals=np.stack([lockin(image,k,sigma) for k in ks])
    phase=np.angle(signals);mask=np.zeros_like(image);dr=2*sigma;mask[dr:-dr,dr:-dr]=1
    weights=np.abs(signals)*(mask+1e-6)
    gx=weighted_projection(wrap_phase(np.diff(phase,axis=2)),ks,weights)
    gy=weighted_projection(wrap_phase(np.diff(phase,axis=1)),ks,weights)
    u=np.stack([integrate_gradient(gx[i],gy[i],np.linalg.norm(weights,axis=0),kmax) for i in range(2)])
    return u*NM_PER_PIXEL
