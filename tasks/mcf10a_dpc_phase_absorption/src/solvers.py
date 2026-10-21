"""Tikhonov direct inverse and original third-order-TV ADMM comparison."""
import numpy as np
from .preprocessing import normalize
from .physics_model import transfer_functions,forward
REG_ABSORPTION=.1
REG_PHASE=.005
def normal_matrix(hu,hp,reg_u=REG_ABSORPTION,reg_p=REG_PHASE):
    return [(hu.conj()*hu).sum(0)+reg_u,(hu.conj()*hp).sum(0),(hp.conj()*hu).sum(0),(hp.conj()*hp).sum(0)+reg_p]
def invert(a,b):
    det=a[0]*a[3]-a[1]*a[2]
    return np.fft.ifft2((a[3]*b[0]-a[1]*b[1])/det).real,np.fft.ifft2((a[0]*b[1]-a[2]*b[0])/det).real
def third_difference(x,axis):return 3*x-np.roll(x,-1,axis)-3*np.roll(x,1,axis)+np.roll(x,2,axis)
def shrink(x,threshold):
    magnitude=np.sqrt(np.sum(x.conj()*x,axis=0))
    # The selected actual run has no exactzero magnitudes; define mathematical
    # continuous limit atzero for analytic tests, rather than permit0/0.
    ratio=np.divide(np.maximum(magnitude-threshold,0),magnitude,out=np.zeros_like(magnitude),where=magnitude!=0)
    return x*ratio[None]
def tv_solve(images,model,iterations=10):
    hu,hp=model['Hu'],model['Hp'];shape=images.shape[-2:]
    dx=np.zeros(shape,dtype='complex64');dy=dx.copy()
    dx[0,0]=3;dx[0,-1]=-1;dx[0,1]=-3;dx[0,2]=1
    dy[0,0]=3;dy[-1,0]=-1;dy[1,0]=-3;dy[2,0]=1
    fx=np.fft.fft2(dx);fy=np.fft.fft2(dy)
    a=normal_matrix(hu,hp);penalty=fx*fx.conj()+fy*fy.conj();a[0]+=penalty;a[3]+=penalty
    fi=np.fft.fft2(images);base=[np.sum(hu.conj()*fi,axis=0),np.sum(hp.conj()*fi,axis=0)]
    z=np.zeros((4,)+shape,dtype='complex64');u=z.copy();d=z.copy();history=[]
    for it in range(iterations):
        y=np.fft.fft2(z-u)
        # Sum the complex64 derivative terms BEFORE adding complex128 data
        # terms, preserving the original ADMM arithmetic convention.
        b=[base[0]+(fx.conj()*y[0]+fy.conj()*y[1]),base[1]+(fx.conj()*y[2]+fy.conj()*y[3])]
        absorption,phase=invert(a,b)
        history.append(float(np.linalg.norm(forward(absorption,phase,hu,hp)-images)))
        if it<iterations-1:
            d[0]=third_difference(absorption,1);d[1]=third_difference(absorption,0)
            d[2]=third_difference(phase,1);d[3]=third_difference(phase,0)
            z=d+u;z[:2]=shrink(z[:2],1);z[2:]=shrink(z[2:],1);u+=d-z
    return {'reconstruction':phase[None],'absorption':absorption[None]},np.asarray(history)
def reconstruct(data,method='Tikhonov',reg_u=REG_ABSORPTION,reg_p=REG_PHASE):
    images=normalize(data['intensities']);m=transfer_functions(images.shape[-2:]);hu,hp=m['Hu'],m['Hp']
    if method=='TV':result,history=tv_solve(images,m)
    elif method=='Tikhonov':
        fi=np.fft.fft2(images);a=normal_matrix(hu,hp,reg_u,reg_p);b=[(hu.conj()*fi).sum(0),(hp.conj()*fi).sum(0)]
        absorption,phase=invert(a,b);result={'reconstruction':phase[None],'absorption':absorption[None]};history=np.empty(0)
    else:raise ValueError('Unknown method')
    return result,{**m,'normalized':images,'history':history}
