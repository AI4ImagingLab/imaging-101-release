"""Explicit original alternating PSF estimation and cross-channel transfer.

Sun/Peng/Heidrich ICCV2017 mathematical extraction, CC BY-NC4.0.
Patches are streamed to reduce memory; their order and accumulation are retained.
"""
import numpy as np
from .physics_model import positions,gradient,weights,psf_prior,estimate_psf,features

def reconstruct_channel(base,blur,psf_size=500,transfer_size=60,alpha=.3,beta=.3,iterations=3,capture=False):
    shape=base.shape;ps= int(np.ceil(psf_size/5));cs=int(np.ceil(transfer_size/5))
    wp,sp=weights(shape,psf_size,ps);wc,sc=weights(shape,transfer_size,cs)
    coords=positions(shape,transfer_size,cs);pcoords=positions(shape,psf_size,ps)
    gx,gy=gradient(base);out=np.zeros(shape)
    for i,j in coords:
        b=base[i:i+transfer_size,j:j+transfer_size];y=blur[i:i+transfer_size,j:j+transfer_size]
        out[i:i+transfer_size,j:j+transfer_size]+=wc*(y.mean()/b.mean())*b
    out/=sc;diag={'initial':out.copy(),'iteration':[]};prior=psf_prior(psf_size,alpha,beta)
    for iteration in range(iterations):
        blurred_base=np.zeros(shape);psfs=[];first=None
        for k,(i,j) in enumerate(pcoords):
            sl=np.s_[i:i+psf_size,j:j+psf_size]
            h=estimate_psf(out[sl],blur[sl],prior)
            if k==0:first=h.copy()
            blurred_base[sl]+=wp*np.fft.ifft2(h*np.fft.fft2(base[sl])).real
            kernel=np.fft.ifft2(h).real;idx=np.r_[np.arange(psf_size-24,psf_size),np.arange(25)]
            psfs.append(kernel[np.ix_(idx,idx)])
        blurred_base/=sp;bx,by=gradient(blurred_base);out=np.zeros(shape);coefficients=[]
        for k,(i,j) in enumerate(coords):
            mat=features(blurred_base,bx,by,i,j,transfer_size)
            target=blur[i:i+transfer_size,j:j+transfer_size].ravel(order='F')
            normal=mat.T@mat;rhs=mat.T@target;coef=np.linalg.solve(normal,rhs)
            if k==0:normal_first=normal.copy();rhs_first=rhs.copy()
            coefficients.append(coef)
            pred=features(base,gx,gy,i,j,transfer_size)@coef
            out[i:i+transfer_size,j:j+transfer_size]+=wc*pred.reshape((transfer_size,transfer_size),order='F')
        out/=sc
        if capture:diag['iteration'].append(dict(psf_ft_first=first,blur_base=blurred_base.copy(),A_first=normal_first,b_first=rhs_first,coefficients=np.array(coefficients).T,deblur=out.copy()))
    return out,np.stack(psfs,axis=-1),diag

def reconstruct(image,**kwargs):
    result=image.copy();details={}
    for c in (0,2):
        result[:,:,c],psf,diag=reconstruct_channel(image[:,:,1],image[:,:,c],**kwargs)
        details[c]=dict(psf=psf,**diag)
    return result[None],details
