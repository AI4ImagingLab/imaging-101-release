"""Explicit local extraction of Wang (2023) spectral ADMM.

Copyright (c) 2023 Ruogu Wang. Adapted under MIT license; see LICENSE.
The reweighting, residual checks and signed scaled dual convention deliberately
follow the published implementation, including checks only every ten iterations.
"""
import numpy as np
from .preprocessing import detector_units,window
from .physics_model import poisson_prox

def group_shrink(x,tau):
    norms=np.sqrt(np.sum(x*x,axis=1,keepdims=True))
    weights=tau/(norms+np.finfo(float).eps)
    a=np.maximum(0,norms-weights)
    return (a/(a+weights))*x if tau else x.copy()

def solve_window(y,m,lambda_rank=0.1,lambda_group=0.1,poisson=True,capture=False):
    """Return R×N abundances, central R-vector and iteration diagnostics.

    y is C×N, m is C×R. Zero windows return zero without dividing by zero.
    Capture stores copies at iterations 1,11,101,501 for upstream parity.
    """
    scale=np.sqrt(np.mean(y*y));c,n=y.shape;r=m.shape[1]
    if scale==0: return np.zeros((r,n)),np.zeros(r),{'iterations':0,'snapshots':{}}
    y=y/scale
    gram=m.T@m
    u=np.maximum(np.linalg.inv(gram)@m.T@y,0)
    inv=np.linalg.inv(gram+3*np.eye(r))
    v=[m@u,u.copy(),u.copy(),u.copy()]
    d=[np.zeros_like(z) for z in v]
    mu=1e-2;tol=np.sqrt((3*r+c)*n)*1e-4;snapshots={}
    for iteration in range(1,1001):
        check=iteration%10==1
        if check: old=[z.copy() for z in v]
        u=inv@(m.T@(v[0]+d[0])+(v[1]+d[1])+(v[2]+d[2])+(v[3]+d[3]))
        mu_u=m@u
        v[0]=poisson_prox(mu_u-d[0],y,mu) if poisson else (y+mu*(mu_u-d[0]))/(1+mu)
        left,sv,right=np.linalg.svd(u-d[1],full_matrices=False)
        shrunk=np.maximum(sv-(lambda_rank/mu)/(sv+np.finfo(float).eps),0)
        v[1]=(left*shrunk)@right
        v[2]=group_shrink(u-d[2],lambda_group/mu)
        v[3]=np.maximum(u-d[3],0)
        for k,op in enumerate([mu_u,u,u,u]):d[k]=d[k]-op+v[k]
        if check:
            rp=np.linalg.norm(np.concatenate([mu_u-v[0],u-v[1],u-v[2],u-v[3]],axis=0))
            rd=mu*np.linalg.norm(m.T@(v[0]-old[0])+(v[1]-old[1])+(v[2]-old[2])+(v[3]-old[3]))
            if rp>10*rd:mu*=2;d=[z/2 for z in d]
            elif rd>10*rp:mu/=2;d=[z*2 for z in d]
        if capture and iteration in (1,11,101,501):
            snapshots[iteration]={'U':u.copy(),'mu':mu,'res_p':rp,'res_d':rd,'Y':y.copy(),'tol':tol,**{f'V{k+1}':z.copy() for k,z in enumerate(v)},**{f'D{k+1}':z.copy() for k,z in enumerate(d)}}
        if rp<tol and rd<tol:break
    u=np.maximum(u*scale,0)
    return u,u[:,int(np.floor(n/2))],{'iterations':iteration,'primal':rp,'dual':rd,'snapshots':snapshots}

def reconstruct(image,spectra,method='SLPRU',lambda_rank=0.1,lambda_group=0.1):
    """Reconstruct complete image; return max-normalized (H,W,R) and iterations.

    Methods NLS/SNLS/SLNLS are fresh upstream comparison methods. The one-pixel
    exterior border is explicitly zero, following the acquisition ROI contract.
    """
    if method not in ('NLS','SNLS','SLNLS','SLPRU'):raise ValueError('unknown method')
    t=detector_units(image);h,w,_=t.shape
    out=np.zeros((h,w,spectra.shape[1]));iters=np.zeros((h,w),dtype=int)
    l1=lambda_rank if method in ('SLPRU','SLNLS') else 0
    l2=lambda_group if method!='NLS' else 0
    for i in range(1,h-1):
        for j in range(1,w-1):
            if np.any(t[i,j]):
                _,out[i,j],diag=solve_window(window(t,i,j),spectra,l1,l2,method=='SLPRU')
                iters[i,j]=diag['iterations']
    maximum=float(out.max())
    if maximum>0:out/=maximum
    return out,{'iterations':iters,'brightness_scale':maximum}
