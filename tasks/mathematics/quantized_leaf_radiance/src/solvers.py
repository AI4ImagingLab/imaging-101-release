"""Local source-faithful TV/TGV primal-dual inverse updates.

Copyright(C)2013,2025 Kristian Bredies, source adaptations Apache-2.0.
General NumPy arithmetic replaces original CUDA kernels; source capture
fixtures predate this implementation. No domain solver is imported.
"""
import numpy as np
from .physics_model import gradient,divergence,symmetric_gradient,symmetric_divergence,vector_norm,tensor_norm

def project_vector(p,alpha):
    """Pixelwise joint RGB/spatial Euclidean dual ball projection."""
    norm=np.zeros(p.shape[:2],dtype=p.dtype)
    for channel in range(p.shape[2]):norm+=p[:,:,channel,0]**2+p[:,:,channel,1]**2
    factor=np.sqrt(norm)*np.float32(1/alpha)
    p*=np.reciprocal(np.maximum(factor,np.float32(1)))[...,None,None]
    return p

def project_tensor(q,alpha):
    """Joint RGB symmetric-tensor Frobenius ball, xy counted twice."""
    norm=np.zeros(q.shape[:2],dtype=q.dtype)
    for channel in range(q.shape[2]):norm+=q[:,:,channel,0]**2+q[:,:,channel,1]**2+np.float32(2)*q[:,:,channel,2]**2
    factor=np.sqrt(norm)*np.float32(1/alpha)
    q*=np.reciprocal(np.maximum(factor,np.float32(1)))[...,None,None]
    return q

def midpoint_prox(value,lower,upper,tau):
    """Prox of .5||u-mid||² plus the interval indicator."""
    midpoint=np.float32(.5)*(lower+upper)
    return np.clip((value-midpoint)/(np.float32(1)+tau)+midpoint,lower,upper)

def objective(u,lower,upper,alpha=.45,w=None,fac=2.):
    if np.any(u<lower) or np.any(u>upper):return float('inf')
    mid=np.float32(.5)*(lower+upper)
    fidelity=.5*float(np.sum((u.astype(np.float64)-mid)**2))
    if w is None:return fidelity+alpha*float(vector_norm(gradient(u)).sum(dtype=np.float64))
    return fidelity+alpha*float(vector_norm(gradient(u)-w).sum(dtype=np.float64))+fac*alpha*float(tensor_norm(symmetric_gradient(w)).sum(dtype=np.float64))

def reconstruct(lower,upper,alpha=.45,fac=2.,iterations=2000,method='tgv',record_every=100,return_state=False):
    """Return RGB float32 and actual sampled objective history.

    Same midpoint initialization and fixed source primal/dual step sizes;
    TV L²8, TGV L²12. No truth, cached reference or upstream package access.
    """
    if method not in ('tv','tgv') or alpha<=0 or fac<=0 or iterations<0:raise ValueError('invalid inverse parameters')
    lower=np.asarray(lower,dtype=np.float32);upper=np.asarray(upper,dtype=np.float32)
    u=np.float32(.5)*(lower+upper);ubar=u.copy()
    p=np.zeros(u.shape+(2,),dtype=np.float32)
    w=np.zeros_like(p) if method=='tgv' else None
    wbar=w.copy() if w is not None else None
    q=np.zeros(u.shape+(3,),dtype=np.float32) if w is not None else None
    bound=12. if w is not None else 8.
    tau=np.float32(1/np.sqrt(bound));sigma=np.float32(1/(1/np.sqrt(bound))/bound)
    history=[(0,objective(u,lower,upper,alpha,w,fac))]
    for iteration in range(1,iterations+1):
        p=project_vector(p+sigma*(gradient(ubar)-(wbar if w is not None else np.float32(0))),alpha)
        if w is not None:q=project_tensor(q+sigma*symmetric_gradient(wbar),fac*alpha)
        old_u=u
        u=midpoint_prox(u+tau*divergence(p),lower,upper,tau)
        if w is not None:
            old_w=w;w=w+tau*(p+symmetric_divergence(q));wbar=np.float32(2)*w-old_w
        ubar=np.float32(2)*u-old_u
        if record_every and (iteration%record_every==0 or iteration==iterations):history.append((iteration,objective(u,lower,upper,alpha,w,fac)))
    if return_state:
        state={'u':u,'u_':ubar,'p':p.reshape(u.shape[:2]+(u.shape[2]*2,))}
        if w is not None:state.update(w=w.reshape(u.shape[:2]+(u.shape[2]*2,)),w_=wbar.reshape(u.shape[:2]+(u.shape[2]*2,)),q=q.reshape(u.shape[:2]+(u.shape[2]*3,)))
        return u,np.asarray(history),state
    return u,np.asarray(history)

def reconstruct_tensor(lower,upper,alpha=.45,fac=2.,iterations=2000,method='tgv',device='cuda',return_state=False):
    """Explicit same primal-dual mathematics with general PyTorch tensors.

    Used by the active notebook for full native parameter comparisons;
    the independent NumPy CPU implementation remains the standard CLI.
    No trained weights, upstream kernels or optimizer wrapper is used.
    Returns float32 NumPy RGB, and optionally every final source-shaped state.
    """
    import torch
    if method not in ('tv','tgv') or alpha<=0 or fac<=0 or iterations<0:raise ValueError('invalid inverse parameters')
    lo=torch.as_tensor(lower,dtype=torch.float32,device=device);hi=torch.as_tensor(upper,dtype=torch.float32,device=device)
    midpoint=.5*(lo+hi);u=midpoint.clone();ubar=u.clone()
    p=torch.zeros((*u.shape,2),device=device,dtype=torch.float32)
    w=torch.zeros_like(p) if method=='tgv' else None
    wbar=w.clone() if w is not None else None
    q=torch.zeros((*u.shape,3),device=device,dtype=torch.float32) if w is not None else None
    bound=12. if w is not None else 8.;tau=float(np.float32(1/np.sqrt(bound)));sigma=float(np.float32(1/(1/np.sqrt(bound))/bound))
    def grad(x):
        g=torch.zeros((*x.shape,2),device=device,dtype=x.dtype)
        g[:-1,...,0]=x[1:]-x[:-1];g[:,:-1,...,1]=x[:,1:]-x[:,:-1]
        return g
    def div(x):
        r=torch.zeros(x.shape[:-1],device=device,dtype=x.dtype)
        r[:-1]+=x[:-1,...,0];r[1:]-=x[:-1,...,0]
        r[:,:-1]+=x[:,:-1,...,1];r[:,1:]-=x[:,:-1,...,1]
        return r
    def symgrad(x):
        a=grad(x[...,0]);b=grad(x[...,1]);return torch.stack([a[...,0],b[...,1],.5*(a[...,1]+b[...,0])],dim=-1)
    def symdiv(x):
        return torch.stack([div(torch.stack([x[...,0],x[...,2]],dim=-1)),div(torch.stack([x[...,2],x[...,1]],dim=-1))],dim=-1)
    def project(x,weight,tensor=False):
        n=torch.zeros(x.shape[:2],device=device,dtype=x.dtype)
        for channel in range(x.shape[2]):
            n+=x[:,:,channel,0]**2+x[:,:,channel,1]**2+(2*x[:,:,channel,2]**2 if tensor else 0.)
        scale=torch.reciprocal(torch.clamp(torch.sqrt(n)*float(np.float32(1/weight)),min=1.))
        return x*scale[...,None,None]
    with torch.no_grad():
        for iteration in range(iterations):
            p=project(p+sigma*(grad(ubar)-(wbar if w is not None else 0.)),alpha)
            if w is not None:q=project(q+sigma*symgrad(wbar),fac*alpha,True)
            old_u=u
            u=torch.minimum(torch.maximum((u+tau*div(p)-midpoint)/(1.+tau)+midpoint,lo),hi)
            if w is not None:
                old_w=w;w=w+tau*(p+symdiv(q));wbar=2*w-old_w
            ubar=2*u-old_u
    out=u.cpu().numpy()
    if return_state:
        state={'u':out,'u_':ubar.cpu().numpy(),'p':p.reshape(*u.shape[:2],u.shape[2]*2).cpu().numpy()}
        if w is not None:state.update(w=w.reshape(*u.shape[:2],u.shape[2]*2).cpu().numpy(),w_=wbar.reshape(*u.shape[:2],u.shape[2]*2).cpu().numpy(),q=q.reshape(*u.shape[:2],u.shape[2]*3).cpu().numpy())
        return out,state
    return out
