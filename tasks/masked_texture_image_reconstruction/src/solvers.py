"""Explicit weighted primal-dual TV/TGV updates; GPL-3.0-or-later."""
import numpy as np
from scipy.linalg import svd
from .physics_model import gradient,gradient_adjoint,symgrad,symgrad_adjoint,symgrad2,symgrad2_adjoint

def project_ball(p,radius=1.0,weights=None):
    """Pointwise projection in the declared tensor Hilbert metric."""
    z=p.copy()
    square=np.square(p) if weights is None else np.asarray(weights)[None,None,:]*np.square(p)
    z/=np.maximum(np.sqrt(square.sum(axis=2,keepdims=True))/radius,1.0)
    return z

def group_norm(p,weights=None):
    square=np.square(p) if weights is None else np.asarray(weights)[None,None,:]*np.square(p)
    return np.sqrt(square.sum(axis=2)).sum()

def reconstruct(observations,mask,order=2,niter=8000,alpha1=np.sqrt(2.),alpha2=np.sqrt(2.),capture_steps=()):
    """Return dictionary with u, objective, stage captures and auxiliary fields.

    Every field uses native row,column axes, float64 and unit pixel spacing.
    Fixed iteration budget is preserved; no convergence claim or final clipping.
    """
    y=np.asarray(observations,dtype=np.float64);mask=np.asarray(mask,dtype=bool)
    if y.ndim!=2 or min(y.shape)<2 or mask.shape!=y.shape or not np.isfinite(y).all():raise ValueError('Invalid image/mask')
    if order not in (1,2,3) or not isinstance(niter,int) or niter<1:raise ValueError('Invalid order or iteration budget')
    if not np.isfinite([alpha1,alpha2]).all() or min(alpha1,alpha2)<=0:raise ValueError('Invalid regularization weights')
    norms=np.diag([np.sqrt(8.)]*order)+np.diag([1.]*(order-1),k=1)
    norm=float(svd(norms,compute_uv=False)[0]);sig=100./norm;tau=1./(norm*100.)
    u=np.zeros_like(y);ux=u.copy();p=np.zeros(y.shape+(2,))
    v=np.zeros_like(p);vx=v.copy();q=np.zeros(y.shape+(3,));w=np.zeros_like(q);wx=w.copy();r=np.zeros(y.shape+(4,))
    history=np.zeros(niter+1);snapshots={}
    def objective():
        val=np.abs(u[mask]-y[mask]).sum()
        if order==1:return val+group_norm(gradient(u))
        if order==2:return val+group_norm(gradient(u)-v)+alpha1*group_norm(symgrad(v),[1,1,2])
        return val+group_norm(gradient(u)-v)+alpha1*group_norm(symgrad(v)-w,[1,1,2])+alpha2*group_norm(symgrad2(w),[1,1,3,3])
    history[0]=objective()
    for k in range(niter):
        p=project_ball(p+sig*(gradient(ux) if order==1 else gradient(ux)-vx))
        if order>1:q=project_ball(q+sig*(symgrad(vx) if order==2 else symgrad(vx)-wx),alpha1,[1,1,2])
        if order>2:r=project_ball(r+sig*symgrad2(wx),alpha2,[1,1,3,3])
        new=u-tau*gradient_adjoint(p);new[mask]=y[mask];ux=2.0*new-u;u=new
        if order>1:
            new=v-tau*(-p+symgrad_adjoint(q));vx=2.0*new-v;v=new
        if order>2:
            new=w-tau*(-q+symgrad2_adjoint(r));wx=2.0*new-w;w=new
        if k+1 in capture_steps:
            fields=dict(u=u,p=p)
            if order>1:fields.update(v=v,q=q)
            if order>2:fields.update(w=w,r=r)
            snapshots[k+1]={name:value.copy() for name,value in fields.items()}
        history[k+1]=objective()
    return dict(u=u,v=v,w=w,objective=history,snapshots=snapshots,sig=sig,tau=tau,operator_norm=norm)

