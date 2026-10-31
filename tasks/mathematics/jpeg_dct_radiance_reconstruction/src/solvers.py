"""Author projected accelerated normalized subgradient iteration; GPL-3.0-or-later."""
import numpy as np
from .physics_model import inverse,transform,tv_gradient
def reconstruct(coeff,q,iterations=2000):
    q=q.astype(np.float32)[:,None,None];center=coeff*q
    u=inverse(center);prev=u.copy();t=np.float32(1)
    step=np.sqrt(np.float32(u.shape[1]*u.shape[2]))/2/np.sqrt(np.float32(1+iterations))
    alpha=np.float32(.001)*2*np.float32(255)*np.sqrt(np.float32(2))
    cos=center.copy()
    for i in range(iterations):
        tn=(np.float32(1)+np.sqrt(np.float32(1)+4*t*t))/2
        old=u.copy();u=u+(t-1)/tn*(u-prev);prev=old;t=tn
        g=alpha*inverse((cos-center)/(q*q))+tv_gradient(u)
        for c in range(3):
            n=np.float32(np.sqrt(np.sum((g[c]*g[c]).astype(float))))
            if n:u[c]-=step*(g[c]/n)
        cos=np.clip(transform(u),center-.5*q,center+.5*q)
        u=inverse(cos)
    return u
