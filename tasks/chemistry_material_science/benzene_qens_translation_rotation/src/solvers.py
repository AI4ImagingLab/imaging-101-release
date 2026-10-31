"""Box-constrained point inference with explicit nuisance profiling.

Uniform priors are in physical N, not 1/N. Profiling is an optimization
coordinate change, never a posterior marginalization or a new prior.
"""
import numpy as np
from scipy.optimize import minimize
from .physics_model import LOWER,UPPER,kernel_basis,predict,weighted_sse

def profile_energy(basis,y,sigma):
    w=1/sigma**2
    s0=w.sum(axis=1);s1=(w*basis).sum(axis=1)
    ybar=(w*y).sum(axis=1)/s0;hbar=s1/s0
    hlo,hhi=1/67.5,1/.0225;blo,bhi=.001,10.
    knots=[hlo,hhi]
    for b in (blo,bhi):
        a=np.divide(s0*(ybar-b),s1,out=np.full_like(s1,np.nan),where=s1!=0)
        knots.extend(a[(a>hlo)&(a<hhi)].tolist())
    knots=np.unique(knots);candidates=knots.tolist()
    for left,right in zip(knots[:-1],knots[1:]):
        middle=(left+right)/2;unclipped=ybar-middle*hbar
        free=(unclipped>blo)&(unclipped<bhi)
        b=np.clip(unclipped,blo,bhi)
        hc=np.where(free[:,None],basis-hbar[:,None],basis)
        yc=np.where(free[:,None],y-ybar[:,None],y-b[:,None])
        denominator=np.sum(w*hc**2)
        numerator=np.sum(w*hc*yc)
        candidates.append(float(np.clip(numerator/denominator if denominator>0 else middle,left,right)))
    best=None
    for a in candidates:
        b=np.clip(ybar-a*hbar,blo,bhi)
        prediction=a*basis+b[:,None]
        objective=float(np.sum(w*(prediction-y)**2))
        if best is None or objective<best[0]:best=(objective,1/a,b,prediction)
    return best

def profile(global_parameters,data):
    p=np.empty(31);p[:3]=global_parameters;predictions={};objective=0.
    for name,index,start,stop in [('360',3,4,16),('197',16,17,31)]:
        d=data[name];value,N,b,curve=profile_energy(kernel_basis(p[:3],d),d['y'],d['sigma'])
        p[index]=N;p[start:stop]=b;predictions[name]=curve;objective+=value
    return objective,p,predictions

def reconstruct(data):
    lo=LOWER[:3];span=UPPER[:3]-lo
    def fun(x):return profile(lo+span*np.asarray(x),data)[0]
    grid=np.array(np.meshgrid(*([np.array([.1,.5,.9])]*3),indexing='ij')).reshape(3,-1).T
    scores=np.array([fun(x) for x in grid]);results=[]
    for index in np.argsort(scores)[:4]:
        result=minimize(fun,grid[index],method='L-BFGS-B',jac='3-point',bounds=[(0.,1.)]*3,options={'ftol':1e-13,'gtol':1e-5,'maxiter':500,'maxls':40,'finite_diff_rel_step':2e-5})
        results.append(result)
    best=min(results,key=lambda r:r.fun)
    value,p,curves=profile(lo+span*best.x,data)
    return p,curves,dict(weighted_sse=value,starts=4,grid_points=27,iterations=int(best.nit),optimizer_success=bool(best.success),optimizer_message=str(best.message))
