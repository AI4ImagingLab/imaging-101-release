"""Bounded Nelder-Mead inverse updates; adapted from Bumps simplex.
From optimize.py module by Travis E. Oliphant.
You may copy and use this module as you see fit with no guarantee implied
provided you keep this notice in all copies. Modified by Paul Kienzle for bounds.
Local convergence reporting corrects upstream's reversed Result status label.
"""
import numpy as np
from .physics_model import predictions,depth_profiles
INITIAL=np.array([1.,.026,0.,5.,5.,10.,.1,5.,160.])
LOWER=np.array([.98,0.,-.02,1.,1.,0.,0.,2.,60.])
UPPER=np.array([1.02,.08,.02,15.,15.,30.,2.,7.,180.])

def bounded_simplex(objective,initial,lower,upper,radius=.15,xtol=1e-7,ftol=1e-9,maxiter=15000):
    initial=np.asarray(initial,float);n=len(initial);points=np.tile(initial,(n+1,1));history=[]
    endpoint=np.clip(initial+(upper-lower)*radius,lower,upper)
    collide=endpoint==initial;endpoint[collide]=initial[collide]-(upper[collide]-lower[collide])*radius
    for j in range(n):points[j+1,j]=endpoint[j]
    calls=0
    def evaluate(x):
        nonlocal calls
        calls+=1
        return np.inf if np.any((x<lower)|(x>upper)) else objective(x)
    values=np.array([evaluate(x) for x in points]);order=np.argsort(values);points=points[order];values=values[order]
    iteration=1
    while iteration<maxiter:
        size=np.max(abs(points[1:]-points[0])/(upper-lower));spread=np.max(abs(values[0]-values[1:]))
        if size<=xtol and spread<=ftol:break
        center=points[:-1].sum(axis=0)/n;reflected=2*center-points[-1];vr=evaluate(reflected);shrink=False
        if vr<values[0]:
            expanded=3*center-2*points[-1];ve=evaluate(expanded)
            if ve<vr:points[-1]=expanded;values[-1]=ve
            else:points[-1]=reflected;values[-1]=vr
        elif vr<values[-2]:points[-1]=reflected;values[-1]=vr
        elif vr<values[-1]:
            contracted=1.5*center-.5*points[-1];vc=evaluate(contracted)
            if vc<=vr:points[-1]=contracted;values[-1]=vc
            else:shrink=True
        else:
            contracted=.5*center+.5*points[-1];vc=evaluate(contracted)
            if vc<values[-1]:points[-1]=contracted;values[-1]=vc
            else:shrink=True
        if shrink:
            for j in range(1,n+1):points[j]=points[0]+.5*(points[j]-points[0]);values[j]=evaluate(points[j])
        order=np.argsort(values);points=points[order];values=values[order]
        if iteration%100==0:history.append([iteration,values[0]])
        iteration+=1
    success=iteration<maxiter and size<=xtol and spread<=ftol
    if not success:raise RuntimeError('Simplex did not meet both declared convergence criteria')
    return dict(parameters=points[0],objective=float(values[0]),iterations=iteration,calls=calls,scaled_size=float(size),objective_spread=float(spread),history=np.array(history))

def fit_model(data,dead=True,initial=None):
    keep=np.arange(9) if dead else np.array([0,1,2,3,4,6,7,8])
    start=INITIAL[keep].copy() if initial is None else np.asarray(initial,float)
    def objective(p):
        residual=(predictions(p,data,dead)-data['reflectivity'])/data['uncertainty']
        return .5*np.sum(residual**2)
    result=bounded_simplex(objective,start,LOWER[keep],UPPER[keep])
    result['prediction']=predictions(result['parameters'],data,dead)
    result['profiles']=depth_profiles(result['parameters'],data['depth'],dead)
    result['reduced_chi_square']=2*result['objective']/(data['reflectivity'].size-len(start))
    return result

def reconstruct(data):
    result=fit_model(data,True)
    return dict(reconstruction=result['profiles'][None],parameters=result['parameters'][None]),result
