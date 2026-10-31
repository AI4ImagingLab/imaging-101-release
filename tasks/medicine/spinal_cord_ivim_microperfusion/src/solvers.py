"""Local extraction of Lévy IVIM one-step v2 and fixed-D two-step fits.

Original task-specific equations/bounds are MIT licensed; see LICENSE.
Generic Minuit box transform is evaluated explicitly, no lmfit dependency.
"""
import numpy as np
from scipy.optimize import minimize,differential_evolution
from .physics_model import signal,log_signal

def from_internal(u,lower,upper):
    return np.array([min(float(hi),max(float(lo),float(lo)+(np.sin(v)+1)*(float(hi)-float(lo))/2.)) for v,lo,hi in zip(u,lower,upper)])

def to_internal(p,lower,upper):
    u=np.arcsin(2*(np.clip(p,lower,upper)-lower)/(upper-lower)-1)
    u[np.abs(u)<1e-15]=0.
    return u

class _EvaluationLimit(Exception):pass

def bounded_fit(model,data,initial,lower,upper,method,rng=None):
    initial=np.asarray(initial,float);lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    max_nfev=2000*(len(initial)+1);calls=0;last=None
    def objective(u):
        nonlocal calls,last
        calls+=1;last=np.array(u,copy=True)
        if calls>max_nfev:raise _EvaluationLimit
        residual=data-model(from_internal(u,lower,upper))
        return (residual*residual).sum()
    capped=False
    try:
        if method=='cg':
            result=minimize(objective,to_internal(initial,lower,upper),method='CG',options={'maxiter':2*max_nfev})
        else:
            result=differential_evolution(objective,[(-np.pi/2,np.pi/2)]*len(initial),strategy='best1bin',maxiter=max_nfev,popsize=15,tol=.01,mutation=(.5,1),recombination=.7,seed=rng,polish=True,init='latinhypercube',atol=0,updating='immediate',workers=1)
        u=result.x
    except _EvaluationLimit:
        capped=True;u=last
    parameters=from_internal(u,lower,upper)
    return parameters,{'evaluations':calls,'capped':capped}

def fit_voxel(values,b,seed=20261027,method='one_step',capture=False):
    if method not in ('one_step','two_step'):raise ValueError('Unknown fitting method')
    rng=np.random.RandomState(seed);stages=[]
    keep=b>=(400 if method=='one_step' else 500)
    slope,intercept=np.polyfit(b[keep],np.log(values[keep]),1)
    p,info=bounded_fit(lambda p:log_signal(b[keep],*p),np.log(values[keep]),[intercept,-slope],[.5*intercept,.00015],[1.5*intercept,.0045],'cg')
    def record(parameters,prediction,data,info):
        stages.append({'params':parameters.copy(),'prediction':prediction.copy(),'residual':data-prediction,'chisqr':float(np.sum((data-prediction)**2)),**info})
    record(p,log_signal(b[keep],*p),np.log(values[keep]),info)
    D=p[1];base=np.exp(p[0]);fixed=method=='two_step'
    lower=np.array([.7*base,.00015,0,2*D]);upper=np.array([1.7*base,1.2*D,.2,.05]);initial=np.array([1.2*base,D,.1,10*D])
    if fixed:lower[3]=D;upper[3]=.0355;upper[2]=.35;initial[3]=(D+.0355)/2
    active=np.array([0,2,3]) if fixed else np.arange(4)
    def expand(v):
        out=initial.copy();out[active]=v;return out
    for iteration in range(2):
        p,info=bounded_fit(lambda v:signal(b,*expand(v)),values,initial[active],lower[active],upper[active],'de',rng)
        p=expand(p);record(p,signal(b,*p),values,info)
        if iteration==0:
            initial=p.copy();lower=.95*p;upper=1.05*p
            if p[2]<=1e-12:lower[2]=0;upper[2]=.05
    return (p,stages) if capture else p

def reconstruct(image,mask,b,method='one_step',seed=20261027):
    result=np.zeros((*mask.shape,4),dtype=float)
    for index,coord in enumerate(np.argwhere(mask)):
        loc=tuple(coord);result[loc]=fit_voxel(image[loc],b,seed+index,method)
    return result
