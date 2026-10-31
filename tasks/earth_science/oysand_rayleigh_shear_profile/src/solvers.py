"""Explicit original immediate-update CPSO with shrinking boundary repair.

BSD3-Clause adaptation of Keurfon Luu's stochopy and evodcinv.
"""
import numpy as np
from .physics_model import phase_velocity,DispersionError
from .preprocessing import BOUNDS,initial_population

def physical_model(parameters):
    thickness=np.append(parameters[:3],1.)
    vs=parameters[3:7];nu=parameters[7:]
    vp=vs*((1-nu)/(.5-nu))**.5
    return np.column_stack([thickness,vp,vs,np.full(4,1.95)])

def objective(parameters,data):
    if np.any(np.diff(parameters[3:7])<0):return np.inf
    model=physical_model(parameters)
    try:predicted=phase_velocity(data['period'],*model.T)
    except DispersionError:return np.inf
    if len(predicted)!=len(data['period']) or np.any(predicted<=0):return np.inf
    return float(np.sqrt(np.mean(((predicted-data['phase_velocity'])/data['uncertainty'])**2)))

def shrink(position,velocity,lower,upper):
    candidate=position+velocity;maskl=candidate<lower;masku=candidate>upper
    if maskl.any() and masku.any():factor=min(((lower[maskl]-position[maskl])/velocity[maskl]).min(),((upper[masku]-position[masku])/velocity[masku]).min())
    elif maskl.any():factor=((lower[maskl]-position[maskl])/velocity[maskl]).min()
    elif masku.any():factor=((upper[masku]-position[masku])/velocity[masku]).min()
    else:factor=1.
    velocity=velocity*factor
    return position+velocity,velocity

def competitive_swarm(data,seed=24713,popsize=40,maxiter=300,callback=None):
    rng=np.random.RandomState(seed);lower,upper=BOUNDS.T;ndim=len(lower)
    x=initial_population(rng,popsize);v=np.zeros_like(x);personal=x.copy()
    fit=np.array([objective(p,data) for p in x]);personal_fit=fit.copy()
    index=np.argmin(fit);best=x[index].copy();best_fit=fit[index]
    history=[x.copy()];losses=[fit.copy()];global_loss=[best_fit]
    delta=np.log(1+.003*popsize)/max(.2,np.log(.01*maxiter))
    for iteration in range(2,maxiter+1):
        r1=rng.rand(popsize,ndim);r2=rng.rand(popsize,ndim);status=None
        for i in range(popsize):
            velocity=.7298*v[i]+1.49618*r1[i]*(personal[i]-x[i])+1.49618*r2[i]*(best-x[i])
            x[i],v[i]=shrink(x[i],velocity,lower,upper)
            fit[i]=objective(x[i],data);status=None
            if fit[i]<=personal_fit[i]:
                personal[i]=x[i].copy();personal_fit[i]=fit[i]
                if fit[i]<=best_fit:
                    difference=np.linalg.norm(best-x[i]);best=x[i].copy();best_fit=fit[i]
                    if difference<=1e-8 and best_fit<=1e-8:status=0
                    elif best_fit<=1e-8:status=1
        history.append(x.copy());losses.append(fit.copy());global_loss.append(best_fit)
        if callback is not None:callback(iteration,best_fit)
        if status is not None or iteration>=maxiter:break
        radius=max(np.linalg.norm(p-best) for p in x)/np.sqrt(4.*ndim)
        if radius<delta:
            nw=int((popsize-1)/(1+np.exp((iteration/maxiter-1.+.5)/.09)))
            if nw>0:
                idx=personal_fit.argsort()[:-nw-1:-1]
                v[idx]=0;x[idx]=rng.uniform(lower,upper,(nw,ndim));personal[idx]=x[idx].copy();personal_fit[idx]=1e30
    return best,{'parameters':np.asarray(history),'objectives':np.asarray(losses),'global_objectives':np.asarray(global_loss),'best_objective':best_fit}

def reconstruct(data,seed=24713,popsize=40,maxiter=300,callback=None):
    best,stages=competitive_swarm(data,seed,popsize,maxiter,callback)
    return {'reconstruction':(best[3:7]*1000)[None],'thickness':(best[:3]*1000)[None]},stages
