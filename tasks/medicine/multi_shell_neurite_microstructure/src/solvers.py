"""Explicit batched extraction of the author's three-vertex Nelder-Mead.

BSD-2-Clause SMT attribution: Enrico Kaden & University College London2016.
Vertices retain storage identities; the unusual storage-slot width criterion
is preserved. An explicit bounded insertion sort reproduces the original
three-element libstdc++ tie ordering without an invalid generic comparator.
"""
import numpy as np
from .physics_model import signal_objective,transform_parameters

def fit_isotropic_signals(observed,bvalues,absolute_precision=1e-12):
    """Review-only v=0 model comparison: bounded one-dimensional golden search.

    Fit the isotropic exp(-bD) null, not a second source-reproduced method.
    Return diffusivity(n,) and full directional intensity SSE(n,). The known
    b0 signal remains fixed. Equal-shell sufficient statistics preserve its
    minimizer; full residuals are recomputed for reported objective values.
    """
    y=np.asarray(observed,float);b=np.asarray(bvalues,float).ravel();positive=b>0
    shells=np.unique(b[positive]);baseline=y[:,b==0].mean(1)
    means=np.column_stack([y[:,b==value].mean(1) for value in shells]);weights=np.array([(b==value).sum() for value in shells])
    lo=np.zeros(len(y));hi=np.full(len(y),3.05e-3);ratio=(np.sqrt(5)-1)/2
    def objective(d):return np.sum(weights[None,:]*(means-baseline[:,None]*np.exp(-d[:,None]*shells))**2,axis=1)
    for unused in range(100):
        if np.max(hi-lo)<=absolute_precision:break
        left=hi-ratio*(hi-lo);right=lo+ratio*(hi-lo);choose_left=objective(left)<=objective(right)
        hi=np.where(choose_left,right,hi);lo=np.where(choose_left,lo,left)
    d=(lo+hi)/2;predicted=baseline[:,None]*np.exp(-d[:,None]*b[None,positive])
    return d,np.sum((y[:,positive]-predicted)**2,axis=1)

def _sort_three(values,indices):
    """Original small stable_sort insertion behavior: equal keys reverse."""
    order=indices.copy();rows=np.arange(len(order))
    for position in [1,2]:
        current=order[:,position].copy();j=np.full(len(order),position)
        for unused in range(position):
            previous=np.maximum(j-1,0);move=(j>0)&(values[rows,current]<=values[rows,order[rows,previous]])
            active=np.flatnonzero(move)
            order[active,j[active]]=order[active,j[active]-1];j[active]-=1
        order[rows,j]=current
    return order

def _sort_last(values,indices):
    """Source strict-less insertion of the updated worst vertex."""
    order=indices.copy();rows=np.arange(len(order));current=order[:,2].copy();j=np.full(len(order),2)
    for unused in range(2):
        previous=np.maximum(j-1,0);move=(j>0)&(values[rows,current]<values[rows,order[rows,previous]])
        active=np.flatnonzero(move);order[active,j[active]]=order[active,j[active]-1];j[active]-=1
    order[rows,j]=current
    return order

def fit_signals(observed,bvalues,max_iter=10000,return_history=False):
    """Fit independent rows using source reflection/expansion/contraction rules.

    Returns (n,2) parameters(v,D) and actual iteration/stopping diagnostics.
    b0 is observed and fixed. No parameter reference or target mask is read.
    """
    y=np.asarray(observed,float);b=np.asarray(bvalues,float).ravel()
    if y.ndim!=2 or y.shape[1]!=len(b) or not np.isfinite(y).all() or not np.any(b==0):raise ValueError('Finite signal rows and observed b0 required')
    n=len(y);rows=np.arange(n);x=np.zeros((n,3,2));x[:,1,0]=1;x[:,2,1]=1
    values=np.column_stack([signal_objective(x[:,k],y,b) for k in range(3)])
    order=_sort_three(values,np.tile(np.arange(3),(n,1)));active=np.ones(n,bool);iterations=np.zeros(n,np.int32)
    parameter_history=[];objective_history=[];relative=1000*np.finfo(float).eps;absolute=10*np.finfo(float).eps
    def evaluate(candidate,which):return signal_objective(candidate,y[which],b)
    for iteration in range(max_iter):
        a=np.flatnonzero(active)
        if not len(a):break
        best=order[a,0];middle=order[a,1];worst=order[a,2]
        centroid=(x[a,best]+x[a,middle])/2
        reflected=2*centroid-x[a,worst];fr=evaluate(reflected,a)
        fb=values[a,best];fm=values[a,middle];fw=values[a,worst]
        chosen=reflected.copy();fc=fr.copy();shrink=np.zeros(len(a),bool)
        expand=fr<fb
        if expand.any():
            k=np.flatnonzero(expand);expanded=3*centroid[k]-2*x[a[k],worst[k]];fe=evaluate(expanded,a[k]);accept=fe<fr[k]
            chosen[k[accept]]=expanded[accept];fc[k[accept]]=fe[accept]
        contract=~expand&~((fb<=fr)&(fr<fm))
        outside=contract&(fm<=fr)&(fr<fw)
        inside=contract&~outside
        for selected,is_outside in [(outside,True),(inside,False)]:
            if selected.any():
                k=np.flatnonzero(selected)
                contracted=1.5*centroid[k]-.5*x[a[k],worst[k]] if is_outside else .5*centroid[k]+.5*x[a[k],worst[k]]
                fcontract=evaluate(contracted,a[k]);accept=(fcontract<=fr[k]) if is_outside else (fcontract<fw[k])
                chosen[k[accept]]=contracted[accept];fc[k[accept]]=fcontract[accept];shrink[k[~accept]]=True
        regular=~shrink
        ar=a[regular];x[ar,worst[regular]]=chosen[regular];values[ar,worst[regular]]=fc[regular]
        if len(ar):order[ar]=_sort_last(values[ar],order[ar])
        ash=a[shrink]
        if len(ash):
            for rank in [1,2]:
                indices=order[ash,rank];bi=order[ash,0]
                x[ash,indices]=x[ash,bi]+.5*(x[ash,indices]-x[ash,bi]);values[ash,indices]=evaluate(x[ash,indices],ash)
            order[ash]=_sort_three(values[ash],order[ash])
        # Preserve source storage-slot width, rather than replacing by full span.
        width=np.max(np.abs(x[a,0]-x[a,2]),axis=1);scale=np.max(np.abs(x[a,2]),axis=1)
        fbest=values[a,order[a,0]];fworst=values[a,order[a,2]]
        converged=(np.abs(fbest-fworst)<=np.maximum(absolute,relative*np.abs(fworst)))&(width<=np.maximum(absolute,relative*scale))
        iterations[a]+=1;active[a[converged]]=False
        if return_history:
            parameter_history.append(x[rows,order[:,0]].copy());objective_history.append(values[rows,order[:,0]].copy())
    fitted=transform_parameters(x[rows,order[:,0]])
    diagnostics=dict(iterations=iterations,converged=~active,objective=values[rows,order[:,0]])
    if return_history:diagnostics.update(parameter_history=np.asarray(parameter_history),objective_history=np.asarray(objective_history))
    return fitted,diagnostics

def reconstruct(data,batch_size=1024):
    """Fit all fixed support voxels independently; preserve all four quantities."""
    mask=data['observation_mask'][0];y=data['diffusion_signal'][0][mask];b=data['bvalues'][0]
    parameters=[];objective=[];iterations=[];converged=[]
    for first in range(0,len(y),batch_size):
        fit,diag=fit_signals(y[first:first+batch_size],b)
        parameters.append(fit);objective.append(diag['objective']);iterations.append(diag['iterations']);converged.append(diag['converged'])
    fit=np.concatenate(parameters);v=fit[:,0];d=fit[:,1]
    result={}
    for name,values in [('intra_fraction',v),('intrinsic_diffusivity',d),('extra_transverse_diffusivity',(1-v)*d),('extra_mean_diffusivity',(1-2/3*v)*d)]:
        field=np.zeros(mask.shape,dtype=np.float64);field[mask]=values;result[name]=field[None]
    return result,dict(objective=np.concatenate(objective),iterations=np.concatenate(iterations),converged=np.concatenate(converged))
