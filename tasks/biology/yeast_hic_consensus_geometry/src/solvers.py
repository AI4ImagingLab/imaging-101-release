"""Explicit MDS initialization followed by published PM1 Poisson estimation.

Task-specific acquisition mathematics and derivatives are local. SciPy supplies
only the general L-BFGS-B optimizer, as in the validated author implementation.
Adapted from Pastis under BSD-3-Clause; see NOTICE.
"""
import time
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from .preprocessing import build_pairs,ice_normalize
from .physics_model import wish_distances,mds_objective,mds_gradient,poisson_objective,poisson_gradient,pair_distances,alpha_profile

MAX_ITER=5000
SEED=0

def fit_mds(counts,bias,seed=SEED,max_iter=MAX_ITER,alpha=-3.,beta=1.,initial=None,factr=1e12,wish_override=None):
    """Fit normalized MDS stress with the author's initialization and stopping."""
    row,col,c=build_pairs(counts)
    wish=wish_distances(row,col,c,bias,alpha,beta) if wish_override is None else np.asarray(wish_override)
    n=np.asarray(counts).shape[0]
    x0=1-2*np.random.RandomState(seed).rand(n*3) if initial is None else np.asarray(initial).ravel().copy()
    history=[]
    def objective(z):return mds_objective(z,row,col,wish)
    def gradient(z):return mds_gradient(z,row,col,wish).ravel()
    def callback(z):history.append(objective(z))
    x,value,info=fmin_l_bfgs_b(objective,x0,gradient,factr=factr,maxiter=max_iter,callback=callback)
    return x.reshape(n,3),dict(objective=float(value),history=np.asarray(history),iterations=int(info['nit']),function_calls=int(info['funcalls']),warnflag=int(info['warnflag']),stop=str(info['task']))

def fit_pm1(counts,bias,seed=SEED,max_iter=MAX_ITER,alpha=-3.,beta=1.,initial=None):
    """MDS2-initialized positive-pair Poisson objective, fixed scale."""
    start=time.perf_counter()
    if initial is None:x0,mds=fit_mds(counts,bias,seed,max_iter,alpha,beta)
    else:x0=np.asarray(initial,dtype=np.float64).reshape(-1,3).copy();mds=None
    row,col,c=build_pairs(counts);history=[]
    def objective(z):return poisson_objective(z,row,col,c,bias,alpha,beta)
    def gradient(z):return poisson_gradient(z,row,col,c,bias,alpha,beta).ravel()
    def callback(z):history.append(objective(z))
    x,value,info=fmin_l_bfgs_b(objective,x0.ravel(),gradient,maxiter=max_iter,callback=callback)
    return x.reshape(-1,3),dict(objective=float(value),history=np.asarray(history),iterations=int(info['nit']),function_calls=int(info['funcalls']),warnflag=int(info['warnflag']),stop=str(info['task']),seconds=time.perf_counter()-start,mds=mds,initial_coordinates=x0)

def reconstruct(data,seed=SEED,max_iter=MAX_ITER):
    """Fit all350 supplied loci; center supported coordinates, pad unsupported0.

    Return batch-first float64 coordinates and identified method diagnostics.
    Original unobserved-locus values are arbitrary initialization values and are
    replaced only outside the input-fixed support, never inside an inferred mask.
    """
    counts=data['contact_counts'][0]
    normalized,bias,ice_history=ice_normalize(counts)
    x,diagnostics=fit_pm1(counts,bias,seed,max_iter)
    support=data['bead_support'][0]
    x[support]-=x[support].mean(axis=0)
    x[~support]=0
    diagnostics.update(bias=bias,normalized_counts=normalized,ice_history=ice_history)
    return x[None],diagnostics

def monotone_fit(abscissa,ordinate):
    """Increasing least squares via tie aggregation and pool-adjacent-violators.

    Equal abscissae share an ordinate. The lexicographic ordering matches the
    upstream general isotonic primitive; every merge is a weighted mean.
    """
    x=np.asarray(abscissa,dtype=np.float64);y=np.asarray(ordinate,dtype=np.float64)
    if x.shape!=y.shape or x.ndim!=1 or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Expected finite matching nonempty1D isotonic arrays')
    order=np.lexsort((y,x));xs=x[order];ys=y[order]
    values=[];weights=[];groups=np.zeros(len(x),dtype=np.int64)
    total=0.;weight=0;group=-1;previous=None
    for k,(xx,yy) in enumerate(zip(xs,ys)):
        if previous is None or xx!=previous:
            if weight:values.append(total/weight);weights.append(weight)
            group+=1;total=0.;weight=0;previous=xx
        total+=yy;weight+=1;groups[k]=group
    values.append(total/weight);weights.append(weight)
    blocks=[]
    for k,(value,weight) in enumerate(zip(values,weights)):
        blocks.append([k,k,value,float(weight)])
        while len(blocks)>=2 and blocks[-2][2]>=blocks[-1][2]:
            right=blocks.pop();left=blocks.pop();w=left[3]+right[3]
            blocks.append([left[0],right[1],(left[2]*left[3]+right[2]*right[3])/w,w])
    fitted=np.empty(len(values))
    for first,last,value,weight in blocks:fitted[first:last+1]=value
    result=np.empty(len(x));result[order]=fitted[groups]
    return result

def fit_nmds(counts,bias,seed=SEED,max_iter=MAX_ITER,outer_iter=5):
    """Published alternating nonmetric MDS; repaired ndarray/string API guard.

    The unmodified author NMDS wrapper fails at its second iteration under
    current NumPy. Its mathematical steps are retained locally and checked
    against the explicitly identified one-line API-guard-repaired oracle.
    """
    x,info=fit_mds(counts,bias,seed,max_iter);r,c,n=build_pairs(counts);hist=[info['objective']]
    for _ in range(1,outer_iter):
        wish=monotone_fit(1./n,pair_distances(x,r,c))
        x,info=fit_mds(counts,bias,seed,max_iter,initial=x,wish_override=wish)
        hist.append(info['objective'])
    return x,dict(outer_stress=np.asarray(hist),source_API_guard_repaired=True)

def fit_pm2(counts,bias,seed=SEED,max_iter=MAX_ITER,outer_iter=5):
    """Alternate analytical beta/profiled-alpha estimation and coordinate PM fit."""
    x,mds=fit_mds(counts,bias,seed,max_iter);row,col,c=build_pairs(counts);alpha=-3.;beta=1.;history=[]
    for _ in range(outer_iter):
        def objective(a):return alpha_profile(x,row,col,c,bias,a[0])[0]
        def gradient(a):return np.array([alpha_profile(x,row,col,c,bias,a[0])[1]])
        a,value,info=fmin_l_bfgs_b(objective,np.array([alpha]),gradient,bounds=np.array([[-100.,1e-2]]),maxiter=1000)
        alpha=float(a[0]);beta=alpha_profile(x,row,col,c,bias,alpha)[2]
        x,diag=fit_pm1(counts,bias,seed,max_iter,alpha,beta,initial=x)
        history.append([alpha,beta,diag['objective']])
    return x,dict(alpha=alpha,beta=beta,outer_history=np.asarray(history),mds=mds)
