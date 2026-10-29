"""Sequential multiplicative CARD updates, explicit source arithmetic.
Adapted from YMa-lab/CARDspa src/CARDref.cpp (GPL-3.0).
"""
import numpy as np
from .physics_model import normalize_weights,car_log_normalizer
PHIS=(.01,.1,.3,.5,.7,.9,.99)


def fit_card(X,basis,kernel,phi=.99,max_iter=1000,epsilon=1e-4,initial=None):
    """Fit one fixed CAR strength; return every spatial unknown and diagnostics."""
    X=np.asarray(X,dtype=np.float64);U=np.asarray(basis,dtype=np.float64)
    W=np.asarray(kernel,dtype=np.float64);n=X.shape[1];k=U.shape[1]
    V=np.full((n,k),1/k) if initial is None else np.array(initial,dtype=float,copy=True)
    b=np.zeros(k);variance=.1;lambdas=np.full(k,10.)
    UtX=U.T@X;XtU=UtX.T;d=W.sum(axis=1)
    L=np.diag(d)-phi*W;accu_L=L.sum();UtU=U.T@U;diag=np.diag(UtU)
    trac_xxt=np.sum(X*X)
    def state_objective():
        UtXV=UtX@V;VtV=V.T@V
        norm=trac_xxt-2*np.trace(UtXV)+np.trace(UtU@VtV)
        Z=V-b[None,:];temp=Z.T@L@Z
        obj=(-X.size*.5*np.log(variance)-.5*norm/variance
             -n*.5*np.log(lambdas).sum()-.5*np.sum(np.diag(temp)/lambdas)
             -2*np.log(lambdas).sum()-np.sum((n/2)/lambdas))
        return float(norm),temp,float(obj)
    norm,temp,obj_old=state_objective();V_old=V.copy();history=[]
    for iteration in range(1,max_iter+1):
        b=(V.T@L).sum(axis=1)/accu_L
        # Source uses the previous iteration's centered quadratic here.
        lambdas=(np.diag(temp)/2+n/2)/(n/2+2)
        part1=variance*(d[:,None]*V+phi*d[:,None]*b[None,:])
        part2=variance*(phi*W@V+d[:,None]*b[None,:])
        for j in range(k):
            denominator=lambdas[j]*(V[:,j]*diag[j]+(V@UtU[:,j]-V[:,j]*diag[j]))+part1[:,j]
            V[:,j]*=(lambdas[j]*XtU[:,j]+part2[:,j])/denominator
        norm=trac_xxt-2*np.trace(UtX@V)+np.trace(UtU@(V.T@V))
        variance=norm/X.size
        _,temp,obj=state_objective()
        change=np.sqrt(np.mean((V-V_old)**2))
        rel=abs(obj-obj_old)*2/abs(obj+obj_old)
        history.append((iteration,obj,change,variance))
        stop=np.isnan(obj) or change<epsilon or (obj>obj_old and rel<epsilon)
        if stop:
            if iteration>5:break
        else:
            obj_old=obj;V_old=V.copy()
    if not np.isfinite(obj) or not np.all(np.isfinite(V)):raise ArithmeticError("Nonfinite native CARD trajectory")
    return {"V":V,"b":b,"lambda":lambdas,"sigma_e2":variance,"Obj":obj,
            "history":np.asarray(history),"phi":float(phi)}


def reconstruct(prepared,phis=PHIS):
    """Evaluate all source phi candidates; last maximizing SI Eq9-normalized score wins (qualified correction)."""
    results=[fit_card(prepared["X"],prepared["basis"],prepared["kernel"],phi=p) for p in phis]
    objectives=np.array([r["Obj"]+car_log_normalizer(prepared["kernel"],r["phi"],prepared["basis"].shape[1]) for r in results]);best=np.flatnonzero(objectives==objectives.max())[-1]
    for r,score in zip(results,objectives):r["normalized_score"]=float(score)
    weights=results[best]["V"]
    return {"mixing_weights":weights[None],"cell_type_proportions":normalize_weights(weights)[None]},results
