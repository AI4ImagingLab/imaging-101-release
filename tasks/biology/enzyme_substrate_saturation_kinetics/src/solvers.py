"""Local extraction of renz0.2.1, Juan Carlos Aledo, GPL>=2.
ECB's known intercept bug is preserved only as historical comparison/seed.
No upstream runtime dependency; direct nls Gauss-Newton is explicit.
"""
import numpy as np
from .physics_model import rate,jacobian
METHODS=('direct','lb','weighted_lb','ecb','hw','eh')
def intersections(s,v,corrected=False):
    """All pairwise Km,V intersections, native erroneous V unless corrected."""
    slope=v/s;out=[]
    for i in range(len(s)-1):
        for j in range(i+1,len(s)):
            with np.errstate(divide='ignore',invalid='ignore'):
                k=(v[j]-v[i])/(slope[i]-slope[j])
                vmax=slope[i]*k+(v[i] if corrected else v[j])
            out.append((k,vmax))
    return np.asarray(out)
def ecb(s,v,corrected=False):
    return np.round(np.nanmedian(intersections(s,v,corrected),axis=0),3)
def line_fit(x,y,weights=None):
    a=np.column_stack((np.ones(len(x)),x))
    if weights is not None:
        w=np.sqrt(weights/np.max(weights));a=a*w[:,None];y=y*w
    return np.linalg.lstsq(a,y,rcond=None)[0]
def linear_fit(s,v,method):
    x=np.round(1/s,4);y=np.round(1/v,4)
    if method in ('lb','weighted_lb'):
        intercept,slope=line_fit(x,y,1/y**4 if method=='weighted_lb' else None)
        vmax=np.round(1/intercept,2);km=np.round(vmax*slope,2)
    elif method=='hw':
        x=1/x;intercept,slope=line_fit(x,x*y)
        vmax=np.round(1/slope,2);km=np.round(vmax*intercept,2)
    elif method=='eh':
        intercept,slope=line_fit(v/s,v)
        km=np.round(-slope,2);vmax=np.round(intercept,2)
    else: raise ValueError('Unknown linear method')
    return np.array([km,vmax])
def direct(s,v,initial=None,return_history=False):
    """R nls default Gauss-Newton: tol1e-5, max50, step halving >=1/1024.
    Analytic derivatives replace R numericalDeriv. Relative offset criterion is
    norm(projected residual)/norm(orthogonal residual), R default scaleOffset0.
    Return author's 3-decimal rounded Km,Vmax; history is optional diagnostic.
    """
    p=ecb(s,v) if initial is None else np.asarray(initial,dtype=float).copy()
    history=[];factor=1.
    for iteration in range(51):
        r=v-rate(s,p);j=jacobian(s,p)
        q,rr=np.linalg.qr(j,mode='reduced');qr=q.T@r
        orth=r-q@qr;den=np.linalg.norm(orth)
        criterion=np.linalg.norm(qr)/den if den else (0. if np.linalg.norm(qr)==0 else np.inf)
        history.append(np.r_[p,r@r,criterion])
        if criterion<=1e-5:
            result=np.round(p,3)
            return (result,np.asarray(history)) if return_history else result
        if iteration==50: raise RuntimeError('nls iteration limit')
        delta=np.linalg.solve(rr,qr);factor=min(1.,2*factor)
        while factor>=1/1024:
            candidate=p+factor*delta;new=v-rate(s,candidate)
            if np.isfinite(new).all() and new@new <= r@r:
                p=candidate;break
            factor*=.5
        else: raise RuntimeError('nls step factor below minimum')
    raise RuntimeError('Unreachable')
def compare(s,rates,author_removal=False):
    """All six methods, all groups, shape (6,8,2)."""
    result=np.empty((6,len(rates),2))
    for g,v in enumerate(rates):
        mask=np.ones(len(s),bool)
        if author_removal and g==1: mask[7]=False
        x,y=s[mask],v[mask]
        for m,method in enumerate(METHODS):
            result[m,g]=direct(x,y) if method=='direct' else ecb(x,y) if method=='ecb' else linear_fit(x,y,method)
    return result
def reconstruct(s,rates):
    p=np.asarray([direct(s,v) for v in rates])
    return {'reconstruction':p[:,0][None],'maximum_rate':p[:,1][None],
            'fitted_rate':np.asarray([rate(s,row) for row in p])[None]}
def sensitivity(s,rates):
    """Measured-data leave-one-out fits; failures explicit NaN in diagnostic only."""
    result=np.full((len(rates),len(s),2),np.nan)
    for g,v in enumerate(rates):
        for i in range(len(s)):
            mask=np.arange(len(s))!=i
            try:result[g,i]=direct(s[mask],v[mask])
            except (ValueError,RuntimeError,np.linalg.LinAlgError):pass
    return result
