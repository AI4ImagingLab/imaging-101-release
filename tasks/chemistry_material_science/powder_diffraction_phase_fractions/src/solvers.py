"""Published BFGS/Rwp with phase removal; optional active-set NNLS comparison.

BFGS conventions adapted from R vmmin, GPL-2-or-later, R Core Team;
full-pattern workflow from Butler/Hillier powdR, GPL-2.
"""
import numpy as np
from .preprocessing import align_pattern,natural_spline
from .physics_model import forward,rwp,finite_gradient,mass_fractions
def bfgs(fun,start,maxiter=100,reltol=np.sqrt(np.finfo(float).eps)):
    """R variable-metric update, central .001 gradients and .2 Armijo backtracking."""
    x=np.array(start,dtype=float); n=len(x); value=fun(x); gradient=finite_gradient(fun,x)
    gradcount=1; iteration=1; last=gradcount; history=[x.copy()]; B=np.eye(n)
    for guard in range(10000):
        if last==gradcount: B=np.eye(n)
        old=x.copy(); oldg=gradient.copy(); direction=-B@gradient; projection=float(direction@gradient)
        if projection<0:
            step=1.; count=0
            while True:
                x=old+step*direction; count=int(np.sum(10+old==10+x))
                if count<n:
                    trial=fun(x); accepted=np.isfinite(trial) and trial<=value+projection*step*.0001
                    if not accepted: step*=.2
                else: accepted=False; trial=value
                if count==n or accepted: break
            enough=abs(trial-value)>reltol*(abs(value)+reltol)
            if not enough: count=n; value=trial
            if count<n:
                value=trial; gradient=finite_gradient(fun,x); gradcount+=1; iteration+=1
                displacement=step*direction; change=gradient-oldg; d1=float(displacement@change)
                if d1>0:
                    bc=B@change; d2=1+float(change@bc)/d1
                    B+=(d2*np.outer(displacement,displacement)-np.outer(bc,displacement)-np.outer(displacement,bc))/d1
                else: last=gradcount
            elif last<gradcount: count=0; last=gradcount
        else:
            count=n if last==gradcount else 0
            last=gradcount
        history.append(x.copy())
        if iteration>=maxiter: break
        if gradcount-last>2*n: last=gradcount
        if count==n and last==gradcount: break
    else: raise RuntimeError('BFGS guard exceeded')
    return x,{'history':np.asarray(history),'iterations':iteration,'converged':iteration<maxiter,'objective':value}
def nnls_active_set(matrix,counts):
    """Lawson-Hanson passive-set least squares with explicit nonnegative boundary steps."""
    n=matrix.shape[1]; x=np.zeros(n); passive=np.zeros(n,dtype=bool)
    tolerance=10*np.finfo(float).eps*np.linalg.norm(matrix,1)*np.linalg.norm(counts)
    for _ in range(10*n*n):
        dual=matrix.T@(counts-matrix@x)
        if np.all(passive) or np.max(dual[~passive])<=tolerance: return x
        enter=np.argmax(np.where(passive,-np.inf,dual)); passive[enter]=True
        while True:
            z=np.zeros(n); z[passive]=np.linalg.lstsq(matrix[:,passive],counts,rcond=None)[0]
            bad=passive&(z<=0)
            if not np.any(bad): x=z; break
            alpha=np.min(x[bad]/(x[bad]-z[bad])); x+=alpha*(z-x)
            remove=passive&(x<=np.finfo(float).eps); passive[remove]=False; x[remove]=0
    raise RuntimeError('NNLS active-set iteration limit')
def fit_coefficients(matrix,counts,method='BFGS'):
    if method=='NNLS': return nnls_active_set(matrix,counts),[]
    if method!='BFGS': raise ValueError('Unknown method')
    active=np.arange(matrix.shape[1]); start=np.zeros(len(active)); records=[]
    for _ in range(matrix.shape[1]):
        a=matrix[:,active]; x,record=bfgs(lambda v:rwp(v,a,counts),start); records.append(record)
        if np.min(x)>=0:
            out=np.zeros(matrix.shape[1]); out[active]=x; return out,records
        keep=x>0; active=active[keep]; start=x[keep]
    raise RuntimeError('No positive phase fit')
def reconstruct(data,method='BFGS',max_shift=.3):
    angles=data['angles']; standards=data['standards']; fractions=[]; shifts=[]; fitted=[]; diagnostics=[]
    for counts in data['counts']:
        shift,grid,y=align_pattern(angles,counts,standards[:,0],max_shift)
        matrix=natural_spline(angles,standards,grid)
        coefficients,history=fit_coefficients(matrix,y,method)
        fit=forward(matrix,coefficients)
        full=natural_spline(grid-shift,fit,angles)
        fractions.append(mass_fractions(coefficients,data['rir'])); shifts.append(shift); fitted.append(full)
        diagnostics.append({'grid':grid,'aligned':y,'matrix':matrix,'coefficients':coefficients,'fit':fit,'history':history})
    return {'reconstruction':np.asarray(fractions)[None],'shifts':np.asarray(shifts)[None],'fitted':np.asarray(fitted)[None]},diagnostics
