"""Explicit local DLSuR inversion; BSD3 adaptation, see LICENSE.
All task-specific regression, CV window selection and inverse operators are local.
"""
import numpy as np
from scipy.optimize import curve_fit
from .physics_model import stretched_exponential,msd_from_field,modulus
from .preprocessing import select_lags,KB,RADIUS_NM,TEMPERATURE_K
def cross_validation(t,g,p0):
    """Leave-one-out mean squared error; original failure penalties retained."""
    errors=[]
    for i in range(len(t)):
        try:
            p=curve_fit(stretched_exponential,np.delete(t,i),np.delete(g,i),p0=p0,maxfev=10000)[0]
            errors.append((g[i]-stretched_exponential(t[i],*p))**2)
        except RuntimeError: errors.append(1e3)
    score=float(np.mean(errors))
    try:curve_fit(stretched_exponential,t,g,p0=p0,maxfev=10000)
    except RuntimeError:score=1e6
    return score
def estimate_intercept(t,g):
    """Return beta, selected inclusive fit interval, 3 parameters, 9 CV scores."""
    p0=[g[1],.01,1.]
    windows=[[2.,float(end)] for end in np.arange(40.,130.,10.)]
    scores=[];parameters=[]
    for lo,hi in windows:
        i,j=np.argmin(abs(t-lo)),np.argmin(abs(t-hi))
        tt,gg=t[i:j+1],g[i:j+1]
        scores.append(cross_validation(tt,gg,p0))
        try:p=curve_fit(stretched_exponential,tt,gg,p0=p0,maxfev=100000)[0]
        except RuntimeError:p=None
        parameters.append(p)
    best=int(np.argmin(scores));p=parameters[best]
    if p is None:raise ValueError('No valid intercept fit')
    return float(p[0]),np.array(windows[best]),p,np.array(scores)
def field_correlation(t,g):
    """Siegert inverse with the original early-window fitted replacement."""
    beta,window,p,scores=estimate_intercept(t,g)
    corrected=g.copy();i,j=[np.argmin(abs(t-v)) for v in window]
    # Upstream executes this assignment unconditionally, upper endpoint exclusive.
    corrected[i:j]=stretched_exponential(t,*p)[i:j]
    return np.sqrt(corrected/beta),beta,window,p,scores
def local_power_law(t,msd,bandwidth=.1):
    """Gaussian weighted log-linear regression with tau=bw*log-lag range."""
    msd=msd.copy()
    neg=msd<0
    if np.any(neg):msd[neg]=np.interp(t[neg],t[msd>0],msd[msd>0])
    if np.any(msd<=0):raise ValueError('Nonpositive MSD cannot be log-regressed')
    x=np.log(t);y=np.log(msd);X=np.column_stack([np.ones(len(x)),x])
    tau=bandwidth*abs(x[0]-x[-1])
    coefficients=np.empty((len(x),2))
    for i in range(len(x)):
        w=np.exp(-(x-x[i])**2/tau)
        # Algebraically identical to upstream X.T @ diag(w) @ X.
        coefficients[i]=np.linalg.inv(X.T@(w[:,None]*X))@(X.T@(w*y))
    return np.exp(np.sum(X*coefficients,axis=1)),coefficients[:,1]
def reconstruct_one(t,g,bandwidth=.1,stride=1):
    """Complete single repeat; optional stride is a labeled robustness ablation."""
    t,g=select_lags(t,g)
    t,g=t[::stride],g[::stride]
    g1,beta,window,p,scores=field_correlation(t,g)
    raw=msd_from_field(g1)
    smooth,alpha=local_power_law(t,raw,bandwidth)
    omega,storage,loss=modulus(t,smooth,alpha)
    return dict(time=t,correlation=g,g1=g1,raw_msd=raw,msd=smooth,exponent=alpha,omega=omega,
                storage=storage,loss=loss,intercept=beta,window=window,parameters=p,cv_scores=scores)
def reconstruct(t,g,bandwidth=.1):
    """Return submission dict and three diagnostic dicts from (3,192) arrays."""
    rows=[reconstruct_one(tt,gg,bandwidth) for tt,gg in zip(t,g)]
    result={'reconstruction':np.concatenate([np.column_stack([r['storage'],r['loss']]) for r in rows])[None],
            'msd':np.concatenate([r['msd'] for r in rows])[None],
            'exponent':np.concatenate([r['exponent'] for r in rows])[None],
            'intercept':np.array([r['intercept'] for r in rows])[None]}
    return result,rows
def laplace_alternative(t,msd,bandwidth=.01):
    """Fresh direct Laplace/analytic-continuation comparison, biased at ends.
    Upstream uses removed scipy.integrate.trapz; numpy.trapezoid is equivalent.
    """
    s=1/t
    transformed=np.array([np.trapezoid(msd*np.exp(-v*t),t) for v in s])
    values=1/(np.pi*RADIUS_NM*s*transformed)
    smooth,alpha=local_power_law(s,values,bandwidth)
    amplitude=KB*TEMPERATURE_K*1e27*smooth
    return amplitude*np.cos(np.pi*alpha/2),amplitude*np.sin(np.pi*alpha/2)
