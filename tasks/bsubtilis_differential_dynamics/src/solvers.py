"""Explicit DDM inverse fit; generic SciPy MINPACK, no lmfit/domain runtime.
The bound transforms and default MINPACK tolerances reproduce lmfit1.3.4;
Copyright fastddm mathematical extraction: University of Vienna, GPL3+.
"""
import numpy as np
from scipy.optimize import leastsq
from .physics_model import structure_model
from .preprocessing import image_structure,radial_average,resample_loglog,Q_MIN,Q_MAX
EPS=np.finfo(float).eps

def fit_curve(tau,data,initial,weights=None,vary_beta=True):
    """Bounded compressed-exponential fit via explicit lower-bound transform.
    A/Gamma>=epsilon, beta>=1, B unbounded. Fixed beta retains initial value.
    Returns physical parameters, covariance-internal, MINPACK status.
    """
    initial=np.asarray(initial,dtype=float);n=4 if vary_beta else 3
    lower=np.array([EPS,-np.inf,EPS,1.])[:n]
    internal=initial[:n].copy()
    for i in (0,2,3):
        if i<n:internal[i]=np.sqrt((max(initial[i],lower[i])-lower[i]+1.)**2-1.)
    def decode(values):
        p=initial.copy()
        for i,v in enumerate(values):p[i]=lower[i]-1.+np.sqrt(v*v+1.) if np.isfinite(lower[i]) else float(v)
        return p
    def residual(values):
        diff=data-structure_model(tau,decode(values))
        if weights is not None:diff=diff*weights
        return diff.ravel()
    with np.errstate(all='ignore'):
        values,cov,info,message,status=leastsq(residual,internal,full_output=True,ftol=1.5e-8,xtol=1.5e-8,gtol=0.,maxfev=4000*(n+1),epsfcn=1e-10,factor=100)
    if status not in (1,2,3,4):raise RuntimeError('MINPACK failure: '+message)
    return decode(values),cov,int(status)

def fit_multifrequency(data,errors,tau,averages,q,movie_index):
    """Original reference10 then descending and ascending warm-start fits.
    First reference fit is unweighted exactly as original fit_multik; later
    fits use inverse spatial SD. Near-coverslip beta varies, bulk beta1.35.
    """
    noise=2*np.nanmean(averages[-7:,-1]);amp=2*averages[:,-1]-noise
    ref=10;y=1-(data[ref]-noise)/amp[ref];tc=tau[np.argmin(np.abs(y-np.exp(-1)))]
    init=np.array([amp[ref],noise,1/tc,1.35]);vary=movie_index==0
    p,_,_=fit_curve(tau,data[ref],init,vary_beta=vary)
    result=np.full((len(q),4),np.nan);result[ref]=p
    for indices in (reversed(range(ref)),range(ref,len(q))):
        current=result[ref].copy()
        for index in indices:
            if np.isnan(data[index,0]):continue
            current,_,_=fit_curve(tau,data[index],current,weights=1/errors[index],vary_beta=vary)
            result[index]=current
    return result,np.full(len(q),noise),amp

def fit_speed(q,Gamma):
    """Fit log(Gamma)=log(v*q), unweighted original lower-bound LM, um/s."""
    def residual(internal):
        v=-1+np.sqrt(internal[0]**2+1)
        return np.log(Gamma)-np.log(v*q)
    initial=np.array([np.sqrt((1.+1)**2-1)])
    value,_,_,message,status=leastsq(residual,initial,full_output=True,ftol=1.5e-8,xtol=1.5e-8,gtol=0.,epsfcn=1e-10,maxfev=8000,factor=100)
    if status not in (1,2,3,4):raise RuntimeError(message)
    return float(-1+np.sqrt(value[0]**2+1))

def analyze_movie(movie,movie_index,num_lags=200,frame_rate=100.,pixel_size=75.59/128):
    """Full native movie -> all fresh numerical stages and selected parameters."""
    packed=image_structure(movie);avg,err,q=radial_average(packed,pixel_size)
    data,error,tau=resample_loglog(avg,err,frame_rate,num_lags)
    pars,noise,amp=fit_multifrequency(data,error,tau,avg,q,movie_index)
    selected=(q>=Q_MIN)&(q<=Q_MAX)
    speed=fit_speed(q[selected],pars[selected,2])
    return dict(packed=packed,averages=avg,errors=err,q=q,data=data,error=error,tau=tau,parameters=pars,noise=noise,amplitude_estimate=amp,selected=selected,speed=speed)

def reconstruct(frames):
    """Full batch-first two-condition recording -> parameter and speed arrays."""
    parameters=[];speeds=[]
    for i,movie in enumerate(frames[0]):
        result=analyze_movie(movie,i);parameters.append(result['parameters'][result['selected']]);speeds.append(result['speed'])
    return np.stack(parameters)[None],np.asarray(speeds)[None]
