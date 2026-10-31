"""Marginalized Gaussian inverse, original MAP multistart and conditional prediction.
JavelinTNG mathematical extraction; Ying Zu and contributors, GPL2.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import fmin
from .physics_model import covariance,unpack
from .preprocessing import prepare
INITIALS=np.array([[np.log(10),np.log(100),10,6,.8,15,6,1.2],[np.log(10),np.log(100),20,10,.8,20,10,1.2],[np.log(10),np.log(100),5,4,.8,5,4,1.2]])

def likelihood(p,data,components=False):
    sigma,tau,lags,widths,scales=unpack(p)
    if sigma<=0 or tau<=0 or np.any(widths<0) or np.any(scales<=0) or np.max(abs(lags))>data['baseline']:return -1e300
    C=covariance(data['time'],data['channel'],data['time'],data['channel'],p);C[np.diag_indices_from(C)]+=data['error']**2
    try:
        U=cho_factor(C,lower=False);a=cho_solve(U,data['value']);b=cho_solve(U,data['design']);Cp=data['design'].T@b;W=cho_factor(Cp,lower=False);d=cho_solve(W,data['design'].T)
    except np.linalg.LinAlgError:return -1e300
    q=d@a;f=a-b@q
    parts=np.array([0.,-.5*np.dot(data['value'],f),-np.log(np.diag(U[0])).sum(),-np.log(np.diag(W[0])).sum()]);parts[0]=parts[1:].sum()
    penalty=sum(np.log(abs(lag)/(.3*data['baseline'])) for lag in lags[1:] if abs(lag)>.3*data['baseline'])
    if np.any(lags[1:]<0) or np.any(lags[1:]>80) or np.any(widths[1:]<.1) or np.any(widths[1:]>80):penalty+=1e300
    parts[0]-=penalty
    return (parts,q) if components else parts[0]

def predict(p,data,times):
    _,q=likelihood(p,data,components=True);means=data['means']+q;y=data['value']-data['design']@q
    C=covariance(data['time'],data['channel'],data['time'],data['channel'],p);C[np.diag_indices_from(C)]+=data['error']**2;U=cho_factor(C,lower=False);a=cho_solve(U,y)
    predictions=[];errors=[]
    for channel in range(1,4):
        ids=np.full(len(times),channel);cross=covariance(times,ids,data['time'],data['channel'],p);prediction=cross@a+means[channel-1]
        prior=np.diag(covariance(times,ids,times,ids,p));variance=prior-np.sum(cross*cho_solve(U,cross.T).T,axis=1)
        if np.any(variance<0):raise ValueError('Negative conditional variance')
        predictions.append(prediction);errors.append(np.sqrt(variance))
    return np.array(predictions),np.array(errors)

def reconstruct(observations):
    data=prepare(observations);runs=[];rounds=[]
    for initial in INITIALS:
        p=initial.copy()
        for attempt in range(5):
            p,value,niter,ncalls,warn=fmin(lambda x:-likelihood(x,data),p,full_output=True,disp=False)
            rounds.append(dict(start=len(runs),attempt=attempt,objective=float(value),iterations=int(niter),evaluations=int(ncalls),warning=int(warn)))
            if not warn:break
        if warn:raise RuntimeError('MAP native restart limit reached')
        runs.append((p,-value))
    best=int(np.argmax([r[1] for r in runs]));p,logp=runs[best];times=np.linspace(data['time'][0]-.1*data['baseline'],data['time'][-1]+.1*data['baseline'],600)
    curves,sd=predict(p,data,times)
    out=dict(reconstruction=curves[None],transfer=p[2:].reshape(1,2,3),continuum=np.exp(p[:2])[None])
    return out,dict(parameters=p,time=times,conditional_sd=sd,logp=float(logp),all_parameters=np.array([r[0] for r in runs]),all_logp=np.array([r[1] for r in runs]),rounds=rounds,selected_start=best)
