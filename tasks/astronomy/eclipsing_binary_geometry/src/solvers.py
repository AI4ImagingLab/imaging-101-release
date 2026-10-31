"""Explicit profiled likelihood and affine-invariant ensemble updates."""
import numpy as np
from scipy.optimize import least_squares
from .physics_model import photometry,orbit,P,T0


def valid(p):
    if np.shape(p)!=(7,) or not np.isfinite(p).all():return False
    r1,r2,inc,fs,fc,sj,k=p;e=fs*fs+fc*fc
    return r1>0 and r2>0 and 0<inc<=90 and e<1 and r1+r2<1-e and sj>=0 and k>0

def models(p,groups):
    result=[]
    for gi,(t,y,s,wt,flag) in enumerate(groups):
        if gi==5:pred=orbit(t,p,True)[2]
        else:
            direct=flag==1;flux=photometry(t[direct],p,gi)
            if gi in (1,4):pred=-2.5*np.log10(np.interp(t,t[direct],flux))
            elif gi==0:pred=np.zeros(len(t));pred[direct]=-2.5*np.log10(flux)
            else:pred=-2.5*np.log10(flux)
        result.append(pred)
    return result

def residuals(p,groups,details=False):
    preds=models(p,groups);res=[];offsets=[];chi=[]
    for pred,(t,y,s,wt,flag) in zip(preds,groups):
        delta=y-pred;zp=np.sum(delta*wt)/np.sum(wt);rr=(delta-zp)*np.sqrt(wt)
        res.append(rr);offsets.append(zp);chi.append(np.sum((delta-zp)**2*wt))
    if details:return np.array(chi),np.array(offsets),np.concatenate(preds)
    return np.concatenate(res)

def log_probability(p,groups):
    if not valid(p):return -np.inf
    return -.5*residuals(p,groups,True)[0].sum()

def initialize(groups):
    """Raw RV multistarts and photometric depth, followed by joint fitting.

    Initial trial grid is generic; no original fitted parameter or uncertainty
    enters this routine. The fitted Hessian only initializes ensemble spread.
    """
    rvgroup=groups[-1];t,y,s,wt,flag=rvgroup
    k0=max(float(np.ptp(y))/2,1.);rvfits=[]
    def rv_res(x):
        e,omega,k=x;p=np.array([.05,.005,90,np.sqrt(e)*np.sin(omega),np.sqrt(e)*np.cos(omega),.1,k])
        pred=orbit(t,p,True)[2];d=y-pred;z=np.sum(d*wt)/np.sum(wt)
        return (d-z)*np.sqrt(wt)
    for e in (.05,.4,.7):
        for omega in (-2.5,-.8,.8,2.5):
            r=least_squares(rv_res,[e,omega,k0],bounds=([0,-2*np.pi,.1],[.9,2*np.pi,200]),x_scale=[.3,2,20],max_nfev=160)
            rvfits.append(r)
    rvfit=min(rvfits,key=lambda f:np.sum(f.fun*f.fun));e,om,k=rvfit.x
    def depth(gi):
        yy=groups[gi][1];return max(float(np.quantile(yy,.9)-np.quantile(yy,.1)),.0001)
    dp=1-10**(-.4*depth(1));ds=1-10**(-.4*depth(4));ratio=np.sqrt(dp);sj=np.clip(ds/(ratio*ratio*(1-ds)),.01,3.)
    lower=[.003,.0001,70,-.95,-.95,.00001,.1];upper=[.3,.15,90,.95,.95,10,200]
    scale=np.array([.05,.008,1,.1,.1,.3,15]);fits=[]
    def fun(p):
        if not valid(p):return np.full(sum(len(g[0]) for g in groups),1e6)
        return residuals(p,groups)
    for radius in (.035,.07,.14):
        for impact in (.2,.8):
            cosi=impact*radius*(1+e*np.sin(om))/(1-e*e)
            p=[radius,radius*ratio,np.rad2deg(np.arccos(np.clip(cosi,0,1))),np.sqrt(e)*np.sin(om),np.sqrt(e)*np.cos(om),sj,k]
            fit=least_squares(fun,p,bounds=(lower,upper),x_scale=scale,diff_step=1e-5,max_nfev=180,ftol=1e-9,xtol=1e-10,gtol=1e-7)
            fits.append(fit)
    best=min(fits,key=lambda f:np.sum(f.fun*f.fun))
    # A fresh centered Jacobian, rather than finite differences of optimizer
    # termination, provides a stable raw likelihood curvature for walker spread.
    p=best.x;steps=np.array([1e-6,1e-6,1e-4,1e-5,1e-5,1e-4,1e-4]);J=np.column_stack([(residuals(p+np.eye(7)[j]*steps[j],groups)-residuals(p-np.eye(7)[j]*steps[j],groups))/(2*steps[j]) for j in range(7)])
    covariance=np.linalg.pinv(J.T@J,rcond=1e-12)
    return p,covariance,{'rv_best_chi_squared':float(np.sum(rvfit.fun**2)),'joint_start_chi_squared':[float(np.sum(f.fun**2)) for f in fits],'raw_depth_ratio':float(ratio),'raw_surface_brightness_start':float(sj)}

def ensemble(groups,initial,covariance,rng,walkers=50,updates=1000,progress=None,probability=None):
    """Two-subensemble Goodman--Weare stretch proposals, scale a=2.

    Samples z with density proportional to z**(-1/2) on [.5,2]. Each
    half update uses the complementary half held fixed, preserving balance.
    """
    probability=(lambda p:log_probability(p,groups)) if probability is None else probability
    positions=[]
    while len(positions)<walkers:
        p=rng.multivariate_normal(initial,covariance)
        if valid(p):positions.append(p)
    positions=np.asarray(positions);lp=np.array([probability(p) for p in positions]);chain=np.empty((walkers,updates,7));accepted=np.zeros(walkers,dtype=int)
    for it in range(updates):
        order=rng.permutation(walkers);halves=np.array_split(order,2)
        for active,complement in (halves,halves[::-1]):
            partners=rng.choice(complement,len(active));z=(1+rng.random(len(active)))**2/2
            proposal=positions[partners]+z[:,None]*(positions[active]-positions[partners]);plp=np.array([probability(p) for p in proposal])
            take=np.log(rng.random(len(active)))<6*np.log(z)+plp-lp[active]
            chosen=active[take];positions[chosen]=proposal[take];lp[chosen]=plp[take];accepted[chosen]+=1
        chain[:,it]=positions
        if progress and (it+1)%100==0:progress(it+1,float(np.mean(accepted/(it+1))))
    return chain,accepted/updates


def solve(groups,rng=None,progress=None):
    """Full 50-walker, 1000-update fit; summarize last 500 updates."""
    if rng is None:rng=np.random.default_rng()
    initial,covariance,initialization=initialize(groups)
    chain,acceptance=ensemble(groups,initial,covariance,rng,progress=progress)
    point=np.median(chain[:,500:].reshape(-1,7),axis=0)
    chi,offsets,prediction=residuals(point,groups,True)
    return {"parameters":point,"chain":chain,"acceptance":acceptance,"chi_squared":chi,"offsets":offsets,"prediction":prediction,"initial_parameters":initial,"initialization":initialization}
