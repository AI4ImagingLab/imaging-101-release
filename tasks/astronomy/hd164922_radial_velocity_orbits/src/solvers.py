"""Explicit RadVel configuration priors/likelihood and native Powell MAP stage (MIT)."""
import warnings
import numpy as np
from scipy.optimize import minimize
from .physics_model import forward,unpack_orbits
INITIAL=np.array([1206.3,2456779.,np.sqrt(.01)*np.cos(np.pi/2),.1,np.log(10.),75.771,2456277.6,np.sqrt(.01)*np.cos(np.pi/2),.1,0.,0.,1.,0.,2.6,2.6,2.6])

def log_prior(vector):
    v=np.asarray(vector)
    if v.shape!=(16,) or not np.isfinite(v).all():return -np.inf
    out=0.
    for j,lo,hi,mu,sd in [(0,500.,5000.,2456779.,300.),(5,1.,500.,2456277.6,25.)]:
        if not lo<=v[j]<=hi:return -np.inf
        out+=-np.log(np.log(hi/lo))-np.log(v[j])
        out+=-.5*((v[j+1]-mu)/sd)**2-.5*np.log(sd*sd*2*np.pi)
        for k,a,b in [(j+2,-1.,1.),(j+3,-1.,1.),(j+4,np.log(.001),np.log(30.))]:
            if not a<v[k]<b:return -np.inf
            out-=np.log(b-a)
        if not 0<=v[j+2]**2+v[j+3]**2<=.99:return -np.inf
    out-=2*np.log(.99)
    for k in range(10,16):
        lo,hi=(-10.,10.) if k<13 else (0.,10.)
        if not lo<v[k]<hi:return -np.inf
        out-=np.log(hi-lo)
    return float(out)

def negative_logposterior(vector,data):
    lp=log_prior(vector)
    if not np.isfinite(lp):return np.inf
    residual=data['velocity']-forward(data['time'],data['instrument'],vector)
    # Preserve native instrument-wise summation order k,j,a.
    ll=0.
    for k in range(3):
        mask=data['instrument']==k
        var=data['uncertainty'][mask]**2+vector[13+k]**2
        ll+=-.5*np.sum(residual[mask]**2/var)-np.sum(np.log(np.sqrt(2*np.pi*var)))
    return float(-lp-ll)

def reconstruct_native(data,initial=None,fixed=None):
    """Return native MAP vector and optimizer diagnostics; fixed indices are notebook ablations."""
    x=INITIAL.copy() if initial is None else np.asarray(initial,dtype=float).copy()
    fixed={} if fixed is None else dict(fixed)
    for k,v in fixed.items():x[k]=v
    active=np.array([i for i in range(16) if i not in fixed],dtype=int)
    latest=x.copy()
    def objective(y):
        nonlocal latest
        latest=x.copy();latest[active]=y
        return negative_logposterior(latest,data)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=RuntimeWarning,message='invalid value encountered')
        r=minimize(objective,x[active],method='Powell',options=dict(maxiter=200,maxfev=100000,xtol=1e-8))
    # The original maxlike_fitting leaves the mutable parameter vector at the last objective evaluation.
    return latest.copy(),{'success':bool(r.success),'message':str(r.message),'iterations':int(r.nit),'evaluations':int(r.nfev),'negative_logposterior':negative_logposterior(latest,data)}

def submission(vector):
    return {'reconstruction':unpack_orbits(vector)[None,...],'instrument_offset':np.asarray(vector)[None,10:13].copy(),'instrument_jitter':np.asarray(vector)[None,13:16].copy()}

# Scaling changes only numerical coordinates, not the physical posterior density.
COORDINATE_SCALE=np.array([10.,100.,.1,.1,.5,1.,10.,.1,.1,.5,1.,1.,1.,1.,1.,1.])

def precision_fit(data,start,fixed=None):
    """Converge the same posterior in affine coordinates; return explicit optimum."""
    fixed={} if fixed is None else dict(fixed)
    start=np.asarray(start,dtype=float).copy()
    for k,v in fixed.items():start[k]=v
    active=np.array([i for i in range(16) if i not in fixed],dtype=int)
    def to_physical(z):
        v=start.copy();v[active]=INITIAL[active]+COORDINATE_SCALE[active]*z
        return v
    def objective(z):return negative_logposterior(to_physical(z),data)
    z0=(start[active]-INITIAL[active])/COORDINATE_SCALE[active]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        r=minimize(objective,z0,method='Powell',options={'maxiter':2000,'maxfev':500000,'xtol':1e-11,'ftol':1e-13})
    vector=to_physical(r.x)
    return vector,{'success':bool(r.success),'message':str(r.message),'iterations':int(r.nit),'evaluations':int(r.nfev),'negative_logposterior':negative_logposterior(vector,data)}

def reconstruct(data,initial=None,fixed=None,multi_start=None):
    """Precision MAP; four native/prior-based starts by default, single start for explicit local experiments."""
    if multi_start is None:multi_start=initial is None and not fixed
    if initial is not None:starts=[np.asarray(initial,dtype=float).copy()]
    else:starts=[INITIAL.copy()]
    if multi_start:
        native,_=reconstruct_native(data)
        starts.append(native)
        for sign in [-1,1]:
            v=INITIAL.copy();v[0]*=1+sign*.005;v[5]*=1-sign*.005
            v[2]=sign*.2;v[3]=.1;v[7]=-sign*.15;v[8]=.1;starts.append(v)
    fits=[precision_fit(data,start,fixed) for start in starts]
    best=int(np.argmin([r['negative_logposterior'] for _,r in fits]))
    v,r=fits[best]
    return v,{**r,'selected_start':best,'runs':[r for _,r in fits]}
