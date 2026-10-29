"""Local MINPACK inverse with the original lower-bound parameter transform."""
import numpy as np
from scipy.optimize import leastsq
from .preprocessing import preprocess
from .physics_model import residual, WEIGHT_DISTANCE

def fit_curve(force, height, segment, spring_constant, width=WEIGHT_DISTANCE, corrected=True):
    """Return (modulus [Pa], absolute contact position [m], diagnostics)."""
    x,y,origin=preprocess(force,height,spring_constant)
    mask=segment==0
    def objective(p):
        return residual((np.sqrt(p[0]**2+1)-1,p[1]),x[mask],y[mask],width,corrected)
    initial=np.array([np.sqrt(51.**2-1),0.])
    p,cov,info,msg,status=leastsq(objective,initial,full_output=True,ftol=1.5e-8,xtol=1.5e-8,gtol=0,epsfcn=1e-10,factor=100,maxfev=12000)
    if status not in (1,2,3,4):
        raise RuntimeError(f'MINPACK failed: {msg}')
    E=np.sqrt(p[0]**2+1)-1
    return float(E),float(p[1]+origin),{'nfev':info['nfev'],'weighted_sse':float(np.dot(info['fvec'],info['fvec']))}

def reconstruct(data, width=WEIGHT_DISTANCE, corrected=True):
    """Fit every complete acquisition; return batch-first modulus/contact arrays."""
    n=len(data['spring_constant']); E=np.empty(n); C=np.empty(n)
    for i in range(n):
        a,b=data['offsets'][i:i+2]
        E[i],C[i],_=fit_curve(data['force'][a:b],data['height'][a:b],data['segment'][a:b],data['spring_constant'][i],width,corrected)
    return {'reconstruction':E[None], 'contact_position':C[None]}
