"""Two-step Lorentzian extraction from Pre-CAT (MIT, copyright2025 JWW).

The task-specific model, background exclusion, centering and staged inverse
are local; SciPy supplies general bounded least-squares optimization only.
"""
import numpy as np
from scipy.optimize import curve_fit,least_squares
from .physics_model import lorentzian,background,pool_sum

P0_CORR=[.8,1.8,0,.15,40,-1]
LB_CORR=[.02,.3,-10,0,30,-2.5]
UB_CORR=[1,10,10,.5,60,0]
P0_BG=[.8,.2,0,.15,40,-1]
LB_BG=[.02,.01,-1e-6,0,30,-2.5]
UB_BG=[1,10,1e-6,.5,60,0]
P0_POOLS=[.05,1.5,3.5,.05,.5,2,.05,1,-3.5,.05,1,-1.6]
LB_POOLS=[0,.5,3.2,0,.5,1.6,0,.5,-4,0,.5,-1.8]
UB_POOLS=[.3,5,4,.5,5,2.6,.25,5,-1.5,.25,5,-1.2]
OPTIONS={'xtol':1e-10,'ftol':1e-4,'maxfev':500}

def fit_spectrum(spectrum,offsets,correct_field=True,capture=False):
    """Return(6x3 parameters, ppm shift, stage diagnostics); raise on failure."""
    if offsets[0]>0:offsets=np.flip(offsets);spectrum=np.flip(spectrum)
    first,_=curve_fit(background,offsets,spectrum,p0=P0_CORR,bounds=(LB_CORR,UB_CORR),**OPTIONS)
    correction=first[2] if correct_field else 0.
    x=offsets-correction
    condition=(x<=-4)|(x>=4)|((x>=-1.4)&(x<=1.4))
    bg,_=curve_fit(background,x[condition],spectrum[condition],p0=P0_BG,bounds=(LB_BG,UB_BG),**OPTIONS)
    baseline=lorentzian(x,*bg[:3])+lorentzian(x,*bg[3:])
    difference=1-(spectrum+baseline)
    pools,_=curve_fit(pool_sum,x,difference,p0=P0_POOLS,bounds=(LB_POOLS,UB_POOLS),**OPTIONS)
    parameters=np.concatenate([bg,pools]).reshape(6,3)
    diagnostics={'initial_background':first,'difference':difference,'corrected_offsets':x,'background_mask':condition}
    if capture:
        diagnostics['stages']=[{'params':first,'x':offsets,'y':spectrum,'prediction':background(offsets,*first)},
                               {'params':bg,'x':x[condition],'y':spectrum[condition],'prediction':background(x[condition],*bg)},
                               {'params':pools,'x':x,'y':difference,'prediction':pool_sum(x,*pools)}]
    return parameters,float(correction),diagnostics

def reconstruct(spectra,offsets,mask,correct_field=True):
    """Fit every supported native pixel; outside support all outputs equal zero."""
    parameters=np.zeros((*mask.shape,6,3));field=np.zeros(mask.shape)
    for y,x in np.argwhere(mask):
        parameters[y,x],field[y,x],_=fit_spectrum(spectra[y,x],offsets,correct_field)
    return parameters,field

def joint_refine_spectrum(spectrum,offsets,parameters,field_shift,max_evaluations=2000):
    """Selected-pixel diagnostic: jointly refine six pools at a fixed field shift."""
    def residual(p):return 1-pool_sum(offsets-field_shift,*p)-spectrum
    result=least_squares(residual,parameters.ravel(),bounds=(LB_BG+LB_POOLS,UB_BG+UB_POOLS),
                         xtol=1e-10,ftol=1e-8,max_nfev=max_evaluations)
    return result.x.reshape(6,3),{'converged':bool(result.success),'evaluations':int(result.nfev),'message':result.message}
