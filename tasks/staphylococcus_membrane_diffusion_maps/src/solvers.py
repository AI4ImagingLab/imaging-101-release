"""Original local per-pixel nonlinear least squares; no domain-library imports."""
import warnings,time
import numpy as np
from scipy.optimize import curve_fit
from .preprocessing import prepare_stack,bin_labels,correct_bleaching,correlate_trace,BINNING
from .physics_model import correlation_model,PSF_SIGMA,CURVATURE

def fit_curve(curve,aperture=.32,sigma=PSF_SIGMA,curvature=CURVATURE):
    """Fit N,D,Ginf using original unconstrained LM and SciPy default stopping.

    Original D initial guess uses .23um, distinct from calibrated .19um PSF.
    Return three parameters and model samples; retain explicit invalid sentinels
    only for scientifically labeled diagnostic branches.
    """
    curve=np.asarray(curve);t=curve[:,0];g=curve[:,1]
    p0=(max(0.,1/g[0]/3),.23**2/4/np.median(t),0.)
    def model(lag,N,D,offset):return correlation_model(lag,N,D,offset,aperture,sigma,curvature)
    try:
        parameters,_=curve_fit(model,t,g,p0=p0)
        fitted=model(t,*parameters)
    except Exception as exc:
        warnings.warn('Original correlation-fit fallback: '+str(exc),RuntimeWarning)
        parameters=np.full(3,-1.);fitted=np.zeros_like(g)
    return parameters,fitted

def reconstruct(frames,cell_labels,binning=BINNING,bleaching=True,curvature=CURVATURE,sigma=PSF_SIGMA,return_details=False,require_physical=True):
    """Fresh movie-to-N,D,offset reconstruction on every fixed public cell bin.

    Standard binning2 output keys: reconstruction=D, molecule_number=N,
    correlation_offset=Ginf; all batch-first float64, background exactly zero.
    Optional diagnostic settings never silently replace the benchmark method.
    """
    start=time.perf_counter();stack=prepare_stack(frames);labels=bin_labels(cell_labels,binning)
    shape=labels.shape;parameters=np.zeros(shape+(3,));curves={};fits={};bleach_details={};bad=[]
    for y,x in zip(*np.nonzero(labels)):
        trace=stack[:,y*binning:(y+1)*binning,x*binning:(x+1)*binning].mean(axis=(1,2))
        if bleaching:corrected,details=correct_bleaching(trace,True)
        else:corrected=trace;details={'success':True,'disabled':True}
        curve=correlate_trace(corrected)
        pars,prediction=fit_curve(curve,aperture=.16*binning,sigma=sigma,curvature=curvature)
        parameters[y,x]=pars
        if np.any(pars[:2]<=0) or not np.isfinite(pars).all() or not details['success']:bad.append((int(y),int(x)))
        if return_details:
            curves[(int(y),int(x))]=curve;fits[(int(y),int(x))]=prediction
            bleach_details[(int(y),int(x))]={'raw_trace':trace,'corrected_trace':corrected,**details}
    if bad and require_physical:raise ValueError(f'Nonphysical or failed selected fits: {bad}')
    result={'reconstruction':parameters[...,1][None],'molecule_number':parameters[...,0][None],'correlation_offset':parameters[...,2][None]}
    details={'labels':labels,'curves':curves,'fits':fits,'bleaching':bleach_details,'invalid_pixels':bad,'runtime_seconds':time.perf_counter()-start,'mean_intensity':stack.mean(axis=0)}
    return (result,details) if return_details else result
