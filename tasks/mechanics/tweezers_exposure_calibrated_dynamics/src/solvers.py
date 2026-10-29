"""Published unbounded Nelder-Mead MLE with observation-derived initialization."""
import numpy as np
from scipy.optimize import minimize
from .preprocessing import spectrum,allan,KT
from .physics_model import prediction,negative_log_likelihood
def reconstruct(trace,kind='psd',tracking_error=True,bins=35,exposure=True):
    data=spectrum(trace,bins) if kind=='psd' else allan(trace)
    initial=[6*np.pi*8.94e-10*530,KT/np.var(trace)]
    if tracking_error:initial.append(10.)
    result=minimize(negative_log_likelihood,initial,args=(data,kind,exposure),method='Nelder-Mead')
    if not result.success:raise RuntimeError(result.message)
    model=prediction(data['x'],result.x,kind,exposure)
    residual=(data['y']-model)/(data['y']/np.sqrt(data['shape']))
    return {'parameters':result.x,'data':data,'model':model,'residual':residual,'nll':float(result.fun),'aic':float(2*(len(result.x)+result.fun)),'iterations':int(result.nit)}
