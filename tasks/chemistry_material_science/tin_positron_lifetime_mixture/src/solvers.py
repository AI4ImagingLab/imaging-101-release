import numpy as np
from scipy.optimize import least_squares
from .physics_model import residual,prediction_native
from .preprocessing import prepare

def reconstruct(observed):
    area,background=prepare(observed);parameters=[];predictions=[];histories=[]
    for y,total,bg in zip(observed,area,background):
        p=np.array([198/50,.9,380/50,.1,2200/50,.005,198/50,5468/50,.5,198/50,5468/50,.5,bg]);previous=float(np.sum((prediction_native(p,total)[33:948]-y[33:948])**2/(y[33:948]+1)));history=[]
        for run in range(20):
            fit=least_squares(residual,p,args=(y,total),method='lm',ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=2800);p=fit.x;value=float(np.sum(fit.fun**2));history.append((value,fit.nfev))
            if not fit.success:raise RuntimeError('Levenberg-Marquardt did not converge')
            stop_value=float(np.sum((prediction_native(p,total)[33:948]-y[33:948])**2/(y[33:948]+1)))
            if previous-stop_value<=1e-5:break
            previous=stop_value
        if not np.all(np.isfinite(p)) or np.any(p[[0,2,4,6,9]]<=0) or np.any(p[[1,3,5,8,11,12]]<0) or total-916*p[12]<=0:raise RuntimeError('Nonphysical mixture result')
        parameters.append(p);predictions.append(prediction_native(p,total));histories.append(history)
    return np.array(parameters),np.array(predictions),histories
