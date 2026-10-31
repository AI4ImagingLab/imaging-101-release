import numpy as np
from scipy.optimize import minimize
from .physics_model import objective,diffuse_reflectance

def reconstruct(observed,frequencies,index=1.4,frequency_indices=(0,1)):
    shape=observed.shape[:2];parameters=np.empty(shape+(2,),dtype=float);predicted=np.empty(observed.shape,dtype=float)
    for row in range(shape[0]):
        for col in range(shape[1]):
            fit=minimize(objective,[.02,1.],args=(np.asarray(frequencies)[list(frequency_indices)],observed[row,col,list(frequency_indices)],index),method='Nelder-Mead',options={'maxiter':300,'xatol':.001,'fatol':.001})
            if not fit.success or not np.all(np.isfinite(fit.x)) or np.any(fit.x<=0):raise RuntimeError('Optical inversion did not produce a valid positive solution')
            parameters[row,col]=fit.x;predicted[row,col]=diffuse_reflectance(fit.x,frequencies,index)
    return parameters,predicted
