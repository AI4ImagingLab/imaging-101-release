"""Corrected standardized-residual NLS interface with explicit log parameters.

The author code supplied scalar chi-square to least_squares, giving a rank1
Gauss-Newton system. This explicit interface correction restores the paper NLS
objective; physical equations, initialization and bounds remain unchanged.
"""
import numpy as np
from scipy.optimize import least_squares
from .physics_model import trajectory

def initial_guess(data):
    mu=data['friction'];x=data['displacement'];jump=data['jump_index']
    a=max((np.max(mu)-np.mean(mu[:jump+1]))/np.log(10),.0021)
    index=int(np.argmax(mu));factor=.33
    while factor>1e-10:
        shift=int(np.rint(factor*(len(mu)-index-1)))
        if mu[index+shift]<mu[-1]:factor/=2
        else:
            Dc=max(-(x[index+shift]-x[index])/1000/np.log((mu[index+shift]-mu[-1])/(mu[0]-mu[-1]+a*np.log(10))),1.1e-7)
            return np.log10([a,Dc])
    raise ValueError('No admissible initial decay estimate')

def chi_square(log_parameters,data):
    return float(np.sum(residual_vector(log_parameters,data)**2))

def residual_vector(log_parameters,data):
    predicted=trajectory(log_parameters,data)['friction'][0]
    return (data['friction']-predicted)/.0004

def reconstruct(data):
    initial=initial_guess(data);history=[]
    def objective(x):
        residual=residual_vector(x,data);value=float(np.sum(residual**2));history.append({'log_parameters':x.tolist(),'chi_square':value});return residual
    fit=least_squares(objective,initial,bounds=((np.log10(.002),-7),(np.log10(.04),-3)),max_nfev=10000)
    result=trajectory(fit.x,data)
    return result,{'history':history,'nfev':fit.nfev,'success':bool(fit.success),'message':fit.message,'objective':float(np.sum(fit.fun**2)),'initial_log_parameters':initial.tolist(),'jacobian':fit.jac}
