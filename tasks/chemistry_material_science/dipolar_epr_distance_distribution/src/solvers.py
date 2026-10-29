"""Separable nonlinear least squares, with nonnegative Gaussian weights."""
import numpy as np
from scipy.optimize import least_squares,nnls
from .physics_model import signal_basis,gaussian_basis,DISTANCE
from .preprocessing import prepare
START=np.array([.01,.3,50,2.5,.2,4.5,.2])
LOWER=np.array([0,.252,.01,1,.05,1,.05]);UPPER=np.array([1,.348,5000,20,2.5,20,2.5])
def solve(time,echo,initial=START,max_nfev=1000):
    t,y=prepare(time,echo)
    def residual(p):
        a=signal_basis(t,p);x,_=nnls(a,y);return a@x-y
    fit=least_squares(residual,initial,bounds=(LOWER,UPPER),ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=max_nfev)
    p=fit.x;a=signal_basis(t,p);amp,_=nnls(a,y);dist=gaussian_basis(DISTANCE,*p[[3,4,5,6]])@amp;dist/=np.trapezoid(dist,DISTANCE)
    pars=np.array([amp[0],amp[1],p[2],p[3],p[5],p[0],p[1],p[4],p[6]])
    if pars[3]>pars[4]:pars=pars[[1,0,2,4,3,5,6,8,7]]
    return dist,a@amp,pars
def reconstruct(data):
    p,signal,pars=solve(data['time'],data['echo'])
    return {'reconstruction':p[None,:],'fitted_echo':signal[None,:],'parameters':pars[None,:]}
