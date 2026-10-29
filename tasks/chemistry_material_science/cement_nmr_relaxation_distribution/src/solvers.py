"""Local nonnegative regularized Laplace inversion; Georget (2020), BSD-3-Clause."""
import numpy as np
from scipy.optimize import nnls
from .preprocessing import phase_echo
from .physics_model import decay_kernel,RELAXATION_TIME
ALPHA=1e-4
def solve(echo_time,echo,alpha=ALPHA):
    y=phase_echo(echo);k=decay_kernel(echo_time[1:]);m=np.vstack([k,np.sqrt(alpha)*np.eye(len(RELAXATION_TIME))])
    amplitudes=np.array([nnls(m,np.r_[v,np.zeros(len(RELAXATION_TIME))])[0] for v in y])
    return amplitudes,amplitudes@k.T
def reconstruct(data):
    amplitude,fit=solve(data['echo_time'],data['echo'])
    return {'reconstruction':amplitude[None,:],'fitted_echo':fit[None,:]}
