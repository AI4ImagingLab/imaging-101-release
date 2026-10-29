"""Local Gaussian conditioning, exact likelihood gradient and original TNC fit."""
import numpy as np
from scipy.optimize import fmin_tnc
from .physics_model import covariance_operators,acceleration_parameter,KERNELS
from .preprocessing import load_observations,reconstruction_grid

def factor_observations(z,y,error,theta,kernel):
    covariance,gradient,_,_=covariance_operators(z,z,theta,kernel)
    matrix=covariance+np.diag(error**2)
    lower=np.linalg.cholesky(matrix)
    alpha=np.linalg.solve(lower.T,np.linalg.solve(lower,y))
    return covariance,gradient,lower,alpha

def negative_log_likelihood(theta,z,y,error,kernel):
    _,gradient,lower,alpha=factor_observations(z,y,error,theta,kernel)
    value=.5*np.dot(y,alpha)+np.log(np.diag(lower)).sum()+len(y)/2*np.log(2*np.pi)
    inverse=np.linalg.solve(lower.T,np.linalg.solve(lower,np.eye(len(y))))
    derivatives=.5*np.einsum('ij,kij->k',inverse-np.outer(alpha,alpha),gradient)
    return float(value),derivatives

def condition(z,y,error,grid,theta,kernel):
    """Return H,Hprime and marginal per-grid-point2x2 covariance of[H,Hprime]."""
    _,_,lower,alpha=factor_observations(z,y,error,theta,kernel)
    kstar,_,dstar,_=covariance_operators(z,grid,theta,kernel)
    mean=kstar.T@alpha;derivative=dstar.T@alpha
    v=np.linalg.solve(lower,kstar);dv=np.linalg.solve(lower,dstar)
    prior,_,_,prior_mixed=covariance_operators(np.array([0.]),np.array([0.]),theta,kernel)
    variance=prior[0,0]-np.sum(v*v,axis=0)
    derivative_variance=prior_mixed[0,0]-np.sum(dv*dv,axis=0)
    cross=-np.sum(v*dv,axis=0)
    point_covariance=np.empty((len(grid),2,2));point_covariance[:,0,0]=variance;point_covariance[:,1,1]=derivative_variance;point_covariance[:,0,1]=point_covariance[:,1,0]=cross
    return mean,derivative,point_covariance

def fit_kernel(z,y,error,grid,kernel,initial_factor=1.):
    """Original data-based initialization, positive bounds and TNC defaults."""
    original_initial=np.array([np.ptp(y)/2,np.ptp(z)/2],dtype=np.float64)
    initial=original_initial*initial_factor
    bounds=[(value/1e15,None) for value in original_initial]
    theta,evaluations,status=fmin_tnc(negative_log_likelihood,initial,args=(z,y,error,kernel),bounds=bounds,messages=0)
    if status not in [0,1,2]:raise RuntimeError(f'TNC did not establish convergence: status{status}')
    h,hp,covariance=condition(z,y,error,grid,theta,kernel)
    return h,hp,covariance,{'theta':theta,'evaluations':int(evaluations),'status':int(status),'negative_log_likelihood':negative_log_likelihood(theta,z,y,error,kernel)[0]}

def reconstruct(path='data/raw_data.npz',initial_factor=1.):
    z,y,error=load_observations(path);grid=reconstruction_grid();hs=[];derivatives=[];covariances=[];records=[]
    for kernel in KERNELS:
        h,hp,covariance,record=fit_kernel(z,y,error,grid,kernel,initial_factor)
        hs.append(h);derivatives.append(hp);covariances.append(covariance);records.append(record)
    h=np.array(hs);hp=np.array(derivatives);covariance=np.array(covariances)
    result={'reconstruction':acceleration_parameter(grid,h,hp)[None],'hubble_rate':h[None],'hubble_derivative':hp[None],'hubble_constant':h[None,:,0],'point_covariance':covariance[None]}
    return result,records
