"""Explicit PALMA PPXA+ updates, batched independent spectral bands.

Adapted from Cherni, Chouzenoux and Delsuc PALMA.py. The actual source
header specifies CC-BY-NC-SA4.0; see NOTICE. No upstream runtime imports.
"""
import numpy as np
from scipy.special import lambertw
from .physics_model import kernel

def approximate_lambert_exponential(values):
    """Preserve original overflow-safe approximation of W(exp(c))."""
    x=np.asarray(values);small=x<50
    with np.errstate(all='ignore'):
        result=x*(1-np.log(x)/(1+x))+.00303583748046/(1+x-50)**.75
        result[small]=np.real(lambertw(np.exp(x[small]),k=0,tol=1e-8))
    return np.nan_to_num(result)

def entropy_l1_prox(values,lamda=.01,a=1.):
    if lamda==0:return np.sign(values)*np.maximum(abs(values)-1,0)
    c=(a*values-a*(1-lamda))/lamda-1-np.log(lamda)+2*np.log(a)
    return lamda/a*approximate_lambert_exponential(c)

def ball_projection(values,observations,eta):
    difference=values-observations
    norm=np.linalg.norm(difference,axis=0)
    with np.errstate(divide='ignore',invalid='ignore'):
        scale=np.minimum(eta/norm,1)
    return values+difference*scale-difference

def ppxa(K,inverse,observations,eta,nbiter=200000,lamda=.01,precision=1e-12):
    """Return bands x diffusion amplitudes and per-band last iteration index.

    A vectorized column performs exactly the scalar author's PPXA+ iteration.
    Each column retains its own scaling, error ball and stopping criterion.
    """
    Y=np.asarray(observations).T.copy();scale=Y[0].copy();Y/=scale
    eta=np.asarray(eta)/scale
    m,n=K.shape;columns=Y.shape[1]
    x0=np.ones((n,columns))*np.sum(Y,axis=0)/(m*n)
    old=x0.copy();tmp1=x0.copy();tmp2=K@x0
    x=inverse@(tmp1+K.T@tmp2)
    active=np.ones(columns,dtype=bool);last=np.full(columns,nbiter-1,dtype=np.int64)
    for iteration in range(nbiter):
        active_idx=np.flatnonzero(active)
        if not len(active_idx):break
        a=tmp1[:,active_idx];b=tmp2[:,active_idx];current=x[:,active_idx]
        first=entropy_l1_prox(a,lamda,1.)
        second=ball_projection(b,Y[:,active_idx],eta[active_idx])
        c=inverse@(first+K.T@second);difference=c-current;reflected=c+difference
        tmp1[:,active_idx]=a+1.99*(reflected-first)
        tmp2[:,active_idx]=b+1.99*(K@reflected-second)
        current=current+1.99*difference;x[:,active_idx]=current
        relative=np.linalg.norm(current-old[:,active_idx],axis=0)/np.linalg.norm(current,axis=0)
        stopped=relative<precision
        if not np.all(np.isfinite(current)):raise FloatingPointError('Nonfinite PPXA+ iterate')
        last[active_idx[stopped]]=iteration;active[active_idx[stopped]]=False
        old[:,active_idx]=current
    return (x*scale).T,last

def reconstruct(prepared,nbiter=200000,lamda=.01,precision=1e-12):
    K=kernel(prepared['b_values']);inverse=np.linalg.inv(np.eye(K.shape[1])+K.T@K)
    result,last=ppxa(K,inverse,prepared['observations'],prepared['eta'],nbiter,lamda,precision)
    return {'reconstruction':result[None]}, {'last_iteration':last,'kernel':K,'eta':prepared['eta'],'residual':np.linalg.norm(result@K.T-prepared['observations'],axis=1)}
