"""Published weighted-droplet iteration and optional published cost comparison.
Adapted from Amit Devra, LGPL-3.0. No ideal target is used by either solver.
"""
import numpy as np
from scipy.optimize import minimize
from .preprocessing import droplets
from .physics_model import basis, quaternion_matrix, matrix_quaternion, overlap_coefficients, coefficients_matrix, euler_quaternion

def correlation_signs(c):
    """Effective upstream final branch: largest diagonal controls relative signs.
    The upstream earlier ideal-condition branches are always overwritten.
    """
    s=np.sign(c.real); k=int(np.argmax(np.abs(np.sqrt(np.diag(c)))))
    signs=np.ones(4)
    if k==3:
        signs[:3]=s[:3,3]
    else:
        signs[:3]=s[:3,k]*s[k,3]
        signs[k]=s[k,3]
    return signs

def reconstruct(data, updates=9):
    """Complete acquisition -> unitary(1,2,2) plus numerical diagnostics.
    Nine author updates followed by normalization; no convergence shortcut.
    """
    f=droplets(data['expectation']);w=data['quadrature_weight']
    b=basis(data['polar_angle'],data['azimuth'])
    corr=(f.conj()*w) @ f.T
    c=corr/0.00995
    q=correlation_signs(c)*np.sqrt(np.diag(c))
    history=[]
    for _ in range(updates+1):
        combined=q@f
        coef=overlap_coefficients(combined,b,w)
        q=np.array([coef[0].imag,coef[1].imag,coef[2].imag,coef[3].real])
        q=q/np.linalg.norm(q)
        history.append(q.copy())
    return {'reconstruction':quaternion_matrix(q)[None], 'quaternion_history':np.array(history),'droplets':f,'basis':b,'correlation':corr}

def optimization_inputs(data):
    """Author per-channel normalization and non-unit normalized initial estimate."""
    f=droplets(data['expectation']);w=data['quadrature_weight'];b=basis(data['polar_angle'],data['azimuth'])
    c=overlap_coefficients(f,b,w)
    channels=np.stack([coefficients_matrix(x)/np.linalg.norm(x) for x in c])
    corr=(f.conj()*w)@f.T/0.00995
    q0=correlation_signs(corr)*np.sqrt(np.diag(corr))
    q=(q0/np.sqrt(np.sqrt(np.sum(q0*q0)))).real
    a,b,c,d=q
    x=np.array([np.arctan2(c,d)-np.arctan2(a,b),np.arccos(1-2*(a*a+b*b)),np.arctan2(c,d)+np.arctan2(a,b)])
    return channels,x

def objective(x, channels):
    """Author equation31: half sum of four squared Frobenius residuals."""
    q=euler_quaternion(x);u=quaternion_matrix(q)
    return float(0.5*np.sum(np.abs(channels-q[:,None,None]*u)**2))

def optimize_comparison(data):
    """Fresh BFGS minimization of published objective; trajectory is not MATLAB parity."""
    channels,x=optimization_inputs(data);history=[x.copy()]
    result=minimize(objective,x,args=(channels,),method='BFGS',options={'maxiter':45,'gtol':1e-8,'eps':1e-8},callback=lambda z:history.append(z.copy()))
    history=np.array(history)
    return {'reconstruction':quaternion_matrix(euler_quaternion(result.x))[None], 'angles':history,'objective':np.array([objective(z,channels) for z in history]),'message':str(result.message)}
