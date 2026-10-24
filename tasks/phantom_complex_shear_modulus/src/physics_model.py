"""Explicit published planar harmonic elasticity and author polynomial derivatives.
Adapted from MRE-PINN fields.py/baseline/filters.py, GPL-3.0; see LICENSE.
"""
import math
from functools import lru_cache
import numpy as np
from scipy.ndimage import convolve,median_filter

@lru_cache(None)
def polynomial_kernels(order=2,kernel_size=9):
    """Return 2D total-degree polynomial-fit convolution kernels in author ordering."""
    if kernel_size%2!=1 or kernel_size<3 or order<0:raise ValueError("Invalid polynomial window")
    half=kernel_size//2;axis=np.arange(-half,half+1)
    coords=np.stack(np.meshgrid(axis,axis),axis=-1).reshape(-1,2)
    powers=np.stack(np.meshgrid(np.arange(order+1),np.arange(order+1)),axis=-1).reshape(-1,2)
    powers=powers[powers.sum(axis=1)<=order]
    if len(coords)<len(powers):raise ValueError("Polynomial order exceeds window rank")
    matrix=np.zeros((len(coords),len(powers)))
    for i in range(len(coords)):
        for j in range(len(powers)):matrix[i,j]=np.power(coords[i],powers[j]).prod()
    kernels=np.linalg.pinv(matrix)
    kernels*=np.array([math.factorial(int(p[0]))*math.factorial(int(p[1])) for p in powers])[:,None]
    kernels=kernels.reshape(-1,kernel_size,kernel_size)
    return {tuple(p):k for p,k in zip(powers,kernels)}

def polynomial_filter(wave,x,y,derivative=(0,0),order=2,kernel_size=9):
    """Convolve each z/component plane with reflect padding; return same shape."""
    kernel=polynomial_kernels(order,kernel_size)[tuple(derivative)]
    result=convolve(wave,kernel[:,:,None,None],mode='reflect')
    if derivative[0]:result=result/np.mean(np.diff(x))**derivative[0]
    if derivative[1]:result=result/np.mean(np.diff(y))**derivative[1]
    return result

def planar_laplacian(wave,x,y,savgol=True,order=2,kernel_size=9):
    """Author planar x/y Laplacian, no z derivative; double np.gradient comparison."""
    if savgol:
        return polynomial_filter(wave,x,y,(2,0),order,kernel_size)+polynomial_filter(wave,x,y,(0,2),order,kernel_size)
    gx=np.gradient(wave,x,axis=0);gy=np.gradient(wave,y,axis=1)
    return np.gradient(gx,x,axis=0)+np.gradient(gy,y,axis=1)

def outlier_filter(a,threshold=3.0):
    """Author separate magnitude/phase median-relative replacement; not clipping."""
    if np.iscomplexobj(a):return outlier_filter(abs(a),threshold)*np.exp(1j*outlier_filter(np.angle(a),threshold))
    weights=np.array([1,2,3,2,1]);footprint=np.einsum('i,j->ij',weights,weights)>1
    footprint=footprint.reshape(5,5,*([1]*(a.ndim-2)))
    median=median_filter(a,footprint=footprint,mode='reflect');relative=(a-median)/median
    result=np.where(relative>threshold,threshold*median,a)
    return np.where(relative<-threshold,-threshold*median,result)

def helmholtz_inversion(wave,laplacian,frequency,rho=1000.,polar=True,eps=1e-8):
    """Return complex shear modulus Pa; retain source nonnegative loss phase branch."""
    omega=2*np.pi*frequency
    if polar:
        numerator=(rho*omega**2*abs(wave)).sum(axis=-1);denominator=abs(laplacian).sum(axis=-1)
        phase_num=(wave.real*laplacian.real+wave.imag*laplacian.imag).sum(axis=-1)
        phase_den=(abs(wave)*abs(laplacian)).sum(axis=-1)
        amplitude=numerator/(denominator+eps);phase=np.arccos(-phase_num/(phase_den+eps))
        return amplitude*np.exp(1j*phase)
    return (-rho*omega**2*wave).sum(axis=-1)/(laplacian.sum(axis=-1)+eps)

def relative_residual(modulus,wave,laplacian,frequency,support,rho=1000.):
    forcing=rho*(2*np.pi*frequency)**2*wave
    return float(np.linalg.norm((modulus[...,None]*laplacian+forcing)[support])/np.linalg.norm(forcing[support]))
