"""Explicit full-pattern intensity model and internal-standard mass calibration."""
import numpy as np
def forward(matrix,coefficients):
    return np.sum(matrix*coefficients[None,:],axis=1)
def rwp(coefficients,matrix,counts):
    residual=counts-forward(matrix,coefficients)
    return float(np.sqrt(np.sum(residual**2/counts)/np.sum(counts)))
def rwp_gradient(coefficients,matrix,counts):
    residual=forward(matrix,coefficients)-counts; value=rwp(coefficients,matrix,counts)
    return matrix.T@(residual/counts)/(np.sum(counts)*value)
def finite_gradient(fun,x,step=.001):
    out=np.empty_like(x)
    for j in range(len(x)):
        plus=x.copy(); minus=x.copy(); plus[j]+=step; minus[j]-=step
        out[j]=(fun(plus)-fun(minus))/(2*step)
    return out
def mass_fractions(coefficients,rir,standard_percent=20.):
    if coefficients[0]<=0: raise ValueError('Positive corundum amplitude required')
    return np.round(standard_percent*coefficients/coefficients[0]*rir[0]/rir,4)
