"""Explicit conductive half-space heat kernel used by CIBOR (MIT)."""
import numpy as np

SECONDS_PER_YEAR=31557600.
TIMES=30.*np.arange(1,22)

def complementary_error(x):
    """Author's finite polynomial approximation, preserved for parity."""
    z=np.abs(np.asarray(x,dtype=float));t=1/(1+.5*z)
    coefficients=[.17087277,-.82215223,1.48851587,-1.13520398,.27886807,-.18628806,.09678418,.37409196,1.00002368]
    p=np.zeros_like(z)+coefficients[0]
    for c in coefficients[1:]:p=c+t*p
    result=t*np.exp(-z*z-1.26551223+t*p)
    return np.where(np.asarray(x)<0,2-result,result)

def heat_matrix(depth,diffusivity=1e-6,times=TIMES):
    """Map piecewise-constant boundary anomalies (K) to depth anomalies (K)."""
    response=complementary_error(np.asarray(depth)[:,None]/(2*np.sqrt(diffusivity*SECONDS_PER_YEAR*np.asarray(times)[None,:])))
    response[:,1:]-=response[:,:-1].copy()
    zero=np.asarray(depth)==0
    response[zero]=0.;response[zero,0]=1.
    return response

def forward(matrix,history):
    return matrix@history

def adjoint(matrix,values):
    return matrix.T@values
