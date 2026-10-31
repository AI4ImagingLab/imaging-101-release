import numpy as np
BOLTZMANN=1.380649e-23
TEMPERATURE=293.
VISCOSITY=8.9e-4
REFRACTIVE_INDEX=1.33
WAVELENGTH_NM=817.
ANGLE_RAD=150/180*np.pi

def relaxation_times(radius_nm):
    q=4*np.pi*REFRACTIVE_INDEX/(WAVELENGTH_NM/1e9)*np.sin(ANGLE_RAD/2)
    diffusion=BOLTZMANN*TEMPERATURE/(6*np.pi*VISCOSITY*(np.asarray(radius_nm)/1e9))
    return 1/(diffusion*q*q)

def kernel(time_s,relaxation_s):
    return np.exp(-np.asarray(time_s)[:,None]/np.asarray(relaxation_s)[None,:])

def curvature_matrix(n):
    m=np.zeros((n,n))
    for j in range(1,n-1):m[j,j-1:j+2]=[-1,2,-1]
    return m

def predict(weights,t,s,beta):
    return 1+beta[None,:]*(kernel(t,s)@weights.T)**2

def objective(weights,k,g1,alpha=.1):
    m=curvature_matrix(len(weights));r=k@weights-g1
    return float(r@r+1000*((weights.sum()-1)**2+weights[0]**2+weights[-1]**2)+alpha*np.sum((m@weights)**2))
