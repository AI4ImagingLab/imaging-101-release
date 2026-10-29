"""Rectilinear point-source point-lens acquisition model."""
import numpy as np
def separation(time,parameters):
    t0,u0,tE=parameters
    return np.sqrt(((time-t0)/tE)**2+u0**2)
def magnification(time,parameters):
    u=separation(time,parameters)
    return (u*u+2)/(u*np.sqrt(u*u+4))
def magnification_derivative(u):
    return -8/(u*u*(u*u+4)**1.5)
def predict(time,parameters,fluxes):
    return fluxes[0]*magnification(time,parameters)+fluxes[1]
