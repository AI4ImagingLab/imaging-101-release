"""Local measured moment surface and explicit physical derivative."""
import numpy as np

def design_matrix(field,reversal_field):
    """Columns 1,H,Hr,H^2,Hr^2,H*Hr,H^3,Hr^3,H^2*Hr,H*Hr^2."""
    h=np.asarray(field);r=np.asarray(reversal_field)
    return np.column_stack((np.ones(h.size),h,r,h*h,r*r,h*r,h**3,r**3,h*h*r,h*r*r))

def local_response(coefficients,field,reversal_field):
    """Evaluate a six- or ten-coefficient moment surface in A m^2."""
    c=np.asarray(coefficients)
    return design_matrix(field,reversal_field)[:,:c.size]@c

def mixed_derivative_density(coefficients):
    """Quadratic rho=-0.5*d^2M/(dH dHr), in A m^2/T^2."""
    return -0.5*np.asarray(coefficients)[5]
