import numpy as np
from .physics_model import carriers,phases,displacement,effective_height,integrate_gradient

def reconstruct(reference,measurements,unwrap=True,radius_factor=1.):
    c=carriers(reference,radius_factor=radius_factor);p=phases(measurements,c,unwrap);u=displacement(p,c);gradient=-u/effective_height();height=integrate_gradient(*gradient,c['calibration'])
    return height,dict(carriers=c,phases=p,displacement=u,gradient=gradient)
