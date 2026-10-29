"""Independent known-particle physics controls only, never benchmark data.

The400 bound observations are existing author simulation. This helper creates
four-camera singleton Gaussian measurements for a deterministic mathematical
unit/control experiment. Inputs: legitimate camera calibration and explicitly
known world position[mm], amplitude[ADU], inverse widths[pixel^-2]. Pixel centres
are integer column,row; no random seed/noise, no clipping or hidden target is
used. Control fields never replace raw_data.npz or the independent author truth.
"""
import numpy as np
from .physics_model import project
def known_particle_images(cameras,point,amplitude=100.,inverse_widths=(1.2,1.4)):
    point=np.asarray(point,dtype=np.float64).reshape(1,3);yy,xx=np.mgrid[:1024,:1024];images=[]
    for camera in cameras:
        u,v=project(point,camera)[0];images.append(amplitude*np.exp(-(inverse_widths[0]*(xx-u)**2+inverse_widths[1]*(yy-v)**2)))
    return np.ascontiguousarray(images)
