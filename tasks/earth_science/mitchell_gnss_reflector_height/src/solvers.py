"""Complete fixed-QC acquisition -> one reflector height per measured arc."""
import numpy as np
from .physics_model import spectrum
def reconstruct(data,precision=.005,stride=1):
    heights=[];offset=0
    for i,count in enumerate(data['counts']):
        e=data['elevation'][offset:offset+count];y=data['signal'][offset:offset+count];offset+=count
        order=np.argsort(e);x=np.sin(e[order]*np.pi/180)/data['half_wavelength'][i];y=y[order]
        h,p=spectrum(x[::stride],y[::stride],precision=precision)
        heights.append(h[np.argmax(p)])
    return np.array(heights)[None]
