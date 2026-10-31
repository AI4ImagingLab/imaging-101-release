"""Local closed-form Helms MTsat/R1 inverse, extracted from qMRLab MTSAT_exec.
MIT copyright(c)2017 NeuroPoly; see NOTICE. No clipping or parameter regularization.
"""
import numpy as np
from .preprocessing import analysis_mask
from .physics_model import ANGLES,TR
def reconstruct(data,repetition_times=None):
    mask=analysis_mask(data)
    p,t,m=(data[k][mask] for k in ('PDw','T1w','MTw'))
    ap,at,am=ANGLES;tp,tt,tm=TR if repetition_times is None else repetition_times
    r=.5*((at/tt)*t-(ap/tp)*p)/(p/ap-t/at)
    amplitude=(tp*at/ap-tt*ap/at)*((p*t)/(tp*at*t-tt*ap*p))
    delta=100*(tm*(am*(amplitude/m)-np.ones_like(m))*r-am**2/2)
    mt=np.zeros(mask.shape);r1=np.zeros(mask.shape)
    mt[mask]=delta;r1[mask]=r
    return {'reconstruction':mt[None],'r1':r1[None]}
def mtr(data,mask):
    out=np.zeros(mask.shape);out[mask]=100*(data['PDw'][mask]-data['MTw'][mask])/data['PDw'][mask]
    return out
