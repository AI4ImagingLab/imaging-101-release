"""Matched-control estimation and complete corrected APb-FRET pipeline.

Adapted from AccPbFRET, Roszik/Szollosi/Vereb GPL-3.0-or-later.
"""
import numpy as np
from .preprocessing import preprocess_sample,registration_shift,translate_zero,SIGNAL_THRESHOLD
from .physics_model import corrected_efficiency

def _display_factor(value):
    """Native float32 factor followed by three-decimal GUI rounding and parse."""
    return np.float32(format(float(np.float32(value)),'.3f'))

def estimate_corrections(donor_control,acceptor_control,target,threshold=SIGNAL_THRESHOLD):
    """Return gamma,delta,epsilon,alpha and the three input-defined masks."""
    d=np.asarray(donor_control,np.float32);a=np.asarray(acceptor_control,np.float32);t=np.asarray(target,np.float32)
    dm=d[0,0]>threshold;dam=d[1,0]>threshold;am=a[0,1]>threshold;tm=(t[0,0]>threshold)&(t[0,1]>threshold)
    if not all(m.any() for m in [dm,dam,am,tm]):raise ValueError('empty input-selected control or target support')
    mean=lambda v,m:np.mean(v[m],dtype=np.float64)
    gamma=mean(d[0,0],dm)/mean(d[1,0],dam)
    delta=mean(a[0,0],am)/mean(a[0,1],am)
    epsilon=mean(a[1,0],am)/mean(a[0,1],am)
    alpha=np.mean(t[1,1][tm]/t[0,1][tm],dtype=np.float64)
    return np.array([_display_factor(v) for v in [gamma,delta,epsilon,alpha]],dtype=np.float32),(dm,am,tm)

def reconstruct(observations,sigma=1.0):
    """Full sample-first STCYX (3,2,2,Y,X) input -> finite maps and identified stages."""
    raw=np.asarray(observations)
    if raw.shape[:3]!=(3,2,2):raise ValueError('three matched samples in STCYX order required')
    shift=registration_shift(raw[2,0,0],raw[2,1,0])
    if shift!=(0,0):
        raw=raw.copy();raw[2,1]=translate_zero(raw[2,1],shift)
    stages=[preprocess_sample(a,sigma=sigma) for a in raw]
    factors,masks=estimate_corrections(*(s['smooth'] for s in stages))
    target=stages[2]['smooth'];support=masks[2]
    efficiency=np.zeros(target.shape[-2:],np.float32)
    efficiency[support]=corrected_efficiency(target[0,0][support],target[1,0][support],target[0,1][support],factors)
    if not np.isfinite(efficiency).all():raise ValueError('nonfinite physical result on fixed support')
    return {'reconstruction':efficiency[None],'correction_factors':factors[None],'stages':stages,'masks':masks,'registration_shift':shift}
