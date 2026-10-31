"""Scientific confidence envelopes using every selected parameter case."""
import numpy as np
from .solvers import profile_batch

def confidence_envelopes(details,observations):
    """Return three (minimum,median,maximum) profiles, each array(5,N)."""
    envelopes=[]
    for cases in details['cloud_parameters']:
        _,profiles=profile_batch(cases,*observations,details['z1'],details['z2'],clip_rates=False)
        envelopes.append((np.min(profiles,axis=0),np.median(profiles,axis=0),np.max(profiles,axis=0)))
    return envelopes
