"""Four explicit power updates from the published example, no eigensolver shortcut."""
import numpy as np
from .preprocessing import backscatter,LUMINANCE
from .physics_model import build_kernel,mass,roughness
def reconstruct(brdf,iterations=4):
    if not isinstance(iterations,int) or iterations<1: raise ValueError('Positive integer iterations required')
    theta,rgb=backscatter(brdf); kernel=build_kernel(theta,rgb@LUMINANCE)
    p=np.ones(89); history=[]
    for _ in range(iterations):
        p=kernel@p; history.append(p.copy())
    pdf=np.r_[p,0.]; pdf/=mass(pdf)
    return {'reconstruction':pdf[None,:],'roughness':roughness(pdf)[None,:]}, {
        'theta':theta,'rgb':rgb,'kernel':kernel,'powers':np.asarray(history)}
