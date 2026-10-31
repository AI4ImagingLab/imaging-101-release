"""Explicit GAP/ADMM projection and native finite spatial-TV dual clipping.

Selected author method from PnP-SCI91fc7aa4; credited Yuan/Liu. No original
library, pretrained model, domain solver or hidden reference is imported.
"""
import numpy as np
from .physics_model import forward,adjoint,difference,difference_adjoint,chronological_cube

def tv_denoise(image,weight=.07,iterations=5):
    """Actual separable-direction finite dual clipping, not an exact TV prox.

    Return x before the final dual update, preserving the author convention.
    TV is spatial per frame; no temporal derivative or positivity constraint.
    """
    horizontal=np.zeros((image.shape[0],image.shape[1],image.shape[2]-1))
    vertical=np.zeros((image.shape[0],image.shape[1]-1,image.shape[2]))
    for _ in range(iterations):
        xh=image-difference_adjoint(horizontal,2)
        xv=image-difference_adjoint(vertical,1)
        x=(xh+xv)/2
        horizontal=np.clip(horizontal+difference(x,2)/5,-weight/2,weight/2)
        vertical=np.clip(vertical+difference(x,1)/5,-weight/2,weight/2)
    return x

def reconstruct_burst(measurement,mask,method='gap',iterations=None,weight=.07,tv_iterations=5):
    """Full native inner sequence; returns mask-order cube and actual history."""
    if method not in ['gap','admm']:raise ValueError('Unknown selected inverse method')
    if iterations is None:iterations=100 if method=='gap' else 200
    energy=np.zeros(mask.shape[1:])
    for code in mask:energy+=code*code
    energy[energy==0]=1 # Native convention; not activated by the bound calibration.
    value=adjoint(measurement,mask);history=[]
    if method=='gap':accumulator=np.zeros_like(measurement)
    else:theta=value.copy();dual=np.zeros_like(value)
    for iteration in range(iterations):
        if method=='gap':
            predicted=forward(value,mask)
            accumulator+=measurement-predicted
            value=value+adjoint((accumulator-predicted)/energy,mask)
            value=tv_denoise(value,weight,tv_iterations)
        else:
            predicted=forward(theta+dual,mask)
            value=theta+dual+adjoint((measurement-predicted)/(energy+1.),mask)
            theta=tv_denoise(value-dual,weight,tv_iterations)
            dual=dual-(value-theta)
        history.append(float(np.linalg.norm(forward(value,mask)-measurement)/np.linalg.norm(measurement)))
    return value,{'iterations':iterations,'relative_observation_residual_history':history,'method':method,'weight':weight}

def reconstruct(data,method='gap',iterations=None,weight=.07):
    """All15 measured exposures -> all210 chronological radiance frames."""
    cubes=[];diagnostics=[]
    for index,y in enumerate(data['observations']):
        cube,info=reconstruct_burst(y,data['mask'],method,iterations,weight)
        cubes.append(chronological_cube(cube,index));diagnostics.append(info)
    return {'reconstruction':np.concatenate(cubes,axis=0)[None]},diagnostics
