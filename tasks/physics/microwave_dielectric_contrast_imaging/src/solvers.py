"""Local extraction of EISPY2D mathematical kernels.
Copyright (c) André Costa Batista and EISPY2D contributors.
Adapted from revision a72831db93d64c7b80282579f48de4c110627eeb.
Distributed under GNU General Public License version3; see LICENSE.
No upstream framework or computational-imaging library is imported.
"""

import numpy as np
from .physics_model import green_matrices,incident_field,prepare_fft,total_field,scattered_field,physical_contrast,GRID
ALPHA=.001
ITERATIONS=15
def linear_kernel(gs,field):
    return (gs[:,None,:]*field.T[None,:,:]).reshape(gs.shape[0]*field.shape[1],gs.shape[1])
def tikhonov(kernel,measurements,alpha=ALPHA):
    return np.linalg.solve(kernel.conj().T@kernel+alpha*np.eye(kernel.shape[1]),kernel.conj().T@measurements.ravel())
def reconstruct(data,alpha=ALPHA,iterations=ITERATIONS):
    gs,gd=green_matrices(data['receiver_angles']);incident=incident_field(data['illumination_angles']);cache=prepare_fft(GRID,incident.shape[1]);field=incident.copy()
    contrasts=[];inputs=[];legacy=[];physical_residual=[];inner=[];states=[]
    for step in range(iterations):
        inputs.append(field.copy());contrast=tikhonov(linear_kernel(gs,field),data['measurements'],alpha)
        contrasts.append(contrast.copy());field,errors=total_field(contrast.reshape(GRID,GRID),incident,cache);inner.append(errors)
        legacy.append(float(np.linalg.norm(data['measurements']-scattered_field(contrast,field,gs))**2))
        physical=physical_contrast(contrast)
        physical_residual.append(float(np.linalg.norm(data['measurements']-scattered_field(physical,field,gs))/np.linalg.norm(data['measurements'])))
        states.append(float(np.linalg.norm(field-gd@(physical.reshape(-1,1)*field)-incident)/np.linalg.norm(incident)))
    result=contrasts[-1].real.reshape(1,GRID,GRID).astype('float64')
    return {'reconstruction':result},{'GS':gs,'GD':gd,'incident':incident,'complex_contrasts':np.asarray(contrasts),'total_field_inputs':np.asarray(inputs),'final_total_field':field,'legacy_objective':np.asarray(legacy),'physical_residual':np.asarray(physical_residual),'state_residual':np.asarray(states),'inner_history':inner}
