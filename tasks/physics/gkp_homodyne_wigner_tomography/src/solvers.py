"""Explicit native positive maximum-likelihood updates, Apache-2.0 adaptation."""
import time
import numpy as np
from .physics_model import hermite_amplitudes,density_update,step_fidelity

def maximum_likelihood(q,theta,cutoff=30,iterations=100,snapshot_steps=(1,2,7,25,50,100)):
    """Full list-mode reconstruction; all samples enter every R update."""
    start=time.perf_counter()
    amplitudes=hermite_amplitudes(q,theta,cutoff=cutoff)
    density=np.identity(cutoff+1)/(cutoff+1)
    previous=density
    L=np.zeros(iterations,dtype=complex);F=np.zeros(iterations,dtype=complex)
    snapshots={}
    for i in range(iterations):
        density,R,p=density_update(density,amplitudes)
        L[i]=np.sum(np.log(p))
        F[i]=step_fidelity(density,previous)
        previous=density
        if i+1 in snapshot_steps:snapshots[i+1]=density.copy()
    return {'density_matrix':density,'likelihood':np.real_if_close(L,tol=1e-6),
            'step_fidelity':np.real_if_close(F,tol=1e-6),'snapshots':snapshots,
            'iterations':iterations,'runtime_seconds':time.perf_counter()-start}
