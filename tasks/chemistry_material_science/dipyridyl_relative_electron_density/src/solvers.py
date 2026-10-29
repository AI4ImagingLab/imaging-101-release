"""Published explicit double-threshold charge flipping on the complete cell.

Copyright2010 UChicago Argonne LLC; local extraction by benchmark maintainers
2026 under retained Argonne license. No domain solver/runtime imports.
"""
import time
import numpy as np
from .preprocessing import initialize_structure_factors
from .physics_model import charge_flip_update,source_origin_shift

def reconstruct(data,seed=20261022,iterations=200,k_factor=.1,k_max=30.,stop_residual=0.,snapshot_steps=(1,2,7,25,100,200)):
    """Full inverse; fixed selected200-cycle budget with source <5% early exit.

    stop_residual optionally represents the author's GUI cancellation criterion;
    the standard benchmark uses only the200-cycle cap and intrinsic5% criterion.
    """
    start=time.perf_counter();initial=initialize_structure_factors(data,seed)
    state=initial.copy();amplitude=abs(initial);history=[];snapshots={}
    for cycle in range(1,iterations+1):
        state,diag=charge_flip_update(state,amplitude,k_factor,k_max)
        history.append([cycle,diag['residual_percent']])
        if cycle in snapshot_steps:snapshots[cycle]=np.fft.fftn(np.fft.fftshift(state)).real/10.
        if diag['residual_percent']<5. or (stop_residual>0 and diag['residual_percent']<=stop_residual):break
    shift,chi2,terms=source_origin_shift(state)
    density=np.fft.fftn(np.fft.fftshift(state)).real/10.
    density=np.roll(density,tuple(shift),axis=(0,1,2))
    return {'reconstruction':density,'history':np.asarray(history),'snapshots':snapshots,
            'iterations':cycle,'origin_shift':shift,'origin_score':chi2,'origin_terms':terms,
            'runtime_seconds':time.perf_counter()-start}
