"""Stolt scalar-wave migration; explicit mathematics and no domain solver."""
import numpy as np
from .preprocessing import prepare,taper,TRACE_PITCH_M,VELOCITY_M_S,HORIZONTAL_TAPER,VERTICAL_TAPER
from .physics_model import dispersion,remap

def migrate(data,dt,dx,velocity=VELOCITY_M_S,horizontal_taper=HORIZONTAL_TAPER,vertical_taper=VERTICAL_TAPER,apply_taper=True,bandlimited=False):
    """Migrate time×trace float64 data; return signed image and diagnostics.

    dt seconds, dx metres, velocity metres/second. Output shape unchanged
    for even temporal length; no temporal/spatial resampling in this stage.
    """
    if data.shape[0]%2:raise ValueError('Migration requires even temporal length')
    weighted=data*taper(data.shape,horizontal_taper,vertical_taper) if apply_taper else data
    fk=np.fft.rfft2(weighted,axes=(1,0))
    omega,kx,kz,target,scaling=dispersion(*data.shape,dt,dx,velocity)
    kk=remap(fk,omega,target)*scaling
    kk[0,0]=0
    if bandlimited:kk[target>omega[-1]]=0
    image=np.fft.irfft2(kk,axes=(1,0))
    return image,{'fk':fk,'kk':kk,'scaling':scaling,'kx':kx,'omega':omega,'kz':kz,'target':target,'weighted':weighted}

def reconstruct(data,time_ns,velocity=VELOCITY_M_S,return_diagnostics=False,bandlimited=False):
    """Full time calibration/background-removal/migration on raw acquisition."""
    _,_,prepared=prepare(data,time_ns)
    image,diagnostics=migrate(prepared,(time_ns[1]-time_ns[0])*1e-9,TRACE_PITCH_M,velocity,bandlimited=bandlimited)
    return (image,diagnostics) if return_diagnostics else image

def boundary_diagnostics(diagnostics):
    """Quantify source endpoint-extension power beyond measured bandwidth."""
    fk,kk=diagnostics['fk'],diagnostics['kk']
    outside=diagnostics['target']>diagnostics['omega'][-1]
    fpower=float(np.sum(abs(fk)**2));kpower=float(np.sum(abs(kk)**2))
    return {'unsupported_grid_fraction':float(outside.mean()),
            'unsupported_migrated_energy_fraction':float(np.sum(abs(kk[outside])**2)/kpower) if kpower else 0.,
            'last_measured_frequency_energy_fraction':float(np.sum(abs(fk[-1])**2)/fpower) if fpower else 0.}
