"""Load all author-processed observations; do not interpolate the primary delay axis."""
from pathlib import Path
import numpy as np
from .physics_model import FourierGrid,SHGFrog

FREQUENCY_STEP=407474458795.6243
CARRIER_ANGULAR_FREQUENCY=903760270468365.0

def load_observations(path):
    """Return (trace[512,512], model) in delay/frequency order, SI coordinates."""
    with np.load(Path(path),allow_pickle=False) as f:
        d={k:f[k][0].copy() for k in ['trace','delay','time','angular_frequency']}
    if d['trace'].shape!=(512,512) or not all(np.isfinite(v).all() for v in d.values()):raise ValueError('Invalid acquisition')
    grid=FourierGrid(512,FREQUENCY_STEP)
    if not np.array_equal(grid.t,d['time']):raise ValueError('Time calibration differs from declared grid')
    if not np.array_equal(grid.w+2*CARRIER_ANGULAR_FREQUENCY,d['angular_frequency']):raise ValueError('Frequency calibration differs from declared grid')
    return d['trace'],SHGFrog(grid,d['delay'])

def gaussian_initialization(grid,width_fs=50.):
    """Independent zero-phase Gaussian, intensity FWHM width_fs; no target field."""
    return grid.forward(np.exp(-2*np.log(2)*(grid.t/(width_fs*1e-15))**2))

def unit_energy_field(grid,spectrum):
    """Return complex envelope [s^-1/2], with sum(abs(E)^2)*dt=1."""
    e=grid.backward(spectrum)
    return e/np.sqrt(np.sum(e.real**2+e.imag**2)*grid.dt)

def regrid_for_pcgpa(measured,model):
    """Explicit comparison-only linear delay regrid; zero outside measured support."""
    g=model.grid
    d=np.column_stack([np.interp(g.t,model.delays,measured[:,j],left=0.,right=0.) for j in range(g.n)])
    return d,SHGFrog(g,g.t.copy())
