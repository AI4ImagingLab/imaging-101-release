"""Strict observation loading and source-preserving signal conversion."""
from pathlib import Path
import numpy as np
from .physics_model import acquisition_log

TR_SECONDS = 0.005
FLIP_RADIANS = np.pi / 6.0
RELAXIVITY = 4.5
RAW_SHAPES = {'signal': (1,1321,60,50), 'relaxation_rate': (1,60,50),
              'time_seconds': (1,1321), 'plasma_concentration': (1,1321)}

def load_observations(path):
    """Load the four exact float64 batch-first observations without coercion."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != set(RAW_SHAPES):
            raise ValueError('Raw observation keys do not match the declared contract')
        result={k:archive[k] for k in RAW_SHAPES}
    for key,shape in RAW_SHAPES.items():
        a=result[key]
        if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError(f'Invalid observation {key}: expected finite float64 {shape}')
    if np.any(result['signal']<=0) or np.any(result['relaxation_rate']<=0):
        raise ValueError('Signal and calibration rates must be positive')
    if not np.all(np.diff(result['time_seconds'][0])>0):
        raise ValueError('Time samples must increase strictly')
    return result

def signal_to_concentration(signal_curves, relaxation_rates):
    """Convert (N,T) signal and (N,) s^-1 calibration into R1 and mM Ct.

    Both arrays are fresh float64 calculations.  The first two measured frames
    define the source's baseline signal.  Source invalid-log and nonpositive-R1
    branches are retained; no fitted kinetic parameter determines this stage.
    """
    s=np.asarray(signal_curves,dtype=np.float64)
    rates=np.asarray(relaxation_rates,dtype=np.float64)
    if s.ndim!=2 or rates.shape!=(len(s),) or s.shape[1]<2:
        raise ValueError('Expected (N,T>=2) signal curves and (N,) relaxation rates')
    if not np.isfinite(s).all() or not np.isfinite(rates).all():
        raise ValueError('Signal conversion requires finite inputs')
    baseline=np.mean(s[:,:2],axis=1)
    if np.any(baseline<=0):raise ValueError('Measured baseline must be positive')
    a=s/baseline[:,None]
    e0=np.exp(-rates*TR_SECONDS)[:,None]
    cosine=np.cos(FLIP_RADIANS)
    e=(1.0-a+a*e0-e0*cosine)/(1.0-a*cosine+a*e0*cosine-e0*cosine)
    r1=np.zeros_like(s)
    valid=e>0
    r1[valid]=(-1.0/TR_SECONDS)*acquisition_log(e[valid])
    ct=np.where(r1>0,(r1-rates[:,None])/RELAXIVITY,0.0)
    ct[~np.isfinite(ct)]=0.0
    return r1,ct

def prepare_curves(observations):
    """Return native Fortran-raster (N,T), t in minutes, and plasma mM."""
    signal=observations['signal'][0]
    if signal.ndim!=3 or signal.shape[1:]!=(60,50):
        raise ValueError('The complete 60 by50 spatial raster must be retained')
    nt=signal.shape[0]
    curves=signal.reshape((nt,3000),order='F').T
    rates=observations['relaxation_rate'][0].ravel(order='F')
    r1,ct=signal_to_concentration(curves,rates)
    return r1,ct,observations['time_seconds'][0]/60.0,observations['plasma_concentration'][0]
