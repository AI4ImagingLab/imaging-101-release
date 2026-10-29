"""Public observation validation and inverse-variance weights."""
from pathlib import Path
import numpy as np

C=299792458.0

def prepare(frequency,polarization,noise):
    """Validate 1-D float64 Hz/complex128 Jy/float64 Jy and prepare weights."""
    frequency=np.asarray(frequency);polarization=np.asarray(polarization);noise=np.asarray(noise)
    if frequency.ndim!=1 or frequency.shape!=polarization.shape or noise.shape!=frequency.shape:
        raise ValueError('Expected matching one-dimensional observation arrays')
    if not all(np.isfinite(x).all() for x in (frequency,polarization,noise)):
        raise ValueError('Observations must be finite')
    if np.any(frequency<=0) or np.any(noise<=0) or np.any(np.diff(frequency)<=0):
        raise ValueError('Frequency must increase strictly and noise must be positive')
    lambda2=(C/frequency)**2
    weights=1/noise**2
    normalizer=1/np.sum(weights)
    lambda0=normalizer*np.sum(weights*lambda2)
    sigma=np.sqrt(np.sum(weights**2*noise**2))*normalizer
    return dict(frequency=frequency,polarization=polarization,noise=noise,lambda2=lambda2,
                weights=weights,lambda0=float(lambda0),sigma_fdf=float(sigma))

def load_observations(path='data/raw_data.npz'):
    """Load the single full 288-channel polarization observation without pickle."""
    with np.load(Path(path),allow_pickle=False) as f:
        required={'frequency':np.float64,'polarization':np.complex128,'noise':np.float64}
        for key,dtype in required.items():
            if f[key].shape!=(1,288) or f[key].dtype!=dtype:raise ValueError(f'Invalid {key} contract')
        return {key:f[key][0].copy() for key in required}
