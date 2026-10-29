"""Load measurements and estimate noise exclusively from line-free channels."""
from pathlib import Path
import numpy as np
from .physics_model import REST_FREQUENCY, C_KMS

def load_observations(path=Path('data/raw_data.npz')):
    """Return brightness (501,3,3) K and channel frequency (501,) Hz, float64."""
    with np.load(path, allow_pickle=False) as data:
        cube = data['brightness']; frequency = data['frequency']
    if cube.shape != (1,501,3,3) or frequency.shape != (1,501):
        raise ValueError('Unexpected acquisition shape')
    if not np.isfinite(cube).all() or not np.isfinite(frequency).all():
        raise ValueError('Nonfinite observations')
    return cube[0].astype(np.float64), frequency[0].astype(np.float64)

def estimate_noise(cube, frequency):
    """Population standard deviation in radio-velocity 20 to 28 km/s, K."""
    velocity = C_KMS*(1-frequency/REST_FREQUENCY)
    selection = (velocity >= 20)&(velocity <= 28)
    if selection.sum() < 2:
        raise ValueError('No line-free calibration channels')
    noise = cube[selection].std(axis=0, ddof=0)
    if np.any(noise <= 0):
        raise ValueError('Noise estimate must be positive')
    return noise
