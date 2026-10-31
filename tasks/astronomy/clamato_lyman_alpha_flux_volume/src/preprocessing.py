"""Read processed real spectral pixels without refitting their continua."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

GRID_SHAPE = (60, 48, 876)
GRID_EXTENT = (30.0, 24.0, 438.0)

@dataclass
class Observations:
    positions: np.ndarray
    noise: np.ndarray
    flux: np.ndarray

def load_observations(path):
    """Return positions (n,3) in Mpc/h and dimensionless noise/flux (n,)."""
    with np.load(Path(path), allow_pickle=False) as z:
        p, s, f = (np.array(z[k][0], dtype=np.float64, copy=True)
                   for k in ('positions', 'noise_std', 'flux_contrast'))
    if p.ndim != 2 or p.shape[1] != 3 or s.shape != (len(p),) or f.shape != s.shape:
        raise ValueError('Inconsistent spectral pixel shapes')
    if not all(np.isfinite(v).all() for v in (p,s,f)) or np.any(s <= 0):
        raise ValueError('Pixels must be finite with positive noise')
    return Observations(p,s,f)

def grid_coordinates(shape=GRID_SHAPE, extent=GRID_EXTENT):
    """Cell centers, flattened with z fastest; no periodic wrapping."""
    axes = [(np.arange(n)+0.5)*length/n for n,length in zip(shape,extent)]
    return np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)

def whiten(noise, flux):
    """Return w=1/sigma and b=w*deltaF; released noise already includes its floor."""
    w = 1.0/np.asarray(noise)
    return w,w*np.asarray(flux)
