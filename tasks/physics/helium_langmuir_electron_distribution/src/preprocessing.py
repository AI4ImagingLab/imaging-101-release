"""Measured probe inputs. Adapted from PlasmaPy; see LICENSE and NOTICE."""
from pathlib import Path
import numpy as np

PROBE_LENGTH_M = 1.145e-3
PROBE_DIAMETER_M = 1.57e-3
PROBE_AREA_M2 = np.pi * PROBE_LENGTH_M * PROBE_DIAMETER_M + np.pi * PROBE_DIAMETER_M**2 / 4

def unique_characteristic(bias, current):
    """Average duplicate voltages, then sort ascending; volts and amperes."""
    v=np.asarray(bias,dtype=np.float64); i=np.asarray(current,dtype=np.float64)
    if v.ndim!=1 or i.shape!=v.shape or not np.isfinite(v).all() or not np.isfinite(i).all():
        raise ValueError('Expected finite matching 1-D voltage and current')
    u=np.unique(v)
    return u, np.array([np.mean(i[v==x]) for x in u])

def load_data(path='data/raw_data.npz'):
    """Return full native characteristic as two 1-D float64 arrays."""
    with np.load(Path(path),allow_pickle=False) as a:
        if a['bias'].shape!=(1,304) or a['current'].shape!=(1,304):
            raise ValueError('Expected the complete 304-sample sweep')
        return unique_characteristic(a['bias'][0],a['current'][0])
