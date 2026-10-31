"""Load visible observations without accessing upstream or hidden assets."""
from pathlib import Path
import numpy as np

KEYS=('period','log_apparent_resistivity','log_resistivity_std','impedance_phase','phase_std','layer_top')

def load_data(path='data/raw_data.npz'):
    """Return unbatched float64 arrays; time [s], depth [m], phase [rad]."""
    with np.load(Path(path),allow_pickle=False) as archive:
        data={k:archive[k][0].copy() for k in KEYS}
    if any(data[k].shape!=(23,) for k in KEYS[:-1]) or data['layer_top'].shape!=(25,):
        raise ValueError('Observation/depth dimensions do not match acquisition.')
    if any(v.dtype!=np.float64 or not np.isfinite(v).all() for v in data.values()):
        raise ValueError('Inputs must be finite float64.')
    if any(np.any(data[k]<=0) for k in ('period','log_resistivity_std','phase_std')):
        raise ValueError('Periods and uncertainties must be positive.')
    if data['layer_top'][0]!=0 or np.any(np.diff(data['layer_top'])<=0):
        raise ValueError('Layer tops must increase from zero.')
    return data
