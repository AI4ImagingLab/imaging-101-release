from pathlib import Path
import numpy as np
def load_observations(path):
    """Return the two original 2048-channel float64 count arrays."""
    with np.load(Path(path),allow_pickle=False) as z:
        low=z['low_loss_counts'];core=z['core_loss_counts']
        if low.shape!=(1,2048) or core.shape!=(1,2048):raise ValueError('Expected (1,2048) counts')
        if low.dtype!=np.float64 or core.dtype!=np.float64:raise ValueError('Preserve original float64 counts')
        for key,offset in [('low_loss_energy',-40.),('core_loss_energy',360.)]:
            if not np.array_equal(z[key],(offset+.2*np.arange(2048))[None]):raise ValueError('Nominal energy grid mismatch')
        if not np.isfinite(low).all() or not np.isfinite(core).all():raise ValueError('Nonfinite counts')
        return low[0].copy(),core[0].copy()
